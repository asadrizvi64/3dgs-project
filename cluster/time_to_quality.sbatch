#!/usr/bin/env bash
# Time-to-quality from the step-15k state (cluster/analysis/time_to_quality.py).
#   1. patch Trainer.py with the FGS_SAVE_STATE hook (idempotent)
#   2. per seed: stock Faster-GS training that also saves state_0015000.pt (skipped if it already exists)
#   3. per seed: cost-model calibration on that checkpoint (CUDA events), then every arm in ARMS (policy:adam)
#   4. check: every arm exited 0 and wrote its final eval row; the job fails otherwise
#   sbatch --export=ALL,SCENE=garden cluster/time_to_quality.sbatch      (from the repo root)
#SBATCH --job-name=gs-ttq
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --exclude=c20
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
SCENE=${SCENE:-garden}
SCALE=${SCALE:-0.25}
SEEDS=${SEEDS:-"0 1 2"}
ARMS=${ARMS:-"dense:dense err_block:dense rand_block:dense err_block:sparse rand_block:sparse dense:sparse"}
IMAGES=${IMAGES:-15000}
HERE=${SLURM_SUBMIT_DIR:-$PWD}/cluster
DATA=$(ws_find -F cat 3dgs-data) || { echo "cat workspace not visible on $(hostname)"; exit 1; }
module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
source "$DATA/fastergs-venv/bin/activate"
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd "$DATA/nerficg"
python "$HERE/patches/fastergs_save_state.py" src/Methods/FasterGS/Trainer.py
OUT=$HERE/../results/ttq; mkdir -p "$OUT"
TAGX=${CSV_TAG:-}
CSV=$OUT/${SCENE}_ttq${TAGX}.csv
LOGS=$OUT/logs${TAGX}; mkdir -p "$LOGS"
FAILED=()

for seed in $SEEDS; do
  RUN=$(ls -dt output/FasterGS/fastergs_${SCENE}_*_s${seed}tq_* 2>/dev/null | head -1 || true)
  if [ -z "$RUN" ] || [ ! -f "$RUN/checkpoints/state_0015000.pt" ]; then
    echo "########## train seed $seed $(date +%T)"
    SCENE=$SCENE SCALE=$SCALE SEED=$seed METHOD=FasterGS TAG=tq BACKUP_INTERVAL=15000 FGS_SAVE_STATE=15000 \
      bash "$HERE/train_fastergs.sbatch" > "$LOGS/train_s$seed.log" 2>&1 || true
    tr '\r' '\n' < "$LOGS/train_s$seed.log" | grep -E "PSNR|FGS_SAVE|Traceback|Error|== run dir" | grep -v "it/s" || true
    RUN=$(ls -dt output/FasterGS/fastergs_${SCENE}_*_s${seed}tq_* 2>/dev/null | head -1 || true)
    if [ -z "$RUN" ] || [ ! -f "$RUN/checkpoints/state_0015000.pt" ]; then
      echo "FAILED training seed $seed (no state_0015000.pt), see $LOGS/train_s$seed.log"; FAILED+=("train:s$seed"); continue
    fi
  fi
  echo "run=$RUN stock-30k: $(tail -1 $RUN/test_*/metrics_8bit.txt 2>/dev/null || echo n/a)"
  CAL=$LOGS/calib_s$seed.json
  if ! python "$HERE/analysis/time_to_quality.py" --run "$RUN" --seed $seed --calibrate "$CAL" > "$LOGS/calib_s$seed.log" 2>&1; then
    echo "FAILED calibration seed $seed:"; tail -25 "$LOGS/calib_s$seed.log"; FAILED+=("calib:s$seed"); CAL=
  else
    grep "^CALIB" "$LOGS/calib_s$seed.log"
  fi
  for arm in $ARMS; do
    pol=${arm%%:*}; adam=${arm#*:}
    L=$LOGS/s${seed}_${pol}_${adam}.log
    echo "########## seed $seed policy $pol adam $adam $(date +%T)"
    if python "$HERE/analysis/time_to_quality.py" --run "$RUN" --policy $pol --adam $adam --seed $seed --images $IMAGES \
         ${CAL:+--calib "$CAL"} --out "$CSV" > "$L" 2>&1; then
      grep -E "^(state|EVAL|DONE)" "$L"
    else
      echo "FAILED seed $seed $pol $adam (exit $?):"; grep -vE "WARNING|INFO" "$L" | tail -30; FAILED+=("s$seed:$pol:$adam")
    fi
  done
done

echo "########## check"
python - "$CSV" "$IMAGES" "$SEEDS" "$ARMS" <<'PY' || FAILED+=("check")
import csv, sys
path, images, seeds, arms = sys.argv[1], float(sys.argv[2]), sys.argv[3].split(), sys.argv[4].split()
rows = list(csv.DictReader(open(path))) if __import__('os').path.exists(path) else []
missing = []
for s in seeds:
    for a in arms:
        pol, adam = a.split(':')
        final = [r for r in rows if r['seed'] == s and r['policy'] == pol and r['adam'] == adam and abs(float(r['image_equiv']) - images) < 1e-6]
        if len(final) != 1:
            missing.append(f's{s}:{a} ({len(final)} final rows)')
print('all arms complete' if not missing else 'MISSING: ' + ', '.join(missing))
sys.exit(1 if missing else 0)
PY
if [ ${#FAILED[@]} -gt 0 ]; then echo "JOB FAILED: ${FAILED[*]}"; exit 1; fi
echo "JOB OK"
