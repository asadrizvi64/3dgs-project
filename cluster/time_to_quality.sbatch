#!/usr/bin/env bash
# Time-to-quality from the step-15k state: dense vs error-guided vs random 64-px block sampling
# (25% of blocks), each with dense Adam, plus error blocks and dense images with selective (gradient-row) Adam.
#   1. patch Trainer.py with the FGS_SAVE_STATE hook (idempotent)
#   2. for each seed: stock Faster-GS training that also saves state_0015000.pt (skipped if it exists)
#   3. for each seed: continue from 15k with each policy for 15,000 image equivalents, test PSNR every 1,500
#   sbatch --export=ALL,SCENE=garden cluster/time_to_quality.sbatch      (from the repo root)
#SBATCH --job-name=gs-ttq
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=c20
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
SCENE=${SCENE:-garden}
SCALE=${SCALE:-0.25}
SEEDS=${SEEDS:-"0 1 2"}
ARMS=${ARMS:-"dense:dense err_block:dense rand_block:dense err_block:sparse dense:sparse"}  # policy:adam
IMAGES=${IMAGES:-15000}
HERE=${SLURM_SUBMIT_DIR:-$PWD}/cluster
DATA=$(ws_find -F cat 3dgs-data) || { echo "cat workspace not visible on $(hostname)"; exit 1; }
module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
source "$DATA/fastergs-venv/bin/activate"
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd "$DATA/nerficg"
python "$HERE/patches/fastergs_save_state.py" src/Methods/FasterGS/Trainer.py
OUT=$HERE/../results/ttq; mkdir -p "$OUT"
CSV=$OUT/${SCENE}_ttq${CSV_TAG:-}.csv
for seed in $SEEDS; do
  RUN=$(ls -dt output/FasterGS/fastergs_${SCENE}_*_s${seed}tq_* 2>/dev/null | head -1 || true)
  if [ -z "$RUN" ] || [ ! -f "$RUN/checkpoints/state_0015000.pt" ]; then
    echo "########## train seed $seed $(date +%T)"
    SCENE=$SCENE SCALE=$SCALE SEED=$seed METHOD=FasterGS TAG=tq BACKUP_INTERVAL=15000 FGS_SAVE_STATE=15000 \
      bash "$HERE/train_fastergs.sbatch" 2>&1 | tr '\r' '\n' | grep -E "PSNR|FGS_SAVE|Traceback|Error|== run dir" | grep -v "it/s" || true
    RUN=$(ls -dt output/FasterGS/fastergs_${SCENE}_*_s${seed}tq_* | head -1)
  fi
  echo "run=$RUN stock-30k: $(tail -1 $RUN/test_*/metrics_8bit.txt 2>/dev/null)"
  for arm in $ARMS; do
    pol=${arm%%:*}; adam=${arm#*:}
    echo "########## seed $seed policy $pol adam $adam $(date +%T)"
    python "$HERE/analysis/time_to_quality.py" --run "$RUN" --policy $pol --adam $adam --seed $seed --images $IMAGES \
      --out "$CSV" 2>&1 | grep -E "^(state|EVAL)|Traceback|Error" || true
  done
done
