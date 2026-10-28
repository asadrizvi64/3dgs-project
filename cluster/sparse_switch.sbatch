#!/usr/bin/env bash
# Stock Faster-GS vs "dense Adam to SWITCH, then selective Adam" in the real trainer, same node, interleaved.
#   - both arms: lean-Adam patch present; base has its flags off (= stock adam_step kernel)
#   - test metrics every EVAL_EVERY steps (BACKUP.INTERVAL; the .pt checkpoints are deleted afterwards)
#   - SEGCLOCK lines (logs/<run>.log.seg): summed CUDA-event time of training_iteration, excluding tests/checkpoints
#   sbatch --export=ALL,SCENE=garden cluster/sparse_switch.sbatch      (from the repo root)
#SBATCH --job-name=gs-switch
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=c20
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
SCENE=${SCENE:-garden}
SCALE=${SCALE:-0.25}
SEEDS=${SEEDS:-"0 1 2"}
SWITCH=${SWITCH:-15000}
EVAL_EVERY=${EVAL_EVERY:-2500}
ITERS=${ITERS:-30000}     # NUM_ITERATIONS (lower only for smoke tests)
HERE=${SLURM_SUBMIT_DIR:-$PWD}/cluster
DATA=$(ws_find -F cat 3dgs-data) || { echo "cat workspace not visible on $(hostname)"; exit 1; }
module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
source "$DATA/fastergs-venv/bin/activate"
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd "$DATA/nerficg"
python "$HERE/patches/fastergs_sparse_from.py" src/Methods/FasterGS/Trainer.py
OUT=$HERE/../results/sparse_switch${CSV_TAG:-}; mkdir -p "$OUT/logs"
FAILED=()

run() {  # tag seed [env...]
  local tag=$1 seed=$2; shift 2
  local L=$OUT/logs/${tag}_s${seed}.log
  echo "########## $tag seed=$seed start $(date +%T)"
  rm -f "$L.seg" "$L.metrics"
  env "$@" FGS_SEGCLOCK=$EVAL_EVERY FGS_LOG="$L.seg" SCENE=$SCENE SCALE=$SCALE SEED=$seed ITERS=$ITERS METHOD=FasterGS TAG=$tag BACKUP_INTERVAL=$EVAL_EVERY \
    bash "$HERE/train_fastergs.sbatch" > "$L" 2>&1 || true
  grep -E "FGS_SPARSE_FROM|SEGCLOCK step=($SWITCH|$ITERS) " "$L.seg" || true
  tr '\r' '\n' < "$L" | grep -E "Traceback|Error|== run dir" | grep -v "it/s" || true
  local RUN; RUN=$(tr '\r' '\n' < "$L" | sed -n 's/^== run dir //p' | tail -1)
  if [ -z "$RUN" ] || [ ! -f "$RUN/test_$ITERS/metrics_8bit.txt" ]; then echo "FAILED $tag s$seed"; FAILED+=("$tag:s$seed"); return; fi
  rm -f "$RUN"/checkpoints/*.pt
  for t in "$RUN"/test_*/metrics_8bit.txt; do echo "METRIC $tag $seed $(basename $(dirname $t)) $(tail -1 $t)"; done | sort -t_ -k2 -n | tee -a "$L.metrics"
  grep -A2 "^training_iteration:" "$RUN/timings.txt" | tr '\n' ' '; echo
  cp "$RUN/timings.txt" "$OUT/logs/${tag}_s${seed}_timings.txt"
}

for seed in $SEEDS; do
  run swbase "$seed"
  run sw15k "$seed" FGS_SPARSE_FROM=$SWITCH
done

echo "########## check"
for seed in $SEEDS; do for tag in swbase sw15k; do
  n=$(grep -c "^SEGCLOCK" "$OUT/logs/${tag}_s${seed}.log.seg" 2>/dev/null || true); n=${n:-0}
  [ "$n" -eq $((ITERS / EVAL_EVERY)) ] || { echo "MISSING SEGCLOCK $tag s$seed ($n)"; FAILED+=("seg:$tag:s$seed"); }
done; done
if [ ${#FAILED[@]} -gt 0 ]; then echo "JOB FAILED: ${FAILED[*]}"; exit 1; fi
echo "JOB OK"
