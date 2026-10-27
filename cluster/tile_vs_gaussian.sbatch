#!/usr/bin/env bash
# Tile fraction vs active-Gaussian fraction on a Faster-GS model (cluster/analysis/tile_vs_gaussian.py).
#   1. train SCENE (stock Faster-GS, seed 0) saving checkpoints every 5000 steps   (skipped if RUN is given)
#   2. analyse checkpoints 5k (densifying), 15k (densification just ended), 30k (final)
#   sbatch --export=ALL,SCENE=garden cluster/tile_vs_gaussian.sbatch      (from the repo root)
#SBATCH --job-name=gs-tile-gauss
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=c20
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
SCENE=${SCENE:-garden}
SCALE=${SCALE:-0.25}
HERE=${SLURM_SUBMIT_DIR:-$PWD}/cluster
DATA=$(ws_find -F cat 3dgs-data) || { echo "cat workspace not visible on $(hostname)"; exit 1; }
module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
source "$DATA/fastergs-venv/bin/activate"
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd "$DATA/nerficg"
if [ -z "${RUN:-}" ]; then
  SCENE=$SCENE SCALE=$SCALE SEED=0 METHOD=FasterGS TAG=ckpt BACKUP_INTERVAL=5000 \
    bash "$HERE/train_fastergs.sbatch" 2>&1 | tr '\r' '\n' | grep -E "PSNR|Traceback|Error|== run dir" | grep -v "it/s" || true
  RUN=$(ls -dt output/FasterGS/fastergs_${SCENE}_*_s0ckpt_* | head -1)
fi
echo "run=$RUN"; ls "$RUN/checkpoints"
OUT=$HERE/../results/tiles; mkdir -p "$OUT"
python "$HERE/analysis/tile_vs_gaussian.py" --run "$RUN" --ckpt 0005000.pt 0015000.pt final.pt \
  --out "$OUT/${SCENE}_tile_vs_gaussian.csv" 2>&1 | grep -vE "it/s|^\s*$"
