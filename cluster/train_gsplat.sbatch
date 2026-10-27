#!/usr/bin/env bash
# gsplat simple_trainer baseline on one Mip-NeRF360 scene.
#   sbatch --export=ALL,SCENE=garden cluster/train_gsplat.sbatch      (from the repo root)
# Results: $DATA/runs/gsplat-<commit>/<scene>/  (stats/*.json hold PSNR, SSIM, LPIPS, time, #GS, memory)
#SBATCH --job-name=gs-train-gsplat
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail

SCENE=${SCENE:-garden}
FACTOR=${FACTOR:-4}   # 4 for outdoor scenes, 2 for indoor (bonsai, counter, kitchen, room), as in 3DGS
DATA=$(ws_find -F cat 3dgs-data)
GSPLAT=$DATA/gsplat

module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
source "$DATA/venv/bin/activate"

COMMIT=$(git -C "$GSPLAT" rev-parse --short HEAD)
OUT=$DATA/runs/gsplat-$COMMIT/$SCENE
mkdir -p "$OUT"

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch, gsplat; print('torch', torch.__version__, 'gsplat', gsplat.__version__, '$COMMIT')"
echo "scene=$SCENE factor=$FACTOR out=$OUT job=${SLURM_JOB_ID:-local}"

cd "$GSPLAT/examples"
TIMER=; [ -x /usr/bin/time ] && TIMER="/usr/bin/time -v"   # not installed on every node
$TIMER python simple_trainer.py default \
    --data_dir "$DATA/data/360_v2/$SCENE" --data_factor "$FACTOR" \
    --result_dir "$OUT" --disable_viewer 2>&1 | grep -vE "^\s+(Page|Socket|Signals|Swaps|File system|Voluntary|Involuntary|Average|Major|Minor|Exit|Percent|System|User)"

echo "== stats"
for f in "$OUT"/stats/*.json; do echo "$(basename "$f"): $(cat "$f")"; done
