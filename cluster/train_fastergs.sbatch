#!/usr/bin/env bash
# Faster-GS (NeRFICG method) baseline on one Mip-NeRF360 scene. First run also builds its venv.
#   sbatch --export=ALL,SCENE=garden cluster/train_fastergs.sbatch      (from the repo root)
# Layout on the cat workspace:  nerficg/ (framework)  nerficg/src/Methods/FasterGS/ (method)  fastergs-venv/
# Results: nerficg/output/FasterGS/<model name>_<timestamp>/  (test_*/metrics_8bit.txt, timing + VRAM stats)
#SBATCH --job-name=gs-train-fastergs
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail

SCENE=${SCENE:-garden}
SCALE=${SCALE:-0.25}   # 0.25 for outdoor scenes, 0.5 for indoor, as in 3DGS
DATA=$(ws_find -F cat 3dgs-data)
NERFICG=$DATA/nerficg
METHOD=$NERFICG/src/Methods/FasterGS
VENV=$DATA/fastergs-venv

module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
export PIP_CACHE_DIR=$DATA/.pipcache TORCH_CUDA_ARCH_LIST="8.0;9.0" MAX_JOBS=14 MKL_THREADING_LAYER=GNU

if [ ! -f "$VENV/.built" ]; then
  echo "== building $VENV"
  [ -d "$VENV" ] || python -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install -q --upgrade pip wheel ninja "setuptools==80.10.2"
  pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  # NeRFICG py311_cu128_no_gui environment, minus torch (pinned above)
  pip install -q numpy tqdm natsort pyyaml munch tabulate wandb opencv-python-headless kornia torchmetrics \
      lpips einops plyfile matplotlib timm plotly pillow jax pyproj scikit-learn pycolmap
  cd "$NERFICG" && python scripts/install.py -m FasterGS
  pip freeze > "$DATA/fastergs-venv-freeze.txt"
  touch "$VENV/.built"
else
  source "$VENV/bin/activate"
fi

NC=$(git -C "$NERFICG" rev-parse --short HEAD)
FC=$(git -C "$METHOD" rev-parse --short HEAD)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__)"
echo "scene=$SCENE scale=$SCALE nerficg=$NC fastergs=$FC job=${SLURM_JOB_ID:-local}"

# authors' garden config, pointed at our data, GUI off, scene/scale substituted
CFG=$NERFICG/configs/fastergs_${SCENE}_cluster.yaml
mkdir -p "$NERFICG/configs"
sed -e "s|PATH: dataset/mipnerf360/garden|PATH: $DATA/data/360_v2/$SCENE|" \
    -e "s|MODEL_NAME: fastergs_garden|MODEL_NAME: fastergs_${SCENE}_${FC}|" \
    -e "s|IMAGE_SCALE_FACTOR: 0.25|IMAGE_SCALE_FACTOR: $SCALE|" \
    -e "/^    GUI:/,/^    [A-Z]/ s|ACTIVATE: true|ACTIVATE: false|" \
    "$METHOD/fastergs_garden.yaml" > "$CFG"
grep -nE "PATH:|MODEL_NAME|IMAGE_SCALE|GUI:|ACTIVATE" "$CFG"

cd "$NERFICG"
/usr/bin/time -v python scripts/train.py -c "$CFG" 2>&1 \
  | grep -vE "^\s+(Page|Socket|Signals|Swaps|File system|Voluntary|Involuntary|Average|Major|Minor|Exit|Percent|System|User)"

RUN=$(ls -dt output/FasterGS/fastergs_${SCENE}_${FC}* | head -1)
echo "== run dir $RUN"
ls "$RUN"
for f in $(find "$RUN" -maxdepth 2 \( -name "metrics_8bit.txt" -o -name "*.txt" -o -name "*.json" \) | head -20); do
  echo "== $f"; head -8 "$f"
done
