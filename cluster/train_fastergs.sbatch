#!/usr/bin/env bash
# Faster-GS (NeRFICG method) baseline on one Mip-NeRF360 scene. First run also builds its venv.
#   sbatch --export=ALL,SCENE=garden cluster/train_fastergs.sbatch      (from the repo root)
#   sbatch --export=ALL,SCENE=garden,METHOD=FasterGSFused cluster/train_fastergs.sbatch
#   (each variant is a branch of the Faster-GS repo cloned to nerficg/src/Methods/<METHOD>)
# Layout on the cat workspace:  nerficg/ (framework)  nerficg/src/Methods/<METHOD>/  fastergs-venv/
# Results: nerficg/output/<METHOD>/<model name>_<timestamp>/  (test_*/metrics_8bit.txt, timing + VRAM stats)
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
METHOD=${METHOD:-FasterGS}
SCALE=${SCALE:-0.25}   # 0.25 for outdoor scenes, 0.5 for indoor, as in 3DGS
DATA=$(ws_find -F cat 3dgs-data)
NERFICG=$DATA/nerficg
MDIR=$NERFICG/src/Methods/$METHOD
LOW=$(echo "$METHOD" | tr 'A-Z' 'a-z')
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
  touch "$VENV/.built"
else
  source "$VENV/bin/activate"
fi
# method-specific CUDA extensions (FasterGS was installed by the first build)
[ "$METHOD" = FasterGS ] && [ -d "$VENV/lib/python3.12/site-packages/FasterGSCudaBackend" ] && touch "$VENV/.built-FasterGS"
if [ ! -f "$VENV/.built-$METHOD" ]; then
  echo "== installing extensions for $METHOD"
  cd "$NERFICG" && python scripts/install.py -m "$METHOD"
  python -c "import ${METHOD}CudaBackend" && touch "$VENV/.built-$METHOD"
  pip freeze > "$DATA/fastergs-venv-freeze.txt"
fi

NC=$(git -C "$NERFICG" rev-parse --short HEAD)
FC=$(git -C "$MDIR" rev-parse --short HEAD)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__)"
echo "method=$METHOD scene=$SCENE scale=$SCALE nerficg=$NC method_commit=$FC job=${SLURM_JOB_ID:-local}"

# authors' garden config, pointed at our data, GUI off, scene/scale substituted
CFG=$NERFICG/configs/${LOW}_${SCENE}_cluster.yaml
mkdir -p "$NERFICG/configs"
sed -e "s|PATH: dataset/mipnerf360/garden|PATH: $DATA/data/360_v2/$SCENE|" \
    -e "s|MODEL_NAME: ${LOW}_garden|MODEL_NAME: ${LOW}_${SCENE}_${FC}|" \
    -e "s|IMAGE_SCALE_FACTOR: 0.25|IMAGE_SCALE_FACTOR: $SCALE|" \
    -e "/^    GUI:/,/^    [A-Z]/ s|ACTIVATE: true|ACTIVATE: false|" \
    "$MDIR/${LOW}_garden.yaml" > "$CFG"
grep -nE "METHOD_TYPE|PATH:|MODEL_NAME|IMAGE_SCALE|GUI:|ACTIVATE" "$CFG"

cd "$NERFICG"
/usr/bin/time -v python scripts/train.py -c "$CFG" 2>&1 \
  | grep -vE "^\s+(Page|Socket|Signals|Swaps|File system|Voluntary|Involuntary|Average|Major|Minor|Exit|Percent|System|User)"

RUN=$(ls -dt output/$METHOD/${LOW}_${SCENE}_${FC}* | head -1)
echo "== run dir $RUN"
ls "$RUN"
cat "$RUN/n_gaussians.txt" "$RUN/vram_stats.txt"; echo
grep -A3 "^training_iteration:" "$RUN/timings.txt"
tail -2 "$RUN"/test_*/metrics_8bit.txt
