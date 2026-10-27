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
SEED=${SEED:-0}
TAG=${TAG:-}          # appended to the run name, e.g. TAG=prof for profiling runs
LAUNCHER=${LAUNCHER:-}  # optional prefix, e.g. "nsys profile ..." or "ncu ..."
OPT_FLAGS=${OPT_FLAGS:-}  # extra TRAINING.OPTIMIZER keys, e.g. "SPARSE_ADAM=true,SH_REST_BF16_MOMENTS=true"
SCALE=${SCALE:-0.25}   # 0.25 for outdoor scenes, 0.5 for indoor, as in 3DGS
DATA=$(ws_find -F cat 3dgs-data) || { echo "cat workspace not visible on $(hostname); resubmit with --exclude=$(hostname)"; exit 1; }   # c20 lacks /data/cat (2026-10-27)
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
echo "method=$METHOD seed=$SEED tag=$TAG opt_flags=$OPT_FLAGS scene=$SCENE scale=$SCALE nerficg=$NC method_commit=$FC job=${SLURM_JOB_ID:-local}"

# authors' garden config, pointed at our data, GUI off, scene/scale/seed substituted
NAME=${LOW}_${SCENE}_${FC}_s${SEED}${TAG}
CFG=$NERFICG/configs/${NAME}_cluster.yaml
mkdir -p "$NERFICG/configs"
sed -e "s|PATH: dataset/mipnerf360/garden|PATH: $DATA/data/360_v2/$SCENE|" \
    -e "s|MODEL_NAME: ${LOW}_garden|MODEL_NAME: $NAME|" \
    -e "s|RANDOM_SEED: 0|RANDOM_SEED: $SEED|" \
    -e "s|IMAGE_SCALE_FACTOR: 0.25|IMAGE_SCALE_FACTOR: $SCALE|" \
    -e "/^    GUI:/,/^    [A-Z]/ s|ACTIVATE: true|ACTIVATE: false|" \
    "$MDIR/${LOW}_garden.yaml" > "$CFG"
for kv in ${OPT_FLAGS//,/ }; do   # insert extra optimizer options right below "    OPTIMIZER:"
  sed -i "/^    OPTIMIZER:/a\        ${kv%%=*}: ${kv#*=}" "$CFG"
done
grep -nE "SPARSE_ADAM|BF16_MOMENTS|RANDOM_SEED|METHOD_TYPE|PATH:|MODEL_NAME|IMAGE_SCALE|GUI:|ACTIVATE" "$CFG"

cd "$NERFICG"
# GPU clock / power / temperature log, 1 sample per second, for the whole training run
CLK=$(mktemp -p "$NERFICG/output" "clock_${NAME}_XXXX.csv")
nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active \
  --format=csv,noheader,nounits -lms 1000 > "$CLK" &
CLKPID=$!
TIMER=; [ -x /usr/bin/time ] && TIMER="/usr/bin/time -v"   # not installed on every node
$TIMER $LAUNCHER python scripts/train.py -c "$CFG" 2>&1 \
  | grep -vE "^\s+(Page|Socket|Signals|Swaps|File system|Voluntary|Involuntary|Average|Major|Minor|Exit|Percent|System|User)" || true
kill $CLKPID 2>/dev/null || true

RUN=$(ls -dt output/$METHOD/${NAME}_* | head -1)
echo "== run dir $RUN"
mv "$CLK" "$RUN/gpu_clock_log.csv"
python - "$RUN/gpu_clock_log.csv" > "$RUN/gpu_clock_summary.txt" <<'PY2'
import sys, statistics as st
rows = [r.split(', ') for r in open(sys.argv[1]) if r.count(',') >= 7]
busy = [r for r in rows if r[6].strip().isdigit() and int(r[6]) >= 50]   # samples with >= 50% GPU utilization
f = lambda i: [float(r[i]) for r in busy if r[i].replace('.', '', 1).strip().isdigit()]
sm, mem, pw, tc = f(2), f(3), f(4), f(5)
reasons = {}
for r in busy:
    reasons[r[7].strip()] = reasons.get(r[7].strip(), 0) + 1
if sm:
    print(f'CLOCK samples_busy={len(busy)} sm_mhz_mean={st.mean(sm):.0f} sm_mhz_min={min(sm):.0f} mem_mhz={st.mean(mem):.0f} '
          f'power_w_mean={st.mean(pw):.0f} power_w_max={max(pw):.0f} temp_c_max={max(tc):.0f} event_reasons={reasons}')
else:
    print(f'CLOCK no busy samples ({len(rows)} rows)')
PY2
cat "$RUN/gpu_clock_summary.txt"
ls "$RUN"
cat "$RUN/n_gaussians.txt" "$RUN/vram_stats.txt" 2>/dev/null || true; echo
grep -A3 "^training_iteration:" "$RUN/timings.txt" 2>/dev/null || true
tail -2 "$RUN"/test_*/metrics_8bit.txt 2>/dev/null || true
