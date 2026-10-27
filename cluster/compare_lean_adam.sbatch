#!/usr/bin/env bash
# Lean Adam vs Faster-GS baseline, same node, one job:
#   1. rebuild the FasterGS CUDA extension (with cluster/patches/fastergs_lean_adam.patch applied)
#   2. GPU unit checks (cluster/tests/test_lean_adam.py); abort if any fail
#   3. SEEDS x {base, lean} interleaved   (lean = SPARSE_ADAM + SH_REST_BF16_MOMENTS)
#   4. single-seed ablations: sparse only, bf16 only
#   5. stage-timing runs (FGS_PROF=1) for base and lean
#   sbatch --export=ALL,SCENE=garden cluster/compare_lean_adam.sbatch      (from the repo root)
#SBATCH --job-name=gs-lean-adam
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --exclude=c20
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail

SCENE=${SCENE:-garden}
SCALE=${SCALE:-0.25}
SEEDS=${SEEDS:-"0 1 2"}
ABLATIONS=${ABLATIONS:-1}
HERE=${SLURM_SUBMIT_DIR:-$PWD}/cluster
DATA=$(ws_find -F cat 3dgs-data) || { echo "cat workspace not visible on $(hostname)"; exit 1; }
MDIR=$DATA/nerficg/src/Methods/FasterGS
LEAN="SPARSE_ADAM=true,SH_REST_BF16_MOMENTS=true"

module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
source "$DATA/fastergs-venv/bin/activate"
export TORCH_CUDA_ARCH_LIST="8.0;9.0" MAX_JOBS=14 PIP_CACHE_DIR=$DATA/.pipcache
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader)"
echo "method commit: $(git -C "$MDIR" log -1 --oneline)"

echo "########## 1. build"
git -C "$MDIR" log -1 --format=%s | grep -q "^Lean Adam" || { echo "lean adam patch not applied"; exit 1; }
pip install -q --no-build-isolation --no-deps --force-reinstall "$MDIR/FasterGSCudaBackend"
python -c "import torch; from FasterGSCudaBackend import _C; assert hasattr(_C, 'adam_step_sparse') and hasattr(_C, 'n_touched_tiles'); print('extension ok')"

echo "########## 2. unit checks"
python -I "$HERE/tests/test_lean_adam.py"

run() {  # tag seed opt_flags [extra env...]
  local tag=$1 seed=$2 flags=$3; shift 3
  echo; echo "########## $tag seed=$seed flags=$flags start $(date +%T)"
  env "$@" SCENE=$SCENE SCALE=$SCALE SEED=$seed METHOD=FasterGS TAG=$tag OPT_FLAGS=$flags \
    bash "$HERE/train_fastergs.sbatch" 2>&1 | tr '\r' '\n' | grep -E "^(VISFRAC|STAGE|CLOCK)|PSNR|Traceback|Error|== run dir" | grep -v "it/s" || true
}

echo "########## 3. base vs lean"
for seed in $SEEDS; do
  run base "$seed" ""
  run lean "$seed" "$LEAN"
done
if [ "$ABLATIONS" = 1 ]; then
  echo "########## 4. ablations"
  run sparse 0 "SPARSE_ADAM=true"
  run bf16 0 "SH_REST_BF16_MOMENTS=true"
  echo "########## 5. stage timing"
  run baseprof 0 "" FGS_PROF=1
  run leanprof 0 "$LEAN" FGS_PROF=1
fi

echo; echo "########## SUMMARY (tag seed ms/it train_s psnr ssim lpips n_gaussians vram_gib run)"
cd "$DATA/nerficg"
for d in $(ls -dt output/FasterGS/fastergs_${SCENE}_*_s*{base,lean,sparse,bf16}_* 2>/dev/null | sort); do
  [ -f "$d/timings.txt" ] || continue
  tag=$(basename "$d" | sed -E 's/.*_s[0-9]+([a-z0-9]+)_[0-9]{4}-.*/\1/'); seed=$(basename "$d" | sed -E 's/.*_s([0-9]+)[a-z].*/\1/')
  ms=$(grep -A2 "^training_iteration:" "$d/timings.txt" | awk -F': ' '/per iteration/{print $2}')
  met=$(tail -1 "$d"/test_*/metrics_8bit.txt | sed 's/PSNR://; s/SSIM://; s/LPIPS://')
  ng=$(grep -o 'N_Gaussians:[0-9]*' "$d/n_gaussians.txt" | cut -d: -f2)
  vr=$(awk '/allocated:/{print $2; exit}' "$d/vram_stats.txt")
  echo "RESULT $tag $seed $ms $(awk -v m="$ms" 'BEGIN{printf "%.1f", m*30}') $met $ng $vr $(basename "$d")"
done | grep -E "_s[0-9]+(base|lean|sparse|bf16)_$(date +%Y-%m-%d)" || true
