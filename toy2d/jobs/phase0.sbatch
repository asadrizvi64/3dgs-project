#!/usr/bin/env bash
# Phase 0 run on TU Dresden ZIH Capella (H100): build, tests, gradient table, benchmark, fit, Nsight Compute.
#   mkdir -p logs && sbatch jobs/phase0.sbatch            (from the toy2d folder)
# Uses the venv in the "3dgs" workspace (torch 2.11 + cu128). Override with GS2D_ENV=/path/to/venv.
#SBATCH --job-name=gs2d-phase0
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
source "${GS2D_ENV:-$(ws_find 3dgs)/venv}/bin/activate"

mkdir -p logs results profiles
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
TAG="${COMMIT}-${SLURM_JOB_ID:-local}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "== commit $COMMIT"

echo "== build"
GS2D_VERBOSE_BUILD=1 python -c "from gs2d.raster import ext; ext()"

echo "== tests"
python -m pytest -q tests || echo "TESTS FAILED (continuing so the report and bench still run)"

echo "== gradient table (G0 accuracy)"
python gradcheck_report.py --json "results/gradcheck-$TAG.json" | tee "results/gradcheck-$TAG.md"

echo "== benchmark (G0 speed)"
python bench.py --json "results/bench-$TAG.json"

echo "== fit demo"
python fit.py --n 20000 --iters 1000 --backend splat --out "results/fit-$TAG"

if command -v ncu >/dev/null; then
  echo "== Nsight Compute"
  ncu --set full -k regex:"backward_(pixel|splat)_kernel" -c 2 -f -o "profiles/bwd-$TAG" \
      python bench.py --sizes 1024 --counts 50000 --warmup 0 --iters 1 \
    || echo "ncu failed (GPU performance counters may need admin permission on this cluster)"
else
  echo "ncu not found"
fi
echo "== done"
