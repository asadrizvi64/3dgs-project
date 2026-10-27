#!/usr/bin/env bash
# Phase 0 run: build, test, gradient table, benchmark, Nsight Compute profile.
#   sbatch jobs/phase0.sbatch            (from the toy2d folder)
#SBATCH --job-name=gs2d-phase0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# TODO: add your partition/account, e.g.
# #SBATCH --partition=gpu
# #SBATCH --account=your_account
set -euo pipefail

module purge || true
module load python/3.11 || true      # TODO: same modules as setup_env.sh
module load cuda/12.4 || true
source "${GS2D_ENV:-$HOME/envs/gs2d}/bin/activate"

mkdir -p logs results profiles
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
TAG="${COMMIT}-${SLURM_JOB_ID:-local}"

nvidia-smi
echo "== commit $COMMIT"

echo "== build + tests"
GS2D_VERBOSE_BUILD=1 python -c "from gs2d.raster import ext; ext()"
python -m pytest -q tests

echo "== gradient table (G0 accuracy)"
python gradcheck_report.py --json "results/gradcheck-$TAG.json" | tee "results/gradcheck-$TAG.md"

echo "== benchmark (G0 speed)"
python bench.py --json "results/bench-$TAG.json"

echo "== fit demo"
python fit.py --n 20000 --iters 1000 --backend splat --out "results/fit-$TAG"

if command -v ncu >/dev/null; then
  echo "== Nsight Compute"
  ncu --set full -k regex:"backward_(pixel|splat)_kernel" -c 2 -f -o "profiles/bwd-$TAG" \
      python bench.py --sizes 1024 --counts 50000 --warmup 0 --iters 1
else
  echo "ncu not found: load the Nsight Compute module to profile"
fi
