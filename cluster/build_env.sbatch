#!/usr/bin/env bash
# Builds the training venv on the cat workspace: torch 2.11 + cu128, gsplat at the pinned
# commit (compiled for A100 + H100), and the gsplat example dependencies.
#   sbatch cluster/build_env.sbatch          (from the repo root; logs go to cluster/logs/)
#SBATCH --job-name=gs-build-env
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail

DATA=$(ws_find -F cat 3dgs-data) || { echo "cat workspace not visible on $(hostname); resubmit with --exclude=$(hostname)"; exit 1; }   # c20 lacks /data/cat (2026-10-27)
VENV=$DATA/venv
GSPLAT=$DATA/gsplat
GSPLAT_COMMIT=512d366b67073d77ca099ede742683c165dfc23b

module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
export PIP_CACHE_DIR=$DATA/.pipcache
export TORCH_CUDA_ARCH_LIST="8.0;9.0" MAX_JOBS=14

[ -d "$VENV" ] || python -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel setuptools ninja

echo "== torch"
pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

echo "== example requirements (torch pins and nvidia-ncore dropped; torch is pinned above)"
grep -vE '^\s*(#|$)|^torch==|^torchvision==|^nvidia-ncore' "$GSPLAT/examples/requirements.txt" > /tmp/gs-req-$$.txt
cat /tmp/gs-req-$$.txt
pip install -q --no-build-isolation -r /tmp/gs-req-$$.txt

echo "== gsplat $GSPLAT_COMMIT"
cd "$GSPLAT" && git checkout -q "$GSPLAT_COMMIT" && git submodule update --init --recursive -q  # glm headers
pip install -q --no-build-isolation -e "$GSPLAT"

echo "== check"
python - <<'PY'
import torch, torchvision, gsplat
from gsplat import rasterization
print("torch", torch.__version__, "torchvision", torchvision.__version__, "gsplat", gsplat.__version__)
d = "cuda"
N = 1000
means = torch.randn(N, 3, device=d) + torch.tensor([0, 0, 5.0], device=d)
quats = torch.nn.functional.normalize(torch.randn(N, 4, device=d), dim=-1)
scales = torch.full((N, 3), 0.05, device=d)
opac = torch.full((N,), 0.5, device=d)
colors = torch.rand(N, 3, device=d)
K = torch.tensor([[300.0, 0, 128], [0, 300.0, 128], [0, 0, 1]], device=d)[None]
img, alpha, _ = rasterization(means, quats, scales, opac, colors, torch.eye(4, device=d)[None], K, 256, 256)
print("rasterization ok", tuple(img.shape), float(img.mean()))
PY
pip freeze > "$DATA/venv-freeze.txt"
echo "== done; freeze saved to $DATA/venv-freeze.txt"
