#!/usr/bin/env bash
# One-time environment setup on the cluster login node.
#   bash jobs/setup_env.sh
# Edit the module lines to match your cluster (`module avail cuda`, `module avail python`).
set -euo pipefail

module purge || true
module load python/3.11 || true      # TODO: your cluster's module name
module load cuda/12.4 || true        # TODO: must match the CUDA version of the torch wheel below

ENV_DIR="${GS2D_ENV:-$HOME/envs/gs2d}"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install ninja numpy pillow pytest

python - <<'EOF'
import torch, sys
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)
EOF
echo "Environment ready at $ENV_DIR. Run jobs with: sbatch jobs/phase0.sbatch"
