#!/usr/bin/env bash
# (Re)creates the FasterGSReinit method next to FasterGS in NeRFICG and builds its CUDA extension (alt_reinit/PLAN.md, R0):
#   1. clone nerficg/src/Methods/FasterGS at BASE (default 1144604 = Faster-GS c60ea26 + lean Adam, options off = stock)
#   2. rename FasterGS -> FasterGSReinit everywhere (method, classes, FasterGSReinitCudaBackend, yaml), one commit
#   3. trainer patches, one commit each: the shared clock hook (cluster/patches/fastergs_sparse_from.py, SELECTIVE_FROM
#      stays unset), the shared save-state hook (cluster/patches/fastergs_save_state.py), the track's hooks
#      (alt_reinit/patches/fastergs_reinit.py)
#   4. copy alt_reinit/reinit_core.py and reinit_trainer.py into the method, one commit
#   5. pip install the extension into fastergs-venv (BUILD=0 skips) and mark it built for train_fastergs.sbatch
# No macro backward, no profiling hooks, nothing from FasterGSPlus. The FasterGS checkout itself is only read.
#   bash alt_reinit/make_reinit_variant.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-$(ws_find -F cat 3dgs-data)}
METHODS=$DATA/nerficg/src/Methods
SRC=$METHODS/FasterGS
VARIANT=FasterGSReinit
VLOW=fastergsreinit
DST=$METHODS/$VARIANT
BASE=${BASE:-1144604}
BUILD=${BUILD:-1}
VENV=$DATA/fastergs-venv
MARK=.alt-reinit-variant
REV=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)

if [ -e "$DST" ]; then
  [ -f "$DST/$MARK" ] || { echo "$DST exists but was not made by this script; not touching it"; exit 1; }
  rm -rf "$DST"
fi
git clone -q "$SRC" "$DST"
cd "$DST"
git config user.name "alt-reinit-variant"; git config user.email "alt-reinit-variant@localhost"
git checkout -q -b alt-reinit "$BASE"

git mv FasterGSCudaBackend/FasterGSCudaBackend FasterGSCudaBackend/${VARIANT}CudaBackend
git mv FasterGSCudaBackend ${VARIANT}CudaBackend
git mv fastergs_garden.yaml ${VLOW}_garden.yaml
git ls-files -z | xargs -0 grep -IlZ -e FasterGS -e fastergs | xargs -0 sed -i -e "s/FasterGS/$VARIANT/g" -e "s/fastergs/$VLOW/g"
git commit -qam "Rename FasterGS -> $VARIANT"

for tp in cluster/patches/fastergs_sparse_from.py cluster/patches/fastergs_save_state.py alt_reinit/patches/fastergs_reinit.py; do
  python3 "$REPO/$tp" "$DST/Trainer.py"
  git commit -qam "trainer patch $tp (3dgs-project $REV)"
done
cp "$REPO/alt_reinit/reinit_core.py" "$REPO/alt_reinit/reinit_trainer.py" "$DST/"
git add reinit_core.py reinit_trainer.py
git commit -qm "alt_reinit reinit_core.py + reinit_trainer.py (3dgs-project $REV)"
echo "$REV" > "$MARK"   # untracked marker: made by this script
echo "$MARK" >> .git/info/exclude
git log --oneline -7

if [ "$BUILD" = 1 ]; then
  source "$VENV/bin/activate"
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;9.0"} MAX_JOBS=${MAX_JOBS:-14}
  pip install -q --no-build-isolation --no-deps --force-reinstall "$DST/${VARIANT}CudaBackend"
  cd /  # not the source dir, which shadows the installed package
  python -c "import torch; from ${VARIANT}CudaBackend import _C; from ${VARIANT}CudaBackend.torch_bindings import rasterization as r; print('${VARIANT}CudaBackend ok')"
  touch "$VENV/.built-$VARIANT"
fi
