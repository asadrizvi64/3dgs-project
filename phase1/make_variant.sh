#!/usr/bin/env bash
# (Re)creates a Faster-GS variant method (default FasterGSMacro) next to FasterGS in NeRFICG and builds its CUDA extension:
#   1. clone nerficg/src/Methods/FasterGS at BASE (default 1144604 = c60ea26 + lean Adam) into FasterGSMacro
#   2. apply phase1/fastergs_macro.patch (macro blend backward + FGS_BLEND_BWD switch + GRADCHECK/BWDTIME hooks)
#   3. rename FasterGS -> FasterGSMacro everywhere (method, classes, FasterGSMacroCudaBackend, yaml), as FasterGSFused does
#   4. add the env-gated profiling hooks (cluster/patches/fastergs_profiling.py, off unless FGS_PROF=1)
#   5. apply each script in TRAINER_PATCHES (repo-relative python patch scripts) to the variant's Trainer.py
#   6. pip install the extension into fastergs-venv (BUILD=0 skips) and mark it built for train_fastergs.sbatch
#   VARIANT=FasterGSPlus TRAINER_PATCHES=cluster/patches/fastergs_sparse_from.py bash phase1/make_variant.sh   (phase2)
# The FasterGS checkout itself is only read. Run on a GPU node (or a login node with CUDA loaded) from anywhere:
#   bash phase1/make_variant.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-$(ws_find -F cat 3dgs-data)}
METHODS=$DATA/nerficg/src/Methods
SRC=$METHODS/FasterGS
VARIANT=${VARIANT:-FasterGSMacro}
VLOW=$(echo "$VARIANT" | tr 'A-Z' 'a-z')
TRAINER_PATCHES=${TRAINER_PATCHES:-}
SRC_PATCHES=${SRC_PATCHES:-}   # e.g. phase2/fastergs_readback.patch
DST=$METHODS/$VARIANT
BASE=${BASE:-1144604}
BUILD=${BUILD:-1}
VENV=$DATA/fastergs-venv
MARK=.phase1-variant

if [ -e "$DST" ]; then
  [ -f "$DST/$MARK" ] || { echo "$DST exists but was not made by this script; not touching it"; exit 1; }
  rm -rf "$DST"
fi
git clone -q "$SRC" "$DST"
cd "$DST"
git config user.name "phase1-variant"; git config user.email "phase1-variant@localhost"
git checkout -q -b phase1-macro "$BASE"
git apply "$REPO/phase1/fastergs_macro.patch"
REV=$(cat "$REPO/.phase1-rev" 2>/dev/null || git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
git commit -qam "Phase 1: macro blend backward (3dgs-project $REV)"
for sp in ${SRC_PATCHES:-}; do   # repo-relative source patches on top of the macro patch (FasterGS names, before the rename)
  git apply "$REPO/$sp"
  git add -A   # the patch may add files
  git commit -qm "source patch $sp"
done

git mv FasterGSCudaBackend/FasterGSCudaBackend FasterGSCudaBackend/${VARIANT}CudaBackend
git mv FasterGSCudaBackend ${VARIANT}CudaBackend
git mv fastergs_garden.yaml ${VLOW}_garden.yaml
git ls-files -z | xargs -0 grep -IlZ -e FasterGS -e fastergs | xargs -0 sed -i -e "s/FasterGS/$VARIANT/g" -e "s/fastergs/$VLOW/g"
git commit -qam "Rename FasterGS -> $VARIANT"

python3 "$REPO/cluster/patches/fastergs_profiling.py" "$DST/Trainer.py"
git commit -qam "fgs-prof hooks (cluster/patches/fastergs_profiling.py)"
for tp in $TRAINER_PATCHES; do
  python3 "$REPO/$tp" "$DST/Trainer.py"
  git commit -qam "trainer patch $tp"
done
touch "$MARK"
git log --oneline -6

if [ "$BUILD" = 1 ]; then
  source "$VENV/bin/activate"
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;9.0"} MAX_JOBS=${MAX_JOBS:-14}
  pip install -q --no-build-isolation --no-deps --force-reinstall "$DST/${VARIANT}CudaBackend"
  cd /  # not the source dir, which shadows the installed package
  python -c "import torch; from ${VARIANT}CudaBackend import _C; from ${VARIANT}CudaBackend.torch_bindings import rasterization as r; print('${VARIANT}CudaBackend ok, default blend backward mode', r.BLEND_BACKWARD_MODE)"
  touch "$VENV/.built-$VARIANT"
fi
