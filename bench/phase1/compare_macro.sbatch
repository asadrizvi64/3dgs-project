#!/usr/bin/env bash
# Phase 1: macro blend backward (FasterGSMacro) vs Faster-GS (FasterGS @1144604, stock Adam), one node, one job.
#   build     phase1/make_variant.sh: clone FasterGS -> FasterGSMacro, apply phase1/fastergs_macro.patch, build
#   test      GPU parity tests (phase1/tests/test_parity.py); the job stops if any fail
#   bench     synthetic backward timing, stock vs macro (bench/phase1/kernel_bench.py)
#   gradcheck one garden run of FasterGSMacro with FGS_GRADCHECK_EVERY / FGS_BWDTIME_REPS: per-step gradient
#             rel error vs the stock kernel on real views, and blend-backward timing of both kernels (+ STAGE lines)
#   compare   SEEDS x {base = FasterGS, macro = FasterGSMacro}, interleaved, clean timing (no hooks)
#   sbatch --export=ALL,STAGES="build test" bench/phase1/compare_macro.sbatch       (from the repo root)
#   sbatch bench/phase1/compare_macro.sbatch                                         (everything)
#SBATCH --job-name=gs-phase1-macro
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --exclude=c20
#SBATCH --output=bench/phase1/logs/%x-%j.out
set -euo pipefail

STAGES=${STAGES:-"build test bench gradcheck compare"}
SCENE=${SCENE:-garden}
SCALE=${SCALE:-0.25}
SEEDS=${SEEDS:-"0 1 2"}
GC_EVERY=${GC_EVERY:-1000}
GC_REPS=${GC_REPS:-20}
REPO=${SLURM_SUBMIT_DIR:-$PWD}
DATA=$(ws_find -F cat 3dgs-data) || { echo "cat workspace not visible on $(hostname)"; exit 1; }
export DATA

module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
source "$DATA/fastergs-venv/bin/activate"
export TORCH_CUDA_ARCH_LIST="8.0;9.0" MAX_JOBS=14 PIP_CACHE_DIR=$DATA/.pipcache
export PYTHONPATH=$HOME/py-extra${PYTHONPATH:+:$PYTHONPATH}
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader) stages=$STAGES"
echo "repo: $REPO ($(cat "$REPO/.phase1-rev" 2>/dev/null || git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown))"
echo "FasterGS: $(git -C "$DATA/nerficg/src/Methods/FasterGS" log -1 --oneline)"
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has build; then
  echo "########## build"
  bash "$REPO/phase1/make_variant.sh"
fi

if has test; then
  echo "########## parity tests"
  python -m pytest -q -s -p no:cacheprovider "$REPO/phase1/tests/test_parity.py" || { echo "parity tests failed"; exit 1; }
fi

if has bench; then
  echo "########## synthetic kernel timing"
  python "$REPO/bench/phase1/kernel_bench.py" --reps 50
fi

run() {  # method tag seed [extra env...]
  local method=$1 tag=$2 seed=$3; shift 3
  echo; echo "########## $method $tag seed=$seed start $(date +%T)"
  env "$@" SCENE=$SCENE SCALE=$SCALE SEED=$seed METHOD=$method TAG=$tag \
    bash "$REPO/cluster/train_fastergs.sbatch" 2>&1 | tr '\r' '\n' \
    | sed -E 's/.*((GRADCHECK|GRADFLOOR|BWDTIME|STAGE) )/\1/' \
    | grep -E "^(GRADCHECK|GRADFLOOR|BWDTIME|STAGE|CLOCK)|PSNR|Traceback|Error|== run dir|method=" | grep -v "it/s" || true
}

if has gradcheck; then
  echo "########## gradcheck + kernel timing on $SCENE (FasterGSMacro)"
  run FasterGSMacro gc 0 FGS_GRADCHECK_EVERY=$GC_EVERY FGS_BWDTIME_REPS=$GC_REPS FGS_PROF=1 FGS_CURVE_EVERY=100000
fi

if has compare; then
  echo "########## base vs macro"
  for seed in $SEEDS; do
    run FasterGS p1base "$seed"
    run FasterGSMacro p1macro "$seed"
  done
fi

echo; echo "########## SUMMARY (method tag seed ms/it train_s psnr ssim lpips n_gaussians vram_gib run)"
cd "$DATA/nerficg"
for d in $(ls -d output/FasterGS/fastergs_${SCENE}_*_s*p1base_* output/FasterGSMacro/fastergsmacro_${SCENE}_*_s*{p1macro,gc}_* 2>/dev/null | sort); do
  [ -f "$d/timings.txt" ] || continue
  tag=$(basename "$d" | sed -E 's/.*_s[0-9]+([a-z0-9]+)_[0-9]{4}-.*/\1/'); seed=$(basename "$d" | sed -E 's/.*_s([0-9]+)[a-z].*/\1/')
  ms=$(grep -A2 "^training_iteration:" "$d/timings.txt" | awk -F': ' '/per iteration/{print $2}')
  met=$(tail -1 "$d"/test_*/metrics_8bit.txt | sed 's/PSNR://; s/SSIM://; s/LPIPS://')
  ng=$(grep -o 'N_Gaussians:[0-9]*' "$d/n_gaussians.txt" | cut -d: -f2)
  vr=$(awk '/allocated:/{print $2; exit}' "$d/vram_stats.txt")
  echo "RESULT $(basename "$(dirname "$d")") $tag $seed $ms $(awk -v m="$ms" 'BEGIN{printf "%.1f", m*30}') $met $ng $vr $(basename "$d")"
done | grep "_$(date +%Y-%m-%d)" || true
