#!/usr/bin/env bash
# Phase 2 step 1: build FasterGSPlus (Faster-GS @1144604 + macro blend backward + lean Adam + SELECTIVE_FROM switch)
# and check it before any benchmark. Flags:
#   A  macro blend backward      env FGS_BLEND_BWD=macro (variant default) | taming (stock kernel)
#   B  selective Adam from n     TRAINING.OPTIMIZER.SELECTIVE_FROM: n   (-1 / absent = off)
#   C  bf16 SH-rest moments      TRAINING.OPTIMIZER.SH_REST_BF16_MOMENTS: true
# Stages (STAGES="..."):
#   build    phase1/make_variant.sh with VARIANT=FasterGSPlus and the switch/clock trainer patch
#   unit     phase1 parity tests and the lean-Adam kernel tests, both against FasterGSPlusCudaBackend
#   switch   two short garden runs (1200 it): all flags on with SELECTIVE_FROM=600, and all flags off;
#            checks the switch fires exactly at 600 (and never when off) and the clock reports dense/sparse correctly
#   sanity   garden seed 0, 30k: stock FasterGS, FasterGSPlus all off, FasterGSPlus A+B+C (one node, interleaved)
#   factorial  (step 2, not in the default STAGES) 8 arms A x B x C on/off on garden, SEEDS (default 0 1 2), 30k,
#            interleaved per seed with the arm order rotated by seed; B = SELECTIVE_FROM=15000
#   sbatch phase2/build_plus.sbatch      (from the repo root)
#   STAGES=factorial sbatch --job-name=gs-phase2-factorial --time=03:30:00 phase2/build_plus.sbatch
#SBATCH --job-name=gs-phase2-build
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=c20
#SBATCH --output=phase2/logs/%x-%j.out
set -euo pipefail
STAGES=${STAGES:-"build unit switch sanity"}
REPO=${SLURM_SUBMIT_DIR:-$PWD}
DATA=$(ws_find -F cat 3dgs-data) || { echo "cat workspace not visible on $(hostname)"; exit 1; }
export DATA
module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
source "$DATA/fastergs-venv/bin/activate"
export TORCH_CUDA_ARCH_LIST="8.0;9.0" MAX_JOBS=14 PIP_CACHE_DIR=$DATA/.pipcache
export PYTHONPATH=$HOME/py-extra${PYTHONPATH:+:$PYTHONPATH}
OUT=$REPO/phase2/logs/run_${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) stages=$STAGES repo=$(git -C "$REPO" rev-parse --short HEAD)"
FAILED=()
has() { [[ " $STAGES " == *" $1 "* ]]; }

# the stock checkout gets the same clock hook (and the B switch, off by default) so all arms are timed alike
python "$REPO/cluster/patches/fastergs_sparse_from.py" "$DATA/nerficg/src/Methods/FasterGS/Trainer.py"

if has build; then
  echo "########## build"
  VARIANT=FasterGSPlus TRAINER_PATCHES=cluster/patches/fastergs_sparse_from.py bash "$REPO/phase1/make_variant.sh"
fi

if has unit; then
  echo "########## unit"
  ( cd /tmp && FGS_BACKEND=FasterGSPlusCudaBackend python -m pytest -q -s -p no:cacheprovider "$REPO/phase1/tests/test_parity.py" ) \
    > "$OUT/parity.log" 2>&1 && tail -3 "$OUT/parity.log" || { echo "FAILED parity"; tail -30 "$OUT/parity.log"; FAILED+=(parity); }
  ( cd /tmp && FGS_BACKEND=FasterGSPlusCudaBackend python -I "$REPO/cluster/tests/test_lean_adam.py" ) \
    > "$OUT/lean_adam.log" 2>&1 && grep -c "^PASS" "$OUT/lean_adam.log" | sed 's/^/lean-adam PASS count: /' || { echo "FAILED lean adam"; grep -E "FAIL|Error" "$OUT/lean_adam.log" | head; FAILED+=(lean_adam); }
  grep "^FAIL" "$OUT/lean_adam.log" && FAILED+=(lean_adam_fail_lines) || true
fi

train() {  # name method seed iters opt_flags blend [segk]
  local name=$1 method=$2 seed=$3 iters=$4 flags=$5 blend=$6 segk=${7:-$4}
  rm -f "$OUT/$name.seg"
  echo "##### $name: method=$method seed=$seed iters=$iters flags=$flags blend=$blend $(date +%T)"
  env FGS_BLEND_BWD=$blend FGS_SEGCLOCK=$segk FGS_LOG="$OUT/$name.seg" SCENE=garden SCALE=0.25 SEED=$seed ITERS=$iters \
      METHOD=$method TAG=p2$name OPT_FLAGS=$flags bash "$REPO/cluster/train_fastergs.sbatch" > "$OUT/$name.log" 2>&1 || true
  local RUN; RUN=$(tr '\r' '\n' < "$OUT/$name.log" | sed -n 's/^== run dir //p' | tail -1)
  cat "$OUT/$name.seg" 2>/dev/null || true
  if [ -z "$RUN" ] || [ ! -f "$DATA/nerficg/$RUN/test_$iters/metrics_8bit.txt" ]; then
    echo "FAILED $name"; tr '\r' '\n' < "$OUT/$name.log" | grep -E "Traceback|Error" | head; FAILED+=("$name"); return
  fi
  echo "RESULT $name $(tail -1 "$DATA/nerficg/$RUN/test_$iters/metrics_8bit.txt") $(grep -o 'N_Gaussians:[0-9]*' "$DATA/nerficg/$RUN/n_gaussians.txt") $(grep -A2 '^training_iteration:' "$DATA/nerficg/$RUN/timings.txt" | awk -F': ' '/per iteration/{print "ms_it=" $2}') $(grep allocated "$DATA/nerficg/$RUN/vram_stats.txt" | head -1 | tr -s ' \t' ' ')"
}

if has switch; then
  echo "########## switch"
  train sw_on FasterGSPlus 0 1200 "SELECTIVE_FROM=600,SH_REST_BF16_MOMENTS=true" macro 300
  train sw_off FasterGSPlus 0 1200 "" taming 300
  python - "$OUT/sw_on.seg" "$OUT/sw_off.seg" <<'PY' || FAILED+=(switch_check)
import re, sys
def parse(p):
    t = open(p).read()
    sw = re.findall(r'selective Adam from iteration (\d+)', t)
    seg = {int(a): int(b) for a, b in re.findall(r'SEGCLOCK step=(\d+) .*?sparse=(\d)', t)}
    return sw, seg
on_sw, on = parse(sys.argv[1]); off_sw, off = parse(sys.argv[2])
checks = {
    'on: switch fires once, at 600': on_sw == ['600'],
    'on: dense before, sparse after': on == {300: 0, 600: 0, 900: 1, 1200: 1},
    'off: never switches': off_sw == [],
    'off: dense throughout': off == {300: 0, 600: 0, 900: 0, 1200: 0},
}
for k, v in checks.items():
    print(('PASS ' if v else 'FAIL ') + k)
sys.exit(0 if all(checks.values()) else 1)
PY
fi

if has sanity; then
  echo "########## sanity (garden seed 0, 30k; train time = SEGCLOCK cum_s)"
  train stock FasterGS 0 30000 "" taming
  train plus_off FasterGSPlus 0 30000 "" taming
  train plus_abc FasterGSPlus 0 30000 "SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true" macro
fi

if has factorial; then
  # 8 arms (A x B x C on/off) x seeds, interleaved per seed; the arm order is rotated by the seed so no arm always runs first
  echo "########## factorial (garden, seeds ${SEEDS:-0 1 2}, 30k; train time = SEGCLOCK cum_s)"
  ARMS=(a0b0c0 a1b0c0 a0b1c0 a0b0c1 a1b1c0 a1b0c1 a0b1c1 a1b1c1)
  for seed in ${SEEDS:-0 1 2}; do
    for k in 0 1 2 3 4 5 6 7; do
      arm=${ARMS[$(( (k + 3 * seed) % 8 ))]}
      blend=taming; [ "${arm:1:1}" = 1 ] && blend=macro
      flags=""; [ "${arm:3:1}" = 1 ] && flags="SELECTIVE_FROM=15000"
      [ "${arm:5:1}" = 1 ] && flags="${flags:+$flags,}SH_REST_BF16_MOMENTS=true"
      train "f_${arm}_s$seed" FasterGSPlus "$seed" 30000 "$flags" "$blend"
    done
  done
fi

if [ ${#FAILED[@]} -gt 0 ]; then echo "JOB FAILED: ${FAILED[*]}"; exit 1; fi
echo "JOB OK"
