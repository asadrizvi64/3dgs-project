#!/usr/bin/env bash
# Phase 2 step 1: build FasterGSPlus (Faster-GS @1144604 + macro blend backward + lean Adam + SELECTIVE_FROM switch)
# and check it before any benchmark. Flags:
#   A  macro blend backward      env FGS_BLEND_BWD=macro (variant default) | taming (stock kernel)
#   B  selective Adam from n     TRAINING.OPTIMIZER.SELECTIVE_FROM: n   (-1 / absent = off)
#   C  bf16 SH-rest moments      TRAINING.OPTIMIZER.SH_REST_BF16_MOMENTS: true
# Current stack (phase2 step 5h, adopted 2026-11-07): BASE (default FasterGSPlusDS) = A+B+C + D + S, D = deferred warp
# reduction v2 (phase2/deferred_reduce_v2.patch, FGS_DEFERRED_REDUCE=0 turns it off at run time), S = coalesced SH-rest access
# in preprocess backward (phase2/sh_coalesce.patch); arm label BASE_ARM (default abcds). BASE=FasterGSPlusD (arm abcd) is the
# step 5f-5h stack A+B+C+D.
# The build stage builds BASE by default; any BUILD_VARIANT is built on top of BASE's patches (STACK_ON_BASE=0: plain
# A+B+C + BUILD_SRC_PATCHES, as in steps 4c-5f) and checked for equivalence against BASE. rbbench, ncublend, timerctl,
# profile and hostprof use BASE as their A+B+C(+D) arm. BASE=FasterGSPlus reproduces the pre-5f setup (arm abc).
# switch, sanity, factorial, bench and stages stay on FasterGSPlus (A+B+C, historical steps 1-3).
# Stages (STAGES="..."):
#   build    phase1/make_variant.sh with VARIANT=FasterGSPlus and the switch/clock trainer patch
#   unit     phase1 parity tests and the lean-Adam kernel tests, both against FasterGSPlusCudaBackend
#   switch   two short garden runs (1200 it): all flags on with SELECTIVE_FROM=600, and all flags off;
#            checks the switch fires exactly at 600 (and never when off) and the clock reports dense/sparse correctly
#   sanity   garden seed 0, 30k: stock FasterGS, FasterGSPlus all off, FasterGSPlus A+B+C (one node, interleaved)
#   factorial  (step 2, not in the default STAGES) 8 arms A x B x C on/off on garden, SEEDS (default 0 1 2), 30k,
#            interleaved per seed with the arm order rotated by seed; B = SELECTIVE_FROM=15000
#   bench    (step 3, not in the default STAGES) one Mip-NeRF360 scene (SCENE, scale 0.25 outdoor / 0.5 indoor), SEEDS,
#            30k: stock FasterGS, FasterGSPlus A+B, FasterGSPlus A+B+C, interleaved per seed, arm order rotated by seed.
#            BENCH_ARMS picks the arms (default "stock ab abc"); step 6 uses BENCH_ARMS="stock $BASE_ARM" (stock vs the
#            adopted stack). With BENCH_ARMS set, every run also logs FGS_WALL (sync only at 0, 15k and 30k).
#   stages   (renders during training, not in the default STAGES) one scene, seed SEEDS (default 0), stock / A+B / A+B+C:
#            full test-set renders and metrics every STAGE_EVERY (default 1000) iterations, training clock at each;
#            renders copied to $ARCHIVE/stages/<scene>/<run>/ (outside the expiring workspace),
#            intermediate checkpoints deleted. Not for timing claims: the intermediate evaluations change the run.
#   profile  (step 4, not in the default STAGES) one scene, seed 0, stock vs A+B+C: per arm one full 30k run with
#            FGS_PROF=1 (STAGE fwd/bwd/opt CUDA events every 250 steps, CURVE ms/step every 500; the hooks sync the
#            host every step, so these are not benchmark times) and two nsys captures of 50 steps at 5000 and 20000
#            (NVTX fwd/bwd/opt); reports in $ARCHIVE/profile/<job>/, CSV summaries in phase2/logs/run_<job>/
#   ctltest  (with build/unit/switch) the timer-control path at 1200 iterations on garden seed 0, whose real output must
#            pass analyze_timerctl.py and record_runs.py before the full timerctl runs are trusted
#   timerctl (step 4b) one scene, SEEDS (default 0 1 2): stock and A+B+C, each with NeRFICG's per-step timer ON (the
#            benchmark protocol: device sync after every step) and OFF; interleaved, order rotated by seed. Every run
#            also logs FGS_WALL (unprofiled wall time over 0-15k and 15k-30k, device synced only at those boundaries)
#            and a hash of the training-view sequence, so matched runs can be checked to have seen the same views
#   rbbench  (step 4c / 5b) one scene, SEEDS (default 0 1 2): A+B+C vs A+B+C built as CMP_VARIANT (default FasterGSPlusRB),
#            timer ON, FGS_WALL; interleaved, order rotated by seed. Step 5b: CMP_VARIANT=FasterGSPlusCull (ellipse culling)
#   ncublend (step 5c) SCENES (default garden room), seed 0: Nsight Compute (--set full) on the forward blend and macro
#            backward of 2 steps at NCU_STEP (default 20000), A+B+C vs FasterGSPlusCull. Not timing runs.
#   ncupre   (step 5g) SCENES (default garden room), seed 0, BASE only: Nsight Compute (--set full) on the forward and
#            backward preprocess kernels of 2 steps at NCU_STEP (default 20000), plus a per-SASS export of one preprocess
#            backward launch (<scene>_<arm>.pre_sass.csv). Not timing runs.
#   blendstats (step 5) SCENES (default garden room), seed 0, A+B+C built as FasterGSPlusStats (BUILD_VARIANT=FasterGSPlusStats
#            BUILD_SRC_PATCHES=phase2/blend_stats.patch): blend-kernel work counters over 50-step windows (BS_STARTS)
#   sbatch phase2/build_plus.sbatch      (from the repo root)
#   STAGES=factorial sbatch --job-name=gs-phase2-factorial --time=03:30:00 phase2/build_plus.sbatch
#   STAGES=bench SCENE=room sbatch --job-name=gs-bench-room --time=02:30:00 phase2/build_plus.sbatch
#   STAGES=bench SCENE=room BENCH_ARMS="stock abcds" sbatch --job-name=gs-bench6-room --time=01:30:00 phase2/build_plus.sbatch   (step 6)
#   STAGES=stages SCENE=room sbatch --job-name=gs-stages-room --time=02:30:00 phase2/build_plus.sbatch
#   STAGES=profile SCENE=room sbatch --job-name=gs-prof-room --time=02:00:00 phase2/build_plus.sbatch
#   STAGES=timerctl SCENE=room sbatch --job-name=gs-timer-room --time=01:30:00 phase2/build_plus.sbatch
#   STAGES="build unit" BUILD_VARIANT=FasterGSPlusCull BUILD_SRC_PATCHES=phase2/ellipse_cull.patch sbatch --job-name=gs-cull-build phase2/build_plus.sbatch
#   STAGES=rbbench SCENE=garden CMP_VARIANT=FasterGSPlusCull CMP_ARM=abce sbatch --job-name=gs-cull-garden --time=00:40:00 phase2/build_plus.sbatch
#   STAGES=ncublend sbatch --job-name=gs-ncublend --time=00:40:00 phase2/build_plus.sbatch
#   STAGES="build unit" BUILD_VARIANT=FasterGSPlusDR BUILD_SRC_PATCHES=phase2/deferred_reduce.patch sbatch --job-name=gs-dr-build phase2/build_plus.sbatch
#   STAGES=ncublend CMP_VARIANT=FasterGSPlusDR CMP_ARM=abcd sbatch --job-name=gs-dr-ncu phase2/build_plus.sbatch
#   STAGES="build unit" sbatch --job-name=gs-base-build phase2/build_plus.sbatch   (builds BASE = FasterGSPlusDS, the adopted stack)
#   STAGES="build unit" BUILD_VARIANT=FasterGSPlusX BUILD_SRC_PATCHES=phase2/x.patch sbatch ...  (X on top of the stack, vs BASE)
#   historical (pre-5f, plain A+B+C): BASE=FasterGSPlus ..., or STACK_ON_BASE=0 for a variant build, e.g.
#   STAGES="build unit" STACK_ON_BASE=0 BUILD_VARIANT=FasterGSPlusDR2 BUILD_SRC_PATCHES=phase2/deferred_reduce_v2.patch sbatch --job-name=gs-dr2-build phase2/build_plus.sbatch
#   STAGES=ncublend CMP_VARIANT=FasterGSPlusDR2 CMP_ARM=abcd2 sbatch --job-name=gs-dr2-ncu phase2/build_plus.sbatch
#   STAGES=ncupre sbatch --job-name=gs-ncupre --time=00:40:00 phase2/build_plus.sbatch
#   STAGES="build unit" BUILD_VARIANT=FasterGSPlusDS BUILD_SRC_PATCHES=phase2/sh_coalesce.patch sbatch --job-name=gs-ds-build phase2/build_plus.sbatch
#   STAGES=ncupre sbatch --job-name=gs-ds-ncu ...   (step 5h ran this before adoption, with BASE=FasterGSPlusDS set explicitly)
#   STAGES=rbbench SCENE=garden CMP_VARIANT=FasterGSPlusDS CMP_ARM=abcds sbatch --job-name=gs-ds-garden --time=00:50:00 phase2/build_plus.sbatch
#   STAGES=exp1smoke sbatch --job-name=gs-x1-smoke --time=00:30:00 phase2/build_plus.sbatch   (phase3/PLAN.md engineering checks)
#   STAGES=exp1 SCENE=garden sbatch --job-name=gs-x1-garden --time=03:00:00 phase2/build_plus.sbatch   (phase 3 Experiment 1)
#   STAGES="upbuild upequiv" sbatch --job-name=gs-up-build --time=01:00:00 phase2/build_plus.sbatch   (upstream/README.md)
#   STAGES=upbench SCENE=garden sbatch --job-name=gs-up-garden --time=01:30:00 phase2/build_plus.sbatch
#   STAGES=upncu sbatch --job-name=gs-up-ncu --time=00:40:00 phase2/build_plus.sbatch
#   STAGES="build unit blendstats" BUILD_VARIANT=FasterGSPlusStats BUILD_SRC_PATCHES=phase2/blend_stats.patch sbatch --job-name=gs-blendstats phase2/build_plus.sbatch
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
ARCHIVE=${ARCHIVE:-/projects/p_scads_mlhpc/syas272h/3dgs-archive}   # durable results (project storage; moved out of $HOME 2026-10-29, which has a 50 GB quota)
# NOTE: /projects is read-only on the compute nodes (job 4503798): stages that write $ARCHIVE (stages, profile, hostprof)
# need ARCHIVE pointed at the workspace, with a copy to /projects from a login node afterwards
export DATA
module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
source "$DATA/fastergs-venv/bin/activate"
export TORCH_CUDA_ARCH_LIST="8.0;9.0" MAX_JOBS=14 PIP_CACHE_DIR=$DATA/.pipcache
export PYTHONPATH=$HOME/py-extra${PYTHONPATH:+:$PYTHONPATH}
OUT=$REPO/phase2/logs/run_${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) stages=$STAGES repo=$(git -C "$REPO" rev-parse --short HEAD)"
FAILED=()
BSCENE=garden; BSCALE=0.25
has() { [[ " $STAGES " == *" $1 "* ]]; }
BASE=${BASE:-FasterGSPlusDS}
case $BASE in
  FasterGSPlusDS) BASE_PATCHES=${BASE_PATCHES-phase2/deferred_reduce_v2.patch phase2/sh_coalesce.patch}; BASE_ARM=${BASE_ARM:-abcds} ;;
  FasterGSPlusD) BASE_PATCHES=${BASE_PATCHES-phase2/deferred_reduce_v2.patch}; BASE_ARM=${BASE_ARM:-abcd} ;;
  FasterGSPlus)  BASE_PATCHES=${BASE_PATCHES-}; BASE_ARM=${BASE_ARM:-abc} ;;
  *) : "${BASE_PATCHES?set BASE_PATCHES for BASE=$BASE}"; BASE_ARM=${BASE_ARM:?set BASE_ARM for BASE=$BASE} ;;
esac
BV=${BUILD_VARIANT:-$BASE}
if [ "$BV" = FasterGSPlus ]; then BPATCHES="${BUILD_SRC_PATCHES:-}"
elif [ "$BV" = "$BASE" ]; then BPATCHES="$BASE_PATCHES"
elif [ "${STACK_ON_BASE:-1}" = 0 ]; then BPATCHES="${BUILD_SRC_PATCHES:-}"
else BPATCHES="$BASE_PATCHES ${BUILD_SRC_PATCHES:-}"; fi
echo "base=$BASE ($BASE_ARM, patches: ${BASE_PATCHES:-none})"

# the stock checkout gets the same clock hook (and the B switch, off by default) so all arms are timed alike
python "$REPO/cluster/patches/fastergs_sparse_from.py" "$DATA/nerficg/src/Methods/FasterGS/Trainer.py"

if has build; then
  echo "########## build"
  echo "variant=$BV source patches: ${BPATCHES:-none}"
  VARIANT=$BV SRC_PATCHES="$BPATCHES" TRAINER_PATCHES=cluster/patches/fastergs_sparse_from.py bash "$REPO/phase1/make_variant.sh"
  # per-kernel registers / spills of the macro backward instantiations (mangled names; template args at the end)
  so=$(ls "$DATA"/fastergs-venv/lib/python3.12/site-packages/${BV}CudaBackend/_C*.so 2>/dev/null | head -1) || true
  if [ -n "$so" ]; then   # diagnostics only, never fails the job; first block = first embedded arch
    { cuobjdump --dump-resource-usage "$so" 2>/dev/null | grep -A1 "macro" | grep -v "^--" | paste - -       | sed -E "s/.*macro_cuILb([01])ELi([48])ELb([01])ELb([01])ELb([01])E[^ ]*/abs=\1 warps=\2 skip=\3 tr=\4 dr=\5/; s/Resource usage://; s/ +/ /g"       | cut -d" " -f1-9 | sed "s/^/RESOURCES /"; } || true
  fi
fi

if has unit; then
  echo "########## unit"
  ( cd /tmp && FGS_BACKEND=${BV}CudaBackend python -m pytest -q -s -p no:cacheprovider "$REPO/phase1/tests/test_parity.py" ) \
    > "$OUT/parity.log" 2>&1 && tail -3 "$OUT/parity.log" || { echo "FAILED parity"; tail -30 "$OUT/parity.log"; FAILED+=(parity); }
  ( cd /tmp && FGS_BACKEND=${BV}CudaBackend python -I "$REPO/cluster/tests/test_lean_adam.py" ) \
    > "$OUT/lean_adam.log" 2>&1 && grep -c "^PASS" "$OUT/lean_adam.log" | sed 's/^/lean-adam PASS count: /' || { echo "FAILED lean adam"; grep -E "FAIL|Error" "$OUT/lean_adam.log" | head; FAILED+=(lean_adam); }
  grep "^FAIL" "$OUT/lean_adam.log" && FAILED+=(lean_adam_fail_lines) || true
  if [ "$BV" != FasterGSPlus ]; then   # the variant must reproduce its reference on identical inputs: BASE, or A+B+C for BASE
    # itself and for STACK_ON_BASE=0 builds
    REF=$BASE; { [ "$BV" = "$BASE" ] || [ "${STACK_ON_BASE:-1}" = 0 ]; } && REF=FasterGSPlus
    echo "equivalence: $BV vs $REF"
    ( cd /tmp && FGS_REF=${REF}CudaBackend FGS_NEW=${BV}CudaBackend python -m pytest -q -s -p no:cacheprovider "$REPO/phase2/tests/test_readback_equiv.py" )       > "$OUT/equiv.log" 2>&1 && { grep "^EQUIV" "$OUT/equiv.log" | sort -u | head -30; tail -2 "$OUT/equiv.log"; } || { echo "FAILED equivalence"; tail -40 "$OUT/equiv.log"; FAILED+=(equivalence); }
  fi
fi

train() {  # name method seed iters opt_flags blend [segk]
  local name=$1 method=$2 seed=$3 iters=$4 flags=$5 blend=$6 segk=${7:-$4}
  rm -f "$OUT/$name.seg"
  echo "##### $name: method=$method seed=$seed iters=$iters flags=$flags blend=$blend $(date +%T)"
  env ${TRAIN_ENV:-} FGS_BLEND_BWD=$blend FGS_SEGCLOCK=$segk FGS_LOG="$OUT/$name.seg" BACKUP_INTERVAL=${TRAIN_BACKUP:--1} SCENE=$BSCENE SCALE=$BSCALE SEED=$seed ITERS=$iters \
      METHOD=$method TAG=p2$name OPT_FLAGS=$flags bash "$REPO/cluster/train_fastergs.sbatch" > "$OUT/$name.log" 2>&1 || true
  local RUN; RUN=$(tr '\r' '\n' < "$OUT/$name.log" | sed -n 's/^== run dir //p' | tail -1)
  cat "$OUT/$name.seg" 2>/dev/null || true
  # keep every run's small result files with the job logs (checkpoints and renders stay on the workspace)
  if [ -n "$RUN" ] && [ -d "$DATA/nerficg/$RUN" ]; then
    echo "RUNDIR $name $RUN"
    mkdir -p "$OUT/runs/$name"; echo "$RUN" > "$OUT/runs/$name/rundir.txt"
    ( cd "$DATA/nerficg/$RUN" && cp --parents -t "$OUT/runs/$name" training_config.yaml timings.txt n_gaussians.txt vram_stats.txt \
        gpu_clock_summary.txt gpu_clock_log.csv test_*/metrics_8bit.txt 2>/dev/null ) || true
    if [ "${TRAIN_BACKUP:--1}" -gt 0 ]; then   # stage-render runs: keep every stage's renders outside the workspace
      local ARC=$ARCHIVE/stages/$BSCENE/$name
      mkdir -p "$ARC"; echo "$RUN" > "$ARC/rundir.txt"
      rsync -a --include='test_30000/rgb_gt/' --include='test_30000/rgb_gt/**' --exclude='test_*/rgb_gt/' --include='test_*/' --include='test_*/**' --include='training_config.yaml' --include='timings.txt'           --include='n_gaussians.txt' --exclude='*' "$DATA/nerficg/$RUN/" "$ARC/" && echo "STAGEARCHIVE $name $ARC $(ls -d "$ARC"/test_* | wc -l) stages"
      find "$DATA/nerficg/$RUN/checkpoints" -name '*.pt' ! -name 'final.pt' -delete 2>/dev/null || true
    fi
  fi
  if [ -z "$RUN" ] || [ ! -f "$DATA/nerficg/$RUN/test_$iters/metrics_8bit.txt" ]; then
    echo "FAILED $name"; tr '\r' '\n' < "$OUT/$name.log" | grep -E "Traceback|Error" | head; FAILED+=("$name"); return
  fi
  # lowest single test image (index, PSNR) and the number of test images, to catch one-view artifacts
  echo "PERIMG $name $(awk -F'	' '/^Image/{f=1;next} f&&$1~/^[0-9]+$/{n++; if(min==""||$2<min){min=$2;i=$1}} END{print "n_images=" n " min_image=" i " min_psnr=" min}' "$DATA/nerficg/$RUN/test_$iters/metrics_8bit.txt")"
  # ms_it= is always written (empty when NeRFICG's timer is off and there is no timings.txt)
  local MSIT; MSIT=$(grep -A2 '^training_iteration:' "$DATA/nerficg/$RUN/timings.txt" 2>/dev/null | awk -F': ' '/per iteration/{gsub(/[ \t]+$/, "", $2); print $2}' || true)
  echo "RESULT $name $(tail -1 "$DATA/nerficg/$RUN/test_$iters/metrics_8bit.txt") $(grep -o 'N_Gaussians:[0-9]*' "$DATA/nerficg/$RUN/n_gaussians.txt") ms_it=${MSIT} $(grep allocated "$DATA/nerficg/$RUN/vram_stats.txt" 2>/dev/null | head -1 | tr -s ' \t' ' ')"
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
for nm, p in (('on', sys.argv[1]), ('off', sys.argv[2])):   # clock hook v4 fields
    t = open(p).read()
    segs = re.findall(r'^SEGCLOCK .*$', t, re.M)
    checks[f'{nm}: v4 SEGCLOCK lines carry N and peak_alloc_gib'] = len(segs) == 4 and all(re.search(r' N=\d+ peak_alloc_gib=[\d.]+$', l) for l in segs)
    checks[f'{nm}: one FGS_TRAIN_END line with N and peaks'] = len(re.findall(r'^FGS_TRAIN_END N=\d+ peak_alloc_gib=[\d.]+ peak_reserved_gib=[\d.]+$', t, re.M)) == 1
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

if has bench; then
  BSCENE=${SCENE:?set SCENE for the bench stage}
  case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
  echo "########## bench (scene $BSCENE scale $BSCALE, seeds ${SEEDS:-0 1 2}, 30k; train time = SEGCLOCK cum_s)"
  BARMS=(${BENCH_ARMS:-stock ab abc}); NA=${#BARMS[@]}
  BWALL=""; [ -n "${BENCH_ARMS:-}" ] && BWALL="FGS_WALL=0,15000,30000"
  echo "arms: ${BARMS[*]}"
  for seed in ${SEEDS:-0 1 2}; do
    for (( k = 0; k < NA; k++ )); do
      arm=${BARMS[$(( (k + seed) % NA ))]}
      case $arm in
        stock) TRAIN_ENV="$BWALL" train "b_${BSCENE}_stock_s$seed" FasterGS "$seed" 30000 "" taming ;;
        ab)    TRAIN_ENV="$BWALL" train "b_${BSCENE}_ab_s$seed" FasterGSPlus "$seed" 30000 "SELECTIVE_FROM=15000" macro ;;
        abc)   TRAIN_ENV="$BWALL" train "b_${BSCENE}_abc_s$seed" FasterGSPlus "$seed" 30000 "SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true" macro ;;
        "$BASE_ARM") TRAIN_ENV="$BWALL" train "b_${BSCENE}_${arm}_s$seed" $BASE "$seed" 30000 "SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true" macro ;;
        *) echo "FAILED bench: unknown arm $arm"; FAILED+=("bench_$arm") ;;
      esac
    done
  done
fi

if has stages; then
  BSCENE=${SCENE:?set SCENE for the stages stage}
  case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
  TRAIN_BACKUP=${STAGE_EVERY:-1000}
  echo "########## stages (scene $BSCENE scale $BSCALE, seeds ${SEEDS:-0}, renders every $TRAIN_BACKUP it; clock every $TRAIN_BACKUP it)"
  for seed in ${SEEDS:-0}; do
    train "r_${BSCENE}_stock_s$seed" FasterGS "$seed" 30000 "" taming "$TRAIN_BACKUP"
    train "r_${BSCENE}_ab_s$seed" FasterGSPlus "$seed" 30000 "SELECTIVE_FROM=15000" macro "$TRAIN_BACKUP"
    train "r_${BSCENE}_abc_s$seed" FasterGSPlus "$seed" 30000 "SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true" macro "$TRAIN_BACKUP"
  done
  TRAIN_BACKUP=-1
fi

nsys_window() {  # name method flags blend start: nsys capture of steps start..start+50; validated before it counts
  local name=$1 method=$2 flags=$3 blend=$4 a=$5 b=$(( $5 + 50 ))
  local NSYS; NSYS=$(command -v nsys || true)
  [ -n "$NSYS" ] || { echo "FAILED $name: nsys not found"; FAILED+=("$name"); return; }
  local REP=$ARCHIVE/profile/${SLURM_JOB_ID:-local}; mkdir -p "$REP"
  echo "##### $name: method=$method seed=0 window=$a-$b flags=$flags blend=$blend $(date +%T)"
  local rc=0
  env FGS_BLEND_BWD=$blend FGS_PROF_RANGE=$a,$b FGS_PROF_EXIT=$b SCENE=$BSCENE SCALE=$BSCALE SEED=0 METHOD=$method TAG=p2$name OPT_FLAGS=$flags LAUNCHER="$NSYS profile -t cuda,nvtx ${NSYS_EXTRA:-}--capture-range=cudaProfilerApi --capture-range-end=stop -f true -o $REP/$name" bash "$REPO/cluster/train_fastergs.sbatch" > "$OUT/$name.log" 2>&1 || rc=$?
  local why=""
  [ "$rc" = 0 ] || why="$why exit status $rc;"
  # (the 'FGS_PROF_EXIT' print is not checked: stdout lines get lost in the progress-bar output; the 50-step kernel count below is the window check)
  ! tr '\r' '\n' < "$OUT/$name.log" | grep -q "^Traceback" || why="$why Python traceback in the log;"
  [ -f "$REP/$name.nsys-rep" ] || why="$why no report;"
  if [ -z "$why" ]; then
    for r in cuda_gpu_kern_sum nvtx_sum nvtx_gpu_proj_sum; do
      "$NSYS" stats --force-export=true --report $r --format csv --output "$OUT/$name" "$REP/$name.nsys-rep" >/dev/null 2>&1 || why="$why nsys report $r failed;"
    done
    # the window must hold exactly 50 training steps: one forward blend kernel per step
    local nb; nb=$(python3 -c "import csv,sys; print(sum(int(r['Instances']) for r in csv.DictReader(open(sys.argv[1])) if 'forward::blend_cu' in r['Name']))" "$OUT/${name}_cuda_gpu_kern_sum.csv" 2>/dev/null || echo 0)
    [ "$nb" = 50 ] || why="$why forward blend launches = $nb, expected 50;"
    if [ -n "${NSYS_EXTRA:-}" ]; then   # backtrace captures: PyTorch op ranges, and resolved Python frames on the kernel launches
      grep -q "aten::" "$OUT/${name}_nvtx_sum.csv" 2>/dev/null || why="$why no PyTorch (aten::) NVTX ranges;"
      # runtime callchainId -> CUDA_CALLCHAINS only; at least half of the kernel launches must carry a .py frame
      python3 "$REPO/phase2/host_gaps.py" "$REP/$name.sqlite" --check-backtraces || why="$why kernel launches lack resolved Python frames;"
    fi
  fi
  if [ -n "$why" ]; then
    echo "FAILED $name:$why"; tr '\r' '\n' < "$OUT/$name.log" | grep -E "Traceback|Error" | head; FAILED+=("$name"); return
  fi
  echo "NSYS $name validated (50 steps, $nb forward blends) rep $REP/$name.nsys-rep"
}

run_timerctl() {  # scene, seeds, iterations: stock / A+B+C x NeRFICG timer on / off, interleaved, order rotated by seed
  BSCENE=$1; local seeds=$2 TC=$3
  case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
  echo "########## timerctl (scene $BSCENE scale $BSCALE, seeds $seeds, $TC iterations; timer on/off x stock/A+B+C)"
  local CONFIGS=(stock_on ${BASE_ARM}_on stock_off ${BASE_ARM}_off) seed k cfg arm tm off m fl bl
  for seed in $seeds; do
    for k in 0 1 2 3; do
      cfg=${CONFIGS[$(( (k + seed) % 4 ))]}
      arm=${cfg%_*}; tm=${cfg#*_}
      off=0; [ "$tm" = off ] && off=1
      if [ $arm = stock ]; then m=FasterGS; fl=""; bl=taming; else m=$BASE; fl="SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true"; bl=macro; fi
      TRAIN_ENV="FGS_WALL=0,$(( TC / 2 )),$TC TIMING_OFF=$off" train "t_${BSCENE}_${arm}_${tm}_s$seed" $m "$seed" "$TC" "$fl" $bl
    done
  done
}

if has ctltest; then   # short end-to-end test of the timer-control path: real runs, real output through analyzer and recorder
  run_timerctl garden 0 1200
  LOG=$REPO/phase2/logs/${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:-local}.out
  echo "########## ctltest checks on $LOG"
  python3 "$REPO/phase2/analyze_timerctl.py" "$LOG" --seeds 0 --iters 1200 > "$OUT/ctltest_analysis.md" 2>&1 \
    && { echo "PASS ctltest: analyzer validated all 4 runs"; grep -E "^\| (stock|A\+B\+C)" "$OUT/ctltest_analysis.md"; } \
    || { echo "FAIL ctltest: analyzer"; cat "$OUT/ctltest_analysis.md"; FAILED+=(ctltest_analyzer); }
  ( cd "$REPO" && python3 phase2/record_runs.py "$LOG" --dry-run ) > "$OUT/ctltest_record.txt" 2>&1 || true
  python3 - "$OUT/ctltest_record.txt" <<'PYC' && echo "PASS ctltest: recorder parsed 4 runs (2 timer ON, 2 timer OFF), none failed" || { echo "FAIL ctltest: recorder"; cat "$OUT/ctltest_record.txt"; FAILED+=(ctltest_recorder); }
import csv, io, sys
t = open(sys.argv[1]).read()
rows = [r for r in csv.reader(io.StringIO(t.split('\n', 1)[1])) if len(r) > 5 and 'run=t_garden_' in ','.join(r)]
notes = [','.join(r) for r in rows]
ok = (len(rows) == 4 and not any('FAILED' in n for n in notes)
      and sum('timer OFF' in n for n in notes) == 2 and sum('timer ON' in n for n in notes) == 2
      and all('wall clock' in n for n in notes))
sys.exit(0 if ok else 1)
PYC
fi

if has rbbench; then   # A+B+C vs A+B+C built with a source patch (CMP_VARIANT, default FasterGSPlusRB = readback patch; arm
  # name CMP_ARM), timer ON (benchmark protocol), wall clock on
  BSCENE=${SCENE:?set SCENE for the rbbench stage}
  CMP_VARIANT=${CMP_VARIANT:-FasterGSPlusRB}; CMP_ARM=${CMP_ARM:-abcr}
  case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
  echo "########## rbbench (scene $BSCENE scale $BSCALE, seeds ${SEEDS:-0 1 2}; A+B+C vs A+B+C as $CMP_VARIANT)"
  for seed in ${SEEDS:-0 1 2}; do
    for k in 0 1; do
      if [ $(( (k + seed) % 2 )) = 0 ]; then arm=$BASE_ARM; m=$BASE; else arm=$CMP_ARM; m=$CMP_VARIANT; fi
      TRAIN_ENV="FGS_WALL=0,15000,30000" train "x_${BSCENE}_${arm}_s$seed" $m "$seed" 30000 "SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true" macro
    done
  done
fi

if has ncublend; then   # step 5c: Nsight Compute on the blend kernels, A+B+C vs CMP_VARIANT (default FasterGSPlusCull, arm
  # CMP_ARM, default abce), steps NCU_STEP..+2
  # one process per scene and arm: training runs normally to NCU_STEP (default 20000), ncu captures the forward blend and the
  # macro backward of 2 steps (--set full), then the process exits. Reports in $DATA/ncu/<job>/, details CSV with the logs.
  NCU=$(command -v ncu || true)
  [ -n "$NCU" ] || { echo "FAILED ncublend: ncu not found"; FAILED+=(ncublend); }
  NS=${NCU_STEP:-20000}; REP=$DATA/ncu/${SLURM_JOB_ID:-local}; mkdir -p "$REP"   # /projects is read-only on compute nodes
  for BSCENE in ${SCENES:-garden room}; do
    case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
    for arm in $BASE_ARM ${CMP_ARM:-abce}; do
      [ -n "$NCU" ] || break
      if [ $arm = "$BASE_ARM" ]; then m=$BASE; else m=${CMP_VARIANT:-FasterGSPlusCull}; fi
      name=u_${BSCENE}_${arm}_$NS
      echo "##### $name: method=$m seed=0 window=$NS-$(( NS + 2 )) $(date +%T)"
      env FGS_BLEND_BWD=macro FGS_PROF_RANGE=$NS,$(( NS + 2 )) FGS_PROF_EXIT=$(( NS + 2 )) SCENE=$BSCENE SCALE=$BSCALE SEED=0 METHOD=$m TAG=p2$name           OPT_FLAGS="SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true"           LAUNCHER="$NCU --profile-from-start off --kernel-name regex:blend_cu|blend_backward_macro_cu --launch-count 4 --set full --import-source no -f -o $REP/$name"           bash "$REPO/cluster/train_fastergs.sbatch" > "$OUT/$name.log" 2>&1 || true
      if [ -f "$REP/$name.ncu-rep" ]; then
        "$NCU" --import "$REP/$name.ncu-rep" --page details --csv > "$OUT/$name.details.csv" 2>/dev/null || true
        "$NCU" --import "$REP/$name.ncu-rep" --page raw --csv > "$OUT/$name.raw.csv" 2>/dev/null || true
        echo "NCU $name kernels=$(python3 -c "import csv,sys; print(len({r['ID'] for r in csv.DictReader(open(sys.argv[1]))}))" "$OUT/$name.details.csv" 2>/dev/null || echo 0) rep $REP/$name.ncu-rep"
      else
        echo "FAILED $name: no report"; tr '' '
' < "$OUT/$name.log" | grep -E "==ERROR==|==WARNING==|Traceback|Error" | head; FAILED+=("$name")
      fi
    done
  done
fi

if has ncupre; then   # step 5g: Nsight Compute on the preprocess kernels of BASE (forward preprocess_cu, preprocess_backward_cu)
  NCU=$(command -v ncu || true)
  [ -n "$NCU" ] || { echo "FAILED ncupre: ncu not found"; FAILED+=(ncupre); }
  NS=${NCU_STEP:-20000}; REP=$DATA/ncu/${SLURM_JOB_ID:-local}; mkdir -p "$REP"   # /projects is read-only on compute nodes
  for BSCENE in ${SCENES:-garden room}; do
    [ -n "$NCU" ] || break
    case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
    name=q_${BSCENE}_${BASE_ARM}_$NS
    echo "##### $name: method=$BASE seed=0 window=$NS-$(( NS + 2 )) $(date +%T)"
    env FGS_BLEND_BWD=macro FGS_PROF_RANGE=$NS,$(( NS + 2 )) FGS_PROF_EXIT=$(( NS + 2 )) SCENE=$BSCENE SCALE=$BSCALE SEED=0 METHOD=$BASE TAG=p2$name \
      OPT_FLAGS="SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true" \
      LAUNCHER="$NCU --profile-from-start off --kernel-name regex:preprocess --launch-count 4 --set full --import-source no -f -o $REP/$name" \
      bash "$REPO/cluster/train_fastergs.sbatch" > "$OUT/$name.log" 2>&1 || true
    if [ -f "$REP/$name.ncu-rep" ]; then
      "$NCU" --import "$REP/$name.ncu-rep" --page details --csv > "$OUT/$name.details.csv" 2>/dev/null || true
      "$NCU" --import "$REP/$name.ncu-rep" --page raw --csv > "$OUT/$name.raw.csv" 2>/dev/null || true
      "$NCU" --import "$REP/$name.ncu-rep" --page source --csv --print-source sass --kernel-name regex:preprocess_backward \
        --launch-count 1 > "$OUT/${BSCENE}_${BASE_ARM}.pre_sass.csv" 2>/dev/null || true
      echo "NCU $name kernels=$(python3 -c "import csv,sys; print(len({r['ID'] for r in csv.DictReader(open(sys.argv[1]))}))" "$OUT/$name.details.csv" 2>/dev/null || echo 0) sass_lines=$(wc -l < "$OUT/${BSCENE}_${BASE_ARM}.pre_sass.csv") rep $REP/$name.ncu-rep"
    else
      echo "FAILED $name: no report"; tr '\r' '\n' < "$OUT/$name.log" | grep -E "==ERROR==|==WARNING==|Traceback|Error" | head; FAILED+=("$name")
    fi
  done
fi

if has blendstats; then   # step 5 (blend kernels): how much blend work is wasted; counters of phase2/blend_stats.patch
  # one A+B+C run per scene (seed 0, 30k) with FasterGSPlusStats; the rasterizer prints BLENDSTATS_FWD / _BWD lines,
  # each summing 50 steps from every start in BS_STARTS. Timing of these runs is not a benchmark value.
  for BSCENE in ${SCENES:-garden room}; do
    case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
    echo "########## blendstats (scene $BSCENE scale $BSCALE, seed 0, A+B+C with blend counters)"
    TRAIN_ENV="FGS_BLENDSTATS=${BS_STARTS:-2000,5000,10000,15000,20000,29000}" train "p_${BSCENE}_abcstats_s0" FasterGSPlusStats 0 30000 "SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true" macro
    tr '\r' '\n' < "$OUT/p_${BSCENE}_abcstats_s0.log" | grep -oE 'BLENDSTATS_(FWD|BWD) .*' > "$OUT/p_${BSCENE}_abcstats_s0.stats" || true
    echo "STATS p_${BSCENE}_abcstats_s0 $(grep -c '^BLENDSTATS_FWD' "$OUT/p_${BSCENE}_abcstats_s0.stats" || true) fwd windows, $(grep -c '^BLENDSTATS_BWD' "$OUT/p_${BSCENE}_abcstats_s0.stats" || true) bwd windows"
    [ "$(grep -c '^BLENDSTATS_' "$OUT/p_${BSCENE}_abcstats_s0.stats" || true)" -gt 0 ] || FAILED+=("stats_${BSCENE}")
  done
fi

if has timerctl; then
  run_timerctl "${SCENE:?set SCENE for the timerctl stage}" "${SEEDS:-0 1 2}" "${TC_ITERS:-30000}"
fi

if has hostprof; then   # where the host-side gaps come from: PyTorch op ranges + Python backtraces on CUDA API calls
  BSCENE=${SCENE:?set SCENE for the hostprof stage}
  case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
  module load CUDA/12.8.0 >/dev/null 2>&1 || true
  echo "########## hostprof (scene $BSCENE, A+B+C, nsys with --pytorch=autograd-nvtx --python-backtrace=cuda at 20000)"
  NSYS_EXTRA="--pytorch=autograd-nvtx --python-backtrace=cuda --python-sampling=true "     nsys_window "h_${BSCENE}_${BASE_ARM}_20000" $BASE "SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true" macro 20000
fi

if has profile; then
  BSCENE=${SCENE:?set SCENE for the profile stage}
  case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
  module load CUDA/12.8.0 >/dev/null 2>&1 || true
  echo "########## profile (scene $BSCENE scale $BSCALE, seed 0; stock vs A+B+C)"
  STEPS=$(seq -s, 250 250 29750)
  for arm in stock $BASE_ARM; do
    if [ $arm = stock ]; then m=FasterGS; fl=""; bl=taming; else m=$BASE; fl="SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true"; bl=macro; fi
    TRAIN_ENV="FGS_PROF=1 FGS_STAGE_STEPS=$STEPS FGS_CURVE_EVERY=500" train "p_${BSCENE}_${arm}_s0" $m 0 30000 "$fl" $bl 1000
    tr '\r' '\n' < "$OUT/p_${BSCENE}_${arm}_s0.log" | grep -E '^(STAGE|CURVE) ' > "$OUT/p_${BSCENE}_${arm}_s0.prof" || true
    echo "PROF p_${BSCENE}_${arm}_s0 $(grep -c '^STAGE' "$OUT/p_${BSCENE}_${arm}_s0.prof") STAGE lines, $(grep -c '^CURVE' "$OUT/p_${BSCENE}_${arm}_s0.prof") CURVE lines"
    for w in 5000 20000; do nsys_window "n_${BSCENE}_${arm}_${w}" $m "$fl" $bl $w; done
  done
fi

# Upstream comparison (upstream/README.md): Faster-GS main (090e8ad) kernels with and without the coalesced SH-rest loads
# (upstream/sh_coalesced_loads.patch, the proposed upstream PR). No macro patch, dense Adam, fp32 moments in all arms.
#   FasterGSUp / FasterGSUpS: 1144604 + upstream c60ea26..090e8ad kernel diff (+ S loads), with the clock hook; timed
#   FasterGSUpPure / FasterGSUpPureS: c60ea26 + the same diff = upstream main's backend exactly (+ S loads); equivalence only
UP_PATCH=upstream/upstream_main_kernels.patch; S_PATCH=upstream/sh_coalesced_loads.patch
if has upbuild; then
  echo "########## upbuild"
  MACRO=0 VARIANT=FasterGSUp SRC_PATCHES="$UP_PATCH" TRAINER_PATCHES=cluster/patches/fastergs_sparse_from.py bash "$REPO/phase1/make_variant.sh" || FAILED+=(build_FasterGSUp)
  MACRO=0 VARIANT=FasterGSUpS SRC_PATCHES="$UP_PATCH $S_PATCH" TRAINER_PATCHES=cluster/patches/fastergs_sparse_from.py bash "$REPO/phase1/make_variant.sh" || FAILED+=(build_FasterGSUpS)
  MACRO=0 PROFILING=0 BASE=c60ea26 VARIANT=FasterGSUpPure SRC_PATCHES="$UP_PATCH" TRAINER_PATCHES= bash "$REPO/phase1/make_variant.sh" || FAILED+=(build_FasterGSUpPure)
  MACRO=0 PROFILING=0 BASE=c60ea26 VARIANT=FasterGSUpPureS SRC_PATCHES="$UP_PATCH $S_PATCH" TRAINER_PATCHES= bash "$REPO/phase1/make_variant.sh" || FAILED+=(build_FasterGSUpPureS)
  for v in FasterGSUp FasterGSUpS FasterGSUpPure FasterGSUpPureS; do
    so=$(ls "$DATA"/fastergs-venv/lib/python3.12/site-packages/${v}CudaBackend/_C*.so 2>/dev/null | head -1) || true
    [ -n "$so" ] && cuobjdump --dump-resource-usage "$so" 2>/dev/null | grep -A1 "preprocess_backward" | grep -v "^--" | paste - - \
      | sed -E "s/ +/ /g" | head -2 | cut -c1-220 | sed "s/^/RESOURCES $v /" || true
  done
fi

if has upequiv; then
  echo "########## upequiv"
  for pair in FasterGSUpPure:FasterGSUpPureS FasterGSUp:FasterGSUpS; do
    R=${pair%%:*}; N=${pair##*:}
    echo "equivalence: $N vs $R"
    ( cd /tmp && FGS_REF=${R}CudaBackend FGS_NEW=${N}CudaBackend python -m pytest -q -s -p no:cacheprovider "$REPO/upstream/tests/test_sh_load_equiv.py" ) \
      > "$OUT/upequiv_$N.log" 2>&1 && { grep "^EQUIV" "$OUT/upequiv_$N.log" | sort -u; tail -1 "$OUT/upequiv_$N.log"; } \
      || { echo "FAILED equivalence $N"; grep -E "^EQUIV|FAILED|Error" "$OUT/upequiv_$N.log" | head -40; FAILED+=("upequiv_$N"); }
  done
fi

if has upbench; then   # stock (c60ea26 kernels) / up (upstream main kernels) / ups (main + S loads), interleaved, rotated by seed
  BSCENE=${SCENE:?set SCENE for the upbench stage}
  case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
  echo "########## upbench (scene $BSCENE scale $BSCALE, seeds ${SEEDS:-0 1 2}, 30k; train time = SEGCLOCK cum_s)"
  UARMS=(${UP_ARMS:-stock up ups}); NA=${#UARMS[@]}
  echo "arms: ${UARMS[*]}"
  for seed in ${SEEDS:-0 1 2}; do
    for (( k = 0; k < NA; k++ )); do
      arm=${UARMS[$(( (k + seed) % NA ))]}
      case $arm in
        stock) m=FasterGS ;;
        up)    m=FasterGSUp ;;
        ups)   m=FasterGSUpS ;;
        *) echo "FAILED upbench: unknown arm $arm"; FAILED+=("upbench_$arm"); continue ;;
      esac
      TRAIN_ENV="FGS_WALL=0,15000,30000" train "b_${BSCENE}_${arm}_s$seed" $m "$seed" 30000 "" taming
    done
  done
fi

if has upncu; then   # Nsight Compute on preprocess_backward_cu at step NCU_STEP, upstream main vs main + S loads
  NCU=$(command -v ncu || true)
  [ -n "$NCU" ] || { echo "FAILED upncu: ncu not found"; FAILED+=(upncu); }
  NS=${NCU_STEP:-20000}; REP=$DATA/ncu/${SLURM_JOB_ID:-local}; mkdir -p "$REP"
  for BSCENE in ${SCENES:-garden room}; do
    [ -n "$NCU" ] || break
    case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
    for arm in up ups; do
      m=FasterGSUp; [ $arm = ups ] && m=FasterGSUpS
      name=q_${BSCENE}_${arm}_$NS
      echo "##### $name: method=$m seed=0 window=$NS-$(( NS + 2 )) $(date +%T)"
      env FGS_BLEND_BWD=taming FGS_PROF_RANGE=$NS,$(( NS + 2 )) FGS_PROF_EXIT=$(( NS + 2 )) SCENE=$BSCENE SCALE=$BSCALE SEED=0 METHOD=$m TAG=p2$name \
        OPT_FLAGS="" \
        LAUNCHER="$NCU --profile-from-start off --kernel-name regex:preprocess --launch-count 4 --set full --import-source no -f -o $REP/$name" \
        bash "$REPO/cluster/train_fastergs.sbatch" > "$OUT/$name.log" 2>&1 || true
      if [ -f "$REP/$name.ncu-rep" ]; then
        "$NCU" --import "$REP/$name.ncu-rep" --page details --csv > "$OUT/$name.details.csv" 2>/dev/null || true
        "$NCU" --import "$REP/$name.ncu-rep" --page raw --csv > "$OUT/$name.raw.csv" 2>/dev/null || true
        echo "NCU $name kernels=$(python3 -c "import csv,sys; print(len({r['ID'] for r in csv.DictReader(open(sys.argv[1]))}))" "$OUT/$name.details.csv" 2>/dev/null || echo 0) rep $REP/$name.ncu-rep"
      else
        echo "FAILED $name: no report"; tr '\r' '\n' < "$OUT/$name.log" | grep -E "==ERROR==|==WARNING==|Traceback|Error" | head; FAILED+=("$name")
      fi
    done
  done
fi

# Phase 3 Experiment 1 (phase3/PLAN.md): branch upstream-main training (FasterGSUp, dense Adam) at step EXP1_AT (5000)
# into arms ctl / stop / mcmc / oracle; test metrics every 1000 steps (cluster/patches/fastergs_resume.py).
#   exp1smoke  garden seed 0, short: prefix to 3000 saving the state at 2000, then ctl / stop / mcmc resumed 2000 -> 3000.
#              ctl's test_2000 must equal the prefix's test_2000 (same state, deterministic render).
#   exp1       SCENE, SEEDS: prefix 0 -> 5000 per seed, then the arms (rotated per seed), oracle after ctl.
x_clean() {  # drop the per-step test renders of a phase 3 run (metrics are archived by train()), keep checkpoints only for prefixes
  local RUN; RUN=$(cat "$OUT/runs/$1/rundir.txt" 2>/dev/null) || return 0
  find "$DATA/nerficg/$RUN" -path '*test_*' -name '*.png' -delete 2>/dev/null || true
  [ "${2:-}" = keepstate ] || rm -rf "$DATA/nerficg/$RUN/checkpoints" 2>/dev/null || true
}
x_arm_cfg() {  # arm N_at_branch N_oracle -> CFG_SET value
  case $1 in
    ctl)    echo "" ;;
    stop)   echo "DENSIFICATION_END_ITERATION=$EXP1_AT" ;;
    mcmc)   echo "USE_MCMC=true,MAX_PRIMITIVES=$2,LAMBDA_OPACITY_REGULARIZATION=0.01,LAMBDA_SCALE_REGULARIZATION=0.01,LEARNING_RATE_OPACITIES=0.05" ;;
    oracle) echo "USE_MCMC=true,MAX_PRIMITIVES=$3,LAMBDA_OPACITY_REGULARIZATION=0.01,LAMBDA_SCALE_REGULARIZATION=0.01,LEARNING_RATE_OPACITIES=0.05" ;;
  esac
}
if has exp1smoke || has exp1; then
  M=${EXP1_METHOD:-FasterGSUp}
  python "$REPO/cluster/patches/fastergs_resume.py" "$DATA/nerficg/src/Methods/$M/Trainer.py"
fi
if has exp1smoke; then
  BSCENE=garden; BSCALE=0.25; EXP1_AT=2000
  echo "########## exp1smoke (garden seed 0; prefix to 3000 with state at 2000, arms resumed 2000 -> 3000)"
  TRAIN_ENV="FGS_SAVE_STATE=2000 FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=2000" train xs_garden_pfx_s0 $M 0 3000 "" taming 1000
  ST=$DATA/nerficg/$(cat "$OUT/runs/xs_garden_pfx_s0/rundir.txt" 2>/dev/null)/checkpoints/state_0002000.pt
  N2=$(sed -n 's/^FGS_SAVE_STATE step=2000 N=\([0-9]*\).*/\1/p' "$OUT/xs_garden_pfx_s0.seg")
  if [ ! -f "$ST" ] || [ -z "$N2" ]; then echo "FAILED exp1smoke: no state file / N"; FAILED+=(exp1smoke_state); else
    for arm in ctl stop mcmc; do
      TRAIN_ENV="FGS_RESUME=$ST FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=2000 CFG_SET=$(x_arm_cfg $arm $N2 $N2)" train xs_garden_${arm}_s0 $M 0 3000 "" taming 1000
      x_clean xs_garden_${arm}_s0
    done
    python3 - "$OUT" <<'PY' || FAILED+=(exp1smoke_check)
import re, sys, os
out = sys.argv[1]
seg = lambda n: open(f'{out}/{n}.seg').read()
met = lambda n, it: open(f'{out}/runs/{n}/test_{it}/metrics_8bit.txt').read().split('Image', 1)[-1].strip()
ok = {}
p = seg('xs_garden_pfx_s0')
ok['prefix wrote the state'] = 'FGS_SAVE_STATE step=2000' in p
for a in ('ctl', 'stop', 'mcmc'):
    t = seg(f'xs_garden_{a}_s0')
    ok[f'{a}: resumed at 2000'] = bool(re.search(r'^FGS_RESUME step=2000 ', t, re.M))
    ok[f'{a}: clock continues (SEGCLOCK step=3000 cum_s > prefix cum at 2000)'] = (
        float(re.search(r'SEGCLOCK step=3000 \S+ cum_s=(\S+)', t).group(1)) > float(re.search(r'SEGCLOCK step=2000 \S+ cum_s=(\S+)', p).group(1)))
    ok[f'{a}: test_2000 identical to the prefix (same state)'] = met(f'xs_garden_{a}_s0', 2000) == met('xs_garden_pfx_s0', 2000)
    print(f'{a}: ' + ' '.join(l for l in t.splitlines() if l.startswith(('FGS_RESUME', 'SEGCLOCK step=3000', 'FGS_TRAIN_END'))))
n = lambda name: int(re.search(r'SEGCLOCK step=3000 .* N=(\d+)', seg(name)).group(1))
n2 = int(re.search(r'FGS_SAVE_STATE step=2000 N=(\d+)', p).group(1))
ok['ctl: N at 3000 within 1% of the uninterrupted prefix'] = abs(n('xs_garden_ctl_s0') / n('xs_garden_pfx_s0') - 1) < 0.01
ok['stop: N at 3000 <= N at 2000 (no densification after 2000)'] = n('xs_garden_stop_s0') <= n2
ok['mcmc: N at 3000 == N at 2000 (population held)'] = n('xs_garden_mcmc_s0') == n2
psnr = lambda name: float(open(f'{out}/runs/{name}/test_3000/metrics_8bit.txt').read().split('PSNR')[1].split()[0])
print('N at 2000', n2, ' N at 3000: ' + ' '.join(f'{a}={n(f"xs_garden_{a}_s0")}' for a in ('pfx', 'ctl', 'stop', 'mcmc')))
print('PSNR at 3000: ' + ' '.join(f'{a}={psnr(f"xs_garden_{a}_s0"):.2f}' for a in ('pfx', 'ctl', 'stop', 'mcmc')))
ok['ctl: PSNR at 3000 within 0.3 dB of the uninterrupted prefix'] = abs(psnr('xs_garden_ctl_s0') - psnr('xs_garden_pfx_s0')) < 0.3
for k, v in ok.items():
    print(('PASS ' if v else 'FAIL ') + k)
sys.exit(0 if all(ok.values()) else 1)
PY
  fi
  x_clean xs_garden_pfx_s0
fi
if has exp1; then
  BSCENE=${SCENE:?set SCENE for the exp1 stage}
  case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
  EXP1_AT=${EXP1_AT:-5000}
  XARMS=(${EXP1_ARMS:-ctl stop mcmc}); NA=${#XARMS[@]}
  echo "########## exp1 (scene $BSCENE scale $BSCALE, seeds ${SEEDS:-0 1 2}, branch at $EXP1_AT, arms ${XARMS[*]} + oracle after ctl)"
  for seed in ${SEEDS:-0 1 2}; do
    pfx=x_${BSCENE}_pfx_s$seed
    TRAIN_ENV="FGS_SAVE_STATE=$EXP1_AT" train $pfx $M "$seed" $EXP1_AT "" taming 1000
    ST=$DATA/nerficg/$(cat "$OUT/runs/$pfx/rundir.txt" 2>/dev/null)/checkpoints/state_$(printf %07d $EXP1_AT).pt
    NB=$(sed -n "s/^FGS_SAVE_STATE step=$EXP1_AT N=\([0-9]*\).*/\1/p" "$OUT/$pfx.seg")
    if [ ! -f "$ST" ] || [ -z "$NB" ]; then echo "FAILED exp1 $pfx: no state file / N"; FAILED+=("$pfx"); continue; fi
    echo "EXP1_BRANCH $BSCENE s$seed step=$EXP1_AT N=$NB state=$ST"
    for (( k = 0; k < NA; k++ )); do
      arm=${XARMS[$(( (k + seed) % NA ))]}
      TRAIN_ENV="FGS_RESUME=$ST FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=$EXP1_AT CFG_SET=$(x_arm_cfg $arm $NB 0)" train x_${BSCENE}_${arm}_s$seed $M "$seed" 30000 "" taming 1000
      x_clean x_${BSCENE}_${arm}_s$seed
    done
    if [ "${EXP1_ORACLE:-1}" = 1 ]; then   # cap = ctl's count at the last step before the end-of-training cleanup
      NC=$(sed -n 's/^FGS_TRAIN_END N=\([0-9]*\).*/\1/p' "$OUT/x_${BSCENE}_ctl_s$seed.seg" 2>/dev/null)
      if [ -z "$NC" ]; then echo "FAILED exp1 oracle s$seed: no ctl count"; FAILED+=("x_${BSCENE}_oracle_s$seed"); else
        echo "EXP1_ORACLE_CAP $BSCENE s$seed N=$NC"
        TRAIN_ENV="FGS_RESUME=$ST FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=$EXP1_AT CFG_SET=$(x_arm_cfg oracle $NB $NC)" train x_${BSCENE}_oracle_s$seed $M "$seed" 30000 "" taming 1000
        x_clean x_${BSCENE}_oracle_s$seed
      fi
    fi
    x_clean $pfx keepstate
  done
fi

if [ ${#FAILED[@]} -gt 0 ]; then echo "JOB FAILED: ${FAILED[*]}"; exit 1; fi
echo "JOB OK"
