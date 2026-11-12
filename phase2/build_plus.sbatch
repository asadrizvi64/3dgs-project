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
#   STAGES=x1adj ADJ_EVALTEST=1 SEEDS="3 4 5" sbatch --job-name=gs-x1adj-a --time=02:45:00 phase2/build_plus.sbatch   (Exp 1 adjudication)
#   STAGES="exp1smoke x1rsmoke" sbatch --job-name=gs-x1r-smoke --time=01:45:00 phase2/build_plus.sbatch   (Exp 1R smoke tests)
#   STAGES=noisepilot sbatch --job-name=gs-x1r-pilot --time=02:15:00 phase2/build_plus.sbatch   (Exp 1R noise pilot)
#   STAGES=x1ra sbatch --job-name=gs-x1ra --time=00:45:00 phase2/build_plus.sbatch   (Exp 1R smoke test A, repeated)
#   STAGES=exp1r SCENE=garden SEEDS="0 1" EXP1_AT=2000 X_ITERS=3000 sbatch --job-name=gs-x1r-stagesmoke --time=00:45:00 phase2/build_plus.sbatch
#   STAGES=exp1r SCENE=garden sbatch --job-name=gs-x1r-garden --time=04:00:00 phase2/build_plus.sbatch   (Exp 1R; room: one job per seed)
#   STAGES=x1asmoke sbatch --job-name=gs-x1a-smoke --time=01:00:00 phase2/build_plus.sbatch   (Exp 1A switch smoke test)
#   STAGES=x1atime sbatch --job-name=gs-x1a-time --time=00:45:00 phase2/build_plus.sbatch   (Exp 1A: counting vs training clock)
#   cd ~/3dgs-project && STAGES=x1rrender sbatch --job-name=gs-x1r-render --time=02:00:00 ~/render-job/build_plus.sbatch   (article renders; copy of this file, checkout untouched)
#   STAGES=exp1a SCENE=garden SEEDS="0 1" EXP1_AT=2000 X_ITERS=3000 sbatch --job-name=gs-x1a-stagesmoke --time=01:00:00 phase2/build_plus.sbatch
#   STAGES=exp1a SCENE=room SEEDS=30 sbatch --job-name=gs-x1a-room-s30 --time=04:30:00 phase2/build_plus.sbatch   (Exp 1A: one job per scene and seed)
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
        echo "FAILED $name: no report"; tr '
' '
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
if has exp1smoke || has exp1 || has x1adj || has x1rsmoke || has noisepilot || has x1ra || has exp1r || has x1asmoke || has exp1a || has x1atime || has x1rrender; then
  M=${EXP1_METHOD:-FasterGSUp}
  python "$REPO/cluster/patches/fastergs_resume.py" "$DATA/nerficg/src/Methods/$M/Trainer.py"
  if has x1asmoke || has exp1a || has x1atime; then python "$REPO/cluster/patches/fastergs_ablate.py" "$DATA/nerficg/src/Methods/$M/Trainer.py"; fi
fi
if has exp1smoke; then
  BSCENE=garden; BSCALE=0.25; EXP1_AT=2000
  echo "########## exp1smoke (garden seed 0; 4 uninterrupted runs to 3000, state saved at 2000; ctl x3 / stop / mcmc resumed 2000 -> 3000)"
  for r in pfx pfx2 pfx3 pfx4; do
    TRAIN_ENV="FGS_SAVE_STATE=2000 FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=2000 FGS_VIEW_TRACE=$OUT/xs_garden_${r}_s0.views" train xs_garden_${r}_s0 $M 0 3000 "" taming 1000
  done
  ST=$DATA/nerficg/$(cat "$OUT/runs/xs_garden_pfx_s0/rundir.txt" 2>/dev/null)/checkpoints/state_0002000.pt
  N2=$(sed -n 's/^FGS_SAVE_STATE step=2000 N=\([0-9]*\).*/\1/p' "$OUT/xs_garden_pfx_s0.seg")
  if [ ! -f "$ST" ] || [ -z "$N2" ]; then echo "FAILED exp1smoke: no state file / N"; FAILED+=(exp1smoke_state); else
    for run in ctl ctl2 ctl3 stop mcmc; do
      arm=${run%%[0-9]*}; segk=1000; [ $arm = ctl ] || segk=100   # stop / mcmc: clock (and N) every 100 steps
      TRAIN_ENV="FGS_RESUME=$ST FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=2000 FGS_VIEW_TRACE=$OUT/xs_garden_${run}_s0.views CFG_SET=$(x_arm_cfg $arm $N2 $N2)" train xs_garden_${run}_s0 $M 0 3000 "" taming $segk
      x_clean xs_garden_${run}_s0
    done
    python3 - "$OUT" <<'PY' || FAILED+=(exp1smoke_check)
import re, sys, statistics as st
out = sys.argv[1]
UN, CT = ('pfx', 'pfx2', 'pfx3', 'pfx4'), ('ctl', 'ctl2', 'ctl3')
seg = lambda n: open(f'{out}/xs_garden_{n}_s0.seg').read()
mfile = lambda n, it: open(f'{out}/runs/xs_garden_{n}_s0/test_{it}/metrics_8bit.txt').read()
rows = lambda n, it: re.findall(r'^\d+\t.*$', mfile(n, it), re.M)                       # per-image rows
psnr = lambda n: float(re.search(r'^PSNR:(\S+) SSIM', mfile(n, 3000), re.M).group(1))  # full-precision mean
nclk = lambda n: {int(a): int(b) for a, b in re.findall(r'SEGCLOCK step=(\d+) .* N=(\d+) peak', seg(n))}
views = lambda n, lo, hi: [l for l in open(f'{out}/xs_garden_{n}_s0.views').read().split('\n') if l and lo <= int(l.split()[0]) < hi]
ok = {}
sv = re.search(r'^FGS_SAVE_STATE step=2000 N=(\d+) \S+ param_sha=(\S+) optim_sha=(\S+) dinfo_sha=(\S+)', seg('pfx'), re.M)
ok['prefix wrote the state'] = bool(sv)
n2 = int(sv.group(1))
fr = [l.split()[2] for l in views('pfx', 0, 3000)]
print(f'view trace: {len(fr)} steps, {len(set(fr))} distinct frame_idx, first lines: {views("pfx", 0, 3)}')
ok['view trace holds real camera identities (frame_idx never None, > 50 distinct in 3000 steps)'] = 'None' not in fr and len(set(fr)) > 50
ok['the 4 uninterrupted runs saw the same views 0-2999'] = all(views(u, 0, 3000) == views('pfx', 0, 3000) for u in UN)
lr = {}
for run in CT + ('stop', 'mcmc'):
    t = seg(run)
    r = re.search(r'^FGS_RESUME step=2000 N=(\d+) \S+ \S+ param_sha=(\S+) optim_sha=(\S+) dinfo_sha=(\S+) .* lr=(\S+) lambda_opacity=(\S+) lambda_scale=(\S+)', t, re.M)
    ok[f'{run}: resumed at 2000'] = bool(r)
    if not r:
        continue
    ok[f'{run}: parameters and optimizer state after loading equal the saved ones (SHA-256 of the full bytes)'] = (r.group(1), r.group(2), r.group(3)) == (sv.group(1), sv.group(2), sv.group(3))
    ok[f'{run}: densification statistics restored (SHA-256), or dropped for MCMC'] = r.group(4) == ('none' if run == 'mcmc' else sv.group(4))
    ok[f'{run}: test_2000 per-image metrics identical to the uninterrupted run (same state)'] = rows(run, 2000) == rows('pfx', 2000)
    ok[f'{run}: clock continues (cum_s at 3000 > prefix cum_s at 2000)'] = (
        float(re.search(r'SEGCLOCK step=3000 \S+ cum_s=(\S+)', t).group(1)) > float(re.search(r'SEGCLOCK step=2000 \S+ cum_s=(\S+)', seg('pfx')).group(1)))
    lr[run] = dict(x.split(':') for x in r.group(5).split(','))
    lr[run]['lambda'] = (float(r.group(6)), float(r.group(7)))
for c in CT:   # only the control reproduces the uninterrupted view order: densification and MCMC draw from the CUDA
    ok[f'{c}: same views 2000-2999 as the uninterrupted run'] = views(c, 2000, 3000) == views('pfx', 2000, 3000)   # generator that reshuffles
print('learning rates / regularisation after resume:', {k: (v['opacities'], v['lambda']) for k, v in lr.items()})
ok['ctl: opacity lr 0.025, regularisation 0'] = all(float(lr[c]['opacities']) == 0.025 and lr[c]['lambda'] == (0.0, 0.0) for c in CT)
ok['mcmc: opacity lr 0.05, regularisation 0.01 / 0.01'] = float(lr['mcmc']['opacities']) == 0.05 and lr['mcmc']['lambda'] == (0.01, 0.01)
up, cp = [psnr(u) for u in UN], [psnr(c) for c in CT]
un, cn = [nclk(u)[3000] for u in UN], [nclk(c)[3000] for c in CT]
print('PSNR at 3000, uninterrupted:', ' '.join(f'{x:.4f}' for x in up), '| resumed ctl:', ' '.join(f'{x:.4f}' for x in cp))
print('N at 3000, uninterrupted:', un, '| resumed ctl:', cn)
ok['ctl: mean PSNR at 3000 within the uninterrupted runs\' range of their mean'] = abs(st.mean(cp) - st.mean(up)) <= max(up) - min(up)
ok['ctl: mean N at 3000 within the uninterrupted runs\' range of their mean'] = abs(st.mean(cn) - st.mean(un)) <= max(un) - min(un)
ns, nm = nclk('stop'), nclk('mcmc')
print('stop N:', {k: v for k, v in ns.items() if k >= 2000}, '\nmcmc N:', {k: v for k, v in nm.items() if k >= 2000})
ok['stop: N constant from 2100 to 3000 (only the branch step densifies)'] = len({v for k, v in ns.items() if 2100 <= k <= 3000}) == 1
ok['mcmc: N == N at 2000 at every 100-step point 2100-3000 (population held)'] = all(v == n2 for k, v in nm.items() if 2100 <= k <= 3000) and 3000 in nm
for k, v in ok.items():
    print(('PASS ' if v else 'FAIL ') + k)
sys.exit(0 if all(ok.values()) else 1)
PY
  fi
  for r in pfx pfx2 pfx3 pfx4; do x_clean xs_garden_${r}_s0; done
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
      TRAIN_ENV="FGS_RESUME=$ST FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=$EXP1_AT FGS_WALL=$EXP1_AT,30000 CFG_SET=$(x_arm_cfg $arm $NB 0)" train x_${BSCENE}_${arm}_s$seed $M "$seed" 30000 "" taming 1000
      x_clean x_${BSCENE}_${arm}_s$seed
    done
    if [ "${EXP1_ORACLE:-1}" = 1 ]; then   # cap = ctl's count at the last step before the end-of-training cleanup
      NC=$(sed -n 's/^FGS_TRAIN_END N=\([0-9]*\).*/\1/p' "$OUT/x_${BSCENE}_ctl_s$seed.seg" 2>/dev/null)
      if [ -z "$NC" ]; then echo "FAILED exp1 oracle s$seed: no ctl count"; FAILED+=("x_${BSCENE}_oracle_s$seed"); else
        echo "EXP1_ORACLE_CAP $BSCENE s$seed N=$NC"
        TRAIN_ENV="FGS_RESUME=$ST FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=$EXP1_AT FGS_WALL=$EXP1_AT,30000 CFG_SET=$(x_arm_cfg oracle $NB $NC)" train x_${BSCENE}_oracle_s$seed $M "$seed" 30000 "" taming 1000
        x_clean x_${BSCENE}_oracle_s$seed
      fi
    fi
    x_clean $pfx keepstate
  done
fi

# Experiment 1 adjudication (phase3/RESULTS.md, "Adjudication", pre-registered before any of it ran). Room only.
#   x1adj  ADJ_EVALTEST=1: part 1, room seed 0, two uninterrupted runs 0 -> 8000 with the view trace, one evaluating every
#          1000 steps from 5000 (as the branched runs do) and one not. Then, for each of SEEDS: the uninterrupted `up` run
#          exactly as in upbench, and prefix 0 -> 5000 plus branched ctl exactly as in exp1; order alternates by seed.
if has x1adj; then
  BSCENE=room; BSCALE=0.5; EXP1_AT=5000
  echo "########## x1adj (room, eval test ${ADJ_EVALTEST:-0}, seeds ${SEEDS:?set SEEDS for x1adj})"
  if [ "${ADJ_EVALTEST:-0}" = 1 ]; then
    TRAIN_ENV="FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=5000 FGS_VIEW_TRACE=$OUT/a_room_ev_s0.views" train a_room_ev_s0 $M 0 8000 "" taming 1000
    x_clean a_room_ev_s0
    TRAIN_ENV="FGS_VIEW_TRACE=$OUT/a_room_ne_s0.views" train a_room_ne_s0 $M 0 8000 "" taming 1000
    x_clean a_room_ne_s0
    python3 - "$OUT" <<'PY' || FAILED+=(x1adj_evaltest)
import re, sys
out = sys.argv[1]
views = lambda n: [l for l in open(f'{out}/a_room_{n}_s0.views').read().split('\n') if l]
ev, ne = views('ev'), views('ne')
nclk = lambda n: {int(a): int(b) for a, b in re.findall(r'SEGCLOCK step=(\d+) .* N=(\d+) peak', open(f'{out}/a_room_{n}_s0.seg').read())}
first = next((i for i, (a, b) in enumerate(zip(ev, ne)) if a != b), None)
print(f'EVALTEST view trace lines: with evaluation {len(ev)}, without {len(ne)}; first difference at line {first}')
print('EVALTEST N every 1000 steps (diagnostic), with evaluation:', nclk('ev'), '\nEVALTEST without:', nclk('ne'))
ok = len(ev) == len(ne) == 8000 and ev == ne
print(('PASS ' if ok else 'FAIL ') + 'part 1: identical training-view sequences 0-7999 with and without evaluation')
sys.exit(0 if ok else 1)
PY
  fi
  for seed in $SEEDS; do
    order="up ctl"; [ $(( seed % 2 )) = 1 ] && order="ctl up"
    for what in $order; do
      if [ $what = up ]; then
        TRAIN_ENV="FGS_WALL=0,15000,30000" train "b_room_up_s$seed" FasterGSUp "$seed" 30000 "" taming
        continue
      fi
      pfx=x_room_pfx_s$seed
      TRAIN_ENV="FGS_SAVE_STATE=$EXP1_AT" train $pfx $M "$seed" $EXP1_AT "" taming 1000
      ST=$DATA/nerficg/$(cat "$OUT/runs/$pfx/rundir.txt" 2>/dev/null)/checkpoints/state_$(printf %07d $EXP1_AT).pt
      NB=$(sed -n "s/^FGS_SAVE_STATE step=$EXP1_AT N=\([0-9]*\).*/\1/p" "$OUT/$pfx.seg")
      if [ ! -f "$ST" ] || [ -z "$NB" ]; then echo "FAILED x1adj $pfx: no state file / N"; FAILED+=("$pfx"); continue; fi
      echo "EXP1_BRANCH room s$seed step=$EXP1_AT N=$NB state=$ST"
      TRAIN_ENV="FGS_RESUME=$ST FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=$EXP1_AT FGS_WALL=$EXP1_AT,30000 CFG_SET=$(x_arm_cfg ctl $NB 0)" train x_room_ctl_s$seed $M "$seed" 30000 "" taming 1000
      x_clean x_room_ctl_s$seed
      x_clean $pfx
    done
  done
fi

# Experiment 1R smoke tests (phase3/EXP1R_PLAN.md; resume patch v7). Room seed 0.
#   x1rsmoke  A: the saved Experiment 1 room seed-0 state (job 4553943), resumed 5000 -> 7000 three times, evaluating at
#                5000 and 6000 with the per-stage profile (FGS_EVAL_PROF): guard on / guard off (v6 behaviour) /
#                guard on with cudnn.benchmark off. Same state at 5000, so the test_5000 metrics must be identical
#                between guard on and off; benchmark off is the memory comparison.
#             B: four uninterrupted runs 0 -> 8000, two evaluating every 1000 from 5000 with FGS_EVAL_CHECK, two not,
#                all with view traces. Every evaluation must leave the random states, parameters, optimizer state,
#                densification statistics, training flag and dataset mode as they were, and all four view traces must
#                be identical.
#   noisepilot  SCENES (default garden room), PILOT_SEEDS (default 20 21 22 23): two identical uninterrupted 30k
#                FasterGSUp runs per seed (b_<scene>_up / _up2), with view traces: the same-seed numerical noise that
#                sets Experiment 1R's check 1 sample size.
if has x1rsmoke; then
  BSCENE=room; BSCALE=0.5
  echo "########## x1rsmoke (room seed 0: A metric path + evaluation profile, B evaluation neutrality)"
  ST=${X1R_STATE:-$DATA/nerficg/$(cat "$REPO/phase2/logs/run_4553943/runs/x_room_pfx_s0/rundir.txt" 2>/dev/null)/checkpoints/state_0005000.pt}
  if [ ! -f "$ST" ]; then echo "FAILED x1rsmoke: no state $ST"; FAILED+=(x1rsmoke_state); else
    for r in guard noguard nobench; do
      e=""; [ $r = noguard ] && e="FGS_EVAL_RNG_GUARD=0"; [ $r = nobench ] && e="FGS_EVAL_CUDNN_BENCH=0"
      TRAIN_ENV="FGS_RESUME=$ST FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=5000 FGS_EVAL_PROF=1 $e" train rs_room_${r}_s0 $M 0 7000 "" taming 1000
      x_clean rs_room_${r}_s0
    done
  fi
  for r in ev ne ev2 ne2; do
    e=""; case $r in ev*) e="FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=5000 FGS_EVAL_CHECK=1" ;; esac
    TRAIN_ENV="$e FGS_VIEW_TRACE=$OUT/rn_room_${r}_s0.views" train rn_room_${r}_s0 $M 0 8000 "" taming 1000
    x_clean rn_room_${r}_s0
  done
  python3 - "$OUT" <<'PY' || FAILED+=(x1rsmoke_check)
import os, re, sys
out = sys.argv[1]
ok = {}
seg = lambda n: open(f'{out}/{n}.seg').read() if os.path.exists(f'{out}/{n}.seg') else ''
mf = lambda n, it: open(f'{out}/runs/{n}/test_{it}/metrics_8bit.txt').read()
rows = lambda n, it: re.findall(r'^\d+\t.*$', mf(n, it), re.M)
full = lambda n, it: mf(n, it).strip().splitlines()[-1]
# ---- A: metric path and profile
G, NG, NB = 'rs_room_guard_s0', 'rs_room_noguard_s0', 'rs_room_nobench_s0'
try:
    ok['A: guard on / off give identical per-image metrics at 5000 (same state; the guard does not touch the metric path)'] = rows(G, 5000) == rows(NG, 5000)
    ok['A: guard on / off give identical full-precision means at 5000'] = full(G, 5000) == full(NG, 5000)
    print('A full-precision means at 5000: guard', full(G, 5000), '| no guard', full(NG, 5000), '| benchmark off', full(NB, 5000))
    rg, rb = rows(G, 5000), rows(NB, 5000)
    d = [[abs(float(a) - float(b)) for a, b in zip(x.split('\t')[1:], y.split('\t')[1:])] for x, y in zip(rg, rb)]
    print('A benchmark off vs on at 5000 (diagnostic): per-image rows identical:', rg == rb,
          '| max abs difference PSNR / SSIM / LPIPS:', [max(c) for c in zip(*d)] if d else None)
except FileNotFoundError as e:
    ok[f'A: metrics files present ({e.filename})'] = False
for n in (G, NG, NB):
    for l in re.findall(r'^FGS_EVALPROF .*$', seg(n), re.M):
        print(n, l)
# ---- B: neutrality
EV, NE = ('rn_room_ev_s0', 'rn_room_ev2_s0'), ('rn_room_ne_s0', 'rn_room_ne2_s0')
views = {n: [l for l in open(f'{out}/{n}.views').read().split('\n') if l] for n in EV + NE if os.path.exists(f'{out}/{n}.views')}
ok['B: four view traces of 8000 steps'] = len(views) == 4 and all(len(v) == 8000 for v in views.values())
ref = views.get(NE[0])
for n in EV + NE[1:]:
    v = views.get(n, [])
    first = next((i for i, (a, b) in enumerate(zip(v, ref or [])) if a != b), None)
    ok[f'B: {n} view trace identical to {NE[0]} (first difference: {first})'] = v == ref
for n in EV:
    chk = re.findall(r'^FGS_EVALCHECK step=(\d+) guard=1 rng_before=(\S+) rng_after_render=(\S+) rng_after_restore=(\S+) '
                     r'state_before=(\S+) state_after=(\S+) training_before=(\S+) training_after_render=(\S+) '
                     r'training_after_restore=(\S+) mode_before=(\S+) mode_after_render=(\S+) mode_after_restore=(\S+)', seg(n), re.M)
    ok[f'B: {n} has evaluation checks at 5000, 6000, 7000'] = [int(c[0]) for c in chk] == [5000, 6000, 7000]
    ok[f'B: {n} random states restored after every evaluation'] = bool(chk) and all(c[1] == c[3] for c in chk)
    ok[f'B: {n} parameters, optimizer state and densification statistics unchanged by every evaluation (SHA-256)'] = bool(chk) and all(c[4] == c[5] for c in chk)
    ok[f'B: {n} training flag and dataset mode restored after every evaluation'] = bool(chk) and all(c[6] == c[8] and c[9] == c[11] for c in chk)
    for c in chk:
        print(f'{n} step {c[0]}: rendering changed the random states: {c[1] != c[2]} '
              f'({",".join(k for k, (a, b) in zip(("cpu", "cuda", "py", "np"), zip(c[1].split(","), c[2].split(","))) if a != b) or "none"}); '
              f'training flag {c[6]} -> {c[7]} -> {c[8]}; mode {c[9]} -> {c[10]} -> {c[11]}')
nclk = lambda n: {int(a): int(b) for a, b in re.findall(r'SEGCLOCK step=(\d+) .* N=(\d+) peak', seg(n))}
for n in EV + NE:
    r = re.search(r'PSNR:(\S+) SSIM:(\S+) LPIPS:(\S+)', full(n, 8000)) if os.path.exists(f'{out}/runs/{n}/test_8000/metrics_8bit.txt') else None
    print(f'B diagnostic {n}: N {nclk(n)} | test at 8000 {r.groups() if r else None}')
for k, v in ok.items():
    print(('PASS ' if v else 'FAIL ') + k)
sys.exit(0 if all(ok.values()) else 1)
PY
fi
# Experiment 1R, revised after review (phase3/EXP1R_PLAN.md, "Revisions after the smoke test and pilot").
#   x1ra   smoke test A, repeated: the same saved room seed-0 state (job 4553943) resumed 5000 -> 5100, evaluated once at
#          5000, without profiling: guard on x2, guard off x2 (gate: tolerances on the full-precision means for every
#          guard-on / guard-off pair, identical per-image rows at output precision), plus cudnn.benchmark off x2. Memory
#          diagnostic: FGS_TRAIN_END peak (never reset, so it includes the evaluation) with benchmark on vs off.
#   exp1r  SCENE, SEEDS (default 10 11 12): per seed the uninterrupted reference b_<scene>_up_s<seed> (no evaluation) and
#          the prefix 0 -> EXP1_AT, then ctl / stop / mcmc (rotated by seed) and oracle, as exp1, with test metrics every
#          1000 steps and FGS_EVAL_CHECK in every branched run; view traces for every run. The reference runs first for
#          even seeds and last for odd seeds (EXP1R_ORDER line). X_ITERS (default 30000) and EXP1_AT (default 5000) are
#          only changed for the stage's own smoke test.
if has x1ra; then
  BSCENE=room; BSCALE=0.5
  echo "########## x1ra (room seed 0, same saved state at 5000: guard on x2 / off x2 / benchmark off x2, unprofiled)"
  ST=${X1R_STATE:-$DATA/nerficg/$(cat "$REPO/phase2/logs/run_4553943/runs/x_room_pfx_s0/rundir.txt" 2>/dev/null)/checkpoints/state_0005000.pt}
  if [ ! -f "$ST" ]; then echo "FAILED x1ra: no state $ST"; FAILED+=(x1ra_state); else
    for r in guard noguard nobench guard2 noguard2 nobench2; do
      e=""; case $r in noguard*) e="FGS_EVAL_RNG_GUARD=0" ;; nobench*) e="FGS_EVAL_CUDNN_BENCH=0" ;; esac
      TRAIN_ENV="FGS_RESUME=$ST FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=5000 $e" train ra_room_${r}_s0 $M 0 5100 "" taming 100
      x_clean ra_room_${r}_s0
    done
    python3 "$REPO/phase3/check_x1ra.py" "$OUT" || FAILED+=(x1ra_check)
  fi
fi
if has exp1r; then
  BSCENE=${SCENE:?set SCENE for the exp1r stage}
  case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
  EXP1_AT=${EXP1_AT:-5000}; XI=${X_ITERS:-30000}
  XARMS=(ctl stop mcmc); NA=3
  echo "########## exp1r (scene $BSCENE scale $BSCALE, seeds ${SEEDS:-10 11 12}, branch at $EXP1_AT, end $XI, arms ${XARMS[*]} + oracle after ctl)"
  x1r_up() {
    TRAIN_ENV="FGS_WALL=0,$XI FGS_VIEW_TRACE=$OUT/b_${BSCENE}_up_s$1.views" train b_${BSCENE}_up_s$1 $M "$1" $XI "" taming 1000
  }
  for seed in ${SEEDS:-10 11 12}; do
    order=(); for (( k = 0; k < NA; k++ )); do order+=(${XARMS[$(( (k + seed) % NA ))]}); done
    if [ $(( seed % 2 )) = 0 ]; then echo "EXP1R_ORDER $BSCENE s$seed up pfx ${order[*]} oracle"; else echo "EXP1R_ORDER $BSCENE s$seed pfx ${order[*]} oracle up"; fi
    [ $(( seed % 2 )) = 0 ] && x1r_up $seed
    pfx=x_${BSCENE}_pfx_s$seed
    TRAIN_ENV="FGS_SAVE_STATE=$EXP1_AT FGS_VIEW_TRACE=$OUT/$pfx.views" train $pfx $M "$seed" $EXP1_AT "" taming 1000
    ST=$DATA/nerficg/$(cat "$OUT/runs/$pfx/rundir.txt" 2>/dev/null)/checkpoints/state_$(printf %07d $EXP1_AT).pt
    NB=$(sed -n "s/^FGS_SAVE_STATE step=$EXP1_AT N=\([0-9]*\).*/\1/p" "$OUT/$pfx.seg")
    if [ ! -f "$ST" ] || [ -z "$NB" ]; then echo "FAILED exp1r $pfx: no state file / N"; FAILED+=("$pfx")
    else
      echo "EXP1_BRANCH $BSCENE s$seed step=$EXP1_AT N=$NB state=$ST"
      XE="FGS_RESUME=$ST FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=$EXP1_AT FGS_EVAL_CHECK=1 FGS_WALL=$EXP1_AT,$XI"
      for arm in "${order[@]}"; do
        TRAIN_ENV="$XE FGS_VIEW_TRACE=$OUT/x_${BSCENE}_${arm}_s$seed.views CFG_SET=$(x_arm_cfg $arm $NB 0)" train x_${BSCENE}_${arm}_s$seed $M "$seed" $XI "" taming 1000
        x_clean x_${BSCENE}_${arm}_s$seed
      done
      NC=$(sed -n 's/^FGS_TRAIN_END N=\([0-9]*\).*/\1/p' "$OUT/x_${BSCENE}_ctl_s$seed.seg" 2>/dev/null)
      if [ -z "$NC" ]; then echo "FAILED exp1r oracle s$seed: no ctl count"; FAILED+=("x_${BSCENE}_oracle_s$seed"); else
        echo "EXP1_ORACLE_CAP $BSCENE s$seed N=$NC"
        TRAIN_ENV="$XE FGS_VIEW_TRACE=$OUT/x_${BSCENE}_oracle_s$seed.views CFG_SET=$(x_arm_cfg oracle $NB $NC)" train x_${BSCENE}_oracle_s$seed $M "$seed" $XI "" taming 1000
        x_clean x_${BSCENE}_oracle_s$seed
      fi
      x_clean $pfx keepstate
    fi
    [ $(( seed % 2 )) = 1 ] && x1r_up $seed
    x_clean b_${BSCENE}_up_s$seed
  done
fi
# x1rrender (article figures, not an experiment): ctl and mcmc resumed again from Experiment 1R's saved seed-11 states
#   at 5000 (garden job 4559237, room job 4559239), exactly as exp1r ran them, but keeping the per-step test renders
#   (no x_clean). Renders stay in the run directories on the workspace; RUNDIR lines name them. Seed via X1R_SEED.
if has x1rrender; then
  RS=${X1R_SEED:-11}
  for BSCENE in garden room; do
    case $BSCENE in garden) BSCALE=0.25; R1=$REPO/phase2/logs/run_4559237 ;; room) BSCALE=0.5; R1=$REPO/phase2/logs/run_4559239 ;; esac
    echo "########## x1rrender ($BSCENE seed $RS, Experiment 1R state at 5000: ctl / mcmc with renders kept)"
    ST=$DATA/nerficg/$(cat "$R1/runs/x_${BSCENE}_pfx_s$RS/rundir.txt" 2>/dev/null)/checkpoints/state_0005000.pt
    NB=$(sed -n 's/^FGS_SAVE_STATE step=5000 N=\([0-9]*\).*/\1/p' "$R1/x_${BSCENE}_pfx_s$RS.seg" 2>/dev/null)
    if [ ! -f "$ST" ] || [ -z "$NB" ]; then echo "FAILED x1rrender $BSCENE: no state $ST / N"; FAILED+=("x1rrender_$BSCENE"); continue; fi
    echo "EXP1_BRANCH $BSCENE s$RS step=5000 N=$NB state=$ST"
    for arm in ctl mcmc; do
      TRAIN_ENV="FGS_RESUME=$ST FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=5000 FGS_EVAL_CHECK=1 FGS_WALL=5000,30000 CFG_SET=$(x_arm_cfg $arm $NB 0)" \
        train xr_${BSCENE}_${arm}_s$RS $M "$RS" 30000 "" taming 1000
    done
  done
fi
# Experiment 1A (phase3/EXP1A_PLAN.md): which parts of the MCMC policy make it faster. Needs the ablation switches
# (cluster/patches/fastergs_ablate.py), applied with the resume patch above.
#   x1asmoke  every switch on Experiment 1R's saved garden seed-10 state at 5000 (job 4559237), with FGS_ABL counters,
#             evaluation checks and view traces: ctl and creset 5000 -> 13000 (the three resets), the six MCMC arms
#             5000 -> 7000; phase3/check_x1a.py.
#   exp1a     SCENE, SEEDS (default 30 31 32): as exp1r (reference b_<scene>_up first for even seeds, last for odd; prefix;
#             FGS_EVAL_CHECK and view traces in every branched run) with the arms ctl mcmc mrep mreloc mnoise mreg mlr
#             creset, rotated by seed, all with FGS_ABL=1. X_ITERS / EXP1_AT only change for the stage's smoke test.
x_abl_env() {  # arm N_at_branch -> extra env and CFG_SET for an Experiment 1A arm
  local m="USE_MCMC=true,MAX_PRIMITIVES=$2"
  case $1 in
    ctl)    echo "CFG_SET=" ;;
    creset) echo "FGS_OPACITY_RESET=0 CFG_SET=" ;;
    mcmc|mrep) echo "CFG_SET=$m,LAMBDA_OPACITY_REGULARIZATION=0.01,LAMBDA_SCALE_REGULARIZATION=0.01,LEARNING_RATE_OPACITIES=0.05" ;;
    mreloc) echo "FGS_MCMC_RELOCATE=0 CFG_SET=$m,LAMBDA_OPACITY_REGULARIZATION=0.01,LAMBDA_SCALE_REGULARIZATION=0.01,LEARNING_RATE_OPACITIES=0.05" ;;
    mnoise) echo "FGS_MCMC_NOISE=0 CFG_SET=$m,LAMBDA_OPACITY_REGULARIZATION=0.01,LAMBDA_SCALE_REGULARIZATION=0.01,LEARNING_RATE_OPACITIES=0.05" ;;
    mreg)   echo "CFG_SET=$m,LAMBDA_OPACITY_REGULARIZATION=0.0,LAMBDA_SCALE_REGULARIZATION=0.0,LEARNING_RATE_OPACITIES=0.05" ;;
    mlr)    echo "CFG_SET=$m,LAMBDA_OPACITY_REGULARIZATION=0.01,LAMBDA_SCALE_REGULARIZATION=0.01,LEARNING_RATE_OPACITIES=0.025" ;;
  esac
}
if has x1asmoke; then
  BSCENE=garden; BSCALE=0.25
  echo "########## x1asmoke (garden seed 10, Experiment 1R state at 5000: ctl / creset to 13000, MCMC arms to 7000)"
  R1=$REPO/phase2/logs/run_4559237
  ST=$DATA/nerficg/$(cat "$R1/runs/x_garden_pfx_s10/rundir.txt" 2>/dev/null)/checkpoints/state_0005000.pt
  NB=$(sed -n 's/^FGS_SAVE_STATE step=5000 N=\([0-9]*\).*/\1/p' "$R1/x_garden_pfx_s10.seg" 2>/dev/null)
  if [ ! -f "$ST" ] || [ -z "$NB" ]; then echo "FAILED x1asmoke: no state $ST / N"; FAILED+=(x1asmoke_state); else
    for arm in ctl creset mcmc mrep mreloc mnoise mreg mlr; do
      it=7000; case $arm in ctl|creset) it=13000 ;; esac
      TRAIN_ENV="FGS_RESUME=$ST FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=5000 FGS_EVAL_CHECK=1 FGS_ABL=1 FGS_VIEW_TRACE=$OUT/sa_garden_${arm}_s10.views $(x_abl_env $arm $NB)" \
        train sa_garden_${arm}_s10 $M 10 $it "" taming 1000
      x_clean sa_garden_${arm}_s10
    done
    python3 "$REPO/phase3/check_x1a.py" "$OUT" "$R1" || FAILED+=(x1asmoke_check)
  fi
fi
#   x1atime   does counting change the training clock? ctl and mcmc from the same Experiment 1R state, 5000 -> 8000, no
#             evaluation, FGS_ABL=1 and FGS_ABL=0, two runs each, alternating; phase3/check_x1atime.py: training time
#             5000 -> 8000 with counting within 2% of without, per arm; elapsed wall time and callback time reported.
if has x1atime; then
  BSCENE=garden; BSCALE=0.25
  echo "########## x1atime (garden seed 10, Experiment 1R state at 5000: ctl / mcmc, FGS_ABL 1 vs 0, two each)"
  R1=$REPO/phase2/logs/run_4559237
  ST=$DATA/nerficg/$(cat "$R1/runs/x_garden_pfx_s10/rundir.txt" 2>/dev/null)/checkpoints/state_0005000.pt
  NB=$(sed -n 's/^FGS_SAVE_STATE step=5000 N=\([0-9]*\).*/\1/p' "$R1/x_garden_pfx_s10.seg" 2>/dev/null)
  if [ ! -f "$ST" ] || [ -z "$NB" ]; then echo "FAILED x1atime: no state $ST / N"; FAILED+=(x1atime_state); else
    for rep in 1 2; do
      for arm in ctl mcmc; do
        for c in 1 0; do
          TRAIN_ENV="FGS_RESUME=$ST FGS_ABL=$c FGS_WALL=5000,8000 $(x_abl_env $arm $NB)" train ta_garden_${arm}_c${c}r${rep}_s10 $M 10 8000 "" taming 1000
          x_clean ta_garden_${arm}_c${c}r${rep}_s10
        done
      done
    done
    python3 "$REPO/phase3/check_x1atime.py" "$OUT" || FAILED+=(x1atime_check)
  fi
fi
if has exp1a; then
  BSCENE=${SCENE:?set SCENE for the exp1a stage}
  case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
  EXP1_AT=${EXP1_AT:-5000}; XI=${X_ITERS:-30000}
  AARMS=(ctl mcmc mrep mreloc mnoise mreg mlr creset); NA=${#AARMS[@]}
  echo "########## exp1a (scene $BSCENE scale $BSCALE, seeds ${SEEDS:-30 31 32}, branch at $EXP1_AT, end $XI, arms ${AARMS[*]})"
  x1a_up() {
    TRAIN_ENV="FGS_WALL=0,$XI FGS_VIEW_TRACE=$OUT/b_${BSCENE}_up_s$1.views" train b_${BSCENE}_up_s$1 $M "$1" $XI "" taming 1000
  }
  for seed in ${SEEDS:-30 31 32}; do
    order=(); for (( k = 0; k < NA; k++ )); do order+=(${AARMS[$(( (k + seed) % NA ))]}); done
    if [ $(( seed % 2 )) = 0 ]; then echo "EXP1R_ORDER $BSCENE s$seed up pfx ${order[*]}"; else echo "EXP1R_ORDER $BSCENE s$seed pfx ${order[*]} up"; fi
    [ $(( seed % 2 )) = 0 ] && x1a_up $seed
    pfx=x_${BSCENE}_pfx_s$seed
    TRAIN_ENV="FGS_SAVE_STATE=$EXP1_AT FGS_ABL=1 FGS_VIEW_TRACE=$OUT/$pfx.views" train $pfx $M "$seed" $EXP1_AT "" taming 1000
    ST=$DATA/nerficg/$(cat "$OUT/runs/$pfx/rundir.txt" 2>/dev/null)/checkpoints/state_$(printf %07d $EXP1_AT).pt
    NB=$(sed -n "s/^FGS_SAVE_STATE step=$EXP1_AT N=\([0-9]*\).*/\1/p" "$OUT/$pfx.seg")
    if [ ! -f "$ST" ] || [ -z "$NB" ]; then echo "FAILED exp1a $pfx: no state file / N"; FAILED+=("$pfx")
    else
      echo "EXP1_BRANCH $BSCENE s$seed step=$EXP1_AT N=$NB state=$ST"
      for arm in "${order[@]}"; do
        TRAIN_ENV="FGS_RESUME=$ST FGS_EVAL_EVERY=1000 FGS_EVAL_FROM=$EXP1_AT FGS_EVAL_CHECK=1 FGS_ABL=1 FGS_WALL=$EXP1_AT,$XI FGS_VIEW_TRACE=$OUT/x_${BSCENE}_${arm}_s$seed.views $(x_abl_env $arm $NB)" \
          train x_${BSCENE}_${arm}_s$seed $M "$seed" $XI "" taming 1000
        x_clean x_${BSCENE}_${arm}_s$seed
      done
      x_clean $pfx keepstate
    fi
    [ $(( seed % 2 )) = 1 ] && x1a_up $seed
    x_clean b_${BSCENE}_up_s$seed
  done
fi
if has noisepilot; then
  for BSCENE in ${SCENES:-garden room}; do
    case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
    echo "########## noisepilot (scene $BSCENE scale $BSCALE, seeds ${PILOT_SEEDS:-20 21 22 23}, two identical uninterrupted runs each)"
    for seed in ${PILOT_SEEDS:-20 21 22 23}; do
      for r in up up2; do
        TRAIN_ENV="FGS_WALL=0,15000,30000 FGS_VIEW_TRACE=$OUT/b_${BSCENE}_${r}_s$seed.views" train "b_${BSCENE}_${r}_s$seed" FasterGSUp "$seed" 30000 "" taming
      done
    done
  done
fi

if [ ${#FAILED[@]} -gt 0 ]; then echo "JOB FAILED: ${FAILED[*]}"; exit 1; fi
echo "JOB OK"
