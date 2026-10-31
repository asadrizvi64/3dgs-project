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
#   bench    (step 3, not in the default STAGES) one Mip-NeRF360 scene (SCENE, scale 0.25 outdoor / 0.5 indoor), SEEDS,
#            30k: stock FasterGS, FasterGSPlus A+B, FasterGSPlus A+B+C, interleaved per seed, arm order rotated by seed
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
#   blendstats (step 5) SCENES (default garden room), seed 0, A+B+C built as FasterGSPlusStats (BUILD_VARIANT=FasterGSPlusStats
#            BUILD_SRC_PATCHES=phase2/blend_stats.patch): blend-kernel work counters over 50-step windows (BS_STARTS)
#   sbatch phase2/build_plus.sbatch      (from the repo root)
#   STAGES=factorial sbatch --job-name=gs-phase2-factorial --time=03:30:00 phase2/build_plus.sbatch
#   STAGES=bench SCENE=room sbatch --job-name=gs-bench-room --time=02:30:00 phase2/build_plus.sbatch
#   STAGES=stages SCENE=room sbatch --job-name=gs-stages-room --time=02:30:00 phase2/build_plus.sbatch
#   STAGES=profile SCENE=room sbatch --job-name=gs-prof-room --time=02:00:00 phase2/build_plus.sbatch
#   STAGES=timerctl SCENE=room sbatch --job-name=gs-timer-room --time=01:30:00 phase2/build_plus.sbatch
#   STAGES="build unit" BUILD_VARIANT=FasterGSPlusCull BUILD_SRC_PATCHES=phase2/ellipse_cull.patch sbatch --job-name=gs-cull-build phase2/build_plus.sbatch
#   STAGES=rbbench SCENE=garden CMP_VARIANT=FasterGSPlusCull CMP_ARM=abce sbatch --job-name=gs-cull-garden --time=00:40:00 phase2/build_plus.sbatch
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

# the stock checkout gets the same clock hook (and the B switch, off by default) so all arms are timed alike
python "$REPO/cluster/patches/fastergs_sparse_from.py" "$DATA/nerficg/src/Methods/FasterGS/Trainer.py"

if has build; then
  echo "########## build"
  VARIANT=${BUILD_VARIANT:-FasterGSPlus} SRC_PATCHES="${BUILD_SRC_PATCHES:-}" TRAINER_PATCHES=cluster/patches/fastergs_sparse_from.py bash "$REPO/phase1/make_variant.sh"
fi

if has unit; then
  echo "########## unit"
  ( cd /tmp && FGS_BACKEND=${BUILD_VARIANT:-FasterGSPlus}CudaBackend python -m pytest -q -s -p no:cacheprovider "$REPO/phase1/tests/test_parity.py" ) \
    > "$OUT/parity.log" 2>&1 && tail -3 "$OUT/parity.log" || { echo "FAILED parity"; tail -30 "$OUT/parity.log"; FAILED+=(parity); }
  ( cd /tmp && FGS_BACKEND=${BUILD_VARIANT:-FasterGSPlus}CudaBackend python -I "$REPO/cluster/tests/test_lean_adam.py" ) \
    > "$OUT/lean_adam.log" 2>&1 && grep -c "^PASS" "$OUT/lean_adam.log" | sed 's/^/lean-adam PASS count: /' || { echo "FAILED lean adam"; grep -E "FAIL|Error" "$OUT/lean_adam.log" | head; FAILED+=(lean_adam); }
  grep "^FAIL" "$OUT/lean_adam.log" && FAILED+=(lean_adam_fail_lines) || true
  if [ "${BUILD_VARIANT:-FasterGSPlus}" != FasterGSPlus ]; then   # the variant must reproduce FasterGSPlus on identical inputs
    ( cd /tmp && FGS_REF=FasterGSPlusCudaBackend FGS_NEW=${BUILD_VARIANT}CudaBackend python -m pytest -q -s -p no:cacheprovider "$REPO/phase2/tests/test_readback_equiv.py" )       > "$OUT/equiv.log" 2>&1 && { grep "^EQUIV" "$OUT/equiv.log" | sort -u | head -30; tail -2 "$OUT/equiv.log"; } || { echo "FAILED equivalence"; tail -40 "$OUT/equiv.log"; FAILED+=(equivalence); }
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
  BARMS=(stock ab abc)
  for seed in ${SEEDS:-0 1 2}; do
    for k in 0 1 2; do
      arm=${BARMS[$(( (k + seed) % 3 ))]}
      case $arm in
        stock) train "b_${BSCENE}_stock_s$seed" FasterGS "$seed" 30000 "" taming ;;
        ab)    train "b_${BSCENE}_ab_s$seed" FasterGSPlus "$seed" 30000 "SELECTIVE_FROM=15000" macro ;;
        abc)   train "b_${BSCENE}_abc_s$seed" FasterGSPlus "$seed" 30000 "SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true" macro ;;
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
  local CONFIGS=(stock_on abc_on stock_off abc_off) seed k cfg arm tm off m fl bl
  for seed in $seeds; do
    for k in 0 1 2 3; do
      cfg=${CONFIGS[$(( (k + seed) % 4 ))]}
      arm=${cfg%_*}; tm=${cfg#*_}
      off=0; [ "$tm" = off ] && off=1
      if [ $arm = stock ]; then m=FasterGS; fl=""; bl=taming; else m=FasterGSPlus; fl="SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true"; bl=macro; fi
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
      if [ $(( (k + seed) % 2 )) = 0 ]; then arm=abc; m=FasterGSPlus; else arm=$CMP_ARM; m=$CMP_VARIANT; fi
      TRAIN_ENV="FGS_WALL=0,15000,30000" train "x_${BSCENE}_${arm}_s$seed" $m "$seed" 30000 "SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true" macro
    done
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
  NSYS_EXTRA="--pytorch=autograd-nvtx --python-backtrace=cuda --python-sampling=true "     nsys_window "h_${BSCENE}_abc_20000" FasterGSPlus "SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true" macro 20000
fi

if has profile; then
  BSCENE=${SCENE:?set SCENE for the profile stage}
  case $BSCENE in room|counter|kitchen|bonsai) BSCALE=0.5 ;; *) BSCALE=0.25 ;; esac
  module load CUDA/12.8.0 >/dev/null 2>&1 || true
  echo "########## profile (scene $BSCENE scale $BSCALE, seed 0; stock vs A+B+C)"
  STEPS=$(seq -s, 250 250 29750)
  for arm in stock abc; do
    if [ $arm = stock ]; then m=FasterGS; fl=""; bl=taming; else m=FasterGSPlus; fl="SELECTIVE_FROM=15000,SH_REST_BF16_MOMENTS=true"; bl=macro; fi
    TRAIN_ENV="FGS_PROF=1 FGS_STAGE_STEPS=$STEPS FGS_CURVE_EVERY=500" train "p_${BSCENE}_${arm}_s0" $m 0 30000 "$fl" $bl 1000
    tr '\r' '\n' < "$OUT/p_${BSCENE}_${arm}_s0.log" | grep -E '^(STAGE|CURVE) ' > "$OUT/p_${BSCENE}_${arm}_s0.prof" || true
    echo "PROF p_${BSCENE}_${arm}_s0 $(grep -c '^STAGE' "$OUT/p_${BSCENE}_${arm}_s0.prof") STAGE lines, $(grep -c '^CURVE' "$OUT/p_${BSCENE}_${arm}_s0.prof") CURVE lines"
    for w in 5000 20000; do nsys_window "n_${BSCENE}_${arm}_${w}" $m "$fl" $bl $w; done
  done
fi

if [ ${#FAILED[@]} -gt 0 ]; then echo "JOB FAILED: ${FAILED[*]}"; exit 1; fi
echo "JOB OK"
