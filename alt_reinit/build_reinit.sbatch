#!/usr/bin/env bash
# Alternative track R (alt_reinit/PLAN.md): R0 build and checks, R0 smoke runs, R1 audit. Separate from phase2/:
# shared files (cluster/train_fastergs.sbatch, cluster/patches/*, the stock FasterGS checkout) are only read.
# Stages (STAGES="..."):
#   build   alt_reinit/make_reinit_variant.sh: FasterGSReinit = FasterGS @1144604 + shared clock / save-state hooks +
#           the track's hooks (no macro backward, nothing from FasterGSPlus)
#   unit    CPU tests (core, driver with stand-ins, analyzer + recorder on synthetic runs), static checks of the variant
#           (tree; Trainer.py minus the marked blocks == stock + shared hooks), GPU checks (depth from the real kernel:
#           one layer, two layers, empty pixels; flags off: forward, gradients and FusedAdam steps equal stock's)
#   smoke   short garden runs: stock and FasterGSReinit with flags off (no hook output, similar per-step clock), apply
#           at 2000 with target match and +500000 (replacement check in the run; densification continues to 2300 on
#           the new rows), audit at 2000 with the final agreement; then the analyzer and the recorder must accept the
#           real output
#   audit   R1: per scene in SCENES (default "garden room"), seed 0, one 30k run with
#           FGS_REINIT=mode=audit,iters=2000:5000,target=<stock N at 15k>/match and the callback clock. The audit works
#           on a copy; training is stock Faster-GS.
#   STAGES="build unit smoke" sbatch alt_reinit/build_reinit.sbatch        (from the repo root)
#   STAGES=audit sbatch --job-name=gs-alt-audit --time=01:30:00 alt_reinit/build_reinit.sbatch
#SBATCH --job-name=gs-alt-r0
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=c20
#SBATCH --output=alt_reinit/logs/%x-%j.out
set -euo pipefail
STAGES=${STAGES:-"build unit smoke"}
REPO=${SLURM_SUBMIT_DIR:-$PWD}
DATA=$(ws_find -F cat 3dgs-data) || { echo "cat workspace not visible on $(hostname)"; exit 1; }
ARCHIVE=${ARCHIVE:-/projects/p_scads_mlhpc/syas272h/3dgs-archive}
export DATA
module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
source "$DATA/fastergs-venv/bin/activate"
export TORCH_CUDA_ARCH_LIST="8.0;9.0" MAX_JOBS=14 PIP_CACHE_DIR=$DATA/.pipcache
OUT=$REPO/alt_reinit/logs/run_${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
LOG=$REPO/alt_reinit/logs/${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:-local}.out
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) stages=$STAGES repo=$(git -C "$REPO" rev-parse --short HEAD)"
FAILED=()
has() { [[ " $STAGES " == *" $1 "* ]]; }
METHODS=$DATA/nerficg/src/Methods
BSCENE=garden; BSCALE=0.25

# the stock arm must carry the same clock hook as in phase2 (applied there; checked here, never modified)
grep -q '# >>> fgs-sparse-from v5' "$METHODS/FasterGS/Trainer.py" \
  || { echo "FAILED stock FasterGS lacks the v5 clock hook (applied by phase2/build_plus.sbatch)"; FAILED+=(stock_hook); }

if has build; then
  echo "########## build"
  bash "$REPO/alt_reinit/make_reinit_variant.sh" || FAILED+=(build)
fi

if has unit; then
  echo "########## unit"
  ( cd "$REPO" && python -m pytest -q -p no:cacheprovider alt_reinit/tests/test_core_cpu.py alt_reinit/tests/test_trainer_cpu.py \
      alt_reinit/tests/test_analyze_record.py alt_reinit/tests/test_check_variant.py ) > "$OUT/cpu_tests.log" 2>&1 \
    && tail -1 "$OUT/cpu_tests.log" | sed 's/^/CPU tests: /' || { echo "FAILED cpu tests"; tail -30 "$OUT/cpu_tests.log"; FAILED+=(cpu_tests); }
  python3 "$REPO/alt_reinit/tests/check_variant.py" tree "$METHODS/FasterGSReinit" --base 1144604 > "$OUT/check_tree.log" 2>&1 \
    && grep -c '^PASS' "$OUT/check_tree.log" | sed 's/^/static tree checks PASS: /' || { echo "FAILED static tree"; cat "$OUT/check_tree.log"; FAILED+=(check_tree); }
  git -C "$METHODS/FasterGS" show 1144604:Trainer.py > "$OUT/stock_Trainer_1144604.py"
  python3 "$REPO/alt_reinit/tests/check_variant.py" trainer "$OUT/stock_Trainer_1144604.py" "$METHODS/FasterGSReinit/Trainer.py" > "$OUT/check_trainer.log" 2>&1 \
    && grep -c '^PASS' "$OUT/check_trainer.log" | sed 's/^/static Trainer checks PASS: /' || { echo "FAILED static Trainer"; cat "$OUT/check_trainer.log"; FAILED+=(check_trainer); }
  ( cd /tmp && FGS_REF=FasterGSCudaBackend FGS_NEW=FasterGSReinitCudaBackend python -m pytest -q -s -p no:cacheprovider "$REPO/alt_reinit/tests/test_gpu.py" ) \
    > "$OUT/gpu_tests.log" 2>&1 && { grep '^CHECK' "$OUT/gpu_tests.log"; tail -1 "$OUT/gpu_tests.log"; } || { echo "FAILED gpu tests"; tail -40 "$OUT/gpu_tests.log"; FAILED+=(gpu_tests); }
fi

train() {  # name method seed iters reinit_spec cbclock(0|1) [segk]
  local name=$1 method=$2 seed=$3 iters=$4 spec=$5 cbclock=$6 segk=${7:-1000}
  rm -f "$OUT/$name.seg"
  echo "##### $name: method=$method scene=$BSCENE seed=$seed iters=$iters reinit=${spec:--} cbclock=$cbclock $(date +%T)"
  local envs=(FGS_SEGCLOCK=$segk FGS_LOG="$OUT/$name.seg")
  [ -n "$spec" ] && envs+=(FGS_REINIT="$spec")
  [ "$cbclock" = 1 ] && envs+=(FGS_CBCLOCK=1)
  env "${envs[@]}" SCENE=$BSCENE SCALE=$BSCALE SEED=$seed ITERS=$iters METHOD=$method TAG=$name \
      bash "$REPO/cluster/train_fastergs.sbatch" > "$OUT/$name.log" 2>&1 || true
  local RUN; RUN=$(tr '\r' '\n' < "$OUT/$name.log" | sed -n 's/^== run dir //p' | tail -1)
  cat "$OUT/$name.seg" 2>/dev/null || true
  if [ -n "$RUN" ] && [ -d "$DATA/nerficg/$RUN" ]; then
    echo "RUNDIR $name $RUN"
    mkdir -p "$OUT/runs/$name"; echo "$RUN" > "$OUT/runs/$name/rundir.txt"
    ( cd "$DATA/nerficg/$RUN" && cp --parents -t "$OUT/runs/$name" training_config.yaml timings.txt n_gaussians.txt vram_stats.txt \
        gpu_clock_summary.txt test_*/metrics_8bit.txt reinit_audit/k_*/stats.json reinit_audit/k_*/agreement_final.json \
        reinit_audit/k_*/masks/*.jpg 2>/dev/null ) || true
    if [ -d "$DATA/nerficg/$RUN/reinit_audit" ]; then   # full audit output (with the depth maps) outside the workspace
      local ARC=$ARCHIVE/alt_reinit/${SLURM_JOB_ID:-local}/$name
      mkdir -p "$ARC" && rsync -a "$DATA/nerficg/$RUN/reinit_audit" "$ARC/" && echo "AUDITARCHIVE $name $ARC"
    fi
  fi
  if [ -z "$RUN" ] || [ ! -f "$DATA/nerficg/$RUN/test_$iters/metrics_8bit.txt" ] || tr '\r' '\n' < "$OUT/$name.log" | grep -q '^Traceback'; then
    echo "FAILED $name"; tr '\r' '\n' < "$OUT/$name.log" | grep -E "Traceback|Error" | head; FAILED+=("$name"); return
  fi
  echo "PERIMG $name $(awk -F'	' '/^Image/{f=1;next} f&&$1~/^[0-9]+$/{n++; if(min==""||$2<min){min=$2;i=$1}} END{print "n_images=" n " min_image=" i " min_psnr=" min}' "$DATA/nerficg/$RUN/test_$iters/metrics_8bit.txt")"
  local MSIT; MSIT=$(grep -A2 '^training_iteration:' "$DATA/nerficg/$RUN/timings.txt" 2>/dev/null | awk -F': ' '/per iteration/{gsub(/[ \t]+$/, "", $2); print $2}' || true)
  echo "RESULT $name $(tail -1 "$DATA/nerficg/$RUN/test_$iters/metrics_8bit.txt") $(grep -o 'N_Gaussians:[0-9]*' "$DATA/nerficg/$RUN/n_gaussians.txt") ms_it=${MSIT} $(grep allocated "$DATA/nerficg/$RUN/vram_stats.txt" 2>/dev/null | head -1 | tr -s ' \t' ' ')"
}

if has smoke; then
  echo "########## smoke (garden seed 0, short runs)"
  BSCENE=garden; BSCALE=0.25
  train ar0_garden_stock_s0 FasterGS 0 1200 "" 0 300
  train ar0_garden_off_s0 FasterGSReinit 0 1200 "" 0 300
  train ar0_garden_applymatch_s0 FasterGSReinit 0 2300 "mode=apply,iters=2000,target=match" 1 100
  train ar0_garden_applyplus_s0 FasterGSReinit 0 2300 "mode=apply,iters=2000,target=+500000" 1 100
  train ar0_garden_audit_s0 FasterGSReinit 0 2100 "mode=audit,iters=2000,target=match/4166000" 1 100
  python3 - "$OUT" <<'PY' || FAILED+=(smoke_checks)
import json, re, sys
from pathlib import Path
out = Path(sys.argv[1])
seg = lambda n: (out / f'{n}.seg').read_text() if (out / f'{n}.seg').exists() else ''
checks = {}
stock, off = seg('ar0_garden_stock_s0'), seg('ar0_garden_off_s0')
cum = lambda t: [float(x) for x in re.findall(r'^SEGCLOCK step=1200 \S+ cum_s=(\S+)', t, re.M)]
checks['flags off: no REINIT / CBCLOCK output'] = off != '' and not re.search(r'^(REINIT|CBCLOCK)', off, re.M)
c0, c1 = cum(stock), cum(off)
checks['flags off: per-step clock within 5% of stock over 1200 steps'] = bool(c0 and c1 and abs(c1[0] / c0[0] - 1) < 0.05)
print(f'flags-off clock: stock {c0} variant {c1}')
for name, target in (('ar0_garden_applymatch_s0', 'match'), ('ar0_garden_applyplus_s0', '+500000')):
    t = seg(name)
    m = re.search(r'^REINIT_COUNTS iteration=2000 n_before=(\d+) .* n_after=(\d+) target=(\d+)', t, re.M)
    want = None if not m else (int(m.group(1)) if target == 'match' else int(m.group(1)) + 500000)
    checks[f'{name}: replacement check passed'] = 'REINIT_CHECK iteration=2000 result=pass' in t
    checks[f'{name}: count after == target'] = bool(m and int(m.group(2)) == want == int(m.group(3)))
    checks[f'{name}: callback clock logged the reinit and later densify calls'] = bool(
        re.search(r'^CBCLOCK name=reinit_gaussians iteration=2000 ', t, re.M) and re.search(r'^CBCLOCK name=densify iteration=2[12]00 ', t, re.M))
    checks[f'{name}: SEGCLOCK N after the reinit equals the target'] = bool(m and re.search(rf'^SEGCLOCK step=2100 .* N={want} ', t, re.M))
t = seg('ar0_garden_audit_s0')
checks['audit: replacement check passed on the copy'] = 'REINIT_CHECK iteration=2000 result=pass' in t
checks['audit: both targets reported'] = len(re.findall(r'^REINIT_TARGET iteration=2000 ', t, re.M)) == 2
checks['audit: diagnostics and final agreement written'] = 'REINIT_AUDIT iteration=2000' in t and 'REINIT_AGREEMENT k=02000' in t
st = out / 'runs' / 'ar0_garden_audit_s0' / 'reinit_audit' / 'k_02000' / 'stats.json'
checks['audit: stats.json archived with the run'] = st.exists()
if st.exists():
    s = json.loads(st.read_text())
    checks['audit: SfM check ran (no error) and reprojection median < 1 px'] = 'error' not in s['sfm'] and s['sfm']['reprojection_px'].get('q50', 9) < 1.0
    print('SfM', json.dumps(s['sfm'])[:600])
    print('timing', s['timing'])
for k, v in checks.items():
    print(('PASS ' if v else 'FAIL ') + k)
sys.exit(0 if all(checks.values()) else 1)
PY
  python3 "$REPO/alt_reinit/analyze_audit.py" "$LOG" > "$OUT/smoke_audit_analysis.md" 2>&1 \
    && echo "PASS analyzer accepted the smoke audit" || { echo "FAIL analyzer"; cat "$OUT/smoke_audit_analysis.md"; FAILED+=(smoke_analyzer); }
  python3 "$REPO/alt_reinit/record_runs.py" "$LOG" --dry-run > "$OUT/smoke_record.txt" 2>&1 \
    && grep -q '^5 new row' "$OUT/smoke_record.txt" && ! grep -q 'FAILED' "$OUT/smoke_record.txt" \
    && echo "PASS recorder parsed the 5 smoke runs, none failed" || { echo "FAIL recorder"; cat "$OUT/smoke_record.txt"; FAILED+=(smoke_recorder); }
fi

if has audit; then
  for BSCENE in ${SCENES:-garden room}; do
    case $BSCENE in
      garden) BSCALE=0.25; N15=4166000 ;;   # stock N at 15k, phase2/logs/bench-analysis.md (3-seed mean)
      room)   BSCALE=0.5;  N15=1345000 ;;
      *) echo "FAILED no stock 15k count recorded for $BSCENE"; FAILED+=("audit_$BSCENE"); continue ;;
    esac
    echo "########## audit (R1) $BSCENE scale $BSCALE: k = 2000, 5000; targets $N15 (stock N at 15k) / match"
    train "ar1_${BSCENE}_audit_s0" FasterGSReinit 0 30000 "mode=audit,iters=2000:5000,target=$N15/match" 1 1000
  done
fi

if [ ${#FAILED[@]} -gt 0 ]; then echo "JOB FAILED: ${FAILED[*]}"; exit 1; fi
echo "JOB OK"
