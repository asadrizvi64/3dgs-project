#!/usr/bin/env bash
# Phase 0 kernel benchmark on one A100: forward, backward (A) per-pixel, backward (B) per-splat.
#   A. timing + CPU-emulation op counts + GPU correctness on three scenes (backward_bench.py)
#   B. ncu on one launch of each kernel per scene: time, DRAM bytes, throughput, atomics
#      (RED/ATOM requests at L1 and L2), instructions, occupancy, warp stall reasons.
# Reports go to $W/prof_phase0/.
#   sbatch bench/phase0/profile_backward.sbatch                      (A100, from the repo root)
#   sbatch -p capella --export=ALL,SKIP_TIMING=1 bench/phase0/profile_backward.sbatch
#        (H100 ncu pass; on Alpha, DCGM monitoring can hold the profiling counters)
#SBATCH --job-name=phase0-bench
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=bench/phase0/logs/%x-%j.out
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$PWD}
module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
PART=${SLURM_JOB_PARTITION:-alpha}
W=$(ws_find -F horse 3dgs)
if [ "$PART" = capella ]; then
  VENV=$(ws_find -F cat 3dgs-data)/venv   # built against the Capella (genoa) module tree
else
  VENV=$W/venv                            # built against the Alpha (rome) module tree
fi
source "$VENV/bin/activate"
export PYTHONPATH=$HOME/py-extra TORCH_CUDA_ARCH_LIST="8.0;9.0" SPLAT2D_WITH_CUDA=1 MAX_JOBS=6
export TORCH_EXTENSIONS_DIR=$W/torch_extensions_phase0_$PART
PROF=$W/prof_phase0; mkdir -p "$PROF"
NCU=$(command -v ncu || true)
cd "$ROOT"
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD) ncu=$NCU"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.memory,power.limit --format=csv,noheader

# name | args
SCENES=(
  "hd50k|--height 1080 --width 1920 --n 50000 --sigma 2 10"
  "hd200k|--height 1080 --width 1920 --n 200000 --sigma 2 10"
  "overlap|--height 512 --width 512 --n 20000 --sigma 4 20"
)

if [ -z "${SKIP_TIMING:-}" ]; then
  echo; echo "########## A. timing, op counts, correctness"
  for s in "${SCENES[@]}"; do
    python bench/phase0/backward_bench.py ${s#*|}
  done
fi

echo; echo "########## B. ncu, one launch per kernel"
if [ -z "$NCU" ]; then echo "ncu not found"; exit 0; fi
# Free the profiling counters if DCGM monitoring holds them (no-op where not permitted).
if command -v dcgmi >/dev/null && dcgmi profile --pause >/dev/null 2>&1; then
  echo "dcgm profiling paused"; trap 'dcgmi profile --resume >/dev/null 2>&1 || true' EXIT
fi
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum
M=$M,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed
M=$M,l1tex__t_requests_pipe_lsu_mem_global_op_red.sum,l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum
M=$M,lts__t_requests_op_red.sum,lts__t_requests_op_atom.sum,smsp__inst_executed.sum
M=$M,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__block_size
# Stall reasons per issued instruction (as in Nsight's warp-state view).
for r in lg_throttle long_scoreboard short_scoreboard mio_throttle barrier; do
  M=$M,smsp__average_warps_issue_stalled_${r}_per_issue_active.ratio
done
for s in "${SCENES[@]}"; do
  name=${s%%|*}
  rep=$PROF/${name}_${SLURM_JOB_ID:-local}
  "$NCU" --profile-from-start off --clock-control none --metrics "$M" -f -o "$rep" \
    python bench/phase0/backward_bench.py ${s#*|} --ncu --no-counters > "$rep.run.log" 2>&1 || { tail -20 "$rep.run.log"; continue; }
  "$NCU" -i "$rep.ncu-rep" --csv --page raw --metrics "$M" > "$rep.csv"
  python - "$rep.csv" "$name" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
hdr, units, data = rows[0], rows[1], rows[2:]
ix = {h: i for i, h in enumerate(hdr)}
UNIT = {'ns': 1e-3, 'nsecond': 1e-3, 'us': 1.0, 'usecond': 1.0, 'ms': 1e3, 'msecond': 1e3,
        's': 1e6, 'second': 1e6, 'byte': 1e-6, 'Kbyte': 1e-3, 'Mbyte': 1.0, 'Gbyte': 1e3}
def v(r, k):
    try: return float(r[ix[k]].replace(',', ''))
    except Exception: return float('nan')
def sc(r, k):  # time -> us, bytes -> MB
    return v(r, k) * UNIT.get(units[ix[k]], 1.0)
def st(r, name):
    return v(r, f'smsp__average_warps_issue_stalled_{name}_per_issue_active.ratio')
short = lambda n: n.split('(')[0].split('<')[0].replace('void ', '').replace('splat2d::', '')
ours = [r for r in data if 'splat2d' in r[ix['Kernel Name']]]
other_us = sum(sc(r, 'gpu__time_duration.sum') for r in data if r not in ours)
tag = f"NCU {sys.argv[2]}:"
print(f"{tag} {'kernel':22s} {'us':>8} {'grid':>6} {'blk':>4} {'DRAM_MB':>8} {'L2_MB':>8} {'sm%':>5} {'dram%':>5} "
      f"{'L1_red':>9} {'L2_red':>10} {'inst_M':>7} {'occ%':>5} | stall/inst: {'lgthr':>6} {'lsb':>5} {'ssb':>5} {'mio':>5} {'bar':>5}")
for r in ours:
    print(f"{tag} {short(r[ix['Kernel Name']])[:22]:22s} {sc(r,'gpu__time_duration.sum'):8.1f} "
          f"{v(r,'launch__grid_size'):6.0f} {v(r,'launch__block_size'):4.0f} "
          f"{sc(r,'dram__bytes_read.sum')+sc(r,'dram__bytes_write.sum'):8.1f} {sc(r,'lts__t_bytes.sum'):8.1f} "
          f"{v(r,'sm__throughput.avg.pct_of_peak_sustained_elapsed'):5.1f} "
          f"{v(r,'dram__throughput.avg.pct_of_peak_sustained_elapsed'):5.1f} "
          f"{v(r,'l1tex__t_requests_pipe_lsu_mem_global_op_red.sum'):9.0f} {v(r,'lts__t_requests_op_red.sum'):10.0f} "
          f"{v(r,'smsp__inst_executed.sum')/1e6:7.1f} {v(r,'sm__warps_active.avg.pct_of_peak_sustained_active'):5.1f} | "
          f"{'':11s} {st(r,'lg_throttle'):6.2f} {st(r,'long_scoreboard'):5.2f} {st(r,'short_scoreboard'):5.2f} "
          f"{st(r,'mio_throttle'):5.2f} {st(r,'barrier'):5.2f}")
print(f"{tag} other kernels (torch zero-fills etc.): {other_us:.1f} us")
PY
done
echo; echo "########## done"
