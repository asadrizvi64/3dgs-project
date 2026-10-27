#!/usr/bin/env bash
# Phase 1 profiling of Faster-GS main on one scene (default garden, seed 0):
#   0. device info + measured memory bandwidth (copy / read)
#   A. full 30k run with STAGE (fwd/bwd/opt CUDA events at 500/5000/20000) + CURVE (ms/step vs #Gaussians every 500 steps)
#   B. nsys capture of steps 5000-5049 (exits after) -> kernel and NVTX summary
#   C. ncu on one training step (5000) with DRAM byte + throughput metrics, real clocks (exits after)
# All runs log GPU clocks/power (see train_fastergs.sbatch). Reports go to $DATA/prof/.
#   sbatch --export=ALL,SCENE=garden cluster/profile_fastergs.sbatch      (from the repo root)
#SBATCH --job-name=gs-prof-fastergs
#SBATCH --nodes=1
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail

SCENE=${SCENE:-garden}
SCALE=${SCALE:-0.25}
SEED=${SEED:-0}
HERE=${SLURM_SUBMIT_DIR:-$PWD}/cluster
DATA=$(ws_find -F cat 3dgs-data)
PROF=$DATA/prof; mkdir -p "$PROF"
TAGBASE=${SCENE}_main_${SLURM_JOB_ID:-local}

module load GCC/13.3.0 CUDA/12.8.0 Python/3.12.3 >/dev/null 2>&1
source "$DATA/fastergs-venv/bin/activate"
NSYS=$(command -v nsys || ls -d "${EBROOTCUDA:-/nonexistent}"/nsight-systems*/bin/nsys 2>/dev/null | head -1 || true)
NCU=$(command -v ncu || true)
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} nsys=$NSYS ncu=$NCU"

python "$HERE/patches/fastergs_profiling.py" "$DATA/nerficg/src/Methods/FasterGS/Trainer.py"

echo; echo "########## 0. device"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.memory,power.limit,pci.device_id --format=csv,noheader
python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print('DEVICE', p)
for k in ('memory_clock_rate', 'memory_bus_width', 'L2_cache_size', 'multi_processor_count'):
    print(f'DEVICE {k} = {getattr(p, k, "n/a")}')
if hasattr(p, 'memory_bus_width') and hasattr(p, 'memory_clock_rate'):
    print(f'DEVICE theoretical_dram_GBps = {2 * p.memory_clock_rate * 1e3 * p.memory_bus_width / 8 / 1e9:.0f}')
n = 2 * 1024**3 // 4                       # 2 GiB of fp32
a = torch.empty(n, device='cuda'); b = torch.empty(n, device='cuda'); a.uniform_()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
def bench(fn, nbytes, reps=20):
    for _ in range(3): fn()
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return nbytes * reps / (s.elapsed_time(e) / 1e3) / 1e9
print(f'BANDWIDTH copy (read+write) = {bench(lambda: b.copy_(a), 2 * a.numel() * 4):.0f} GB/s')
print(f'BANDWIDTH read (sum)        = {bench(lambda: a.sum(), a.numel() * 4):.0f} GB/s')
PY

echo; echo "########## A. stage timing + step curve (full run)"
FGS_PROF=1 SCENE=$SCENE SCALE=$SCALE SEED=$SEED METHOD=FasterGS TAG=prof \
  bash "$HERE/train_fastergs.sbatch" | tee "$PROF/${TAGBASE}_stage_curve.log" | grep -E "^(STAGE|CURVE|CLOCK)|PSNR|SSIM|LPIPS|Gaussians|Error|Traceback" || true

echo; echo "########## B. nsys, steps 5000-5049"
if [ -n "$NSYS" ]; then
  FGS_PROF_RANGE=5000,5050 FGS_PROF_EXIT=5050 SCENE=$SCENE SCALE=$SCALE SEED=$SEED METHOD=FasterGS TAG=nsys \
  LAUNCHER="$NSYS profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o $PROF/${TAGBASE}_5k" \
    bash "$HERE/train_fastergs.sbatch" > "$PROF/${TAGBASE}_nsys_run.log" 2>&1 || true
  tail -5 "$PROF/${TAGBASE}_nsys_run.log"
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv --output "$PROF/${TAGBASE}_5k" "$PROF/${TAGBASE}_5k.nsys-rep" >/dev/null 2>&1 || true
  "$NSYS" stats --report nvtx_sum --format csv --output "$PROF/${TAGBASE}_5k" "$PROF/${TAGBASE}_5k.nsys-rep" >/dev/null 2>&1 || true
  ls "$PROF" | grep "${TAGBASE}_5k"
  python - "$PROF/${TAGBASE}_5k_cuda_gpu_kern_sum.csv" "$PROF/${TAGBASE}_5k_nvtx_sum.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r['Total Time (ns)']) for r in rows)
print(f'NSYS kernels: {len(rows)} distinct, total GPU kernel time {tot/1e6:.1f} ms over 50 steps = {tot/50/1e6:.3f} ms/step')
print(f'NSYS {"%":>5} {"ms/step":>8} {"calls":>6} {"avg_us":>8}  kernel')
for r in rows[:15]:
    t = float(r['Total Time (ns)'])
    print(f'NSYS {100*t/tot:5.1f} {t/50/1e6:8.3f} {int(r["Instances"]):6d} {float(r["Avg (ns)"])/1e3:8.1f}  {r["Name"][:110]}')
try:
    for r in csv.DictReader(open(sys.argv[2])):
        print(f'NVTX {r["Range"]:>6} total {float(r["Total Time (ns)"])/1e6:9.2f} ms  avg {float(r["Avg (ns)"])/1e6:7.3f} ms  n={r["Instances"]}')
except Exception as ex:
    print('NVTX summary unavailable:', ex)
PY
else
  echo "nsys not found"
fi

echo; echo "########## C. ncu, one training step (5000), DRAM traffic per kernel"
if [ -n "$NCU" ]; then
  M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum.peak_sustained,dram__cycles_elapsed.avg.per_second
  FGS_PROF_RANGE=5000,5001 FGS_PROF_EXIT=5001 SCENE=$SCENE SCALE=$SCALE SEED=$SEED METHOD=FasterGS TAG=ncu \
  LAUNCHER="$NCU --profile-from-start off --clock-control none --target-processes all --metrics $M -f -o $PROF/${TAGBASE}_step5000" \
    bash "$HERE/train_fastergs.sbatch" > "$PROF/${TAGBASE}_ncu_run.log" 2>&1 || true
  tail -5 "$PROF/${TAGBASE}_ncu_run.log"
  "$NCU" -i "$PROF/${TAGBASE}_step5000.ncu-rep" --csv --page raw --metrics "$M" > "$PROF/${TAGBASE}_step5000.csv" 2>/dev/null || true
  python - "$PROF/${TAGBASE}_step5000.csv" <<'PY'
import csv, sys, collections
rows = list(csv.reader(open(sys.argv[1])))
if len(rows) < 3:
    print('NCU no data'); sys.exit()
hdr, units, data = rows[0], rows[1], rows[2:]
ix = {h: i for i, h in enumerate(hdr)}
def v(r, k):
    try: return float(r[ix[k]].replace(',', ''))
    except Exception: return 0.0
unit = units[ix['gpu__time_duration.sum']]
tscale = {'nsecond': 1e-9, 'usecond': 1e-6, 'msecond': 1e-3, 'second': 1.0}.get(unit, 1e-9)
bscale = lambda k: {'byte': 1, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9}.get(units[ix[k]], 1)
agg = collections.OrderedDict()
peak = 0.0
for r in data:
    name = r[ix['Kernel Name']]
    t = v(r, 'gpu__time_duration.sum') * tscale
    rd = v(r, 'dram__bytes_read.sum') * bscale('dram__bytes_read.sum')
    wr = v(r, 'dram__bytes_write.sum') * bscale('dram__bytes_write.sum')
    l2 = v(r, 'lts__t_bytes.sum') * bscale('lts__t_bytes.sum')
    a = agg.setdefault(name, [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    a[0] += 1; a[1] += t; a[2] += rd; a[3] += wr; a[4] += l2
    a[5] = max(a[5], v(r, 'dram__throughput.avg.pct_of_peak_sustained_elapsed'))
    a[6] = max(a[6], v(r, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'))
    pk = v(r, 'dram__bytes.sum.peak_sustained') * v(r, 'dram__cycles_elapsed.avg.per_second')
    peak = max(peak, pk)
T = sum(a[1] for a in agg.values()); R = sum(a[2] for a in agg.values()); W = sum(a[3] for a in agg.values())
print(f'NCU step 5000: {len(data)} kernel launches, kernel time {T*1e3:.3f} ms, DRAM read {R/1e6:.1f} MB, write {W/1e6:.1f} MB, '
      f'avg {(R+W)/T/1e9 if T else 0:.0f} GB/s; peak DRAM BW (ncu) {peak/1e9:.0f} GB/s')
print(f'NCU {"%time":>5} {"us":>8} {"n":>3} {"rd_MB":>8} {"wr_MB":>8} {"GB/s":>6} {"dram%":>6} {"sm%":>5}  kernel')
for name, a in sorted(agg.items(), key=lambda kv: -kv[1][1])[:12]:
    bw = (a[2] + a[3]) / a[1] / 1e9 if a[1] else 0
    print(f'NCU {100*a[1]/T:5.1f} {a[1]*1e6:8.1f} {a[0]:3d} {a[2]/1e6:8.1f} {a[3]/1e6:8.1f} {bw:6.0f} {a[5]:6.1f} {a[6]:5.1f}  {name[:90]}')
PY
else
  echo "ncu not found"
fi
echo; echo "########## done"
