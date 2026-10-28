"""Attribute GPU idle gaps in an nsys capture to what the CPU was doing while the GPU waited.

    python3 phase2/host_gaps.py <capture>.sqlite [--steps 50]
    python3 phase2/host_gaps.py <capture>.sqlite --check-backtraces   # exit 0 only if Python frames are resolved
    python3 phase2/host_gaps.py <capture>.sqlite --schema             # print callchain-related tables

For captures made with  nsys profile --pytorch=autograd-nvtx --python-backtrace=cuda  (phase2/build_plus.sbatch,
STAGES=hostprof); also works without them, with coarser ranges and no call sites.

Gap attribution. The GPU goes idle at g0 (end of the last running op) and resumes at g1 with an op whose CUDA runtime
call (joined by correlationId) ran on the CPU from E (API entry) to R (API return). The gap is split into:
  - before API entry  [g0, min(E, g1)]: the CPU had not yet asked for the next op. This part is charged to what the
    launching thread was doing (sampled every 2 us: innermost NVTX range open on that thread, e.g. a PyTorch op such as
    aten::item, or the trainer's fwd / bwd / opt range; '(between ranges)' if none).
  - inside the API call  [max(g0, E), min(R, g1)]: the CPU was in the runtime call that submits the op (reported by
    API name; a slow call here is host cost, not GPU latency).
  - after API return  [max(g0, R), g1]: the op had been submitted and the GPU had not started it yet (launch latency).
This says what the CPU was busy with during each gap; it does not by itself prove that this work caused the gap.

Python call sites. Only the documented relationship is used: CUPTI_ACTIVITY_KIND_RUNTIME.callchainId ->
CUDA_CALLCHAINS.id (frames ordered by stackDepth; symbol and module resolved through StringIds). A frame counts as a
Python frame only if its resolved module or symbol names a .py file. No other table is used as a fallback.
"""
import argparse
import bisect
import re
import sqlite3
import sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('db')
ap.add_argument('--steps', type=int, default=50)
ap.add_argument('--min-gap-us', type=float, default=3.0)
ap.add_argument('--schema', action='store_true')
ap.add_argument('--check-backtraces', action='store_true')
args = ap.parse_args()
c = sqlite3.connect(args.db)
S = dict(c.execute('select id, value from StringIds'))
tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
cols_of = lambda t: [r[1] for r in c.execute(f'pragma table_info({t})')]

if args.schema:
    for t in sorted(tables):
        if re.search('CALLCHAIN|PYTHON|BACKTRACE|SAMPLING|COMPOSITE', t, re.I):
            print(t, c.execute(f'select count(*) from {t}').fetchone()[0], cols_of(t))
            for row in c.execute(f'select * from {t} limit 4'):
                print('   ', row)
    print('runtime calls with callchainId:',
          c.execute('select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME where callchainId is not null').fetchone()[0])
    sys.exit(0)

# --- CUDA runtime calls: correlationId -> (entry, return, thread, api name, callchainId)
rt = {}
for start, end, tid, corr, nid, cc in c.execute(
        'select start, end, globalTid, correlationId, nameId, callchainId from CUPTI_ACTIVITY_KIND_RUNTIME'):
    rt[corr] = (start, end, tid, S.get(nid, '?'), cc)

# --- Python frames per callchain, via CUDA_CALLCHAINS only
resolve = lambda v: S.get(v, v) if isinstance(v, int) else v
is_py = lambda s: bool(s) and bool(re.search(r'\.py\b', str(s)))
py_frames = {}
if 'CUDA_CALLCHAINS' in tables:
    cols = cols_of('CUDA_CALLCHAINS')
    if {'id', 'symbol', 'stackDepth'} <= set(cols):
        mod = 'module' if 'module' in cols else 'NULL'
        frames = defaultdict(list)
        for cid, depth, sym, md in c.execute(f'select id, stackDepth, symbol, {mod} from CUDA_CALLCHAINS'):
            frames[cid].append((depth, str(resolve(sym) or ''), str(resolve(md) or '')))
        for cid, fr in frames.items():
            py = [(d, s, m) for d, s, m in sorted(fr) if is_py(s) or is_py(m)]
            if py:
                py_frames[cid] = py

if args.check_backtraces:
    launches = [v for v in rt.values() if re.match(r'cudaLaunchKernel|cuLaunchKernel', v[3])]
    with_py = sum(1 for v in launches if v[4] in py_frames)
    frac = with_py / max(len(launches), 1)
    print(f'kernel launch calls: {len(launches)}; with resolved Python frames (CUDA_CALLCHAINS via callchainId): '
          f'{with_py} ({100 * frac:.0f}%)')
    sys.exit(0 if launches and frac >= 0.5 else 1)


def callsite(cc):
    """First Python frame outside torch / site-packages / the NeRFICG base framework, else the first Python frame."""
    fr = py_frames.get(cc)
    if not fr:
        return '(no Python frame)'
    txt = lambda f: f'{f[2] or ""} {f[1]}'.strip()
    pick = next((f for f in fr if not re.search(r'site-packages|/torch/|Methods/Base/|/lib/python', txt(f))), fr[0])
    return txt(pick).split('/src/')[-1]


# --- NVTX push/pop ranges per thread
ranges = defaultdict(list)
for start, end, text, tid, tidx in c.execute('select start, end, text, globalTid, textId from NVTX_EVENTS where end is not null'):
    t = text if text is not None else S.get(tidx, '')
    t = re.sub(r',? *(seq|op_id|sizes|input_op_ids|Input Dims|input_dims) *[=:].*', '', t or '').strip()
    ranges[tid].append((start, end, t))
for tid in ranges:
    ranges[tid].sort()
rstarts = {tid: [r[0] for r in rs] for tid, rs in ranges.items()}


def open_ranges(tid, t):
    rs = ranges.get(tid, [])
    i = bisect.bisect_right(rstarts.get(tid, []), t)
    return [txt for s, e, txt in rs[max(0, i - 800):i] if s <= t <= e]   # outermost first


ops = []
for start, end, corr, name in c.execute('select start, end, correlationId, coalesce(demangledName, shortName) from CUPTI_ACTIVITY_KIND_KERNEL'):
    ops.append((start, end, corr, 'K:' + re.sub(r'\(.*', '', str(resolve(name))).replace('void ', '').split('::')[-1][:50]))
for start, end, corr, kind in c.execute('select start, end, correlationId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY'):
    ops.append((start, end, corr, {1: 'memcpy HtoD', 2: 'memcpy DtoH', 8: 'memcpy DtoD'}.get(kind, f'memcpy kind {kind}')))
for start, end, corr in c.execute('select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMSET'):
    ops.append((start, end, corr, 'memset'))
ops.sort()

before = defaultdict(float)     # (outer, innermost) range on the launching thread -> ns
inside = defaultdict(float)     # api name -> ns
site_before = defaultdict(float)
resumed = defaultdict(lambda: [0.0, 0])
after_total = inside_total = before_total = unknown = total = 0.0
cur_end = None
STEP = 2000
for start, end, corr, name in ops:
    if cur_end is not None and start > cur_end and start - cur_end >= args.min_gap_us * 1e3:
        g0, g1 = cur_end, start
        total += g1 - g0
        resumed[name][0] += g1 - g0
        resumed[name][1] += 1
        call = rt.get(corr)
        if not call:
            unknown += g1 - g0
        else:
            E, R, tid, api, cc = call
            b = max(0, min(E, g1) - g0)
            ins = max(0, min(R, g1) - max(g0, E))
            aft = max(0, g1 - max(g0, R))
            before_total += b; inside_total += ins; after_total += aft
            inside[api] += ins
            if b > 0:
                n = max(1, int(b // STEP))
                for k in range(n):
                    t = g0 + (k + 0.5) * b / n
                    rs = open_ranges(tid, t)
                    before[(rs[0] if rs else '(between ranges)', rs[-1] if rs else '(between ranges)')] += b / n
                if py_frames:
                    site_before[callsite(cc)] += b
    cur_end = end if cur_end is None else max(cur_end, end)

per = lambda ns: ns / args.steps / 1e3
span = ops[-1][1] - ops[0][0]
print(f'GPU span {per(span) / 1e3:.3f} ms/step; idle gaps >= {args.min_gap_us:g} us: {per(total):.0f} us/step = '
      f'{per(before_total):.0f} before API entry + {per(inside_total):.0f} inside the API call + '
      f'{per(after_total):.0f} after API return (launch latency)' + (f' + {per(unknown):.0f} without a runtime call' if unknown else '') + '\n')
print('Before API entry: what the launching CPU thread was doing (us/step):\n')
print('| us/step | outer range | innermost range |')
print('|---|---|---|')
for (o, i), v in sorted(before.items(), key=lambda x: -x[1])[:25]:
    print(f'| {per(v):.1f} | {o[:45]} | {i[:70]} |')
print('\nInside the API call that submitted the next op (us/step):\n')
print('| us/step | API |')
print('|---|---|')
for k, v in sorted(inside.items(), key=lambda x: -x[1])[:10]:
    print(f'| {per(v):.1f} | {k} |')
if py_frames:
    print('\nPython call site of the late launch, for the before-entry part (us/step):\n')
    print('| us/step | call site |')
    print('|---|---|')
    for k, v in sorted(site_before.items(), key=lambda x: -x[1])[:20]:
        print(f'| {per(v):.1f} | {k[:110]} |')
else:
    print('\n(no resolved Python frames in CUDA_CALLCHAINS: call sites not reported)')
print('\nGPU op that resumed after each gap (where execution restarts, not necessarily the cause):\n')
print('| us/step | gaps/step | op |')
print('|---|---|---|')
for k, (v, n) in sorted(resumed.items(), key=lambda x: -x[1][0])[:12]:
    print(f'| {per(v):.1f} | {n / args.steps:.1f} | {k} |')

print('\nBlocking host calls per step:\n')
blk = defaultdict(lambda: [0, 0.0])
for corr, (E, R, tid, api, cc) in rt.items():
    if re.search(r'cudaStreamSynchronize|cudaDeviceSynchronize|cudaEventSynchronize|^cudaMemcpy(_v\d+)?$', api):
        rs = open_ranges(tid, E)
        k = (api, rs[0] if rs else '(between ranges)', rs[-1] if rs else '(between ranges)', callsite(cc) if py_frames else '')
        blk[k][0] += 1
        blk[k][1] += R - E
print('| calls/step | host us/step in call | call | outer range | innermost range | call site |')
print('|---|---|---|---|---|---|')
for (api, o, i, site), (n, t) in sorted(blk.items(), key=lambda x: -x[1][0])[:20]:
    print(f'| {n / args.steps:.1f} | {per(t):.0f} | {api} | {o[:30]} | {i[:50]} | {site[:70]} |')
print('\nNote: time inside a blocking call is CPU waiting; the GPU may be busy during it, so it is not a GPU stall.')
