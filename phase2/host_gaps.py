"""Attribute GPU idle gaps in an nsys capture to what the CPU was doing while the GPU waited.

    python3 phase2/host_gaps.py <capture>.sqlite [--steps 50] [--schema]

For captures made with  nsys profile --pytorch=autograd-nvtx --python-backtrace=cuda  (phase2/build_plus.sbatch,
STAGES=hostprof); also works without them, with coarser ranges.

How a gap is attributed:
  The GPU goes idle at g0 (the end of the last running op) and resumes at g1. The next op was launched on the CPU at L
  (its CUDA runtime call, joined by correlationId). Between g0 and min(L, g1) the GPU waited for that launch, so the
  gap is charged to what the launching CPU thread was doing in that interval: the interval is sampled every 2 us and
  each sample goes to the innermost NVTX range open on that thread (a PyTorch op such as aten::item, or the trainer's
  fwd / bwd / opt range; '(between ranges)' if none). Any part of the gap after L (launch already issued, GPU not yet
  started) is reported separately as launch latency. This says what the CPU was busy with during the gap; it does not
  by itself prove that this work caused the gap (it would have, had it finished earlier).

Also listed: the GPU op that resumed after each gap (where execution restarts), blocking host calls with their ranges,
and, if the capture has Python backtraces, the Python call site of the launch that ended each gap.
"""
import argparse
import bisect
import re
import sqlite3
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('db')
ap.add_argument('--steps', type=int, default=50)
ap.add_argument('--min-gap-us', type=float, default=3.0)
ap.add_argument('--schema', action='store_true', help='print the callchain/backtrace table layouts and exit')
args = ap.parse_args()
c = sqlite3.connect(args.db)
S = dict(c.execute('select id, value from StringIds'))
tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
cols_of = lambda t: [r[1] for r in c.execute(f'pragma table_info({t})')]

if args.schema:
    for t in sorted(tables):
        if re.search('CALLCHAIN|PYTHON|BACKTRACE|SAMPLING|COMPOSITE', t, re.I):
            print(t, c.execute(f'select count(*) from {t}').fetchone()[0], cols_of(t))
            for row in c.execute(f'select * from {t} limit 3'):
                print('   ', row)
    raise SystemExit

# --- CUDA runtime calls: correlationId -> (start, end, thread, name, callchainId)
rt = {}
for start, end, tid, corr, nid, cc in c.execute(
        'select start, end, globalTid, correlationId, nameId, callchainId from CUPTI_ACTIVITY_KIND_RUNTIME'):
    rt[corr] = (start, end, tid, S.get(nid, '?'), cc)

# --- NVTX push/pop ranges per thread
ranges = defaultdict(list)
for start, end, text, tid, tidx in c.execute('select start, end, text, globalTid, textId from NVTX_EVENTS where end is not null'):
    t = text if text is not None else S.get(tidx, '')
    t = re.sub(r',? *(seq|op_id|sizes|input_op_ids|Input Dims|input_dims) *[=:].*', '', t or '').strip()
    ranges[tid].append((start, end, t))
for tid in ranges:
    ranges[tid].sort()
rstarts = {tid: [r[0] for r in rs] for tid, rs in ranges.items()}


def innermost(tid, t):
    rs = ranges.get(tid, [])
    i = bisect.bisect_right(rstarts.get(tid, []), t)
    best = None
    for s, e, txt in reversed(rs[max(0, i - 600):i]):
        if s <= t <= e and (best is None or s >= best[0]):
            best = (s, txt)
            break   # sorted by start: the latest-starting open range is the innermost
    return best[1] if best else '(between ranges)'


def outermost(tid, t):
    out = '(between ranges)'
    rs = ranges.get(tid, [])
    i = bisect.bisect_right(rstarts.get(tid, []), t)
    for s, e, txt in rs[max(0, i - 600):i]:
        if s <= t <= e:
            return txt
    return out


# --- Python call sites (if the capture has backtraces): callchainId -> 'file:line function' of the first frame
#     outside torch / site-packages / this NeRFICG base framework
callsite_of = {}
cc_tables = [t for t in tables if re.search('PYTHON.*CALLCHAIN|CALLCHAIN.*PYTHON', t, re.I)] or \
            [t for t in tables if re.search('CALLCHAIN', t, re.I)]
for t in cc_tables:
    cols = cols_of(t)
    idc = next((x for x in cols if x.lower() in ('id', 'callchainid')), None)
    depth = next((x for x in cols if 'depth' in x.lower()), None)
    func = next((x for x in cols if re.search('symbol|function|name', x, re.I) and x != idc), None)
    filec = next((x for x in cols if re.search('file|module|source', x, re.I)), None)
    linec = next((x for x in cols if re.search('line', x, re.I)), None)
    if not (idc and func):
        continue
    frames = defaultdict(list)
    q = f'select {idc}, {depth or 0}, {func}, {filec or "NULL"}, {linec or "NULL"} from {t}'
    for cid, d, fn, fl, ln in c.execute(q):
        fn = S.get(fn, fn) if isinstance(fn, int) else fn
        fl = S.get(fl, fl) if isinstance(fl, int) else fl
        frames[cid].append((d or 0, str(fn), str(fl or ''), ln))
    for cid, fr in frames.items():
        fr.sort()
        pick = next((f for f in fr if f[2] and not re.search(r'site-packages|/torch/|Methods/Base/|/lib/python', f[2])), fr[0])
        callsite_of[cid] = f'{pick[2].split("/src/")[-1]}:{pick[3]} {pick[1]}'.strip()
    if callsite_of:
        print(f'(Python call sites from table {t})')
        break

# --- GPU ops in time order
ops = []
for start, end, corr, name in c.execute('select start, end, correlationId, coalesce(demangledName, shortName) from CUPTI_ACTIVITY_KIND_KERNEL'):
    ops.append((start, end, corr, 'K:' + re.sub(r'\(.*', '', S.get(name, str(name))).replace('void ', '').split('::')[-1][:50]))
for start, end, corr, kind in c.execute('select start, end, correlationId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY'):
    ops.append((start, end, corr, {1: 'memcpy HtoD', 2: 'memcpy DtoH', 8: 'memcpy DtoD'}.get(kind, f'memcpy kind {kind}')))
for start, end, corr in c.execute('select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMSET'):
    ops.append((start, end, corr, 'memset'))
ops.sort()

cpu_during = defaultdict(float)     # (outer range, innermost range on the launching thread) -> ns
resumed_by = defaultdict(lambda: [0.0, 0])
site_gap = defaultdict(float)
launch_lat = 0.0
total = 0.0
cur_end = None
STEP = 2000   # 2 us sampling of the CPU thread during a gap
for start, end, corr, name in ops:
    if cur_end is not None and start > cur_end and start - cur_end >= args.min_gap_us * 1e3:
        g0, g1 = cur_end, start
        total += g1 - g0
        launch = rt.get(corr)
        resumed_by[name][0] += g1 - g0
        resumed_by[name][1] += 1
        if launch:
            L, tid, cc = launch[0], launch[2], launch[4]
            w_end = min(L, g1)
            if w_end > g0:
                n = max(1, int((w_end - g0) // STEP))
                for k in range(n):
                    t = g0 + (k + 0.5) * (w_end - g0) / n
                    cpu_during[(outermost(tid, t), innermost(tid, t))] += (w_end - g0) / n
            launch_lat += max(0, g1 - max(L, g0))
            if callsite_of:
                site_gap[callsite_of.get(cc, '(no backtrace)')] += g1 - g0
    cur_end = end if cur_end is None else max(cur_end, end)

per = lambda ns: ns / args.steps / 1e3
span = ops[-1][1] - ops[0][0]
print(f'GPU span {per(span) / 1e3:.3f} ms/step; idle gaps >= {args.min_gap_us:g} us: {per(total):.0f} us/step, '
      f'of which {per(launch_lat):.0f} us/step after the launch was already issued (launch latency)\n')
print('What the launching CPU thread was doing while the GPU waited (us/step):\n')
print('| us/step | outer range | innermost range |')
print('|---|---|---|')
for (o, i), v in sorted(cpu_during.items(), key=lambda x: -x[1])[:25]:
    print(f'| {per(v):.1f} | {o[:45]} | {i[:70]} |')
if site_gap:
    print('\nPython call site of the launch that ended each gap (idle us/step before it):\n')
    print('| us/step | call site |')
    print('|---|---|')
    for k, v in sorted(site_gap.items(), key=lambda x: -x[1])[:20]:
        print(f'| {per(v):.1f} | {k[:110]} |')
print('\nGPU op that resumed after each gap (where execution restarts, not necessarily the cause):\n')
print('| us/step | gaps/step | op |')
print('|---|---|---|')
for k, (v, n) in sorted(resumed_by.items(), key=lambda x: -x[1][0])[:12]:
    print(f'| {per(v):.1f} | {n / args.steps:.1f} | {k} |')

print('\nBlocking host calls per step (calls, host time inside the call, innermost range, call site):\n')
blk = defaultdict(lambda: [0, 0.0])
for corr, (start, end, tid, name, cc) in rt.items():
    if re.search(r'cudaStreamSynchronize|cudaDeviceSynchronize|cudaEventSynchronize|^cudaMemcpy(_v\d+)?$', name):
        k = (name, outermost(tid, start), innermost(tid, start), callsite_of.get(cc, ''))
        blk[k][0] += 1
        blk[k][1] += end - start
print('| calls/step | host us/step in call | call | outer range | innermost range | call site |')
print('|---|---|---|---|---|---|')
for (name, o, i, site), (n, t) in sorted(blk.items(), key=lambda x: -x[1][0])[:20]:
    print(f'| {n / args.steps:.1f} | {per(t):.0f} | {name} | {o[:30]} | {i[:50]} | {site[:70]} |')
print('\nNote: time inside a blocking call is CPU waiting; the GPU may be busy during it, so it is not a GPU stall.')
