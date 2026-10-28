"""Attribute GPU idle gaps and blocking host calls in an nsys capture to the PyTorch ops / Python code that caused them.

    python3 phase2/host_gaps.py <capture>.sqlite [--steps 50]

For captures made with  nsys profile --pytorch=autograd-nvtx --python-backtrace=cuda  (phase2/build_plus.sbatch,
STAGES=hostprof). Each GPU op (kernel, memcpy, memset) is joined to the CUDA runtime call that launched it
(correlationId). A gap on the GPU before an op is charged to that op's launch: the innermost NVTX range open on
the launching thread at launch time (a PyTorch op such as aten::copy_, or the trainer's fwd/bwd/opt range), and the
Python call site if backtraces were captured. Blocking calls (cudaStreamSynchronize, cudaDeviceSynchronize,
cudaMemcpy) are listed with their ranges and call sites. Times are per training step (total / --steps).
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
args = ap.parse_args()
c = sqlite3.connect(args.db)
S = dict(c.execute('select id, value from StringIds'))
tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}

# CUDA runtime calls by correlation id, with their thread and start time
rt = {}
for start, end, tid, corr, nid, cc in c.execute(
        'select start, end, globalTid, correlationId, nameId, callchainId from CUPTI_ACTIVITY_KIND_RUNTIME'):
    rt[corr] = (start, end, tid, S.get(nid, '?'), cc)

# NVTX push/pop ranges per thread, for innermost-range lookup
ranges = defaultdict(list)
for start, end, text, tid, tidx in c.execute('select start, end, text, globalTid, textId from NVTX_EVENTS where end is not null'):
    t = text if text is not None else S.get(tidx, '')
    t = re.sub(r', (seq|op_id|sizes|input_op_ids) = .*', '', t or '')
    ranges[tid].append((start, end, t))
for tid in ranges:
    ranges[tid].sort()
starts = {tid: [r[0] for r in rs] for tid, rs in ranges.items()}


def enclosing(tid, t):
    """Innermost and outermost NVTX range on thread tid open at time t."""
    rs = ranges.get(tid, [])
    i = bisect.bisect_right(starts.get(tid, []), t)
    inner = outer = None
    for s, e, txt in reversed(rs[max(0, i - 400):i]):
        if s <= t <= e:
            if inner is None:
                inner = txt
            outer = txt
    return inner or '(no range)', outer or '(no range)'


# Python backtraces, if the capture has them: find a table that maps callchainId -> frames
def callsite(cc):
    return ''


for t in tables:
    if 'PYTHON' in t.upper() and 'CALLCHAIN' in t.upper() or t == 'PYTHON_CALLCHAINS':
        cols = [r[1] for r in c.execute(f'pragma table_info({t})')]
        print(f'(python backtrace table {t}: {cols})')

ops = []
for start, end, corr, name in c.execute('select start, end, correlationId, coalesce(demangledName, shortName) from CUPTI_ACTIVITY_KIND_KERNEL'):
    ops.append((start, end, corr, 'K:' + re.sub(r'\(.*', '', S.get(name, str(name))).replace('void ', '').split('::')[-1][:50]))
for start, end, corr, kind in c.execute('select start, end, correlationId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY'):
    ops.append((start, end, corr, f'memcpy kind {kind}'))
for start, end, corr in c.execute('select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMSET'):
    ops.append((start, end, corr, 'memset'))
ops.sort()

gap_by = defaultdict(lambda: [0.0, 0])
total_gap = 0.0
cur_end = None
for start, end, corr, name in ops:
    if cur_end is not None and start > cur_end:
        g = start - cur_end
        if g >= args.min_gap_us * 1e3:
            total_gap += g
            launch = rt.get(corr)
            inner, outer = enclosing(launch[2], launch[0]) if launch else ('?', '?')
            key = (outer, inner, name)
            gap_by[key][0] += g
            gap_by[key][1] += 1
    cur_end = end if cur_end is None else max(cur_end, end)
span = ops[-1][1] - ops[0][0]
print(f'GPU span {span / args.steps / 1e6:.3f} ms/step; idle gaps >= {args.min_gap_us:g} us: {total_gap / args.steps / 1e6:.3f} ms/step\n')
print('| idle us/step | gaps/step | outer range | innermost range at launch | GPU op that waited |')
print('|---|---|---|---|---|')
for (outer, inner, name), (g, n) in sorted(gap_by.items(), key=lambda x: -x[1][0])[:25]:
    print(f'| {g / args.steps / 1e3:.1f} | {n / args.steps:.1f} | {outer[:40]} | {inner[:60]} | {name} |')

print('\nBlocking host calls per step (by innermost range):\n')
blk = defaultdict(lambda: [0, 0.0])
for corr, (start, end, tid, name, cc) in rt.items():
    if re.search(r'cudaStreamSynchronize|cudaDeviceSynchronize|cudaMemcpy(_|$|v2$)|cudaMemcpy$|cudaEventSynchronize', name):
        inner, outer = enclosing(tid, start)
        blk[(name, outer, inner)][0] += 1
        blk[(name, outer, inner)][1] += end - start
print('| calls/step | host wait us/step | call | outer range | innermost range |')
print('|---|---|---|---|---|')
for (name, outer, inner), (n, t) in sorted(blk.items(), key=lambda x: -x[1][1])[:20]:
    print(f'| {n / args.steps:.1f} | {t / args.steps / 1e3:.1f} | {name} | {outer[:40]} | {inner[:60]} |')
