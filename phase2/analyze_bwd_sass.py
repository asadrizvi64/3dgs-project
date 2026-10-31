"""Where the macro blend backward's instructions go: per-SASS-instruction counts from Nsight Compute, grouped by region.

    ncu --import u_<scene>_abc_20000.ncu-rep --page source --csv --print-source sass \
        --kernel-name regex:blend_backward_macro --launch-count 1 > <scene>.bwd_sass.csv
    python phase2/analyze_bwd_sass.py phase2/logs/run_<job>/<scene>.bwd_sass.csv [...] [--md out.md]

The export repeats the kernel's SASS once per captured launch; only the first copy is used (one launch). Regions follow
the SASS of blend_backward_macro_cu (mode 3: macro + subtile skip, plain shfl_down reduction):
  staging     everything before the splat loop: bucket lookup, splat staging, per-pixel checkpoint loads (once per bucket)
  loop head   per splat step: shared-memory reads of the splat and its flags, row predicates
  row check   per row (6 unrolled rows): the subtile / lim branch around the row body
  evaluate    per row: delta, exponent, expf, alpha cut-off (warp executes it when any lane evaluates the pixel)
  gradient    per row: the contributing-pixel arithmetic (any lane above the cut-off)
  vote        per step: ballot of `used` (skip the reduction when no lane contributed)
  reduce      per step with a contribution: 9 values x 5 shfl_down + adds, and the 9 global atomics (lane 0)
  loop tail   per step: loop counter and branch
Region boundaries: the step count is the count of the VOTE.ANY ballot (once per splat step); the loop starts at the first
instruction with that count. In the row blocks, instructions with their row's MUFU.EX2 count are 'evaluate', with its
MUFU.RCP count 'gradient', the rest 'row check'. The reduce region runs from VOTE.ANY's branch to the loop tail.
Stall samples are ncu's warp-state samples (all samples). CHECK fails if the regions don't add up to the total.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


def load(path):
    rows = list(csv.reader(open(path, newline='')))
    hi = next(i for i, r in enumerate(rows) if 'Address' in r and 'Source' in r)
    h = rows[hi]
    data = [r for r in rows[hi + 1:] if len(r) == len(h) and r[h.index('Address')].startswith('0x')]
    num = lambda v: int(float(v.replace(',', '') or 0))
    ins = [(r[h.index('Address')], r[h.index('Source')].strip(), num(r[h.index('Instructions Executed')]),
            num(r[h.index('Warp Stall Sampling (All Samples)')])) for r in data]
    # keep the first launch: addresses repeat for the second
    seen, first = set(), []
    for x in ins:
        if x[0] in seen:
            break
        seen.add(x[0]); first.append(x)
    return first


def op(src):
    t = [w for w in src.split() if not w.startswith('@')]
    return t[0] if t else ''


def classify(ins):
    counts = [c for _, _, c, _ in ins]
    ops = [op(s) for _, s, _, _ in ins]
    buckets = counts[0]
    vote = next(i for i in range(len(ins)) if ops[i].startswith('VOTE') and counts[i] > 10 * buckets)
    steps = counts[vote]   # the ballot of `used` runs exactly once per splat step
    start = next(i for i, c in enumerate(counts) if c == steps)
    red_start = next(i for i in range(vote, len(ins)) if ops[i].startswith('BRA')) + 1
    tail = next(i for i in range(red_start, len(ins)) if counts[i] == steps)
    end = next(i for i in range(tail, len(ins)) if ops[i] == 'EXIT')
    first_bssy = next(i for i in range(start, vote) if ops[i].startswith('BSSY'))
    reg = ['staging'] * len(ins)
    # first row's branch is folded into the loop head: the head ends at the first instruction executed less than once per step
    head_end = next(i for i in range(first_bssy, vote) if counts[i] < steps)
    for i in range(start, head_end):
        reg[i] = 'loop head'
    ex2 = {counts[i] for i in range(head_end, vote) if ops[i].startswith('MUFU.EX2')}
    rcp = {counts[i] for i in range(head_end, vote) if ops[i].startswith('MUFU.RCP')}
    for i in range(head_end, vote):
        reg[i] = 'evaluate' if counts[i] in ex2 else 'gradient' if counts[i] in rcp else 'row check'
    for i in range(vote, red_start):
        reg[i] = 'vote'
    for i in range(red_start, tail):
        reg[i] = 'reduce'
    for i in range(tail, end + 1):
        reg[i] = 'loop tail'
    return reg, buckets, steps


def main(argv):
    md = None
    if '--md' in argv:
        i = argv.index('--md'); md = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    lines, ok = [], True
    order = ['staging', 'loop head', 'row check', 'evaluate', 'gradient', 'vote', 'reduce', 'loop tail']
    for path in argv:
        ins = load(path)
        reg, buckets, steps = classify(ins)
        tot = sum(c for _, _, c, _ in ins); tot_s = sum(s for *_, s in ins)
        agg = {r: [0, 0, 0] for r in order}   # instructions executed, static instructions, stall samples
        for (a, s, c, smp), r in zip(ins, reg):
            agg[r][0] += c; agg[r][1] += 1; agg[r][2] += smp
        reduce_runs = max((c for (a, s, c, smp), r in zip(ins, reg) if r == 'reduce' and op(s).startswith('SHFL')), default=0)
        eval_rows = sum(c for (a, s, c, smp), r in zip(ins, reg) if r == 'evaluate' and op(s).startswith('MUFU.EX2'))
        grad_rows = sum(c for (a, s, c, smp), r in zip(ins, reg) if r == 'gradient' and op(s).startswith('MUFU.RCP'))
        lines.append(f'### {Path(path).name}\n')
        lines.append(f'One launch: {tot / 1e6:.1f}M warp instructions; {buckets:,} launched buckets, {steps:,} splat steps '
                     f'(reduction run on {reduce_runs:,} = {100 * reduce_runs / steps:.1f}%), {eval_rows:,} warp row evaluations, '
                     f'{grad_rows:,} warp row gradients.\n')
        lines.append('| region | instructions executed | share | static instructions | per splat step | stall samples share |')
        lines.append('|---|---|---|---|---|---|')
        for r in order:
            c, n, smp = agg[r]
            lines.append(f'| {r} | {c / 1e6:.1f}M | {100 * c / tot:.1f}% | {n} | {c / steps:.1f} | {100 * smp / tot_s:.1f}% |')
        lines.append(f'| **total** | {tot / 1e6:.1f}M | | {len(ins)} | {tot / steps:.1f} | |')
        lines.append('')
        if abs(sum(v[0] for v in agg.values()) - tot) > 0:
            ok = False
    lines.append('CHECK ' + ('PASSED: regions add up to the total' if ok else 'FAILED'))
    text = '\n'.join(lines)
    print(text)
    if md:
        Path(md).write_text(text + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
