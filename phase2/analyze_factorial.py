"""Validate and analyze the Phase 2 step 2 factorial (A x B x C on/off, several seeds) from the job log.

    python phase2/analyze_factorial.py phase2/logs/gs-phase2-factorial-<job>.out [--seeds 0 1 2]

Validation comes first; nothing is analyzed unless every check passes ('JOB OK' alone is not enough, because the
launcher tolerates trainer errors and only checks that final metrics exist). Required for every arm x seed:
  - exactly one run block '##### f_<arm>_s<seed>', and no 'FAILED' line for it;
  - exactly one 'SEGCLOCK step=30000' record (the training clock), with a finite cum_s > 0;
  - exactly one RESULT line with finite PSNR / SSIM / LPIPS, a Gaussian count and ms/it;
  - B on : exactly one 'selective Adam from iteration 15000' line, and the final SEGCLOCK says sparse=1;
    B off: no switch line, and the final SEGCLOCK says sparse=0.

Reported:
  - per arm: train time (CUDA events), ms/it, quality, paired dPSNR vs all-off, Gaussians, allocated VRAM;
  - standalone gains: each single-flag arm vs all-off, paired per seed (what one flag alone buys);
  - incremental gains inside the stack (e.g. C given A and B on): a1b1c1 vs a1b1c0 etc.;
  - factorial main effects and interactions (2^3 contrasts per seed, mean [min..max] over seeds). A main effect
    averages a flag's on/off difference over all settings of the other two, so it differs from the standalone gain
    when flags overlap (B and C both cut optimizer memory traffic).
The VRAM figure is the trainer's own 'allocated' line, read after the final test evaluation, so it may include
the final render; it is not a training-only peak.
"""
import argparse
import itertools
import math
import re
import statistics as st
import sys

ap = argparse.ArgumentParser()
ap.add_argument('log')
ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
args = ap.parse_args()

arms = [f'a{a}b{b}c{c}' for a, b, c in itertools.product((0, 1), repeat=3)]
text = open(args.log, encoding='utf-8', errors='replace').read()

blocks = {}   # (arm, seed) -> list of lines (one list per header occurrence)
order = []
cur = None
for line in text.splitlines():
    m = re.match(r'##### f_(a[01]b[01]c[01])_s(\d+):', line)
    if m:
        cur = (m.group(1), int(m.group(2)))
        blocks.setdefault(cur, []).append([])
        order.append(cur)
        continue
    if line.startswith('#####'):   # any other section header ends the current block
        cur = None
    if cur is not None:
        blocks[cur][-1].append(line)

errors = []
runs = {}
fin = lambda x: x is not None and math.isfinite(x)
for seed in args.seeds:
    for arm in arms:
        key = (arm, seed)
        name = f'f_{arm}_s{seed}'
        if key not in blocks:
            errors.append(f'{name}: no run block')
            continue
        if len(blocks[key]) != 1:
            errors.append(f'{name}: {len(blocks[key])} run blocks (expected 1)')
        lines = blocks[key][-1]
        if any(l.startswith(f'FAILED {name}') for l in lines):
            errors.append(f'{name}: FAILED line')
        clocks = [re.search(r'cum_s=([\d.eE+-]+|nan|inf) sparse=(\d)', l) for l in lines if l.startswith('SEGCLOCK step=30000 ')]
        switches = [l for l in lines if 'selective Adam from iteration' in l]
        results = [re.match(r'RESULT \S+ PSNR:(\S+) SSIM:(\S+) LPIPS:(\S+) N_Gaussians:(\d+) ms_it=(\S+)(?:\s+allocated: ([\d.]+) GiB)?', l)
                   for l in lines if l.startswith('RESULT ')]
        if len(clocks) != 1 or clocks[0] is None:
            errors.append(f'{name}: {len(clocks)} SEGCLOCK step=30000 records (expected 1)')
            continue
        if len(results) != 1 or results[0] is None:
            errors.append(f'{name}: {len(results)} RESULT lines (expected 1, parseable)')
            continue
        t, sparse = float(clocks[0].group(1)), int(clocks[0].group(2))
        r = results[0]
        rec = dict(t=t, psnr=float(r.group(1)), ssim=float(r.group(2)), lpips=float(r.group(3)), n=int(r.group(4)),
                   msit=float(r.group(5)), vram=float(r.group(6)) if r.group(6) else float('nan'))
        if not (fin(t) and t > 0):
            errors.append(f'{name}: bad train time {t}')
        for k in ('psnr', 'ssim', 'lpips', 'msit'):
            if not fin(rec[k]):
                errors.append(f'{name}: non-finite {k}')
        b_on = arm[3] == '1'
        if b_on and (len(switches) != 1 or not switches[0].rstrip().endswith('iteration 15000')):
            errors.append(f'{name}: B on but switch lines = {switches}')
        if not b_on and switches:
            errors.append(f'{name}: B off but switch lines = {switches}')
        if sparse != int(b_on):
            errors.append(f'{name}: final optimizer state sparse={sparse}, expected {int(b_on)}')
        runs[key] = rec

extra = sorted(set(blocks) - {(a, s) for a in arms for s in args.seeds})
if extra:
    errors.append(f'unexpected run blocks: {extra}')
print(f'## Validation: {len(runs)}/{len(arms) * len(args.seeds)} arm-seed runs complete; '
      f'{"all checks passed" if not errors else f"{len(errors)} problem(s)"}')
for e in errors:
    print('  -', e)
if errors:
    sys.exit(1)
print('Run order:', ' '.join(f'{a}_s{s}' for a, s in order))
seeds = args.seeds


def paired(arm, ref, key):
    return [runs[(arm, s)][key] - runs[(ref, s)][key] for s in seeds]


def pct(arm, ref):
    return [100 * (runs[(arm, s)]['t'] / runs[(ref, s)]['t'] - 1) for s in seeds]


fmt3 = lambda v, f: f'{f % st.mean(v)} ({" / ".join(f % x for x in v)})'

print('\n## Per arm (mean over seeds; per seed in brackets)\n')
print('| arm | A | B | C | train s | vs all-off (paired) | ms/it | PSNR | dPSNR vs all-off (paired) | SSIM | LPIPS | Gaussians | allocated GiB* |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for arm in arms:
    rs = [runs[(arm, s)] for s in seeds]
    on = lambda i: 'on' if arm[i] == '1' else ''
    print(f"| {arm} | {on(1)} | {on(3)} | {on(5)} | {fmt3([r['t'] for r in rs], '%.1f')} | "
          f"{fmt3(pct(arm, 'a0b0c0'), '%+.1f')}% | {st.mean(r['msit'] for r in rs):.2f} | "
          f"{st.mean(r['psnr'] for r in rs):.3f} | {fmt3(paired(arm, 'a0b0c0', 'psnr'), '%+.3f')} | "
          f"{st.mean(r['ssim'] for r in rs):.4f} | {st.mean(r['lpips'] for r in rs):.4f} | "
          f"{st.mean(r['n'] for r in rs):,.0f} | {st.mean(r['vram'] for r in rs):.2f} |")
print('\n*allocated: the trainer\'s VRAM line, read after the final test evaluation (not a training-only peak).')

print('\n## Standalone and incremental gains (paired per seed: mean (s0 / s1 / s2))\n')
print('| comparison | train time | dPSNR | dSSIM | dLPIPS |')
print('|---|---|---|---|---|')
comps = [('A alone', 'a1b0c0', 'a0b0c0'), ('B alone', 'a0b1c0', 'a0b0c0'), ('C alone', 'a0b0c1', 'a0b0c0'),
         ('C given B', 'a0b1c1', 'a0b1c0'), ('C given A+B', 'a1b1c1', 'a1b1c0'),
         ('B given A+C', 'a1b1c1', 'a1b0c1'), ('A given B+C', 'a1b1c1', 'a0b1c1'),
         ('full stack A+B+C', 'a1b1c1', 'a0b0c0')]
for label, arm, ref in comps:
    print(f"| {label} ({arm} vs {ref}) | {fmt3(pct(arm, ref), '%+.1f')}% | {fmt3(paired(arm, ref, 'psnr'), '%+.3f')} | "
          f"{fmt3(paired(arm, ref, 'ssim'), '%+.4f')} | {fmt3(paired(arm, ref, 'lpips'), '%+.4f')} |")
base_t = st.mean(runs[('a0b0c0', s)]['t'] for s in seeds)
full_t = st.mean(runs[('a1b1c1', s)]['t'] for s in seeds)
add_s = sum(st.mean(paired(a, 'a0b0c0', 't')) for a in ('a1b0c0', 'a0b1c0', 'a0b0c1'))
mult = math.prod(st.mean(runs[(a, s)]['t'] for s in seeds) / base_t for a in ('a1b0c0', 'a0b1c0', 'a0b0c1'))
print(f'\nFull stack {full_t:.1f} s vs all-off {base_t:.1f} s ({100 * (full_t / base_t - 1):+.1f}%). '
      f'Adding the standalone savings in seconds predicts {base_t + add_s:.1f} s ({100 * add_s / base_t:+.1f}%); '
      f'multiplying the standalone ratios predicts {base_t * mult:.1f} s ({100 * (mult - 1):+.1f}%).')


def contrasts(val):
    """2^3 effects for one seed: mean over the 4 'on' arms minus mean over the 4 'off' arms (sign products for interactions)."""
    out = {}
    for name, idx in [('A', (1,)), ('B', (3,)), ('C', (5,)), ('AB', (1, 3)), ('AC', (1, 5)), ('BC', (3, 5)), ('ABC', (1, 3, 5))]:
        tot = 0.0
        for arm in arms:
            sign = 1
            for i in idx:
                sign *= 1 if arm[i] == '1' else -1
            tot += sign * val[arm]
        out[name] = tot / 4
    return out


print('\n## Factorial effects (2^3 contrasts per seed; mean [min..max] over seeds)\n')
print('Main effects average over the other two flags, so they are not the standalone gains above.\n')
print('| effect | train time, s | train time, % (log-ratio) | PSNR, dB | LPIPS |')
print('|---|---|---|---|---|')
per = {k: [contrasts({a: f(runs[(a, s)]) for a in arms}) for s in seeds]
       for k, f in [('t', lambda r: r['t']), ('logt', lambda r: math.log(r['t'])),
                    ('psnr', lambda r: r['psnr']), ('lpips', lambda r: r['lpips'])]}
for name in ('A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC'):
    def cell(key, f, scale=1.0):
        v = [scale * d[name] for d in per[key]]
        return f'{f % st.mean(v)} [{f % min(v)}..{f % max(v)}]'
    print(f"| {name} | {cell('t', '%+.1f')} | {cell('logt', '%+.1f', 100.0)} | {cell('psnr', '%+.3f')} | {cell('lpips', '%+.4f')} |")
print('\nInteraction terms near 0 mean the savings add (in seconds, or in % for the log column); '
      'a positive BC time term means B and C overlap.')
