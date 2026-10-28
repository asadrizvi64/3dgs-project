"""Validate and analyze the Phase 2 step 3 benchmark (stock / A+B / A+B+C, one job per Mip-NeRF360 scene).

    python phase2/analyze_bench.py phase2/logs/gs-bench-*.out [--seeds 0 1 2]

Validation first, per scene. A scene is analyzed only if every run passes:
  - exactly one run block '##### b_<scene>_<arm>_s<seed>' per arm x seed, and no FAILED line for it;
  - one 'SEGCLOCK step=30000' record with a finite train time, N and training-only peak VRAM (clock hook v4);
  - one 'FGS_N iteration=15000' line and one 'FGS_TRAIN_END' line (v4);
  - one PERIMG line, with the same number of test images in every run of the scene;
  - one RESULT line with finite PSNR / SSIM / LPIPS;
  - A+B and A+B+C: exactly one switch line at 15000 and sparse=1 at the end; stock: no switch line and sparse=0.
Runs whose lowest single-image PSNR is far below the scene's other runs (> 2 dB under their median) are flagged as
possible one-view artifacts; they are reported, not removed.

Report: one table per scene (paired deltas against stock per seed), then a summary across scenes and the
pre-registered win criterion from phase2/PLAN.md, for each of A+B and A+B+C:
  median training-time reduction >= 20%, mean dPSNR >= -0.05 dB, no scene below -0.10 dB, mean dLPIPS <= +0.002.
"""
import argparse
import math
import re
import statistics as st
import sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('logs', nargs='+')
ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

ARMS = ('stock', 'ab', 'abc')
LABEL = {'stock': 'stock Faster-GS', 'ab': 'A+B', 'abc': 'A+B+C'}
ORDER = ['bicycle', 'flowers', 'garden', 'stump', 'treehill', 'room', 'counter', 'kitchen', 'bonsai']
fin = lambda x: x is not None and math.isfinite(x)

blocks = defaultdict(list)          # (scene, arm, seed) -> [lines] per header occurrence
jobs = {}
for path in args.logs:
    cur = None
    head = open(path, encoding='utf-8', errors='replace').read()
    mjob = re.search(r'^node=(\S+) job=(\S+)', head, re.M)
    for line in head.splitlines():
        m = re.match(r'##### b_([a-z]+)_(stock|ab|abc)_s(\d+):', line)
        if m:
            cur = (m.group(1), m.group(2), int(m.group(3)))
            blocks[cur].append([])
            if mjob:
                jobs[cur[0]] = f'{mjob.group(2)} ({mjob.group(1)})'
            continue
        if line.startswith('#####'):
            cur = None
        if cur is not None:
            blocks[cur][-1].append(line)

scenes = sorted({k[0] for k in blocks}, key=lambda s: ORDER.index(s) if s in ORDER else 99)
runs, errors, flags = {}, defaultdict(list), defaultdict(list)
for sc in scenes:
    for arm in ARMS:
        for seed in args.seeds:
            key, name = (sc, arm, seed), f'b_{sc}_{arm}_s{seed}'
            err = errors[sc]
            if key not in blocks:
                err.append(f'{name}: no run block'); continue
            if len(blocks[key]) != 1:
                err.append(f'{name}: {len(blocks[key])} run blocks')
            L = blocks[key][-1]
            one = lambda pat: [m for m in (re.match(pat, l) for l in L) if m]
            if any(l.startswith(f'FAILED {name}') for l in L):
                err.append(f'{name}: FAILED line')
            clk = one(r'SEGCLOCK step=30000 \S+ cum_s=(\S+) sparse=(\d)(?: visfrac=(\S+))? N=(\d+) peak_alloc_gib=(\S+)$')
            n15 = one(r'FGS_N iteration=15000 N=(\d+)$')
            end = one(r'FGS_TRAIN_END N=(\d+) peak_alloc_gib=(\S+) peak_reserved_gib=(\S+)$')
            per = one(r'PERIMG \S+ n_images=(\d+) min_image=(\d+) min_psnr=(\S+)$')
            res = one(r'RESULT \S+ PSNR:(\S+) SSIM:(\S+) LPIPS:(\S+) N_Gaussians:(\d+) ms_it=(\S+)')
            sw = [l for l in L if 'selective Adam from iteration' in l]
            bad = False
            for what, lst in (('SEGCLOCK step=30000 (v4)', clk), ('FGS_N iteration=15000', n15), ('FGS_TRAIN_END', end),
                              ('PERIMG', per), ('RESULT', res)):
                if len(lst) != 1:
                    err.append(f'{name}: {len(lst)} {what} lines (expected 1)'); bad = True
            if bad:
                continue
            c, r, e, p = clk[0], res[0], end[0], per[0]
            rec = dict(t=float(c.group(1)), sparse=int(c.group(2)), n15=int(n15[0].group(1)), n_end=int(e.group(1)),
                       peak=float(e.group(2)), peak_res=float(e.group(3)), psnr=float(r.group(1)), ssim=float(r.group(2)),
                       lpips=float(r.group(3)), n=int(r.group(4)), msit=float(r.group(5)),
                       n_img=int(p.group(1)), min_img=int(p.group(2)), min_psnr=float(p.group(3)))
            for k in ('t', 'psnr', 'ssim', 'lpips', 'msit', 'peak', 'min_psnr'):
                if not fin(rec[k]):
                    err.append(f'{name}: non-finite {k}')
            sel = arm != 'stock'
            if sel and (len(sw) != 1 or not sw[0].rstrip().endswith('iteration 15000')):
                err.append(f'{name}: expected one switch at 15000, got {sw}')
            if not sel and sw:
                err.append(f'{name}: stock run has switch lines {sw}')
            if rec['sparse'] != int(sel):
                err.append(f'{name}: final sparse={rec["sparse"]}, expected {int(sel)}')
            runs[key] = rec
    nimg = {runs[k]['n_img'] for k in runs if k[0] == sc}
    if len(nimg) > 1:
        errors[sc].append(f'test image counts differ between runs: {sorted(nimg)}')
    mins = [runs[k]['min_psnr'] for k in runs if k[0] == sc]
    for k in [k for k in runs if k[0] == sc]:
        others = [runs[j]['min_psnr'] for j in runs if j[0] == sc and j != k]
        if others and runs[k]['min_psnr'] < st.median(others) - 2.0:
            flags[sc].append(f'b_{sc}_{k[1]}_s{k[2]}: image {runs[k]["min_img"]} at {runs[k]["min_psnr"]:.2f} dB, '
                             f'other runs\' lowest image median {st.median(others):.2f} dB')

ok = [sc for sc in scenes if not errors[sc]]
print(f'## Validation: {len(ok)}/{len(scenes)} scenes pass ({", ".join(scenes)})')
for sc in scenes:
    for e in errors[sc]:
        print(f'  - {sc}: {e}')
    for f in flags[sc]:
        print(f'  - {sc}: FLAG possible one-view artifact: {f}')
missing = [s for s in ORDER if s not in scenes]
if missing:
    print(f'  - not in these logs: {", ".join(missing)}')
if not ok:
    sys.exit(1)

S = args.seeds
get = lambda sc, arm, k: [runs[(sc, arm, s)][k] for s in S]
dif = lambda sc, arm, k, ref='stock': [a - b for a, b in zip(get(sc, arm, k), get(sc, ref, k))]
red = lambda sc, arm, ref='stock': [100 * (1 - a / b) for a, b in zip(get(sc, arm, 't'), get(sc, ref, 't'))]
f3 = lambda v, f: f'{f % st.mean(v)} ({" / ".join(f % x for x in v)})'

summary = {}
for sc in ok:
    print(f'\n## {sc} (job {jobs.get(sc, "?")}; per seed in brackets; deltas paired against stock)\n')
    print('| arm | train s | time reduction | ms/it | PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | N at 15k | N at 30k, before cleanup | N final | train peak GiB |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for arm in ARMS:
        cells = [LABEL[arm], f3(get(sc, arm, 't'), '%.1f')]
        if arm == 'stock':
            cells += ['', f'{st.mean(get(sc, arm, "msit")):.2f}', f'{st.mean(get(sc, arm, "psnr")):.3f}', '', '', '']
        else:
            cells += [f3(red(sc, arm), '%.1f') + '%', f'{st.mean(get(sc, arm, "msit")):.2f}', f'{st.mean(get(sc, arm, "psnr")):.3f}',
                      f3(dif(sc, arm, 'psnr'), '%+.3f'), f'{st.mean(dif(sc, arm, "ssim")):+.4f}', f'{st.mean(dif(sc, arm, "lpips")):+.4f}']
        cells += [f'{st.mean(get(sc, arm, k)) / 1e6:.3f} M' for k in ('n15', 'n_end', 'n')]
        cells += [f'{st.mean(get(sc, arm, "peak")):.2f}']
        print('| ' + ' | '.join(cells) + ' |')
    print(f'\nC on top of A+B (A+B+C vs A+B, paired): time {f3(red(sc, "abc", "ab"), "%.1f")}% reduction, '
          f'ΔPSNR {f3(dif(sc, "abc", "psnr", "ab"), "%+.3f")}, ΔLPIPS {st.mean(dif(sc, "abc", "lpips", "ab")):+.4f}')
    summary[sc] = {arm: dict(red=st.mean(red(sc, arm)), dpsnr=st.mean(dif(sc, arm, 'psnr')), dssim=st.mean(dif(sc, arm, 'ssim')),
                             dlpips=st.mean(dif(sc, arm, 'lpips')), seeds_faster=sum(x > 0 for x in red(sc, arm)))
                   for arm in ('ab', 'abc')}

print('\n## Across scenes (each scene = mean of its paired seeds)\n')
print('| scene | A+B time | A+B ΔPSNR | A+B ΔLPIPS | A+B+C time | A+B+C ΔPSNR | A+B+C ΔLPIPS |')
print('|---|---|---|---|---|---|---|')
for sc in ok:
    a, b = summary[sc]['ab'], summary[sc]['abc']
    print(f"| {sc} | −{a['red']:.1f}% | {a['dpsnr']:+.3f} | {a['dlpips']:+.4f} | −{b['red']:.1f}% | {b['dpsnr']:+.3f} | {b['dlpips']:+.4f} |")
print()
for arm in ('ab', 'abc'):
    reds = [summary[sc][arm]['red'] for sc in ok]
    dps = [summary[sc][arm]['dpsnr'] for sc in ok]
    dls = [summary[sc][arm]['dlpips'] for sc in ok]
    worst = min(ok, key=lambda sc: summary[sc][arm]['dpsnr'])
    crit = {'median time reduction >= 20%': st.median(reds) >= 20, 'mean dPSNR >= -0.05 dB': st.mean(dps) >= -0.05,
            'no scene dPSNR < -0.10 dB': min(dps) >= -0.10, 'mean dLPIPS <= +0.002': st.mean(dls) <= 0.002}
    print(f'**{LABEL[arm]}** over {len(ok)} scenes: time reduction median {st.median(reds):.1f}%, mean {st.mean(reds):.1f}%, '
          f'range {min(reds):.1f}–{max(reds):.1f}%; ΔPSNR mean {st.mean(dps):+.3f}, worst {worst} {min(dps):+.3f}; '
          f'ΔLPIPS mean {st.mean(dls):+.4f}; paired runs faster: {sum(summary[sc][arm]["seeds_faster"] for sc in ok)}/{len(ok) * len(S)}')
    for k, v in crit.items():
        print(f'  - {"PASS" if v else "FAIL"} {k}')
    if len(ok) < len(ORDER):
        print(f'  - (criterion is defined over all {len(ORDER)} scenes; only {len(ok)} analyzed)')
