"""Experiment 1R noise pilot (phase3/EXP1R_PLAN.md, "Check 1"): same-seed numerical noise of uninterrupted runs, and the
number of seeds check 1 needs.

    python phase3/noise_pilot.py phase2/logs/gs-x1r-pilot-<job>.out [--margin-psnr 0.05] [--margin-n 0.01]

Each pilot seed has two identical uninterrupted FasterGSUp runs, b_<scene>_up_s<seed> and b_<scene>_up2_s<seed>
(build_plus.sbatch stage noisepilot). They differ only through GPU non-determinism (the order of atomic additions),
which is all that should separate a branched ctl run from its reference once evaluation is neutral (EXP1R check 1a).

Reported per scene:
- whether each pair saw identical training views (view traces);
- the same-seed differences d = up2 - up of full-precision final PSNR and final N, their SD, and the largest per-view
  PSNR difference;
- the TOST power of check 1 for 3..12 seeds per scene, at true difference 0, with the SD estimated here.
  Check 1 passes when the 90% t interval of the mean paired (ctl - up) difference lies within +-margin, for PSNR and
  for N (relative to mean N). The sample-size rule is fixed in EXP1R_PLAN.md: the smallest n with >= 80% power on
  both measures and both scenes.
"""
import argparse
import os
import re
import statistics as st
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('logs', nargs='+')
ap.add_argument('--margin-psnr', type=float, default=0.05)
ap.add_argument('--margin-n', type=float, default=0.01, help='relative to mean final N')
ap.add_argument('--sims', type=int, default=20000)
args = ap.parse_args()
T95 = {2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796}  # one-sided 0.95, df

res, per, views = {}, {}, {}
for p in args.logs:
    text = open(p, encoding='utf-8', errors='replace').read()
    job = re.search(r'^node=\S+ job=(\S+)', text, re.M).group(1)
    base = os.path.dirname(p)
    for m in re.finditer(r'^RESULT b_([a-z]+)_(up2?)_s(\d+) PSNR:(\S+) .*?N_Gaussians:(\d+)', text, re.M):
        sc, r, s = m.group(1), m.group(2), int(m.group(3))
        res[(sc, r, s)] = (float(m.group(4)), int(m.group(5)))
        f = os.path.join(base, f'run_{job}', 'runs', f'b_{sc}_{r}_s{s}', 'test_30000', 'metrics_8bit.txt')
        if os.path.exists(f):
            t = open(f).read().partition('Image')[2]
            per[(sc, r, s)] = {int(a): float(b) for a, b in re.findall(r'^(\d+)\t([\d.]+)', t, re.M)}
        v = os.path.join(base, f'run_{job}', f'b_{sc}_{r}_s{s}.views')
        if os.path.exists(v):
            views[(sc, r, s)] = open(v).read()

rng = np.random.default_rng(0)
need = {}
for sc in sorted({k[0] for k in res}):
    seeds = sorted(s for (c, r, s) in res if c == sc and r == 'up' and (sc, 'up2', s) in res)
    print(f'## {sc}: {len(seeds)} same-seed pairs (seeds {" ".join(map(str, seeds))})\n')
    if len(seeds) < 2:
        print('not enough pairs\n')
        continue
    print('| seed | views identical | d PSNR (dB) | d N | largest per-view d (dB, view) |\n|---|---|---|---|---|')
    dp, dn, nn = [], [], []
    for s in seeds:
        a, b = res[(sc, 'up', s)], res[(sc, 'up2', s)]
        dp.append(b[0] - a[0]); dn.append(b[1] - a[1]); nn.append(a[1])
        va, vb = views.get((sc, 'up', s)), views.get((sc, 'up2', s))
        same = 'n/a' if va is None or vb is None else ('yes' if va == vb else 'NO')
        pv = ''
        if (sc, 'up', s) in per and (sc, 'up2', s) in per:
            dd = {i: per[(sc, 'up2', s)][i] - per[(sc, 'up', s)][i] for i in per[(sc, 'up', s)]}
            w = max(dd, key=lambda i: abs(dd[i]))
            pv = f'{dd[w]:+.2f} (view {w})'
        print(f'| {s} | {same} | {dp[-1]:+.4f} | {dn[-1]:+d} | {pv} |')
    sdp, sdn = st.stdev(dp), st.stdev(dn) / st.mean(nn)
    print(f'\nSD of same-seed differences: PSNR {sdp:.4f} dB, N {sdn * 100:.3f}% of mean N (from {len(seeds)} pairs; '
          f'an SD from this few pairs is imprecise)')
    print(f'\nTOST power of check 1 at true difference 0 (margins: PSNR {args.margin_psnr} dB, N {args.margin_n * 100:.1f}%):\n')
    print('| seeds per scene | power PSNR | power N |\n|---|---|---|')
    need[sc] = None
    for n in range(3, 13):
        pw = []
        for sd, mg in ((sdp, args.margin_psnr), (sdn, args.margin_n)):
            x = rng.normal(0.0, sd, size=(args.sims, n))
            m, s = x.mean(1), x.std(1, ddof=1)
            pw.append(float(np.mean(np.abs(m) + T95[n - 1] * s / np.sqrt(n) < mg)))
        print(f'| {n} | {pw[0]:.2f} | {pw[1]:.2f} |')
        if need[sc] is None and min(pw) >= 0.80:
            need[sc] = n
    print(f'\nsmallest n with >= 80% power on both: {need[sc] if need[sc] else "more than 12"}\n')
if need:
    ns = [v for v in need.values()]
    print('Seeds per scene for check 1, by the rule in EXP1R_PLAN.md: '
          + ('more than 12 on at least one scene: check 1 cannot be made conclusive at these margins; the plan returns '
             'to review before launch' if None in ns else str(max(ns))))
