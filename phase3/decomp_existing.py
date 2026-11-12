"""Descriptive, post hoc: split each arm's time-to-target saving into fewer steps and cheaper steps, from the completed
Experiment 1R and 1A logs. No new training. Not pre-registered; not a decision rule.

    python phase3/decomp_existing.py

From the branch (step 5000) to the first crossing of the target (ctl's PSNR at 30000 minus 0.05 dB, as in 1R/1A):
    T' = training-step seconds (SEGCLOCK), S' = steps, c = T'/S' = mean seconds per step over that window.
The crossing step is interpolated linearly between evaluation points, in the same way as the crossing time, so
    ln(T'_ctl / T'_arm) = ln(S'_ctl / S'_arm) + ln(c_ctl / c_arm)     (exact)
step share = first term / total. This is an accounting identity. It says *where* the time went (fewer steps or
cheaper steps), not *why*: the number of Gaussians and the policy can each affect both factors.
"""
import math
import os
import re
import statistics as st
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
BR, END, DELTA = 5000, 30000, 0.05
LOG = 'phase2/logs'
SETS = {
    '1R': (['gs-x1r-garden-4559237.out', 'gs-x1r-room-s10-4559238.out', 'gs-x1r-room-s11-4559239.out',
            'gs-x1r-room-s12-4559240.out'], ('stop', 'mcmc', 'oracle')),
    '1A': (['gs-x1a-garden-s30-4561623.out', 'gs-x1a-garden-s31-4561625.out', 'gs-x1a-garden-s32-4561627.out',
            'gs-x1a-room-s30-4561624.out', 'gs-x1a-room-s31-4561626.out', 'gs-x1a-room-s32-4561628.out'],
           ('mcmc', 'mrep', 'mreloc', 'mnoise', 'mreg', 'mlr', 'creset')),
}


def blocks_of(path):
    out, arch = {}, None
    text = open(path, encoding='utf-8', errors='replace').read()
    job = re.search(r'^node=\S+ job=(\S+)', text, re.M).group(1)
    arch = os.path.join(os.path.dirname(path), f'run_{job}', 'runs')
    cur = None
    for line in text.splitlines():
        m = re.match(r'##### x_([a-z]+)_([a-z]+)_s(\d+):', line)
        if m:
            cur = (m.group(1), m.group(2), int(m.group(3)))
            out[cur] = []
            continue
        if line.startswith('#####'):
            cur = None
        if cur is not None:
            out[cur].append(line)
    return out, arch


def psnr(path):
    return float(re.search(r'^PSNR\t(\S+)', open(path).read(), re.M).group(1))


def load(name, lines, arch):
    clk = {}
    for l in lines:
        m = re.match(r'SEGCLOCK step=(\d+) \S+ cum_s=(\S+) .*N=(\d+) peak', l)
        if m:
            clk[int(m.group(1))] = (float(m.group(2)), int(m.group(3)))
        m = re.match(r'FGS_RESUME step=(\d+) N=(\d+) cum_s=(\S+)', l)
        if m:
            clk[int(m.group(1))] = (float(m.group(3)), int(m.group(2)))
    steps = list(range(BR, END + 1, 1000))
    if any(s not in clk for s in steps):
        return None
    curve = {s: psnr(os.path.join(arch, name, f'test_{s}', 'metrics_8bit.txt')) for s in steps}
    return dict(clk=clk, curve=curve)


def cross(r, tgt):
    """(T', S', mean N over the window) at the first crossing, or None."""
    pts = sorted(r['curve'])
    t0 = r['clk'][BR][0]
    for i, s in enumerate(pts):
        if r['curve'][s] >= tgt:
            if i == 0:
                return None   # at the branch already: no window
            a = pts[i - 1]
            pa, pb = r['curve'][a], r['curve'][s]
            w = (tgt - pa) / (pb - pa) if pb != pa else 1.0
            step = a + w * (s - a)
            t = r['clk'][a][0] + w * (r['clk'][s][0] - r['clk'][a][0])
            ns = [r['clk'][k][1] for k in pts if k <= s]
            return t - t0, step - BR, st.mean(ns)
    return None


rows = defaultdict(list)   # (exp, scene, arm) -> per-seed dicts
for exp, (files, arms) in SETS.items():
    runs = {}
    for f in files:
        b, arch = blocks_of(os.path.join(LOG, f))
        for (sc, arm, sd), L in b.items():
            if arm != 'pfx':
                runs[(sc, arm, sd)] = load(f'x_{sc}_{arm}_s{sd}', L, arch)
    for (sc, arm, sd), r in sorted(runs.items()):
        if arm not in arms or r is None or runs.get((sc, 'ctl', sd)) is None:
            continue
        c = runs[(sc, 'ctl', sd)]
        tgt = c['curve'][END] - DELTA
        xc, xa = cross(c, tgt), cross(r, tgt)
        d = dict(seed=sd, ctl=xc, arm=xa)
        if xc and xa:
            lt, ls, lc = math.log(xc[0] / xa[0]), math.log(xc[1] / xa[1]), math.log((xc[0] / xc[1]) / (xa[0] / xa[1]))
            d.update(lt=lt, ls=ls, lc=lc, share=ls / lt if abs(lt) > 1e-9 else float('nan'))
        rows[(exp, sc, arm)].append(d)

print('# Steps vs cost per step, from the completed 1R and 1A runs (descriptive, post hoc)\n')
print('Window: branch (5000) to the first crossing of ctl\'s final PSNR − 0.05 dB. T\' in training-step seconds, S\' in')
print('steps, c = T\'/S\' in ms per step, N̄ = mean Gaussian count at the evaluation points in the window (millions).')
print('Step share = ln(S\'ctl/S\'arm) / ln(T\'ctl/T\'arm); cost share = 1 − step share. "–" = arm never reached the target.\n')
print('| exp | scene | arm | seed | T\' ctl / arm (s) | S\' ctl / arm | c ctl / arm (ms) | N̄ ctl / arm (M) | time saved | step share |')
print('|---|---|---|---|---|---|---|---|---|---|')
summ = []
for (exp, sc, arm), ds in sorted(rows.items(), key=lambda kv: (kv[0][0] != '1R', kv[0][1], SETS[kv[0][0]][1].index(kv[0][2]))):
    shares, saves = [], []
    for d in ds:
        xc, xa = d['ctl'], d['arm']
        if not (xc and xa):
            print(f'| {exp} | {sc} | {arm} | {d["seed"]} | {xc[0]:.1f} / – | {xc[1]:.0f} / – | – | – | – | – |' if xc else
                  f'| {exp} | {sc} | {arm} | {d["seed"]} | ctl not crossed | | | | | |')
            continue
        save = 1 - xa[0] / xc[0]
        saves.append(save)
        shares.append(d['share'])
        print(f'| {exp} | {sc} | {arm} | {d["seed"]} | {xc[0]:.1f} / {xa[0]:.1f} | {xc[1]:.0f} / {xa[1]:.0f} | '
              f'{1e3 * xc[0] / xc[1]:.2f} / {1e3 * xa[0] / xa[1]:.2f} | {xc[2] / 1e6:.2f} / {xa[2] / 1e6:.2f} | '
              f'{100 * save:+.1f}% | {d["share"]:.2f} |')
    summ.append((exp, sc, arm, len(ds), len(shares), saves, shares, ds))

print('\n## Summary per scene and arm\n')
print('| exp | scene | arm | pairs used | median time saved | median step share | step-share range | median c ratio arm/ctl | median S\' ratio arm/ctl |')
print('|---|---|---|---|---|---|---|---|---|')
for exp, sc, arm, n, k, saves, shares, ds in summ:
    if not k:
        print(f'| {exp} | {sc} | {arm} | 0 of {n} | | | | | |')
        continue
    cr = [math.exp(-d['lc']) for d in ds if 'lc' in d]
    sr = [math.exp(-d['ls']) for d in ds if 'ls' in d]
    print(f'| {exp} | {sc} | {arm} | {k} of {n} | {100 * st.median(saves):+.1f}% | {st.median(shares):.2f} | '
          f'{min(shares):.2f} – {max(shares):.2f} | {st.median(cr):.2f} | {st.median(sr):.2f} |')
