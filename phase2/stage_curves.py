"""Test PSNR vs training time for the stage-render runs (seed 0, test set every 1000 iterations), one panel per scene.

    python phase2/stage_curves.py phase2/logs/gs-stages-*.out  [--out phase2/stages/psnr_vs_time.png]

Training time is the CUDA-event clock at each stage (SEGCLOCK every 1000 iterations, from run_<job>/r_*.seg);
PSNR is the test-set mean at that stage (archived run_<job>/runs/r_*/test_<it>/metrics_8bit.txt). The dashed
gray line is stock's PSNR at 30k. Stages at 3k, 6k, 9k and 12k are left out: they fall right after Faster-GS's
opacity resets (every 3000 iterations while densifying), where the test PSNR briefly collapses to about 8-10 dB.
The y range starts at the 4k stage, so the 1k-2k points may sit below the plot. These runs evaluate every 1000 iterations, so they show progress, not benchmark values.
Also writes a CSV next to the PNG with every point.
"""
import argparse
import csv
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('logs', nargs='+')
ap.add_argument('--out', default='phase2/stages/psnr_vs_time.png')
args = ap.parse_args()

RESET = lambda it: it % 3000 == 0 and it < 15000   # stage right after an opacity reset
ORDER = ['bicycle', 'flowers', 'garden', 'stump', 'treehill', 'room', 'counter', 'kitchen', 'bonsai']
# categorical slots 1-3 of the reference palette (validated: CVD dE 9.2, normal 27.6); slot 3 is < 3:1 on white,
# so every line is also direct-labeled and has its own dash pattern
ARMS = [('stock', 'stock Faster-GS', '#2a78d6', '-'), ('ab', 'A+B', '#eb6834', '--'), ('abc', 'A+B+C', '#1baf7a', ':')]
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5e0'

data = {}
for path in args.logs:
    head = open(path, encoding='utf-8', errors='replace').read()
    m = re.search(r'^node=\S+ job=(\S+)', head, re.M)
    if not m:
        continue
    base = os.path.join(os.path.dirname(path), f'run_{m.group(1)}')
    for arm, *_ in ARMS:
        for seg in [f for f in os.listdir(base) if re.match(rf'r_([a-z]+)_{arm}_s0\.seg$', f)]:
            sc = seg.split('_')[1]
            clk = {int(a): float(b) for a, b in re.findall(r'SEGCLOCK step=(\d+) \S+ cum_s=([\d.]+)', open(os.path.join(base, seg)).read())}
            rdir = os.path.join(base, 'runs', seg[:-4])
            pts = []
            for d in os.listdir(rdir) if os.path.isdir(rdir) else []:
                mm = re.match(r'test_(\d+)$', d)
                mf = os.path.join(rdir, d, 'metrics_8bit.txt')
                if mm and os.path.exists(mf) and int(mm.group(1)) in clk:
                    p = re.search(r'^PSNR\t([\d.]+)', open(mf).read(), re.M)
                    pts.append((int(mm.group(1)), clk[int(mm.group(1))], float(p.group(1))))
            data[(sc, arm)] = sorted(pts)

scenes = [s for s in ORDER if any((s, a) in data for a, *_ in ARMS)]
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
for ax, sc in zip(axes.flat, scenes):
    stock = data.get((sc, 'stock'), [])
    tail = [p for p in stock if p[0] == 30000]
    for arm, label, color, ls in ARMS:
        pts = [p for p in data.get((sc, arm), []) if not RESET(p[0])]
        if not pts:
            continue
        ax.plot([p[1] for p in pts], [p[2] for p in pts], color=color, lw=2, ls=ls, label=label)
        x, y = pts[-1][1], pts[-1][2]
        ax.plot([x], [y], 'o', ms=5, color=color, mec='white', mew=1.5)
    xmax = max(p[1] for a, *_ in ARMS for p in data.get((sc, a), []))
    if tail:
        ax.hlines(tail[0][2], 0, xmax, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
    lo = min(p[2] for a, *_ in ARMS for p in data.get((sc, a), []) if p[0] >= 4000 and not RESET(p[0]))
    hi = max(p[2] for a, *_ in ARMS for p in data.get((sc, a), []))
    ax.set_ylim(lo - 0.15, hi + 0.12)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_title(sc, loc='left', color=INK, fontsize=11)
    ax.set_xlabel('training time, s (CUDA events)')
    ax.set_ylabel('test PSNR, dB')
    # direct labels at the right end, nudged apart
    ends = sorted([(data[(sc, a)][-1][2], lab, c) for a, lab, c, _ in ARMS if data.get((sc, a))], reverse=True)
    last_y = None
    span = hi + 0.12 - (lo - 0.15)
    for y, lab, c in ends:
        yy = y if last_y is None else min(y, last_y - 0.07 * span)
        ax.annotate(lab, xy=(xmax, yy), xytext=(8, 0),
                    textcoords='offset points', va='center', fontsize=8, color=INK)
        last_y = yy
    ax.margins(x=0.02)
    ax.set_xlim(0, xmax * 1.32)
for ax in list(axes.flat)[len(scenes):]:
    ax.axis('off')
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.suptitle('Test PSNR during training vs training time (seed 0, test set every 1,000 iterations)\n'
             'dashed gray = stock at 30k; stages right after the opacity resets (3k, 6k, 9k, 12k) left out; y axis starts near the 4k value',
             color=INK, fontsize=11)
fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False, fontsize=10)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, dpi=150)
print('wrote', args.out, 'scenes:', ', '.join(scenes))
with open(os.path.splitext(args.out)[0] + '.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['scene', 'arm', 'iteration', 'train_time_s', 'test_psnr'])
    for (sc, arm), pts in sorted(data.items()):
        for it, t, p in pts:
            w.writerow([sc, arm, it, f'{t:.3f}', f'{p:.3f}'])
