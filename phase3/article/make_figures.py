"""Figures for the Part 2 article, from the completed Experiment 1R and 1A logs (no new training).

    python phase3/article/make_figures.py      -> phase3/article/fig*.png
"""
import math
import os
import re
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

LOG = 'phase2/logs'
OUT = 'phase3/article'
BR, END, DELTA = 5000, 30000, 0.05
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
C = {'ctl': '#eb6834', 'mcmc': '#2a78d6', 'oracle': '#1baf7a', 'stop': '#6250d6', 'pfx': '#8a8984'}
SCENE = {'garden': ('#008300', 'o'), 'room': ('#e87ba4', 'D')}
LABEL = {'ctl': 'Standard Faster-GS', 'mcmc': 'Switch to MCMC', 'oracle': 'MCMC, grown to standard\'s size',
         'stop': 'Standard, growth frozen'}
R1 = ['gs-x1r-garden-4559237.out', 'gs-x1r-room-s10-4559238.out', 'gs-x1r-room-s11-4559239.out', 'gs-x1r-room-s12-4559240.out']
R1A = ['gs-x1a-garden-s30-4561623.out', 'gs-x1a-garden-s31-4561625.out', 'gs-x1a-garden-s32-4561627.out',
       'gs-x1a-room-s30-4561624.out', 'gs-x1a-room-s31-4561626.out', 'gs-x1a-room-s32-4561628.out']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.titlesize': 12, 'axes.titlecolor': INK,
    'legend.frameon': False, 'savefig.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.25})


def blocks(path):
    text = open(path, encoding='utf-8', errors='replace').read()
    job = re.search(r'^node=\S+ job=(\S+)', text, re.M).group(1)
    arch = os.path.join(os.path.dirname(path), f'run_{job}', 'runs')
    out, cur = {}, None
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


def clock(lines):
    clk = {}
    for l in lines:
        m = re.match(r'SEGCLOCK step=(\d+) \S+ cum_s=(\S+) .*N=(\d+) peak', l)
        if m:
            clk[int(m.group(1))] = (float(m.group(2)), int(m.group(3)))
        m = re.match(r'FGS_RESUME step=(\d+) N=(\d+) cum_s=(\S+)', l)
        if m:
            clk[int(m.group(1))] = (float(m.group(3)), int(m.group(2)))
    return clk


def psnr(path):
    return float(re.search(r'^PSNR\t(\S+)', open(path).read(), re.M).group(1))


def load(files):
    runs = {}
    for f in files:
        b, arch = blocks(os.path.join(LOG, f))
        for (sc, arm, sd), L in b.items():
            clk = clock(L)
            if arm == 'pfx':
                runs[(sc, arm, sd)] = dict(clk=clk)
                continue
            steps = range(BR, END + 1, 1000)
            curve = {s: psnr(os.path.join(arch, f'x_{sc}_{arm}_s{sd}', f'test_{s}', 'metrics_8bit.txt')) for s in steps}
            runs[(sc, arm, sd)] = dict(clk=clk, curve=curve)
    return runs


def cross(r, tgt, from_branch=False):
    pts = sorted(r['curve'])
    for i, s in enumerate(pts):
        if r['curve'][s] >= tgt:
            if i == 0:
                return (r['clk'][s][0], s)
            a = pts[i - 1]
            pa, pb = r['curve'][a], r['curve'][s]
            w = (tgt - pa) / (pb - pa)
            t = r['clk'][a][0] + w * (r['clk'][s][0] - r['clk'][a][0])
            step = a + w * (s - a)
            if from_branch:
                return t - r['clk'][BR][0], step - BR
            return t, step
    return None


def note(fig, text, y=-0.01):
    fig.text(0.01, y, text, ha='left', va='top', fontsize=8.5, color=MUTED)


r1, r1a = load(R1), load(R1A)
allruns = {('1R',) + k: v for k, v in r1.items()}
allruns.update({('1A',) + k: v for k, v in r1a.items()})

# ------------------------------------------------------------------ 1. population size over training
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
for ax, sc, sd in [(axes[0], 'garden', 10), (axes[1], 'room', 10)]:
    p = r1[(sc, 'pfx', sd)]['clk']
    xs = [0] + sorted(k for k in p if k <= BR)
    ax.plot(xs, [0] + [p[k][1] / 1e6 for k in xs[1:]], color=C['pfx'], lw=2, label='Shared start (standard growth)')
    for arm in ('ctl', 'oracle', 'mcmc', 'stop'):
        c = r1[(sc, arm, sd)]['clk']
        ks = sorted(k for k in c if k >= BR)
        ax.plot(ks, [c[k][1] / 1e6 for k in ks], color=C[arm], lw=2, label=LABEL[arm])
    ax.axvline(BR, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(BR + 400, ax.get_ylim()[1] * 0.04, 'switch at\nstep 5,000', fontsize=9, color=INK2)
    ax.set_title(f'{sc.capitalize()} (seed {sd})', loc='left')
    ax.set_xlabel('Training step')
    ax.set_xlim(0, END)
    ax.set_xticks([0, 5000, 10000, 15000, 20000, 25000, 30000], ['0', '5k', '10k', '15k', '20k', '25k', '30k'])
axes[0].set_ylabel('Number of Gaussians (millions)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.07), fontsize=9.5)
note(fig, 'Experiment 1R, one seed shown. Standard Faster-GS keeps growing until step 15,000; the MCMC switch freezes the '
          'count at step 5,000 and recycles Gaussians instead.')
fig.savefig(f'{OUT}/fig1_population.png')
plt.close(fig)

# ------------------------------------------------------------------ 2. quality against training time
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, sc, sd in [(axes[0], 'garden', 11), (axes[1], 'room', 11)]:
    c, m = r1[(sc, 'ctl', sd)], r1[(sc, 'mcmc', sd)]
    tgt = c['curve'][END] - DELTA
    lo = min(min(r['curve'].values()) for r in (c, m)) - 0.15
    for arm, r in (('ctl', c), ('mcmc', m)):
        ks = sorted(r['curve'])
        ax.plot([r['clk'][k][0] for k in ks], [r['curve'][k] for k in ks], color=C[arm], lw=2, marker='o', ms=3.5,
                label=LABEL[arm])
    ax.axhline(tgt, color=INK2, lw=1, ls=(0, (3, 3)), label='Target: standard\'s final PSNR − 0.05 dB')
    tc, tm = cross(c, tgt)[0], cross(m, tgt)[0]
    for t, arm in ((tc, 'ctl'), (tm, 'mcmc')):
        ax.plot([t, t], [lo, tgt], color=C[arm], lw=1.2, ls=(0, (2, 2)))
        ax.plot(t, tgt, 'o', ms=9, mfc=C[arm], mec=SURF, mew=2, zorder=5)
        ax.text(t, lo + 0.05, f'{t:.0f} s', ha='center', fontsize=9.5, color=INK, fontweight='bold',
                bbox=dict(fc=SURF, ec='none', pad=1))
    ya = lo + 0.3 * (tgt - lo)
    ax.annotate('', xy=(tm, ya), xytext=(tc, ya), arrowprops=dict(arrowstyle='<->', color=INK2, lw=1))
    ax.text((tc + tm) / 2, ya + 0.03, f'{100 * (1 - tm / tc):.0f}% less time', ha='center', va='bottom', fontsize=10,
            color=INK, bbox=dict(fc=SURF, ec='none', pad=1))
    ax.set_ylim(lo, c['curve'][END] + 0.5)
    ax.set_title(f'{sc.capitalize()} (seed {sd})', loc='left')
    ax.set_xlabel('Training time (seconds, GPU training steps only)')
axes[0].set_ylabel('Test PSNR (dB, higher is better)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.06))
note(fig, 'Experiment 1R, the seed with the median saving on each scene. Points are test evaluations every 1,000 steps '
          'from the switch at step 5,000; time counts from step 0.')
fig.savefig(f'{OUT}/fig2_quality_vs_time.png')
plt.close(fig)

# ------------------------------------------------------------------ 3. all twelve pairs
rows = []
for exp, rs, seeds in (('1R', r1, (10, 11, 12)), ('1A', r1a, (30, 31, 32))):
    for sc in ('garden', 'room'):
        for sd in seeds:
            c, m = rs[(sc, 'ctl', sd)], rs[(sc, 'mcmc', sd)]
            tgt = c['curve'][END] - DELTA
            rows.append((exp, sc, sd, cross(c, tgt)[0], cross(m, tgt)[0]))
fig, ax = plt.subplots(figsize=(9, 5.6))
for i, (exp, sc, sd, tc, tm) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([tm, tc], [y, y], color=GRID, lw=3, solid_capstyle='round', zorder=1)
    ax.plot(tc, y, 'o', ms=9, mfc=C['ctl'], mec=SURF, mew=2, zorder=3)
    ax.plot(tm, y, 'o', ms=9, mfc=C['mcmc'], mec=SURF, mew=2, zorder=3)
    ax.text(tc + 4, y, f'−{100 * (1 - tm / tc):.0f}%', va='center', fontsize=9.5, color=INK)
ax.set_yticks(range(len(rows)), [f'{sc.capitalize()} · seed {sd} ({exp})' for exp, sc, sd, _, _ in reversed(rows)])
ax.grid(axis='y', visible=False)
ax.set_xlabel('Training time to reach the target quality (seconds)')
ax.set_xlim(60, 205)
ax.plot([], [], 'o', ms=9, mfc=C['ctl'], mec=SURF, label=LABEL['ctl'])
ax.plot([], [], 'o', ms=9, mfc=C['mcmc'], mec=SURF, label=LABEL['mcmc'])
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.45, 1.08))
ax.axhline(5.5, color=MUTED, lw=0.8)
note(fig, 'Every paired comparison from Experiments 1R (seeds 10–12) and 1A (fresh seeds 30–32). In each pair both '
          'runs start from the same saved model at step 5,000.')
fig.savefig(f'{OUT}/fig3_twelve_pairs.png')
plt.close(fig)

# ------------------------------------------------------------------ 4. the evaluation side-effect (schematic)
rng = np.random.default_rng(3)
cmap = plt.get_cmap('viridis')
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.axis('off')
n_before, n_after, w = 14, 14, 1.0
same = rng.permutation(40)[:n_before]
after_a, after_b = rng.permutation(40)[:n_after], rng.permutation(40)[:n_after]
for row, (y, label, after) in enumerate(((1.6, 'Run that never evaluates', after_a),
                                         (0.2, 'Run that evaluates at step 5,000', after_b))):
    ax.text(-0.4, y + 0.4, label, ha='right', va='center', fontsize=10.5, color=INK)
    for i, v in enumerate(list(same) + list(after)):
        x = i * w + (1.4 if i >= n_before else 0)
        ax.add_patch(Rectangle((x, y), w * 0.86, 0.8, fc=cmap(v / 40), ec='none'))
ax.add_patch(Rectangle((-0.1, -0.05), n_before * w, 2.7, fc='none', ec=C['mcmc'], lw=1.5, ls=(0, (3, 2))))
ax.text(n_before * w / 2, 2.85, 'Identical photo order', ha='center', fontsize=10, color=INK)
ax.text(n_before * w + 1.4 + n_after * w / 2, 2.85, 'Different photo order after the next reshuffle', ha='center',
        fontsize=10, color=INK)
ax.annotate('evaluation builds the LPIPS network,\nwhich draws from the same random generator',
            xy=(n_before * w + 0.6, 0.1), xytext=(n_before * w + 0.6, -1.0), ha='center', fontsize=9.5, color=INK2,
            arrowprops=dict(arrowstyle='->', color=INK2))
ax.set_xlim(-7.5, (n_before + n_after) * w + 2)
ax.set_ylim(-1.4, 3.2)
note(fig, 'Schematic. Each square is one training photograph, in the order the trainer visits them. Measured on room: '
          'identical through step 5,167, different from step 5,168 (the first reshuffle after the evaluation).')
fig.savefig(f'{OUT}/fig4_evaluation_side_effect.png')
plt.close(fig)

# ------------------------------------------------------------------ 5. ablation
ARMS = [('mcmc', 'Full MCMC switch'), ('mrep', 'Same again (identical rerun)'), ('mlr', 'Without the higher opacity\nlearning rate'),
        ('mreg', 'Without regularisation'), ('mnoise', 'Without noise injection'), ('mreloc', 'Without relocation'),
        ('creset', 'Standard, without\nopacity resets')]
NEVER = 1.55
fig, ax = plt.subplots(figsize=(10, 5.4))
for i, (arm, lab) in enumerate(ARMS):
    y = len(ARMS) - 1 - i
    for j, sc in enumerate(('garden', 'room')):
        for k, sd in enumerate((30, 31, 32)):
            c, r = r1a[(sc, 'ctl', sd)], r1a[(sc, arm, sd)]
            tgt = c['curve'][END] - DELTA
            tc, xa = cross(c, tgt)[0], cross(r, tgt)
            x = NEVER + 0.03 * (k - 1) if xa is None else xa[0] / tc
            ax.plot(x, y + (j - 0.5) * 0.28, SCENE[sc][1], ms=7.5 if j else 8.5, mfc=SCENE[sc][0], mec=SURF, mew=1.6, zorder=3)
ax.axvline(1.0, color=INK2, lw=1)
ax.text(1.0, len(ARMS) - 0.35, 'standard\nFaster-GS', ha='center', fontsize=9, color=INK2)
ax.axvspan(1.42, 1.7, color='#f0efec', zorder=0)
ax.text(NEVER, len(ARMS) - 0.35, 'never reached\nthe target', ha='center', fontsize=9, color=INK2)
ax.set_yticks(range(len(ARMS)), [lab for _, lab in reversed(ARMS)])
ax.grid(axis='y', visible=False)
ax.set_xlim(0.45, 1.7)
ax.set_ylim(-0.6, len(ARMS) + 0.2)
ax.set_xticks([0.5, 0.75, 1.0, 1.25], ['0.5×', '0.75×', '1×', '1.25×'])
ax.set_xlabel('Time to reach the target, relative to standard Faster-GS on the same seed (lower is faster)')
for sc in ('garden', 'room'):
    ax.plot([], [], SCENE[sc][1], ms=8.5, mfc=SCENE[sc][0], mec=SURF, label=sc.capitalize())
ax.legend(loc='lower left')
note(fig, 'Experiment 1A, seeds 30–32. One dot per run; each is compared with standard Faster-GS on the same scene and '
          'seed, branched from the same saved model.')
fig.savefig(f'{OUT}/fig5_ablation.png')
plt.close(fig)

# ------------------------------------------------------------------ 6. fewer steps vs cheaper steps
pairs = []   # (group, scene, steps ratio, cost ratio)
for exp, rs, seeds, arms in (('1R', r1, (10, 11, 12), ('mcmc', 'oracle')), ('1A', r1a, (30, 31, 32), ('mcmc',))):
    for sc in ('garden', 'room'):
        for sd in seeds:
            c = rs[(sc, 'ctl', sd)]
            tgt = c['curve'][END] - DELTA
            xc = cross(c, tgt, True)
            for arm in arms:
                xa = cross(rs[(sc, arm, sd)], tgt, True)
                pairs.append((arm, sc, xa[1] / xc[1], (xa[0] / xa[1]) / (xc[0] / xc[1])))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, gridspec_kw=dict(wspace=0.12))
groups = [('mcmc', 'garden'), ('mcmc', 'room'), ('oracle', 'garden'), ('oracle', 'room')]
names = ['Switch to MCMC · garden', 'Switch to MCMC · room', 'MCMC at standard\'s size · garden',
         'MCMC at standard\'s size · room']
for ax, idx, title in ((axes[0], 2, 'Steps needed to reach the target'), (axes[1], 3, 'Time per training step')):
    ax.set_axisbelow(True)
    for g, (arm, sc) in enumerate(groups):
        y = len(groups) - 1 - g
        vals = [p[idx] for p in pairs if p[0] == arm and p[1] == sc]
        med = st.median(vals)
        ax.barh(y, med - 1, left=1, height=0.5, color=C[arm], alpha=0.9)
        ax.plot(vals, [y] * len(vals), 'o', ms=5, mfc=SURF, mec=INK2, mew=1, zorder=3)
        ax.text(min(vals + [med]) - 0.03 if med < 1 else max(vals + [med]) + 0.03, y, f'{med:.2f}×', va='center',
                ha='right' if med < 1 else 'left', fontsize=10, color=INK, fontweight='bold')
    ax.axvline(1, color=INK2, lw=1)
    ax.set_xlim(0.25, 1.35)
    ax.set_title(title, loc='left')
    ax.set_xlabel('relative to standard Faster-GS (= 1×)')
    ax.grid(axis='y', visible=False)
axes[0].set_yticks(range(len(groups)), list(reversed(names)))
note(fig, 'Measured from the switch at step 5,000 to the target. Bars: median; dots: individual pairs (MCMC switch: 1R and '
          '1A, 6 per scene; MCMC at standard\'s size: 1R only, 3 per scene).', y=-0.07)
fig.savefig(f'{OUT}/fig6_steps_vs_cost.png')
plt.close(fig)
print('written', sorted(f for f in os.listdir(OUT) if f.endswith('.png')))
