"""Phase 2 step 4 profiling: where the training step's time goes, stock vs A+B+C (garden = outdoor, room = indoor).

    python phase2/analyze_profile.py phase2/logs/gs-prof-*.out [--plot phase2/profile/stage_split.png]

1. Step split from the full 30k profiling runs (STAGE lines: CUDA events around forward = render + loss, backward,
   optimizer = step + zero_grad + post-step, sampled every 250 steps). Reported as means over two phases:
   densifying (steps < 15000; B is off in both arms) and after densification (>= 15000; B on in A+B+C).
   The hooks synchronize the host every step, so absolute times are a little above benchmark times.
2. Kernels from the nsys captures (50 steps at 5000 and at 20000): GPU time per step by kernel group (by name)
   and the top kernels. GPU kernel time excludes launch gaps, so it is lower than the step time.
"""
import argparse
import csv
import glob
import os
import re
import statistics as st
import sys

ap = argparse.ArgumentParser()
ap.add_argument('logs', nargs='+')
ap.add_argument('--plot', default='phase2/profile/stage_split.png')
args = ap.parse_args()
sys.stdout.reconfigure(encoding='utf-8')

ARMS = [('stock', 'stock Faster-GS'), ('abc', 'A+B+C')]
GROUPS = [  # (group, regex on the kernel name); first match wins
    ('optimizer (Adam)', r'adam'),
    ('backward: blend', r'blend_backward'),
    ('backward: preprocess', r'preprocess_backward'),
    ('backward: other', r'backward'),
    ('forward: blend', r'blend_cu|blend_forward|render_cu|forward::blend'),
    ('forward: preprocess', r'preprocess'),
    ('forward: sort', r'[Ss]ort|radix|Radix|cub::'),
    ('PyTorch ops (loss, fills, copies)', r'at::native|elementwise|reduce|copy|Fill'),
    ('other', r'.'),
]

runs = {}
for path in args.logs:
    text = open(path, encoding='utf-8', errors='replace').read()
    m = re.search(r'^node=\S+ job=(\S+)', text, re.M)
    sc = re.search(r'########## profile \(scene (\w+)', text)
    if not (m and sc):
        continue
    runs[sc.group(1)] = dict(job=m.group(1), base=os.path.join(os.path.dirname(path), f'run_{m.group(1)}'))
if not runs:
    sys.exit('no profile logs')

print('# Profiling: where the training step goes (seed 0)\n')
split = {}
for sc, r in runs.items():
    print(f'## {sc} (job {r["job"]})\n')
    print('Step split from CUDA events (mean ms per step over the sampled steps):\n')
    print('| arm | phase | forward | backward | optimizer | sum | forward % | backward % | optimizer % | Gaussians (mean) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for arm, label in ARMS:
        f = os.path.join(r['base'], f'p_{sc}_{arm}_s0.prof')
        if not os.path.exists(f):
            print(f'| {label} | missing {f} | | | | | | | | |'); continue
        rows = [tuple(map(float, m.groups())) for m in re.finditer(
            r'^STAGE step=(\d+) fwd=([\d.]+) bwd=([\d.]+) opt=([\d.]+) n_gauss=(\d+)', open(f).read(), re.M)]
        for ph, lo, hi in (('densifying (<15k)', 0, 15000), ('after (>=15k)', 15000, 10 ** 9), ('whole run', 0, 10 ** 9)):
            sel = [x for x in rows if lo <= x[0] < hi]
            if not sel:
                continue
            fw, bw, op = (st.mean(x[i] for x in sel) for i in (1, 2, 3))
            tot = fw + bw + op
            split[(sc, arm, ph)] = (fw, bw, op)
            print(f'| {label} | {ph} | {fw:.2f} | {bw:.2f} | {op:.2f} | {tot:.2f} | {100 * fw / tot:.0f}% | {100 * bw / tot:.0f}% | '
                  f'{100 * op / tot:.0f}% | {st.mean(x[4] for x in sel) / 1e6:.2f} M |')
    print()
    for arm, label in ARMS:
        for w in (5000, 20000):
            name = f'n_{sc}_{arm}_{w}'
            kf = glob.glob(os.path.join(r['base'], f'{name}*cuda_gpu_kern_sum*.csv'))
            if not kf:
                print(f'- {label}, steps {w}-{w + 50}: no kernel summary\n'); continue
            ks = list(csv.DictReader(open(kf[0])))
            tot = sum(float(k['Total Time (ns)']) for k in ks)
            grp = {}
            for k in ks:
                g = next(g for g, rx in GROUPS if re.search(rx, k['Name']))
                grp[g] = grp.get(g, 0.0) + float(k['Total Time (ns)'])
            print(f'Kernels, {label}, steps {w}–{w + 50}: {tot / 50 / 1e6:.2f} ms of GPU kernel time per step\n')
            print('| kernel group | ms/step | share |')
            print('|---|---|---|')
            for g, _ in GROUPS:
                if g in grp:
                    print(f'| {g} | {grp[g] / 50 / 1e6:.3f} | {100 * grp[g] / tot:.1f}% |')
            print('\nTop kernels:\n')
            print('| share | ms/step | calls/step | kernel |')
            print('|---|---|---|---|')
            for k in ks[:8]:
                t = float(k['Total Time (ns)'])
                nm = re.sub(r'\(.*', '', k['Name']).replace('void ', '')[:90]
                print(f'| {100 * t / tot:.1f}% | {t / 50 / 1e6:.3f} | {int(k["Instances"]) / 50:g} | `{nm}` |')
            print()

# stacked bars: per scene, arm x phase, forward / backward / optimizer
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    COL = {'forward': '#2a78d6', 'backward': '#eb6834', 'optimizer': '#1baf7a'}   # validated categorical slots 1-3
    INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5e0'
    scenes = list(runs)
    fig, axes = plt.subplots(1, len(scenes), figsize=(5.2 * len(scenes), 4.4), constrained_layout=True, squeeze=False)
    for ax, sc in zip(axes[0], scenes):
        bars = [(f'{lab}\n{ph.split(" ")[0]}', split.get((sc, arm, ph))) for arm, lab in ARMS for ph in ('densifying (<15k)', 'after (>=15k)')]
        xs = range(len(bars))
        bottom = [0.0] * len(bars)
        for i, part in enumerate(('forward', 'backward', 'optimizer')):
            vals = [b[1][i] if b[1] else 0 for b in bars]
            ax.bar(xs, vals, bottom=bottom, color=COL[part], width=0.62, label=part, edgecolor='white', linewidth=2)
            for x, v, b0 in zip(xs, vals, bottom):
                if v > 0.35:
                    ax.text(x, b0 + v / 2, f'{v:.2f}', ha='center', va='center', fontsize=8, color='white')
            bottom = [b + v for b, v in zip(bottom, vals)]
        for x, b in zip(xs, bottom):
            ax.text(x, b + 0.08, f'{b:.2f} ms', ha='center', va='bottom', fontsize=8, color=INK)
        ax.set_xticks(list(xs), [b[0] for b in bars], fontsize=8)
        ax.set_ylabel('ms per training step (CUDA events)')
        ax.set_title(sc, loc='left', color=INK)
        ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Training step split, stock vs A+B+C, while densifying (<15k) and after (>=15k); seed 0, profiling runs',
                 color=INK, fontsize=10)
    os.makedirs(os.path.dirname(args.plot), exist_ok=True)
    fig.savefig(args.plot, dpi=150)
    print(f'(figure: {args.plot})')
except ImportError:
    pass
