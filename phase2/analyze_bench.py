"""Validate and analyze the Phase 2 step 3 benchmark (stock / A+B / A+B+C, one job per Mip-NeRF360 scene).

    python phase2/analyze_bench.py phase2/logs/gs-bench-*.out [--seeds 0 1 2]
    python phase2/analyze_bench.py phase2/logs/gs-bench6-*.out --arms stock abcds   (step 6: stock vs A+B+C+D+S)

Validation first, per scene. A scene is analyzed only if every run passes:
  - exactly one run block '##### b_<scene>_<arm>_s<seed>' per arm x seed, and no FAILED line for it;
  - one 'SEGCLOCK step=30000' record with a finite train time, N and training-only peak VRAM (clock hook v4);
  - one 'FGS_N iteration=15000' line and one 'FGS_TRAIN_END' line (v4);
  - one PERIMG line, with the same number of test images in every run of the scene;
  - one RESULT line with finite PSNR / SSIM / LPIPS;
  - A+B and A+B+C: exactly one switch line at 15000 and sparse=1 at the end; stock: no switch line and sparse=0.
Every run's archived per-image test metrics (phase2/logs/run_<job>/runs/<run>/test_30000/metrics_8bit.txt) are
required. Each test image is compared with the same image in the scene's other runs (and the same-seed runs of the
other arms are listed): an image more than 1 dB below that median is flagged as a possible one-view artifact.
Flagged runs stay in all results; the flags are reported next to them.

The overall verdict is given only when all 9 scenes x arms x 3 seeds (81 runs for 3 arms) pass validation. Otherwise the
summary is printed as partial, the verdict reads 'incomplete - not evaluated', and the exit code is 2.

Report: one table per scene (paired deltas against stock per seed), then a summary across scenes and the
pre-registered win criterion from phase2/PLAN.md, for each arm other than stock (--arms, default A+B and A+B+C):
  median training-time reduction >= 20%, mean dPSNR >= -0.05 dB, no scene below -0.10 dB, mean dLPIPS <= +0.002.
"""
import argparse
import math
import os
import re
import statistics as st
import sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('logs', nargs='+')
ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
ap.add_argument('--arms', nargs='+', default=['stock', 'ab', 'abc'])
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

ARMS = tuple(args.arms)
assert ARMS[0] == 'stock', 'the first arm must be stock (the reference)'
TREAT = ARMS[1:]
LABEL = {'stock': 'stock Faster-GS', 'ab': 'A+B', 'abc': 'A+B+C', 'abcd': 'A+B+C+D', 'abcds': 'A+B+C+D+S'}
ORDER = ['bicycle', 'flowers', 'garden', 'stump', 'treehill', 'room', 'counter', 'kitchen', 'bonsai']
IMG_DROP = 1.0   # flag a test image this many dB below the median of the same image in the scene's other runs
fin = lambda x: x is not None and math.isfinite(x)

blocks = defaultdict(list)          # (scene, arm, seed) -> [lines] per header occurrence
jobs, archives = {}, {}   # archives: per run (scene, arm, seed), so several jobs of one scene can be pooled
for path in args.logs:
    cur = None
    head = open(path, encoding='utf-8', errors='replace').read()
    mjob = re.search(r'^node=(\S+) job=(\S+)', head, re.M)
    for line in head.splitlines():
        m = re.match(r'##### b_([a-z]+)_(' + '|'.join(sorted(ARMS, key=len, reverse=True)) + r')_s(\d+):', line)
        if m:
            cur = (m.group(1), m.group(2), int(m.group(3)))
            blocks[cur].append([])
            if mjob:
                jobs[cur[0]] = f'{mjob.group(2)} ({mjob.group(1)})'
                archives[cur] = os.path.join(os.path.dirname(path), f'run_{mjob.group(2)}', 'runs')
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
    # per-image comparison: the same test image across the scene's runs (archived metrics_8bit.txt of every run)
    for k in [k for k in runs if k[0] == sc]:
        name = f'b_{sc}_{k[1]}_s{k[2]}'
        mf = os.path.join(archives.get(k, ''), name, 'test_30000', 'metrics_8bit.txt')
        if not os.path.exists(mf):
            errors[sc].append(f'{name}: no archived per-image metrics ({mf})'); continue
        rows_ = list(re.finditer(r'^(\d+)\t([\d.]+)\t([\d.]+)\t([\d.]+)', open(mf).read().split('Image', 1)[-1], re.M))
        img = {int(m.group(1)): float(m.group(2)) for m in rows_}
        if len(img) != runs[k]['n_img']:
            errors[sc].append(f'{name}: {len(img)} per-image values, PERIMG says {runs[k]["n_img"]}')
        runs[k]['img'] = img
        runs[k]['img_lpips'] = {int(m.group(1)): float(m.group(4)) for m in rows_}
    for k in [k for k in runs if k[0] == sc and 'img' in runs[k]]:
        for i, v in runs[k]['img'].items():
            others = [runs[j]['img'][i] for j in runs if j[0] == sc and j != k and i in runs[j].get('img', {})]
            paired = {a: runs[(sc, a, k[2])]['img'].get(i) for a in ARMS if a != k[1] and 'img' in runs.get((sc, a, k[2]), {})}
            if others and v < st.median(others) - IMG_DROP:
                flags[sc].append(f'b_{sc}_{k[1]}_s{k[2]}: test image {i} at {v:.2f} dB; same image in the other runs: median '
                                 f'{st.median(others):.2f} dB, same seed ' + ', '.join(f'{LABEL[a]} {p:.2f}' for a, p in paired.items() if p is not None))

ok = [sc for sc in scenes if not errors[sc]]
print(f'## Validation: {len(ok)}/{len(ORDER)} expected scenes pass ({", ".join(scenes)})')
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
    if 'ab' in ARMS and 'abc' in ARMS:
        print(f'\nC on top of A+B (A+B+C vs A+B, paired): time {f3(red(sc, "abc", "ab"), "%.1f")}% reduction, '
              f'ΔPSNR {f3(dif(sc, "abc", "psnr", "ab"), "%+.3f")}, ΔLPIPS {st.mean(dif(sc, "abc", "lpips", "ab")):+.4f}')
    summary[sc] = {arm: dict(red=st.mean(red(sc, arm)), dpsnr=st.mean(dif(sc, arm, 'psnr')), dssim=st.mean(dif(sc, arm, 'ssim')),
                             dlpips=st.mean(dif(sc, arm, 'lpips')), seeds_faster=sum(x > 0 for x in red(sc, arm)))
                   for arm in TREAT}

complete = set(ok) == set(ORDER) and all((sc, a, s) in runs for sc in ORDER for a in ARMS for s in S)
verdicts = {}
print('\n## Across scenes (each scene = mean of its paired seeds)' + ('' if complete else f' - PARTIAL, {len(ok)}/{len(ORDER)} scenes') + '\n')
print('| scene | ' + ' | '.join(f'{LABEL[a]} time | {LABEL[a]} ΔPSNR | {LABEL[a]} ΔLPIPS' for a in TREAT) + ' |')
print('|---|' + '---|---|---|' * len(TREAT))
for sc in ok:
    print(f'| {sc} | ' + ' | '.join(f"−{summary[sc][a]['red']:.1f}% | {summary[sc][a]['dpsnr']:+.3f} | {summary[sc][a]['dlpips']:+.4f}" for a in TREAT) + ' |')
print()
for arm in TREAT:
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
        print(f'  - {"meets" if v else "misses"} {k}' + ('' if complete else ' (partial data, not a verdict)'))
    verdicts[arm] = all(crit.values())

# Outdoor / indoor split and speedup factors (descriptive, not part of the verdict). A scene's speedup is stock time /
# arm time, from the mean of its paired per-seed ratios; "aggregate" is total stock time / total arm time over the scenes.
OUT_SC, IN_SC = ORDER[:5], ORDER[5:]
print('\n## Outdoor / indoor and speedup factors (descriptive; scene speedup = mean over seeds of stock time / arm time)\n')
print('| arm | group | scenes | time reduction median / mean | speedup median / mean | speedup range | aggregate speedup | ΔPSNR mean |')
print('|---|---|---|---|---|---|---|---|')
for arm in TREAT:
    for gname, grp in (('all', ORDER), ('outdoor', OUT_SC), ('indoor', IN_SC)):
        g = [sc for sc in grp if sc in ok]
        if not g:
            continue
        reds = [summary[sc][arm]['red'] for sc in g]
        spd = [st.mean(b / a for a, b in zip(get(sc, arm, 't'), get(sc, 'stock', 't'))) for sc in g]
        agg = sum(sum(get(sc, 'stock', 't')) for sc in g) / sum(sum(get(sc, arm, 't')) for sc in g)
        print(f'| {LABEL[arm]} | {gname} | {len(g)} | {st.median(reds):.1f}% / {st.mean(reds):.1f}% | '
              f'{st.median(spd):.3f}× / {st.mean(spd):.3f}× | {min(spd):.3f}–{max(spd):.3f}× | {agg:.3f}× | '
              f'{st.mean(summary[sc][arm]["dpsnr"] for sc in g):+.3f} |')

# Sensitivity (not the verdict): one-view failures, i.e. a test image more than SEVERE dB below the same image's
# median in the scene's other runs. Count them per arm, then drop every such image from all runs of that scene and
# recompute the paired deltas on the remaining images.
SEVERE = 3.0
print(f'\n## Sensitivity: one-view failures (a test image > {SEVERE:.0f} dB below the same image in the scene\'s other runs)\n')
severe = defaultdict(set)
count = defaultdict(list)
for sc in ok:
    for k in [k for k in runs if k[0] == sc]:
        for i, v in runs[k]['img'].items():
            others = [runs[j]['img'][i] for j in runs if j[0] == sc and j != k]
            if v < st.median(others) - SEVERE:
                severe[sc].add(i)
                count[k[1]].append((f'{sc} s{k[2]}', f'{sc} s{k[2]} image {i} ({v - st.median(others):+.1f} dB)'))
for arm in ARMS:
    nruns = len({r for r, _ in count[arm]})
    print(f'- {LABEL[arm]}: {len(count[arm])} failing image(s) in {nruns} of {len(ok) * len(S)} runs'
          + (': ' + '; '.join(t for _, t in count[arm]) if count[arm] else ''))
print()
print('These happen without the changes (stock has them), but these counts cannot show whether the changes alter how often '
      'they happen. The exclusion below is a secondary analysis; the verdict uses all images.')
print('\nDeltas with those images removed from every run of the scene (paired per seed, mean of 3 seeds):\n')
CTOP = 'ab' in ARMS and 'abc' in ARMS
print('| scene | images removed | ' + ' | '.join(f'{LABEL[a]} ΔPSNR | {LABEL[a]} ΔLPIPS' for a in TREAT)
      + (' | C on top of A+B ΔPSNR' if CTOP else '') + ' |')
print('|---|---|' + '---|---|' * len(TREAT) + ('---|' if CTOP else ''))
rob = defaultdict(dict)
for sc in ok:
    keep = lambda k, key: st.mean(v for i, v in runs[k][key].items() if i not in severe[sc])
    d = lambda arm, key, ref='stock': st.mean(keep((sc, arm, s), key) - keep((sc, ref, s), key) for s in S)
    rob[sc] = {arm: (d(arm, 'img'), d(arm, 'img_lpips')) for arm in TREAT}
    print(f"| {sc} | {', '.join(map(str, sorted(severe[sc]))) or '-'} | "
          + ' | '.join(f'{rob[sc][a][0]:+.3f} | {rob[sc][a][1]:+.4f}' for a in TREAT)
          + (f" | {d('abc', 'img', 'ab'):+.3f}" if CTOP else '') + ' |')
for arm in TREAT:
    dps = [rob[sc][arm][0] for sc in ok]
    worst = min(ok, key=lambda sc: rob[sc][arm][0])
    print(f'\n{LABEL[arm]} with one-view failures removed: ΔPSNR mean {st.mean(dps):+.3f}, worst {worst} {min(dps):+.3f}, '
          f'ΔLPIPS mean {st.mean(rob[sc][arm][1] for sc in ok):+.4f}; tolerances '
          + ('still met' if st.mean(dps) >= -0.05 and min(dps) >= -0.10 and st.mean(rob[sc][arm][1] for sc in ok) <= 0.002 else 'NOT met'))

n_valid = sum(1 for k in runs if k[0] in ok)
print()
if complete:
    for arm in TREAT:
        print(f'VERDICT {LABEL[arm]}: {"PASS" if verdicts[arm] else "FAIL"} (all 9 scenes, {len(ORDER) * len(ARMS) * len(S)} runs validated)')
else:
    print(f'VERDICT: incomplete - not evaluated ({len(ok)}/{len(ORDER)} scenes and {n_valid}/{len(ORDER) * len(ARMS) * len(S)} runs validated)')
    sys.exit(2)
