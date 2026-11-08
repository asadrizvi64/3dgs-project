"""Phase 3 Experiment 1 analysis: time to quality per arm, with the pre-registered decision rule (phase3/PLAN.md).

    python phase3/analyze_ttq.py phase2/logs/gs-x1-garden-<job>.out phase2/logs/gs-x1-room-<job>.out \
        [--ref phase2/logs/gs-up-garden-<job>.out phase2/logs/gs-up-room-<job>.out] [--seeds 0 1 2]

Inputs: the exp1 job logs (build_plus.sbatch stage exp1: run blocks '##### x_<scene>_<arm>_s<seed>:', each with the
clock hook's lines) and the archived per-step test metrics phase2/logs/run_<job>/runs/<run>/test_<i>/metrics_8bit.txt.
--ref: the upbench logs with uninterrupted FasterGSUp runs (arm 'up'), for check 1.

Time axis: SEGCLOCK cum_s, the CUDA-event training time from step 0 (the prefix's time carries into each arm).
Evaluation i is the state after exactly i steps, so it pairs with 'SEGCLOCK step=i'. The point at 30000 is the
end-of-training render (after the opacity cleanup), as in phase 2.
T (time to target): first crossing of target = ctl's PSNR at 30000 minus 0.05 dB (per scene and seed), linearly
interpolated between the evaluation points around it; 'not reached' if the arm never reaches it.
"""
import argparse
import os
import re
import statistics as st
import sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('logs', nargs='+')
ap.add_argument('--ref', nargs='*', default=[])
ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
ap.add_argument('--branch', type=int, default=5000)
args = ap.parse_args()
ARMS = ('ctl', 'stop', 'mcmc', 'oracle')
CAND = ('stop', 'mcmc')
DELTA, SENS = 0.05, (0.10, 0.20)
SEVERE = 3.0


def blocks_of(paths, pat):
    out, arch = defaultdict(list), {}
    for path in paths:
        text = open(path, encoding='utf-8', errors='replace').read()
        job = re.search(r'^node=\S+ job=(\S+)', text, re.M)
        cur = None
        for line in text.splitlines():
            m = re.match(pat, line)
            if m:
                cur = m.groups()
                out[cur].append([])
                if job:
                    arch[cur] = os.path.join(os.path.dirname(path), f'run_{job.group(1)}', 'runs')
                continue
            if line.startswith('#####'):
                cur = None
            if cur is not None:
                out[cur][-1].append(line)
    return out, arch


def metrics(path):
    t = open(path).read()
    head, _, imgs = t.partition('Image')
    mean = {k: float(re.search(rf'^{k}\t(\S+)', head, re.M).group(1)) for k in ('PSNR', 'SSIM', 'LPIPS')}
    per = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r'^(\d+)\t([\d.]+)', imgs, re.M)}
    return mean, per


blocks, arch = blocks_of(args.logs, r'##### x_([a-z]+)_([a-z]+)_s(\d+):')
scenes = sorted({k[0] for k in blocks})
errors = defaultdict(list)
runs = {}
for (sc, arm, seed), bl in blocks.items():
    seed = int(seed)
    if arm == 'pfx':
        continue
    name = f'x_{sc}_{arm}_s{seed}'
    L = bl[-1]
    if len(bl) != 1:
        errors[sc].append(f'{name}: {len(bl)} blocks')
    if any(l.startswith(f'FAILED {name}') for l in L):
        errors[sc].append(f'{name}: FAILED line'); continue
    res = [re.match(r'FGS_RESUME step=(\d+) N=(\d+) cum_s=(\S+)', l) for l in L]
    res = [m for m in res if m]
    clk = {int(m.group(1)): (float(m.group(2)), int(m.group(3))) for m in
           (re.match(r'SEGCLOCK step=(\d+) \S+ cum_s=(\S+) .*N=(\d+) peak', l) for l in L) if m}
    end = [re.match(r'FGS_TRAIN_END N=(\d+) peak_alloc_gib=(\S+)', l) for l in L]
    end = [m for m in end if m]
    if len(res) != 1 or int(res[0].group(1)) != args.branch:
        errors[sc].append(f'{name}: expected one FGS_RESUME at {args.branch}, got {[m.group(0) for m in res]}'); continue
    clk[args.branch] = (float(res[0].group(3)), int(res[0].group(2)))
    steps = list(range(args.branch, 30001, 1000))
    missing = [s for s in steps if s not in clk]
    if missing or len(end) != 1:
        errors[sc].append(f'{name}: clock points missing {missing[:5]} / FGS_TRAIN_END lines {len(end)}'); continue
    curve, per = {}, {}
    for s in steps:
        f = os.path.join(arch.get((sc, arm, str(seed)), ''), name, f'test_{s}', 'metrics_8bit.txt')
        if not os.path.exists(f):
            errors[sc].append(f'{name}: no metrics at {s}'); break
        curve[s], per[s] = metrics(f)
    else:
        runs[(sc, arm, seed)] = dict(clk=clk, curve=curve, per=per, n_end=int(end[0].group(1)), peak=float(end[0].group(2)),
                                    resume=res[0].group(0))

# ---------------------------------------------------------------- checks
print('## Validation\n')
for sc in scenes:
    for a in ARMS:
        for s in args.seeds:
            if (sc, a, s) not in runs and not any(e.startswith(f'x_{sc}_{a}_s{s}') for e in errors[sc]):
                errors[sc].append(f'x_{sc}_{a}_s{s}: missing')
    for s in args.seeds:   # all arms start from the same state: identical test metrics at the branch step
        b = [runs[(sc, a, s)]['curve'][args.branch] for a in ARMS if (sc, a, s) in runs]
        if b and any(x != b[0] for x in b):
            errors[sc].append(f's{s}: test metrics at the branch step differ between arms {b}')
    for s in args.seeds:   # check 2: each arm did what it says
        g = lambda a: runs.get((sc, a, s))
        if g('ctl'):
            if not g('ctl')['clk'][15000][1] > g('ctl')['clk'][args.branch][1]:
                errors[sc].append(f's{s} ctl: no growth after the branch')
        if g('stop'):
            ns = [g('stop')['clk'][k][1] for k in sorted(g('stop')['clk']) if k >= args.branch]
            if any(b > a for a, b in zip(ns, ns[1:])):
                errors[sc].append(f's{s} stop: N grew after the branch')
        if g('mcmc'):
            n0 = g('mcmc')['clk'][args.branch][1]
            if any(v[1] != n0 for k, v in g('mcmc')['clk'].items() if k >= args.branch):
                errors[sc].append(f's{s} mcmc: N not held at {n0}')
        if g('oracle') and g('ctl'):
            cap = runs[(sc, 'ctl', s)]['n_end']
            if g('oracle')['clk'][15000][1] != cap:
                errors[sc].append(f's{s} oracle: N at 15000 {g("oracle")["clk"][15000][1]} != cap {cap}')
for sc in scenes:
    print(f'- {sc}: ' + ('all checks pass' if not errors[sc] else f'{len(errors[sc])} problem(s)'))
    for e in errors[sc]:
        print(f'  - {e}')

# check 1: branched ctl vs uninterrupted runs of the same build
ref, rarch = blocks_of(args.ref, r'##### b_([a-z]+)_(up)_s(\d+):')
print('\n## Check 1: ctl (branched) vs uninterrupted FasterGSUp runs\n')
check1 = {}
for sc in scenes:
    up = []
    for (rsc, _, s), bl in ref.items():
        if rsc != sc:
            continue
        r = [re.match(r'RESULT \S+ PSNR:(\S+) .*N_Gaussians:(\d+)', l) for l in bl[-1]]
        r = [m for m in r if m]
        if r:
            up.append((float(r[0].group(1)), int(r[0].group(2))))
    ctl = [(runs[(sc, 'ctl', s)]['curve'][30000]['PSNR'], None) for s in args.seeds if (sc, 'ctl', s) in runs]
    nctl = []
    for s in args.seeds:
        bl = blocks.get((sc, 'ctl', str(s)))
        m = bl and [re.match(r'RESULT \S+ PSNR:\S+ .*N_Gaussians:(\d+)', l) for l in bl[-1]]
        m = [x for x in (m or []) if x]
        if m:
            nctl.append(int(m[0].group(1)))
    if len(up) < 2 or len(ctl) < len(args.seeds) or len(nctl) < len(args.seeds):
        print(f'- {sc}: not evaluable (uninterrupted runs {len(up)}, ctl runs {len(ctl)})')
        check1[sc] = False
        continue
    up_p, up_n = [u[0] for u in up], [u[1] for u in up]
    cp = st.mean(c[0] for c in ctl)
    ok_p = abs(cp - st.mean(up_p)) <= max(up_p) - min(up_p)
    ok_n = abs(st.mean(nctl) - st.mean(up_n)) <= max(up_n) - min(up_n)
    check1[sc] = ok_p and ok_n
    print(f'- {sc}: ctl PSNR {cp:.3f} vs uninterrupted {st.mean(up_p):.3f} (range {min(up_p):.3f}–{max(up_p):.3f}) '
          f'{"ok" if ok_p else "OUTSIDE"}; final N {st.mean(nctl) / 1e6:.3f} M vs {st.mean(up_n) / 1e6:.3f} M '
          f'(range {min(up_n) / 1e6:.3f}–{max(up_n) / 1e6:.3f} M) {"ok" if ok_n else "OUTSIDE"}')
print('Criterion: |mean difference| <= the uninterrupted runs\' range (max - min), for PSNR and for final N.')


# ---------------------------------------------------------------- time to target
def ttq(run, target, stay=False):
    pts = [(run['clk'][s][0], run['curve'][s]['PSNR']) for s in sorted(run['curve'])]
    for i, (t, p) in enumerate(pts):
        if p >= target and (not stay or all(q >= target for _, q in pts[i:])):
            if i == 0:
                return t
            (t0, p0) = pts[i - 1]
            return t0 + (t - t0) * (target - p0) / (p - p0) if p != p0 else t
    return None


print('\n## Time to target (target = ctl PSNR at 30k minus 0.05 dB; training seconds from step 0)\n')
summary = {}
for sc in scenes:
    print(f'### {sc}\n')
    print('| arm | T per seed (s) | ΔT vs ctl per seed | final PSNR | ΔPSNR | ΔLPIPS | final N | train time to 30k (s) | peak GiB |')
    print('|---|---|---|---|---|---|---|---|---|')
    for a in ARMS:
        ts, dts, dps, dls, rows = [], [], [], [], []
        for s in args.seeds:
            r, c = runs.get((sc, a, s)), runs.get((sc, 'ctl', s))
            if not r or not c:
                ts.append(None); continue
            tgt = c['curve'][30000]['PSNR'] - DELTA
            t, tc = ttq(r, tgt), ttq(c, tgt)
            ts.append(t)
            dts.append(None if t is None or tc is None else 100 * (1 - t / tc))
            dps.append(r['curve'][30000]['PSNR'] - c['curve'][30000]['PSNR'])
            dls.append(r['curve'][30000]['LPIPS'] - c['curve'][30000]['LPIPS'])
            rows.append(r)
        if not rows:
            continue
        fmt = lambda v, f: ' / '.join('n/r' if x is None else f % x for x in v)
        print(f'| {a} | {fmt(ts, "%.1f")} | {"" if a == "ctl" else fmt(dts, "%+.1f%%")} | '
              f'{st.mean(r["curve"][30000]["PSNR"] for r in rows):.3f} | {"" if a == "ctl" else f"{st.mean(dps):+.3f}"} | '
              f'{"" if a == "ctl" else f"{st.mean(dls):+.4f}"} | {st.mean(r["n_end"] for r in rows) / 1e6:.3f} M | '
              f'{st.mean(r["clk"][30000][0] for r in rows):.1f} | {st.mean(r["peak"] for r in rows):.2f} |')
        summary[(sc, a)] = dict(dts=dts, dps=dps, dls=dls, n=len(rows))
    print()
    for d in SENS:   # sensitivity: other targets, and "reaches and stays"
        line = []
        for a in ARMS:
            v = []
            for s in args.seeds:
                r, c = runs.get((sc, a, s)), runs.get((sc, 'ctl', s))
                if r and c:
                    tgt = c['curve'][30000]['PSNR'] - d
                    t = ttq(r, tgt)
                    v.append('n/r' if t is None else f'{t:.0f}')
            line.append(f'{a} {"/".join(v)}')
        print(f'- sensitivity, target −{d:.2f} dB: ' + '; '.join(line))
    line = []
    for a in ARMS:
        v = []
        for s in args.seeds:
            r, c = runs.get((sc, a, s)), runs.get((sc, 'ctl', s))
            if r and c:
                t = ttq(r, c['curve'][30000]['PSNR'] - DELTA, stay=True)
                v.append('n/r' if t is None else f'{t:.0f}')
        line.append(f'{a} {"/".join(v)}')
    print('- sensitivity, reaches −0.05 dB and stays: ' + '; '.join(line) + '\n')

# one-view failures at 30k (phase 2 definition, within each scene)
print('## One-view failures at 30k (a test image > 3 dB below the median of the same image in the scene\'s other runs)\n')
fails = defaultdict(int)
for sc in scenes:
    keys = [k for k in runs if k[0] == sc]
    for k in keys:
        for i, v in runs[k]['per'][30000].items():
            others = [runs[j]['per'][30000][i] for j in keys if j != k and i in runs[j]['per'][30000]]
            if others and v < st.median(others) - SEVERE:
                fails[(sc, k[1])] += 1
                print(f'- {sc} {k[1]} s{k[2]}: image {i} {v - st.median(others):+.1f} dB')
print('\n' + '; '.join(f'{sc} {a}: {fails[(sc, a)]}' for sc in scenes for a in ARMS) + '\n')

# ---------------------------------------------------------------- decision (pre-registered)
print('## Decision (phase3/PLAN.md)\n')
valid = all(not errors[sc] for sc in scenes) and all(check1.get(sc) for sc in scenes) and len(scenes) == 2
if not valid:
    print('NOT EVALUATED: validation or check 1 failed (see above). Fix the cause before reading the arms.')
    sys.exit(2)
passed = {}
for a in CAND + ('oracle',):
    crit = {}
    for sc in scenes:
        d = summary[(sc, a)]
        dts = d['dts']
        crit[f'{sc}: median T reduction >= 10%'] = None not in dts and st.median(dts) >= 10
        crit[f'{sc}: mean ΔPSNR >= -0.10 dB'] = st.mean(d['dps']) >= -0.10
        crit[f'{sc}: mean ΔLPIPS <= +0.002'] = st.mean(d['dls']) <= 0.002
    faster = sum(x is not None and x > 0 for sc in scenes for x in summary[(sc, a)]['dts'])
    crit[f'T lower than ctl in >= 5 of 6 pairs ({faster})'] = faster >= 5
    crit['one-view failures not above ctl'] = sum(fails[(sc, a)] for sc in scenes) <= sum(fails[(sc, 'ctl')] for sc in scenes)
    passed[a] = all(crit.values())
    print(f'**{a}**{" (oracle: upper-bound reference, not a method)" if a == "oracle" else ""}: {"PASS" if passed[a] else "FAIL"}')
    for k, v in crit.items():
        print(f'  - {"meets" if v else "misses"} {k}')
if any(passed[a] for a in CAND):
    print('\nOUTCOME: a non-oracle arm passes -> Experiment 2 studies that mechanism as an adaptive, training-signal policy.')
elif passed['oracle']:
    print('\nOUTCOME: only the oracle passes -> Experiment 2 predicts the target count from training signals.')
else:
    print('\nOUTCOME: nothing passes -> population control after step 5000 is not the lever here; switch to selective update scheduling.')
