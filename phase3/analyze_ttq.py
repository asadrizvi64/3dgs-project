"""Phase 3 Experiment 1 analysis: time to quality per arm, with the pre-registered decision rule (phase3/PLAN.md).

    python phase3/analyze_ttq.py phase2/logs/gs-x1-garden-<job>.out phase2/logs/gs-x1-room-<job>.out \
        [--ref phase2/logs/gs-up-garden-<job>.out phase2/logs/gs-up-room-<job>.out] [--seeds 0 1 2] \n        [--rerun phase2/logs/gs-x1-room-s1-<job>.out ...]

Inputs: the exp1 job logs (build_plus.sbatch stage exp1: run blocks '##### x_<scene>_<arm>_s<seed>:', each with the
clock hook's lines) and the archived per-step test metrics phase2/logs/run_<job>/runs/<run>/test_<i>/metrics_8bit.txt.
--ref: the upbench logs with uninterrupted FasterGSUp runs (arm 'up'), for check 1.

Two clocks are reported: training-only time (the T values and the decision) and, descriptively, the elapsed wall time
from the branch to 30000 including the 26 test evaluations and all callbacks (FGS_WALL), so monitoring cost is visible.
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
ap.add_argument('--rerun', nargs='*', default=[], help='rerun logs: every (scene, seed) they contain replaces that whole block (prefix and all arms) of the main logs')
ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
ap.add_argument('--branch', type=int, default=5000)
ap.add_argument('--descriptive', action='store_true', help='if validation or check 1 failed, still print what the rule would give, labelled as not the decision')
ap.add_argument('--exp1r', action='store_true', help='Experiment 1R: check 1 = exact checks 1a-1d (gates) + paired report 1e, with '
                'the references b_<scene>_up_s<seed> from the same logs (phase3/EXP1R_PLAN.md)')
ap.add_argument('--end', type=int, default=30000, help='last step (only changed for the exp1r stage smoke test)')
ap.add_argument('--pilot-sd', default='garden=0.0504,room=0.0664',
                help='same-seed PSNR SD of the difference from the noise pilot (phase3/noise-pilot-analysis.md), for 1e')
ap.add_argument('--checks-only', action='store_true', help='stop after validation and check 1 (stage smoke test)')
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
END = args.end
MID = 15000 if END == 30000 else END   # check 2's mid-run point
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
if args.rerun:
    rb, ra = blocks_of(args.rerun, r'##### x_([a-z]+)_([a-z]+)_s(\d+):')
    rep = sorted({(k[0], k[2]) for k in rb})
    for k in [k for k in blocks if (k[0], k[2]) in rep]:
        del blocks[k]
        arch.pop(k, None)
    blocks.update(rb)
    arch.update(ra)
    print('REPLACED from rerun logs (whole scene-seed blocks): ' + ', '.join(f'{sc} s{sd}' for sc, sd in rep) + '\n')
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
    steps = list(range(args.branch, END + 1, 1000))
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
        wall = [re.match(rf'FGS_WALL from={args.branch} to={END} s=(\S+)', l) for l in L]
        wall = [float(m.group(1)) for m in wall if m]
        runs[(sc, arm, seed)] = dict(clk=clk, curve=curve, per=per, n_end=int(end[0].group(1)), peak=float(end[0].group(2)),
                                    resume=res[0].group(0), wall=wall[0] if len(wall) == 1 else None, lines=L)

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
            if not g('ctl')['clk'][MID][1] > g('ctl')['clk'][args.branch][1]:
                errors[sc].append(f's{s} ctl: no growth after the branch')
        if g('stop'):
            # densification at the branch step itself still runs (found in the smoke test, phase3/PLAN.md), so N must be
            # constant from the first evaluation point after the branch on
            ns = [g('stop')['clk'][k][1] for k in sorted(g('stop')['clk']) if k >= args.branch + 1000]
            if len(set(ns)) != 1:
                errors[sc].append(f's{s} stop: N changed after step {args.branch + 1000}: {sorted(set(ns))[:4]}')
        if g('mcmc'):
            n0 = g('mcmc')['clk'][args.branch][1]
            if any(v[1] != n0 for k, v in g('mcmc')['clk'].items() if k >= args.branch):
                errors[sc].append(f's{s} mcmc: N not held at {n0}')
        if g('oracle') and g('ctl') and END == 30000:   # a short smoke run need not reach the cap
            cap = runs[(sc, 'ctl', s)]['n_end']
            if g('oracle')['clk'][15000][1] != cap:
                errors[sc].append(f's{s} oracle: N at 15000 {g("oracle")["clk"][15000][1]} != cap {cap}')
for sc in scenes:
    print(f'- {sc}: ' + ('all checks pass' if not errors[sc] else f'{len(errors[sc])} problem(s)'))
    for e in errors[sc]:
        print(f'  - {e}')

def yaml_flat(path):
    """training_config.yaml -> {dotted key: normalised value}; comments dropped, numeric underscores removed."""
    out, stack = {}, []
    for line in open(path, encoding='utf-8'):
        m = re.match(r'^(\s*)([A-Za-z0-9_]+):\s*(.*?)\s*(#.*)?$', line.rstrip('\n'))
        if not m:
            continue
        ind = len(m.group(1))
        while stack and stack[-1][0] >= ind:
            stack.pop()
        stack.append((ind, m.group(2)))
        v = m.group(3)
        if v:
            try:
                v = repr(float(v.replace('_', '')))
            except ValueError:
                pass
            out['.'.join(k for _, k in stack)] = v
    return out


def ydiff(a, b):
    return {k.split('.')[-1] for k in set(a) | set(b) if a.get(k) != b.get(k)}


ALLOWED = {'ctl': {'MODEL_NAME'}, 'stop': {'MODEL_NAME', 'DENSIFICATION_END_ITERATION'},
           'mcmc': {'MODEL_NAME', 'USE_MCMC', 'MAX_PRIMITIVES', 'LAMBDA_OPACITY_REGULARIZATION', 'LAMBDA_SCALE_REGULARIZATION',
                    'LEARNING_RATE_OPACITIES'}}
ALLOWED['oracle'] = ALLOWED['mcmc']
EVALCHECK = re.compile(r'^FGS_EVALCHECK step=(\d+) guard=1 rng_before=(\S+) rng_after_render=(\S+) rng_after_restore=(\S+) '
                       r'state_before=(\S+) state_after=(\S+) training_before=(\S+) training_after_render=(\S+) '
                       r'training_after_restore=(\S+) mode_before=(\S+) mode_after_render=(\S+) mode_after_restore=(\S+)')
RESUME = re.compile(r'FGS_RESUME step=(\d+) N=(\d+) \S+ \S+ param_sha=(\S+) optim_sha=(\S+) dinfo_sha=(\S+) .*use_mcmc=(\S+) '
                    r'max_primitives=(\S+) densification_end=(\S+) lr=(\S+) lambda_opacity=(\S+) lambda_scale=(\S+)')
RESULT = re.compile(r'RESULT \S+ PSNR:(\S+) .*N_Gaussians:(\d+)')


def exp1r_views(arch_dir, name, lo, hi):
    f = os.path.join(os.path.dirname(arch_dir), f'{name}.views')
    if not os.path.exists(f):
        return None
    return [l for l in open(f).read().split('\n') if l and lo <= int(l.split()[0]) < hi]


def exp1r_check1():
    """Experiment 1R check 1 (phase3/EXP1R_PLAN.md): 1a-1d exact gates, 1e paired report against the noise pilot."""
    ub, uarch = blocks_of(args.logs, r'##### b_([a-z]+)_(up)_s(\d+):')
    sd = {k: float(v) for k, v in (x.split('=') for x in args.pilot_sd.split(','))}
    order = {}
    for path in args.logs:
        for m in re.finditer(r'^EXP1R_ORDER (\S+) s(\d+) (.*)$', open(path, encoding='utf-8', errors='replace').read(), re.M):
            order[(m.group(1), int(m.group(2)))] = m.group(3).split()
    cfg = lambda arch_dir, name: yaml_flat(os.path.join(arch_dir, name, 'training_config.yaml'))
    print('\n## Check 1 (Experiment 1R): exact gates 1a-1d, paired report 1e\n')
    res = {}
    for sc in scenes:
        bad, rep = [], []
        for s in args.seeds:
            key, pk = (sc, 'up', str(s)), (sc, 'pfx', str(s))
            if key not in ub or (sc, 'ctl', s) not in runs or pk not in blocks:
                bad.append(f's{s}: reference, prefix or ctl missing')
                continue
            ra, xa = uarch[key], arch[(sc, 'ctl', str(s))]
            # 1a: prefix (0 .. branch-1) + ctl (branch .. END-1) == reference (0 .. END-1)
            u = exp1r_views(ra, f'b_{sc}_up_s{s}', 0, END)
            p = exp1r_views(arch[pk], f'x_{sc}_pfx_s{s}', 0, args.branch)
            c = exp1r_views(xa, f'x_{sc}_ctl_s{s}', args.branch, END)
            if None in (u, p, c):
                bad.append(f's{s} 1a: view trace missing (reference {u is not None}, prefix {p is not None}, ctl {c is not None})')
            elif len(u) != END or p + c != u:
                first = next((i for i, (a, b) in enumerate(zip(p + c, u)) if a != b), min(len(p + c), len(u)))
                bad.append(f's{s} 1a: training views differ from the reference ({len(p + c)} vs {len(u)} lines; first difference at line {first})')
            sv = [re.match(r'FGS_SAVE_STATE step=(\d+) N=(\d+) \S+ param_sha=(\S+) optim_sha=(\S+) dinfo_sha=(\S+)', l) for l in blocks[pk][-1]]
            sv = [m for m in sv if m]
            for a in ARMS:
                r = runs.get((sc, a, s))
                if not r:
                    bad.append(f's{s}: {a} missing')
                    continue
                # 1b: every branched arm loaded exactly the saved state (densification statistics dropped for MCMC)
                m = [x for x in (RESUME.match(l) for l in r['lines']) if x]
                if len(sv) != 1 or len(m) != 1 or not (m[0].group(1) == sv[0].group(1) == str(args.branch)):
                    bad.append(f's{s} 1b: {a}: no single matching FGS_SAVE_STATE / FGS_RESUME at {args.branch}')
                    continue
                m = m[0]
                want = sv[0].group(2, 3, 4) + (('none',) if a in ('mcmc', 'oracle') else (sv[0].group(5),))
                if m.group(2, 3, 4, 5) != want:
                    bad.append(f's{s} 1b: {a}: loaded state differs from the saved one ({m.group(2, 3, 4, 5)} vs {want})')
                # 1c: every in-training evaluation left random states, model / optimizer state, flag and mode as they were
                ch = [x.groups() for x in (EVALCHECK.match(l) for l in r['lines']) if x]
                if [int(x[0]) for x in ch] != list(range(args.branch, END, 1000)):
                    bad.append(f's{s} 1c: {a}: evaluation checks at {[int(x[0]) for x in ch][:4]}..., expected every 1000 from {args.branch} to {END - 1000}')
                elif not all(x[1] == x[3] and x[4] == x[5] and x[6] == x[8] and x[9] == x[11] for x in ch):
                    bad.append(f's{s} 1c: {a}: an evaluation changed the random states, the model or optimizer state, or the flag / mode')
                # 1d: ctl's configuration equals the reference's except the name; the other arms differ from ctl only in
                # their own keys; the settings in effect after loading match the arm's configuration
                try:
                    ca, cc = cfg(arch[(sc, a, str(s))], f'x_{sc}_{a}_s{s}'), cfg(xa, f'x_{sc}_ctl_s{s}')
                    d = ydiff(ca, cfg(ra, f'b_{sc}_up_s{s}') if a == 'ctl' else cc)
                    if not d <= ALLOWED[a]:
                        bad.append(f's{s} 1d: {a}: configuration differs in {sorted(d - ALLOWED[a])}')
                    get = lambda k: next(v for kk, v in ca.items() if kk.endswith('.' + k))
                    lr = dict(x.split(':') for x in m.group(9).split(','))
                    eff = {'USE_MCMC': str(m.group(6) == 'True').lower(), 'DENSIFICATION_END_ITERATION': repr(float(m.group(8))),
                           'LEARNING_RATE_OPACITIES': repr(float(lr['opacities'])),
                           'LAMBDA_OPACITY_REGULARIZATION': repr(float(m.group(10))), 'LAMBDA_SCALE_REGULARIZATION': repr(float(m.group(11)))}
                    if a in ('mcmc', 'oracle'):
                        eff['MAX_PRIMITIVES'] = repr(float(m.group(7)))
                    wrong = {k: (v, get(k)) for k, v in eff.items() if v != get(k)}
                    if wrong:
                        bad.append(f's{s} 1d: {a}: settings in effect after loading differ from its configuration (in effect, configured): {wrong}')
                except (OSError, StopIteration) as e:
                    bad.append(f's{s} 1d: {a}: configuration not readable ({e!r})')
            # 1d, prefix: differs from the reference only in its name and length; NUM_ITERATIONS has no effect before
            # the branch while PPISP and the 3D filter are off (checked here)
            try:
                cp_, cu_ = cfg(arch[pk], f'x_{sc}_pfx_s{s}'), cfg(ra, f'b_{sc}_up_s{s}')
                d = ydiff(cp_, cu_) - {'MODEL_NAME', 'NUM_ITERATIONS'}
                on = [k for k, v in cu_.items() if (k.endswith('PPISP.USE') or k.endswith('FILTER_3D.USE')) and v != 'false']
                if d or on:
                    bad.append(f's{s} 1d: prefix: configuration differs from the reference in {sorted(d)}; PPISP / 3D filter on: {on}')
            except OSError as e:
                bad.append(f's{s} 1d: prefix: configuration not readable ({e!r})')
            # 1e: paired ctl - reference at full precision, reported only
            up = [x for x in (RESULT.match(l) for l in ub[key][-1]) if x]
            cr = [x for x in (RESULT.match(l) for l in runs[(sc, 'ctl', s)]['lines']) if x]
            if up and cr:
                dp, dn = float(cr[0].group(1)) - float(up[0].group(1)), int(cr[0].group(2)) - int(up[0].group(2))
                f = os.path.join(ra, f'b_{sc}_up_s{s}', f'test_{END}', 'metrics_8bit.txt')
                pv = ''
                if os.path.exists(f):
                    uper, cper = metrics(f)[1], runs[(sc, 'ctl', s)]['per'][END]
                    dd = sorted(((cper[i] - uper[i], i) for i in uper if i in cper), key=lambda x: -abs(x[0]))[:2]
                    pv = ', '.join(f'view {i} {v:+.2f}' for v, i in dd)
                o = order.get((sc, s), [])
                rep.append((s, dp, dn, int(up[0].group(2)), pv, 'up first' if o[:1] == ['up'] else 'up last' if o[-1:] == ['up'] else '?'))
        res[sc] = not bad
        print(f'- {sc}: 1a-1d ' + ('all pass' if not bad else f'{len(bad)} problem(s)'))
        for b in bad:
            print(f'  - {b}')
        if rep:
            print('\n  1e (reported, not a gate): ctl - reference, per seed\n')
            print('  | seed | order | d PSNR (dB) | d PSNR / pilot SD | d N | d N (% of N) | largest per-view d (dB) |')
            print('  |---|---|---|---|---|---|---|')
            for s, dp, dn, nu, pv, o in rep:
                z = f'{dp / sd[sc]:+.2f}' if sc in sd else 'n/a'
                print(f'  | {s} | {o} | {dp:+.4f} | {z} | {dn:+d} | {100 * dn / nu:+.3f}% | {pv} |')
            dps = [r[1] for r in rep]
            print(f'\n  mean d PSNR {st.mean(dps):+.4f} dB over {len(dps)} seed(s). Noise pilot, same-seed differences on this scene: '
                  f'SD {sd.get(sc, float("nan")):.4f} dB. Passing 1a-1d establishes implementation fidelity, not statistical '
                  'equivalence of final PSNR.\n')
    return res


if args.exp1r:
    check1 = exp1r_check1()
else:   # Experiment 1's check 1 (range rule)
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
        ctl = [(runs[(sc, 'ctl', s)]['curve'][END]['PSNR'], None) for s in args.seeds if (sc, 'ctl', s) in runs]
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
if args.checks_only:
    sys.exit(0 if all(not errors[sc] for sc in scenes) and all(check1.get(sc) for sc in scenes) else 1)


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
    print('| arm | T per seed (s) | ΔT vs ctl per seed | final PSNR | ΔPSNR | ΔLPIPS | final N | train time to 30k (s) | elapsed 5k→30k incl. evaluations (s) | peak GiB |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for a in ARMS:
        ts, dts, dps, dls, rows = [], [], [], [], []
        for s in args.seeds:
            r, c = runs.get((sc, a, s)), runs.get((sc, 'ctl', s))
            if not r or not c:
                ts.append(None); continue
            tgt = c['curve'][END]['PSNR'] - DELTA
            t, tc = ttq(r, tgt), ttq(c, tgt)
            ts.append(t)
            dts.append(None if t is None or tc is None else 100 * (1 - t / tc))
            dps.append(r['curve'][END]['PSNR'] - c['curve'][END]['PSNR'])
            dls.append(r['curve'][END]['LPIPS'] - c['curve'][END]['LPIPS'])
            rows.append(r)
        if not rows:
            continue
        fmt = lambda v, f: ' / '.join('n/r' if x is None else f % x for x in v)
        walls = [r['wall'] for r in rows]
        wall_s = 'n/a' if None in walls else f'{st.mean(walls):.1f}'
        print(f'| {a} | {fmt(ts, "%.1f")} | {"" if a == "ctl" else fmt(dts, "%+.1f%%")} | '
              f'{st.mean(r["curve"][END]["PSNR"] for r in rows):.3f} | {"" if a == "ctl" else f"{st.mean(dps):+.3f}"} | '
              f'{"" if a == "ctl" else f"{st.mean(dls):+.4f}"} | {st.mean(r["n_end"] for r in rows) / 1e6:.3f} M | '
              f'{st.mean(r["clk"][END][0] for r in rows):.1f} | {wall_s} | {st.mean(r["peak"] for r in rows):.2f} |')
        summary[(sc, a)] = dict(dts=dts, dps=dps, dls=dls, n=len(rows))
    print()
    for d in SENS:   # sensitivity: other targets, and "reaches and stays"
        line = []
        for a in ARMS:
            v = []
            for s in args.seeds:
                r, c = runs.get((sc, a, s)), runs.get((sc, 'ctl', s))
                if r and c:
                    tgt = c['curve'][END]['PSNR'] - d
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
                t = ttq(r, c['curve'][END]['PSNR'] - DELTA, stay=True)
                v.append('n/r' if t is None else f'{t:.0f}')
        line.append(f'{a} {"/".join(v)}')
    print('- sensitivity, reaches −0.05 dB and stays: ' + '; '.join(line) + '\n')

# one-view failures at 30k (phase 2 definition, within each scene)
print('## One-view failures at 30k (a test image > 3 dB below the median of the same image in the scene\'s other runs)\n')
fails = defaultdict(int)
for sc in scenes:
    keys = [k for k in runs if k[0] == sc]
    for k in keys:
        for i, v in runs[k]['per'][END].items():
            others = [runs[j]['per'][END][i] for j in keys if j != k and i in runs[j]['per'][END]]
            if others and v < st.median(others) - SEVERE:
                fails[(sc, k[1])] += 1
                print(f'- {sc} {k[1]} s{k[2]}: image {i} {v - st.median(others):+.1f} dB')
print('\n' + '; '.join(f'{sc} {a}: {fails[(sc, a)]}' for sc in scenes for a in ARMS) + '\n')

# ---------------------------------------------------------------- decision (pre-registered)
print('## Decision (phase3/PLAN.md)\n')
valid = all(not errors[sc] for sc in scenes) and all(check1.get(sc) for sc in scenes) and len(scenes) == 2
if not valid:
    print('NOT EVALUATED: validation or check 1 failed (see above). Fix the cause before reading the arms.')
    if not args.descriptive:
        sys.exit(2)
    print('\n**Descriptive only (--descriptive): the rule applied anyway. This is NOT the pre-registered decision.**\n')
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
