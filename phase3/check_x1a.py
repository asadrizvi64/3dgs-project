"""Experiment 1A smoke test of the ablation switches (phase3/EXP1A_PLAN.md, "Before launch").

    python phase3/check_x1a.py phase2/logs/run_<job> phase2/logs/run_4559237

Runs (build_plus.sbatch stage x1asmoke), all resumed from Experiment 1R's saved garden seed-10 state at 5000 (job
4559237), with FGS_ABL=1, evaluation every 1000 from 5000 with FGS_EVAL_CHECK, and view traces:
  sa_garden_ctl_s10, sa_garden_creset_s10                      5000 -> 13000
  sa_garden_{mcmc,mrep,mreloc,mnoise,mreg,mlr}_s10             5000 -> 7000
Gates:
  defaults unchanged: ctl (no switch set) sees exactly the training views of Experiment 1R's ctl over 5000-12999, and
    ctl and mcmc loaded exactly the saved state;
  every run: every evaluation restored the random states, model / optimizer state, flag and mode;
  counters and switches per arm (EXP1A_PLAN.md, check 2), N held in the MCMC arms, settings in effect per arm.
Diagnostics: mcmc / mrep / creset view traces against the corresponding references, relocated and dead counts, N and
PSNR at the end.
"""
import os
import re
import sys

out, ref = sys.argv[1], sys.argv[2]
N = lambda a: f'sa_garden_{a}_s10'
seg = lambda p, n: open(f'{p}/{n}.seg').read() if os.path.exists(f'{p}/{n}.seg') else ''
views = lambda p, n, lo, hi: ([l for l in open(f'{p}/{n}.views').read().split('\n') if l and lo <= int(l.split()[0]) < hi]
                              if os.path.exists(f'{p}/{n}.views') else None)
MC = ('mcmc', 'mrep', 'mreloc', 'mnoise', 'mreg', 'mlr')
END = {a: 13000 if a in ('ctl', 'creset') else 7000 for a in MC + ('ctl', 'creset')}
SW = {'mreloc': (0, 1, 1), 'mnoise': (1, 0, 1), 'creset': (1, 1, 0)}
ok = {}
sv = re.search(r'^FGS_SAVE_STATE step=5000 N=(\d+) \S+ param_sha=(\S+) optim_sha=(\S+) dinfo_sha=(\S+)', seg(ref, 'x_garden_pfx_s10'), re.M)
ok['reference: Experiment 1R garden seed-10 FGS_SAVE_STATE found'] = bool(sv)
for a in ('ctl', 'creset') + MC:
    t = seg(out, N(a))
    r = re.search(r'^FGS_RESUME step=(\d+) N=(\d+) \S+ \S+ param_sha=(\S+) optim_sha=(\S+) dinfo_sha=(\S+) .*use_mcmc=(\S+) max_primitives=(\S+) '
                  r'densification_end=(\S+) lr=(\S+) lambda_opacity=(\S+) lambda_scale=(\S+)', t, re.M)
    ab = re.search(r'^FGS_ABL relocate=(\d) noise=(\d) reset=(\d) use_mcmc=(\S+) relocation_events=(\d+) relocated=(\d+) dead_left=(\d+) '
                   r'noise_steps=(\d+) resets=(\S+)', t, re.M)
    ch = re.findall(r'^FGS_EVALCHECK step=(\d+) guard=1 rng_before=(\S+) rng_after_render=\S+ rng_after_restore=(\S+) state_before=(\S+) '
                    r'state_after=(\S+) training_before=(\S+) training_after_render=\S+ training_after_restore=(\S+) mode_before=(\S+) '
                    r'mode_after_render=\S+ mode_after_restore=(\S+)', t, re.M)
    ok[f'{a}: resumed at 5000, FGS_ABL line present'] = bool(r and ab and r.group(1) == '5000')
    if not (r and ab and sv):
        continue
    mc = a in MC
    ok[f'{a}: loaded exactly the saved state (N, parameters, optimizer{", densification statistics" if not mc else "; densification statistics dropped"})'] = (
        r.group(2, 3, 4) == sv.group(1, 2, 3) and r.group(5) == ('none' if mc else sv.group(4)))
    ok[f'{a}: evaluations at 5000..{END[a] - 1000} all restored random states, state, flag and mode'] = (
        [int(c[0]) for c in ch] == list(range(5000, END[a], 1000)) and all(c[1] == c[2] and c[3] == c[4] and c[5] == c[6] and c[7] == c[8] for c in ch))
    ok[f'{a}: switches (relocate, noise, reset) = {SW.get(a, (1, 1, 1))}'] = tuple(int(x) for x in ab.group(1, 2, 3)) == SW.get(a, (1, 1, 1))
    resets = [] if ab.group(9) == 'none' else [int(x) for x in ab.group(9).split(',')]
    lr = dict(x.split(':') for x in r.group(9).split(','))
    if mc:
        ok[f'{a}: noise steps {ab.group(8)} == {0 if a == "mnoise" else 2000}'] = int(ab.group(8)) == (0 if a == 'mnoise' else 2000)
        ok[f'{a}: relocation events {ab.group(5)} {"== 0" if a == "mreloc" else "> 0"}'] = (int(ab.group(5)) == 0) == (a == 'mreloc')
        ok[f'{a}: no opacity resets'] = resets == []
        nn = [int(x) for x in re.findall(r'SEGCLOCK step=\d+ .* N=(\d+) peak', t)]
        ok[f'{a}: N held at N at the branch ({sv.group(1)})'] = bool(nn) and all(x == int(sv.group(1)) for x in nn)
        want = (0.025 if a == 'mlr' else 0.05, 0.0 if a == 'mreg' else 0.01)
        ok[f'{a}: in effect: MCMC on, cap N, opacity lr {want[0]}, regularisation {want[1]} / {want[1]}'] = (
            r.group(6) == 'True' and r.group(7) == sv.group(1) and float(lr['opacities']) == want[0]
            and float(r.group(10)) == want[1] and float(r.group(11)) == want[1])
    else:
        ok[f'{a}: opacity resets at {resets or "none"}, expected {"none" if a == "creset" else [6000, 9000, 12000]}'] = resets == ([] if a == 'creset' else [6000, 9000, 12000])
        ok[f'{a}: no noise steps, no relocation'] = int(ab.group(8)) == 0 and int(ab.group(5)) == 0
        ok[f'{a}: in effect: MCMC off, opacity lr 0.025, regularisation 0 / 0'] = (
            r.group(6) == 'False' and float(lr['opacities']) == 0.025 and float(r.group(10)) == 0.0 and float(r.group(11)) == 0.0)
    nend = re.findall(r'N=(\d+) peak', t)[-1:]
    print(f'{a}: relocation events {ab.group(5)}, relocated {ab.group(6)}, dead left {ab.group(7)}, noise steps {ab.group(8)}, '
          f'resets {ab.group(9)}; N at end {nend}')
# defaults unchanged: ctl's views are Experiment 1R's ctl's
c, rc = views(out, N('ctl'), 5000, 13000), views(ref, 'x_garden_ctl_s10', 5000, 13000)
first = None if c is None or rc is None else next((5000 + i for i, (x, y) in enumerate(zip(c, rc)) if x != y), None)
ok[f'defaults unchanged: ctl views 5000-12999 identical to Experiment 1R ctl seed 10 (first difference: {first})'] = (
    c is not None and rc is not None and len(c) == 8000 and c == rc)
for a, r_, lab in (('mcmc', 'x_garden_mcmc_s10', 'Experiment 1R mcmc'), ('mrep', None, 'mcmc in this job'), ('creset', None, 'ctl in this job')):
    hi = 7000 if a != 'creset' else 13000
    v = views(out, N(a), 5000, hi)
    w = views(ref, r_, 5000, hi) if r_ else views(out, N('mcmc' if a == 'mrep' else 'ctl'), 5000, hi)
    d = None if v is None or w is None else next((5000 + i for i, (x, y) in enumerate(zip(v, w)) if x != y), 'none')
    print(f'diagnostic: {a} views vs {lab}: first difference at step {d}')
for k, v in ok.items():
    print(('PASS ' if v else 'FAIL ') + k)
sys.exit(0 if all(ok.values()) else 1)
