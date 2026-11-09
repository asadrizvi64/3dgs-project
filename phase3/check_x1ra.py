"""Experiment 1R smoke test A, repeated (phase3/EXP1R_PLAN.md, "Revisions after the smoke test and pilot").

    python phase3/check_x1ra.py phase2/logs/run_<job>

Six resumes of the same saved room seed-0 state at 5000, each evaluated once at 5000 (build_plus.sbatch stage x1ra):
ra_room_{guard,guard2,noguard,noguard2,nobench,nobench2}_s0.

Gate (revised criterion, set after smoke job 4556213 failed the bit-identity version):
- every guard-on / guard-off pair (4 pairs): per-image rows identical at their output precision, and full-precision
  means within PSNR 1e-5 dB, SSIM 1e-6, LPIPS 1e-6.
Reported, not gated: the same differences within each condition (guard / guard2, noguard / noguard2, nobench /
nobench2) and with benchmark off; FGS_TRAIN_END peak allocated memory per run (never reset, so it includes the
evaluation at 5000) with cudnn.benchmark on and off.
"""
import os
import re
import sys

out = sys.argv[1]
TOL = {'PSNR': 1e-5, 'SSIM': 1e-6, 'LPIPS': 1e-6}
N = lambda r: f'ra_room_{r}_s0'
RUNS = ('guard', 'guard2', 'noguard', 'noguard2', 'nobench', 'nobench2')


def metrics(r):
    f = f'{out}/runs/{N(r)}/test_5000/metrics_8bit.txt'
    if not os.path.exists(f):
        return None
    t = open(f).read()
    rows = re.findall(r'^\d+\t.*$', t, re.M)
    m = re.search(r'^PSNR:(\S+) SSIM:(\S+) LPIPS:(\S+)', t, re.M)
    return rows, dict(zip(('PSNR', 'SSIM', 'LPIPS'), map(float, m.groups()))) if m else None


ok, M = {}, {r: metrics(r) for r in RUNS}
for r in RUNS:
    ok[f'{N(r)}: test_5000 metrics present with full-precision means'] = bool(M[r] and M[r][1])


def diff(a, b):
    if not (M[a] and M[b] and M[a][1] and M[b][1]):
        return None
    return {k: abs(M[a][1][k] - M[b][1][k]) for k in TOL}


fmt = lambda d: 'n/a' if d is None else ' '.join(f'{k} {v:.2e}' for k, v in d.items())
for g in ('guard', 'guard2'):
    for n in ('noguard', 'noguard2'):
        d = diff(g, n)
        ok[f'A: {g} vs {n}: per-image rows identical at output precision'] = bool(M[g] and M[n]) and M[g][0] == M[n][0] and len(M[g][0]) > 0
        ok[f'A: {g} vs {n}: full-precision means within tolerance ({fmt(d)})'] = d is not None and all(d[k] <= TOL[k] for k in TOL)
print('Within-condition differences (reported, not gated):')
for a, b in (('guard', 'guard2'), ('noguard', 'noguard2'), ('nobench', 'nobench2'), ('guard', 'nobench'), ('guard2', 'nobench2')):
    print(f'  {a} vs {b}: {fmt(diff(a, b))}; per-image rows identical: {bool(M[a] and M[b]) and M[a][0] == M[b][0]}')
print('Memory (FGS_TRAIN_END peak allocated, never reset; includes the evaluation at 5000):')
for r in RUNS:
    s = f'{out}/{N(r)}.seg'
    m = re.search(r'^FGS_TRAIN_END N=\d+ peak_alloc_gib=(\S+) peak_reserved_gib=(\S+)', open(s).read(), re.M) if os.path.exists(s) else None
    print(f'  {N(r)}: cudnn.benchmark {"off" if r.startswith("nobench") else "on"}: '
          + (f'peak allocated {m.group(1)} GiB, peak reserved {m.group(2)} GiB' if m else 'no FGS_TRAIN_END line'))
for k, v in ok.items():
    print(('PASS ' if v else 'FAIL ') + k)
sys.exit(0 if all(ok.values()) else 1)
