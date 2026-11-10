"""Experiment 1A: does counting (FGS_ABL=1) change the training clock? (phase3/EXP1A_PLAN.md, "Before launch")

    python phase3/check_x1atime.py phase2/logs/run_<job>

Runs ta_garden_{ctl,mcmc}_c{1,0}r{1,2}_s10 (build_plus.sbatch stage x1atime): resumed from the same state at 5000, to
8000, no evaluation. Gate: per arm, the mean training time 5000 -> 8000 (SEGCLOCK cum_s at 8000 minus the resumed
cum_s) with counting is within 2% of the mean without. Reported: each run's training time, elapsed wall time 5000 -> 8000
(FGS_WALL, includes callbacks), and the callback time measured with counting on (FGS_ABL_CB).
"""
import os
import re
import statistics as st
import sys

out = sys.argv[1]
ok, rows = {}, {}
for arm in ('ctl', 'mcmc'):
    for c in (1, 0):
        for r in (1, 2):
            n = f'ta_garden_{arm}_c{c}r{r}_s10'
            t = open(f'{out}/{n}.seg').read() if os.path.exists(f'{out}/{n}.seg') else ''
            r0 = re.search(r'^FGS_RESUME step=5000 N=\d+ cum_s=(\S+)', t, re.M)
            r1 = re.search(r'^SEGCLOCK step=8000 \S+ cum_s=(\S+)', t, re.M)
            w = re.search(r'^FGS_WALL from=5000 to=8000 s=(\S+)', t, re.M)
            cb = re.search(r'^FGS_ABL_CB total_s=(\S+) by=(\S+) calls=(\d+)', t, re.M)
            rows[(arm, c, r)] = (float(r1.group(1)) - float(r0.group(1)) if r0 and r1 else None,
                                 float(w.group(1)) if w else None, cb.groups() if cb else None)
            ok[f'{n}: training time and wall time present'] = rows[(arm, c, r)][0] is not None and rows[(arm, c, r)][1] is not None
            if c == 1:
                ok[f'{n}: FGS_ABL_CB present, callback time > 0 and below the wall time'] = bool(
                    cb and 0 < float(cb.group(1)) < (rows[(arm, c, r)][1] or 0))
            else:
                ok[f'{n}: no FGS_ABL_CB line without counting'] = cb is None
print('| run | training 5000->8000 (s) | wall 5000->8000 (s) | callbacks (s, by callback) |\n|---|---|---|---|')
for (arm, c, r), (tr, w, cb) in sorted(rows.items()):
    print(f'| {arm} counting={c} rep {r} | {tr if tr is None else f"{tr:.3f}"} | {w if w is None else f"{w:.3f}"} | '
          f'{"-" if not cb else cb[0] + " (" + cb[1] + ", " + cb[2] + " calls)"} |')
for arm in ('ctl', 'mcmc'):
    on = [rows[(arm, 1, r)][0] for r in (1, 2)]
    off = [rows[(arm, 0, r)][0] for r in (1, 2)]
    if None in on + off:
        ok[f'{arm}: training time with counting within 2% of without'] = False
        continue
    d = st.mean(on) / st.mean(off) - 1
    print(f'{arm}: training time with counting {st.mean(on):.3f} s, without {st.mean(off):.3f} s ({100 * d:+.2f}%); '
          f'spread within each condition {abs(on[0] - on[1]):.3f} / {abs(off[0] - off[1]):.3f} s')
    ok[f'{arm}: training time with counting within 2% of without ({100 * d:+.2f}%)'] = abs(d) <= 0.02
for k, v in ok.items():
    print(('PASS ' if v else 'FAIL ') + k)
sys.exit(0 if all(ok.values()) else 1)
