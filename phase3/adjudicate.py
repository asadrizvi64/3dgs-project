"""Experiment 1 adjudication, part 2 (phase3/RESULTS.md, "Adjudication"): check 1 on room with all seeds.

    python phase3/adjudicate.py --ctl <exp1 room log> <rerun logs...> <adjudication logs...> \
        --ref <upbench room logs...> <adjudication logs...> [--seeds 0 1 2 3 4 5 6 7 8]

Same sources and criterion as check 1 in analyze_ttq.py:
- branched ctl final PSNR = the mean PSNR in run_<job>/runs/x_room_ctl_s<seed>/test_30000/metrics_8bit.txt,
  final N = the RESULT line's N_Gaussians;
- uninterrupted `up` final PSNR and N = the RESULT line of run b_room_up_s<seed>;
- pass when |mean ctl - mean up| <= the uninterrupted runs' range (max - min), for PSNR and for final N.
For --ctl, logs listed later replace whole seeds of earlier ones (the rerun rule). For --ref, every seed must
appear exactly once with a RESULT line.
Part 1 (identical view sequences with and without evaluation) is checked inside the job and printed as a PASS/FAIL
line; this script reports that line from the log that has it.
"""
import argparse
import os
import re
import statistics as st
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--ctl', nargs='+', required=True)
ap.add_argument('--ref', nargs='+', required=True)
ap.add_argument('--seeds', type=int, nargs='+', default=list(range(9)))
args = ap.parse_args()


def results(path, pat):
    """{seed: (RESULT line fields, archive dir)} for run names matching pat in one job log"""
    text = open(path, encoding='utf-8', errors='replace').read()
    job = re.search(r'^node=\S+ job=(\S+)', text, re.M).group(1)
    arch = os.path.join(os.path.dirname(path), f'run_{job}', 'runs')
    out = {}
    for m in re.finditer(rf'^RESULT {pat} PSNR:(\S+) .*?N_Gaussians:(\d+)', text, re.M):
        out.setdefault(int(m.group(1)), []).append((float(m.group(2)), int(m.group(3)), arch))
    return out, text


ctl, part1 = {}, {}   # part 1 lines keyed by log path: one log may be given in both --ctl and --ref
for p in args.ctl:
    r, text = results(p, r'x_room_ctl_s(\d+)')
    part1[os.path.abspath(p)] = re.findall(r'^(?:PASS|FAIL) part 1: .*$', text, re.M)
    for s, v in r.items():
        ctl[s] = v[-1]
ref = {}
for p in args.ref:
    r, text = results(p, r'b_room_up_s(\d+)')
    part1[os.path.abspath(p)] = re.findall(r'^(?:PASS|FAIL) part 1: .*$', text, re.M)
    for s, v in r.items():
        ref.setdefault(s, []).extend(v)

problems = [f'ctl s{s} missing' for s in args.seeds if s not in ctl]
problems += [f'up s{s}: {len(ref.get(s, []))} RESULT lines' for s in args.seeds if len(ref.get(s, [])) != 1]
cp, cn, up, un = [], [], [], []
for s in args.seeds:
    if s in ctl:
        f = os.path.join(ctl[s][2], f'x_room_ctl_s{s}', 'test_30000', 'metrics_8bit.txt')
        if not os.path.exists(f):
            problems.append(f'ctl s{s}: no {f}')
            continue
        cp.append(float(re.search(r'^PSNR\t(\S+)', open(f).read(), re.M).group(1)))
        cn.append(ctl[s][1])
    if len(ref.get(s, [])) == 1:
        up.append(ref[s][0][0]); un.append(ref[s][0][1])
print('## Adjudication\n')
part1 = [l for v in part1.values() for l in v]
print('Part 1: ' + (' / '.join(part1) if part1 else 'no PASS/FAIL line found'))
p1 = len(part1) == 1 and part1[0].startswith('PASS')
print('\nPart 2, check 1 on room, seeds ' + ' '.join(map(str, args.seeds)) + ':\n')
print('| seed | ctl PSNR | up PSNR | ctl N | up N |\n|---|---|---|---|---|')
for s in args.seeds:
    c = ctl.get(s); u = ref.get(s, [None])[0]
    print(f'| {s} | {cp[args.seeds.index(s)]:.3f} | ' if c and len(cp) == len(args.seeds) else f'| {s} | — | ', end='')
    print(f'{u[0]:.3f} | ' if u else '— | ', end='')
    print(f'{c[1]} | ' if c else '— | ', end='')
    print(f'{u[1]} |' if u else '— |')
if problems:
    print('\nNOT EVALUABLE: ' + '; '.join(problems))
    sys.exit(1)
ok_p = abs(st.mean(cp) - st.mean(up)) <= max(up) - min(up)
ok_n = abs(st.mean(cn) - st.mean(un)) <= max(un) - min(un)
print(f'\nPSNR: ctl {st.mean(cp):.3f} vs up {st.mean(up):.3f}, difference {st.mean(cp) - st.mean(up):+.3f}, '
      f'allowed {max(up) - min(up):.3f} -> {"pass" if ok_p else "FAIL"}')
print(f'N: ctl {st.mean(cn) / 1e6:.3f} M vs up {st.mean(un) / 1e6:.3f} M, difference {(st.mean(cn) - st.mean(un)) / 1e6:+.3f} M, '
      f'allowed {(max(un) - min(un)) / 1e6:.3f} M -> {"pass" if ok_n else "FAIL"}')
p2 = ok_p and ok_n
print(f'\nPart 1 {"PASS" if p1 else "FAIL"}, part 2 {"PASS" if p2 else "FAIL"}: '
      + ('apply the original decision rule to the original 24 continuations.' if p1 and p2
         else 'Experiment 1 stays invalid; the arms are not compared.'))

# ------------------------------------------------------------ supplementary diagnostics (declared 2026-11-09, not the rule)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def per_image(arch, name, it=30000):
    t = open(os.path.join(arch, name, f'test_{it}', 'metrics_8bit.txt')).read().partition('Image')[2]
    return {int(m.group(1)): float(m.group(2)) for m in re.finditer(r'^(\d+)\t([\d.]+)', t, re.M)}


print('\n## Supplementary diagnostics (declared before results; they do not change the rule above)\n')
# part 1: final state of the two 8000-step runs
p1r = {}
for p in args.ctl + args.ref:
    text = open(p, encoding='utf-8', errors='replace').read()
    job = re.search(r'^node=\S+ job=(\S+)', text, re.M).group(1)
    for m in re.finditer(r'^RESULT a_room_(ev|ne)_s0 PSNR:(\S+) SSIM:(\S+) LPIPS:(\S+) N_Gaussians:(\d+)', text, re.M):
        p1r[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5)),
                           os.path.join(os.path.dirname(p), f'run_{job}', 'runs'))
if len(p1r) == 2:
    e, n = p1r['ev'], p1r['ne']
    print('Part 1 at step 8000 (with evaluation / without):')
    print(f'- N {e[3]} / {n[3]} (difference {e[3] - n[3]:+d}); PSNR {e[0]:.3f} / {n[0]:.3f} ({e[0] - n[0]:+.3f}); '
          f'SSIM {e[1]:.4f} / {n[1]:.4f}; LPIPS {e[2]:.4f} / {n[2]:.4f}')
    ie, ine = per_image(e[4], 'a_room_ev_s0', 8000), per_image(n[4], 'a_room_ne_s0', 8000)
    dd = {i: ie[i] - ine[i] for i in ie if i in ine}
    w = max(dd, key=lambda i: abs(dd[i]))
    print(f'- per-view PSNR difference: largest {dd[w]:+.2f} dB (view {w}); views 34 / 35: {dd.get(34, 0):+.2f} / {dd.get(35, 0):+.2f}')
    print('- Two runs without evaluation would also differ here (atomic-addition order), and no such pair exists at 8000,'
          ' so these numbers have no noise reference. Descriptive only.')
else:
    print('Part 1 at step 8000: runs a_room_ev_s0 / a_room_ne_s0 not found')
# part 2: differences paired by seed (ctl and up share the seed and, up to the branch, the view sequence)
print('\nPart 2, ctl minus up paired by seed, full-precision final PSNR (RESULT line), and per view for 34 and 35:\n')
print('| seed | ΔPSNR | Δ without views 34, 35 | Δ view 34 | Δ view 35 |\n|---|---|---|---|---|')
d, dx = [], []
for s in args.seeds:
    ic, iu = per_image(ctl[s][2], f'x_room_ctl_s{s}'), per_image(ref[s][0][2], f'b_room_up_s{s}')
    keep = [i for i in ic if i not in (34, 35)]
    d.append(ctl[s][0] - ref[s][0][0])
    dx.append(st.mean(ic[i] for i in keep) - st.mean(iu[i] for i in keep))
    print(f'| {s} | {d[-1]:+.3f} | {dx[-1]:+.3f} | {ic[34] - iu[34]:+.2f} | {ic[35] - iu[35]:+.2f} |')
for label, v in (('ΔPSNR', d), ('Δ without views 34, 35 (per-view values at 0.01 dB)', dx)):
    h = T975[len(v) - 1] * st.stdev(v) / len(v) ** 0.5
    print(f'- {label}: mean {st.mean(v):+.3f} dB, 95% t interval [{st.mean(v) - h:+.3f}, {st.mean(v) + h:+.3f}] '
          f'(n={len(v)}), {sum(x < 0 for x in v)} of {len(v)} seeds negative')
print('- The rule\'s allowed difference is the range of the uninterrupted runs, which tends to grow with the number of'
      ' seeds; the interval above shows whether a systematic gap remains.')
sys.exit(0 if p1 and p2 else 1)
