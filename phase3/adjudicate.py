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


ctl, part1 = {}, []
for p in args.ctl:
    r, text = results(p, r'x_room_ctl_s(\d+)')
    part1 += re.findall(r'^(?:PASS|FAIL) part 1: .*$', text, re.M)
    for s, v in r.items():
        ctl[s] = v[-1]
ref = {}
for p in args.ref:
    r, text = results(p, r'b_room_up_s(\d+)')
    part1 += re.findall(r'^(?:PASS|FAIL) part 1: .*$', text, re.M)
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
sys.exit(0 if p1 and p2 else 1)
