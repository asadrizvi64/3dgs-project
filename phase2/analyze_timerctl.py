"""Timer on/off control (Phase 2 step 4b): stock vs A+B+C, NeRFICG's per-step timer ON (benchmark protocol) vs OFF.

    python phase2/analyze_timerctl.py phase2/logs/gs-timer-*.out [--seeds 0 1 2]

Validation first (exit 1 if anything fails):
  - each scene x arm (stock, abc) x timer (on, off) x seed has exactly one run, with no FAILED line;
  - each run has FGS_WALL lines for 0-15000 and 15000-30000 and one SEGCLOCK step=30000 record, and a RESULT line;
  - timer-ON runs have NeRFICG's ms/it (timings.txt written); timer-OFF runs have none (timer really off);
  - for each seed, all four runs of a scene saw the same training views (equal view hashes at 15000 and 30000).

Reported per scene (means of the seeds, per seed in brackets):
  - unprofiled wall time over 0-30k (device synced only at 0, 15k and 30k; includes everything between steps), its two
    halves, and the summed per-step CUDA-event time (SEGCLOCK, the benchmark's clock);
  - paired effects: timer OFF vs ON within each arm, and A+B+C vs stock within each timer setting;
  - test PSNR, to confirm the timer setting does not change results beyond run-to-run noise.
"""
import argparse
import math
import re
import statistics as st
import sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('logs', nargs='+')
ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
args = ap.parse_args()
sys.stdout.reconfigure(encoding='utf-8')

ARMS, TIMERS = ('stock', 'abc'), ('on', 'off')
LABEL = {'stock': 'stock Faster-GS', 'abc': 'A+B+C'}
runs, errors, jobs = {}, defaultdict(list), {}
blocks = defaultdict(list)
for path in args.logs:
    text = open(path, encoding='utf-8', errors='replace').read()
    mj = re.search(r'^node=(\S+) job=(\S+)', text, re.M)
    cur = None
    for line in text.splitlines():
        m = re.match(r'##### t_([a-z]+)_(stock|abc)_(on|off)_s(\d+):', line)
        if m:
            cur = (m.group(1), m.group(2), m.group(3), int(m.group(4)))
            blocks[cur].append([])
            if mj:
                jobs[cur[0]] = f'{mj.group(2)} ({mj.group(1)})'
            continue
        if line.startswith('#####'):
            cur = None
        if cur is not None:
            blocks[cur][-1].append(line)

scenes = sorted({k[0] for k in blocks})
for sc in scenes:
    for arm in ARMS:
        for tm in TIMERS:
            for seed in args.seeds:
                key, name = (sc, arm, tm, seed), f't_{sc}_{arm}_{tm}_s{seed}'
                if len(blocks.get(key, [])) != 1:
                    errors[sc].append(f'{name}: {len(blocks.get(key, []))} run blocks'); continue
                L = blocks[key][0]
                if any(l.startswith(f'FAILED {name}') for l in L):
                    errors[sc].append(f'{name}: FAILED')
                walls = {(int(a), int(b)): (float(t), h) for a, b, t, h in
                         (m.groups() for m in (re.match(r'FGS_WALL from=(\d+) to=(\d+) s=([\d.]+) views=(\w+)', l) for l in L) if m)}
                clk = [float(m.group(1)) for m in (re.match(r'SEGCLOCK step=30000 \S+ cum_s=([\d.]+)', l) for l in L) if m]
                res = [m for m in (re.match(r'RESULT \S+ PSNR:(\S+) SSIM:(\S+) LPIPS:(\S+) N_Gaussians:(\d+) ms_it=(\S*)', l) for l in L) if m]
                if set(walls) != {(0, 15000), (15000, 30000)}:
                    errors[sc].append(f'{name}: FGS_WALL windows {sorted(walls)}'); continue
                if len(clk) != 1 or len(res) != 1:
                    errors[sc].append(f'{name}: {len(clk)} SEGCLOCK step=30000, {len(res)} RESULT'); continue
                msit = res[0].group(5)
                if tm == 'on' and not msit:
                    errors[sc].append(f'{name}: timer ON but no NeRFICG ms/it')
                if tm == 'off' and msit:
                    errors[sc].append(f'{name}: timer OFF but NeRFICG ms/it {msit} present (timer not off?)')
                runs[key] = dict(w1=walls[(0, 15000)][0], w2=walls[(15000, 30000)][0], h1=walls[(0, 15000)][1],
                                 h2=walls[(15000, 30000)][1], seg=clk[0], psnr=float(res[0].group(1)),
                                 msit=float(msit) if msit else math.nan)
                runs[key]['wall'] = runs[key]['w1'] + runs[key]['w2']
    for seed in args.seeds:
        hs = {(runs[k]['h1'], runs[k]['h2']) for k in runs if k[0] == sc and k[3] == seed}
        if len(hs) > 1:
            errors[sc].append(f'seed {seed}: runs saw different view sequences {sorted(hs)}')

ok = [sc for sc in scenes if not errors[sc]]
print(f'## Validation: {len(ok)}/{len(scenes)} scenes pass ({", ".join(scenes)})')
for sc in scenes:
    for e in errors[sc]:
        print(f'  - {sc}: {e}')
if len(ok) != len(scenes) or not ok:
    sys.exit(1)
print('  - all runs: both wall windows, clock, results present; timer state confirmed; view sequences identical per seed')

S = args.seeds
f3 = lambda v, f: f'{f % st.mean(v)} ({" / ".join(f % x for x in v)})'
get = lambda sc, arm, tm, k: [runs[(sc, arm, tm, s)][k] for s in S]
pct = lambda a, b: [100 * (x / y - 1) for x, y in zip(a, b)]
for sc in ok:
    print(f'\n## {sc} (job {jobs.get(sc, "?")}; seeds {", ".join(map(str, S))})\n')
    print('| arm | timer | wall 0–30k, s | wall 0–15k | wall 15k–30k | summed step events (SEGCLOCK), s | wall − SEGCLOCK, s | NeRFICG ms/it | PSNR |')
    print('|---|---|---|---|---|---|---|---|---|')
    for arm in ARMS:
        for tm in TIMERS:
            w, seg = get(sc, arm, tm, 'wall'), get(sc, arm, tm, 'seg')
            mi = get(sc, arm, tm, 'msit')
            print(f'| {LABEL[arm]} | {tm.upper()} | {f3(w, "%.1f")} | {st.mean(get(sc, arm, tm, "w1")):.1f} | {st.mean(get(sc, arm, tm, "w2")):.1f} | '
                  f'{f3(seg, "%.1f")} | {st.mean(a - b for a, b in zip(w, seg)):.1f} | '
                  f'{"" if math.isnan(mi[0]) else "%.2f" % st.mean(mi)} | {st.mean(get(sc, arm, tm, "psnr")):.3f} |')
    print('\nPaired effects (per seed in brackets):\n')
    for arm in ARMS:
        print(f'- {LABEL[arm]}, timer OFF vs ON: wall {f3(pct(get(sc, arm, "off", "wall"), get(sc, arm, "on", "wall")), "%+.1f")}%, '
              f'SEGCLOCK {f3(pct(get(sc, arm, "off", "seg"), get(sc, arm, "on", "seg")), "%+.1f")}%, '
              f'ΔPSNR {f3([a - b for a, b in zip(get(sc, arm, "off", "psnr"), get(sc, arm, "on", "psnr"))], "%+.3f")}')
    for tm in TIMERS:
        print(f'- A+B+C vs stock, timer {tm.upper()}: wall {f3(pct(get(sc, "abc", tm, "wall"), get(sc, "stock", tm, "wall")), "%+.1f")}%, '
              f'SEGCLOCK {f3(pct(get(sc, "abc", tm, "seg"), get(sc, "stock", tm, "seg")), "%+.1f")}%')
