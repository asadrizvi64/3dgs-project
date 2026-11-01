"""R6 analysis (alt_reinit/PLAN.md, "R6", pre-registered 2026-11-01): do kept opacity resets recover garden's SSIM?

    python3 alt_reinit/analyze_r6.py alt_reinit/logs/gs-alt-r2-garden-<job>.out alt_reinit/logs/gs-alt-r6-<job>.out

Comparators are the R2 garden runs ar2_garden_{ref,stop,target}_s<seed>; the new arms are ar6_garden_{stopr,targetr}_s<seed>.

**Primary rule (3-seed means at 30k), per pair X -> X+resets (stop -> stopr, target -> targetr):**
- gap closed g = (SSIM(X+resets) - SSIM(X)) / (SSIM(ref) - SSIM(X));
- supported for the pair when g >= 0.5 and SSIM(X+resets) - SSIM(X) > the larger of the two arms' seed sds;
- hypothesis S supported (both pairs) / partly supported (one) / not supported (neither).

Also printed: PSNR / LPIPS differences and N, and analyze_r2's table with the two new arms added (their R2 outcome
category against R2's ref target; track times of the new arms are from another job, so time comparisons are cross-job).
Exit status 1 when a run is missing, failed, or lacks the 30k evaluation.
"""
from __future__ import annotations

import re
import statistics as stt
import sys
from collections import defaultdict

import analyze_r2 as a2

PAIRS = (('stop', 'stopr'), ('target', 'targetr'))
LABEL = dict(a2.ARM_LABEL, stopr='stop+resets', targetr='reinit-target+resets')
NAME = re.compile(r'^ar([26])_garden_([a-z]+)_s(\d+)$')


def collect(runs: list[dict]) -> tuple[dict, list[str]]:
    by, problems = defaultdict(dict), []
    for r in runs:
        m = NAME.match(r['name'])
        if not m:
            continue
        stage, arm, seed = m.group(1), m.group(2), int(m.group(3))
        if (stage == '2') != (arm in a2.ARMS):
            continue   # ar2 runs of R6 arms or ar6 runs of R2 arms are not expected
        by[arm][seed] = r
        if r['failed'] or 30000 not in r['ev'] or any(e['check'] != 'pass' for e in r['ev'].values()):
            problems.append(f'{r["name"]}: failed, no evaluation at 30000, or an EVAL check failed')
    for arm in ('ref', 'stop', 'target', 'stopr', 'targetr'):
        if sorted(by.get(arm, {})) != [0, 1, 2]:
            problems.append(f'{arm}: seeds {sorted(by.get(arm, {}))}, expected [0, 1, 2]')
    return by, problems


def at30(rs: dict, k: str) -> tuple[float, float]:
    return a2.mean_sd(r['ev'][30000][k] for r in rs.values() if 30000 in r['ev'])


def primary(by: dict) -> tuple[list[str], str]:
    ref = at30(by['ref'], 'ssim')[0]
    out = ['| pair | SSIM X | SSIM X+resets | difference | larger seed sd | gap closed g | PSNR diff | LPIPS diff | N at 30k (X+resets) | supported |',
           '|---|---|---|---|---|---|---|---|---|---|']
    ok = []
    for x, xr in PAIRS:
        (sx, dx), (sr, dr) = at30(by[x], 'ssim'), at30(by[xr], 'ssim')
        diff, sd = sr - sx, max(dx, dr)
        g = diff / (ref - sx) if ref != sx else float('nan')
        sup = g >= 0.5 and diff > sd
        ok.append(sup)
        dp = at30(by[xr], 'psnr')[0] - at30(by[x], 'psnr')[0]
        dl = at30(by[xr], 'lpips')[0] - at30(by[x], 'lpips')[0]
        n = a2.mean_sd(r['ev'][30000]['N'] for r in by[xr].values())[0]
        out.append(f'| {LABEL[x]} -> {LABEL[xr]} | {sx:.5f} ± {dx:.5f} | {sr:.5f} ± {dr:.5f} | {diff:+.5f} | {sd:.5f} | '
                   f'{g:.2f} | {dp:+.3f} | {dl:+.4f} | {n:,.0f} | {"yes" if sup else "no"} |')
    verdict = {2: 'supported', 1: 'partly supported', 0: 'not supported'}[sum(ok)]
    return out, verdict


def main() -> None:
    runs = []
    for p in sys.argv[1:]:
        runs += a2.parse_log(p)
    by, problems = collect(runs)
    out = ['# Alternative track R, R6 analysis (garden, k = 5000, opacity resets kept after k)', '',
           f'Jobs: {", ".join(sorted({r["job"] for r in runs}))}. Rule: PLAN.md, "R6" (pre-registered).', '']
    if not problems:
        ref = {k: at30(by['ref'], k) for k in a2.METRICS}
        out += [f'R2 ref at 30k: PSNR {ref["psnr"][0]:.3f}, SSIM {ref["ssim"][0]:.5f} (sd {ref["ssim"][1]:.5f}), '
                f'LPIPS {ref["lpips"][0]:.4f}.', '', '## Primary rule', '']
        rows, verdict = primary(by)
        out += rows + ['', f'**Hypothesis S (the SSIM loss comes from the missing opacity resets): {verdict}.**', '']
        # analyze_r2's table with the new arms (renamed into its scheme)
        a2.ARMS = a2.ARMS + ('stopr', 'targetr')
        a2.ARM_LABEL.update(LABEL)
        merged = []
        for arm, rs in by.items():
            for seed, r in rs.items():
                merged.append(dict(r, name=f'ar2_garden_{arm}_s{seed}'))
        merged += [r for r in runs if r['name'] in {f'ar2_garden_same_s{s}' for s in range(3)}]
        text, p2, _ = a2.analyze(merged)
        out += ['## R2-style outcomes with the new arms (time comparisons for stop+resets / reinit-target+resets are '
                'cross-job)', ''] + text.splitlines()[5:]
        problems += p2
    if problems:
        out += ['## Problems', ''] + [f'- {p}' for p in problems]
    print('\n'.join(out))
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
