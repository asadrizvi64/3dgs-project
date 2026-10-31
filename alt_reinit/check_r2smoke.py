"""Checks of the r2smoke stage (alt_reinit/build_reinit.sbatch): four short garden runs, k = 5000, to 6500 steps.

    python3 alt_reinit/check_r2smoke.py <job OUT dir> <nerficg dir>

- every arm evaluated at 5000, 5500, 6000 and 6500, each EVAL check=pass (model and RNG untouched);
- the evaluator's numbers at 6500 equal NeRFICG's own metrics_8bit.txt for that run (means within 1e-4; per-image PSNR
  within 0.006, since the file rounds to 0.01);
- ref: densification and the opacity reset at 6000 still run (stock);
- stop / same / target: no densify call after 5000, no opacity reset at 6000, N constant from step 5100 on;
- same / target: the reinit at 5000 passed its in-run check and N afterwards equals the target (match / 4,166,000);
- stop: no reinit ran.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EV = re.compile(r'^EVAL step=(\d+) psnr=(\S+) ssim=(\S+) lpips=(\S+) n_images=(\d+) N=(\d+) t_eval=(\S+) check=(\S+)', re.M)


def check(out: Path, nerficg: Path, final: int = 6500, k: int = 5000, n15: int = 4166000) -> dict[str, bool]:
    checks = {}
    want_steps = list(range(k, final + 1, 500))
    for arm in ('ref', 'stop', 'same', 'target'):
        name = f'ar0_garden_r2{arm}_s0'
        f = out / f'{name}.seg'
        t = f.read_text() if f.exists() else ''
        ev = {int(m.group(1)): m for m in EV.finditer(t)}
        checks[f'{arm}: evaluations at {want_steps}, all check=pass'] = (
            sorted(ev) == want_steps and all(m.group(8) == 'pass' for m in ev.values()))
        if ev:
            print(f'{arm}: t_eval per call {[float(m.group(7)) for m in ev.values()]}')
        key = f'{arm}: evaluator == metrics_8bit.txt at {final} (means within 1e-4, per-image PSNR within 0.006)'
        checks[key] = False
        rd = out / 'runs' / name / 'rundir.txt'
        js = out / 'runs' / name / 'fgs_eval' / 'metrics.jsonl'
        mt = nerficg / rd.read_text().strip() / f'test_{final}' / 'metrics_8bit.txt' if rd.exists() else None
        if mt is not None and mt.exists() and js.exists():
            lines = mt.read_text().splitlines()
            fin = dict(re.findall(r'(\w+):(\S+)', lines[-1]))
            per = [line.split('\t') for line in lines if re.match(r'^\d+\t', line)]
            last = [j for j in map(json.loads, js.read_text().splitlines()) if j['step'] == final]
            if last and fin:
                last = last[0]
                d = {m: abs(float(fin[M]) - last['mean'][m]) for m, M in (('psnr', 'PSNR'), ('ssim', 'SSIM'), ('lpips', 'LPIPS'))}
                ip = last['per_image']['psnr']
                dimg = max(abs(float(p[1]) - v) for p, v in zip(per, ip)) if len(per) == len(ip) else 99.0
                print(f'{arm}: evaluator vs metrics_8bit at {final}: mean diffs {d}, max per-image PSNR diff {dimg:.4f}')
                checks[key] = d['psnr'] < 1e-4 and d['ssim'] < 1e-4 and d['lpips'] < 1e-4 and dimg < 0.006
        seg = {int(a): int(b) for a, b in re.findall(r'^SEGCLOCK step=(\d+) .*? N=(\d+) ', t, re.M)}
        dens_after = [int(i) for i in re.findall(r'^CBCLOCK name=densify iteration=(\d+) ', t, re.M) if int(i) > k]
        reset = bool(re.search(r'^CBCLOCK name=reset_opacities iteration=6000 ', t, re.M))   # stock resets every 3000
        after = {n for s, n in seg.items() if s >= k + 100}
        if arm == 'ref':
            checks['ref: densification and the opacity reset at 6000 continue (stock)'] = (
                bool(dens_after) and reset and seg.get(final, 0) > seg.get(k + 100, 0))
        else:
            checks[f'{arm}: no densify after {k}, no opacity reset at 6000, N constant from {k + 100} on'] = (
                not dens_after and not reset and (k + 100) in seg and len(after) == 1)
        if arm in ('same', 'target'):
            m = re.search(rf'^REINIT_COUNTS iteration={k} n_before=(\d+) .* n_after=(\d+) target=(\d+)', t, re.M)
            want = None if not m else (int(m.group(1)) if arm == 'same' else n15)
            print(f'{arm}: {m.group(0) if m else "no REINIT_COUNTS"}')
            checks[f'{arm}: reinit at {k} passed its check and N afterwards == target'] = bool(
                m and f'REINIT_CHECK iteration={k} result=pass' in t and int(m.group(2)) == want and after == {want})
        if arm == 'stop':
            checks['stop: no reinit ran'] = 'REINIT iteration' not in t and t != ''
    return checks


def main() -> None:
    checks = check(Path(sys.argv[1]), Path(sys.argv[2]))
    for name, ok in checks.items():
        print(('PASS ' if ok else 'FAIL ') + name)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
