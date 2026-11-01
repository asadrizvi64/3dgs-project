"""Checks of the r6smoke stage (alt_reinit/build_reinit.sbatch): the two keep_resets arms, garden seed 0, to 6500 steps.

    python3 alt_reinit/check_r6smoke.py <job OUT dir>

- both arms evaluated at 5000, 5500, 6000 and 6500, each EVAL check=pass;
- the R6 reset callback ran at 6000 and nowhere else up to 6500; stock's reset callback did not run after 5000;
- no densify call after 5000, and N constant from step 5100 on;
- targetr: the reinit at 5000 passed its in-run check and N afterwards is 4,166,000; stopr: no reinit ran.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

EV = re.compile(r'^EVAL step=(\d+) .*? check=(\S+)', re.M)


def check(out: Path, final: int = 6500, k: int = 5000, n15: int = 4166000) -> dict[str, bool]:
    checks = {}
    want = list(range(k, final + 1, 500))
    for arm in ('stopr', 'targetr'):
        f = out / f'ar0_garden_r6{arm}_s0.seg'
        t = f.read_text() if f.exists() else ''
        ev = {int(m.group(1)): m.group(2) for m in EV.finditer(t)}
        checks[f'{arm}: evaluations at {want}, all check=pass'] = sorted(ev) == want and set(ev.values()) == {'pass'}
        r6 = [int(i) for i in re.findall(r'^CBCLOCK name=reinit_reset_opacities iteration=(\d+) ', t, re.M)]
        stock = [int(i) for i in re.findall(r'^CBCLOCK name=reset_opacities iteration=(\d+) ', t, re.M)]
        print(f'{arm}: R6 resets at {r6}, stock resets at {stock}')
        checks[f'{arm}: R6 reset at 6000 only; stock resets end by {k}'] = r6 == [6000] and all(i <= k for i in stock)
        seg = {int(a): int(b) for a, b in re.findall(r'^SEGCLOCK step=(\d+) .*? N=(\d+) ', t, re.M)}
        dens_after = [int(i) for i in re.findall(r'^CBCLOCK name=densify iteration=(\d+) ', t, re.M) if int(i) > k]
        after = {n for s, n in seg.items() if s >= k + 100}
        checks[f'{arm}: no densify after {k}, N constant from {k + 100} on'] = (
            not dens_after and (k + 100) in seg and len(after) == 1)
        if arm == 'targetr':
            m = re.search(rf'^REINIT_COUNTS iteration={k} .* n_after=(\d+)', t, re.M)
            checks[f'{arm}: reinit at {k} passed its check and N afterwards == {n15}'] = bool(
                m and f'REINIT_CHECK iteration={k} result=pass' in t and int(m.group(1)) == n15 and after == {n15})
        else:
            checks[f'{arm}: no reinit ran'] = 'REINIT iteration' not in t and t != ''
    return checks


def main() -> None:
    checks = check(Path(sys.argv[1]))
    for name, ok in checks.items():
        print(('PASS ' if ok else 'FAIL ') + name)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
