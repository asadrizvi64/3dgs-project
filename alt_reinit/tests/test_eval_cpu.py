"""CPU tests for the R2 additions: FGS_REINIT stop / end_densify and the FGS_EVAL schedule (alt_reinit/reinit_eval.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reinit_core as rc  # noqa: E402
import reinit_eval as re_  # noqa: E402


def test_stop_and_end_densify():
    c = rc.parse_env('mode=stop,iters=5000')
    assert c.mode == 'stop' and c.end_densify and c.start == c.end == 5000
    c = rc.parse_env('mode=apply,iters=5000,target=4166000,end_densify=1')
    assert c.end_densify and c.target == '4166000'
    assert not rc.parse_env('mode=apply,iters=5000').end_densify
    for bad in ('mode=stop,iters=2000:5000', 'mode=audit,iters=5000,end_densify=1', 'mode=stop'):
        with pytest.raises(ValueError):
            rc.parse_env(bad)


def test_eval_every():
    c = re_.parse_eval('every=500,from=5000', 30000)
    assert c.labels[0] == 5000 and c.labels[-1] == 30000 and len(c.labels) == 51
    assert c.schedule == (4999, 29999, 500)
    assert re_.parse_eval('every=1000', 3500).labels == [1000, 2000, 3000]


def test_eval_at():
    c = re_.parse_eval('at=12500', 30000)
    assert c.labels == [12500] and c.schedule == (12499, 12499, 1)
    c = re_.parse_eval('at=13000:12000:12500', 30000)
    assert c.labels == [12000, 12500, 13000] and c.schedule == (11999, 12999, 500)
    c = re_.parse_eval('at=10000:10300', 30000)
    assert c.schedule == (9999, 10299, 300)


@pytest.mark.parametrize('bad', ['', 'every=500,at=1', 'at=40000', 'at=0', 'every=0', 'at=5,from=1', 'bogus=1',
                                 'every=500,from=40000'])
def test_eval_bad(bad):
    with pytest.raises(ValueError):
        re_.parse_eval(bad, 30000)


def test_schedule_hits_every_label():
    """NeRFICG runs a callback at it with start <= it <= end and (it - start) % stride == 0: every label must be hit."""
    for spec in ('every=500,from=5000', 'at=12000:12500:20000', 'at=7', 'every=700,from=300'):
        c = re_.parse_eval(spec, 30000)
        s, e, k = c.schedule
        hit = {it + 1 for it in range(30000) if s <= it <= e and (it - s) % k == 0}
        assert set(c.labels) <= hit, spec
