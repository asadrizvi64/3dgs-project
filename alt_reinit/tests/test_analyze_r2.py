"""alt_reinit/analyze_r2.py on a synthetic R2 job: candidates, track clock, outcomes, image failures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_r2 as ar  # noqa: E402

STEPS = (5000, 10000, 20000, 30000)
N_IMG = 4


def _curve(arm: str, seed: int, step: int) -> tuple[float, float, float]:
    """ref ends at 25.0 / 0.80 / 0.20 (seed mean); target gets there by 10000; stop never does; same at 30000 only."""
    jitter = (seed - 1) * 0.01
    frac = {5000: 0.0, 10000: 0.5, 20000: 0.8, 30000: 1.0}[step]
    if arm == 'ref':
        p = 23.0 + 2.0 * frac
    elif arm == 'target':
        p = 23.0 + 2.2 * min(1.0, 2 * frac)
    elif arm == 'stop':
        p = 23.0 + 1.0 * frac
    else:
        p = 23.0 + 2.1 * frac
    q = (p - 23.0) / 2.0
    return p + jitter, 0.70 + 0.10 * q, 0.30 - 0.10 * q


def _write_job(tmp: Path, bad_image_arm: str | None = None) -> Path:
    job = '42'
    lines = ['node=c1 job=42 gpu=H100 stages=r2 repo=abc1234', '########## R2 garden']
    for seed in range(3):
        for arm in ar.ARMS:
            name = f'ar2_garden_{arm}_s{seed}'
            spec = {'ref': '-', 'stop': 'mode=stop,iters=5000'}.get(arm, f'mode=apply,iters=5000,target=match,end_densify=1')
            lines.append(f'##### {name}: method=FasterGSReinit scene=garden seed={seed} iters=30000 reinit={spec} cbclock=1 eval=every=500,from=5000 00:00:00')
            rate = {'ref': 0.010, 'stop': 0.006, 'same': 0.009, 'target': 0.008}[arm]
            lines.append('CBCLOCK name=densify iteration=600 s=0.050000')
            if arm in ('same', 'target'):
                lines.append('REINIT iteration=5000 mode=apply stride=2 retries=0 timed_target=1 t_total=1.5000')
                lines.append('REINIT_COUNTS iteration=5000 n_before=100 n_candidates_grid=9 n_filtered=9 n_deduped=9 '
                             'n_covered=10 n_survivors=90 n_added=10 n_after=100 target=100 max_reachable=200')
                lines.append('CBCLOCK name=reinit_gaussians iteration=5000 s=1.600000')
            js = []
            for st in STEPS:
                lines.append(f'SEGCLOCK step={st} seg_s=1 cum_s={rate * st:.3f} sparse=0 N=100 peak_alloc_gib=1')
                p, s, lp = _curve(arm, seed, st)
                lines.append(f'EVAL step={st} psnr={p} ssim={s} lpips={lp} n_images={N_IMG} N=100 t_eval=0.5 check=pass')
                imgs = [p + d for d in (-0.5, 0.0, 0.2, 0.3)]
                if arm == bad_image_arm and seed == 0:
                    imgs[1] -= 5.0
                js.append(json.dumps(dict(step=st, per_image=dict(psnr=imgs, ssim=[s] * N_IMG, lpips=[lp] * N_IMG))))
            d = tmp / f'run_{job}' / 'runs' / name / 'fgs_eval'
            d.mkdir(parents=True)
            (d / 'metrics.jsonl').write_text('\n'.join(js) + '\n')
    log = tmp / 'gs-alt-r2-garden-42.out'
    log.write_text('\n'.join(lines) + '\nJOB OK\n')
    return log


def _outcomes(text: str) -> dict[str, str]:
    rows = [line.split(' | ') for line in text.splitlines() if line.startswith('| ') and 'candidate step' not in line and '---' not in line]
    return {r[0][2:]: r[-1].rstrip(' |') for r in rows}


def test_outcomes(tmp_path):
    text, problems, confirm = ar.analyze(ar.parse_log(str(_write_job(tmp_path))))
    assert not problems, problems
    o = _outcomes(text)
    assert o['ref'] == 'reference'
    assert o['reinit-target'].startswith('improves time-to-quality')
    assert o['stop'].startswith('reduces quality')
    assert o['reinit-same'].startswith('improves') or o['reinit-same'].startswith('keeps')
    assert ('garden', 'target', 10000) in confirm and ('garden', 'ref', 30000) in confirm
    assert not any(a == 'stop' for _, a, _ in confirm)


def test_clock_counts_callbacks_before_the_step(tmp_path):
    runs = ar.parse_log(str(_write_job(tmp_path)))
    r = [x for x in runs if x['name'] == 'ar2_garden_target_s0'][0]
    assert abs(ar.clock(r, 5000) - (0.008 * 5000 + 0.05)) < 1e-9          # reinit at iteration 5000 is after step 5000
    assert abs(ar.clock(r, 10000) - (0.008 * 10000 + 0.05 + 1.6)) < 1e-9


def test_new_image_failure_blocks_a_positive_outcome(tmp_path):
    text, problems, _ = ar.analyze(ar.parse_log(str(_write_job(tmp_path, bad_image_arm='target'))))
    assert _outcomes(text)['reinit-target'].startswith('reduces quality (new image failures)')
    assert '**1**' in text


def test_missing_run_is_a_problem(tmp_path):
    runs = [r for r in ar.parse_log(str(_write_job(tmp_path))) if r['name'] != 'ar2_garden_same_s2']
    _, problems, _ = ar.analyze(runs)
    assert any('same' in p for p in problems)


def test_confirmation(tmp_path):
    lines = ['node=c1 job=43 gpu=H100 stages=r4 repo=abc1234']
    for arm, step, p in (('ref', 30000, 25.0), ('target', 10000, 25.1), ('same', 30000, 24.9)):
        lines.append(f'########## R4 confirm garden {arm} at step {step}')
        for seed in range(3):
            lines.append(f'##### ar4_garden_{arm}_s{seed}: method=FasterGSReinit scene=garden seed={seed} iters=30000 reinit=x cbclock=1 eval=at={step} 00:00:00')
            lines.append(f'SEGCLOCK step={step} seg_s=1 cum_s={step * 0.01:.3f} sparse=0 N=1 peak_alloc_gib=1')
            lines.append(f'EVAL step={step} psnr={p} ssim=0.8 lpips=0.2 n_images=4 N=1 t_eval=0.5 check=pass')
    log = tmp_path / 'gs-alt-r4-43.out'
    log.write_text('\n'.join(lines) + '\n')
    text, problems = ar.confirmation(ar.parse_log(str(log)))
    assert not problems, problems
    assert 'reinit-target | 10000 | 3 |' in text and 'confirmed: meets the target, earlier than ref' in text
    assert 'not reproduced: psnr below target' in text
