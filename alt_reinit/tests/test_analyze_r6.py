"""alt_reinit/analyze_r6.py on synthetic R2 + R6 logs: the pre-registered SSIM-recovery rule."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analyze_r6 as a6  # noqa: E402
import analyze_r2 as a2  # noqa: E402

STEPS = (5000, 30000)


def _log(tmp: Path, job: str, prefix: str, ssim: dict[str, float], stage: str) -> Path:
    lines = [f'node=c1 job={job} gpu=H100 stages={stage} repo=abc1234', f'########## {stage} garden']
    for seed in range(3):
        for arm, s30 in ssim.items():
            name = f'{prefix}_garden_{arm}_s{seed}'
            lines.append(f'##### {name}: method=FasterGSReinit scene=garden seed={seed} iters=30000 reinit=x cbclock=1 eval=every=500,from=5000 00:00:00')
            js = []
            for st in STEPS:
                s = s30 + (seed - 1) * 0.0002 if st == 30000 else 0.8
                lines.append(f'SEGCLOCK step={st} seg_s=1 cum_s={st * 0.01:.3f} sparse=0 N=100 peak_alloc_gib=1')
                lines.append(f'EVAL step={st} psnr=27.4 ssim={s} lpips=0.12 n_images=2 N=100 t_eval=0.5 check=pass')
                js.append(json.dumps(dict(step=st, per_image=dict(psnr=[27.0, 27.8], ssim=[s] * 2, lpips=[0.12] * 2))))
            d = tmp / f'run_{job}' / 'runs' / name / 'fgs_eval'
            d.mkdir(parents=True)
            (d / 'metrics.jsonl').write_text('\n'.join(js) + '\n')
    log = tmp / f'gs-alt-{stage}-{job}.out'
    log.write_text('\n'.join(lines) + '\nJOB OK\n')
    return log


def _run(tmp: Path, stopr: float, targetr: float):
    r2 = _log(tmp, '41', 'ar2', dict(ref=0.8668, stop=0.8618, same=0.8642, target=0.8654), 'r2')
    r6 = _log(tmp, '46', 'ar6', dict(stopr=stopr, targetr=targetr), 'r6')
    return subprocess.run([sys.executable, str(HERE / 'analyze_r6.py'), str(r2), str(r6)], capture_output=True, text=True,
                          cwd=str(HERE))


def test_supported(tmp_path):
    r = _run(tmp_path, 0.8665, 0.8667)
    assert r.returncode == 0, r.stdout + r.stderr
    assert '**Hypothesis S (the SSIM loss comes from the missing opacity resets): supported.**' in r.stdout
    assert 'stop+resets' in r.stdout and 'reinit-target+resets' in r.stdout


def test_partly(tmp_path):
    r = _run(tmp_path, 0.8665, 0.8655)   # targetr: +0.0001, inside the seed sd
    assert 'partly supported.**' in r.stdout


def test_not_supported(tmp_path):
    r = _run(tmp_path, 0.8620, 0.8640)
    assert ': not supported.**' in r.stdout


def test_missing_run(tmp_path):
    runs = a2.parse_log(str(_log(tmp_path, '41', 'ar2', dict(ref=0.8668, stop=0.8618, target=0.8654), 'r2')))
    runs += [x for x in a2.parse_log(str(_log(tmp_path, '46', 'ar6', dict(stopr=0.866, targetr=0.866), 'r6')))
             if x['name'] != 'ar6_garden_targetr_s2']
    _, problems = a6.collect(runs)
    assert any(p.startswith('targetr: seeds [0, 1]') for p in problems)
