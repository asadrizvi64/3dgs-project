"""CPU tests of alt_reinit/analyze_audit.py and alt_reinit/record_runs.py on a synthetic job:

    python -m pytest -q alt_reinit/tests/test_analyze_record.py

The audit run's files (stats.json, agreement_final.json, masks) come from a real run of the driver on the stand-in scene
(test_trainer_cpu.py), so the analyzer reads the format the driver writes. The job also has a shortened apply run and a
failed run, as the recorder must handle both.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from test_trainer_cpu import BASE, make_trainer, mods, run  # noqa: E402,F401  (mods is the shared fixture)

JOB = '777'


@pytest.fixture(scope='module')
def job(mods, tmp_path_factory):
    tmp = tmp_path_factory.mktemp('job')
    trainer, ds, log = make_trainer(mods, tmp / 'nerficg_run', f'mode=audit,target=+300/match,{BASE}', n_views=24)
    run(mods.rt.run_reinit, trainer, 2000, ds)
    run(mods.rt.run_audit_final, trainer, 2099, ds)
    reinit_lines = [ln for ln in log.read_text().splitlines() if ln.startswith('REINIT')]
    logs = tmp / 'logs'
    adir = logs / f'run_{JOB}' / 'runs' / 'ar1_garden_audit_s0'
    adir.mkdir(parents=True)
    shutil.copytree(tmp / 'nerficg_run' / 'reinit_audit', adir / 'reinit_audit')
    (adir / 'rundir.txt').write_text('output/FasterGSReinit/fastergsreinit_garden_x_s0ar1_garden_audit_s0_2026-10-30-10-00-00')
    (adir / 'gpu_clock_summary.txt').write_text('CLOCK samples_busy=250 sm_mhz_mean=1980 sm_mhz_min=1980 mem_mhz=2619 power_w_mean=400 power_w_max=500 temp_c_max=60 event_reasons={}')
    audit = ['##### ar1_garden_audit_s0: method=FasterGSReinit scene=garden seed=0 iters=30000 '
             f'reinit=mode=audit,iters=2000,target=+300/match,{BASE.split(",", 1)[1]} cbclock=1 10:00:00',
             'CBCLOCK_INSTALLED iteration=0 callbacks=densify,morton_ordering,reset_opacities,reinit_gaussians',
             'SEGCLOCK step=1000 seg_s=1.800 cum_s=1.800 sparse=0 N=500000 peak_alloc_gib=1.00',
             'CBCLOCK name=densify iteration=600 s=0.005000', 'CBCLOCK name=densify iteration=2000 s=0.007000',
             'SEGCLOCK step=2000 seg_s=2.700 cum_s=4.500 sparse=0 N=900000 peak_alloc_gib=1.50',
             *reinit_lines,
             'CBCLOCK name=reinit_gaussians iteration=2000 s=30.000000',
             'SEGCLOCK step=30000 seg_s=6.600 cum_s=250.000 sparse=0 N=4100000 peak_alloc_gib=6.50',
             'FGS_TRAIN_END N=4100000 peak_alloc_gib=6.50 peak_reserved_gib=8.00',
             'RUNDIR ar1_garden_audit_s0 output/FasterGSReinit/x',
             'PERIMG ar1_garden_audit_s0 n_images=24 min_image=3 min_psnr=25.10',
             'RESULT ar1_garden_audit_s0 PSNR:27.500 SSIM:0.8670 LPIPS:0.1200 N_Gaussians:4100000 ms_it=8.40 allocated: 6.40']
    apply = ['##### ar0_garden_applymatch_s0: method=FasterGSReinit scene=garden seed=0 iters=2300 '
             'reinit=mode=apply,iters=2000,target=match cbclock=1 10:10:00',
             'SEGCLOCK step=2300 seg_s=1.000 cum_s=6.000 sparse=0 N=1000000 peak_alloc_gib=2.00',
             'CBCLOCK name=densify iteration=2000 s=0.010000', 'CBCLOCK name=reinit_gaussians iteration=2000 s=3.500000',
             'REINIT_CHECK iteration=2000 result=pass',
             'REINIT_COUNTS iteration=2000 n_before=1000000 n_candidates_grid=5 n_filtered=4 n_deduped=3 n_covered=400000 '
             'n_survivors=600000 n_added=400000 n_after=1000000 target=1000000 max_reachable=3000000',
             'RESULT ar0_garden_applymatch_s0 PSNR:22.000 SSIM:0.6000 LPIPS:0.4000 N_Gaussians:1000000 ms_it=3.00 allocated: 2.00']
    failed = ['##### ar0_garden_off_s0: method=FasterGSReinit scene=garden seed=0 iters=1200 reinit=- cbclock=0 10:20:00',
              'Traceback (most recent call last):', 'FAILED ar0_garden_off_s0']
    text = '\n'.join([f'node=c1 job={JOB} gpu=NVIDIA H100 stages=smoke audit repo=abc1234', '########## audit',
                      *audit, *apply, *failed, 'JOB FAILED: ar0_garden_off_s0']) + '\n'
    path = logs / f'gs-alt-r1-{JOB}.out'
    path.write_text(text)
    return dict(path=path, logs=logs, reinit=reinit_lines)


def analyze(*args):
    return subprocess.run([sys.executable, str(REPO / 'alt_reinit' / 'analyze_audit.py'), *map(str, args)],
                          capture_output=True, text=True)


def test_analyzer_validates_and_reports(job):
    r = analyze(job['path'])
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert '- ar1_garden_audit_s0: valid' in out
    assert '## garden: ar1_garden_audit_s0' in out and '### Counts and targets' in out
    # budget: reference (stock stage run, time to its own 30k PSNR) - warm-up (4.5 s steps + 0.012 s callbacks) - reinit
    t_re = float(next(ln for ln in job['reinit'] if ln.startswith('REINIT iteration')).split('t_total=')[1].split()[0])
    ref = next(float(ln.split(': ')[1].split()[0]) for ln in out.splitlines() if ln.startswith('Reference estimate'))
    assert 150 < ref < 260
    row = next(ln for ln in out.splitlines() if ln.startswith('| garden | 2000 |'))
    assert abs(float(row.split('|')[3]) - (ref - 4.512 - t_re)) < 0.11 and '| PASS |' in row
    assert 'earliest k passing G1 and G2' in out
    assert 'Agreement with the final model' in out and 'not accuracy' in out
    assert 'masks/view000.jpg' in out


def test_analyzer_flags_invalid_audit(job, tmp_path):
    bad = tmp_path / f'gs-alt-r1-{JOB}.out'
    shutil.copytree(job['logs'] / f'run_{JOB}', tmp_path / f'run_{JOB}')
    bad.write_text('\n'.join(ln for ln in job['path'].read_text().splitlines() if not ln.startswith('REINIT_AGREEMENT')))
    r = analyze(bad)
    assert r.returncode == 1 and 'INVALID' in r.stdout and 'no final agreement' in r.stdout


def test_recorder_rows(job, tmp_path):
    csv_path = tmp_path / 'log.csv'
    csv_path.write_text(open(REPO / 'experiment_log.csv', encoding='utf-8').readline())
    rec = [sys.executable, str(REPO / 'alt_reinit' / 'record_runs.py'), str(job['path']), '--csv', str(csv_path)]
    r = subprocess.run(rec + ['--dry-run'], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith('3 new row(s) (dry run')
    assert csv_path.read_text().count('\n') == 1   # dry run wrote nothing
    r = subprocess.run(rec, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
    by = {row['notes'].split(';')[0]: row for row in rows}
    a = by[f'run=ar1_garden_audit_s0 job={JOB}']
    p = by[f'run=ar0_garden_applymatch_s0 job={JOB}']
    f = by[f'run=ar0_garden_off_s0 job={JOB}']
    assert (a['role'], p['role'], f['role']) == ('alt-reinit-audit', 'alt-reinit-check', 'alt-reinit-check')
    assert float(a['train_time_s']) == pytest.approx(250.012, abs=0.06)        # the audit's dry-run reinit is excluded
    assert float(p['train_time_s']) == pytest.approx(6.0 + 0.01 + 3.5, abs=0.06)  # an applied reinit counts
    assert 'FAILED' in f['notes'] and f['psnr'] == '' and 'FAILED' not in a['notes'] + p['notes']
    assert 'shortened run (2300 iterations' in p['notes'] and 'shortened' not in a['notes']
    assert 'replacement check at 2000: pass' in a['notes'] + p['notes']
    assert all('phase1 macro' not in row['commit'] and 'FasterGSReinit (FasterGS@1144604' in row['commit'] for row in rows)
    assert a['date'] == '2026-10-30' and a['train_resolution'] == '1297x840' and a['psnr'] == '27.500'
    assert 'audit (dry run on a copy)' in a['method'] and 'flags off' in f['method']
    r = subprocess.run(rec, capture_output=True, text=True)                   # second time: nothing new
    assert r.stdout.startswith('0 new row(s)') and len(list(csv.DictReader(open(csv_path, encoding='utf-8')))) == 3
