"""Record every run of an alt_reinit/build_reinit.sbatch job in experiment_log.csv (one row per run, failed runs too).

    python alt_reinit/record_runs.py alt_reinit/logs/<job-name>-<job>.out [more logs]  [--csv experiment_log.csv] [--dry-run]

The track's own recorder (phase2/record_runs.py is left as it is): it reads this track's run headers, describes the
FasterGSReinit variant correctly, and sets the role from the run-name prefix:
  ar0_ -> alt-reinit-check (R0 smoke), ar1_ -> alt-reinit-audit (R1), ar2_ -> alt-reinit-compare (R2),
  ar4_ -> alt-reinit-confirm (R4), anything else -> alt-reinit-other.
Shortened runs (fewer than 30k iterations) keep their role and say so in the notes. Each row's notes start with
'run=<run name> job=<job>', the de-duplication key: a run already in the CSV is not added again.

train_time_s is this track's clock: SEGCLOCK cum_s (CUDA events of every training step) plus the CBCLOCK time of the
method's between-step callbacks (densify, Morton ordering, opacity reset, the reinit in apply runs). In audit runs the
reinit callback is a dry run and is not counted. The notes give both parts.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys

RES = {'bicycle': '1237x822', 'flowers': '1256x828', 'garden': '1297x840', 'stump': '1245x825', 'treehill': '1267x832',
       'room': '1557x1038', 'counter': '1558x1038', 'kitchen': '1558x1039', 'bonsai': '1559x1039'}
CONST = dict(gpu='NVIDIA H100 SXM 94GB', driver='580.159.04', cuda='12.8', torch='2.11.0+cu128')
ROLES = {'ar0_': 'alt-reinit-check', 'ar1_': 'alt-reinit-audit', 'ar2_': 'alt-reinit-compare', 'ar4_': 'alt-reinit-confirm'}


def role_of(name: str) -> str:
    return next((v for k, v in ROLES.items() if name.startswith(k)), 'alt-reinit-other')


def describe(method: str, spec: str | None) -> str:
    if method == 'FasterGS':
        return 'Faster-GS main, stock (taming backward, dense Adam, fp32 moments)'
    if not spec:
        return f'{method}, flags off (stock Faster-GS behaviour: taming backward, dense Adam, fp32 moments)'
    s = dict(re.findall(r'(\w+)=([^,]+)', spec))
    filt = 'unfiltered' if s.get('min_agree') == '0' else f'filter {s.get("min_agree", "2")} of {s.get("n_nbr", "4")} neighbours'
    if s.get('mode') == 'audit':
        return f'{method}, stock training + reinit audit (dry run on a copy) at {s.get("iters")}, targets {s.get("target", "match")}, {filt}'
    return f'{method}, depth reinit at {s.get("iters")} (target {s.get("target", "match")}, {filt}), dense Adam, fp32 moments'


def provenance(method: str, repo: str) -> str:
    if method == 'FasterGS':
        return f'FasterGS@1144604 + sparse_from v5 hooks; 3dgs-project {repo}'
    return f'{method} (FasterGS@1144604 + sparse_from v5 + save_state hooks + alt_reinit patch); 3dgs-project {repo}'


def parse(path: str, seen: set) -> list[dict]:
    text = open(path, encoding='utf-8', errors='replace').read()
    hd = re.search(r'^node=(\S+) job=(\S+) gpu=.*? repo=(\S+)', text, re.M)
    if not hd:
        sys.exit(f'{path}: no job header line')
    node, job, repo = hd.groups()
    archive = os.path.join(os.path.dirname(path), f'run_{job}', 'runs')
    blocks, cur = [], None
    for line in text.splitlines():
        m = re.match(r'##### (\S+): method=(\S+) scene=(\S+) seed=(\d+) iters=(\d+) reinit=(\S+) cbclock=(\d)', line)
        if m:
            cur = dict(name=m.group(1), method=m.group(2), scene=m.group(3), seed=m.group(4), iters=int(m.group(5)),
                       spec=None if m.group(6) == '-' else m.group(6), lines=[])
            blocks.append(cur)
            continue
        if line.startswith('#####'):
            cur = None
        if cur is not None:
            cur['lines'].append(line)
    rows = []
    for b in blocks:
        key = f'run={b["name"]} job={job}'
        if key in seen:
            continue
        L = b['lines']
        grab = lambda pat: next((m for m in (re.match(pat, ln) for ln in L) if m), None)  # noqa: E731
        segs = [m for m in (re.match(r'SEGCLOCK step=(\d+) \S+ cum_s=(\S+)', ln) for ln in L) if m]
        seg = segs[-1] if segs else None
        audit = bool(b['spec'] and 'mode=audit' in b['spec'])
        cbs = [(m.group(1), float(m.group(2))) for m in (re.match(r'CBCLOCK name=(\S+) iteration=\d+ s=(\S+)', ln) for ln in L) if m]
        cb_total = sum(s for n, s in cbs if not (audit and n == 'reinit_gaussians'))
        res = grab(r'RESULT \S+ PSNR:(\S+) SSIM:(\S+) LPIPS:(\S+) N_Gaussians:(\d+)(?:\s+ms_it=(\S*))?(?:\s+allocated: ([\d.]+))?')
        end = grab(r'FGS_TRAIN_END N=(\d+) peak_alloc_gib=(\S+) peak_reserved_gib=(\S+)')
        per = grab(r'PERIMG \S+ n_images=(\d+) min_image=(\d+) min_psnr=(\S+)')
        failed = any(ln.startswith(f'FAILED {b["name"]}') for ln in L) or not res or not seg
        checks = [m.groups() for m in (re.match(r'REINIT_CHECK iteration=(\d+) result=(\S+)', ln) for ln in L) if m]
        counts = [kv for kv in (dict(re.findall(r'(\w+)=(\S+)', ln)) for ln in L if ln.startswith('REINIT_COUNTS '))]
        fails = [ln for ln in L if ln.startswith('REINIT_FAIL ')]
        rundir, clock = '', ''
        adir = os.path.join(archive, b['name'])
        if os.path.exists(os.path.join(adir, 'rundir.txt')):
            rundir = open(os.path.join(adir, 'rundir.txt')).read().strip()
        if os.path.exists(os.path.join(adir, 'gpu_clock_summary.txt')):
            mm = re.search(r'sm_mhz_mean=(\d+) sm_mhz_min=(\d+).*?power_w_mean=(\d+).*?temp_c_max=(\d+)', open(os.path.join(adir, 'gpu_clock_summary.txt')).read())
            if mm:
                clock = f'SM clock mean {mm.group(1)} / min {mm.group(2)} MHz, {mm.group(3)} W mean, max {mm.group(4)} C'
        dm = re.search(r'_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}-\d{2}$', rundir)
        notes = [key, f'alt_reinit {b["iters"]}-iteration run' + (f', FGS_REINIT {b["spec"]}' if b['spec'] else ', flags off')]
        if b['iters'] < 30000:
            notes.append(f'shortened run ({b["iters"]} iterations; metrics are not comparable with 30k runs)')
        if failed:
            why = [] if res else ['no final metrics']
            why += [] if seg else ['no training clock record']
            notes.append('FAILED: ' + ('; '.join(why) if why else 'see job log'))
        if seg:
            notes.append(f'train time = SEGCLOCK {float(seg.group(2)):.1f} s + callbacks {cb_total:.1f} s (CBCLOCK'
                         + ('; audit dry runs excluded' if audit else '') + ')' if cbs else
                         f'train time = SEGCLOCK {float(seg.group(2)):.1f} s (no callback clock)')
        for it, result in checks:
            notes.append(f'replacement check at {it}: {result}')
        for c in counts:
            notes.append(f'reinit at {c.get("iteration")}: N {c.get("n_before")} -> {c.get("n_after")} (removed {c.get("n_covered")}, '
                         f'survivors {c.get("n_survivors")}, added {c.get("n_added")}, target {c.get("target")})')
        for f in fails:
            notes.append('reinit failed: ' + f.split(' reason=', 1)[-1][:200])
        if res and res.group(5):
            notes.append(f'{res.group(5)} ms/it (NeRFICG timer)')
        if end:
            notes.append(f'N at last step before cleanup {end.group(1)}; training-only peak {end.group(2)} GiB allocated / {end.group(3)} GiB reserved')
        if per:
            notes.append(f'lowest test image {per.group(2)} at {per.group(3)} dB ({per.group(1)} images)')
        if clock:
            notes.append(clock)
        if rundir:
            notes.append(f'run dir {rundir}')
        notes.append('LPIPS is VGG')
        train_s = (float(seg.group(2)) + cb_total) if seg else None
        rows.append(dict(date=dm.group(1) if dm else '', commit=provenance(b['method'], repo),
                         instance_type=f'ZIH Capella node {node} (job {job})', scene=b['scene'],
                         method=describe(b['method'], b['spec']), seed=b['seed'],
                         train_time_s=f'{train_s:.1f}' if train_s is not None else '',
                         peak_vram_gb=end.group(2) if end else (res.group(6) or '' if res else ''),
                         psnr=f'{float(res.group(1)):.3f}' if res else '', ssim=f'{float(res.group(2)):.4f}' if res else '',
                         lpips=f'{float(res.group(3)):.4f}' if res else '', gaussian_count=res.group(4) if res else '',
                         notes='; '.join(notes), train_resolution=RES.get(b['scene'], ''), role=role_of(b['name']), **CONST))
        seen.add(key)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--csv', default='experiment_log.csv')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    with open(a.csv, newline='', encoding='utf-8') as f:
        rd = csv.reader(f)
        header = next(rd)
        existing = list(rd)
    seen = {m.group(0) for r in existing for m in [re.match(r'run=[^;\s]+ job=[^;\s]+', r[header.index('notes')])] if m}
    rows = []
    for path in a.logs:
        rows += parse(path, seen)
    out = io.StringIO()
    w = csv.writer(out, lineterminator='\n')
    for r in rows:
        w.writerow([r.get(h, '') for h in header])
    print(f'{len(rows)} new row(s)' + (' (dry run, nothing written)' if a.dry_run else f' appended to {a.csv}'))
    if a.dry_run:
        print(out.getvalue())
    else:
        with open(a.csv, 'a', newline='', encoding='utf-8') as f:
            f.write(out.getvalue())


if __name__ == '__main__':
    main()
