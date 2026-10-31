"""Record every training run of a phase2/build_plus.sbatch job in experiment_log.csv (one row per run, failed runs too).

    python phase2/record_runs.py phase2/logs/<job-name>-<job>.out [more logs]  [--dry-run]

Reads the job log (run headers, SEGCLOCK / FGS_N / FGS_TRAIN_END / PERIMG / RESULT / FAILED lines) and the archived
run files in phase2/logs/run_<job>/runs/<run name>/ (rundir.txt, gpu_clock_summary.txt). Each row's notes start with
'run=<run name> job=<job>', which is the de-duplication key: a run already in the log is not added again.

Columns follow the existing log. Train time is the CUDA-event clock (SEGCLOCK cum_s). peak_vram_gb is the
training-only peak (GiB) from clock hook v4 when present; otherwise the trainer's own figure, read after the final
evaluation (the notes say which). role: stage-render runs (r_*, intermediate renders every N iterations; not
for timing claims) -> stage-render, stock -> baseline, runs shorter than 30k -> check, everything else -> experiment.
"""
import argparse
import csv
import io
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument('logs', nargs='+')
ap.add_argument('--csv', default='experiment_log.csv')
ap.add_argument('--dry-run', action='store_true')
args = ap.parse_args()
sys.stdout.reconfigure(encoding='utf-8')

RES = {'bicycle': '1237x822', 'flowers': '1256x828', 'garden': '1297x840', 'stump': '1245x825', 'treehill': '1267x832',
       'room': '1557x1038', 'counter': '1558x1038', 'kitchen': '1558x1039', 'bonsai': '1559x1039'}
CONST = dict(gpu='NVIDIA H100 SXM 94GB', driver='580.159.04', cuda='12.8', torch='2.11.0+cu128')

with open(args.csv, newline='', encoding='utf-8') as f:
    rd = csv.reader(f)
    header = next(rd)
    existing = list(rd)
seen = {m.group(0) for r in existing for m in [re.match(r'run=\S+ job=\d+', r[header.index('notes')])] if m}


def describe(method, flags, blend):
    if method == 'FasterGS':
        return 'Faster-GS main, stock (taming backward, dense Adam, fp32 moments)'
    parts = ['A macro backward' if blend == 'macro' else 'taming backward']
    m = re.search(r'SELECTIVE_FROM=(\d+)', flags)
    parts.append(f'B selective Adam from {m.group(1)}' if m else 'dense Adam')
    parts.append('C bf16 SH-rest moments' if 'SH_REST_BF16_MOMENTS=true' in flags else 'fp32 moments')
    if method == 'FasterGSPlusRB':
        parts.append('fewer host read-backs (phase2/fastergs_readback.patch)')
    if method == 'FasterGSPlusCull':
        parts.append('per-subtile ellipse culling in both blend kernels (phase2/ellipse_cull.patch)')
    if method == 'FasterGSPlusStats':
        parts.append('blend work counters (phase2/blend_stats.patch)')
    return f'{method} ({", ".join(parts)})'


new_rows = []
for path in args.logs:
    text = open(path, encoding='utf-8', errors='replace').read()
    hd = re.search(r'^node=(\S+) job=(\S+) gpu=.*? repo=(\S+)', text, re.M)
    if not hd:
        sys.exit(f'{path}: no job header line')
    node, job, repo = hd.groups()
    archive = os.path.join(os.path.dirname(path), f'run_{job}', 'runs')
    blocks, cur = [], None
    for line in text.splitlines():
        m = re.match(r'##### (\S+): method=(\S+) seed=(\d+) iters=(\d+) flags=(\S*) blend=(\S+)', line)
        if m:
            cur = dict(name=m.group(1), method=m.group(2), seed=m.group(3), iters=int(m.group(4)), flags=m.group(5),
                       blend=m.group(6), lines=[])
            blocks.append(cur)
            continue
        if line.startswith('#####'):
            cur = None
        if cur is not None:
            cur['lines'].append(line)
    for b in blocks:
        key = f'run={b["name"]} job={job}'
        if key in seen:
            continue
        L = b['lines']
        grab = lambda pat: next((m for m in (re.match(pat, l) for l in L) if m), None)
        clk = [m for m in (re.match(r'SEGCLOCK step=(\d+) \S+ cum_s=(\S+) sparse=(\d)(?: visfrac=(\S+))?(?: N=(\d+) peak_alloc_gib=(\S+))?', l) for l in L) if m]
        clk = clk[-1] if clk else None
        res = grab(r'RESULT \S+ PSNR:(\S+) SSIM:(\S+) LPIPS:(\S+) N_Gaussians:(\d+)(?:\s+ms_it=(\S*))?(?:\s+allocated: ([\d.]+))?')
        n15 = grab(r'FGS_N iteration=15000 N=(\d+)')
        end = grab(r'FGS_TRAIN_END N=(\d+) peak_alloc_gib=(\S+) peak_reserved_gib=(\S+)')
        per = grab(r'PERIMG \S+ n_images=(\d+) min_image=(\d+) min_psnr=(\S+)')
        failed = any(l.startswith(f'FAILED {b["name"]}') for l in L)
        sm = re.match(r'[brptx]_([a-z]+)_', b['name'])
        scene = sm.group(1) if sm else 'garden'
        rundir, clock = '', ''
        adir = os.path.join(archive, b['name'])
        if os.path.exists(os.path.join(adir, 'rundir.txt')):
            rundir = open(os.path.join(adir, 'rundir.txt')).read().strip()
        if os.path.exists(os.path.join(adir, 'gpu_clock_summary.txt')):
            g = open(os.path.join(adir, 'gpu_clock_summary.txt')).read()
            mm = re.search(r'sm_mhz_mean=(\d+) sm_mhz_min=(\d+).*?power_w_mean=(\d+).*?temp_c_max=(\d+)', g)
            if mm:
                clock = f'SM clock mean {mm.group(1)} / min {mm.group(2)} MHz, {mm.group(3)} W mean, max {mm.group(4)} C'
        if not per:   # older jobs have no PERIMG line: take the lowest test image from the archived metrics file
            mf = [os.path.join(adir, d, 'metrics_8bit.txt') for d in sorted(os.listdir(adir))] if os.path.isdir(adir) else []
            mf = [p for p in mf if os.path.exists(p)]
            if mf:
                imgs = [(float(m.group(2)), m.group(1)) for m in re.finditer(r'^(\d+)\t([\d.]+)', open(mf[-1]).read().split('Image', 1)[-1], re.M)]
                if imgs:
                    lo = min(imgs)
                    per = re.match(r'(\d+) (\d+) (\S+)', f'{len(imgs)} {lo[1]} {lo[0]:.2f}')
        dm = re.search(r'_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}-\d{2}$', rundir)
        notes = [key, f'phase2 {b["iters"]}-iteration run' + (f', flags {b["flags"]}' if b['flags'] else '') + f', blend backward {b["blend"]}',
                 'train time = CUDA-event clock (SEGCLOCK)']
        if failed or not res or not clk:
            notes.append('FAILED: no final metrics or no training clock record' if not failed else 'FAILED (see job log)')
        if res and res.group(5):
            notes.append(f'{res.group(5)} ms/it (NeRFICG timer)')
        if b['name'].startswith('t_'):
            notes.append('timer-control run: NeRFICG per-step timer ' + ('OFF (no device sync after each step)' if '_off_' in b['name'] else 'ON (device sync after each step, the benchmark protocol)'))
        walls = [m.groups() for m in (re.match(r'FGS_WALL from=(\d+) to=(\d+) s=([\d.]+) views=(\w+)', l) for l in L) if m]
        if walls:
            notes.append('wall clock (sync only at window ends) ' + ', '.join(f'{a}-{b2}: {t} s' for a, b2, t, _ in walls) + f'; view hash {walls[-1][3]}')
        if clk and clk.group(4):
            notes.append(f'visible fraction {clk.group(4)} at the end')
        if n15:
            notes.append(f'N at 15k {n15.group(1)}')
        if end:
            notes.append(f'N at last step before cleanup {end.group(1)}; training-only peak {end.group(2)} GiB allocated / {end.group(3)} GiB reserved')
        elif res:
            if res.group(6):
                notes.append(f'VRAM {res.group(6)} GiB allocated, read after the final evaluation (not training-only)')
        if per:
            notes.append(f'lowest test image {per.group(2)} at {per.group(3)} dB ({per.group(1)} images)')
        if b['name'].startswith('p_') and b['method'] == 'FasterGSPlusStats':
            notes.append('blend-counter run (phase2/blend_stats.patch: extra atomics and an ellipse test per splat in both blend kernels), so its timing is not a benchmark value')
        elif b['name'].startswith('p_'):
            notes.append('profiling run (FGS_PROF=1: host sync every step, CUDA events every 250 steps), so its timing is not a benchmark value')
        if b['name'].startswith('r_'):
            notes.append('stage-render run: test renders and metrics every 1000 iterations, so its timing and quality are not benchmark values')
        if clock:
            notes.append(clock)
        if rundir:
            notes.append(f'run dir {rundir}')
        notes.append('LPIPS is VGG')
        hook = 'v5' if walls else ('v4 or later' if (clk and clk.group(5)) else 'v3')
        commit = (f'FasterGS@1144604 + sparse_from {hook} hooks' if b['method'] == 'FasterGS'
                  else f'{b["method"]} (1144604 + phase1 macro' + (' + phase2 readback patch' if b['method'] == 'FasterGSPlusRB' else ' + phase2 blend counters' if b['method'] == 'FasterGSPlusStats' else ' + phase2 ellipse culling' if b['method'] == 'FasterGSPlusCull' else '') + f' + sparse_from {hook})') + f'; 3dgs-project {repo}'
        row = dict(date=dm.group(1) if dm else '', commit=commit, instance_type=f'ZIH Capella node {node} (job {job})',
                   scene=scene, method=describe(b['method'], b['flags'], b['blend']), seed=b['seed'],
                   train_time_s=f'{float(clk.group(2)):.1f}' if clk else '',
                   peak_vram_gb=end.group(2) if end else (res.group(6) or '' if res else ''),
                   psnr=f'{float(res.group(1)):.3f}' if res else '', ssim=f'{float(res.group(2)):.4f}' if res else '',
                   lpips=f'{float(res.group(3)):.4f}' if res else '', gaussian_count=res.group(4) if res else '',
                   notes='; '.join(notes), train_resolution=RES.get(scene, ''),
                   role='stage-render' if b['name'].startswith('r_') else 'timer-control' if b['name'].startswith('t_') else 'profile' if b['name'].startswith('p_') else 'check' if b['iters'] < 30000 else ('baseline' if b['method'] == 'FasterGS' or b['name'].endswith(('a0b0c0_s0', 'a0b0c0_s1', 'a0b0c0_s2')) else 'experiment'),
                   **CONST)
        new_rows.append([row.get(h, '') for h in header])
        seen.add(key)

out = io.StringIO()
w = csv.writer(out, lineterminator='\n')
for r in new_rows:
    w.writerow(r)
print(f'{len(new_rows)} new row(s)' + (' (dry run, nothing written)' if args.dry_run else f' appended to {args.csv}'))
if args.dry_run:
    print(out.getvalue())
else:
    with open(args.csv, 'a', newline='', encoding='utf-8') as f:
        f.write(out.getvalue())
