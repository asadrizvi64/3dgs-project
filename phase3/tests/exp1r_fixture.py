# Usage: python phase3/tests/exp1r_fixture.py <dir> [fault]
#        python phase3/analyze_ttq.py <dir>/gs-x1r-*.out --exp1r --seeds 10 11 12
# Synthetic exp1r logs (build_plus.sbatch stage exp1r) with known answers for analyze_ttq.py's Experiment 1R check 1.
# fault (applied to garden seed 11 only):
#   none      every check passes; 1e reports ctl - reference = -0.01 dB on every seed
#   views     ctl sees a different view at step 12000                       -> 1a fails
#   resume    stop loaded a different parameter digest                      -> 1b fails
#   evalcheck mcmc's evaluation at 9000 left the random state changed        -> 1c fails
#   config    ctl's configuration differs from the reference in LAMBDA_DSSIM -> 1d fails
#   effect    ctl runs with opacity learning rate 0.05 against 0.025 configured -> 1d fails
#   prefix    the prefix's configuration differs from the reference in SH_DEGREE -> 1d fails
import math
import os
import re
import sys

root, fault = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'none')
here = os.path.dirname(os.path.abspath(__file__))
BASECFG = open(os.path.join(here, 'exp1r_fixture_config.yaml'), encoding='utf-8').read()
ARMS = ('ctl', 'stop', 'mcmc', 'oracle')
N0 = 2_000_000
CAP = N0 + 2_000_000


def cfg(name, iters=30000, **kv):
    t = BASECFG.replace('MODEL_NAME: X', f'MODEL_NAME: {name}').replace('NUM_ITERATIONS: 30000', f'NUM_ITERATIONS: {iters}')
    for k, v in kv.items():
        t, n = re.subn(rf'^(\s*{k}): [^#\n]*', rf'\g<1>: {v}', t, flags=re.M)
        assert n == 1, k
    return t


def psnr_at(arm, step, base):
    off = {'ctl': 0.0, 'stop': 0.0, 'mcmc': -0.2, 'oracle': 0.02}[arm]
    return base + off - 3.0 * math.exp(-(step - 5000) / 4000)


def cum(arm, step):
    return 40.0 + {'ctl': 0.0080, 'stop': 0.0055, 'mcmc': 0.0060, 'oracle': 0.0078}[arm] * (step - 5000)


def metrics(d, p):
    os.makedirs(d, exist_ok=True)
    imgs = '\n'.join(f'{i}\t{p:.2f}\t0.8\t0.12' for i in range(10))
    open(os.path.join(d, 'metrics_8bit.txt'), 'w').write(
        f'Metric\tMean\tMedian\tPixelMean\nPSNR\t{p:.4f}\t{p:.4f}\t{p:.4f}\nSSIM\t0.8\t0.8\t0.8\nLPIPS\t0.12\t0.12\t0.12\n\n'
        f'Image\tPSNR\tSSIM\tLPIPS\n{imgs}\nPSNR:{p:.10f} SSIM:0.8 LPIPS:0.12\n')


def views(path, seed, lo, hi, change=None):
    with open(path, 'w') as f:
        for i in range(lo, hi):
            v = (i * 7 + seed) % 272
            if i == change:
                v = (v + 1) % 272
            f.write(f'{i} 0 {v} {v}\n')


for sc, base, job in (('garden', 27.5, 'j1'), ('room', 31.0, 'j2')):
    run = os.path.join(root, f'run_{job}')
    os.makedirs(os.path.join(run, 'runs'), exist_ok=True)
    L = [f'node=c1 job={job} gpu=H100 stages=exp1r repo=x']
    for s in (10, 11, 12):
        f = (lambda kind: fault == kind and sc == 'garden' and s == 11)
        L.append(f'EXP1R_ORDER {sc} s{s} ' + ('up pfx ctl stop mcmc oracle' if s % 2 == 0 else 'pfx ctl stop mcmc oracle up'))
        b = base + 0.01 * (s - 10)
        # reference
        name = f'b_{sc}_up_s{s}'
        L += [f'##### {name}: method=FasterGSUp seed={s} iters=30000',
              f'RESULT {name} PSNR:{psnr_at("ctl", 30000, b) + 0.01:.10f} SSIM:0.8 LPIPS:0.12 N_Gaussians:{CAP + 5000} ms_it=6']
        os.makedirs(os.path.join(run, 'runs', name), exist_ok=True)
        open(os.path.join(run, 'runs', name, 'training_config.yaml'), 'w').write(cfg(name))
        metrics(os.path.join(run, 'runs', name, 'test_30000'), psnr_at('ctl', 30000, b) + 0.01)
        views(os.path.join(run, f'{name}.views'), s, 0, 30000)
        # prefix
        name = f'x_{sc}_pfx_s{s}'
        L += [f'##### {name}: method=FasterGSUp seed={s} iters=5000',
              f'FGS_SAVE_STATE step=5000 N={N0} cum_s=40.000 param_sha=p{s} optim_sha=o{s} dinfo_sha=d{s} path=x']
        os.makedirs(os.path.join(run, 'runs', name), exist_ok=True)
        open(os.path.join(run, 'runs', name, 'training_config.yaml'), 'w').write(
            cfg(name, 5000, **({'SH_DEGREE': 2} if f('prefix') else {})))
        views(os.path.join(run, f'{name}.views'), s, 0, 5000)
        for a in ARMS:
            name = f'x_{sc}_{a}_s{s}'
            mc = a in ('mcmc', 'oracle')
            kv = {}
            if a == 'stop':
                kv = {'DENSIFICATION_END_ITERATION': 5000}
            if mc:
                kv = {'USE_MCMC': 'true', 'MAX_PRIMITIVES': N0 if a == 'mcmc' else CAP, 'LAMBDA_OPACITY_REGULARIZATION': 0.01,
                      'LAMBDA_SCALE_REGULARIZATION': 0.01, 'LEARNING_RATE_OPACITIES': 0.05}
            if a == 'ctl' and f('config'):
                kv = {'LAMBDA_DSSIM': 0.3}
            os.makedirs(os.path.join(run, 'runs', name), exist_ok=True)
            open(os.path.join(run, 'runs', name, 'training_config.yaml'), 'w').write(cfg(name, **kv))
            lr_op = 0.05 if mc or (a == 'ctl' and f('effect')) else 0.025
            lam = 0.01 if mc else 0.0
            L += [f'##### {name}: method=FasterGSUp seed={s} iters=30000',
                  f'FGS_RESUME step=5000 N={N0} cum_s=40.000 sh_degree=3 param_sha={"pX" if a == "stop" and f("resume") else f"p{s}"} '
                  f'optim_sha=o{s} dinfo_sha={"none" if mc else f"d{s}"} densification_info=(2, {N0}) use_mcmc={mc} '
                  f'max_primitives={N0 if a == "mcmc" else CAP if a == "oracle" else 1000000} '
                  f'densification_end={5000 if a == "stop" else 14900} lr=means:0.0001,opacities:{lr_op} '
                  f'lambda_opacity={lam} lambda_scale={lam} from=x']
            for st in range(5000, 30000, 1000):
                rb = 'cpu:a,cuda:b,py:c,np:d'
                ra = 'cpu:a,cuda:Z,py:c,np:d' if a == 'mcmc' and st == 9000 and f('evalcheck') else rb
                L.append(f'FGS_EVALCHECK step={st} guard=1 rng_before={rb} rng_after_render=cpu:a,cuda:e,py:c,np:d '
                         f'rng_after_restore={ra} state_before=s{st} state_after=s{st} training_before=True '
                         f'training_after_render=False training_after_restore=True mode_before=train mode_after_render=test '
                         f'mode_after_restore=train')
            for st in range(6000, 30001, 1000):
                n = {'ctl': N0 + min(st - 5000, 10000) * 200, 'stop': N0, 'mcmc': N0,
                     'oracle': CAP if st >= 15000 else N0 + 1000}[a]
                L.append(f'SEGCLOCK step={st} seg_s=1 cum_s={cum(a, st):.3f} sparse=0 N={n} peak_alloc_gib=5.00')
            nend = {'ctl': CAP, 'stop': N0, 'mcmc': N0, 'oracle': CAP}[a]
            L += [f'FGS_TRAIN_END N={nend} peak_alloc_gib=5.00 peak_reserved_gib=6.00',
                  f'FGS_WALL from=5000 to=30000 s={cum(a, 30000) - 40.0 + 130:.3f} views=0',
                  f'RESULT {name} PSNR:{psnr_at(a, 30000, b):.10f} SSIM:0.8 LPIPS:0.12 N_Gaussians:{nend} ms_it=6']
            for st in range(5000, 30001, 1000):
                metrics(os.path.join(run, 'runs', name, f'test_{st}'), psnr_at(a, st, b) if st > 5000 else b - 3.0)
            views(os.path.join(run, f'{name}.views'), s, 5000, 30000, 12000 if a == 'ctl' and f('views') else None)
    open(os.path.join(root, f'gs-x1r-{sc}-{job}.out'), 'w').write('\n'.join(L) + '\n')
print(f'fixture written to {root} (fault: {fault})')
