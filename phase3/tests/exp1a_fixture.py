# Usage: python phase3/tests/exp1a_fixture.py <dir> [scenario]
#        python phase3/analyze_ttq.py <dir>/gs-x1a-*.out --abl --seeds 30 31 32
# Synthetic exp1a logs (build_plus.sbatch stage exp1a) with known answers for analyze_ttq.py --abl.
# Every arm follows the same PSNR curve, so all cross the target at the same step and T = 40 + rate * (step - 5000):
# f_k = (rate_k - rate_mcmc) / (rate_ctl - rate_mcmc) exactly. Rates in the good scenario:
#   mrep 0.05 -> resolved; mreloc 0.9 -> R needed; mnoise 0.1 -> Z not needed; mreg 0.35 -> G not resolved;
#   mlr 0.6 -> L needed; creset g = 0.1 -> resets explain little.
# scenario:
#   good        as above
#   norep       mcmc is as slow as ctl                       -> step 1 fails, nothing classified
#   noisy       mrep's f = 0.5                               -> step 2 fails, descriptive only
#   notreached  mreloc ends 0.3 dB low and never reaches     -> f = inf, R still needed
#   resets      creset g = 0.8                               -> resets explain a major part
#   badnoise    mnoise logs noise steps (garden seed 31)     -> check 2 error, not evaluated
#   badswitch   mreloc runs with relocate=1 (garden seed 31) -> 1d and check 2 errors, not evaluated
#   fivewins    mcmc slower than ctl on garden seed 31        -> replication passes (5 of 6); 5 pairs used; classes as good
#   repnotreached  mrep never reaches the target on room seed 30 -> step 2 fails (not resolved)
#   lpips       mnoise ends +0.005 LPIPS worse than mcmc      -> Z not resolved instead of not needed
#   onefail     mcmc garden seed 30 has one image 4 dB low    -> one view failure against the control-based reference;
#               replication fails
#   clock       mreloc has no callback time, the others 0.002 s per step -> R needed on the training clock, not needed
#               with callbacks included: flagged as sensitive to the clock boundary
#   nocb        no FGS_ABL_CB lines (patch v1 logs)           -> classes as good; callback clock reported as not available
# Callback time (FGS_ABL_CB) is otherwise 0.001 s per step in every arm, so the two clocks agree.
import math
import os
import re
import sys

root, scen = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'good')
here = os.path.dirname(os.path.abspath(__file__))
BASECFG = open(os.path.join(here, 'exp1r_fixture_config.yaml'), encoding='utf-8').read()
ARMS = ('ctl', 'mcmc', 'mrep', 'mreloc', 'mnoise', 'mreg', 'mlr', 'creset')
MC = {'mcmc', 'mrep', 'mreloc', 'mnoise', 'mreg', 'mlr'}
RC, RM = 0.0080, 0.0060
FR = {'ctl': 1.0, 'mcmc': 0.0, 'mrep': 0.05, 'mreloc': 0.9, 'mnoise': 0.1, 'mreg': 0.35, 'mlr': 0.6, 'creset': 0.9}
if scen == 'norep':
    FR = {k: 1.0 for k in FR}
if scen == 'noisy':
    FR['mrep'] = 0.5
if scen == 'resets':
    FR['creset'] = 0.2
RATE = {a: RM + f * (RC - RM) for a, f in FR.items()}
OFF = {a: 0.0 for a in ARMS}
if scen == 'notreached':
    OFF['mreloc'] = -0.3
N0, BR, END = 2_000_000, 5000, 30000
SW = {'mreloc': (0, 1, 1), 'mnoise': (1, 0, 1), 'creset': (1, 1, 0)}


def cfg(name, iters=30000, **kv):
    t = BASECFG.replace('MODEL_NAME: X', f'MODEL_NAME: {name}').replace('NUM_ITERATIONS: 30000', f'NUM_ITERATIONS: {iters}')
    for k, v in kv.items():
        t, n = re.subn(rf'^(\s*{k}): [^#\n]*', rf'\g<1>: {v}', t, flags=re.M)
        assert n == 1, k
    return t


CB = {a: ((0.0 if a == 'mreloc' else 0.002) if scen == 'clock' else 0.001) for a in ARMS}
LP = {a: 0.125 if (scen == 'lpips' and a == 'mnoise') else 0.12 for a in ARMS}


def rate(a, sc, s):
    if scen == 'fivewins' and a == 'mcmc' and sc == 'garden' and s == 31:
        return RC * 1.02
    return RATE[a]


def off(a, sc, s):
    return -0.3 if scen == 'repnotreached' and a == 'mrep' and sc == 'room' and s == 30 else OFF[a]


def metrics(d, p, lp=0.12, low=None):
    os.makedirs(d, exist_ok=True)
    imgs = '\n'.join(f'{i}\t{p - (4.0 if i == low else 0.0):.2f}\t0.8\t{lp}' for i in range(10))
    open(os.path.join(d, 'metrics_8bit.txt'), 'w').write(
        f'Metric\tMean\tMedian\tPixelMean\nPSNR\t{p:.4f}\t{p:.4f}\t{p:.4f}\nSSIM\t0.8\t0.8\t0.8\nLPIPS\t{lp}\t{lp}\t{lp}\n\n'
        f'Image\tPSNR\tSSIM\tLPIPS\n{imgs}\nPSNR:{p:.10f} SSIM:0.8 LPIPS:{lp}\n')


def cbline(r_, lo, hi):
    if scen == 'nocb':
        return 'no callback timing in patch v1'
    return (f'FGS_ABL_CB total_s={r_ * (hi - lo):.4f} by=densify:{r_ * (hi - lo):.4f} calls=100 cum='
            + ','.join(f'{s_}:{r_ * max(0, min(s_, hi) - lo):.4f}' for s_ in range(1000, hi + 1, 1000)))


def views(path, seed, lo, hi):
    with open(path, 'w') as f:
        for i in range(lo, hi):
            v = (i * 7 + seed) % 272
            f.write(f'{i} 0 {v} {v}\n')


for sc, base, job in (('garden', 27.5, 'j1'), ('room', 31.0, 'j2')):
    run = os.path.join(root, f'run_{job}')
    os.makedirs(os.path.join(run, 'runs'), exist_ok=True)
    L = [f'node=c1 job={job} gpu=H100 stages=exp1a repo=x']
    for s in (30, 31, 32):
        fault = lambda kind: scen == kind and sc == 'garden' and s == 31
        L.append(f'EXP1R_ORDER {sc} s{s} ' + ('up pfx ' + ' '.join(ARMS) if s % 2 == 0 else 'pfx ' + ' '.join(ARMS) + ' up'))
        b = base + 0.01 * (s - 30)
        psnr = lambda a, st, b: b + off(a, sc, s) - 3.0 * math.exp(-(st - BR) / 4000)
        cum = lambda a, st: 40.0 + rate(a, sc, s) * (st - BR)
        name = f'b_{sc}_up_s{s}'
        L += [f'##### {name}: method=FasterGSUp seed={s} iters=30000',
              f'RESULT {name} PSNR:{psnr("ctl", END, b) + 0.01:.10f} SSIM:0.8 LPIPS:0.12 N_Gaussians:{N0 + 2_000_000} ms_it=6']
        os.makedirs(os.path.join(run, 'runs', name), exist_ok=True)
        open(os.path.join(run, 'runs', name, 'training_config.yaml'), 'w').write(cfg(name))
        metrics(os.path.join(run, 'runs', name, 'test_30000'), psnr('ctl', END, b) + 0.01)
        views(os.path.join(run, f'{name}.views'), s, 0, END)
        name = f'x_{sc}_pfx_s{s}'
        L += [f'##### {name}: method=FasterGSUp seed={s} iters=5000',
              f'FGS_SAVE_STATE step=5000 N={N0} cum_s=40.000 param_sha=p{s} optim_sha=o{s} dinfo_sha=d{s} path=x', cbline(0.002, 0, BR)]
        os.makedirs(os.path.join(run, 'runs', name), exist_ok=True)
        open(os.path.join(run, 'runs', name, 'training_config.yaml'), 'w').write(cfg(name, 5000))
        views(os.path.join(run, f'{name}.views'), s, 0, BR)
        for a in ARMS:
            name = f'x_{sc}_{a}_s{s}'
            mc = a in MC
            kv = {}
            if mc:
                kv = {'USE_MCMC': 'true', 'MAX_PRIMITIVES': N0, 'LAMBDA_OPACITY_REGULARIZATION': 0.0 if a == 'mreg' else 0.01,
                      'LAMBDA_SCALE_REGULARIZATION': 0.0 if a == 'mreg' else 0.01, 'LEARNING_RATE_OPACITIES': 0.025 if a == 'mlr' else 0.05}
            os.makedirs(os.path.join(run, 'runs', name), exist_ok=True)
            open(os.path.join(run, 'runs', name, 'training_config.yaml'), 'w').write(cfg(name, **kv))
            lr_op = (0.025 if a == 'mlr' else 0.05) if mc else 0.025
            lam = (0.0 if a == 'mreg' else 0.01) if mc else 0.0
            L += [f'##### {name}: method=FasterGSUp seed={s} iters=30000',
                  f'FGS_RESUME step=5000 N={N0} cum_s=40.000 sh_degree=3 param_sha=p{s} optim_sha=o{s} '
                  f'dinfo_sha={"none" if mc else f"d{s}"} densification_info=(2, {N0}) use_mcmc={mc} '
                  f'max_primitives={N0 if mc else 1000000} densification_end=14900 lr=means:0.0001,opacities:{lr_op} '
                  f'lambda_opacity={lam} lambda_scale={lam} from=x']
            for st in range(BR, END, 1000):
                L.append(f'FGS_EVALCHECK step={st} guard=1 rng_before=cpu:a,cuda:b,py:c,np:d rng_after_render=cpu:a,cuda:e,py:c,np:d '
                         f'rng_after_restore=cpu:a,cuda:b,py:c,np:d state_before=s{st} state_after=s{st} training_before=True '
                         f'training_after_render=False training_after_restore=True mode_before=train mode_after_render=test '
                         f'mode_after_restore=train')
            for st in range(BR + 1000, END + 1, 1000):
                n = N0 if mc else N0 + min(st - BR, 10000) * 200
                L.append(f'SEGCLOCK step={st} seg_s=1 cum_s={cum(a, st):.3f} sparse=0 N={n} peak_alloc_gib=5.00')
            nend = N0 if mc else N0 + 2_000_000
            sw = (1, 1, 1) if a == 'mreloc' and fault('badswitch') else SW.get(a, (1, 1, 1))
            events = 0 if (mc and sw[0] == 0) else (99 if mc else 0)
            noise = (END - BR if sw[1] and mc else 0) + (500 if a == 'mnoise' and fault('badnoise') else 0)
            resets = 'none' if mc or a == 'creset' else '6000,9000,12000'
            L += [f'FGS_TRAIN_END N={nend} peak_alloc_gib=5.00 peak_reserved_gib=6.00',
                  f'FGS_ABL relocate={sw[0]} noise={sw[1]} reset={sw[2]} use_mcmc={mc} relocation_events={events} '
                  f'relocated={events * 1000} dead_left={0 if events else 5000} noise_steps={noise} resets={resets}',
                  cbline(CB[a], BR, END),
                  f'FGS_WALL from=5000 to=30000 s={cum(a, END) - 40.0 + 130:.3f} views=0',
                  f'RESULT {name} PSNR:{psnr(a, END, b):.10f} SSIM:0.8 LPIPS:{LP[a]} N_Gaussians:{nend} ms_it=6']
            for st in range(BR, END + 1, 1000):
                low = 3 if (scen == 'onefail' and a == 'mcmc' and sc == 'garden' and s == 30 and st == END) else None
                metrics(os.path.join(run, 'runs', name, f'test_{st}'), psnr(a, st, b) if st > BR else b - 3.0,
                        LP[a] if st == END else 0.12, low)
            views(os.path.join(run, f'{name}.views'), s, BR, END)
    open(os.path.join(root, f'gs-x1a-{sc}-{job}.out'), 'w').write('\n'.join(L) + '\n')
print(f'fixture written to {root} (scenario: {scen})')
