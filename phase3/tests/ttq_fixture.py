# Usage: python phase3/tests/ttq_fixture.py <dir>; python phase3/analyze_ttq.py <dir>/gs-x1-*.out --ref <dir>/gs-up-*.out
# Synthetic exp1 logs with known answers for phase3/analyze_ttq.py:
#   stop reaches the target 20% earlier than ctl in every pair, same final quality -> stop PASS
#   mcmc never reaches the target (final 0.2 dB lower) -> FAIL; oracle reaches it 12% earlier -> PASS (reference)
import os, sys, math
root = sys.argv[1]
os.makedirs(root, exist_ok=True)
ARMS = ('ctl', 'stop', 'mcmc', 'oracle')
def psnr_at(arm, step, base):
    # saturating curve; final value at 30000 = base + offset
    off = {'ctl': 0.0, 'stop': 0.0, 'mcmc': -0.2, 'oracle': 0.02}[arm]
    return base + off - 3.0 * math.exp(-(step - 5000) / 4000)
def cum(arm, step):
    rate = {'ctl': 0.0080, 'stop': 0.0055, 'mcmc': 0.0060, 'oracle': 0.0078}[arm]  # s per step after the branch
    return 40.0 + rate * (step - 5000)
for sc, base, job in (('garden', 27.5, 'j1'), ('room', 31.0, 'j2')):
    lines = [f'node=c1 job={job} gpu=H100 stages=exp1 repo=x']
    for s in (0, 1, 2):
        lines.append(f'##### x_{sc}_pfx_s{s}: method=FasterGSUp')
        for a in ARMS:
            name = f'x_{sc}_{a}_s{s}'
            n0 = 2_000_000
            lines.append(f'##### {name}: method=FasterGSUp seed={s}')
            lines.append(f'FGS_RESUME step=5000 N={n0} cum_s=40.000 sh_degree=3')
            for st in range(6000, 30001, 1000):
                n = {'ctl': n0 + min(st - 5000, 10000) * 200, 'stop': n0, 'mcmc': n0, 'oracle': n0 + 2_000_000 if st >= 15000 else n0 + 1000}[a]
                lines.append(f'SEGCLOCK step={st} seg_s=1 cum_s={cum(a, st):.3f} sparse=0 N={n} peak_alloc_gib=5.00')
            nend = {'ctl': n0 + 2_000_000, 'stop': n0, 'mcmc': n0, 'oracle': n0 + 2_000_000}[a]
            lines.append(f'FGS_TRAIN_END N={nend} peak_alloc_gib=5.00 peak_reserved_gib=6.00')
            lines.append(f'RESULT {name} PSNR:{psnr_at(a, 30000, base + 0.01 * s):.2f} SSIM:0.8 LPIPS:0.12 N_Gaussians:{nend} ms_it=6')
            for st in range(5000, 30001, 1000):
                d = os.path.join(root, f'run_{job}', 'runs', name, f'test_{st}')
                os.makedirs(d, exist_ok=True)
                p = psnr_at(a, st, base + 0.01 * s) if st > 5000 else base - 3.0
                imgs = '\n'.join(f'{i}\t{p:.2f}\t0.8\t0.12' for i in range(10))
                open(os.path.join(d, 'metrics_8bit.txt'), 'w').write(
                    f'Metric\tMean\tMedian\tPixelMean\nPSNR\t{p:.4f}\t{p:.4f}\t{p:.4f}\nSSIM\t0.8\t0.8\t0.8\nLPIPS\t0.12\t0.12\t0.12\n\nImage\tPSNR\tSSIM\tLPIPS\n{imgs}\n')
    open(os.path.join(root, f'gs-x1-{sc}-{job}.out'), 'w').write('\n'.join(lines) + '\n')
    ref = [f'node=c1 job=r{job} gpu=H100']
    for s in (0, 1, 2):
        ref.append(f'##### b_{sc}_up_s{s}: method=FasterGSUp')
        ref.append(f'RESULT b_{sc}_up_s{s} PSNR:{base + 0.02 * (s - 1) + 0.01:.2f} SSIM:0.8 LPIPS:0.12 N_Gaussians:{4_000_000 + 30_000 * (s - 1)} ms_it=6')
    open(os.path.join(root, f'gs-up-{sc}-r{job}.out'), 'w').write('\n'.join(ref) + '\n')
print('ok')
