"""Time-to-quality from a mid-training Faster-GS state: dense images vs sampled 64-px blocks.

Starts from checkpoints/state_<n>.pt (parameters + Adam state, written by cluster/patches/fastergs_save_state.py)
and continues training with no densification (as Faster-GS does after step 14,900).

Pixel policies:
  dense       stock Faster-GS step: full image, loss 0.8 * L1 + 0.2 * (1 - SSIM)
  err_block   FRAC of the 64x64 blocks, drawn without replacement with probability ~ block L1 error of the current
              render (so low-error blocks still get picked sometimes); same loss, averaged over the selected pixels
  rand_block  FRAC of the blocks, uniformly at random
Optimizer (--adam):
  dense       stock Faster-GS fused Adam over all rows
  sparse      lean-Adam kernel (fp32 moments) over the rows that received a nonzero gradient this step; the row mask
              is built after backward by scanning all parameter gradients

Every arm uses the same code path, so only the pixel selection and the optimizer rule differ. Progress is counted in
image equivalents (pixels supervised / pixels per image); the means learning-rate schedule follows image equivalents.

Timing. The current Faster-GS kernels rasterize and backpropagate the full image even when a block mask is used, so
measured wall time cannot show what a sparse rasterizer would save. Each eval row therefore also carries projected
seconds from a cost model calibrated on this same checkpoint by --calibrate (CUDA events):
  fwd                   dense render + loss
  bwd                   dense backward, split blend : per-Gaussian = 0.6875 : 0.3125 (nsys kernel shares at step 5k)
  dense_opt             dense Adam step + zero_grad + post-step
  sparse_opt(a)         linear fit of sparse Adam step + zero_grad + post-step against the active-row fraction a
  scan                  the gradient scan that builds the sparse row mask
  projected step = fwd + bwd * (0.6875 * pix + 0.3125 * (a / a_full if pixels are sampled else 1)) + opt
  where pix = measured fraction of pixels with a nonzero image gradient, a = measured active-row fraction, a_full = mean
  active fraction of a full image. proj_s excludes the mask scan, proj_scan_s includes it (sparse Adam arms only).
These projections are provisional: they assume blend work scales with pixels and per-Gaussian backward with active rows.

    cd $NERFICG && python <repo>/cluster/analysis/time_to_quality.py --run output/FasterGS/<run> --calibrate calib.json
    cd $NERFICG && python <repo>/cluster/analysis/time_to_quality.py --run ... --policy err_block --adam sparse \
        --calib calib.json --out ttq.csv
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd() / 'scripts'))
import utils  # noqa: E402  (nerficg/scripts/utils.py)
with utils.DiscoverSourcePath():
    import Framework
    from Implementations import Methods as MI
    from Implementations import Datasets as DI
    from Datasets.utils import apply_background_color

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--state', default='state_0015000.pt')
p.add_argument('--model-ckpt', default='0015000.pt')
p.add_argument('--policy', choices=['dense', 'err_block', 'rand_block'], default='dense')
p.add_argument('--adam', choices=['dense', 'sparse'], default='dense')
p.add_argument('--frac', type=float, default=0.25)
p.add_argument('--images', type=int, default=15000, help='image equivalents to train (15000 = rest of the 30k schedule)')
p.add_argument('--eval-every', type=int, default=1500, help='in image equivalents')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--block', type=int, default=64)
p.add_argument('--calibrate', help='write the cost-model calibration JSON here and exit (no training)')
p.add_argument('--calib', help='calibration JSON from --calibrate (needed for projected times)')
p.add_argument('--out')
args = p.parse_args()
run = Path(args.run)
BLEND, PERG = 0.6875, 0.3125

Framework.setup(config_path=str(run / 'training_config.yaml'), require_custom_config=True)
cfg = Framework.config
dataset = DI.get_dataset(dataset_type=cfg.GLOBAL.DATASET_TYPE, path=cfg.DATASET.PATH)
model = MI.get_model(method=cfg.GLOBAL.METHOD_TYPE, checkpoint=str(run / 'checkpoints' / args.model_ckpt))
renderer = MI.get_renderer(method=cfg.GLOBAL.METHOD_TYPE, model=model)
with utils.DiscoverSourcePath():
    try:
        from fused_ssim import FusedSSIMMap
    except ImportError:
        from Thirdparty.FusedSSIM import FusedSSIMMap
    from Methods.FasterGS import Renderer as _R
_Rasterize = sys.modules[_R.diff_rasterize.__module__]._Rasterize
_Rasterize.track_visibility = False

# restore parameters and Adam state exactly as they were after step n
st = torch.load(run / 'checkpoints' / args.state, weights_only=False)
g = model.gaussians
for k, v in st['params'].items():
    setattr(g, k, torch.nn.Parameter(v.cuda().contiguous()))
g.training_setup(cfg.TRAINING, st['extent'])
g.optimizer.load_state_dict(st['optimizer'])
start_iter = st['iteration']
params = [g._means, g._sh_coefficients_0, g._sh_coefficients_rest, g._opacities, g._scales, g._rotations]
N = g._means.shape[0]
print(f'state {args.state}: step {start_iter}, N={N}, active_sh_bases={g.active_sh_bases}, '
      f'policy={args.policy} adam={args.adam} frac={args.frac}', flush=True)


def set_adam(mode):
    g.optimizer.sparse = g.optimizer.lean = (mode == 'sparse')


def gt_of(view, bg):
    gt = view.rgb.cuda()
    if (a := view.alpha) is not None:
        gt = apply_background_color(gt, a.cuda(), bg)
    return gt


train_views = list(dataset.train())
test_views = list(dataset.test())
train_gt = [gt_of(v, v.camera.background_color.cuda()) for v in train_views]
test_gt = [gt_of(v, v.camera.background_color.cuda()) for v in test_views]
gen = torch.Generator(device='cuda').manual_seed(args.seed)
vgen = torch.Generator(device='cpu').manual_seed(args.seed)  # view order
C1, C2 = 0.01 ** 2, 0.03 ** 2
B = args.block


def block_mask(policy, frac, err):
    """[H, W] float mask (1 on selected blocks) or None for dense."""
    if policy == 'dense':
        return None
    H, W = err.shape
    by, bx = math.ceil(H / B), math.ceil(W / B)
    nb = by * bx
    k = max(1, round(frac * nb))
    if policy == 'err_block':
        pe = torch.zeros(by * B, bx * B, device='cuda')
        pe[:H, :W] = err
        w = pe.view(by, B, bx, B).sum((1, 3)).flatten() + 1e-8
    else:
        w = torch.ones(nb, device='cuda')
    idx = torch.multinomial(w, k, replacement=False, generator=gen)
    m = torch.zeros(nb, device='cuda')
    m[idx] = 1.0
    return m.view(by, bx).repeat_interleave(B, 0).repeat_interleave(B, 1)[:H, :W]


def forward_loss(vi, policy, frac):
    view, gt = train_views[vi], train_gt[vi]
    img = renderer.render_image_training(view=view, update_densification_info=False, bg_color=view.camera.background_color.cuda())
    img.retain_grad()
    l1 = (img - gt).abs()
    ssim = FusedSSIMMap.apply(C1, C2, img[None].contiguous(), gt[None], 'same', True)[0]
    m = block_mask(policy, frac, l1.detach().mean(0))
    if m is None:
        loss = 0.8 * l1.mean() + 0.2 * (1.0 - ssim.mean())
    else:
        denom = m.sum() * 3
        loss = 0.8 * (l1 * m).sum() / denom + 0.2 * (1.0 - (ssim * m).sum() / denom)
    return img, loss


def active_rows():
    act = torch.zeros(N, dtype=torch.bool, device='cuda')
    for t in params:
        act |= t.grad.reshape(N, -1).ne(0).any(1)
    return act


def opt_step(act):
    if g.optimizer.sparse:
        _Rasterize.last_visibility = act.int()
    g.optimizer.step()
    g.optimizer.zero_grad()
    g.post_optimizer_step(inject_noise=False)


# ---------------------------------------------------------------- calibration mode
if args.calibrate:
    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
    rec = {k: [] for k in ('fwd', 'bwd', 'scan', 'dense_opt')}
    pairs = []  # (active fraction, sparse opt ms)
    a_full = []
    for rep in range(36):
        vi = int(torch.randint(len(train_views), (1,), generator=vgen))
        policy = ['dense', 'err_block', 'rand_block'][rep % 3]
        e = [ev() for _ in range(5)]
        torch.cuda.synchronize()
        e[0].record()
        img, loss = forward_loss(vi, policy, args.frac)
        e[1].record()
        loss.backward()
        e[2].record()
        act = active_rows()
        e[3].record()
        mode = 'dense' if rep % 2 == 0 and policy == 'dense' else 'sparse'
        set_adam(mode)
        opt_step(act)
        e[4].record()
        torch.cuda.synchronize()
        if rep < 3:
            continue  # warm-up
        a = float(act.float().mean())
        rec['scan'].append(e[2].elapsed_time(e[3]))
        if policy == 'dense':
            rec['fwd'].append(e[0].elapsed_time(e[1]))
            rec['bwd'].append(e[1].elapsed_time(e[2]))
            a_full.append(a)
        if mode == 'dense':
            rec['dense_opt'].append(e[3].elapsed_time(e[4]))
        else:
            pairs.append((a, e[3].elapsed_time(e[4])))
    xs = torch.tensor([x for x, _ in pairs], dtype=torch.float64)
    ys = torch.tensor([y for _, y in pairs], dtype=torch.float64)
    A = torch.stack([torch.ones_like(xs), xs], 1)
    c0, c1 = torch.linalg.lstsq(A, ys[:, None]).solution.flatten().tolist()
    med = lambda v: float(torch.tensor(v).median())  # noqa: E731
    out = dict(N=N, fwd=med(rec['fwd']), bwd=med(rec['bwd']), scan=med(rec['scan']), dense_opt=med(rec['dense_opt']),
               sparse_opt_c0=c0, sparse_opt_c1=c1, a_full=sum(a_full) / len(a_full),
               sparse_pairs=[[round(x, 4), round(y, 3)] for x, y in pairs], n_dense_opt=len(rec['dense_opt']))
    Path(args.calibrate).write_text(json.dumps(out, indent=1))
    print('CALIB ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k != 'sparse_pairs'}), flush=True)
    sys.exit(0)

# ---------------------------------------------------------------- training mode
assert args.out, '--out is required for training'
frac = 1.0 if args.policy == 'dense' else args.frac
set_adam(args.adam)
cal = json.loads(Path(args.calib).read_text()) if args.calib else None


def proj_ms(pix, a):
    if cal is None:
        return float('nan'), float('nan')
    sampled = args.policy != 'dense'
    bwd = cal['bwd'] * (BLEND * pix + PERG * (a / cal['a_full'] if sampled else 1.0))
    if args.adam == 'sparse':
        opt = cal['sparse_opt_c0'] + cal['sparse_opt_c1'] * a
        return cal['fwd'] + bwd + opt, cal['fwd'] + bwd + opt + cal['scan']
    t = cal['fwd'] + bwd + cal['dense_opt']
    return t, t


@torch.no_grad()
def evaluate():
    model.eval()
    ps = []
    for v, gt in zip(test_views, test_gt):
        img = renderer.render_image_inference(v, to_chw=True)['rgb']
        img8, gt8 = (img * 255).round() / 255, (gt.clamp(0, 1) * 255).round() / 255
        ps.append(-10 * torch.log10(((img8 - gt8) ** 2).mean()).item())
    model.train()
    return sum(ps) / len(ps)


rows = []
steps_total = round(args.images / frac)
eval_every_steps = round(args.eval_every / frac)
train_s = proj_s = proj_scan_s = 0.0
psnr0 = evaluate()
base = dict(policy=args.policy, adam=args.adam, frac=frac, seed=args.seed, n=N)
rows.append(dict(base, step=0, image_equiv=0, iteration_equiv=start_iter, train_s=0.0, proj_s=0.0, proj_scan_s=0.0,
                 test_psnr=psnr0, active_of_n='', pix_frac=''))
print(f'EVAL step=0 img_eq=0 train_s=0.0 psnr={psnr0:.3f}', flush=True)
acc = torch.zeros(3, device='cuda')  # sums of active fraction and pixel fraction, step count (no per-step host sync)
torch.cuda.synchronize()
t0 = time.perf_counter()
for step in range(1, steps_total + 1):
    g.update_learning_rate(int(start_iter + (step - 1) * frac) + 1)
    vi = int(torch.randint(len(train_views), (1,), generator=vgen))
    img, loss = forward_loss(vi, args.policy, frac)
    loss.backward()
    act = active_rows()
    pix = img.grad.ne(0).any(0).float().mean()  # pixels that actually carry an image gradient (blocks + SSIM halo)
    acc += torch.stack([act.float().mean(), pix, torch.ones((), device='cuda')])
    opt_step(act)
    if step % eval_every_steps == 0 or step == steps_total:
        torch.cuda.synchronize()
        train_s += time.perf_counter() - t0
        ps = evaluate()
        ie = step * frac
        n_m = acc[2].item()
        a_m, pix_m = acc[0].item() / n_m, acc[1].item() / n_m
        pm, pms = proj_ms(pix_m, a_m)
        proj_s += n_m * pm / 1e3
        proj_scan_s += n_m * pms / 1e3
        acc.zero_()
        rows.append(dict(base, step=step, image_equiv=ie, iteration_equiv=start_iter + ie, train_s=round(train_s, 2),
                         proj_s=round(proj_s, 2), proj_scan_s=round(proj_scan_s, 2), test_psnr=ps,
                         active_of_n=round(a_m, 4), pix_frac=round(pix_m, 4)))
        print(f'EVAL step={step} img_eq={ie:.0f} train_s={train_s:.1f} proj_s={proj_s:.1f} proj_scan_s={proj_scan_s:.1f} '
              f'psnr={ps:.3f} active={a_m:.4f} pix={pix_m:.3f}', flush=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

new = not Path(args.out).exists()
with open(args.out, 'a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if new:
        w.writeheader()
    w.writerows(rows)
print(f'DONE policy={args.policy} adam={args.adam} seed={args.seed} final_img_eq={rows[-1]["image_equiv"]:.0f} psnr={rows[-1]["test_psnr"]:.3f}', flush=True)
