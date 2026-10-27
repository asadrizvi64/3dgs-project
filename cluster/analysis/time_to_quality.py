"""Time-to-quality from a mid-training Faster-GS state: dense images vs sampled 64-px blocks.

Starts from checkpoints/state_<n>.pt (parameters + Adam state, written by cluster/patches/fastergs_save_state.py)
and continues training with no densification (as Faster-GS does after step 14,900). Policies:

  dense       stock Faster-GS step: full image, loss 0.8 * L1 + 0.2 * (1 - SSIM)
  err_block   FRAC of the 64x64 blocks, drawn without replacement with probability ~ block L1 error of the current
              render (so low-error blocks still get picked sometimes); same loss, averaged over the selected pixels
  rand_block  FRAC of the blocks, uniformly at random

Every policy uses the same code path (render, SSIM map, masked mean, stock fused Adam), so only the pixel selection
differs. Progress is counted in "image equivalents" (pixels supervised / pixels per image). The means learning-rate
schedule follows image equivalents, so the sampled runs take 1/FRAC more steps for the same schedule.

The current Faster-GS kernels still rasterize and backpropagate the full image and update all Gaussians even when a
block mask is used, so measured wall time does not show the savings a sparse rasterizer could give. Each eval row
therefore also carries a projected time from the cost model in results/tiles/README.md.

    cd $NERFICG && python <repo>/cluster/analysis/time_to_quality.py --run output/FasterGS/<run> --policy err_block
"""
import argparse
import csv
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
p.add_argument('--policy', choices=['dense', 'err_block', 'rand_block'], required=True)
p.add_argument('--frac', type=float, default=0.25)
p.add_argument('--images', type=int, default=15000, help='image equivalents to train (15000 = rest of the 30k schedule)')
p.add_argument('--eval-every', type=int, default=1500, help='in image equivalents')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--block', type=int, default=64)
p.add_argument('--adam', choices=['dense', 'sparse'], default='dense',
               help='sparse: update only Gaussian rows that received a nonzero gradient this step (lean-Adam kernel)')
p.add_argument('--out', required=True)
args = p.parse_args()
run = Path(args.run)
frac = 1.0 if args.policy == 'dense' else args.frac

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

# restore parameters and Adam state exactly as they were after step n
st = torch.load(run / 'checkpoints' / args.state, weights_only=False)
g = model.gaussians
for k, v in st['params'].items():
    setattr(g, k, torch.nn.Parameter(v.cuda().contiguous()))
g.training_setup(cfg.TRAINING, st['extent'])
g.optimizer.load_state_dict(st['optimizer'])
start_iter = st['iteration']
if args.adam == 'sparse':  # lean FusedAdam path; the row mask is set by hand before every step (see training loop)
    g.optimizer.sparse, g.optimizer.lean = True, True
    _Rasterize.track_visibility = False
params = [g._means, g._sh_coefficients_0, g._sh_coefficients_rest, g._opacities, g._scales, g._rotations]
N = g._means.shape[0]
print(f'state {args.state}: step {start_iter}, N={N}, active_sh_bases={g.active_sh_bases}, policy={args.policy} frac={frac}', flush=True)


def gt_of(view, bg):
    gt = view.rgb.cuda()
    if (a := view.alpha) is not None:
        gt = apply_background_color(gt, a.cuda(), bg)
    return gt


train_views = list(dataset.train())
test_views = list(dataset.test())
train_gt = [gt_of(v, v.camera.background_color.cuda()) for v in train_views]
test_gt = [gt_of(v, v.camera.background_color.cuda()) for v in test_views]


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


gen = torch.Generator(device='cuda').manual_seed(args.seed)
vgen = torch.Generator(device='cpu').manual_seed(args.seed)  # view order
C1, C2 = 0.01 ** 2, 0.03 ** 2
B = args.block


def block_mask(err):
    """[H, W] float mask (1 on selected 64-px blocks) or None for dense."""
    if args.policy == 'dense':
        return None
    H, W = err.shape
    by, bx = math.ceil(H / B), math.ceil(W / B)
    nb = by * bx
    k = max(1, round(frac * nb))
    if args.policy == 'err_block':
        pe = torch.zeros(by * B, bx * B, device='cuda')
        pe[:H, :W] = err
        w = pe.view(by, B, bx, B).sum((1, 3)).flatten() + 1e-8
    else:
        w = torch.ones(nb, device='cuda')
    idx = torch.multinomial(w, k, replacement=False, generator=gen)
    m = torch.zeros(nb, device='cuda')
    m[idx] = 1.0
    return m.view(by, bx).repeat_interleave(B, 0).repeat_interleave(B, 1)[:H, :W]


# Projected ms per step on a rasterizer that skips unselected pixels (results/tiles/README.md cost model; garden, step-20k
# stage times on H100: fwd 1.67, blend bwd 3.11, per-Gaussian bwd 1.42, dense Adam 3.23, lean sparse Adam 1.21 ms when it
# updated the 35.1% of rows touched by a full view). Inputs are measured every step with the actual training loss:
#   pix = fraction of pixels that can receive a gradient (selected blocks + 5-px SSIM halo)
#   a   = fraction of all Gaussian rows with a nonzero gradient
# Forward stays full (the error map comes from it); per-Gaussian backward scales with a when pixels are sampled; the
# optimizer term is dense Adam unless --adam sparse.
TOUCHED = 0.351


def proj_ms(pix, a):
    sampled = args.policy != 'dense'
    pre = 1.42 * (a / TOUCHED if sampled else 1.0)
    opt = 1.21 * a / TOUCHED if args.adam == 'sparse' else 3.23
    return 1.67 + 3.11 * pix + pre + opt

rows = []
steps_total = round(args.images / frac)
eval_every_steps = round(args.eval_every / frac)
train_s = 0.0
psnr0 = evaluate()
rows.append(dict(policy=args.policy, adam=args.adam, frac=frac, seed=args.seed, step=0, image_equiv=0, iteration_equiv=start_iter,
                 train_s=0.0, proj_s=0.0, test_psnr=psnr0, n=N, active_of_n='', pix_frac=''))
proj_s = 0.0
acc = torch.zeros(3, device='cuda')  # sums of active fraction and pixel fraction, step count (no per-step host sync)
print(f'EVAL step=0 img_eq=0 train_s=0.0 psnr={psnr0:.3f}', flush=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for step in range(1, steps_total + 1):
    it_eq = start_iter + (step - 1) * frac
    g.update_learning_rate(int(it_eq) + 1)
    vi = int(torch.randint(len(train_views), (1,), generator=vgen, device='cpu'))
    view, gt = train_views[vi], train_gt[vi]
    img = renderer.render_image_training(view=view, update_densification_info=False, bg_color=view.camera.background_color.cuda())
    l1 = (img - gt).abs()
    ssim = FusedSSIMMap.apply(C1, C2, img[None].contiguous(), gt[None], 'same', True)[0]
    m = block_mask(l1.detach().mean(0))
    if m is None:
        loss = 0.8 * l1.mean() + 0.2 * (1.0 - ssim.mean())
    else:
        denom = m.sum() * 3
        loss = 0.8 * (l1 * m).sum() / denom + 0.2 * (1.0 - (ssim * m).sum() / denom)
    loss.backward()
    act = torch.zeros(N, dtype=torch.bool, device='cuda')
    for t in params:
        act |= t.grad.reshape(N, -1).ne(0).any(1)
    pix = torch.ones((), device='cuda') if m is None else torch.nn.functional.max_pool2d(m[None, None], 11, 1, 5).mean()
    acc += torch.stack([act.float().mean(), pix.float(), torch.ones((), device='cuda')])
    if args.adam == 'sparse':
        _Rasterize.last_visibility = act.int()
    g.optimizer.step()
    g.optimizer.zero_grad()
    g.post_optimizer_step(inject_noise=False)
    if step % eval_every_steps == 0 or step == steps_total:
        torch.cuda.synchronize()
        train_s += time.perf_counter() - t0
        ps = evaluate()
        ie = step * frac
        a_m, pix_m, n_m = acc[0].item() / acc[2].item(), acc[1].item() / acc[2].item(), int(acc[2].item())
        proj_s += n_m * proj_ms(pix_m, a_m) / 1e3
        acc.zero_()
        rows.append(dict(policy=args.policy, adam=args.adam, frac=frac, seed=args.seed, step=step, image_equiv=ie, iteration_equiv=start_iter + ie,
                         train_s=round(train_s, 2), proj_s=round(proj_s, 2), test_psnr=ps, n=N, active_of_n=round(a_m, 4), pix_frac=round(pix_m, 4)))
        print(f'EVAL step={step} img_eq={ie:.0f} train_s={train_s:.1f} proj_s={proj_s:.1f} psnr={ps:.3f} active={a_m:.4f} pix={pix_m:.3f}', flush=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

new = not Path(args.out).exists()
with open(args.out, 'a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if new:
        w.writeheader()
    w.writerows(rows)
