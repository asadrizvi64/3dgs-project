"""How many Gaussians does a subset of image tiles activate?

For a trained Faster-GS checkpoint and a set of training views, this script selects a fraction of 16x16 tiles with
several policies, backpropagates an L1 loss restricted to those tiles through the real Faster-GS rasterizer, and counts
the Gaussians that receive a nonzero gradient. Those are exactly the rows a visibility-sparse Adam step would touch.

It also compares the lean-Adam visibility mask (n_touched_tiles > 0) with the Gaussians that receive a gradient from
the full image, which shows how many touched-but-not-contributing Gaussians sparse Adam updates anyway.

    cd $NERFICG && python <repo>/cluster/analysis/tile_vs_gaussian.py --run output/FasterGS/<run> --ckpt 0015000.pt

Output: one CSV row per (checkpoint, view, policy, budget) plus a summary printed as SUMMARY lines.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd() / 'scripts'))
import utils  # noqa: E402  (nerficg/scripts/utils.py)
with utils.DiscoverSourcePath():
    import Framework
    from Implementations import Methods as MI
    from Implementations import Datasets as DI
    from Datasets.utils import apply_background_color

TILE = 16
BLOCK = 4  # a block is BLOCK x BLOCK tiles (64 px)
BUDGETS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
POLICIES = ['rand_tile', 'rand_block', 'err_tile', 'err_block']

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--ckpt', nargs='+', required=True)
p.add_argument('--view-stride', type=int, default=4)
p.add_argument('--out', required=True)
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()
run = Path(args.run)

Framework.setup(config_path=str(run / 'training_config.yaml'), require_custom_config=True)
dataset = DI.get_dataset(dataset_type=Framework.config.GLOBAL.DATASET_TYPE, path=Framework.config.DATASET.PATH)
dataset.train()
views = [dataset[i] for i in range(0, len(dataset), args.view_stride)]
gen = torch.Generator(device='cpu').manual_seed(args.seed)

with utils.DiscoverSourcePath():
    from Methods.FasterGS import Renderer as _R
_Rasterize = sys.modules[_R.diff_rasterize.__module__]._Rasterize  # the class the renderer actually calls


def select(policy, frac, tile_err):
    """Boolean [ty, tx] tile mask with round(frac * n_tiles) tiles (block policies round to whole blocks)."""
    ty, tx = tile_err.shape
    n = ty * tx
    if frac >= 1.0:
        return torch.ones_like(tile_err, dtype=torch.bool)
    if policy.endswith('_tile'):
        k = max(1, round(frac * n))
        score = tile_err.flatten() if policy == 'err_tile' else torch.rand(n, generator=gen).to(tile_err.device)
        idx = score.topk(k).indices
        m = torch.zeros(n, dtype=torch.bool, device=tile_err.device)
        m[idx] = True
        return m.view(ty, tx)
    by, bx = math.ceil(ty / BLOCK), math.ceil(tx / BLOCK)
    pad = torch.zeros(by * BLOCK, bx * BLOCK, device=tile_err.device)
    pad[:ty, :tx] = tile_err
    berr = pad.view(by, BLOCK, bx, BLOCK).sum((1, 3)).flatten()
    score = berr if policy == 'err_block' else torch.rand(by * bx, generator=gen).to(tile_err.device)
    order = score.argsort(descending=True)
    m = torch.zeros(by * BLOCK, bx * BLOCK, dtype=torch.bool, device=tile_err.device)
    target, have = round(frac * n), 0
    for b in order.tolist():  # add whole blocks until the tile budget is reached
        if have >= target:
            break
        y, x = divmod(b, bx)
        m[y * BLOCK:(y + 1) * BLOCK, x * BLOCK:(x + 1) * BLOCK] = True
        have = int(m[:ty, :tx].sum())
    return m[:ty, :tx]


rows = []
for ck in args.ckpt:
    model = MI.get_model(method=Framework.config.GLOBAL.METHOD_TYPE, checkpoint=str(run / 'checkpoints' / ck))
    renderer = MI.get_renderer(method=Framework.config.GLOBAL.METHOD_TYPE, model=model)
    g = model.gaussians
    params = [g._means, g._scales, g._rotations, g._opacities, g._sh_coefficients_0, g._sh_coefficients_rest]
    for t in params:
        t.requires_grad_(True)
    N = g._means.shape[0]
    print(f'checkpoint {ck}: N={N} active_sh_bases={g.active_sh_bases} views={len(views)}', flush=True)

    def render(view, track=False):
        _Rasterize.track_visibility = track
        bg = view.camera.background_color.cuda()
        img = renderer.render_image_training(view=view, update_densification_info=False, bg_color=bg)
        gt = view.rgb.cuda()
        if (a := view.alpha) is not None:
            gt = apply_background_color(gt, a.cuda(), bg)
        return img, gt

    def active_rows():
        a = torch.zeros(N, dtype=torch.bool, device='cuda')
        for t in params:
            if t.grad is not None:
                a |= t.grad.reshape(N, -1).ne(0).any(1)
                t.grad = None
        return a

    for vi, view in enumerate(views):
        img, gt = render(view, track=True)
        touched = _Rasterize.last_visibility > 0
        err = (img.detach() - gt).abs().mean(0)  # [H, W]
        H, W = err.shape
        ty, tx = math.ceil(H / TILE), math.ceil(W / TILE)
        pe = torch.zeros(ty * TILE, tx * TILE, device='cuda')
        pe[:H, :W] = err
        tile_err = pe.view(ty, TILE, tx, TILE).mean((1, 3))
        (img - gt).abs().sum().backward()
        full = active_rows()
        n_full = int(full.sum())
        n_touched = int(touched.sum())
        rows.append(dict(ckpt=ck, view=vi, policy='visibility', budget=1.0, tile_frac=1.0, err_frac=1.0, n=N,
                         active=n_touched, active_of_n=n_touched / N, active_of_full=n_touched / max(n_full, 1),
                         touched_not_contributing=int((touched & ~full).sum()), contributing_not_touched=int((full & ~touched).sum())))
        for policy in POLICIES:
            for frac in BUDGETS:
                m = select(policy, frac, tile_err)
                pm = m.repeat_interleave(TILE, 0).repeat_interleave(TILE, 1)[:H, :W]
                img, gt = render(view)
                ((img - gt).abs() * pm).sum().backward()
                act = active_rows()
                na = int(act.sum())
                rows.append(dict(ckpt=ck, view=vi, policy=policy, budget=frac, tile_frac=float(m.float().mean()),
                                 err_frac=float((tile_err * m).sum() / tile_err.sum()), n=N, active=na, active_of_n=na / N,
                                 active_of_full=na / max(n_full, 1), touched_not_contributing='', contributing_not_touched=''))
        if vi % 10 == 0:
            print(f'  view {vi}: full-image contributors {n_full / N:.3f} of N, touched {n_touched / N:.3f}', flush=True)
    del model, renderer, g, params
    torch.cuda.empty_cache()

with open(args.out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# summary: mean over views
from collections import defaultdict  # noqa: E402
acc = defaultdict(list)
for r in rows:
    acc[(r['ckpt'], r['policy'], r['budget'])].append(r)
print('SUMMARY ckpt policy budget tile_frac err_frac active_of_n active_of_full')
for (ck, pol, b), rs in sorted(acc.items()):
    mean = lambda k: sum(float(r[k]) for r in rs) / len(rs)  # noqa: E731
    extra = ''
    if pol == 'visibility':
        extra = f" touched_not_contributing_of_n {mean('touched_not_contributing') / rs[0]['n']:.4f} contributing_not_touched {mean('contributing_not_touched'):.1f}"
    print(f"SUMMARY {ck} {pol} {b:.2f} {mean('tile_frac'):.3f} {mean('err_frac'):.3f} {mean('active_of_n'):.4f} {mean('active_of_full'):.4f}{extra}")
