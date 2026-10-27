"""Fit one image with 2D Gaussians.

    python fit.py --image photo.jpg --n 20000 --backend splat        # GPU
    python fit.py --size 64 --n 300 --iters 300 --backend ref        # CPU, slow but works
With no --image, a synthetic test pattern is used.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gs2d import Gaussians2D, bin_gaussians, rasterize
from gs2d.params import radii_from_cov


def synthetic(size):
    y, x = np.mgrid[0:size, 0:size] / size
    r = np.sin(6 * x) * 0.5 + 0.5
    g = ((x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.1).astype(float) * 0.8 + 0.1
    b = np.cos(9 * y + 3 * x) * 0.5 + 0.5
    return np.stack([r, g, b])


def load_target(path, size):
    if path is None:
        return torch.tensor(synthetic(size), dtype=torch.float32)
    img = Image.open(path).convert("RGB")
    if size:
        img.thumbnail((size, size))
    return torch.tensor(np.asarray(img), dtype=torch.float32).permute(2, 0, 1) / 255.0


def save(img, path):
    arr = (img.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--size", type=int, default=512, help="max side (synthetic: exact side)")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--backend", choices=["ref", "pixel", "splat"], default="splat")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cpu" if args.backend == "ref" else "cuda"
    target = load_target(args.image, args.size).to(device)
    _, H, W = target.shape
    torch.manual_seed(args.seed)
    model = Gaussians2D(args.n, H, W, seed=args.seed).to(device)
    opt = torch.optim.Adam([
        {"params": [model.xy], "lr": 0.002 * max(H, W) / 10},
        {"params": [model.log_scale], "lr": 0.01},
        {"params": [model.theta], "lr": 0.01},
        {"params": [model.rgb_logit], "lr": 0.02},
        {"params": [model.opacity_logit], "lr": 0.02},
    ])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for it in range(1, args.iters + 1):
        means, conic, color, opacity, cov = model.activated()
        bins = bin_gaussians(means, radii_from_cov(cov.detach()), model.depth, H, W)
        img = rasterize(means, conic, color, opacity, bins, backend=args.backend)
        loss = (img - target).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == args.iters:
            mse = ((img.detach().clamp(0, 1) - target) ** 2).mean().item()
            psnr = -10 * math.log10(max(mse, 1e-12))
            print(f"iter {it:5d}  L1 {loss.item():.4f}  PSNR {psnr:.2f} dB  {time.perf_counter() - t0:.1f}s")
    save(img, out_dir / f"fit_{args.backend}.png")
    save(target, out_dir / "target.png")
    print(f"saved to {out_dir}/")


if __name__ == "__main__":
    main()
