"""Small test scenes that hit the kernels' edge cases."""
import torch

from gs2d import Gaussians2D, bin_gaussians
from gs2d.params import radii_from_cov

# name: (n, H, W, scale_mult, opacity_shift, seed)
SCENES = {
    # small enough for finite-difference gradcheck
    "tiny": (12, 18, 20, 1.5, 0.0, 4),
    # odd size, so edge tiles have pixels outside the image
    "sparse_odd": (60, 37, 41, 1.0, 0.0, 1),
    # > 256 splats per tile (several shared-memory rounds, many buckets) and early termination
    "dense": (300, 20, 20, 6.0, 0.0, 2),
    # near-opaque: alpha clamping at 0.99 and early stop
    "opaque": (80, 32, 48, 1.5, 6.0, 3),
}


def make_scene(name, dtype=torch.float64, device="cpu", bg=(0.2, 0.5, 0.8)):
    n, H, W, scale_mult, op_shift, seed = SCENES[name]
    g = Gaussians2D(n, H, W, seed=seed, dtype=torch.float64)
    with torch.no_grad():
        g.log_scale += torch.log(torch.tensor(scale_mult, dtype=torch.float64))
        g.opacity_logit += op_shift
    means, conic, color, opacity, cov = (t.detach() for t in g.activated())
    bins = bin_gaussians(means, radii_from_cov(cov), g.depth, H, W)
    leaves = [t.to(device=device, dtype=dtype).clone().requires_grad_(True) for t in (means, conic, color, opacity)]
    bg = torch.tensor(bg, dtype=dtype, device=device)
    return leaves, bins.to(device), bg


def upstream_grad(H, W, seed=0, dtype=torch.float64, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(3, H, W, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)


def rel_err(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


PARAM_NAMES = ("means", "conic", "color", "opacity")
