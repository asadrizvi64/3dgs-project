"""Learnable 2D Gaussians and their activation into rasterizer inputs.

The rasterizer only sees four tensors per Gaussian:
  means   (N, 2)  pixel coordinates
  conic   (N, 3)  inverse covariance (A, B, C) = [[A, B], [B, C]]
  color   (N, 3)  RGB in [0, 1]
  opacity (N,)    in [0, 1]
Everything upstream of those (scale, rotation, sigmoid) stays in PyTorch autograd,
so the CUDA kernels only have to produce gradients for these four.
"""
import math

import torch


class Gaussians2D(torch.nn.Module):
    def __init__(self, n: int, H: int, W: int, seed: int = 0, dtype=torch.float32):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        base = math.sqrt(H * W / n) * 0.5  # rough spacing so they tile the image
        self.xy = torch.nn.Parameter(torch.rand(n, 2, generator=g, dtype=dtype) * torch.tensor([W, H], dtype=dtype))
        self.log_scale = torch.nn.Parameter(
            torch.log(base * (0.5 + torch.rand(n, 2, generator=g, dtype=dtype)))
        )
        self.theta = torch.nn.Parameter(torch.rand(n, generator=g, dtype=dtype) * math.pi)
        self.rgb_logit = torch.nn.Parameter(0.5 * torch.randn(n, 3, generator=g, dtype=dtype))
        self.opacity_logit = torch.nn.Parameter(0.5 * torch.randn(n, generator=g, dtype=dtype))
        # Fixed draw order (2D has no camera). Smaller depth = drawn first (front).
        self.register_buffer("depth", torch.rand(n, generator=g, dtype=dtype))

    def activated(self):
        return activate(self.xy, self.log_scale, self.theta, self.rgb_logit, self.opacity_logit)


def activate(xy, log_scale, theta, rgb_logit, opacity_logit):
    """Returns (means, conic, color, opacity, cov). cov = (xx, xy, yy) is used for radii."""
    s2 = torch.exp(2.0 * log_scale)
    c, sn = torch.cos(theta), torch.sin(theta)
    sx2, sy2 = s2[:, 0], s2[:, 1]
    cxx = c * c * sx2 + sn * sn * sy2
    cxy = c * sn * (sx2 - sy2)
    cyy = sn * sn * sx2 + c * c * sy2
    det = cxx * cyy - cxy * cxy
    conic = torch.stack([cyy / det, -cxy / det, cxx / det], dim=-1)
    color = torch.sigmoid(rgb_logit)
    opacity = torch.sigmoid(opacity_logit)
    cov = torch.stack([cxx, cxy, cyy], dim=-1)
    return xy, conic, color, opacity, cov


@torch.no_grad()
def radii_from_cov(cov: torch.Tensor) -> torch.Tensor:
    """3-sigma radius of the larger eigenvalue, as in the original 3DGS preprocess."""
    cxx, cxy, cyy = cov.unbind(-1)
    det = cxx * cyy - cxy * cxy
    mid = 0.5 * (cxx + cyy)
    lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    r = torch.ceil(3.0 * torch.sqrt(lam))
    return torch.where(det > 0, r, torch.zeros_like(r)).to(torch.int32)
