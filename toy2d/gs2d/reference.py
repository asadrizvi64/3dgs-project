"""Dense PyTorch rasterizer that follows the CUDA kernels' rules exactly.

Rules shared with csrc/rasterize.cu (change both or neither):
  * a Gaussian only reaches pixels in tiles inside its 3-sigma tile rect
  * power = -0.5 (A dx^2 + C dy^2) - B dx dy; skip if power > 0
  * alpha = min(0.99, opacity * exp(power)); skip if alpha < 1/255
  * front to back; a pixel stops before the splat that would push T below 1e-4
  * out = sum(c_i alpha_i T_i) + T_final * bg

Gradients come from autograd, so this is the ground truth for the kernels.
Cost is O(N * H * W) with a Python loop over N: use it on small scenes only.
"""
import torch

from .binning import TILE, Bins

ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
T_MIN = 1e-4


def render_reference(means, conic, color, opacity, bins: Bins, bg=None):
    H, W = bins.H, bins.W
    dtype, device = means.dtype, means.device
    if bg is None:
        bg = torch.zeros(3, dtype=dtype, device=device)
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    px, py = xs.to(dtype), ys.to(dtype)
    tx, ty = xs // TILE, ys // TILE
    rect = bins.rect.to(device).long()

    P = H * W
    T = torch.ones(P, dtype=dtype, device=device)
    C = torch.zeros(P, 3, dtype=dtype, device=device)
    done = torch.zeros(P, dtype=torch.bool, device=device)

    for g in bins.order.tolist():
        x0, y0, x1, y1 = rect[g].tolist()
        if x1 <= x0 or y1 <= y0:
            continue
        inrect = (tx >= x0) & (tx < x1) & (ty >= y0) & (ty < y1)
        dx = means[g, 0] - px
        dy = means[g, 1] - py
        power = -0.5 * (conic[g, 0] * dx * dx + conic[g, 2] * dy * dy) - conic[g, 1] * dx * dy
        alpha = torch.clamp(opacity[g] * torch.exp(power), max=ALPHA_MAX)
        with torch.no_grad():
            ok = inrect & (power <= 0) & (alpha >= ALPHA_MIN) & ~done
            test_T = T * (1 - alpha)
            term = ok & (test_T < T_MIN)
            done = done | term
            inc = ok & ~term
        a = torch.where(inc, alpha, torch.zeros_like(alpha))
        C = C + (a * T)[:, None] * color[g]
        T = T * (1 - a)

    out = C + T[:, None] * bg
    return out.T.reshape(3, H, W)
