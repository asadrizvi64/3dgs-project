"""GPU checks for alternative track R (R0 in alt_reinit/PLAN.md). Needs a GPU and the built backends:

    FGS_REF=FasterGSCudaBackend FGS_NEW=FasterGSReinitCudaBackend python -m pytest -q -s alt_reinit/tests/test_gpu.py

1. Depth from the real inference kernel with colours (z, 1, z^2): one opaque Gaussian, two overlapping layers at 1 and
   3 (nearly opaque together, expected depth ~2, flagged by the variance), and empty pixels (alpha exactly 0).
2. Flags off: the variant's backend reproduces the stock backend on identical inputs (forward image, all six gradients
   and the densification info, one FusedAdam step), within the stock-vs-stock run-to-run floor (atomics).
Prints CHECK lines for the job log.
"""
from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reinit_core as rc  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip('needs a CUDA GPU', allow_module_level=True)
REF = os.environ.get('FGS_REF', 'FasterGSCudaBackend')
NEW = os.environ.get('FGS_NEW', 'FasterGSReinitCudaBackend')
DEV = 'cuda'


def backend(name):
    r = importlib.import_module(f'{name}.torch_bindings.rasterization')
    a = importlib.import_module(f'{name}.torch_bindings.adam')
    return r, a


def settings(R, W, H, f, sh_bases=1, bg=None):
    # principal point on a pixel centre, so a Gaussian on the optical axis projects exactly onto pixel (W/2, H/2)
    return R.RasterizerSettings(torch.eye(4, device=DEV), torch.zeros(3, device=DEV),
                                torch.zeros(3, device=DEV) if bg is None else bg, sh_bases, W, H, f, f,
                                W / 2 + 0.5, H / 2 + 0.5, 0.2, 1000.0, False)


def gaussians(z, sigma_xy, opacity):
    n = len(z)
    means = torch.tensor([[0.0, 0.0, zz] for zz in z], device=DEV)
    scales = torch.log(torch.tensor([[s, s, 1e-3] for s in sigma_xy], device=DEV))
    rot = torch.tensor([[1.0, 0, 0, 0]] * n, device=DEV)
    op = torch.tensor([[math.log(o / (1 - o))] for o in opacity], device=DEV)
    return means.contiguous(), scales.contiguous(), rot.contiguous(), op.contiguous()


def depth_render(R, means, scales, rot, op, W=64, H=48, f=60.0):
    colors = rc.depth_colors(means, torch.eye(4, device=DEV))
    sh0 = rc.sh0_from_colors(colors)
    sh_rest = torch.zeros(means.shape[0], 15, 3, device=DEV)
    img = R.rasterize(means, scales, rot, op, sh0, sh_rest, settings(R, W, H, f), to_chw=True, clamp_output=False)
    torch.cuda.synchronize()
    return img


@pytest.mark.parametrize('name', [NEW])
def test_depth_single_layer(name):
    R, _ = backend(name)
    img = depth_render(R, *gaussians([2.0], [0.3], [0.999]))
    D, A, rel, has = rc.depth_from_render(img)
    c = (24, 32)
    print(f'CHECK depth single layer: D={D[c].item():.6f} A={A[c].item():.6f} rel_std={rel[c].item():.2e}')
    assert abs(D[c].item() - 2.0) < 1e-4 and abs(A[c].item() - 0.999) < 1e-4 and rel[c].item() < 1e-3
    assert img[0].max() > 1.0, 'output was clamped'
    inner = A > 0.5
    assert (D[inner] - 2.0).abs().max() < 1e-4   # a single Gaussian contributes its centre depth everywhere


@pytest.mark.parametrize('name', [NEW])
def test_depth_two_layers_and_empty(name):
    R, _ = backend(name)
    # front layer at 1 (alpha .5) and back layer at 3 (alpha .999); both cover the centre, neither reaches the corners
    img = depth_render(R, *gaussians([1.0, 3.0], [0.05, 0.15], [0.5, 0.999]))
    D, A, rel, has = rc.depth_from_render(img)
    c = (24, 32)
    w1, w2 = 0.5, 0.5 * 0.999
    d_exp = (w1 * 1.0 + w2 * 3.0) / (w1 + w2)
    var_exp = (w1 * 1.0 + w2 * 9.0) / (w1 + w2) - d_exp ** 2
    print(f'CHECK depth two layers: D={D[c].item():.4f} (expected {d_exp:.4f}) A={A[c].item():.4f} '
          f'rel_std={rel[c].item():.3f} (expected {math.sqrt(var_exp) / d_exp:.3f})')
    assert abs(D[c].item() - d_exp) < 1e-3 and abs(A[c].item() - (w1 + w2)) < 1e-3
    assert abs(rel[c].item() - math.sqrt(var_exp) / d_exp) < 1e-2
    cfg = rc.ReinitConfig()
    st = rc.pixel_status(A, rel, cfg)
    assert A[c] >= cfg.alpha_min and st[c] == rc.MIXTURE, 'nearly opaque mixture must be flagged by the variance'
    corners = [(0, 0), (0, 63), (47, 0), (47, 63)]
    print('CHECK depth empty corners: A=' + ','.join(f'{A[p].item():.1e}' for p in corners))
    assert all(A[p].item() == 0.0 and not has[p] and st[p] == rc.EMPTY for p in corners)


def random_scene(n=200_000, W=1297, H=840, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(*s, generator=g, device=DEV)  # noqa: E731
    f = 0.9 * W
    z = u(2.0, 8.0, n)
    means = torch.stack([u(-0.6, 0.6, n) * W / f * z, u(-0.6, 0.6, n) * H / f * z, z], -1).contiguous()
    scales = torch.log(u(0.002, 0.03, n, 3)).contiguous()
    rot = torch.randn(n, 4, generator=g, device=DEV).contiguous()
    o = u(0.02, 0.99, n, 1)
    op = torch.log(o / (1 - o)).contiguous()
    sh0 = (0.8 * torch.randn(n, 1, 3, generator=g, device=DEV)).contiguous()
    sh_rest = (0.1 * torch.randn(n, 15, 3, generator=g, device=DEV)).contiguous()
    grad_image = torch.randn(3, H, W, generator=g, device=DEV)
    return (means, scales, rot, op, sh0, sh_rest), (W, H, f), grad_image


def forward_backward(R, params, cam, grad_image):
    W, H, f = cam
    leaves = [p.detach().clone().requires_grad_(True) for p in params]
    info = torch.zeros(2, params[0].shape[0], device=DEV)
    s = R.RasterizerSettings(torch.eye(4, device=DEV), torch.zeros(3, device=DEV), torch.tensor([0.1, 0.2, 0.3], device=DEV),
                             16, W, H, f, f, W / 2, H / 2, 0.2, 1000.0, False)
    img = R.diff_rasterize(*leaves, info, s)
    img.backward(grad_image)
    torch.cuda.synchronize()
    return img.detach(), [p.grad.detach() for p in leaves], info


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    nb = torch.linalg.vector_norm(b).item()
    return torch.linalg.vector_norm(a - b).item() / nb if nb > 0 else torch.linalg.vector_norm(a - b).item()


NAMES = ('means', 'scales', 'rotations', 'opacities', 'sh0', 'sh_rest')


def test_flags_off_backend_equivalence():
    Rr, Ar = backend(REF)
    Rn, An = backend(NEW)
    params, cam, gimg = random_scene()
    img_r, g_r, info_r = forward_backward(Rr, params, cam, gimg)
    img_r2, g_r2, info_r2 = forward_backward(Rr, params, cam, gimg)    # stock vs stock: the run-to-run floor
    img_n, g_n, info_n = forward_backward(Rn, params, cam, gimg)
    f_floor, f_err = (img_r2 - img_r).abs().max().item(), (img_n - img_r).abs().max().item()
    print(f'CHECK forward image: max |new - stock| = {f_err:.3e} (stock vs stock {f_floor:.3e})')
    assert f_err <= 4 * f_floor if f_floor > 0 else torch.equal(img_n, img_r), 'forward differs'
    for name, a, b, c in zip(NAMES, g_n, g_r, g_r2):
        floor, err = rel(c, b), rel(a, b)
        print(f'CHECK grad {name}: rel(new, stock) = {err:.2e}, rel(stock, stock) = {floor:.2e}')
        assert err <= max(4 * floor, 1e-6), name
    for r in range(2):
        floor, err = rel(info_r2[r], info_r[r]), rel(info_n[r], info_r[r])
        print(f'CHECK densification info row {r}: rel(new, stock) = {err:.2e}, floor {floor:.2e}')
        assert err <= max(4 * floor, 1e-6)
    # one optimizer step from identical parameters and gradients: an elementwise kernel, so bit-identical
    group_names = ('means', 'scales', 'rotations', 'opacities', 'sh_coefficients_0', 'sh_coefficients_rest')
    outs = []
    for A in (Ar, An):
        ps = [torch.nn.Parameter(p.detach().clone()) for p in params]
        opt = A.FusedAdam([{'params': [p], 'lr': 1e-3, 'name': n} for p, n in zip(ps, group_names)], lr=0.0, eps=1e-15)
        for _ in range(2):
            for p, g in zip(ps, g_r):
                p.grad = g.clone()
            opt.step()
        torch.cuda.synchronize()
        outs.append(([p.detach() for p in ps], [(opt.state[p]['exp_avg'], opt.state[p]['exp_avg_sq']) for p in ps]))
    same = all(torch.equal(a, b) for a, b in zip(outs[0][0], outs[1][0])) and \
        all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(outs[0][1], outs[1][1]))
    print(f'CHECK FusedAdam 2 steps: parameters and both moments bit-identical: {same}')
    assert same
