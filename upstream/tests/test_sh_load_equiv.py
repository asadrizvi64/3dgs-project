"""GPU equivalence: upstream Faster-GS main + upstream/sh_coalesced_loads.patch vs upstream main, on identical inputs.

The patch only changes how preprocess_backward_cu reads the SH-rest coefficients (warp-wide 16-byte loads into the shared
buffer that upstream already uses for the gradient stores, then the SH backward in place). The arithmetic is unchanged, so:
  - forward images bit-identical (forward is not touched), same instance and bucket counts;
  - SH-rest gradients bit-identical (they do not depend on atomics: grad_color comes from the blend backward, whose atomic
    order can differ, so they are compared bit-exactly only where the reference matches itself, else within TOL);
  - every other gradient within TOL relative error, with the reference-vs-reference atomic noise printed alongside;
  - for every SH degree (active_sh_bases 1, 4, 9, 16), for primitive counts that are not a multiple of 32, for a scene with
    most primitives off screen (many fully inactive warps), and for a coefficient tensor that is not 16-byte aligned
    (scalar fallback path). No empty scene: upstream launches the blend backward with zero blocks there, which leaves a
    pending CUDA error that breaks every later call in the process.

    FGS_REF=FasterGSUpPureCudaBackend FGS_NEW=FasterGSUpPureSCudaBackend python -m pytest -q -s upstream/tests/test_sh_load_equiv.py
"""
from __future__ import annotations

import importlib
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('needs a CUDA GPU', allow_module_level=True)
REF = importlib.import_module(f"{os.environ.get('FGS_REF', 'FasterGSUpPureCudaBackend')}._C")
NEW = importlib.import_module(f"{os.environ.get('FGS_NEW', 'FasterGSUpPureSCudaBackend')}._C")
TOL = 1e-5
SCENES = [  # name, n, width, height, scale range, opacity range, depth range
    ('tiny', 1_000, 160, 120, (0.01, 0.1), (0.05, 0.95), (2.0, 8.0)),
    ('odd_count', 20_001, 333, 250, (0.005, 0.08), (0.05, 0.95), (2.0, 8.0)),
    ('garden_res', 200_000, 1297, 840, (0.002, 0.03), (0.02, 0.99), (2.0, 8.0)),
    ('room_res', 300_007, 1557, 1038, (0.002, 0.03), (0.02, 0.99), (2.0, 8.0)),
    ('heavy_overlap', 50_000, 640, 480, (0.05, 0.4), (0.3, 0.99), (2.0, 8.0)),
    ('mostly_offscreen', 40_000, 320, 240, (0.01, 0.05), (0.05, 0.95), (2.0, 8.0)),
]


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    nb = torch.linalg.vector_norm(b).item()
    nd = torch.linalg.vector_norm(a - b).item()
    return nd / nb if nb > 0 else nd


def inputs(scene, seed, misaligned=False):
    name, n, w, h, sc, op, dp = scene
    g = torch.Generator(device='cuda').manual_seed(seed)
    u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(*s, generator=g, device='cuda')
    focal = 0.9 * w
    # distinct depths, so the depth sort (and the forward image) does not depend on atomic order
    z = dp[0] + (dp[1] - dp[0]) * (torch.randperm(n, generator=g, device='cuda').float() + 0.5) / n
    spread = 3.0 if name == 'mostly_offscreen' else 0.6   # most primitives outside the image: many inactive warps
    x = (u(-spread, spread, n) * w / focal) * z
    y = (u(-spread, spread, n) * h / focal) * z
    o = u(*op, n, 1)
    shr = 0.1 * torch.randn(n, 15, 3, generator=g, device='cuda')
    if misaligned:   # same values, data pointer 4 bytes past a 16-byte boundary
        buf = torch.empty(n * 45 + 1, device='cuda')
        shr_m = buf[1:].view(n, 15, 3)
        shr_m.copy_(shr)
        assert shr_m.data_ptr() % 16 == 4 and shr_m.is_contiguous()
        shr = shr_m
    params = (torch.stack([x, y, z], -1).contiguous(), torch.log(u(*sc, n, 3)).contiguous(),
              torch.randn(n, 4, generator=g, device='cuda').contiguous(), torch.log(o / (1 - o)).contiguous(),
              (0.8 * torch.randn(n, 1, 3, generator=g, device='cuda')).contiguous(), shr)
    return params, (w, h, focal), torch.randn(3, h, w, generator=g, device='cuda'), g


def settings(geom, g, active_sh_bases):
    w, h, focal = geom
    return (torch.eye(4, device='cuda'), torch.zeros(3, device='cuda'), torch.rand(3, generator=g, device='cuda'),
            active_sh_bases, w, h, focal, focal, w / 2 + 0.3, h / 2 - 0.2, 0.2, 1000.0, False)


def fwd_bwd(C, params, st, grad_image, rows):
    image, pb, tb, ib, bb, n_inst, n_buckets, sel = C.forward(*params, *st)
    di = torch.zeros(rows, params[0].shape[0], device='cuda') if rows else torch.empty(0, device='cuda')
    m, s, r, o, sh0, shr = params
    grads = C.backward(di, grad_image.reshape(image.shape), image, m, s, r, o, shr, pb, tb, ib, bb, *st,
                       n_inst, n_buckets, sel)
    torch.cuda.synchronize()
    return image, n_inst, n_buckets, grads, di


@pytest.mark.parametrize('scene', SCENES, ids=[s[0] for s in SCENES])
@pytest.mark.parametrize('sh', [1, 4, 9, 16], ids=['deg0', 'deg1', 'deg2', 'deg3'])
@pytest.mark.parametrize('rows', [0, 3], ids=['no_densify', 'densify_abs'])
@pytest.mark.parametrize('misaligned', [False, True], ids=['aligned', 'misaligned'])
def test_equivalent(scene, sh, rows, misaligned):
    params, geom, grad_image, g = inputs(scene, seed=7, misaligned=misaligned)
    st = settings(geom, g, sh)
    new = fwd_bwd(NEW, params, st, grad_image, rows)
    ref = fwd_bwd(REF, params, st, grad_image, rows)
    ref2 = fwd_bwd(REF, params, st, grad_image, rows)
    assert torch.equal(ref[0], new[0]), 'forward image differs'
    assert (ref[1], ref[2]) == (new[1], new[2]), f'instances/buckets {ref[1:3]} vs {new[1:3]}'
    names = ['means', 'scales', 'rotations', 'opacities', 'sh0', 'sh_rest']
    errs = {k: rel(a, b) for k, a, b in zip(names, new[3], ref[3])}
    errs.update({f'densify{r}': rel(new[4][r], ref[4][r]) for r in range(rows)})
    noise = max([rel(a, b) for a, b in zip(ref2[3], ref[3])] + [0.0])
    shr_exact = torch.equal(new[3][5], ref[3][5])
    ref_self_exact = torch.equal(ref2[3][5], ref[3][5])
    if sh == 1:   # degree 0: every SH-rest gradient must be exactly zero in both builds
        assert not new[3][5].any() and not ref[3][5].any(), 'degree 0: nonzero SH-rest gradients'
    if ref_self_exact:
        assert shr_exact, 'SH-rest gradients differ although the reference reproduces itself bit-exactly'
    print(f'EQUIV scene={scene[0]} sh={sh} rows={rows} misaligned={misaligned} instances={new[1]} '
          f'max_rel={max(errs.values()):.2e} ref_noise={noise:.2e} sh_rest_bitexact={shr_exact} ref_self_bitexact={ref_self_exact}')
    assert all(v <= TOL for v in errs.values()), errs
