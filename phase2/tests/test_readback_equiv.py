"""GPU equivalence: FasterGSPlusRB (phase2/fastergs_readback.patch) vs FasterGSPlus on identical inputs.

The readback patch merges the two count read-backs after preprocessing into one round trip, replaces the host wait on the
memset stream with a GPU-side event wait, and drops the host read of the bucket count (buffers sized for an upper bound;
both backward kernels read the real count on the GPU). None of this may change results:
  - forward images bit-identical, same instance count;
  - bucket count: >= the reference's (exact) count, <= floor((n_instances + 31 * n_tiles) / 32) with 16x12 tiles, and 0
    when there are no instances;
  - backward (stock taming kernel, mode 0, and the macro kernel, mode 3): every gradient and densification row within
    1e-6 relative error of the reference (only atomic order differs);
  - 30 forward/backward calls in a row on different inputs, without host syncs in between, keep agreeing (no stale reuse
    of the pinned count buffer).

    FGS_REF=FasterGSPlusCudaBackend FGS_NEW=FasterGSPlusRBCudaBackend python -m pytest -q phase2/tests/test_readback_equiv.py
"""
from __future__ import annotations

import importlib
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('needs a CUDA GPU', allow_module_level=True)
REF = importlib.import_module(f"{os.environ.get('FGS_REF', 'FasterGSPlusCudaBackend')}._C")
NEW = importlib.import_module(f"{os.environ.get('FGS_NEW', 'FasterGSPlusRBCudaBackend')}._C")
TOL = 1e-6
SCENES = [  # name, n, width, height, scale range, opacity range, depth range
    ('tiny', 1_000, 160, 120, (0.01, 0.1), (0.05, 0.95), (2.0, 8.0)),
    ('partial_tiles', 20_000, 333, 250, (0.005, 0.08), (0.05, 0.95), (2.0, 8.0)),
    ('garden_res', 200_000, 1297, 840, (0.002, 0.03), (0.02, 0.99), (2.0, 8.0)),
    ('room_res', 300_000, 1557, 1038, (0.002, 0.03), (0.02, 0.99), (2.0, 8.0)),
    ('heavy_overlap', 50_000, 640, 480, (0.05, 0.4), (0.3, 0.99), (2.0, 8.0)),
    ('empty', 100, 64, 48, (0.01, 0.1), (0.1, 0.9), (-5.0, -2.0)),
]


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    nb = torch.linalg.vector_norm(b).item()
    nd = torch.linalg.vector_norm(a - b).item()
    return nd / nb if nb > 0 else nd


def inputs(scene, seed):
    name, n, w, h, sc, op, dp = scene
    g = torch.Generator(device='cuda').manual_seed(seed)
    u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(*s, generator=g, device='cuda')
    focal = 0.9 * w
    # distinct depths (a shuffled even grid): with tied depths the order of tied primitives after the depth sort follows
    # preprocess's atomics, so even stock Faster-GS is not bit-reproducible and an exact image comparison would be noise
    z = dp[0] + (dp[1] - dp[0]) * (torch.randperm(n, generator=g, device='cuda').float() + 0.5) / n
    x = (u(-0.6, 0.6, n) * w / focal) * z
    y = (u(-0.6, 0.6, n) * h / focal) * z
    o = u(*op, n, 1)
    params = (torch.stack([x, y, z], -1).contiguous(), torch.log(u(*sc, n, 3)).contiguous(),
              torch.randn(n, 4, generator=g, device='cuda').contiguous(), torch.log(o / (1 - o)).contiguous(),
              (0.8 * torch.randn(n, 1, 3, generator=g, device='cuda')).contiguous(),
              (0.1 * torch.randn(n, 15, 3, generator=g, device='cuda')).contiguous())
    settings = (torch.eye(4, device='cuda'), torch.zeros(3, device='cuda'), torch.rand(3, generator=g, device='cuda'),
                16, w, h, focal, focal, w / 2 + 0.3, h / 2 - 0.2, 0.2, 1000.0, False)
    grad_image = torch.randn(3, h, w, generator=g, device='cuda')
    return params, settings, grad_image


def fwd_bwd(C, params, settings, grad_image, mode, rows, sync=True, backward=True):
    image, pb, tb, ib, bb, n_inst, n_buckets, sel = C.forward(*params, *settings)
    if not backward:
        torch.cuda.synchronize()
        return image, n_inst, n_buckets, None, None
    di = torch.zeros(rows, params[0].shape[0], device='cuda') if rows else torch.empty(0, device='cuda')
    m, s, r, o, sh0, shr = params
    grads = C.backward(di, grad_image.reshape(image.shape), image, m, s, r, o, shr, pb, tb, ib, bb, *settings,
                       n_inst, n_buckets, sel, mode)
    if sync:
        torch.cuda.synchronize()
    return image, n_inst, n_buckets, grads, di


@pytest.mark.parametrize('scene', SCENES, ids=[s[0] for s in SCENES])
@pytest.mark.parametrize('mode', [0, 3], ids=['taming', 'macro'])
@pytest.mark.parametrize('rows', [0, 3], ids=['no_densify', 'densify_abs'])
def test_equivalent(scene, mode, rows):
    params, settings, grad_image = inputs(scene, seed=7)
    # stock Faster-GS launches the taming backward with zero blocks when nothing is visible, which leaves a pending CUDA
    # error that breaks the next call (the RB patch guards it, like the macro path already did): skip the reference's
    # backward there and check only that the new build runs it cleanly
    ref_bwd = not (scene[0] == 'empty' and mode == 0)
    new = fwd_bwd(NEW, params, settings, grad_image, mode, rows)
    ref = fwd_bwd(REF, params, settings, grad_image, mode, rows, backward=ref_bwd)
    assert torch.equal(ref[0], new[0]), 'forward image differs'
    assert ref[1] == new[1], f'instances {ref[1]} vs {new[1]}'
    n_tiles = ((scene[2] + 15) // 16) * ((scene[3] + 11) // 12)  # 16x12 tiles (rasterization_config.h)
    bound = 0 if new[1] == 0 else (new[1] + 31 * n_tiles) // 32
    assert ref[2] <= new[2] <= bound, f'buckets ref {ref[2]} new {new[2]} bound {bound}'
    if new[1] == 0:
        assert new[2] == 0
    if not ref_bwd:
        assert all(torch.isfinite(t).all() and not t.any() for t in new[3] if t.numel()), 'empty scene: nonzero gradients'
        print(f'EQUIV scene={scene[0]} mode={mode} rows={rows} instances=0 (reference backward skipped)')
        return
    errs = {f'g{i}': rel(a, b) for i, (a, b) in enumerate(zip(new[3], ref[3]))}
    errs.update({f'densify{r}': rel(new[4][r], ref[4][r]) for r in range(rows)})
    print(f'EQUIV scene={scene[0]} mode={mode} rows={rows} instances={new[1]} buckets ref={ref[2]} new={new[2]} '
          f'bound={bound} max_rel={max(errs.values()) if errs else 0:.2e}')
    assert all(v <= TOL for v in errs.values()), errs


def test_back_to_back_without_sync():
    """30 calls on changing inputs, no host sync in between: instance counts and images keep matching the reference."""
    scene = SCENES[1]
    outs_new, outs_ref = [], []
    for k in range(30):
        params, settings, grad_image = inputs(scene, seed=100 + k)
        outs_new.append(fwd_bwd(NEW, params, settings, grad_image, 3, 0, sync=False)[:2])
    torch.cuda.synchronize()
    for k in range(30):
        params, settings, grad_image = inputs(scene, seed=100 + k)
        outs_ref.append(fwd_bwd(REF, params, settings, grad_image, 3, 0)[:2])
    for k, ((i_new, n_new), (i_ref, n_ref)) in enumerate(zip(outs_new, outs_ref)):
        assert n_new == n_ref and torch.equal(i_new, i_ref), f'call {k}'
