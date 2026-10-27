"""GPU parity: macro blend backward vs Faster-GS's stock (Taming-3DGS) blend backward.

One forward per scene, then _C.backward with mode 0 (taming, the reference) and mode 1 (macro) on the
same forward buffers. Every output is compared with the phase0 metric, ||macro - taming|| / ||taming||
(float64): the six parameter gradients and the rows of densification_info.

Needs the FasterGSMacro extension (phase1/make_variant.sh) and a GPU:
    python -m pytest -q phase1/tests/test_parity.py
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

import pytest
import torch

BACKEND = os.environ.get('FGS_BACKEND', 'FasterGSMacroCudaBackend')
TOL = float(os.environ.get('PARITY_TOL', '1e-5'))
TAMING, MACRO = 0, 1
GRAD_NAMES = ('means', 'scales', 'rotations', 'opacities', 'sh0', 'sh_rest')

if not torch.cuda.is_available():
    pytest.skip('needs a CUDA GPU', allow_module_level=True)
_C = importlib.import_module(f'{BACKEND}._C')


@dataclass(frozen=True)
class Scene:
    name: str
    n: int
    width: int
    height: int
    scale: tuple[float, float]       # world-space std-dev range (uniform)
    opacity: tuple[float, float]     # range of sigmoid(raw opacity)
    depth: tuple[float, float] = (2.0, 8.0)


SCENES = [
    Scene('tiny', 1_000, 160, 120, (0.01, 0.1), (0.05, 0.95)),
    Scene('partial_tiles', 20_000, 333, 250, (0.005, 0.08), (0.05, 0.95)),
    Scene('garden_res', 200_000, 1297, 840, (0.002, 0.03), (0.02, 0.99)),
    Scene('small_splats', 500_000, 1297, 840, (0.0005, 0.006), (0.02, 0.99)),
    Scene('heavy_overlap', 50_000, 640, 480, (0.05, 0.4), (0.3, 0.99)),
]


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    nb = torch.linalg.vector_norm(b).item()
    nd = torch.linalg.vector_norm(a - b).item()
    return nd / nb if nb > 0 else nd


def make_inputs(scene: Scene, seed: int, proper_antialiasing: bool) -> tuple[tuple, tuple, dict]:
    g = torch.Generator(device='cuda').manual_seed(seed)
    dev = 'cuda'

    def u(lo: float, hi: float, *shape: int) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(*shape, generator=g, device=dev)

    n, w, h = scene.n, scene.width, scene.height
    focal = 0.9 * w
    z = u(*scene.depth, n)
    # spread slightly beyond the frustum so some splats straddle / leave the image
    x = (u(-0.6, 0.6, n) * w / focal) * z
    y = (u(-0.6, 0.6, n) * h / focal) * z
    means = torch.stack([x, y, z], dim=-1).contiguous()
    scales = torch.log(u(*scene.scale, n, 3)).contiguous()
    rotations = torch.randn(n, 4, generator=g, device=dev).contiguous()
    o = u(*scene.opacity, n, 1)
    opacities = torch.log(o / (1 - o)).contiguous()
    sh0 = (0.8 * torch.randn(n, 1, 3, generator=g, device=dev)).contiguous()  # some colors < 0 (clamp mask)
    sh_rest = (0.1 * torch.randn(n, 15, 3, generator=g, device=dev)).contiguous()
    w2c = torch.eye(4, device=dev)
    cam_position = torch.zeros(3, device=dev)
    bg = torch.rand(3, generator=g, device=dev)
    settings = (w2c, cam_position, bg, 16, w, h, focal, focal, w / 2 + 0.3, h / 2 - 0.2, 0.2, 1000.0, proper_antialiasing)
    params = (means, scales, rotations, opacities, sh0, sh_rest)
    return params, settings, {'generator': g}


def run_both(scene: Scene, seed: int, abs_grad: bool | None, proper_antialiasing: bool) -> dict[str, float]:
    params, settings, extra = make_inputs(scene, seed, proper_antialiasing)
    means, scales, rotations, opacities, sh0, sh_rest = params
    image, pb, tb, ib, bb, n_inst, n_buckets, sel = _C.forward(*params, *settings)
    assert torch.isfinite(image).all()
    grad_image = torch.randn(image.shape, generator=extra['generator'], device='cuda')
    rows = 0 if abs_grad is None else (3 if abs_grad else 2)
    out = {}
    for mode in (TAMING, MACRO):
        di = torch.zeros(rows, scene.n, device='cuda') if rows else torch.empty(0, device='cuda')
        grads = _C.backward(di, grad_image, image, means, scales, rotations, opacities, sh_rest, pb, tb, ib, bb,
                            *settings, n_inst, n_buckets, sel, mode)
        torch.cuda.synchronize()
        out[mode] = (grads, di)
    errs = {}
    for name, m, t in zip(GRAD_NAMES, out[MACRO][0], out[TAMING][0]):
        assert torch.isfinite(m).all(), name
        errs[name] = rel_error(m, t)
    for r in range(rows):
        errs[f'densify{r}'] = rel_error(out[MACRO][1][r], out[TAMING][1][r])
    # the reference must actually produce gradients, else the comparison is vacuous
    assert torch.linalg.vector_norm(out[TAMING][0][0]).item() > 0
    print(f'PARITY scene={scene.name} n={scene.n} buckets={n_buckets} abs={abs_grad} pa={proper_antialiasing} '
          f'max={max(errs.values()):.3e} ' + ' '.join(f'{k}={v:.2e}' for k, v in errs.items()))
    return errs


@pytest.mark.parametrize('scene', SCENES, ids=[s.name for s in SCENES])
@pytest.mark.parametrize('abs_grad', [None, False, True], ids=['no_densify', 'densify', 'densify_abs'])
def test_parity(scene: Scene, abs_grad: bool | None) -> None:
    errs = run_both(scene, seed=0, abs_grad=abs_grad, proper_antialiasing=False)
    bad = {k: v for k, v in errs.items() if not v <= TOL}
    assert not bad, f'rel error above {TOL}: {bad}'


@pytest.mark.parametrize('scene', SCENES[:3], ids=[s.name for s in SCENES[:3]])
def test_parity_proper_antialiasing(scene: Scene) -> None:
    errs = run_both(scene, seed=1, abs_grad=True, proper_antialiasing=True)
    bad = {k: v for k, v in errs.items() if not v <= TOL}
    assert not bad, f'rel error above {TOL}: {bad}'


def test_macro_is_deterministic_up_to_atomics() -> None:
    """Two macro runs agree as closely as two taming runs do (only atomic order differs)."""
    scene = SCENES[1]
    params, settings, extra = make_inputs(scene, 2, False)
    image, pb, tb, ib, bb, n_inst, n_buckets, sel = _C.forward(*params, *settings)
    grad_image = torch.randn(image.shape, generator=extra['generator'], device='cuda')
    means, scales, rotations, opacities, sh0, sh_rest = params
    runs = [_C.backward(torch.empty(0, device='cuda'), grad_image, image, means, scales, rotations, opacities, sh_rest,
                        pb, tb, ib, bb, *settings, n_inst, n_buckets, sel, MACRO) for _ in range(2)]
    assert max(rel_error(a, b) for a, b in zip(*runs)) < 1e-6


def test_empty_scene() -> None:
    """All splats behind the camera: no instances, no buckets, zero gradients, no launch errors."""
    scene = Scene('empty', 100, 64, 48, (0.01, 0.1), (0.1, 0.9), depth=(-5.0, -2.0))
    params, settings, extra = make_inputs(scene, 3, False)
    image, pb, tb, ib, bb, n_inst, n_buckets, sel = _C.forward(*params, *settings)
    assert n_buckets == 0
    means, scales, rotations, opacities, sh0, sh_rest = params
    grads = _C.backward(torch.empty(0, device='cuda'), torch.ones_like(image), image, means, scales, rotations, opacities,
                        sh_rest, pb, tb, ib, bb, *settings, n_inst, n_buckets, sel, MACRO)
    torch.cuda.synchronize()
    assert all(float(gr.abs().sum()) == 0.0 for gr in grads)
