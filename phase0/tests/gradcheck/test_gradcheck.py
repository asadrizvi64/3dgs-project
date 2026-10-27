"""Gradient correctness of both custom backward passes.

(a) torch.autograd.gradcheck on the extension, float64, both modes, N in {8, 32}, 32x32.
(b) fp32 analytic grads of each mode vs autograd of ref_render() in float64.
(c) the two modes agree with each other (fp32 and fp64).

All comparisons use the norm-based metric per parameter group:
    rel = ||g - g_ref||_2 / max(||g_ref||_2, 1e-12)
never an elementwise relative error.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from conftest import Scene
from splat2d.ops import MODES, Mode, ref_render, render

GROUPS = ("means", "chol", "opacity_logit", "colors")


def rel_error(g: Tensor, g_ref: Tensor) -> float:
    """Norm-based relative error of one parameter group (NOT elementwise)."""
    diff = torch.linalg.vector_norm((g.double().cpu() - g_ref.double().cpu()).flatten())
    ref = torch.linalg.vector_norm(g_ref.double().cpu().flatten())
    return float(diff) / max(float(ref), 1e-12)


def mode_grads(
    scene: Scene, mode: Mode, dtype: torch.dtype, device: torch.device
) -> dict[str, Tensor]:
    params = scene.params(dtype, device)
    image = render(*params, scene.height, scene.width, mode=mode)
    grads = torch.autograd.grad(image, params, grad_outputs=scene.grad_image(dtype, device))
    return {name: g.detach().cpu() for name, g in zip(GROUPS, grads, strict=True)}


def ref_grads(scene: Scene, input_dtype: torch.dtype) -> dict[str, Tensor]:
    """float64 autograd of ref_render at the parameters rounded to ``input_dtype``."""
    rounded = scene.params(input_dtype)
    m, c, o, col = (p.detach().double().requires_grad_(True) for p in rounded)
    params = (m, c, o, col)
    image = ref_render(*params, scene.height, scene.width)
    g_img = scene.grad_image(input_dtype).double()
    grads = torch.autograd.grad(image, params, grad_outputs=g_img)
    return {name: g.detach() for name, g in zip(GROUPS, grads, strict=True)}


@pytest.mark.parametrize("mode", MODES)
def test_gradcheck_float64(gradcheck_scene: Scene, mode: Mode, device: torch.device) -> None:
    scene = gradcheck_scene

    def fn(means: Tensor, chol: Tensor, opacity_logit: Tensor, colors: Tensor) -> Tensor:
        return render(means, chol, opacity_logit, colors, scene.height, scene.width, mode=mode)

    inputs = scene.params(torch.float64, device)
    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_fp32_matches_fp64_reference(
    accuracy_scene: Scene, mode: Mode, device: torch.device
) -> None:
    got = mode_grads(accuracy_scene, mode, torch.float32, device)
    want = ref_grads(accuracy_scene, torch.float32)
    for name in GROUPS:
        rel = rel_error(got[name], want[name])
        assert rel < 1e-5, f"{accuracy_scene.name} {mode} {name}: rel {rel:.3e}"


@pytest.mark.parametrize("mode", MODES)
def test_fp64_matches_fp64_reference(
    accuracy_scene: Scene, mode: Mode, device: torch.device
) -> None:
    got = mode_grads(accuracy_scene, mode, torch.float64, device)
    want = ref_grads(accuracy_scene, torch.float64)
    for name in GROUPS:
        rel = rel_error(got[name], want[name])
        assert rel < 1e-10, f"{accuracy_scene.name} {mode} {name}: rel {rel:.3e}"


@pytest.mark.parametrize(("dtype", "tol"), [(torch.float32, 3e-6), (torch.float64, 1e-12)])
def test_modes_agree(
    accuracy_scene: Scene, dtype: torch.dtype, tol: float, device: torch.device
) -> None:
    pixel = mode_grads(accuracy_scene, "pixel", dtype, device)
    splat = mode_grads(accuracy_scene, "splat", dtype, device)
    for name in GROUPS:
        rel = rel_error(splat[name], pixel[name])
        assert rel < tol, f"{accuracy_scene.name} {dtype} {name}: rel {rel:.3e}"


def test_forward_matches_reference(accuracy_scene: Scene, device: torch.device) -> None:
    s = accuracy_scene
    params = s.params(torch.float64, device)
    got = render(*params, s.height, s.width, mode="splat").detach().cpu()
    want = ref_render(*s.params(torch.float64), s.height, s.width).detach()
    assert rel_error(got, want) < 1e-12
