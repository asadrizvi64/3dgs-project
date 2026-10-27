"""CUDA kernels vs the fp64 PyTorch reference. Gate G0: fp32 relative error < 1e-5."""
import pytest
import torch

from gs2d import rasterize, render_reference
from tests.scenes import PARAM_NAMES, SCENES, make_scene, rel_err, upstream_grad

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOL = 1e-5


def _cuda_inputs(leaves):
    return [t.detach().float().cuda().requires_grad_(True) for t in leaves]


@pytest.mark.parametrize("scene", list(SCENES))
@pytest.mark.parametrize("backend", ["pixel", "splat"])
def test_cuda_matches_reference(scene, backend):
    leaves, bins, bg = make_scene(scene)
    dout = upstream_grad(bins.H, bins.W)
    ref = render_reference(*leaves, bins, bg)
    ref_grads = torch.autograd.grad(ref, leaves, dout)

    x = _cuda_inputs(leaves)
    out = rasterize(*x, bins.to("cuda"), bg.float().cuda(), backend=backend)
    assert rel_err(out, ref) < TOL, f"forward {rel_err(out, ref):.2e}"
    grads = torch.autograd.grad(out, x, dout.float().cuda())
    for name, a, b in zip(PARAM_NAMES, grads, ref_grads):
        e = rel_err(a, b)
        assert e < TOL, f"{backend}/{scene}/{name}: {e:.2e}"
