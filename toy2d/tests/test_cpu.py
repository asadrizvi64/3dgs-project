"""CPU tests: binning, the PyTorch reference, and the Python mirror of the kernels."""
import pytest
import torch

from gs2d import TILE, render_reference
from tests import kernel_emulator as emu
from tests.scenes import PARAM_NAMES, SCENES, make_scene, rel_err, upstream_grad


def test_binning_lists_are_complete_and_depth_sorted():
    leaves, bins, _ = make_scene("dense")
    depth_rank = torch.empty_like(bins.order)
    depth_rank[bins.order] = torch.arange(len(bins.order))
    for tile, (s, e) in enumerate(bins.ranges.tolist()):
        lst = bins.point_list[s:e].long()
        assert torch.all(depth_rank[lst][1:] > depth_rank[lst][:-1]), "tile list not front-to-back"
        tx, ty = tile % bins.tiles_x, tile // bins.tiles_x
        r = bins.rect.long()
        expect = ((r[:, 0] <= tx) & (tx < r[:, 2]) & (r[:, 1] <= ty) & (ty < r[:, 3])).nonzero().flatten()
        assert set(lst.tolist()) == set(expect.tolist())
    assert bins.num_buckets == int(((bins.ranges[:, 1] - bins.ranges[:, 0] + 31) // 32).sum())


def test_dense_scene_exercises_multiple_rounds():
    _, bins, _ = make_scene("dense")
    per_tile = bins.ranges[:, 1] - bins.ranges[:, 0]
    assert per_tile.max() > TILE * TILE


@pytest.mark.parametrize("scene", ["dense", "opaque"])
def test_scene_hits_early_termination(scene):
    leaves, bins, bg = make_scene(scene)
    _, final_T, n_contrib, _ = emu.forward(*[t.detach() for t in leaves], bg, bins)
    assert min(min(row) for row in final_T) < 0.01, "no pixel got close to opaque"


def test_reference_gradcheck_fp64():
    leaves, bins, bg = make_scene("tiny")
    f = lambda *x: render_reference(*x, bins, bg)
    assert torch.autograd.gradcheck(f, tuple(leaves), eps=1e-6, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("scene", list(SCENES))
def test_emulated_forward_matches_reference(scene):
    leaves, bins, bg = make_scene(scene)
    ref = render_reference(*leaves, bins, bg)
    out, *_ = emu.forward(*[t.detach() for t in leaves], bg, bins)
    assert rel_err(torch.tensor(out, dtype=torch.float64), ref.detach()) < 1e-12


@pytest.mark.parametrize("scene", list(SCENES))
@pytest.mark.parametrize("variant", ["pixel", "splat"])
def test_emulated_backward_matches_autograd(scene, variant):
    leaves, bins, bg = make_scene(scene)
    dout = upstream_grad(bins.H, bins.W)
    ref = render_reference(*leaves, bins, bg)
    ref_grads = torch.autograd.grad(ref, leaves, dout)

    x = [t.detach() for t in leaves]
    out, final_T, n_contrib, bstate = emu.forward(*x, bg, bins)
    if variant == "pixel":
        g = emu.backward_pixel(*x, bg, bins, final_T, n_contrib, dout)
    else:
        g = emu.backward_splat(*x, bins, out, n_contrib, bstate, dout)
    for name, a, b in zip(PARAM_NAMES, emu.grads_to_tensors(g), ref_grads):
        assert rel_err(a, b) < 1e-9, f"{variant}/{scene}/{name}: {rel_err(a, b):.2e}"
