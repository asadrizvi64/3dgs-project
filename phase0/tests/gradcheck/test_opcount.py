"""Op-count proxies from the CPU emulation of the two backward thread mappings.

These are proxies only: real instruction counts come from Nsight on the A100 later.
Counters: atomic_adds, checkpoint_bytes, splat_pixel_evals. The extension always runs on
the CPU here (the counters are incremented by the CPU emulation), in float64 so the
Python-side contribution masks match the kernel's decisions exactly.
"""

from __future__ import annotations

import pytest
import torch

from conftest import Scene
from splat2d.ops import (
    BUCKET,
    CKPT_STRIDE,
    MODES,
    NUM_PARAMS,
    TILE,
    TILE_PIXELS,
    Mode,
    TileBins,
    bin_tiles,
    load_extension,
    ref_render,
    ref_state,
    render,
)


def run_counted(scene: Scene, mode: Mode) -> dict[str, int]:
    ext = load_extension(with_cuda=False)
    params = scene.params(torch.float64)
    ext.reset_counters()
    image = render(*params, scene.height, scene.width, mode=mode)
    torch.autograd.grad(image, params, grad_outputs=scene.grad_image(torch.float64))
    return dict(ext.get_counters())


def scene_bins(scene: Scene) -> TileBins:
    return bin_tiles(scene.means, scene.chol, scene.opacity_logit, scene.height, scene.width)


def n_contrib_of(scene: Scene) -> torch.Tensor:
    bins = scene_bins(scene)
    ext = load_extension(with_cuda=False)
    out = ext.forward(
        scene.means,
        scene.chol,
        scene.opacity_logit,
        scene.colors,
        bins.tile_offsets,
        bins.tile_ids,
        bins.bucket_offsets,
        bins.bucket_tile,
        scene.height,
        scene.width,
    )
    return out[2]


def expected_pixel_atomics(scene: Scene) -> int:
    """sum_pixels n_contrib(p) * P."""
    return int(n_contrib_of(scene).sum()) * NUM_PARAMS


def expected_splat_atomics(scene: Scene) -> int:
    """sum_tiles sum_buckets |splats in bucket that touch the tile| * P.

    A splat touches a tile if it is composited into at least one pixel of that tile.
    """
    st = ref_state(
        scene.means, scene.chol, scene.opacity_logit, scene.colors, scene.height, scene.width
    )
    bins = scene_bins(scene)
    offs = bins.tile_offsets.tolist()
    ids = bins.tile_ids.long()
    total = 0
    for t in range(bins.tiles_x * bins.tiles_y):
        x0, y0 = (t % bins.tiles_x) * TILE, (t // bins.tiles_x) * TILE
        region = st.contributes[y0 : y0 + TILE, x0 : x0 + TILE]
        for start in range(offs[t], offs[t + 1], BUCKET):
            bucket = ids[start : min(start + BUCKET, offs[t + 1])]
            total += int(region[:, :, bucket].flatten(0, 1).any(dim=0).sum())
    return total * NUM_PARAMS


def test_n_contrib_matches_reference(opcount_scene: Scene) -> None:
    s = opcount_scene
    st = ref_state(s.means, s.chol, s.opacity_logit, s.colors, s.height, s.width)
    assert torch.equal(n_contrib_of(s).long(), st.contributes.sum(dim=-1))


def test_pixel_atomics_formula(opcount_scene: Scene) -> None:
    counted = run_counted(opcount_scene, "pixel")
    assert counted["atomic_adds"] == expected_pixel_atomics(opcount_scene)


def test_splat_atomics_formula(opcount_scene: Scene) -> None:
    counted = run_counted(opcount_scene, "splat")
    assert counted["atomic_adds"] == expected_splat_atomics(opcount_scene)


def test_dense_splat_atomics_at_most_one_eighth(dense_scene: Scene) -> None:
    pixel = run_counted(dense_scene, "pixel")["atomic_adds"]
    splat = run_counted(dense_scene, "splat")["atomic_adds"]
    print(f"\n[dense] pixel atomics={pixel} splat atomics={splat} ratio={pixel / splat:.1f}x")
    assert splat <= pixel / 8


@pytest.mark.parametrize("mode", MODES)
def test_checkpoint_bytes(dense_scene: Scene, mode: Mode) -> None:
    counted = run_counted(dense_scene, mode)
    bins = scene_bins(dense_scene)
    expected = bins.num_buckets * TILE_PIXELS * CKPT_STRIDE * 8  # float64
    print(
        f"\n[dense/{mode}] checkpoint_bytes={counted['checkpoint_bytes']} "
        f"buckets={bins.num_buckets} splat_pixel_evals={counted['splat_pixel_evals']}"
    )
    assert counted["checkpoint_bytes"] > 0
    assert counted["checkpoint_bytes"] == expected
    assert counted["splat_pixel_evals"] > 0


def test_dense_crosses_two_checkpoints(dense_scene: Scene) -> None:
    bins = scene_bins(dense_scene)
    per_tile = (bins.tile_offsets[1:] - bins.tile_offsets[:-1]).max()
    assert int(per_tile) > 2 * BUCKET  # bucket 2 starts from a replayed checkpoint


@pytest.mark.parametrize("mode", MODES)
def test_sparse_scene_matches_reference_fp64(sparse_scene: Scene, mode: Mode) -> None:
    """Culling, partial tiles, skipped pairs and early termination keep exact gradients."""
    s = sparse_scene
    st = ref_state(s.means, s.chol, s.opacity_logit, s.colors, s.height, s.width)
    assert float(st.t_final.min()) < 1e-4, "sparse scene should terminate somewhere"
    params = s.params(torch.float64)
    image = render(*params, s.height, s.width, mode=mode)
    g_img = s.grad_image(torch.float64)
    got = torch.autograd.grad(image, params, grad_outputs=g_img)
    ref_params = s.params(torch.float64)
    want = torch.autograd.grad(ref_render(*ref_params, s.height, s.width), ref_params, g_img)
    assert torch.allclose(image.detach(), st.image.detach(), rtol=0, atol=1e-12)
    for g, w in zip(got, want, strict=True):
        rel = float(torch.linalg.vector_norm(g - w)) / max(
            float(torch.linalg.vector_norm(w)), 1e-12
        )
        assert rel < 1e-10
