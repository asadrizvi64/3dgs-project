"""Seeds and synthetic scenes for the Phase 0 tests.

Gradient fixtures are guarded: every (pixel, splat) alpha lies strictly inside
(1/255 + 1e-3, 0.99 - 1e-3) and every pixel keeps T > 1e-3, so no pair sits near the
skip threshold, the 0.99 clamp or early termination (non-smooth points break gradcheck).
The "sparse" scene is NOT guarded: it exercises culling, partial tiles, skipped pairs and
early termination for the op-count formulas only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import cast

import pytest
import torch
from torch import Tensor

from splat2d.ops import ALPHA_MAX, ALPHA_MIN, ref_state

SEEDS = (0, 1, 2)
GRADCHECK_N = (8, 32)
DENSE_N = 96  # > 64 splats over every pixel: backward (B) crosses 2 checkpoints
IMAGE_HW = (32, 32)
GUARD_MARGIN = 1e-3
GUARD_MIN_T = 1e-3


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=("cpu", "cuda"),
        help="device for the gradient tests (cuda needs the extension built with -DWITH_CUDA)",
    )


@dataclass(frozen=True)
class Scene:
    name: str
    means: Tensor  # float64 [N, 2]
    chol: Tensor  # float64 [N, 3]
    opacity_logit: Tensor  # float64 [N]
    colors: Tensor  # float64 [N, 3]
    height: int
    width: int

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    def params(
        self, dtype: torch.dtype, device: torch.device | str = "cpu"
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Fresh leaf copies of the parameters (requires_grad) in ``dtype`` on ``device``."""
        out = tuple(
            t.detach().to(dtype=dtype, device=device).clone().requires_grad_(True)
            for t in (self.means, self.chol, self.opacity_logit, self.colors)
        )
        return out[0], out[1], out[2], out[3]

    def grad_image(self, dtype: torch.dtype, device: torch.device | str = "cpu") -> Tensor:
        """Fixed random dL/dimage (seeded by the scene name)."""
        g = torch.Generator().manual_seed(sum(map(ord, self.name)) + 12345)
        w = torch.randn(self.height, self.width, 3, generator=g, dtype=torch.float64)
        return w.to(dtype=dtype, device=device)


def _uniform(g: torch.Generator, lo: float, hi: float, *shape: int) -> Tensor:
    return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=torch.float64)


def make_scene(
    name: str,
    seed: int,
    n: int,
    height: int,
    width: int,
    sigma: tuple[float, float],
    opacity: tuple[float, float],
    mean_box: tuple[float, float] = (0.3, 0.7),
) -> Scene:
    g = torch.Generator().manual_seed(seed)
    means = torch.stack(
        [
            _uniform(g, mean_box[0] * width, mean_box[1] * width, n),
            _uniform(g, mean_box[0] * height, mean_box[1] * height, n),
        ],
        dim=-1,
    )
    l0 = 1.0 / _uniform(g, sigma[0], sigma[1], n)
    l2 = 1.0 / _uniform(g, sigma[0], sigma[1], n)
    l1 = _uniform(g, -0.3, 0.3, n) * torch.minimum(l0, l2)
    o = _uniform(g, opacity[0], opacity[1], n)
    colors = _uniform(g, 0.05, 0.95, n, 3)
    return Scene(
        name=name,
        means=means,
        chol=torch.stack([l0, l1, l2], dim=-1),
        opacity_logit=torch.log(o / (1.0 - o)),
        colors=colors,
        height=height,
        width=width,
    )


def assert_smooth(scene: Scene) -> None:
    """Fixture guard: no (pixel, splat) pair near a non-smooth point of the MODEL."""
    st = ref_state(
        scene.means, scene.chol, scene.opacity_logit, scene.colors, scene.height, scene.width
    )
    lo, hi = ALPHA_MIN + GUARD_MARGIN, ALPHA_MAX - GUARD_MARGIN
    a_min, a_max = float(st.raw_alpha.min()), float(st.raw_alpha.max())
    assert a_min > lo, f"{scene.name}: alpha {a_min:.5f} <= {lo:.5f}"
    assert a_max < hi, f"{scene.name}: alpha {a_max:.5f} >= {hi:.5f}"
    t_min = float(st.t_final.min())
    assert t_min > GUARD_MIN_T, f"{scene.name}: T {t_min:.2e} <= {GUARD_MIN_T}"
    assert bool(st.contributes.all()), f"{scene.name}: a pair was skipped"


@cache
def guarded_scene(seed: int, n: int) -> Scene:
    h, w = IMAGE_HW
    scene = make_scene(f"seed{seed}_n{n}", seed, n, h, w, sigma=(22.0, 40.0), opacity=(0.05, 0.2))
    assert_smooth(scene)
    return scene


@cache
def dense_scene_cached() -> Scene:
    h, w = IMAGE_HW
    scene = make_scene("dense_n96", 7, DENSE_N, h, w, sigma=(30.0, 45.0), opacity=(0.02, 0.06))
    assert_smooth(scene)
    st = ref_state(scene.means, scene.chol, scene.opacity_logit, scene.colors, h, w)
    over_centre = int(st.contributes[h // 2, w // 2].sum())
    assert over_centre > 64, f"dense scene has only {over_centre} splats over the centre pixel"
    return scene


@cache
def sparse_scene_cached() -> Scene:
    """Small splats on a 40x56 image (partial tiles), with an opaque stack that terminates."""
    base = make_scene(
        "sparse_n150", 11, 144, 40, 56, sigma=(1.5, 6.0), opacity=(0.2, 0.9), mean_box=(-0.05, 1.05)
    )
    # Six near-opaque splats on one spot: alpha clamps at 0.99 and T drops below 1e-4.
    k = 6
    stack_means = torch.tensor([[20.3, 17.7]], dtype=torch.float64).repeat(k, 1)
    stack_chol = torch.tensor([[1 / 4.0, 0.01, 1 / 5.0]], dtype=torch.float64).repeat(k, 1)
    stack_logit = torch.full((k,), math.log(0.999 / 0.001), dtype=torch.float64)
    stack_colors = torch.linspace(0.1, 0.9, 3 * k, dtype=torch.float64).reshape(k, 3)
    return Scene(
        name="sparse_n150",
        means=torch.cat([stack_means, base.means]),
        chol=torch.cat([stack_chol, base.chol]),
        opacity_logit=torch.cat([stack_logit, base.opacity_logit]),
        colors=torch.cat([stack_colors, base.colors]),
        height=base.height,
        width=base.width,
    )


GRADCHECK_IDS = [(s, n) for s in SEEDS for n in GRADCHECK_N]


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> torch.device:
    name = cast(str, request.config.getoption("--device"))
    if name == "cuda":
        assert torch.cuda.is_available(), "--device=cuda but torch.cuda.is_available() is False"
    return torch.device(name)


@pytest.fixture(params=GRADCHECK_IDS, ids=[f"seed{s}-n{n}" for s, n in GRADCHECK_IDS])
def gradcheck_scene(request: pytest.FixtureRequest) -> Scene:
    seed, n = cast(tuple[int, int], request.param)
    return guarded_scene(seed, n)


@pytest.fixture(
    params=[*GRADCHECK_IDS, "dense"], ids=[*(f"seed{s}-n{n}" for s, n in GRADCHECK_IDS), "dense"]
)
def accuracy_scene(request: pytest.FixtureRequest) -> Scene:
    if request.param == "dense":
        return dense_scene_cached()
    seed, n = cast(tuple[int, int], request.param)
    return guarded_scene(seed, n)


@pytest.fixture
def dense_scene() -> Scene:
    return dense_scene_cached()


@pytest.fixture
def sparse_scene() -> Scene:
    return sparse_scene_cached()


@pytest.fixture(params=["seed0-n32", "dense", "sparse"])
def opcount_scene(request: pytest.FixtureRequest) -> Scene:
    if request.param == "dense":
        return dense_scene_cached()
    if request.param == "sparse":
        return sparse_scene_cached()
    return guarded_scene(0, 32)
