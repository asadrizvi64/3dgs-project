"""Autograd wrapper around the splat2d extension, plus a pure-PyTorch reference.

``render(..., mode="pixel" | "splat")`` runs the extension forward and the chosen custom
backward. ``ref_render`` is the same MODEL written with plain differentiable torch ops
(no tiles, no checkpoints) and is the ground truth for the tests.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import torch
from torch import Tensor

Mode = Literal["pixel", "splat"]
MODES: tuple[Mode, ...] = ("pixel", "splat")

TILE = 16
TILE_PIXELS = TILE * TILE
BUCKET = 32
NUM_PARAMS = 9
CKPT_STRIDE = 4
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
T_STOP = 1e-4

CSRC = Path(__file__).resolve().parent.parent / "csrc" / "macrotile"
CPU_SOURCES = (CSRC / "splat2d_cpu.cpp", CSRC / "bindings.cpp")
CUDA_SOURCES = (
    CSRC / "splat2d_fwd.cu",
    CSRC / "splat2d_bwd_pixel.cu",
    CSRC / "splat2d_bwd_splat.cu",
)
# Host flags from the spec (g++/clang). MSVC spells them differently: its /Wall is not
# gcc's -Wall (it enables every off-by-default diagnostic) and it has no -Werror, so on
# MSVC the same intent is /W4 /WX with warnings from <system> headers (torch) silenced.
HOST_CFLAGS = ["-O2", "-Wall", "-Werror"]
MSVC_CFLAGS = ["/O2", "/W4", "/WX", "/external:anglebrackets", "/external:W0"]
CUDA_CFLAGS = ["-O2", "-std=c++17", "-DWITH_CUDA", "-gencode", "arch=compute_80,code=sm_80"]


class Extension(Protocol):
    """Typed view of the compiled extension module."""

    def forward(
        self,
        means: Tensor,
        chol: Tensor,
        opacity_logit: Tensor,
        colors: Tensor,
        tile_offsets: Tensor,
        tile_ids: Tensor,
        bucket_offsets: Tensor,
        bucket_tile: Tensor,
        height: int,
        width: int,
    ) -> list[Tensor]: ...

    def backward_pixel(
        self,
        grad_image: Tensor,
        means: Tensor,
        chol: Tensor,
        opacity_logit: Tensor,
        colors: Tensor,
        tile_offsets: Tensor,
        tile_ids: Tensor,
        bucket_offsets: Tensor,
        bucket_tile: Tensor,
        t_final: Tensor,
        last_contrib: Tensor,
        height: int,
        width: int,
    ) -> list[Tensor]: ...

    def backward_splat(
        self,
        grad_image: Tensor,
        means: Tensor,
        chol: Tensor,
        opacity_logit: Tensor,
        colors: Tensor,
        tile_offsets: Tensor,
        tile_ids: Tensor,
        bucket_offsets: Tensor,
        bucket_tile: Tensor,
        image: Tensor,
        ckpt: Tensor,
        height: int,
        width: int,
    ) -> list[Tensor]: ...

    def get_counters(self) -> dict[str, int]: ...

    def reset_counters(self) -> None: ...


_EXTENSIONS: dict[bool, Extension] = {}


def _use_msvc() -> bool:
    return os.name == "nt"


def _ensure_msvc_env() -> None:
    """On Windows, import the MSVC build environment if ``cl`` is not on PATH yet."""
    if shutil.which("cl") is not None:
        return
    program_files = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        raise RuntimeError("MSVC not found: install Visual Studio Build Tools (C++ workload)")
    install = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    vcvars = Path(install) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not install or not vcvars.exists():
        raise RuntimeError("MSVC C++ tools (vcvars64.bat) not found")
    dump = subprocess.run(
        f'cmd /s /c ""{vcvars}" >nul && set"',
        capture_output=True,
        text=True,
        check=True,
        shell=True,
    ).stdout
    for line in dump.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            os.environ[key] = value


def host_cflags() -> list[str]:
    """Host compiler flags for the extension on this platform."""
    return list(MSVC_CFLAGS if _use_msvc() else HOST_CFLAGS)


def load_extension(with_cuda: bool | None = None, verbose: bool = False) -> Extension:
    """JIT-build (or load the cached) extension. CUDA kernels only with ``with_cuda``."""
    if with_cuda is None:
        with_cuda = os.environ.get("SPLAT2D_WITH_CUDA", "0") == "1"
    cached = _EXTENSIONS.get(with_cuda)
    if cached is not None:
        return cached
    from torch.utils.cpp_extension import load

    if _use_msvc():
        _ensure_msvc_env()
    sources = [str(p) for p in CPU_SOURCES]
    cflags = host_cflags()
    if with_cuda:
        sources += [str(p) for p in CUDA_SOURCES]
        cflags.append("/DWITH_CUDA" if _use_msvc() else "-DWITH_CUDA")
    module = load(
        name="splat2d_cuda" if with_cuda else "splat2d_cpu",
        sources=sources,
        extra_cflags=cflags,
        extra_cuda_cflags=list(CUDA_CFLAGS) if with_cuda else None,
        extra_include_paths=[str(CSRC)],
        with_cuda=with_cuda,
        verbose=verbose,
    )
    ext = cast(Extension, module)
    _EXTENSIONS[with_cuda] = ext
    return ext


def _ext_for(t: Tensor) -> Extension:
    # CPU tensors always use the CPU-only build, so its op counters are the ones that move.
    return load_extension(with_cuda=t.is_cuda)


@dataclass(frozen=True)
class TileBins:
    """CSR tile lists (splat ids in depth order) and the 32-splat bucket tables."""

    tile_offsets: Tensor  # int32 [num_tiles + 1]
    tile_ids: Tensor  # int32 [tile_offsets[-1]]
    bucket_offsets: Tensor  # int32 [num_tiles + 1]
    bucket_tile: Tensor  # int32 [num_buckets]
    tiles_x: int
    tiles_y: int

    @property
    def num_buckets(self) -> int:
        return int(self.bucket_tile.numel())


def bin_tiles(
    means: Tensor, chol: Tensor, opacity_logit: Tensor, height: int, width: int
) -> TileBins:
    """Assign splats to 16x16 tiles with the exact bounding box of their alpha >= 1/255 ellipse.

    alpha >= 1/255 needs q <= r^2 = 2 ln(255 o); the ellipse |L d| <= r has half-extents
    r * ||row of L^-1||. One pixel of slack on each side keeps the test conservative, so a
    splat left out of a tile can never reach alpha >= 1/255 there.
    """
    tiles_x = (width + TILE - 1) // TILE
    tiles_y = (height + TILE - 1) // TILE
    device = means.device
    with torch.no_grad():
        m = means.detach().double().cpu()
        lc = chol.detach().double().cpu()
        o = torch.sigmoid(opacity_logit.detach().double().cpu())
        r2 = 2.0 * torch.log(255.0 * o)
        r = torch.sqrt(torch.clamp(r2, min=0.0))
        l0, l1, l2 = lc[:, 0], lc[:, 1], lc[:, 2]
        ex = r * torch.sqrt(1.0 / l0**2 + (l1 / (l0 * l2)) ** 2)
        ey = r / l2.abs()
        x0 = torch.floor(m[:, 0] - ex - 0.5) - 1
        x1 = torch.ceil(m[:, 0] + ex - 0.5) + 1
        y0 = torch.floor(m[:, 1] - ey - 0.5) - 1
        y1 = torch.ceil(m[:, 1] + ey - 0.5) + 1
        valid = (r2 > 0) & (x1 >= 0) & (x0 <= width - 1) & (y1 >= 0) & (y0 <= height - 1)
        tx0 = torch.div(torch.clamp(x0, 0, width - 1), TILE, rounding_mode="floor")
        tx1 = torch.div(torch.clamp(x1, 0, width - 1), TILE, rounding_mode="floor")
        ty0 = torch.div(torch.clamp(y0, 0, height - 1), TILE, rounding_mode="floor")
        ty1 = torch.div(torch.clamp(y1, 0, height - 1), TILE, rounding_mode="floor")
        lists: list[Tensor] = []
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                hit = valid & (tx0 <= tx) & (tx1 >= tx) & (ty0 <= ty) & (ty1 >= ty)
                lists.append(torch.nonzero(hit).flatten())
        counts = torch.tensor([len(ids) for ids in lists], dtype=torch.int64)
        zero = torch.zeros(1, dtype=torch.int64)
        tile_offsets = torch.cat([zero, torch.cumsum(counts, 0)])
        tile_ids = torch.cat(lists) if lists else torch.zeros(0, dtype=torch.int64)
        n_buckets = torch.div(counts + BUCKET - 1, BUCKET, rounding_mode="floor")
        bucket_offsets = torch.cat([zero, torch.cumsum(n_buckets, 0)])
        bucket_tile = torch.repeat_interleave(torch.arange(len(lists)), n_buckets)
    return TileBins(
        tile_offsets=tile_offsets.to(device=device, dtype=torch.int32),
        tile_ids=tile_ids.to(device=device, dtype=torch.int32),
        bucket_offsets=bucket_offsets.to(device=device, dtype=torch.int32),
        bucket_tile=bucket_tile.to(device=device, dtype=torch.int32),
        tiles_x=tiles_x,
        tiles_y=tiles_y,
    )


class _Splat2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means: Tensor,
        chol: Tensor,
        opacity_logit: Tensor,
        colors: Tensor,
        bins: TileBins,
        height: int,
        width: int,
        mode: Mode,
    ) -> Tensor:
        ext = _ext_for(means)
        image, t_final, _n_contrib, last_contrib, ckpt = ext.forward(
            means,
            chol,
            opacity_logit,
            colors,
            bins.tile_offsets,
            bins.tile_ids,
            bins.bucket_offsets,
            bins.bucket_tile,
            height,
            width,
        )
        ctx.save_for_backward(
            means, chol, opacity_logit, colors, image, t_final, last_contrib, ckpt
        )
        ctx.bins = bins
        ctx.height = height
        ctx.width = width
        ctx.mode = mode
        return image

    @staticmethod
    def backward(ctx: Any, grad_image: Tensor) -> tuple[Tensor | None, ...]:
        means, chol, opacity_logit, colors, image, t_final, last_contrib, ckpt = ctx.saved_tensors
        bins: TileBins = ctx.bins
        ext = _ext_for(means)
        tables = (bins.tile_offsets, bins.tile_ids, bins.bucket_offsets, bins.bucket_tile)
        if ctx.mode == "pixel":
            grads = ext.backward_pixel(
                grad_image,
                means,
                chol,
                opacity_logit,
                colors,
                *tables,
                t_final,
                last_contrib,
                ctx.height,
                ctx.width,
            )
        else:
            grads = ext.backward_splat(
                grad_image,
                means,
                chol,
                opacity_logit,
                colors,
                *tables,
                image,
                ckpt,
                ctx.height,
                ctx.width,
            )
        return grads[0], grads[1], grads[2], grads[3], None, None, None, None


def render(
    means: Tensor,
    chol: Tensor,
    opacity_logit: Tensor,
    colors: Tensor,
    height: int,
    width: int,
    mode: Mode = "splat",
    depths: Tensor | None = None,
) -> Tensor:
    """Render an [H, W, 3] image. Splats are taken in index order unless ``depths`` is given."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if depths is not None:
        order = torch.argsort(depths, stable=True)
        means, chol = means[order], chol[order]
        opacity_logit, colors = opacity_logit[order], colors[order]
    bins = bin_tiles(means, chol, opacity_logit, height, width)
    apply = cast(Callable[..., Tensor], _Splat2dFunction.apply)
    return apply(means, chol, opacity_logit, colors, bins, height, width, mode)


@dataclass(frozen=True)
class RefState:
    image: Tensor  # [H, W, 3]
    t_final: Tensor  # [H, W]
    raw_alpha: Tensor  # [H, W, N] o * exp(-q/2) before the 0.99 clamp
    contributes: Tensor  # [H, W, N] bool: splat composited into the pixel


def ref_state(
    means: Tensor, chol: Tensor, opacity_logit: Tensor, colors: Tensor, height: int, width: int
) -> RefState:
    """Pure-PyTorch front-to-back compositing of the MODEL over the whole image."""
    dtype, device = means.dtype, means.device
    ys = torch.arange(height, dtype=dtype, device=device) + 0.5
    xs = torch.arange(width, dtype=dtype, device=device) + 0.5
    py, px = torch.meshgrid(ys, xs, indexing="ij")
    o = torch.sigmoid(opacity_logit)
    image = torch.zeros(height, width, 3, dtype=dtype, device=device)
    t = torch.ones(height, width, dtype=dtype, device=device)
    raw: list[Tensor] = []
    used: list[Tensor] = []
    for i in range(means.shape[0]):
        dx = px - means[i, 0]
        dy = py - means[i, 1]
        a = chol[i, 0] * dx + chol[i, 1] * dy
        b = chol[i, 2] * dy
        raw_i = o[i] * torch.exp(-0.5 * (a * a + b * b))
        alpha = torch.clamp(raw_i, max=ALPHA_MAX)
        active = (alpha >= ALPHA_MIN) & (t >= T_STOP)
        weight = torch.where(active, alpha * t, torch.zeros_like(t))
        image = image + weight[..., None] * colors[i]
        t = torch.where(active, t * (1.0 - alpha), t)
        raw.append(raw_i.detach())
        used.append(active)
    n = means.shape[0]
    empty = torch.zeros(height, width, 0, dtype=dtype, device=device)
    return RefState(
        image=image,
        t_final=t,
        raw_alpha=torch.stack(raw, dim=-1) if n else empty,
        contributes=torch.stack(used, dim=-1) if n else empty.bool(),
    )


def ref_render(
    means: Tensor, chol: Tensor, opacity_logit: Tensor, colors: Tensor, height: int, width: int
) -> Tensor:
    """Differentiable pure-PyTorch reference image [H, W, 3]."""
    return ref_state(means, chol, opacity_logit, colors, height, width).image
