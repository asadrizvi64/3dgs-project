"""Autograd wrapper around the CUDA rasterizer (JIT-compiled on first use)."""
import os
from pathlib import Path

import torch

from .binning import Bins
from .reference import render_reference

_CSRC = Path(__file__).resolve().parent.parent / "csrc"
_ext = None


def ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        flags = ["-O3", "-lineinfo"]  # -lineinfo lets Nsight Compute map stalls to source lines
        _ext = load(
            name="gs2d_cuda",
            sources=[str(_CSRC / "rasterize.cu")],
            extra_cuda_cflags=flags,
            verbose=os.environ.get("GS2D_VERBOSE_BUILD") == "1",
        )
    return _ext


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, conic, color, opacity, bg, bins: Bins, mode: str):
        save = mode == "splat"
        out, final_T, n_contrib, bstate = ext().forward(
            means.contiguous(), conic.contiguous(), color.contiguous(), opacity.contiguous(), bg,
            bins.ranges, bins.point_list, bins.bucket_offsets, bins.num_buckets, bins.H, bins.W, save,
        )
        ctx.bins, ctx.mode = bins, mode
        ctx.save_for_backward(means, conic, color, opacity, bg, out, final_T, n_contrib, bstate)
        return out

    @staticmethod
    def backward(ctx, dout):
        means, conic, color, opacity, bg, out, final_T, n_contrib, bstate = ctx.saved_tensors
        b = ctx.bins
        dout = dout.contiguous()
        args = (means.contiguous(), conic.contiguous(), color.contiguous(), opacity.contiguous(), bg)
        if ctx.mode == "pixel":
            g = ext().backward_pixel(*args, b.ranges, b.point_list, final_T, n_contrib, dout)
        else:
            g = ext().backward_splat(*args, b.ranges, b.point_list, b.bucket_offsets, b.bucket_to_tile,
                                     out, final_T, n_contrib, bstate, dout)
        dmeans, dconic, dcolor, dopacity = g
        return dmeans, dconic, dcolor, dopacity, None, None, None


def rasterize(means, conic, color, opacity, bins: Bins, bg=None, backend: str = "splat"):
    """Render a (3, H, W) image. backend: 'ref' (PyTorch), 'pixel' or 'splat' (CUDA backward variant)."""
    if bg is None:
        bg = torch.zeros(3, dtype=means.dtype, device=means.device)
    if backend == "ref":
        return render_reference(means, conic, color, opacity, bins, bg)
    if backend not in ("pixel", "splat"):
        raise ValueError(f"unknown backend {backend!r}")
    return _Rasterize.apply(means, conic, color, opacity, bg, bins, backend)
