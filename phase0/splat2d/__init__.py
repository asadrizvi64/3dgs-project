"""Phase 0: 2D Gaussian image fitter with two custom backward passes.

Mode "pixel" is the per-pixel atomic baseline (A); mode "splat" is the per-splat,
32-splat warp-bucketed backward (B) that replays from forward checkpoints.
"""

from splat2d.ops import MODES, Mode, TileBins, bin_tiles, load_extension, ref_render, render

__all__ = ["MODES", "Mode", "TileBins", "bin_tiles", "load_extension", "ref_render", "render"]
