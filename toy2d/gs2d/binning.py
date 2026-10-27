"""Tile binning in PyTorch: which Gaussians touch which 16x16 tile, in depth order.

Done in PyTorch on purpose: the toy is about the rasterizer backward, and sharing
one binning between the reference and the CUDA kernels means any mismatch in a
test comes from the rasterizer, not from culling.
"""
from dataclasses import dataclass

import torch

TILE = 16     # tile side in pixels; one thread block per tile
BUCKET = 32   # splats per bucket in the per-splat backward (one warp each)


@dataclass
class Bins:
    H: int
    W: int
    tiles_x: int
    tiles_y: int
    rect: torch.Tensor            # (N, 4) int32 xmin, ymin, xmax, ymax in tiles (max exclusive)
    order: torch.Tensor           # (N,) int64 Gaussian ids, front to back
    point_list: torch.Tensor      # (M,) int32 Gaussian id per (tile, gaussian) pair, tile-major, depth-sorted
    ranges: torch.Tensor          # (T, 2) int32 [start, end) into point_list per tile
    bucket_offsets: torch.Tensor  # (T,) int32 first bucket of each tile
    bucket_to_tile: torch.Tensor  # (B,) int32 tile of each bucket
    num_buckets: int

    def to(self, device):
        kw = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in self.__dict__.items()}
        return Bins(**kw)


@torch.no_grad()
def bin_gaussians(means: torch.Tensor, radii: torch.Tensor, depth: torch.Tensor, H: int, W: int) -> Bins:
    device = means.device
    n = means.shape[0]
    tiles_x = (W + TILE - 1) // TILE
    tiles_y = (H + TILE - 1) // TILE
    num_tiles = tiles_x * tiles_y

    m = means.detach().float()
    r = radii.float()
    # Same as getRect() in 3DGS: truncate toward zero, then clamp to the grid.
    xmin = torch.trunc((m[:, 0] - r) / TILE).clamp(0, tiles_x).long()
    ymin = torch.trunc((m[:, 1] - r) / TILE).clamp(0, tiles_y).long()
    xmax = torch.trunc((m[:, 0] + r + TILE - 1) / TILE).clamp(0, tiles_x).long()
    ymax = torch.trunc((m[:, 1] + r + TILE - 1) / TILE).clamp(0, tiles_y).long()
    valid = radii > 0
    w = torch.where(valid, xmax - xmin, torch.zeros_like(xmax))
    h = torch.where(valid, ymax - ymin, torch.zeros_like(ymax))
    counts = (w * h).clamp(min=0)
    # Gaussians that touch no tile get an empty rect so the reference skips them too.
    empty = counts == 0
    xmax = torch.where(empty, xmin, xmax)
    ymax = torch.where(empty, ymin, ymax)
    rect = torch.stack([xmin, ymin, xmax, ymax], dim=-1).to(torch.int32)

    order = torch.argsort(depth, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(n, device=device)

    M = int(counts.sum())
    if M > 0:
        gid = torch.repeat_interleave(torch.arange(n, device=device), counts)
        first = torch.cumsum(counts, 0) - counts
        k = torch.arange(M, device=device) - torch.repeat_interleave(first, counts)
        wg = w[gid]
        tx = xmin[gid] + k % wg
        ty = ymin[gid] + k // wg
        tile = ty * tiles_x + tx
        key = tile * n + rank[gid]
        perm = torch.argsort(key)
        point_list = gid[perm].to(torch.int32)
        tile_sorted = tile[perm]
        per_tile = torch.bincount(tile_sorted, minlength=num_tiles)
    else:
        point_list = torch.zeros(0, dtype=torch.int32, device=device)
        per_tile = torch.zeros(num_tiles, dtype=torch.long, device=device)

    start = torch.cumsum(per_tile, 0) - per_tile
    ranges = torch.stack([start, start + per_tile], dim=-1).to(torch.int32)
    nb = (per_tile + BUCKET - 1) // BUCKET
    bucket_offsets = (torch.cumsum(nb, 0) - nb).to(torch.int32)
    bucket_to_tile = torch.repeat_interleave(torch.arange(num_tiles, device=device), nb).to(torch.int32)

    return Bins(H, W, tiles_x, tiles_y, rect, order, point_list.contiguous(), ranges.contiguous(),
                bucket_offsets.contiguous(), bucket_to_tile.contiguous(), int(nb.sum()))
