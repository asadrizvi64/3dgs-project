"""Line-by-line Python mirror of csrc/rasterize.cu, for checking kernel logic without a GPU.

Same tile lists, same bucket state, same warp hand-off schedule, same gradient
formulas, just scalar Python in float64. If this matches autograd on the reference
but the CUDA build does not, the bug is in CUDA mechanics (indexing, sync, races),
not in the math.
"""
import math

import torch

from gs2d.binning import BUCKET, TILE, Bins

BLOCK = TILE * TILE
ALPHA_MIN, ALPHA_MAX, T_MIN = 1.0 / 255.0, 0.99, 1e-4


def _eval(m, con, o, px, py):
    dx, dy = m[0] - px, m[1] - py
    power = -0.5 * (con[0] * dx * dx + con[2] * dy * dy) - con[1] * dx * dy
    if power > 0:
        return None
    G = math.exp(power)
    a = o * G
    return dx, dy, G, min(ALPHA_MAX, a), a > ALPHA_MAX


def _chain(con, o, dx, dy, G, clamped, dL_dalpha, g):
    if clamped:
        return
    g["dop"] += G * dL_dalpha
    dpow = o * G * dL_dalpha
    g["dm"][0] += (-con[0] * dx - con[1] * dy) * dpow
    g["dm"][1] += (-con[2] * dy - con[1] * dx) * dpow
    g["dcon"][0] += -0.5 * dx * dx * dpow
    g["dcon"][1] += -dx * dy * dpow
    g["dcon"][2] += -0.5 * dy * dy * dpow


def _new_grads(n):
    return [{"dm": [0.0, 0.0], "dcon": [0.0, 0.0, 0.0], "dcol": [0.0, 0.0, 0.0], "dop": 0.0} for _ in range(n)]


def _tile_pixels(b: Bins, tile):
    tx, ty = tile % b.tiles_x, tile // b.tiles_x
    for tid in range(BLOCK):
        yield tid, tx * TILE + tid % TILE, ty * TILE + tid // TILE


def _lists(b: Bins):
    ranges = b.ranges.tolist()
    pl = b.point_list.tolist()
    return [pl[s:e] for s, e in ranges]


def forward(means, conic, color, opacity, bg, b: Bins):
    H, W = b.H, b.W
    m, con, col, op, bg = means.tolist(), conic.tolist(), color.tolist(), opacity.tolist(), bg.tolist()
    lists = _lists(b)
    out = [[[0.0] * W for _ in range(H)] for _ in range(3)]
    final_T = [[0.0] * W for _ in range(H)]
    n_contrib = [[0] * W for _ in range(H)]
    bstate = [[(0.0, 0.0, 0.0, 0.0)] * BLOCK for _ in range(b.num_buckets)]
    boff = b.bucket_offsets.tolist()
    for tile, lst in enumerate(lists):
        for tid, px, py in _tile_pixels(b, tile):
            T, C, last, done = 1.0, [0.0, 0.0, 0.0], 0, not (px < W and py < H)
            for idx, gid in enumerate(lst):
                if idx % BUCKET == 0:
                    bstate[boff[tile] + idx // BUCKET][tid] = (T, *C)
                if done:
                    continue
                e = _eval(m[gid], con[gid], op[gid], px, py)
                if e is None or e[3] < ALPHA_MIN:
                    continue
                alpha = e[3]
                test_T = T * (1 - alpha)
                if test_T < T_MIN:
                    done = True
                    continue
                for ch in range(3):
                    C[ch] += col[gid][ch] * alpha * T
                T, last = test_T, idx + 1
            if px < W and py < H:
                for ch in range(3):
                    out[ch][py][px] = C[ch] + T * bg[ch]
                final_T[py][px] = T
                n_contrib[py][px] = last
    return out, final_T, n_contrib, bstate


def backward_pixel(means, conic, color, opacity, bg, b: Bins, final_T, n_contrib, dout):
    H, W = b.H, b.W
    m, con, col, op, bg = means.tolist(), conic.tolist(), color.tolist(), opacity.tolist(), bg.tolist()
    d = dout.tolist()
    grads = _new_grads(len(m))
    for tile, lst in enumerate(_lists(b)):
        for _, px, py in _tile_pixels(b, tile):
            if not (px < W and py < H):
                continue
            Tf, last = final_T[py][px], n_contrib[py][px]
            dpix = [d[ch][py][px] for ch in range(3)]
            bg_dot = sum(bg[ch] * dpix[ch] for ch in range(3))
            T, acc, la, lc = Tf, [0.0] * 3, 0.0, [0.0] * 3
            for idx in range(len(lst) - 1, -1, -1):
                if idx >= last:
                    continue
                gid = lst[idx]
                e = _eval(m[gid], con[gid], op[gid], px, py)
                if e is None or e[3] < ALPHA_MIN:
                    continue
                dx, dy, G, alpha, clamped = e
                T = T / (1 - alpha)
                acc = [la * lc[ch] + (1 - la) * acc[ch] for ch in range(3)]
                c = col[gid]
                la, lc = alpha, c
                dL_dalpha = sum((c[ch] - acc[ch]) * dpix[ch] for ch in range(3)) * T
                dL_dalpha += (-Tf / (1 - alpha)) * bg_dot
                g = grads[gid]
                for ch in range(3):
                    g["dcol"][ch] += alpha * T * dpix[ch]
                _chain(con[gid], op[gid], dx, dy, G, clamped, dL_dalpha, g)
    return grads


def backward_splat(means, conic, color, opacity, b: Bins, out, n_contrib, bstate, dout):
    H, W = b.H, b.W
    m, con, col, op = means.tolist(), conic.tolist(), color.tolist(), opacity.tolist()
    d = dout.tolist()
    lists = _lists(b)
    boff = b.bucket_offsets.tolist()
    grads = _new_grads(len(m))
    for bk, tile in enumerate(b.bucket_to_tile.tolist()):
        lst = lists[tile]
        tx0, ty0 = (tile % b.tiles_x) * TILE, (tile // b.tiles_x) * TILE
        lanes = []
        for lane in range(32):
            idx = (bk - boff[tile]) * BUCKET + lane
            lanes.append((idx, lst[idx] if idx < len(lst) else None))
        local = _new_grads(32)
        state = [(0.0, [0.0] * 3)] * 32
        for t in range(BLOCK + BUCKET - 1):
            shifted = [None] + state[:-1]  # __shfl_up_sync by 1
            new_state = list(state)
            for lane in range(32):
                p = t - lane
                if lane == 0:
                    if p < BLOCK:
                        s = bstate[bk][p]
                        new_state[0] = (s[0], [s[1], s[2], s[3]])
                else:
                    new_state[lane] = shifted[lane]
                idx, gid = lanes[lane]
                if gid is None or p < 0 or p >= BLOCK:
                    continue
                px, py = tx0 + p % TILE, ty0 + p // TILE
                if px >= W or py >= H or idx >= n_contrib[py][px]:
                    continue
                e = _eval(m[gid], con[gid], op[gid], px, py)
                if e is None or e[3] < ALPHA_MIN:
                    continue
                dx, dy, G, alpha, clamped = e
                T, C = new_state[lane]
                dpix = [d[ch][py][px] for ch in range(3)]
                w = alpha * T
                C_after = [C[ch] + col[gid][ch] * w for ch in range(3)]
                behind = [out[ch][py][px] - C_after[ch] for ch in range(3)]
                inv = 1.0 / (1.0 - alpha)
                dL_dalpha = sum(dpix[ch] * (T * col[gid][ch] - behind[ch] * inv) for ch in range(3))
                g = local[lane]
                for ch in range(3):
                    g["dcol"][ch] += w * dpix[ch]
                _chain(con[gid], op[gid], dx, dy, G, clamped, dL_dalpha, g)
                new_state[lane] = (T * (1 - alpha), C_after)
            state = new_state
        for lane, (_, gid) in enumerate(lanes):
            if gid is None:
                continue
            for k in ("dm", "dcon", "dcol"):
                for i, v in enumerate(local[lane][k]):
                    grads[gid][k][i] += v
            grads[gid]["dop"] += local[lane]["dop"]
    return grads


def grads_to_tensors(grads, dtype=torch.float64):
    dm = torch.tensor([g["dm"] for g in grads], dtype=dtype)
    dcon = torch.tensor([g["dcon"] for g in grads], dtype=dtype)
    dcol = torch.tensor([g["dcol"] for g in grads], dtype=dtype)
    dop = torch.tensor([g["dop"] for g in grads], dtype=dtype)
    return dm, dcon, dcol, dop
