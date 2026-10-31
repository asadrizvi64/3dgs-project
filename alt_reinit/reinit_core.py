"""alt_reinit/reinit_core.py: geometry and bookkeeping for the depth reinitialization (alternative track R).

Pure torch and device-agnostic: the trainer hook (reinit_trainer.py, copied into the FasterGSReinit method) runs it on
the GPU, alt_reinit/tests/test_core_cpu.py on the CPU.

Conventions follow Faster-GS @c60ea26 (rasterization/include/kernels_inference.cuh):
  camera space = w2c @ world with +z forward; u = fx * x / z + cx, v = fy * y / z + cy;
  pixel (i, j) covers [i, i+1) x [j, j+1), its centre is (i + 0.5, j + 0.5) (the same continuous convention as COLMAP).

Depth comes from one inference render per view with the per-Gaussian colour set to (z, 1, z^2), one SH band and a black
background. The rasterizer then returns (sum w z, sum w, sum w z^2) per pixel, with w = T * alpha the blending weights:
expected depth D = sum w z / sum w, alpha A = sum w, depth variance = sum w z^2 / sum w - D^2.
A large variance flags pixels that mix surfaces at different depths (for example 1 m and 3 m giving D near 2 m, where
there may be no surface at all). A small variance does not show that the depth is correct.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch

SH_C0 = 0.28209479177387814  # 1 / (2 sqrt(pi)); the rasterizer's colour is max(0.5 + SH_C0 * sh0, 0) with one SH band

# per-candidate status (one reference pixel)
KEPT, EMPTY, LOW_ALPHA, MIXTURE, GEOMETRY = 0, 1, 2, 3, 4
STATUS_NAMES = ('kept', 'empty', 'low_alpha', 'mixture', 'geometry')
# per-neighbour outcome of the geometric test
AGREE, OUT_OF_VIEW, TARGET_INVALID, DEPTH_MISMATCH, REPROJ_MISMATCH = 0, 1, 2, 3, 4
NEIGHBOUR_NAMES = ('agree', 'out_of_view', 'target_invalid', 'depth_mismatch', 'reproj_mismatch')
PARAM_GROUPS = ('means', 'sh_coefficients_0', 'sh_coefficients_rest', 'opacities', 'scales', 'rotations')


# ----------------------------------------------------------------------------------------------------------- config
@dataclass
class ReinitConfig:
    """Settings, parsed from the FGS_REINIT environment variable (see parse_env)."""
    mode: str = 'off'          # off | apply | audit | stop (audit: dry run on a copy of the model and optimizer; training
                               # unchanged. stop: no reinit, densification ends at iters=k: the R2 'stop' arm)
    start: int = -1            # first iteration at which the callback runs
    end: int = -1              # last iteration (inclusive)
    every: int = 1             # iteration stride between start and end
    target: str = 'match'      # count after the reinit: 'match' (= count before), '<int>', '+<int>', '-<int>'; audit mode
                               # takes several separated by '/': the first is timed in full, all are checked for feasibility
    stride: int = 2            # candidate pixel grid stride
    alpha_min: float = 0.95    # candidate pixels need accumulated alpha >= this
    rel_std_max: float = 0.05  # and weighted depth std / expected depth <= this (mixture test)
    n_nbr: int = 4             # neighbour views per reference view
    min_agree: int = 2         # neighbours that must agree; 0 = unfiltered (no geometric test)
    depth_tol: float = 0.01    # relative depth tolerance in the neighbour
    reproj_tol: float = 1.0    # round-trip reprojection tolerance in reference pixels
    min_angle: float = 3.0     # neighbours need a median triangulation angle >= this (degrees)
    cover_ring: int = 1        # old rows within this many voxels of a candidate voxel are replaced (0 = same voxel)
    init_opacity: float = 0.1  # stock initialization value
    seed: int = 0              # seeds the reinit's own generator (the training RNG is never used)
    check: bool = True         # verify the replacement on the real optimizer after it ran (untimed)
    max_retries: int = 1       # retries with half the stride when there are too few candidates
    diag_views: int = 16       # audit: views kept for depth agreement, coverage and SfM checks
    mask_views: int = 4        # audit: views whose status maps are saved as images
    zncc_samples: int = 20000  # audit: candidates per class for the photo-consistency check
    nbr_samples: int = 1024    # points per view used to choose neighbours
    end_densify: bool = False  # set DENSIFICATION_END_ITERATION = start (R2 arms; implied by mode=stop). In Faster-GS this
                               # also ends opacity resets and densification-info accumulation there

    def describe(self) -> str:
        return ','.join(f'{k}={v}' for k, v in asdict(self).items())


_INT = ('start', 'end', 'every', 'stride', 'n_nbr', 'min_agree', 'cover_ring', 'seed', 'max_retries', 'diag_views', 'mask_views',
        'zncc_samples', 'nbr_samples')
_FLOAT = ('alpha_min', 'rel_std_max', 'depth_tol', 'reproj_tol', 'min_angle', 'init_opacity')


def parse_env(spec: str) -> ReinitConfig:
    """'mode=audit,iters=2000:5000,target=match,...'. iters is 'k', 'a:b' (stride b - a) or 'a:b:s'."""
    cfg = ReinitConfig()
    if not spec.strip():
        return cfg
    for item in spec.split(','):
        if not item.strip():
            continue
        key, _, val = item.partition('=')
        key, val = key.strip(), val.strip()
        if key == 'iters':
            parts = [int(p) for p in val.split(':')]
            if len(parts) == 1:
                cfg.start = cfg.end = parts[0]
                cfg.every = 1
            elif len(parts) == 2:
                cfg.start, cfg.end = parts
                cfg.every = max(parts[1] - parts[0], 1)
            elif len(parts) == 3:
                cfg.start, cfg.end, cfg.every = parts
            else:
                raise ValueError(f'FGS_REINIT: bad iters {val!r}')
        elif key in _INT:
            setattr(cfg, key, int(val))
        elif key in _FLOAT:
            setattr(cfg, key, float(val))
        elif key in ('check', 'end_densify'):
            setattr(cfg, key, val not in ('0', 'false', 'False', 'no'))
        elif key in ('mode', 'target'):
            setattr(cfg, key, val)
        else:
            raise ValueError(f'FGS_REINIT: unknown key {key!r}')
    if cfg.mode not in ('off', 'apply', 'audit', 'stop'):
        raise ValueError(f'FGS_REINIT: mode must be off, apply, audit or stop, got {cfg.mode!r}')
    if cfg.mode == 'stop':
        cfg.end_densify = True
    if cfg.mode != 'off':
        if cfg.start < 1 or cfg.end < cfg.start or cfg.every < 1:
            raise ValueError(f'FGS_REINIT: bad iteration range start={cfg.start} end={cfg.end} every={cfg.every}')
        if cfg.mode in ('apply', 'stop') and cfg.end != cfg.start:
            raise ValueError(f'FGS_REINIT: {cfg.mode} mode runs once (iters=k)')
    if cfg.end_densify and cfg.mode not in ('apply', 'stop'):
        raise ValueError('FGS_REINIT: end_densify needs mode apply or stop')
    if cfg.stride < 1 or cfg.n_nbr < 1 or cfg.min_agree < 0 or cfg.min_agree > cfg.n_nbr:
        raise ValueError('FGS_REINIT: need stride >= 1, n_nbr >= 1, 0 <= min_agree <= n_nbr')
    for t in cfg.target.split('/'):
        resolve_target(t, 1)  # validates the syntax
    if cfg.mode == 'apply' and '/' in cfg.target:
        raise ValueError('FGS_REINIT: apply mode takes one target')
    return cfg


def resolve_target(spec: str, n_before: int) -> int:
    if spec == 'match':
        return n_before
    if spec[:1] in '+-':
        return n_before + int(spec)
    return int(spec)


# ----------------------------------------------------------------------------------------------------------- cameras
@dataclass
class Camera:
    w2c: torch.Tensor  # (4, 4) world -> camera
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    near: float = 0.0
    c2w: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.c2w = torch.linalg.inv(self.w2c.double()).to(self.w2c.dtype)

    @property
    def center(self) -> torch.Tensor:
        return self.c2w[:3, 3]


def project(cam: Camera, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World points (M, 3) -> continuous pixel coordinates u, v and camera depth z."""
    pc = X @ cam.w2c[:3, :3].T + cam.w2c[:3, 3]
    z = pc[:, 2]
    return cam.fx * pc[:, 0] / z + cam.cx, cam.fy * pc[:, 1] / z + cam.cy, z


def backproject(cam: Camera, u: torch.Tensor, v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Continuous pixel coordinates and camera depth -> world points (M, 3)."""
    pc = torch.stack([(u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z], dim=-1)
    return pc @ cam.c2w[:3, :3].T + cam.c2w[:3, 3]


def pixel_index(cam: Camera, u: torch.Tensor, v: torch.Tensor, z: torch.Tensor | None = None):
    """Pixel containing each continuous coordinate; inside = in the image (and in front of the near plane)."""
    ok = torch.isfinite(u) & torch.isfinite(v)
    if z is not None:
        ok &= z > cam.near
    ix = torch.floor(torch.where(ok, u, torch.full_like(u, -1.0))).long()
    iy = torch.floor(torch.where(ok, v, torch.full_like(v, -1.0))).long()
    inside = ok & (ix >= 0) & (ix < cam.width) & (iy >= 0) & (iy < cam.height)
    return ix.clamp(0, cam.width - 1), iy.clamp(0, cam.height - 1), inside


def sample_bilinear(img: torch.Tensor, valid: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
    """Bilinear sample of (H, W) `img` at continuous coordinates (pixel centres at i + 0.5); ok only when all 4 taps
    are inside the image and valid."""
    H, W = img.shape
    finite = torch.isfinite(u) & torch.isfinite(v)
    x = torch.where(finite, u, torch.zeros_like(u)) - 0.5
    y = torch.where(finite, v, torch.zeros_like(v)) - 0.5
    x0, y0 = torch.floor(x), torch.floor(y)
    wx, wy = x - x0, y - y0
    x0, y0 = x0.long(), y0.long()
    ok = finite & (x0 >= 0) & (y0 >= 0) & (x0 + 1 < W) & (y0 + 1 < H)
    xa, ya = x0.clamp(0, W - 2), y0.clamp(0, H - 2)
    out = torch.zeros_like(u)
    for dy, dx, w in ((0, 0, (1 - wx) * (1 - wy)), (0, 1, wx * (1 - wy)), (1, 0, (1 - wx) * wy), (1, 1, wx * wy)):
        out = out + w * img[ya + dy, xa + dx]
        ok &= valid[ya + dy, xa + dx]
    return out, ok


# ----------------------------------------------------------------------------------------------------------- depth
def depth_colors(means: torch.Tensor, w2c: torch.Tensor) -> torch.Tensor:
    z = means @ w2c[2, :3] + w2c[2, 3]
    return torch.stack([z, torch.ones_like(z), z * z], dim=-1)


def sh0_from_colors(colors: torch.Tensor) -> torch.Tensor:
    """(N, 3) colours -> (N, 1, 3) DC coefficients; exact inverse of the rasterizer's colour for colours >= 0."""
    return ((colors - 0.5) / SH_C0)[:, None, :].contiguous()


def depth_from_render(img: torch.Tensor):
    """(3, H, W) render of the (z, 1, z^2) colours -> expected depth D, alpha A, relative depth std, A > 0."""
    A = img[1]
    has = A > 0
    Ac = A.clamp_min(1e-12)
    D = img[0] / Ac
    var = (img[2] / Ac - D * D).clamp_min(0.0)
    rel_std = var.sqrt() / D.abs().clamp_min(1e-12)
    D = torch.where(has, D, torch.zeros_like(D))
    rel_std = torch.where(has, rel_std, torch.full_like(rel_std, math.inf))
    return D, A, rel_std, has


def valid_depth(A: torch.Tensor, rel_std: torch.Tensor, cfg: ReinitConfig) -> torch.Tensor:
    return (A >= cfg.alpha_min) & (rel_std <= cfg.rel_std_max)


# ----------------------------------------------------------------------------------------------------------- candidates
def grid_pixels(height: int, width: int, stride: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(stride // 2, height, stride, device=device)
    xs = torch.arange(stride // 2, width, stride, device=device)
    iy, ix = torch.meshgrid(ys, xs, indexing='ij')
    return iy.reshape(-1), ix.reshape(-1)


def pixel_status(A: torch.Tensor, rel_std: torch.Tensor, cfg: ReinitConfig) -> torch.Tensor:
    """Status from the depth render alone (KEPT means: eligible for the geometric test)."""
    status = torch.full(A.shape, KEPT, dtype=torch.int8, device=A.device)
    status[A <= 0] = EMPTY
    status[(A > 0) & (A < cfg.alpha_min)] = LOW_ALPHA
    status[(A >= cfg.alpha_min) & ~(rel_std <= cfg.rel_std_max)] = MIXTURE
    return status


def geometric_test(ref: Camera, u: torch.Tensor, v: torch.Tensor, d: torch.Tensor, neighbours, cfg: ReinitConfig):
    """Test reference pixels (centres u, v, depth d) against neighbour depth maps.

    neighbours: list of (Camera, D (H, W), valid (H, W) bool). A neighbour agrees when the point projects inside it, the
    neighbour's depth there (bilinear over the 4 surrounding pixel centres, all of which must be valid) is within
    depth_tol (relative), and the point on the neighbour's ray at that depth reprojects into the reference within
    reproj_tol pixels (the depth difference expressed as reference-image parallax).
    Bilinear depth removes the pixel quantization on smooth slanted surfaces; at depth edges the 4-tap validity and
    blended depths make the test fail (reject) rather than pass.
    Returns the world points (M, 3) and the outcome codes (M, K) int8.
    """
    X = backproject(ref, u, v, d)
    codes = torch.full((X.shape[0], len(neighbours)), OUT_OF_VIEW, dtype=torch.int8, device=X.device)
    for j, (cam, Dn, valn) in enumerate(neighbours):
        pu, pv, z = project(cam, X)
        _, _, inside = pixel_index(cam, pu, pv, z)
        dn, target_ok = sample_bilinear(Dn, valn, pu, pv)
        target_ok &= inside
        depth_ok = target_ok & ((z - dn).abs() <= cfg.depth_tol * dn)
        X2 = backproject(cam, pu, pv, dn)
        ru, rv, rz = project(ref, X2)
        reproj_ok = depth_ok & (rz > 0) & (torch.hypot(ru - u, rv - v) <= cfg.reproj_tol)
        c = torch.full_like(codes[:, j], AGREE)
        c[~reproj_ok] = REPROJ_MISMATCH
        c[~depth_ok] = DEPTH_MISMATCH
        c[~target_ok] = TARGET_INVALID
        c[~inside] = OUT_OF_VIEW
        codes[:, j] = c
    return X, codes


def select_neighbours(cams: list[Camera], samples: list[torch.Tensor], k: int, min_angle_deg: float):
    """For each view, the k views that see the largest share of its sample points in-frustum, among views whose median
    triangulation angle over those points is >= min_angle_deg. samples[v]: (S, 3) world points of view v."""
    dev = cams[0].w2c.device
    W2C = torch.stack([c.w2c for c in cams])
    centers = torch.stack([c.center for c in cams])
    fx = torch.tensor([c.fx for c in cams], device=dev)[:, None]
    fy = torch.tensor([c.fy for c in cams], device=dev)[:, None]
    cx = torch.tensor([c.cx for c in cams], device=dev)[:, None]
    cy = torch.tensor([c.cy for c in cams], device=dev)[:, None]
    wd = torch.tensor([c.width for c in cams], device=dev)[:, None]
    ht = torch.tensor([c.height for c in cams], device=dev)[:, None]
    nr = torch.tensor([c.near for c in cams], device=dev)[:, None]
    out, info = [], []
    for v, X in enumerate(samples):
        if X.shape[0] == 0:
            out.append([])
            info.append(dict(frac=[], angle=[]))
            continue
        pc = torch.einsum('vij,sj->vsi', W2C[:, :3, :3], X) + W2C[:, None, :3, 3]
        z = pc[..., 2]
        pu = fx * pc[..., 0] / z + cx
        pv = fy * pc[..., 1] / z + cy
        inside = (z > nr) & (pu >= 0) & (pu < wd) & (pv >= 0) & (pv < ht)
        frac = inside.to(X.dtype).mean(dim=1)
        rv = X - centers[v]
        ru = X[None] - centers[:, None]
        cos = (rv[None] * ru).sum(-1) / (rv.norm(dim=-1)[None] * ru.norm(dim=-1)).clamp_min(1e-12)
        ang = torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0))).median(dim=1).values
        score = torch.where((frac > 0) & (ang >= min_angle_deg), frac, torch.full_like(frac, -1.0))
        score[v] = -1.0
        top = torch.topk(score, min(k, len(cams) - 1))
        chosen = [int(i) for s, i in zip(top.values.tolist(), top.indices.tolist()) if s > 0]
        out.append(chosen)
        info.append(dict(frac=[round(float(frac[i]), 4) for i in chosen], angle=[round(float(ang[i]), 2) for i in chosen]))
    return out, info


# ----------------------------------------------------------------------------------------------------------- voxels
def footprint(z: torch.Tensor, focal: float, stride: int) -> torch.Tensor:
    """World size of `stride` pixels at depth z: the candidate spacing on a fronto-parallel surface."""
    return z * stride / focal


def levels(fp: torch.Tensor, base: float) -> torch.Tensor:
    """Level l has voxel edge base * 2^l, which lies in (fp / 2, fp]."""
    return torch.floor(torch.log2(fp / base)).clamp_min(0).long()


def _coords(X: torch.Tensor, edge: float) -> torch.Tensor:
    return torch.floor(X / edge).long()


def _packer(lo: torch.Tensor, hi: torch.Tensor):
    span = (hi - lo + 1).tolist()
    if span[0] * span[1] * span[2] >= 2 ** 62:
        raise OverflowError(f'voxel key span {span} too large to pack')

    def pack(c: torch.Tensor) -> torch.Tensor:
        c = c - lo
        return (c[:, 0] * span[1] + c[:, 1]) * span[2] + c[:, 2]
    return pack


def dedupe(X: torch.Tensor, lvl: torch.Tensor, base: float, score: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """One candidate per voxel at its own level (highest score, ties broken at random); across levels the finest
    observation wins: a candidate is dropped if its voxel already holds a kept candidate of a finer level.
    Returns the kept indices into X, ascending."""
    kept, finer = [], X.new_zeros((0, 3))
    for L in torch.unique(lvl).tolist():
        idx = (lvl == L).nonzero().squeeze(1)
        edge = base * 2.0 ** L
        c = _coords(X[idx], edge)
        lo, hi = c.min(dim=0).values, c.max(dim=0).values
        pack = _packer(lo, hi)
        if finer.shape[0]:
            cf = _coords(finer, edge)
            cf = cf[((cf >= lo) & (cf <= hi)).all(dim=1)]
            if cf.shape[0]:
                occupied = torch.isin(pack(c), pack(cf))
                idx, c = idx[~occupied], c[~occupied]
        if idx.numel() == 0:
            continue
        key = pack(c)
        r = torch.rand(idx.shape[0], generator=generator, device=X.device)
        order = torch.argsort(r, stable=True)
        order = order[torch.argsort(-score[idx][order], stable=True)]
        order = order[torch.argsort(key[order], stable=True)]
        ks = key[order]
        first = torch.ones_like(ks, dtype=torch.bool)
        first[1:] = ks[1:] != ks[:-1]
        chosen = idx[order[first]]
        kept.append(chosen)
        finer = torch.cat([finer, X[chosen]])
    if not kept:
        return torch.zeros(0, dtype=torch.long, device=X.device)
    return torch.sort(torch.cat(kept)).values


def covered_by(old_X: torch.Tensor, cand_X: torch.Tensor, cand_lvl: torch.Tensor, base: float, ring: int = 1) -> torch.Tensor:
    """Old Gaussian centres within `ring` voxels (Chebyshev, at the candidate's level) of a voxel occupied by a candidate:
    ring 0 = the same voxel; ring 1 = the 3 x 3 x 3 block around it. Candidates sit at the rendered (expected) depth
    while the old centres scatter around the surface by about a footprint, so ring 0 misses most of them."""
    covered = torch.zeros(old_X.shape[0], dtype=torch.bool, device=old_X.device)
    r = range(-ring, ring + 1)
    offsets = torch.tensor([(a, b, c) for a in r for b in r for c in r], dtype=torch.long, device=old_X.device)
    for L in torch.unique(cand_lvl).tolist():
        edge = base * 2.0 ** L
        c = _coords(cand_X[cand_lvl == L], edge)
        lo, hi = c.min(dim=0).values - ring, c.max(dim=0).values + ring
        co = _coords(old_X, edge)
        inside = ((co >= lo) & (co <= hi)).all(dim=1)
        if not inside.any():
            continue
        pack = _packer(lo - ring, hi + ring)
        occupied = pack(c)
        ci = co[inside]
        hit = torch.zeros(ci.shape[0], dtype=torch.bool, device=old_X.device)
        for o in offsets:
            hit |= torch.isin(pack(ci + o), occupied)
        covered[inside.nonzero().squeeze(1)[hit]] = True
    return covered


# ----------------------------------------------------------------------------------------------------------- counts
class CountError(RuntimeError):
    """The requested Gaussian count cannot be reached with the rule below; the run must not silently differ."""


def plan_selection(n_before: int, covered: torch.Tensor, n_candidates: int, target: int, generator: torch.Generator):
    """The replacement rule, with an exact final count.

    1. Removal uses all deduplicated candidates (before any subsampling), so the surviving rows do not depend on which
       candidates are drawn: an old row is removed iff covered_by() flags its centre (within cover_ring voxels of a
       candidate's voxel, at that candidate's level).
    2. n_add = target - n_survivors new points are drawn uniformly at random (seeded) from the candidates.
    3. Final count = n_survivors + n_add = target, asserted. n_add < 0 or n_add > n_candidates raises CountError.
    Returns (keep mask over old rows, sorted indices of the drawn candidates, counts).
    """
    n_cov = int(covered.sum())
    n_surv = n_before - n_cov
    n_add = target - n_surv
    counts = dict(n_before=n_before, n_covered=n_cov, n_survivors=n_surv, n_candidates=n_candidates, target=target,
                  n_added=n_add, max_reachable=n_surv + n_candidates)
    if n_add < 0:
        raise CountError(f'target {target} is below the {n_surv} surviving rows ({counts})')
    if n_add > n_candidates:
        raise CountError(f'need {n_add} new points but only {n_candidates} candidates ({counts})')
    pick = torch.randperm(n_candidates, generator=generator, device=covered.device)[:n_add]
    pick = torch.sort(pick).values
    counts['n_after'] = n_surv + n_add
    assert counts['n_after'] == target
    return ~covered, pick, counts


def new_rows(X: torch.Tensor, rgb: torch.Tensor, knn_rms: torch.Tensor, n_sh_rest: int, init_opacity: float) -> dict:
    """Stock initialization rules (Gaussians.initialize_from_point_cloud) except position and colour."""
    n = X.shape[0]
    rot = torch.zeros((n, 4), dtype=X.dtype, device=X.device)
    rot[:, 0] = 1.0
    return {
        'means': X.contiguous(),
        'sh_coefficients_0': ((rgb - 0.5) / SH_C0)[:, None, :].contiguous(),
        'sh_coefficients_rest': torch.zeros((n, n_sh_rest, 3), dtype=X.dtype, device=X.device),
        'opacities': torch.full((n, 1), math.log(init_opacity / (1.0 - init_opacity)), dtype=X.dtype, device=X.device),
        'scales': knn_rms.clamp_min(1e-7).log()[:, None].repeat(1, 3).contiguous(),
        'rotations': rot,
    }


def knn_rms_reference(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Brute-force RMS distance to the k nearest neighbours (tests; the trainer uses NeRFICG's fused kNN)."""
    d = torch.cdist(points, points)
    d.fill_diagonal_(math.inf)
    return d.topk(k, largest=False).values.square().mean(dim=1).sqrt()


# ----------------------------------------------------------------------------------------------------------- optimizer rows
def replace_rows(optimizer: torch.optim.Optimizer, keep: torch.Tensor, rows: dict) -> dict:
    """Reference replacement (prune `~keep`, append `rows` with zero moments), the same semantics as NeRFICG's
    Optim.adam_utils.prune_param_groups + extend_param_groups. Used by the CPU tests; the trainer uses NeRFICG's."""
    out = {}
    for group in optimizer.param_groups:
        p = group['params'][0]
        st = optimizer.state.pop(p, None)
        newp = torch.nn.Parameter(torch.cat([p.data[keep], rows[group['name']].to(p.dtype)]))
        if st:
            for key in ('exp_avg', 'exp_avg_sq'):
                st[key] = torch.cat([st[key][keep], torch.zeros_like(rows[group['name']], dtype=st[key].dtype)])
            optimizer.state[newp] = st
        group['params'][0] = newp
        out[group['name']] = newp
    return out


def reorder_rows(optimizer: torch.optim.Optimizer, order: torch.Tensor) -> dict:
    """Reference reordering (NeRFICG sort_param_groups semantics)."""
    out = {}
    for group in optimizer.param_groups:
        p = group['params'][0]
        st = optimizer.state.pop(p, None)
        newp = torch.nn.Parameter(p.data[order])
        if st:
            for key in ('exp_avg', 'exp_avg_sq'):
                st[key] = st[key][order]
            optimizer.state[newp] = st
        group['params'][0] = newp
        out[group['name']] = newp
    return out


def _step_value(step):
    return step.item() if torch.is_tensor(step) else step


def snapshot(optimizer: torch.optim.Optimizer, keep_idx: torch.Tensor) -> dict:
    """Copies of the surviving rows (parameters, both moments) and every group's step, before the replacement."""
    snap = {}
    for group in optimizer.param_groups:
        p = group['params'][0]
        st = optimizer.state.get(p, {})
        snap[group['name']] = dict(
            param=p.data[keep_idx].clone(),
            exp_avg=st['exp_avg'][keep_idx].clone() if 'exp_avg' in st else None,
            exp_avg_sq=st['exp_avg_sq'][keep_idx].clone() if 'exp_avg_sq' in st else None,
            step=_step_value(st['step']) if 'step' in st else None,
        )
    return snap


def verify(optimizer: torch.optim.Optimizer, snap: dict, order: torch.Tensor, n_survivors: int, rows: dict) -> list[str]:
    """Post-conditions of replace (+ reorder by `order`, where final row j came from intermediate row order[j]; the
    intermediate layout is [survivors in their old order, new rows in draw order]):
    survivors keep their parameters and both moments exactly, new rows hold the constructed values with zero moments,
    every group's step counter is unchanged, and every group has len(order) rows. Returns the failed checks."""
    fails = []
    for group in optimizer.param_groups:
        name = group['name']
        p = group['params'][0]
        st = optimizer.state.get(p, {})
        s = snap[name]
        if p.shape[0] != order.shape[0]:
            fails.append(f'{name}: {p.shape[0]} rows, expected {order.shape[0]}')
            continue
        old = order < n_survivors
        src_old, src_new = order[old], order[~old] - n_survivors
        if not torch.equal(p.data[old], s['param'][src_old]):
            fails.append(f'{name}: survivor parameters changed')
        if not torch.equal(p.data[~old], rows[name].to(p.dtype)[src_new]):
            fails.append(f'{name}: new-row parameters differ from the constructed rows')
        for key in ('exp_avg', 'exp_avg_sq'):
            if s[key] is None:
                if key in st:
                    fails.append(f'{name}: {key} appeared')
                continue
            if key not in st:
                fails.append(f'{name}: {key} missing')
                continue
            if not torch.equal(st[key][old], s[key][src_old]):
                fails.append(f'{name}: survivor {key} changed')
            if bool((st[key][~old] != 0).any()):
                fails.append(f'{name}: new-row {key} not zero')
        if s['step'] is not None and _step_value(st.get('step')) != s['step']:
            fails.append(f'{name}: step changed {s["step"]} -> {_step_value(st.get("step"))}')
    return fails


# ----------------------------------------------------------------------------------------------------------- photo evidence
def to_gray(img: torch.Tensor) -> torch.Tensor:
    return img[:3].mean(dim=0)


def sample_patches(gray: torch.Tensor, u: torch.Tensor, v: torch.Tensor, radius: int) -> torch.Tensor:
    """(H, W) image, continuous centres -> (M, (2r+1)^2) bilinear patches (axis-aligned, fronto-parallel)."""
    H, W = gray.shape
    off = torch.arange(-radius, radius + 1, device=gray.device, dtype=u.dtype)
    oy, ox = torch.meshgrid(off, off, indexing='ij')
    x = u[:, None] + ox.reshape(1, -1)
    y = v[:, None] + oy.reshape(1, -1)
    grid = torch.stack([2.0 * x / W - 1.0, 2.0 * y / H - 1.0], dim=-1)[None]
    out = torch.nn.functional.grid_sample(gray[None, None].to(u.dtype), grid, mode='bilinear', padding_mode='border',
                                          align_corners=False)
    return out[0, 0]


def zncc(a: torch.Tensor, b: torch.Tensor):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    na, nb = a.square().sum(dim=1), b.square().sum(dim=1)
    return (a * b).sum(dim=1) / (na * nb).sqrt().clamp_min(1e-12), a.std(dim=1), b.std(dim=1)


def photo_consistency(gray_a, ua, va, gray_b, ub, vb, radius: int = 3, min_std: float = 0.02):
    """ZNCC of 7x7 patches in the two training photos. verifiable = both patches inside their images and textured."""
    Ha, Wa = gray_a.shape
    Hb, Wb = gray_b.shape
    inside = ((ua - radius >= 0) & (ua + radius < Wa) & (va - radius >= 0) & (va + radius < Ha) &
              (ub - radius >= 0) & (ub + radius < Wb) & (vb - radius >= 0) & (vb + radius < Hb) &
              torch.isfinite(ub) & torch.isfinite(vb))
    score, sa, sb = zncc(sample_patches(gray_a, ua, va, radius), sample_patches(gray_b, ub, vb, radius))
    verifiable = inside & (sa >= min_std) & (sb >= min_std)
    return torch.where(verifiable, score, torch.full_like(score, math.nan)), verifiable


def photo_class_counts(score: torch.Tensor, verifiable: torch.Tensor) -> dict:
    s = score[verifiable]
    return dict(n=int(score.numel()), unverifiable=int((~verifiable).sum()), consistent=int((s >= 0.7).sum()),
                ambiguous=int(((s >= 0.3) & (s < 0.7)).sum()), inconsistent=int((s < 0.3).sum()),
                median=float(s.median()) if s.numel() else None)


# ----------------------------------------------------------------------------------------------------------- summaries
def quantiles(x: torch.Tensor, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    x = x[torch.isfinite(x)].double()
    if x.numel() == 0:
        return dict(n=0)
    if x.numel() > 10_000_000:
        x = x[torch.randperm(x.numel(), generator=torch.Generator(device=x.device).manual_seed(0), device=x.device)[:10_000_000]]
    q = torch.quantile(x, torch.tensor(qs, dtype=x.dtype, device=x.device))
    return dict(n=int(x.numel()), **{f'q{int(100 * a)}': float(b) for a, b in zip(qs, q.tolist())})


def count_table(codes: torch.Tensor, groups: torch.Tensor, n_codes: int, n_groups: int) -> list[list[int]]:
    """counts[group][code]."""
    flat = groups.long() * n_codes + codes.long()
    return torch.bincount(flat, minlength=n_codes * n_groups).reshape(n_groups, n_codes).tolist()


def depth_bin(d: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(d, edges).clamp(0, edges.numel())


def region_id(iy: torch.Tensor, ix: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """3 x 3 image regions, row-major."""
    return (iy * 3 // height).clamp(0, 2) * 3 + (ix * 3 // width).clamp(0, 2)


def world_transform(c2w_dataset: list, c2w_colmap: list):
    """Transform from the COLMAP world to the dataset's world (NeRFICG applies a PCA alignment), recovered from the
    camera poses: T = c2w_dataset @ inv(c2w_colmap) for every view; returns T (from the first view) and the largest
    element-wise deviation across views (a rigid or similarity transform makes it ~0)."""
    Ts = [torch.as_tensor(a, dtype=torch.float64) @ torch.linalg.inv(torch.as_tensor(b, dtype=torch.float64))
          for a, b in zip(c2w_dataset, c2w_colmap)]
    dev = max(float((T - Ts[0]).abs().max()) for T in Ts)
    return Ts[0], dev


class PartTimer:
    """Wall time per named part; synchronizes the device at both ends when sync is given."""

    def __init__(self, sync=None, clock=None):
        import time
        self.parts: dict[str, float] = {}
        self._sync = sync or (lambda: None)
        self._clock = clock or time.perf_counter

    def __call__(self, name: str):
        timer = self

        class _Part:
            def __enter__(self_inner):
                timer._sync()
                self_inner.t0 = timer._clock()

            def __exit__(self_inner, *exc):
                timer._sync()
                timer.parts[name] = timer.parts.get(name, 0.0) + timer._clock() - self_inner.t0
                return False
        return _Part()

    @property
    def total(self) -> float:
        return sum(self.parts.values())

