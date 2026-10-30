"""CPU unit tests of alt_reinit/reinit_core.py (no GPU, no NeRFICG):  python -m pytest -q alt_reinit/tests/test_core_cpu.py

Synthetic scenes with analytic depth maps (ray casting of planes) stand in for the rasterizer's depth here; the GPU
checks of the real depth render are in test_gpu.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reinit_core as rc  # noqa: E402

torch.set_default_dtype(torch.float32)


# ------------------------------------------------------------------------------------------------ synthetic cameras
def look_at(center, target, up=(0.0, -1.0, 0.0)) -> torch.Tensor:
    """w2c for a camera at `center` looking at `target` (+z forward, +y down in the image, as in Faster-GS)."""
    c = torch.tensor(center, dtype=torch.float64)
    f = torch.tensor(target, dtype=torch.float64) - c
    f = f / f.norm()
    u = torch.tensor(up, dtype=torch.float64)
    r = torch.linalg.cross(u, f)  # image +x
    r = r / r.norm()
    d = torch.linalg.cross(f, r)  # image +y (down)
    R = torch.stack([r, d, f])
    w2c = torch.eye(4, dtype=torch.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ c
    return w2c.float()


def orbit(n=12, radius=4.0, height=-1.5, W=320, H=208, f=290.0, target=(0.0, 0.0, 0.0)) -> list[rc.Camera]:
    cams = []
    for i in range(n):
        a = 2 * math.pi * i / n
        w2c = look_at((radius * math.cos(a), height, radius * math.sin(a)), target)
        cams.append(rc.Camera(w2c, f, f, W / 2, H / 2, W, H, near=0.05))
    return cams


def plane_depth(cam: rc.Camera, normal, offset) -> torch.Tensor:
    """Camera depth of the plane normal . X + offset = 0 at every pixel centre (inf where the ray misses)."""
    iy, ix = torch.meshgrid(torch.arange(cam.height), torch.arange(cam.width), indexing='ij')
    u, v = ix.reshape(-1) + 0.5, iy.reshape(-1) + 0.5
    X1 = rc.backproject(cam, u, v, torch.ones_like(u))   # point at camera depth 1 on each ray
    C = cam.center.expand_as(X1)
    d = X1 - C
    n = torch.tensor(normal, dtype=torch.float32)
    t = -(C @ n + offset) / (d @ n)
    z = torch.where(t > 0, t, torch.full_like(t, math.inf))   # camera depth = t, since d has camera depth 1
    return z.reshape(cam.height, cam.width)


GROUND = ((0.0, -1.0, 0.0), 1.0)   # plane y = 1 (the ground; +y is down)


def scene_maps(cams, normal=GROUND[0], offset=GROUND[1]):
    maps = []
    for c in cams:
        D = plane_depth(c, normal, offset)
        A = torch.where(torch.isfinite(D), torch.ones_like(D), torch.zeros_like(D))
        D = torch.where(torch.isfinite(D), D, torch.zeros_like(D))
        rel = torch.where(A > 0, torch.zeros_like(D), torch.full_like(D, math.inf))
        maps.append((D, A, rel))
    return maps


CFG = rc.ReinitConfig(mode='apply', start=2000, end=2000)


# ------------------------------------------------------------------------------------------------ config
def test_parse_env():
    assert rc.parse_env('').mode == 'off'
    c = rc.parse_env('mode=audit,iters=2000:5000,min_agree=0,check=0,depth_tol=0.02')
    assert (c.mode, c.start, c.end, c.every, c.min_agree, c.check, c.depth_tol) == ('audit', 2000, 5000, 3000, 0, False, 0.02)
    c = rc.parse_env('mode=apply,iters=2000,target=+500')
    assert (c.start, c.end, c.every) == (2000, 2000, 1) and rc.resolve_target(c.target, 1000) == 1500
    assert rc.resolve_target('match', 7) == 7 and rc.resolve_target('4166000', 7) == 4166000
    assert rc.parse_env('mode=audit,iters=2000:5000,target=4166000/match').target == '4166000/match'
    for bad in ('mode=apply,iters=2000:5000', 'mode=x,iters=1', 'mode=apply', 'mode=apply,iters=10,min_agree=5',
                'mode=apply,iters=10,bogus=1', 'mode=apply,iters=10,target=lots', 'mode=apply,iters=10,target=match/+5',
                'mode=audit,iters=10,target=match/x'):
        with pytest.raises(ValueError):
            rc.parse_env(bad)


# ------------------------------------------------------------------------------------------------ cameras and depth
def test_round_trip_with_garden_intrinsics():
    g = torch.Generator().manual_seed(0)
    for cam in orbit(W=1297, H=840, f=1160.0)[:4]:
        u = torch.rand(5000, generator=g) * cam.width
        v = torch.rand(5000, generator=g) * cam.height
        z = 0.2 + 20 * torch.rand(5000, generator=g)
        X = rc.backproject(cam, u, v, z)
        pu, pv, pz = rc.project(cam, X)
        assert (pu - u).abs().max() < 5e-3 and (pv - v).abs().max() < 5e-3   # fp32 world coordinates (the rasterizer is fp32 too)
        assert ((pz - z).abs() / z).max() < 1e-5


def test_pixel_index_convention():
    cam = orbit()[0]
    u = torch.tensor([0.0, 0.49, 0.5, 1.0, cam.width - 1e-3, cam.width + 0.0, -1e-3])
    v = torch.full_like(u, 10.5)
    ix, iy, inside = rc.pixel_index(cam, u, v)
    assert ix[:5].tolist() == [0, 0, 0, 1, cam.width - 1] and iy[0] == 10
    assert inside.tolist() == [True] * 5 + [False, False]
    _, _, inside = rc.pixel_index(cam, u, v, z=torch.full_like(u, 0.01))   # behind the near plane
    assert not inside.any()


def test_depth_from_render_layers_and_empty():
    # pixel 0: one opaque layer at 2; pixel 1: 1 m (alpha .5) over 3 m (alpha .999) -> nearly opaque, mean ~2, big std;
    # pixel 2: empty
    w1 = torch.tensor([0.999, 0.5, 0.0])
    w2 = torch.tensor([0.0, 0.5 * 0.999, 0.0])
    z1, z2 = torch.tensor([2.0, 1.0, 0.0]), torch.tensor([0.0, 3.0, 0.0])
    img = torch.stack([w1 * z1 + w2 * z2, w1 + w2, w1 * z1 ** 2 + w2 * z2 ** 2])[:, None, :]   # (3, 1, 3)
    D, A, rel, has = rc.depth_from_render(img)
    assert torch.allclose(D[0, :2], torch.tensor([2.0, (0.5 + 0.5 * 0.999 * 3) / (0.5 + 0.5 * 0.999)]), atol=1e-5)
    assert A[0, 1] > 0.95 and rel[0, 0] < 1e-3 and rel[0, 1] > 0.4
    assert not has[0, 2] and math.isinf(rel[0, 2]) and D[0, 2] == 0
    st = rc.pixel_status(A, rel, CFG)
    assert st[0].tolist() == [rc.KEPT, rc.MIXTURE, rc.EMPTY]
    assert rc.pixel_status(torch.tensor([0.5]), torch.tensor([0.0]), CFG).tolist() == [rc.LOW_ALPHA]


def test_sh0_colour_inverse():
    c = torch.tensor([[2.5, 1.0, 6.25], [0.2, 1.0, 0.04], [150.0, 1.0, 22500.0]])
    sh0 = rc.sh0_from_colors(c)
    assert sh0.shape == (3, 1, 3)
    assert torch.allclose(0.5 + rc.SH_C0 * sh0[:, 0], c, rtol=1e-6)


# ------------------------------------------------------------------------------------------------ geometric test
def _neighbours(cams, maps, idx, cfg=CFG):
    return [(cams[j], maps[j][0], rc.valid_depth(maps[j][1], maps[j][2], cfg)) for j in idx]


def test_plane_seen_by_all_views_is_kept_and_floater_rejected():
    cams = orbit()
    maps = scene_maps(cams)
    ref = 0
    D, A, rel = maps[ref]
    # a floater only in the reference's depth map: a patch rendered at half its true depth
    D_ref = D.clone()
    D_ref[150:180, 140:180] *= 0.5
    iy, ix = rc.grid_pixels(cams[ref].height, cams[ref].width, 4)
    ok = rc.pixel_status(A, rel, CFG)[iy, ix] == rc.KEPT
    iy, ix = iy[ok], ix[ok]
    X, codes = rc.geometric_test(cams[ref], ix + 0.5, iy + 0.5, D_ref[iy, ix], _neighbours(cams, maps, [1, 11, 2, 10]), CFG)
    agree = (codes == rc.AGREE).sum(1)
    floater = (iy >= 150) & (iy < 180) & (ix >= 140) & (ix < 180)
    seen = (codes != rc.OUT_OF_VIEW).sum(1) >= 2
    assert floater.any() and (agree[floater] == 0).all()
    assert (agree[~floater & seen] >= 2).float().mean() > 0.99
    assert torch.allclose(X[~floater][:, 1], torch.ones(int((~floater).sum())), atol=1e-3)   # kept points lie on y = 1
    assert (codes[floater] == rc.DEPTH_MISMATCH).sum() > 0


def test_consistent_wrong_surface_passes():
    """Documents the limit of the filter: a wrong surface rendered consistently in every view (here a plane 5 cm above
    the true ground) is kept. Agreement between depth maps is not correctness."""
    cams = orbit()
    wrong = scene_maps(cams, offset=0.95)   # y = 0.95 in every depth map
    iy, ix = rc.grid_pixels(cams[0].height, cams[0].width, 8)
    D = wrong[0][0]
    ok = D[iy, ix] > 0
    X, codes = rc.geometric_test(cams[0], ix[ok] + 0.5, iy[ok] + 0.5, D[iy[ok], ix[ok]], _neighbours(cams, wrong, [1, 11]), CFG)
    both = (codes != rc.OUT_OF_VIEW).all(1)
    assert both.sum() > 100 and ((codes[both] == rc.AGREE).all(1)).float().mean() > 0.99
    assert torch.allclose(X[:, 1], torch.full((X.shape[0],), 0.95), atol=1e-3)


def test_select_neighbours_prefers_overlap_and_rejects_zero_baseline():
    cams = orbit(n=12)
    dup = rc.Camera(cams[0].w2c.clone(), cams[0].fx, cams[0].fy, cams[0].cx, cams[0].cy, cams[0].width, cams[0].height, 0.05)
    cams = cams + [dup]  # index 12: same pose as view 0
    maps = scene_maps(cams)
    g = torch.Generator().manual_seed(0)
    samples = []
    for c, (D, A, rel) in zip(cams, maps):
        iy, ix = rc.grid_pixels(c.height, c.width, 8)
        ok = A[iy, ix] > 0
        sel = torch.randperm(int(ok.sum()), generator=g)[:256]
        samples.append(rc.backproject(c, ix[ok][sel] + 0.5, iy[ok][sel] + 0.5, D[iy[ok][sel], ix[ok][sel]]))
    nbrs, info = rc.select_neighbours(cams, samples, k=4, min_angle_deg=3.0)
    assert 12 not in nbrs[0] and 0 not in nbrs[12]          # zero triangulation angle
    assert set(nbrs[0][:2]) == {1, 11}                      # adjacent views overlap most
    assert all(a >= 3.0 for a in info[0]['angle'])


# ------------------------------------------------------------------------------------------------ voxels
def test_levels_and_footprint():
    fp = torch.tensor([1.0, 1.5, 1.99, 2.0, 7.9, 0.5])
    lv = rc.levels(fp, 1.0)
    edge = 2.0 ** lv.double()
    assert lv.tolist() == [0, 0, 0, 1, 2, 0]
    big = fp >= 1.0
    assert ((edge[big] <= fp[big].double()) & (edge[big] > fp[big].double() / 2)).all()
    assert torch.allclose(rc.footprint(torch.tensor([2.0]), 1000.0, 2), torch.tensor([0.004]))


def test_dedupe_one_per_voxel_and_finest_wins():
    g = torch.Generator().manual_seed(0)
    X = torch.tensor([[0.10, 0.10, 0.10],   # level 0 (edge 1), voxel (0,0,0), score 1
                      [0.90, 0.20, 0.30],   # same voxel, score 3 -> wins
                      [1.50, 0.20, 0.30],   # level 0, voxel (1,0,0)
                      [0.50, 0.50, 0.50],   # level 1 (edge 2), voxel (0,0,0): holds finer kept points -> dropped
                      [5.00, 5.00, 5.00]])  # level 1, voxel (2,2,2): kept
    lvl = torch.tensor([0, 0, 0, 1, 1])
    score = torch.tensor([1.0, 3.0, 0.0, 9.0, 0.0])
    keep = rc.dedupe(X, lvl, 1.0, score, g)
    assert keep.tolist() == [1, 2, 4]


def test_dedupe_ties_are_seeded():
    X = torch.rand(1000, 3, generator=torch.Generator().manual_seed(1)) * 0.5   # all in voxel 0 at edge 1
    lvl = torch.zeros(1000, dtype=torch.long)
    score = torch.zeros(1000)
    a = rc.dedupe(X, lvl, 1.0, score, torch.Generator().manual_seed(5))
    b = rc.dedupe(X, lvl, 1.0, score, torch.Generator().manual_seed(5))
    c = rc.dedupe(X, lvl, 1.0, score, torch.Generator().manual_seed(6))
    assert a.numel() == 1 and torch.equal(a, b) and not torch.equal(a, c)


def test_covered_by():
    cand = torch.tensor([[0.5, 0.5, 0.5], [10.2, 0.1, 0.1]])
    lvl = torch.tensor([0, 1])   # voxel (0,0,0) at edge 1, voxel (5,0,0) at edge 2
    old = torch.tensor([[0.9, 0.1, 0.99],    # in voxel (0,0,0) at edge 1 -> covered
                        [1.1, 0.1, 0.1],     # voxel (1,0,0) at edge 1, (0,0,0) at edge 2 (not occupied) -> not covered
                        [11.9, 1.9, 1.9],    # (5,0,0) at edge 2 -> covered
                        [-0.1, 0.5, 0.5],    # (-1,0,0) -> not covered
                        [1e4, 1e4, 1e4]])    # far away -> not covered
    assert rc.covered_by(old, cand, lvl, 1.0, ring=0).tolist() == [True, False, True, False, False]
    # ring 1: also the 3x3x3 block around each occupied voxel ((1,0,0) and (-1,0,0) neighbour (0,0,0) at edge 1)
    assert rc.covered_by(old, cand, lvl, 1.0, ring=1).tolist() == [True, True, True, True, False]
    assert rc.covered_by(old, cand, lvl, 1.0).tolist() == [True, True, True, True, False]   # default ring 1
    far = torch.tensor([[2.5, 0.5, 0.5]])   # (2,0,0) at edge 1: two voxels away
    assert rc.covered_by(far, cand[:1], lvl[:1], 1.0, ring=1).tolist() == [False]


# ------------------------------------------------------------------------------------------------ counts
def test_plan_selection_exact_counts():
    g = torch.Generator().manual_seed(0)
    covered = torch.zeros(100, dtype=torch.bool)
    covered[:30] = True
    keep, pick, counts = rc.plan_selection(100, covered, 50, 100, g)          # matched count: adds = removals
    assert counts['n_added'] == 30 and counts['n_after'] == 100 and int(keep.sum()) + pick.numel() == 100
    assert counts['n_survivors'] == 70 and counts['max_reachable'] == 120
    keep, pick, counts = rc.plan_selection(100, covered, 50, 110, g)          # target count
    assert int(keep.sum()) + pick.numel() == 110 and pick.unique().numel() == pick.numel()
    with pytest.raises(rc.CountError):
        rc.plan_selection(100, covered, 20, 100, g)                           # 30 needed, 20 available
    with pytest.raises(rc.CountError):
        rc.plan_selection(100, covered, 50, 60, g)                            # below the 70 survivors


def test_new_rows_follow_stock_init():
    X = torch.randn(10, 3)
    rgb = torch.rand(10, 3)
    rows = rc.new_rows(X, rgb, torch.full((10,), 0.02), 15, 0.1)
    assert rows['sh_coefficients_rest'].shape == (10, 15, 3) and rows['sh_coefficients_0'].shape == (10, 1, 3)
    assert torch.allclose(0.5 + rc.SH_C0 * rows['sh_coefficients_0'][:, 0], rgb, atol=1e-6)
    assert torch.allclose(torch.sigmoid(rows['opacities']), torch.full((10, 1), 0.1))
    assert torch.allclose(rows['scales'].exp(), torch.full((10, 3), 0.02))
    assert torch.equal(rows['rotations'][:, 0], torch.ones(10)) and not rows['rotations'][:, 1:].any()


# ------------------------------------------------------------------------------------------------ optimizer rows
def _model(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = dict(means=(3,), sh_coefficients_0=(1, 3), sh_coefficients_rest=(15, 3), opacities=(1,), scales=(3,), rotations=(4,))
    params = {k: torch.nn.Parameter(torch.randn(n, *s, generator=g)) for k, s in shapes.items()}
    opt = torch.optim.Adam([{'params': [p], 'name': k, 'lr': 1e-2} for k, p in params.items()], eps=1e-15)
    for _ in range(3):
        opt.zero_grad()
        sum((p ** 2).sum() for p in params.values()).backward()
        opt.step()
    return params, opt


def _replace(opt, n, n_cov, n_new, seed):
    g = torch.Generator().manual_seed(seed)
    covered = torch.zeros(n, dtype=torch.bool)
    covered[torch.randperm(n, generator=g)[:n_cov]] = True
    keep, pick, counts = rc.plan_selection(n, covered, n_new + 5, n - n_cov + n_new, g)
    rows = {gr['name']: torch.randn(pick.numel(), *gr['params'][0].shape[1:], generator=g) for gr in opt.param_groups}
    snap = rc.snapshot(opt, keep.nonzero().squeeze(1))
    rc.replace_rows(opt, keep, rows)
    order = torch.randperm(counts['n_after'], generator=g)
    rc.reorder_rows(opt, order)
    return snap, order, counts, rows


def test_replace_verify_passes_and_state_moves_with_rows():
    params, opt = _model()
    before = {gr['name']: (gr['params'][0].data.clone(), opt.state[gr['params'][0]]['exp_avg'].clone()) for gr in opt.param_groups}
    snap, order, counts, rows = _replace(opt, 64, 20, 25, 1)
    assert rc.verify(opt, snap, order, counts['n_survivors'], rows) == []
    # independent check of one surviving row through the reordering
    p = opt.param_groups[0]['params'][0]
    j = int((order < counts['n_survivors']).nonzero()[0])
    keep_idx = snap['means']['param']
    assert torch.equal(p.data[j], keep_idx[order[j]])
    orig_row = int(((before['means'][0] == p.data[j]).all(1)).nonzero())
    assert torch.equal(opt.state[p]['exp_avg'][j], before['means'][1][orig_row])
    # training can continue on the new rows
    opt.zero_grad()
    sum((gr['params'][0] ** 2).sum() for gr in opt.param_groups).backward()
    opt.step()


@pytest.mark.parametrize('corrupt', ['survivor_moment', 'new_moment', 'survivor_param', 'new_param', 'step', 'count'])
def test_verify_detects_corruption(corrupt):
    params, opt = _model()
    snap, order, counts, rows = _replace(opt, 64, 20, 25, 2)
    gr = opt.param_groups[3]
    p = gr['params'][0]
    st = opt.state[p]
    old = (order < counts['n_survivors']).nonzero().squeeze(1)
    new = (order >= counts['n_survivors']).nonzero().squeeze(1)
    if corrupt == 'survivor_moment':
        st['exp_avg_sq'][old[0]] += 1.0
    elif corrupt == 'new_moment':
        st['exp_avg'][new[0]] = 1e-3
    elif corrupt == 'survivor_param':
        p.data[old[-1]] += 1e-6
    elif corrupt == 'new_param':
        p.data[new[0]] = 0.0
    elif corrupt == 'step':
        st['step'] = st['step'] + 1
    elif corrupt == 'count':
        order = order[:-1]
    fails = rc.verify(opt, snap, order, counts['n_survivors'], rows)
    assert fails, corrupt


# ------------------------------------------------------------------------------------------------ photo evidence
def test_photo_consistency():
    g = torch.Generator().manual_seed(0)
    a = torch.rand(80, 120, generator=g)
    b = torch.roll(a, shifts=5, dims=1)       # b(x) = a(x - 5)
    u = torch.tensor([30.5, 60.5, 90.5])
    v = torch.tensor([40.5, 40.5, 40.5])
    s, ver = rc.photo_consistency(a, u, v, b, u + 5, v)
    assert ver.all() and (s > 0.99).all()
    s, ver = rc.photo_consistency(a, u, v, b, u + 17, v)
    assert ver.all() and (s.abs() < 0.5).all()
    flat = torch.full((80, 120), 0.4)
    s, ver = rc.photo_consistency(flat, u, v, b, u + 5, v)
    assert not ver.any() and torch.isnan(s).all()
    s, ver = rc.photo_consistency(a, torch.tensor([1.0]), torch.tensor([40.5]), b, torch.tensor([6.0]), torch.tensor([40.5]))
    assert not ver.any()                     # patch leaves the image
    c = rc.photo_class_counts(torch.tensor([0.9, 0.5, 0.1, math.nan]), torch.tensor([True, True, True, False]))
    assert (c['consistent'], c['ambiguous'], c['inconsistent'], c['unverifiable']) == (1, 1, 1, 1)


def test_world_transform_recovers_similarity():
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(3)))[0]
    T[:3, 3] = torch.tensor([0.3, -1.0, 2.0], dtype=torch.float64)
    col = [torch.linalg.inv(look_at((math.cos(a) * 3, -1.0, math.sin(a) * 3), (0, 0, 0)).double()) for a in (0.1, 1.3, 2.9)]
    ds = [T @ c for c in col]
    Tr, dev = rc.world_transform(ds, col)
    assert torch.allclose(Tr, T, atol=1e-10) and dev < 1e-10


# ------------------------------------------------------------------------------------------------ end to end
def test_pipeline_on_synthetic_scene_hits_exact_count():
    """Candidates from 12 orbit views of the ground plane (one view has a floater), filter, dedupe, cover, select."""
    cams = orbit()
    maps = scene_maps(cams)
    maps[3][0][100:140, 100:160] *= 0.6   # floater in view 3 only
    g = torch.Generator().manual_seed(0)
    samples = []
    for c, (D, A, rel) in zip(cams, maps):
        iy, ix = rc.grid_pixels(c.height, c.width, 8)
        ok = A[iy, ix] > 0
        samples.append(rc.backproject(c, ix[ok] + 0.5, iy[ok] + 0.5, D[iy[ok], ix[ok]])[:512])
    nbrs, _ = rc.select_neighbours(cams, samples, 4, 3.0)
    stride = 4
    Xs, scores, fps, floater_pts = [], [], [], 0
    for v, (c, (D, A, rel)) in enumerate(zip(cams, maps)):
        iy, ix = rc.grid_pixels(c.height, c.width, stride)
        st = rc.pixel_status(A, rel, CFG)[iy, ix]
        e = st == rc.KEPT
        X, codes = rc.geometric_test(c, ix[e] + 0.5, iy[e] + 0.5, D[iy[e], ix[e]], _neighbours(cams, maps, nbrs[v]), CFG)
        agree = (codes == rc.AGREE).sum(1)
        k = agree >= CFG.min_agree
        if v == 3:
            fl = (iy[e] >= 100) & (iy[e] < 140) & (ix[e] >= 100) & (ix[e] < 160)
            floater_pts = int((k & fl).sum())
        Xs.append(X[k])
        scores.append(agree[k].float())
        fps.append(rc.footprint(D[iy[e], ix[e]][k], c.fx, stride))
    X, score, fp = torch.cat(Xs), torch.cat(scores), torch.cat(fps)
    assert floater_pts == 0
    assert torch.allclose(X[:, 1], torch.ones(X.shape[0]), atol=1e-3)
    base = float(fp.min())
    lvl = rc.levels(fp, base)
    keep = rc.dedupe(X, lvl, base, score, g)
    assert 0 < keep.numel() < X.shape[0]
    old = torch.cat([X[keep][:300] + 1e-4, torch.tensor([[0.0, -3.0, 0.0]] * 50)])   # 300 near the ground, 50 elsewhere
    covered = rc.covered_by(old, X[keep], lvl[keep], base, ring=0)
    assert covered[:300].float().mean() > 0.95 and not covered[300:].any()
    keep_old, pick, counts = rc.plan_selection(old.shape[0], covered, keep.numel(), old.shape[0], g)
    assert counts['n_after'] == old.shape[0] == int(keep_old.sum()) + pick.numel()
    d = rc.knn_rms_reference(torch.cat([old[keep_old], X[keep][pick]]))[int(keep_old.sum()):]
    assert torch.isfinite(d).all() and (d > 0).all()
