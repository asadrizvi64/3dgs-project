"""CPU integration test of alt_reinit/reinit_trainer.py with stand-ins for Faster-GS / NeRFICG:

    python -m pytest -q alt_reinit/tests/test_trainer_cpu.py

The stand-ins: a reference software compositor with the rasterizer's colour rule (max(0.5 + C0 sh0, 0), front-to-back
alpha blending, black background, 3x3-pixel splats), a Gaussians class with the same prune/sort/reset_densification_info
interface and a torch Adam optimizer with Faster-GS's six named groups, NeRFICG's Optim.adam_utils semantics, a
brute-force kNN, a Morton-like key and a fake pycolmap reconstruction. They exercise the driver's control flow (apply,
audit, final agreement, count failures and retries, the callback clock, the SfM check) before any GPU time; the real
kernels, kNN and Morton code are covered by the GPU checks (test_gpu.py) and the R0 smoke runs.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent


# ------------------------------------------------------------------------------------------------ package + stubs
@pytest.fixture(scope='module')
def mods(tmp_path_factory):
    pkg = tmp_path_factory.mktemp('pkg') / 'fakemethod'
    pkg.mkdir()
    (pkg / '__init__.py').write_text('')
    for f in ('reinit_core.py', 'reinit_trainer.py'):
        shutil.copy(SRC / f, pkg / f)
    sys.path.insert(0, str(pkg.parent))
    _install_stubs()
    import importlib
    rc = importlib.import_module('fakemethod.reinit_core')
    rt = importlib.import_module('fakemethod.reinit_trainer')
    yield SimpleNamespace(rc=rc, rt=rt)
    sys.path.remove(str(pkg.parent))


def _install_stubs():
    adam = types.ModuleType('Optim.adam_utils')

    def prune_param_groups(optimizer, mask):
        out = {}
        for g in optimizer.param_groups:
            p = g['params'][0]
            st = optimizer.state.pop(p, None)
            q = torch.nn.Parameter(p[mask])
            if st:
                for k in ('exp_avg', 'exp_avg_sq'):
                    st[k] = st[k][mask]
                optimizer.state[q] = st
            g['params'][0] = q
            out[g['name']] = q
        return out

    def extend_param_groups(optimizer, add):
        out = {}
        for g in optimizer.param_groups:
            e = add.get(g['name'])
            if e is None:
                continue
            p = g['params'][0]
            st = optimizer.state.pop(p, None)
            q = torch.nn.Parameter(torch.cat((p, e), dim=0))
            if st:
                for k in ('exp_avg', 'exp_avg_sq'):
                    st[k] = torch.cat((st[k], torch.zeros_like(e)), dim=0)
                optimizer.state[q] = st
            g['params'][0] = q
            out[g['name']] = q
        return out

    def sort_param_groups(optimizer, order):
        out = {}
        for g in optimizer.param_groups:
            p = g['params'][0]
            st = optimizer.state.pop(p, None)
            q = torch.nn.Parameter(p[order])
            if st:
                for k in ('exp_avg', 'exp_avg_sq'):
                    st[k] = st[k][order]
                optimizer.state[q] = st
            g['params'][0] = q
            out[g['name']] = q
        return out
    adam.prune_param_groups, adam.extend_param_groups, adam.sort_param_groups = prune_param_groups, extend_param_groups, sort_param_groups

    knn = types.ModuleType('Optim.knn_utils')

    def compute_root_mean_squared_knn_distances(points, k=3, eps=1e-7):
        d = torch.cdist(points, points)
        d.fill_diagonal_(math.inf)
        return d.topk(k, largest=False).values.square().mean(-1).clamp_min(eps).sqrt()
    knn.compute_root_mean_squared_knn_distances = compute_root_mean_squared_knn_distances

    morton = types.ModuleType('CudaUtils.MortonEncoding')

    def morton_encode(x):
        q = ((x - x.min(0).values) / (x.max(0).values - x.min(0).values).clamp_min(1e-9) * 1023).long()
        key = torch.zeros(x.shape[0], dtype=torch.long)
        for b in range(10):
            for a in range(3):
                key |= ((q[:, a] >> b) & 1) << (3 * b + a)
        return key
    morton.morton_encode = morton_encode

    optim = types.ModuleType('Optim')
    cuda_utils = types.ModuleType('CudaUtils')
    sys.modules.update({'Optim': optim, 'Optim.adam_utils': adam, 'Optim.knn_utils': knn, 'CudaUtils': cuda_utils,
                        'CudaUtils.MortonEncoding': morton})


# ------------------------------------------------------------------------------------------------ fake renderer
C0 = 0.28209479177387814
RENDER = types.ModuleType('fake_renderer_mod')


def extract_settings(view, active_sh_bases, bg_color, proper_antialiasing):
    return dict(w2c=view.w2c, cam=view.camera, bg=bg_color, sh=active_sh_bases)


def rasterize(means, scales, rotations, opacities, sh_coefficients_0, sh_coefficients_rest, rasterizer_settings,
              to_chw, clamp_output=True):
    s = rasterizer_settings
    cam = s['cam']
    assert s['sh'] == 1 and to_chw and not clamp_output
    W, H = cam.width, cam.height
    pc = means @ s['w2c'][:3, :3].T + s['w2c'][:3, 3]
    z = pc[:, 2]
    ok = z > cam.near_plane
    u = cam.focal_x * pc[:, 0] / z + cam.center_x
    v = cam.focal_y * pc[:, 1] / z + cam.center_y
    col = (0.5 + C0 * sh_coefficients_0[:, 0]).clamp_min(0.0)
    alpha = torch.sigmoid(opacities[:, 0]).clamp(max=0.99)
    pid, zs, gid = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            px = torch.floor(u).long() + dx
            py = torch.floor(v).long() + dy
            m = ok & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            pid.append(py[m] * W + px[m])
            zs.append(z[m])
            gid.append(m.nonzero().squeeze(1))
    pid, zs, gid = torch.cat(pid), torch.cat(zs), torch.cat(gid)
    o = torch.argsort(zs, stable=True)
    o = o[torch.argsort(pid[o], stable=True)]
    pid, gid = pid[o], gid[o]
    a = alpha[gid]
    lg = torch.log1p(-a)
    cs = torch.cumsum(lg, 0)
    start = torch.ones_like(pid, dtype=torch.bool)
    start[1:] = pid[1:] != pid[:-1]
    sidx = torch.cummax(torch.where(start, torch.arange(pid.numel()), torch.zeros_like(pid)), 0).values
    excl = cs - lg - (cs[sidx] - lg[sidx])
    w = torch.exp(excl) * a
    img = torch.zeros(3, H * W)
    img.index_add_(1, pid, (w[:, None] * col[gid]).T)
    logT = torch.zeros(H * W).index_add_(0, pid, lg)
    img += torch.exp(logT)[None] * s['bg'][:, None]
    return img.reshape(3, H, W)


RENDER.extract_settings, RENDER.rasterize = extract_settings, rasterize
sys.modules['fake_renderer_mod'] = RENDER


class FakeRenderer:
    PROPER_ANTIALIASING = False


FakeRenderer.__module__ = 'fake_renderer_mod'


# ------------------------------------------------------------------------------------------------ fake scene
GROUPS = ('means', 'sh_coefficients_0', 'sh_coefficients_rest', 'opacities', 'scales', 'rotations')


class FakeGaussians(torch.nn.Module):
    def __init__(self, means, rgb, opacity=0.95):
        super().__init__()
        n = means.shape[0]
        self._means = torch.nn.Parameter(means.clone())
        self._sh_coefficients_0 = torch.nn.Parameter(((rgb - 0.5) / C0)[:, None, :].clone())
        self._sh_coefficients_rest = torch.nn.Parameter(torch.zeros(n, 15, 3))
        self._opacities = torch.nn.Parameter(torch.full((n, 1), math.log(opacity / (1 - opacity))))
        self._scales = torch.nn.Parameter(torch.full((n, 3), math.log(0.02)))
        self._rotations = torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1))
        self._densification_info = None
        self._filter_3d = None
        self.optimizer = torch.optim.Adam([{'params': [getattr(self, '_' + g)], 'name': g, 'lr': 0.0} for g in GROUPS], eps=1e-15)
        g = torch.Generator().manual_seed(0)
        for _ in range(2):   # lr 0: moments without moving anything
            for grp in self.optimizer.param_groups:
                grp['params'][0].grad = torch.randn(grp['params'][0].shape, generator=g)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    means = property(lambda self: self._means)
    raw_scales = property(lambda self: self._scales)
    raw_rotations = property(lambda self: self._rotations)
    raw_opacities = property(lambda self: self._opacities)
    sh_coefficients_0 = property(lambda self: self._sh_coefficients_0)
    sh_coefficients_rest = property(lambda self: self._sh_coefficients_rest)

    def _set(self, groups):
        for g in GROUPS:
            setattr(self, '_' + g, groups[g])

    def prune(self, prune_mask):
        valid = ~prune_mask
        self._set(sys.modules['Optim.adam_utils'].prune_param_groups(self.optimizer, valid))
        if self._densification_info is not None:
            self._densification_info = self._densification_info[:, valid]

    def sort(self, order):
        self._set(sys.modules['Optim.adam_utils'].sort_param_groups(self.optimizer, order))

    def reset_densification_info(self, use_abs_grad):
        self._densification_info = torch.zeros((3 if use_abs_grad else 2, self._means.shape[0]))


def look_at(c, t=(0.0, 0.0, 0.0), up=(0.0, -1.0, 0.0)):
    c, t, up = (torch.tensor(x, dtype=torch.float64) for x in (c, t, up))
    f = (t - c) / (t - c).norm()
    r = torch.linalg.cross(up, f)
    r = r / r.norm()
    d = torch.linalg.cross(f, r)
    w2c = torch.eye(4, dtype=torch.float64)
    w2c[:3, :3] = torch.stack([r, d, f])
    w2c[:3, 3] = -w2c[:3, :3] @ c
    return w2c


def texture(x, z):
    return torch.stack([0.5 + 0.4 * torch.sin(7 * x) * torch.cos(5 * z), 0.5 + 0.3 * torch.cos(11 * x + 3 * z),
                        0.4 + 0.3 * torch.sin(13 * z)], -1).clamp(0, 1)


def make_scene(n_views=8, W=96, H=64, f=80.0, spacing=0.06):
    xs = torch.arange(-2.4, 2.4, spacing)
    gx, gz = torch.meshgrid(xs, xs, indexing='ij')
    means = torch.stack([gx.reshape(-1), torch.ones(gx.numel()), gz.reshape(-1)], -1)
    rgb = texture(means[:, 0], means[:, 2])
    gs = FakeGaussians(means, rgb)
    views = []
    for i in range(n_views):
        a = 2 * math.pi * i / n_views
        w2c = look_at((3.0 * math.cos(a), -1.2, 3.0 * math.sin(a)))
        cam = SimpleNamespace(focal_x=f, focal_y=f, center_x=W / 2, center_y=H / 2, width=W, height=H, near_plane=0.05,
                              far_plane=100.0, distortion=None)
        view = SimpleNamespace(camera=cam, w2c=w2c.float(), c2w_numpy=torch.linalg.inv(w2c).numpy(), global_frame_idx=2 * i)
        view.rgb = rasterize(gs.means.detach(), None, None, gs.raw_opacities.detach(), gs.sh_coefficients_0.detach(),
                             None, dict(w2c=view.w2c, cam=cam, bg=torch.zeros(3), sh=1), True, False).clamp(0, 1)
        views.append(view)
    return gs, views


class FakeDataset:
    def __init__(self, views, path):
        self.views, self.dataset_path = views, path

    def train(self):
        return self

    def __iter__(self):
        return iter(self.views)


def make_trainer(mods, tmp_path, spec, densify_end=2000, n_views=8):
    gs, views = make_scene(n_views=n_views)
    trainer = SimpleNamespace(model=SimpleNamespace(gaussians=gs, num_iterations_trained=2000), renderer=FakeRenderer(),
                              output_directory=tmp_path, DENSIFICATION_END_ITERATION=densify_end, USE_MCMC=False,
                              DENSIFICATION_USE_ABS_GRAD=False, NUM_ITERATIONS=2100, requires_empty_cache=False,
                              _fgs_rcfg=mods.rc.parse_env(spec))
    tmp_path.mkdir(parents=True, exist_ok=True)
    log = tmp_path / 'fgs.log'
    trainer._fgs_rlog = open(log, 'a', buffering=1)
    return trainer, FakeDataset(views, tmp_path / 'nodata'), log


def _state(gs):
    return {g['name']: (g['params'][0].detach().clone(), gs.optimizer.state[g['params'][0]]['exp_avg'].clone())
            for g in gs.optimizer.param_groups}


def run(fn, *args):
    """The trainer's callbacks are decorated with @torch.no_grad()."""
    with torch.no_grad():
        return fn(*args)


BASE = 'iters=2000,stride=1,min_agree=1,n_nbr=2,min_angle=1,diag_views=4,mask_views=2,zncc_samples=400,nbr_samples=256'


# ------------------------------------------------------------------------------------------------ tests
def test_depth_render_of_fake_scene(mods):
    gs, views = make_scene()
    D, A, rel, has = run(mods.rt.render_depth, SimpleNamespace(renderer=FakeRenderer()), gs, views[0])
    ok = A > 0.95
    assert ok.float().mean() > 0.3
    # every confident pixel sees the ground plane y = 1. Each Gaussian contributes its centre depth over its whole
    # footprint (3x3 pixels here), as in the real colour trick, so on a slanted plane D is off by up to about
    # footprint x slope: near the plane, not on it
    cam = mods.rt._camera(views[0])
    iy, ix = ok.nonzero(as_tuple=True)
    X = mods.rc.backproject(cam, ix + 0.5, iy + 0.5, D[iy, ix])
    assert (X[:, 1] - 1.0).abs().median() < 0.1


@pytest.mark.parametrize('target', ['match', '+300'])
def test_apply_reaches_target_and_passes_check(mods, tmp_path, target):
    trainer, ds, log = make_trainer(mods, tmp_path, f'mode=apply,target={target},{BASE}')
    n0 = trainer.model.gaussians.means.shape[0]
    before = trainer.model.gaussians.means.detach().clone()
    run(mods.rt.run_reinit, trainer, 2000, ds)
    text = log.read_text()
    assert 'REINIT_CHECK iteration=2000 result=pass' in text, text
    n1 = trainer.model.gaussians.means.shape[0]
    assert n1 == mods.rc.resolve_target(target, n0)
    assert f'n_after={n1}' in text and 'REINIT iteration=2000 mode=apply' in text
    # rows were replaced, not just kept
    after = trainer.model.gaussians.means.detach()
    kept_old = (after[:, None] == before[None]).all(-1).any(1)
    assert 0 < int(kept_old.sum()) < n1
    assert trainer.model.gaussians._densification_info is None   # densification ended at k


def test_apply_with_continuing_densification_resets_buffers(mods, tmp_path):
    trainer, ds, log = make_trainer(mods, tmp_path, f'mode=apply,target=match,{BASE}', densify_end=14900)
    run(mods.rt.run_reinit, trainer, 2000, ds)
    gs = trainer.model.gaussians
    assert gs._densification_info is not None and gs._densification_info.shape == (2, gs.means.shape[0])


def test_audit_leaves_live_model_untouched_and_writes_diagnostics(mods, tmp_path):
    # 24 views (15 degrees apart): the 7x7 axis-aligned ZNCC assumes small baselines, as between neighbouring
    # Mip-NeRF360 views; at 45 degrees the patches are too distorted to agree even on the right surface
    trainer, ds, log = make_trainer(mods, tmp_path, f'mode=audit,target=+300/match/+1000000,{BASE}', n_views=24)
    gs = trainer.model.gaussians
    before = _state(gs)
    run(mods.rt.run_reinit, trainer, 2000, ds)
    after = _state(gs)
    assert trainer.model.gaussians is gs
    for k in before:
        assert torch.equal(before[k][0], after[k][0]) and torch.equal(before[k][1], after[k][1]), k
    text = log.read_text()
    assert 'REINIT_CHECK iteration=2000 result=pass' in text
    assert text.count('REINIT_TARGET iteration=2000') == 3 and 'UNREACHABLE' in text
    kdir = tmp_path / 'reinit_audit' / 'k_02000'
    stats = json.loads((kdir / 'stats.json').read_text())
    for key in ('timing', 'counts', 'status', 'neighbour_outcomes', 'by_depth', 'by_region', 'photo', 'sfm', 'coverage',
                'neighbours', 'rel_std_where_alpha_ok'):
        assert key in stats, key
    assert stats['counts']['n_after'] == stats['n_before'] + 300
    assert 'error' in stats['sfm']                      # no COLMAP data here: reported, not fatal
    kept = stats['photo']['kept']
    assert kept['mean_over_neighbours']['n'] > 0 and set(kept) == {'mean_over_neighbours', 'best_neighbour'}
    assert kept['mean_over_neighbours']['median'] > 0.5     # same textured plane in every photo
    assert kept['best_neighbour']['median'] >= kept['mean_over_neighbours']['median']
    assert 0.0 <= stats['coverage']['hole_fraction'] <= 1.0
    assert len(list((kdir / 'masks').glob('*.jpg'))) == 2 and (kdir / 'diag_maps.npz').exists()
    # final agreement: the model did not change, so the audited depth agrees with it
    run(mods.rt.run_audit_final, trainer, 2099, ds)
    agr = json.loads((kdir / 'agreement_final.json').read_text())
    assert agr['by_status']['kept']['q50'] < 1e-3 and agr['by_status']['kept']['within_1pct'] > 0.99
    assert 'REINIT_AGREEMENT k=02000' in log.read_text()


def test_unreachable_target(mods, tmp_path):
    trainer, ds, log = make_trainer(mods, tmp_path, f'mode=apply,target=+10000000,{BASE}'.replace('stride=1', 'stride=2'))
    with pytest.raises(mods.rc.CountError):
        run(mods.rt.run_reinit, trainer, 2000, ds)
    text = log.read_text()
    assert 'REINIT_RETRY iteration=2000 stride=2 -> 1' in text and 'REINIT_FAIL iteration=2000 mode=apply' in text
    trainer, ds, log = make_trainer(mods, tmp_path / 'b', f'mode=audit,target=+10000000,{BASE}')
    run(mods.rt.run_reinit, trainer, 2000, ds)            # audit: reported, not fatal
    assert 'REINIT_FAIL' in log.read_text() and (tmp_path / 'b' / 'reinit_audit' / 'k_02000' / 'stats.json').exists()
    # audit with a fallback: the unreachable first target is reported, the next reachable one is timed and checked
    trainer, ds, log = make_trainer(mods, tmp_path / 'c', f'mode=audit,target=+10000000/match,{BASE}')
    n0 = trainer.model.gaussians.means.shape[0]
    run(mods.rt.run_reinit, trainer, 2000, ds)
    text = log.read_text()
    assert 'REINIT_FAIL' in text and f'timed_target={n0}' in text and 'result=pass' in text
    assert text.count('REINIT_TARGET') == 2 and 'UNREACHABLE' in text


def test_unfiltered_mode(mods, tmp_path):
    trainer, ds, log = make_trainer(mods, tmp_path, f'mode=apply,target=match,{BASE}'.replace('min_agree=1', 'min_agree=0'))
    run(mods.rt.run_reinit, trainer, 2000, ds)
    text = log.read_text()
    assert 'result=pass' in text and 'geometry=0' in text


def test_callback_clock(mods, tmp_path):
    class Inner:
        n = 0

        def __enter__(self):
            Inner.n += 1

        def __exit__(self, *a):
            return False

        def get_values(self):
            return (1.0, 1.0, Inner.n)

    class T:
        pass

    def densify(self, it, ds):
        return None
    densify.timer = Inner()
    T.densify = densify
    trainer = T()
    trainer.model = SimpleNamespace(num_iterations_trained=600)
    trainer._fgs_rlog = open(tmp_path / 'cb.log', 'a', buffering=1)
    mods.rt.install_cbclock(trainer)
    mods.rt.install_cbclock(trainer)                  # idempotent
    with type(trainer).densify.timer:
        pass
    text = (tmp_path / 'cb.log').read_text()
    assert text.count('CBCLOCK_INSTALLED') == 2 and 'callbacks=densify' in text.splitlines()[0]
    assert 'CBCLOCK name=densify iteration=600 s=' in text and Inner.n == 1
    assert type(trainer).densify.timer.get_values() == (1.0, 1.0, 1)


def test_sfm_check_with_fake_colmap(mods, tmp_path):
    """COLMAP world = a similarity of the dataset world; observations at 2x the training resolution. The check must
    recover the transform and land every projected point on its observation."""
    gs, views = make_scene()
    rc = mods.rc
    Tds = torch.eye(4, dtype=torch.float64)
    Tds[:3, :3] = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1)))[0]
    Tds[:3, 3] = torch.tensor([0.5, -0.2, 1.0], dtype=torch.float64)       # colmap -> dataset
    Tinv = torch.linalg.inv(Tds)
    pts = gs.means.detach().double()[::7]
    pts_col = pts @ Tinv[:3, :3].T + Tinv[:3, 3]
    cams = [mods.rt._camera(v) for v in views]

    class P2:
        def __init__(self, xy, pid):
            self.xy, self.point3D_id = xy, pid

        def has_point3D(self):
            return self.point3D_id >= 0

    images = {}
    for v, cam in zip(views, cams):
        u, vv, z = rc.project(cam, pts.float())
        ok = (z > 0) & (u >= 0) & (u < cam.width) & (vv >= 0) & (vv < cam.height)
        p2 = [P2(np.array([2 * float(a), 2 * float(b)]), int(i)) for a, b, i in zip(u[ok], vv[ok], ok.nonzero().squeeze(1))]
        p2.append(P2(np.array([1.0, 1.0]), -1))
        c2w_col = Tinv @ torch.as_tensor(v.c2w_numpy)
        pose = SimpleNamespace(inverse=lambda m=c2w_col: SimpleNamespace(matrix=lambda m=m: m[:3].numpy()))
        images[v.global_frame_idx] = SimpleNamespace(name=f'img{v.global_frame_idx:03d}.jpg', camera=SimpleNamespace(camera_id=1),
                                                     cam_from_world=lambda pose=pose: pose, points2D=p2)
    # the loader sorts all images by name; global_frame_idx indexes that list (test views are the odd indices here)
    allimgs = [images.get(i) or SimpleNamespace(name=f'img{i:03d}.jpg', camera=SimpleNamespace(camera_id=1),
                                                cam_from_world=None, points2D=[]) for i in range(2 * len(views))]
    rec = SimpleNamespace(camera=lambda i: SimpleNamespace(camera_id=1, width=2 * views[0].camera.width, height=2 * views[0].camera.height),
                          images={i: im for i, im in enumerate(allimgs)},
                          points3D={i: SimpleNamespace(xyz=pts_col[i].numpy()) for i in range(pts_col.shape[0])})
    sys.modules['pycolmap'] = SimpleNamespace(Reconstruction=lambda path: rec)
    try:
        cfg = rc.parse_env('mode=audit,iters=2000,min_agree=1,n_nbr=2,min_angle=1')
        trainer = SimpleNamespace(renderer=FakeRenderer())
        maps, valid = [], []
        for v in views:
            D, A, rel, _ = run(mods.rt.render_depth, trainer, gs, v)
            maps.append((D, A, rel))
            valid.append(rc.valid_depth(A, rel, cfg))
        nbrs = [[(i + 1) % len(views), (i - 1) % len(views)] for i in range(len(views))]
        out = run(mods.rt._sfm_check, FakeDataset(views, tmp_path), views, cams, maps, valid, nbrs, cfg, [0, 3, 5])
    finally:
        del sys.modules['pycolmap']
    assert 'error' not in out, out
    assert out['transform_max_deviation'] < 1e-9 and out['n_observations'] > 100
    assert out['reprojection_px']['q90'] < 1e-3
    # SfM points lie on the plane; the rendered depth is off by the splats' centre-depth bias (see the first test)
    assert out['depth_rel_error_kept']['q50'] < 0.06
