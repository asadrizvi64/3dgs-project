"""alt_reinit/reinit_trainer.py: trainer side of the depth reinitialization (alternative track R, alt_reinit/PLAN.md).

alt_reinit/make_reinit_variant.sh copies this file and reinit_core.py into the FasterGSReinit method; the hooks that
alt_reinit/patches/fastergs_reinit.py adds to its Trainer.py call run_reinit, run_audit_final and install_cbclock.
Nothing here runs unless FGS_REINIT (reinitialization) or FGS_CBCLOCK (callback clock) is set.

Log lines (to FGS_LOG, like the clock hooks): REINIT_CONFIG, REINIT (timing), REINIT_COUNTS, REINIT_STATUS,
REINIT_TARGET (audit: feasibility of each target), REINIT_CHECK, REINIT_FAIL, REINIT_AUDIT, REINIT_AGREEMENT, CBCLOCK.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

from . import reinit_core as rc

COLORS = {rc.KEPT: (0, 200, 0), rc.GEOMETRY: (220, 0, 0), rc.MIXTURE: (255, 150, 0), rc.LOW_ALPHA: (0, 90, 255),
          rc.EMPTY: (0, 0, 0)}
DEPTH_EDGES = torch.tensor([0.1 * 2.0 ** i for i in range(14)])  # 0.1 .. 819 scene units, log2-spaced


def _log(trainer, line: str) -> None:
    f = getattr(trainer, '_fgs_rlog', None)
    if f is None:
        path = os.environ.get('FGS_LOG')
        f = open(path, 'a', buffering=1) if path else sys.stdout
        trainer._fgs_rlog = f
    print(line, file=f, flush=True)


def _sync() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------------------------- callback clock
class _CallbackClock:
    """Wraps a callback's NeRFICG timer (CallbackTimer with the timer on, nullcontext with it off): synchronized wall time
    per call, logged as 'CBCLOCK name=.. iteration=.. s=..'. NeRFICG's own totals (timings.txt) are left as they were."""

    def __init__(self, inner, name: str, trainer) -> None:
        self.inner, self.name, self.trainer = inner, name, trainer

    def __enter__(self):
        _sync()
        self.t0 = time.perf_counter()
        return self.inner.__enter__()

    def __exit__(self, *exc):
        result = self.inner.__exit__(*exc)
        _sync()
        it = self.trainer.model.num_iterations_trained
        _log(self.trainer, f'CBCLOCK name={self.name} iteration={it} s={time.perf_counter() - self.t0:.6f}')
        return result

    def __getattr__(self, item):
        if item == 'inner':
            raise AttributeError(item)
        return getattr(self.inner, item)


# callbacks that run between training steps and belong to the method (evaluation and checkpointing do not)
CLOCKED = ('increase_sh_degree', 'densify', 'morton_ordering', 'recompute_3d_filter', 'reset_opacities',
           'reset_opacities_extra', 'hard_pruning', 'reinit_gaussians')


def install_cbclock(trainer) -> None:
    """Called once, at the start of the first training step (NeRFICG assigns the callback timers when training starts,
    so earlier wrapping would be overwritten). Callbacks that already ran in iteration 0 before that step (Morton
    ordering, identical in every arm) are not counted."""
    names = []
    for name in CLOCKED:
        fn = getattr(type(trainer), name, None)
        if fn is None or not hasattr(fn, 'timer') or isinstance(fn.timer, _CallbackClock):
            continue
        fn.timer = _CallbackClock(fn.timer, name, trainer)
        names.append(name)
    _log(trainer, f'CBCLOCK_INSTALLED iteration={trainer.model.num_iterations_trained} callbacks={",".join(names)}')


# ----------------------------------------------------------------------------------------------- rendering
def _camera(view) -> rc.Camera:
    c = view.camera
    return rc.Camera(view.w2c.float(), float(c.focal_x), float(c.focal_y), float(c.center_x), float(c.center_y),
                     int(c.width), int(c.height), float(c.near_plane))


def _render_channels(trainer, gs, view, colors: torch.Tensor, opacity: float | None = None) -> torch.Tensor:
    """Inference render of per-Gaussian colours (one SH band, black background, no clamping) -> (3, H, W)."""
    R = sys.modules[type(trainer.renderer).__module__]
    settings = R.extract_settings(view, 1, torch.zeros(3, device=colors.device), trainer.renderer.PROPER_ANTIALIASING)
    op = gs.raw_opacities
    if opacity is not None:
        op = torch.full_like(op, math.log(opacity / (1.0 - opacity)))
    return R.rasterize(
        means=gs.means, scales=gs.raw_scales, rotations=gs.raw_rotations, opacities=op,
        sh_coefficients_0=rc.sh0_from_colors(colors), sh_coefficients_rest=gs.sh_coefficients_rest,
        rasterizer_settings=settings, to_chw=True, clamp_output=False,
    )


def render_depth(trainer, gs, view):
    """Expected depth, alpha and relative depth std from one render with colours (z, 1, z^2)."""
    return rc.depth_from_render(_render_channels(trainer, gs, view, rc.depth_colors(gs.means, view.w2c.float())))


def render_alpha(trainer, gs, view, opacity: float | None = None) -> torch.Tensor:
    ones = torch.zeros_like(gs.means)
    ones[:, 1] = 1.0
    return _render_channels(trainer, gs, view, ones, opacity)[1]


# ----------------------------------------------------------------------------------------------- candidates
def _neighbours_of(v, nbrs, cams, maps, valid):
    return [(cams[j], maps[j][0], valid[j]) for j in nbrs[v]]


def _candidates(trainer, views, cams, maps, valid, stride, cfg, gen, T):
    """Grid candidates -> neighbours -> geometric filter. Returns the kept candidates and the status counts."""
    V = len(views)
    with T('candidates'):
        grids, eligible = [], []
        for v in range(V):
            D, A, rel = maps[v]
            iy, ix = rc.grid_pixels(cams[v].height, cams[v].width, stride, device=D.device)
            st = rc.pixel_status(A, rel, cfg)[iy, ix]
            grids.append((iy, ix, st))
            eligible.append((st == rc.KEPT).nonzero().squeeze(1))
    with T('neighbours'):
        samples = []
        for v in range(V):
            e = eligible[v]
            if e.numel() > cfg.nbr_samples:
                e = e[torch.randperm(e.numel(), generator=gen, device=e.device)[:cfg.nbr_samples]]
            iy, ix, _ = grids[v]
            D = maps[v][0]
            samples.append(rc.backproject(cams[v], ix[e] + 0.5, iy[e] + 0.5, D[iy[e], ix[e]]))
        nbrs, nbr_info = rc.select_neighbours(cams, samples, cfg.n_nbr, cfg.min_angle)
    status = torch.zeros(5, dtype=torch.long, device=cams[0].w2c.device)
    outcomes = torch.zeros(5, dtype=torch.long, device=status.device)
    Xs, rgbs, fps, scores = [], [], [], []
    with T('filter'):
        for v in range(V):
            iy, ix, st = grids[v]
            status += torch.bincount(st.long(), minlength=5)
            e = eligible[v]
            if e.numel() == 0:
                continue
            D = maps[v][0]
            u, vv, d = ix[e] + 0.5, iy[e] + 0.5, D[iy[e], ix[e]]
            if cfg.min_agree > 0:
                X, codes = rc.geometric_test(cams[v], u, vv, d, _neighbours_of(v, nbrs, cams, maps, valid), cfg)
                agree = (codes == rc.AGREE).sum(dim=1)
                outcomes += torch.bincount(codes.long().reshape(-1), minlength=5)
                keep = agree >= cfg.min_agree
            else:
                X = rc.backproject(cams[v], u, vv, d)
                agree = torch.zeros_like(d, dtype=torch.long)
                keep = torch.ones_like(d, dtype=torch.bool)
            n_rej = int((~keep).sum())
            status[rc.KEPT] -= n_rej
            status[rc.GEOMETRY] += n_rej
            k = keep.nonzero().squeeze(1)
            Xs.append(X[k])
            rgbs.append(views[v].rgb[:3, iy[e][k], ix[e][k]].T.float())
            fps.append(rc.footprint(d[k], cams[v].fx, stride))
            scores.append(agree[k].float())
    cat = lambda xs, w: torch.cat(xs) if xs else torch.zeros((0, w), device=status.device)  # noqa: E731
    return dict(X=cat(Xs, 3), rgb=cat(rgbs, 3), fp=torch.cat(fps) if fps else torch.zeros(0, device=status.device),
                score=torch.cat(scores) if scores else torch.zeros(0, device=status.device),
                status=status.tolist(), outcomes=outcomes.tolist(), nbrs=nbrs, nbr_info=nbr_info, grids=grids)


def _dedupe_and_cover(gs, cand, gen, T, ring: int):
    with T('dedupe'):
        if cand['X'].shape[0] == 0:
            return None
        base = float(cand['fp'].min().clamp_min(1e-9))
        lvl = rc.levels(cand['fp'], base)
        keep = rc.dedupe(cand['X'], lvl, base, cand['score'], gen)
        X, rgb, lvl = cand['X'][keep], cand['rgb'][keep], lvl[keep]
    with T('select'):
        covered = rc.covered_by(gs.means.detach(), X, lvl, base, ring)
    return dict(X=X, rgb=rgb, lvl=lvl, base=base, covered=covered)


# ----------------------------------------------------------------------------------------------- reinitialization
def run_reinit(trainer, iteration: int, dataset) -> None:
    """The reinit callback. apply: replace in the live model and optimizer. audit: the same on a deep copy (timed the
    same way), then untimed diagnostics; the live model, optimizer and the training RNG are left untouched."""
    cfg = trainer._fgs_rcfg
    if not getattr(trainer, '_fgs_rcfg_logged', False):
        trainer._fgs_rcfg_logged = True
        _log(trainer, f'REINIT_CONFIG {cfg.describe()} densification_end={trainer.DENSIFICATION_END_ITERATION}')
    live = trainer.model.gaussians
    dev = live.means.device
    gen = torch.Generator(device=dev).manual_seed(cfg.seed * 1_000_003 + iteration)
    dataset.train()
    views = list(dataset)
    cams = [_camera(v) for v in views]
    n_before = int(live.means.shape[0])
    targets = [rc.resolve_target(t, n_before) for t in cfg.target.split('/')]
    if cfg.mode == 'apply' and len(targets) != 1:
        raise ValueError('FGS_REINIT: apply mode takes one target')
    gs = copy.deepcopy(live) if cfg.mode == 'audit' else live   # untimed
    T = rc.PartTimer(sync=_sync)
    with T('depth'):
        maps, valid = [], []
        for v in views:
            D, A, rel, _ = render_depth(trainer, gs, v)
            maps.append((D, A, rel))
            valid.append(rc.valid_depth(A, rel, cfg))
    stride, retries = cfg.stride, 0
    while True:
        cand = _candidates(trainer, views, cams, maps, valid, stride, cfg, gen, T)
        sel = _dedupe_and_cover(gs, cand, gen, T, cfg.cover_ring)
        n_cand = 0 if sel is None else int(sel['X'].shape[0])
        covered = sel['covered'] if sel is not None else torch.zeros(n_before, dtype=torch.bool, device=dev)
        try:
            with T('select'):
                keep_old, pick, counts = rc.plan_selection(n_before, covered, n_cand, targets[0], gen)
            timed = targets[0]
            break
        except rc.CountError as err:
            short = 'candidates' in str(err)
            if short and stride > 1 and retries < cfg.max_retries:
                _log(trainer, f'REINIT_RETRY iteration={iteration} stride={stride} -> {max(1, stride // 2)}: {err}')
                stride, retries = max(1, stride // 2), retries + 1
                continue
            _log(trainer, f'REINIT_FAIL iteration={iteration} mode={cfg.mode} stride={stride} target={targets[0]} reason={err}')
            if cfg.mode == 'apply':
                raise
            # audit: time the first reachable of the remaining targets instead, so the cost is still measured
            counts = None
            for t in targets[1:]:
                try:
                    with T('select'):
                        keep_old, pick, counts = rc.plan_selection(n_before, covered, n_cand, t, gen)
                    timed = t
                    break
                except rc.CountError:
                    continue
            if counts is None:
                _audit_targets(trainer, iteration, n_before, covered, n_cand, targets, gen)
                _finish_audit(trainer, dataset, iteration, live, None, views, cams, maps, valid, cand, sel, stride,
                              retries, T, None, cfg, targets, n_before)
                return
            break
    n_surv = counts['n_survivors']
    with T('knn'):
        from Optim.knn_utils import compute_root_mean_squared_knn_distances
        Xnew = sel['X'][pick]
        knn = compute_root_mean_squared_knn_distances(torch.cat([gs.means.detach()[keep_old], Xnew]))[n_surv:]
        rows = rc.new_rows(Xnew, sel['rgb'][pick], knn, gs.sh_coefficients_rest.shape[1], cfg.init_opacity)
    snap = rc.snapshot(gs.optimizer, keep_old.nonzero().squeeze(1)) if cfg.check else None   # untimed (between parts)
    with T('rebuild'):
        from Optim.adam_utils import extend_param_groups
        gs.prune(~keep_old)
        groups = extend_param_groups(gs.optimizer, rows)
        gs._means = groups['means']
        gs._sh_coefficients_0 = groups['sh_coefficients_0']
        gs._sh_coefficients_rest = groups['sh_coefficients_rest']
        gs._opacities = groups['opacities']
        gs._scales = groups['scales']
        gs._rotations = groups['rotations']
        gs._densification_info = None
        gs._filter_3d = None
        if iteration < trainer.DENSIFICATION_END_ITERATION and not trainer.USE_MCMC:
            # densification continues after this step (not the R2 design, but allowed): fresh buffers of the new size
            gs.reset_densification_info(trainer.DENSIFICATION_USE_ABS_GRAD)
    with T('morton'):
        from CudaUtils.MortonEncoding import morton_encode
        order = torch.argsort(morton_encode(gs._means.data))
        gs.sort(order)
    n_after = int(gs.means.shape[0])
    parts = ' '.join(f't_{k}={v:.4f}' for k, v in T.parts.items())
    _log(trainer, f'REINIT iteration={iteration} mode={cfg.mode} stride={stride} retries={retries} timed_target={timed} '
                  f't_total={T.total:.4f} {parts}')
    _log(trainer, f'REINIT_COUNTS iteration={iteration} n_before={n_before} n_candidates_grid={sum(cand["status"])} '
                  f'n_filtered={int(cand["X"].shape[0])} n_deduped={n_cand} n_covered={counts["n_covered"]} '
                  f'n_survivors={n_surv} n_added={counts["n_added"]} n_after={n_after} target={timed} '
                  f'max_reachable={counts["max_reachable"]}')
    _log(trainer, f'REINIT_STATUS iteration={iteration} stride={stride} ' +
         ' '.join(f'{n}={c}' for n, c in zip(rc.STATUS_NAMES, cand['status'])) + ' | neighbour outcomes ' +
         ' '.join(f'{n}={c}' for n, c in zip(rc.NEIGHBOUR_NAMES, cand['outcomes'])))
    failures = []
    if n_after != timed:
        failures.append(f'count {n_after} != target {timed}')
    if cfg.check:
        failures += rc.verify(gs.optimizer, snap, order, n_surv, rows)
        del snap
    _log(trainer, f'REINIT_CHECK iteration={iteration} result={"pass" if not failures else "FAIL"}'
                  + ('' if not failures else ' ' + ' | '.join(failures)))
    if failures:
        raise RuntimeError(f'REINIT_CHECK failed at iteration {iteration}: {failures}')
    if cfg.mode == 'audit':
        _audit_targets(trainer, iteration, n_before, sel['covered'], n_cand, targets, gen)
        _finish_audit(trainer, dataset, iteration, live, gs, views, cams, maps, valid, cand, sel, stride, retries, T,
                      counts, cfg, targets, n_before)
        del gs
    if getattr(trainer, 'requires_empty_cache', False):
        torch.cuda.empty_cache()


def _audit_targets(trainer, iteration, n_before, covered, n_cand, targets, gen) -> None:
    for t in targets:
        try:
            _, _, c = rc.plan_selection(n_before, covered, n_cand, t, torch.Generator(device=covered.device).manual_seed(0))
            ok = 'reachable'
        except rc.CountError:
            n_cov = int(covered.sum())
            c = dict(n_survivors=n_before - n_cov, n_added=t - (n_before - n_cov), max_reachable=n_before - n_cov + n_cand)
            ok = 'UNREACHABLE'
        _log(trainer, f'REINIT_TARGET iteration={iteration} target={t} {ok} n_survivors={c["n_survivors"]} '
                      f'n_added={c["n_added"]} max_reachable={c["max_reachable"]}')


# ----------------------------------------------------------------------------------------------- audit diagnostics
def _diag_views(V: int, n: int) -> list[int]:
    return sorted(set(torch.linspace(0, V - 1, min(n, V)).round().long().tolist()))


def _status_map(cams, maps, valid, nbrs, v, cfg, stride=1):
    """Status of every `stride`-th pixel of view v (the timed pass's rule, recomputed untimed)."""
    D, A, rel = maps[v]
    iy, ix = rc.grid_pixels(cams[v].height, cams[v].width, stride, device=D.device)
    st = rc.pixel_status(A, rel, cfg)[iy, ix].clone()
    e = (st == rc.KEPT).nonzero().squeeze(1)
    codes = None
    if e.numel() and cfg.min_agree > 0:
        _, codes = rc.geometric_test(cams[v], ix[e] + 0.5, iy[e] + 0.5, D[iy[e], ix[e]],
                                     _neighbours_of(v, nbrs, cams, maps, valid), cfg)
        st[e[(codes == rc.AGREE).sum(dim=1) < cfg.min_agree]] = rc.GEOMETRY
    return iy, ix, st, e, codes


def _finish_audit(trainer, dataset, iteration, live, gs_new, views, cams, maps, valid, cand, sel, stride, retries, T,
                  counts, cfg, targets, n_before) -> None:
    t0 = time.perf_counter()
    out = Path(trainer.output_directory) / 'reinit_audit' / f'k_{iteration:05d}'
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    V = len(views)
    diag = _diag_views(V, cfg.diag_views)
    mask_v = diag[::max(1, len(diag) // max(cfg.mask_views, 1))][:cfg.mask_views]
    nbrs = cand['nbrs']
    gen = torch.Generator(device=live.means.device).manual_seed(cfg.seed * 7 + iteration)
    stats = dict(iteration=iteration, mode=cfg.mode, config=cfg.describe(), stride=stride, retries=retries,
                 timing=dict(T.parts, total=T.total), n_before=n_before, targets=targets, counts=counts,
                 n_filtered=int(cand['X'].shape[0]), n_deduped=None if sel is None else int(sel['X'].shape[0]),
                 status=dict(zip(rc.STATUS_NAMES, cand['status'])),
                 neighbour_outcomes=dict(zip(rc.NEIGHBOUR_NAMES, cand['outcomes'])),
                 views=V, diag_views=diag, mask_views=mask_v)
    fr = [f for info in cand['nbr_info'] for f in info['frac']]
    ang = [a for info in cand['nbr_info'] for a in info['angle']]
    stats['neighbours'] = dict(views_without=sum(1 for n in nbrs if not n), per_view=[len(n) for n in nbrs],
                               overlap=rc.quantiles(torch.tensor(fr)), angle_deg=rc.quantiles(torch.tensor(ang)))
    # status by depth bin and by image region, every view, the timed pass's grid
    by_depth = torch.zeros(len(DEPTH_EDGES) + 1, 5, dtype=torch.long)
    by_region = torch.zeros(9, 5, dtype=torch.long)
    samples = {s: [] for s in (rc.KEPT, rc.GEOMETRY, rc.MIXTURE, rc.LOW_ALPHA)}
    per_view_quota = max(1, math.ceil(cfg.zncc_samples / V))
    rel_all = []
    for v in range(V):
        iy, ix, st, e, codes = _status_map(cams, maps, valid, nbrs, v, cfg, stride)
        d = maps[v][0][iy, ix]
        by_depth += torch.tensor(rc.count_table(st, rc.depth_bin(d, DEPTH_EDGES.to(d.device)), 5, len(DEPTH_EDGES) + 1))
        by_region += torch.tensor(rc.count_table(st, rc.region_id(iy, ix, cams[v].height, cams[v].width), 5, 9))
        r = maps[v][2][iy, ix]
        rel_all.append(r[(maps[v][1][iy, ix] >= cfg.alpha_min)][:20000])
        for s in samples:
            idx = (st == s).nonzero().squeeze(1)
            if idx.numel() > per_view_quota:
                idx = idx[torch.randperm(idx.numel(), generator=gen, device=idx.device)[:per_view_quota]]
            samples[s].append((v, iy[idx], ix[idx], d[idx]))
    stats['by_depth'] = dict(edges=DEPTH_EDGES.tolist(), counts=by_depth.tolist(), columns=list(rc.STATUS_NAMES))
    stats['by_region'] = dict(counts=by_region.tolist(), columns=list(rc.STATUS_NAMES), layout='3x3 row-major')
    stats['rel_std_where_alpha_ok'] = rc.quantiles(torch.cat(rel_all), qs=(0.5, 0.9, 0.95, 0.99))
    stats['photo'] = _photo_evidence(views, cams, nbrs, samples, cand)
    stats['sfm'] = _sfm_check(dataset, views, cams, maps, valid, nbrs, cfg, diag)
    if gs_new is not None:
        stats['coverage'] = _coverage(trainer, live, gs_new, views, diag)
    np_saved = _save_maps(out, views, cams, maps, valid, nbrs, cfg, diag, mask_v)
    stats['files'] = np_saved
    stats['diagnostics_s'] = time.perf_counter() - t0
    with open(out / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=1)
    _log(trainer, f'REINIT_AUDIT iteration={iteration} dir={out} diagnostics_s={stats["diagnostics_s"]:.1f}')


def _photo_evidence(views, cams, nbrs, samples, cand) -> dict:
    """Training-photo patch agreement (7x7 ZNCC) between a candidate's pixel and its projection into each of the view's
    neighbours where it lands inside the image with textured patches. The same two statistics for every status class,
    so kept and rejected candidates are compared like for like: the mean over those neighbours and the best one.
    Axis-aligned patches assume small baselines (the neighbour angles are in stats['neighbours'])."""
    out = {}
    grays = {}
    for s, items in samples.items():
        means, bests, has_all = [], [], []
        for v, iy, ix, d in items:
            if iy.numel() == 0 or not nbrs[v]:
                continue
            ga = grays.setdefault(v, rc.to_gray(views[v].rgb.float()))
            u, vv = ix + 0.5, iy + 0.5
            X = rc.backproject(cams[v], u, vv, d)
            best = torch.full_like(d, -math.inf)
            acc = torch.zeros_like(d)
            n_ok = torch.zeros_like(d)
            for j in nbrs[v]:
                gb = grays.setdefault(j, rc.to_gray(views[j].rgb.float()))
                pu, pv, z = rc.project(cams[j], X)
                sc, ok = rc.photo_consistency(ga, u, vv, gb, pu, pv)
                ok &= z > cams[j].near
                best = torch.where(ok, torch.maximum(best, sc.nan_to_num(-math.inf)), best)
                acc = acc + torch.where(ok, sc.nan_to_num(0.0), torch.zeros_like(sc))
                n_ok = n_ok + ok.float()
            has = n_ok > 0
            nan = torch.full_like(d, math.nan)
            means.append(torch.where(has, acc / n_ok.clamp_min(1), nan))
            bests.append(torch.where(has, best, nan))
            has_all.append(has)
        if means:
            has = torch.cat(has_all)
            out[rc.STATUS_NAMES[s]] = dict(mean_over_neighbours=rc.photo_class_counts(torch.cat(means), has),
                                           best_neighbour=rc.photo_class_counts(torch.cat(bests), has))
    return out


def _sfm_check(dataset, views, cams, maps, valid, nbrs, cfg, diag) -> dict:
    """COLMAP sparse points, observed in each diagnostic view, against the rendered depth there. The points come from
    matched image features, so they are evidence independent of the model's geometry (sparse, biased to texture).
    Also validates the camera conventions: projecting each point with our cameras must land on its COLMAP observation."""
    try:
        import numpy as np
        import pycolmap
        rec = pycolmap.Reconstruction(Path(dataset.dataset_path) / 'sparse' / '0')
        cam0 = rec.camera(1)
        images = sorted([im for im in rec.images.values() if im.camera.camera_id == cam0.camera_id], key=lambda im: im.name)

        def c2w_colmap(im):
            m = torch.as_tensor(im.cam_from_world().inverse().matrix(), dtype=torch.float64)
            return torch.cat([m, torch.tensor([[0, 0, 0, 1]], dtype=torch.float64)]) if m.shape[0] == 3 else m
        T, dev = rc.world_transform([torch.as_tensor(v.c2w_numpy) for v in views],
                                    [c2w_colmap(images[v.global_frame_idx]) for v in views])
        reproj, rel_by = [], {n: [] for n in rc.STATUS_NAMES}
        n_obs = 0
        for vi in diag:
            view, cam = views[vi], cams[vi]
            im = images[view.global_frame_idx]
            xy, xyz = [], []
            for p in im.points2D:
                if p.has_point3D():
                    xy.append(p.xy)
                    xyz.append(rec.points3D[p.point3D_id].xyz)
            if not xyz:
                continue
            Xc = torch.as_tensor(np.asarray(xyz), dtype=torch.float64)
            Xw = (Xc @ T[:3, :3].T + T[:3, 3]).float().to(cam.w2c.device)
            obs = torch.as_tensor(np.asarray(xy), dtype=torch.float32, device=cam.w2c.device)
            obs = obs * torch.tensor([cam.width / cam0.width, cam.height / cam0.height], device=obs.device)
            pu, pv, z = rc.project(cam, Xw)
            ix, iy, inside = rc.pixel_index(cam, pu, pv, z)
            reproj.append(torch.hypot(pu - obs[:, 0], pv - obs[:, 1])[inside])
            D, A, rel = maps[vi]
            st = rc.pixel_status(A, rel, cfg)[iy, ix].clone()
            e = ((st == rc.KEPT) & inside).nonzero().squeeze(1)
            if e.numel() and cfg.min_agree > 0:
                _, codes = rc.geometric_test(cam, ix[e] + 0.5, iy[e] + 0.5, D[iy[e], ix[e]],
                                             _neighbours_of(vi, nbrs, cams, maps, valid), cfg)
                st[e[(codes == rc.AGREE).sum(dim=1) < cfg.min_agree]] = rc.GEOMETRY
            relerr = (D[iy, ix] - z).abs() / z
            for s, name in enumerate(rc.STATUS_NAMES):
                m = inside & (st == s) & (D[iy, ix] > 0)
                rel_by[name].append(relerr[m])
            n_obs += int(inside.sum())
        out = dict(transform_max_deviation=dev, n_observations=n_obs,
                   reprojection_px=rc.quantiles(torch.cat(reproj)) if reproj else dict(n=0))
        for name, xs in rel_by.items():
            x = torch.cat(xs) if xs else torch.zeros(0)
            q = rc.quantiles(x)
            if x.numel():
                q['within_1pct'] = float((x <= 0.01).float().mean())
                q['within_5pct'] = float((x <= 0.05).float().mean())
            out[f'depth_rel_error_{name}'] = q
        return out
    except Exception as err:  # diagnostics must not kill the audit run; the analyzer reports the error
        return dict(error=f'{type(err).__name__}: {err}')


def _coverage(trainer, live, gs_new, views, diag) -> dict:
    """Silhouette coverage before and after the replacement, with every opacity forced to 0.99 (the new rows start at
    the stock 0.1, so the real alpha would say nothing about where they are)."""
    holes, added, before = 0, 0, 0
    for v in diag:
        a0 = render_alpha(trainer, live, views[v], opacity=0.99) >= 0.5
        a1 = render_alpha(trainer, gs_new, views[v], opacity=0.99) >= 0.5
        holes += int((a0 & ~a1).sum())
        added += int((~a0 & a1).sum())
        before += int(a0.sum())
    return dict(pixels_covered_before=before, holes=holes, added=added,
                hole_fraction=holes / max(before, 1), added_fraction=added / max(before, 1))


def _save_maps(out: Path, views, cams, maps, valid, nbrs, cfg, diag, mask_v) -> list[str]:
    """Full-resolution status maps of the diagnostic views (npz, for the final-model agreement) and images of the
    mask views (photo | status | relative depth std)."""
    import numpy as np
    files = []
    arrays = {}
    for v in diag:
        H, W = cams[v].height, cams[v].width
        iy, ix, st, _, _ = _status_map(cams, maps, valid, nbrs, v, cfg, stride=1)
        smap = torch.empty(H * W, dtype=torch.int8, device=st.device)
        smap[iy * W + ix] = st
        smap = smap.reshape(H, W)
        D, A, rel = maps[v]
        arrays[f'D_{v}'] = D.half().cpu().numpy()
        arrays[f'status_{v}'] = smap.cpu().numpy()
        if v in mask_v:
            files.append(_mask_image(out / 'masks' / f'view{v:03d}.jpg', views[v], smap, rel))
    arrays['views'] = np.array(diag)
    np.savez_compressed(out / 'diag_maps.npz', **arrays)
    files.append(str(out / 'diag_maps.npz'))
    return files


def _mask_image(path: Path, view, smap: torch.Tensor, rel: torch.Tensor) -> str:
    import numpy as np
    from PIL import Image
    photo = (view.rgb[:3].float().clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
    col = np.zeros_like(photo)
    s = smap.cpu().numpy()
    for code, rgb in COLORS.items():
        col[s == code] = rgb
    overlay = (0.45 * photo + 0.55 * col).astype(np.uint8)
    r = (rel.clamp(0, 0.2) / 0.2 * 255).nan_to_num(255).byte().cpu().numpy()
    heat = np.stack([r, r, r], axis=-1)
    img = Image.fromarray(np.concatenate([photo, overlay, heat], axis=1))
    img = img.resize((img.width // 2, img.height // 2), Image.BILINEAR)
    img.save(path, quality=85)
    return str(path)


# ----------------------------------------------------------------------------------------------- final agreement
def run_audit_final(trainer, iteration: int, dataset) -> None:
    """After the last training step of an audit run: depth of each audited step against the final model's depth, on
    the diagnostic views, by the audited status. This is agreement with a later reconstruction, not accuracy."""
    import numpy as np
    root = Path(trainer.output_directory) / 'reinit_audit'
    if not root.is_dir():
        return
    cfg = trainer._fgs_rcfg
    dataset.train()
    views = list(dataset)
    final_cache = {}
    for kdir in sorted(root.glob('k_*')):
        f = kdir / 'diag_maps.npz'
        if not f.exists():
            continue
        z = np.load(f)
        res = {}
        for v in z['views'].tolist():
            if v not in final_cache:
                D, A, rel, _ = render_depth(trainer, trainer.model.gaussians, views[v])
                final_cache[v] = (D, A)
            Df, Af = final_cache[v]
            Dk = torch.from_numpy(z[f'D_{v}'].astype('float32')).to(Df.device)
            sk = torch.from_numpy(z[f'status_{v}']).to(Df.device)
            ok = (Af >= cfg.alpha_min) & (Df > 0) & (Dk > 0)
            relerr = (Dk - Df).abs() / Df.clamp_min(1e-12)
            for s, name in enumerate(rc.STATUS_NAMES):
                res.setdefault(name, []).append(relerr[ok & (sk == s)])
        summary = {}
        for name, xs in res.items():
            x = torch.cat(xs)
            q = rc.quantiles(x)
            if x.numel():
                q['within_1pct'] = float((x <= 0.01).float().mean())
                q['within_5pct'] = float((x <= 0.05).float().mean())
            summary[name] = q
        with open(kdir / 'agreement_final.json', 'w') as fo:
            json.dump(dict(final_iteration=iteration + 1, note='agreement with the final model, not accuracy',
                           by_status=summary), fo, indent=1)
        kept = summary.get('kept', {})
        geo = summary.get('geometry', {})
        _log(trainer, f'REINIT_AGREEMENT k={kdir.name[2:]} final={iteration + 1} kept_median={kept.get("q50")} '
                      f'kept_within_1pct={kept.get("within_1pct")} geometry_median={geo.get("q50")} '
                      f'geometry_within_1pct={geo.get("within_1pct")}')
