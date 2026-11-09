"""Branching a Faster-GS run from a mid-training state, plus timed-out test evaluations (phase 3, Experiment 1).

    python cluster/patches/fastergs_resume.py <nerficg>/src/Methods/<Variant>/Trainer.py

Needs the clock hook (cluster/patches/fastergs_sparse_from.py v5) applied first. Idempotent. All env-gated: with none
of the variables below set, training is unchanged.

FGS_SAVE_STATE=n    after the step that completes step n (and after the clock hook has booked it), write
                    checkpoints/state_<n>.pt: the six parameter tensors, the optimizer state_dict, the densification
                    statistics, the active SH degree, the camera extent, the training-view sampler, the CPU and CUDA
                    random states and the clock's cumulative training time. NeRFICG's own BACKUP.TRAINING_STATE cannot
                    be used: it pickles the whole trainer, which holds lambdas.
FGS_RESUME=path     run the normal pre-training callbacks (dataset, sampler, model and optimizer setup, loss) with the
                    CURRENT config, then load the state file over them and continue the normal training loop from
                    its step. Config changes (densification end, USE_MCMC, MAX_PRIMITIVES, loss weights, learning
                    rates) therefore take effect from the resumed step on. With USE_MCMC the densification statistics
                    are dropped (MCMC does not use them). SEGCLOCK cum_s continues from the saved value, so training
                    time is counted from step 0.
                    v2: the saved optimizer state is loaded, then every group's learning rate is set back to the
                    current config's (load_state_dict would otherwise keep the saved ones, e.g. the opacity rate of
                    the control config in an MCMC arm). Python and NumPy random states are saved and restored too
                    (training itself only draws from torch). The FGS_RESUME line lists the groups' learning rates.
FGS_VIEW_TRACE=path append '<iteration> <camera_index> <frame_idx> <global_frame_idx>' for every step, each field as
                    the view holds it (None if absent), to check that a resumed run sees the same view sequence.
                    v3: FGS_SAVE_STATE and FGS_RESUME print SHA-256 digests (first 16 hex digits) of the full bytes of
                    the parameters, the optimizer state and the densification statistics, so a save/load round trip
                    can be checked exactly.
FGS_EVAL_EVERY=k    at every step i with i % k == 0 and i >= FGS_EVAL_FROM (default k), render the test set with
                    metrics into test_<i>/: the state after exactly i steps. It is its own training callback with the
                    highest priority, so it runs before that step's SH-degree, densification, ordering and
                    opacity-reset callbacks, and outside the clock hook's timer (not counted as training time).
                    v7 (2026-11-09): evaluation saves and restores the random states, the training flag and the
                    dataset mode, so it no longer changes training. fgs_eval documents FGS_EVAL_RNG_GUARD,
                    FGS_EVAL_CHECK, FGS_EVAL_PROF and FGS_EVAL_CUDNN_BENCH.
"""
import sys
from pathlib import Path

import re

MARK = '# >>> fgs-resume v7'
path = Path(sys.argv[1])
src = path.read_text()
if MARK in src:
    print('already patched:', path)
    sys.exit(0)
# upgrade: remove an older version's blocks
src = re.sub(r'    # >>> fgs-resume v\d+ \(run\).*?    # <<< fgs-resume \(run\)\n\n', '', src, flags=re.S)
src = re.sub(r'        # >>> fgs-resume v\d+ \(save\).*?        # <<< fgs-resume \(save\)\n', '', src, flags=re.S)
assert 'fgs-resume' not in src, 'could not remove the older fgs-resume blocks'
assert '# >>> fgs-sparse-from v5' in src and '        # <<< fgs-segclock\n' in src, 'apply fastergs_sparse_from.py v5 first'

# 2. save hook: after the clock hook has booked the step
A_SAVE = '        # <<< fgs-segclock\n'
SAVE = A_SAVE + '''        # >>> fgs-resume v7 (save)
        if __import__('os').environ.get('FGS_VIEW_TRACE'):
            if not hasattr(self, '_fgs_vt'):
                self._fgs_vt = open(__import__('os').environ['FGS_VIEW_TRACE'], 'a', buffering=1)
            print(iteration, getattr(view, 'camera_index', None), getattr(view, 'frame_idx', None), getattr(view, 'global_frame_idx', None), file=self._fgs_vt)
        if __import__('os').environ.get('FGS_SAVE_STATE') and iteration + 1 == int(__import__('os').environ['FGS_SAVE_STATE']):
            _g = self.model.gaussians
            torch.cuda.synchronize()
            torch.save({
                'iteration': iteration + 1,
                'params': {k: getattr(_g, k).detach().clone() for k in
                           ('_means', '_sh_coefficients_0', '_sh_coefficients_rest', '_opacities', '_scales', '_rotations')},
                'optimizer': _g.optimizer.state_dict(),
                'densification_info': None if _g._densification_info is None else _g._densification_info.clone(),
                'active_sh_degree': _g.active_sh_degree,
                'extent': _g.training_cameras_extent,
                'id_sampler': self.train_sampler.id_sampler,
                'rng_cpu': torch.get_rng_state(),
                'rng_cuda': torch.cuda.get_rng_state(),
                'rng_python': __import__('random').getstate(),
                'rng_numpy': __import__('numpy').random.get_state(),
                'fgs_cum': getattr(self, '_fgs_cum', 0.0),
            }, self.checkpoint_directory / f'state_{iteration + 1:07d}.pt')
            _hp, _ho, _hd = self._fgs_digest(_g)
            print(f'FGS_SAVE_STATE step={iteration + 1} N={_g._means.shape[0]} cum_s={getattr(self, "_fgs_cum", 0.0):.3f} '
                  f'param_sha={_hp} optim_sha={_ho} dinfo_sha={_hd} '
                  f'path={self.checkpoint_directory / f"state_{iteration + 1:07d}.pt"}', file=getattr(self, '_fgs_log', None) or __import__('sys').stdout, flush=True)
        # <<< fgs-resume (save)
'''
assert src.count(A_SAVE) == 1
src = src.replace(A_SAVE, SAVE, 1)

# 3. run() override: pre-training callbacks with the current config, then the saved state, then the normal loop
A_RUN = '    @training_callback(priority=80)\n    def training_iteration('
RUN = '''    # >>> fgs-resume v7 (run)
    def run(self, dataset: 'BaseDataset') -> None:
        _os = __import__('os')
        _p = _os.environ.get('FGS_RESUME')
        if not _p:
            return super().run(dataset)
        for callback in self._gather_callbacks(-1):
            with callback.timer:
                callback(self, 0, dataset)
        st = torch.load(_p, map_location='cpu', weights_only=False)   # RNG states must stay CPU ByteTensors
        g = self.model.gaussians
        with torch.no_grad():
            for k, v in st['params'].items():
                setattr(g, k, torch.nn.Parameter(v.cuda().contiguous().requires_grad_(True)))
        g.training_setup(self, st['extent'])
        _lr = {pg['name']: pg['lr'] for pg in g.optimizer.param_groups}   # from the current config
        g.optimizer.load_state_dict(st['optimizer'])   # moves moments to the parameters' device; also restores the saved groups' learning rates ...
        for pg in g.optimizer.param_groups:              # ... so put the current config's back (means: set every step anyway)
            pg['lr'] = _lr[pg['name']]
        _saved_steps = [s.get('step') for s in st['optimizer']['state'].values()]
        for _s in g.optimizer.state.values():            # torch.optim.Adam's loader turns an int 'step' into a float tensor;
            if torch.is_tensor(_s.get('step')) and all(isinstance(x, int) for x in _saved_steps):   # FusedAdam keeps an int
                _s['step'] = int(_s['step'].item())
        g.active_sh_degree = st['active_sh_degree']
        g.active_sh_bases = (g.active_sh_degree + 1) ** 2
        g._densification_info = None if (self.USE_MCMC or st['densification_info'] is None) else st['densification_info'].cuda().contiguous()
        _fresh, _ids = self.train_sampler.id_sampler, st['id_sampler']
        for _k, _v in vars(_ids).items():   # tensors back to the device the freshly built sampler uses (cuda by default)
            if torch.is_tensor(_v):
                _ref = getattr(_fresh, _k, None)
                setattr(_ids, _k, _v.to(_ref.device if torch.is_tensor(_ref) else torch.empty(0).device))
        self.train_sampler.id_sampler = _ids
        torch.set_rng_state(st['rng_cpu'])
        torch.cuda.set_rng_state(st['rng_cuda'])
        if 'rng_python' in st:
            __import__('random').setstate(st['rng_python'])
            __import__('numpy').random.set_state(st['rng_numpy'])
        self.model.num_iterations_trained = st['iteration']
        self._fgs_log = open(_os.environ['FGS_LOG'], 'a', buffering=1) if _os.environ.get('FGS_LOG') else __import__('sys').stdout
        self._fgs_evs, self._fgs_cum = [], float(st['fgs_cum'])
        _hp, _ho, _hd = self._fgs_digest(g)
        print(f'FGS_RESUME step={st["iteration"]} N={g._means.shape[0]} cum_s={self._fgs_cum:.3f} sh_degree={g.active_sh_degree} '
              f'param_sha={_hp} optim_sha={_ho} dinfo_sha={_hd} '
              f'densification_info={"none" if g._densification_info is None else tuple(g._densification_info.shape)} '
              f'use_mcmc={self.USE_MCMC} max_primitives={self.MAX_PRIMITIVES} densification_end={self.DENSIFICATION_END_ITERATION} '
              f'lr=' + ','.join(f'{pg["name"]}:{pg["lr"]:.6g}' for pg in g.optimizer.param_groups) + f' '
              f'lambda_opacity={self.LOSS.LAMBDA_OPACITY_REGULARIZATION} lambda_scale={self.LOSS.LAMBDA_SCALE_REGULARIZATION} from={_p}',
              file=self._fgs_log, flush=True)
        return super().run(dataset)

    @staticmethod
    @torch.no_grad()
    def _fgs_digest(g) -> tuple:
        """SHA-256 over the full bytes of the parameters, of the optimizer state (group order, then each parameter's
        state keys in sorted order) and of the densification statistics ('none' if absent)."""
        _hl = __import__('hashlib')
        raw = lambda t: t.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes()
        hp = _hl.sha256()
        for k in ('_means', '_sh_coefficients_0', '_sh_coefficients_rest', '_opacities', '_scales', '_rotations'):
            t = getattr(g, k)
            hp.update(f'{k}{tuple(t.shape)}{t.dtype}'.encode())
            hp.update(raw(t))
        ho = _hl.sha256()
        for grp in g.optimizer.param_groups:
            for prm in grp['params']:
                s = g.optimizer.state.get(prm, {})
                for key in sorted(s):
                    v = s[key]
                    ho.update(f'{grp["name"]}.{key}'.encode())
                    ho.update(raw(v) if torch.is_tensor(v) else repr(v).encode())
        d = g._densification_info
        return hp.hexdigest()[:16], ho.hexdigest()[:16], 'none' if d is None else _hl.sha256(raw(d)).hexdigest()[:16]

    @staticmethod
    def _fgs_rng_digest() -> str:
        """Short SHA-256 digests of the torch CPU, torch CUDA (all devices), Python and NumPy random states."""
        _hl, _np, _rd = __import__('hashlib'), __import__('numpy'), __import__('random')
        d = lambda b: _hl.sha256(b).hexdigest()[:12]
        ns = _np.random.get_state()
        return (f'cpu:{d(torch.get_rng_state().numpy().tobytes())},'
                f'cuda:{d(b"".join(s.numpy().tobytes() for s in torch.cuda.get_rng_state_all()))},'
                f'py:{d(repr(_rd.getstate()).encode())},'
                f'np:{d(repr((ns[0], ns[2], ns[3], ns[4])).encode() + ns[1].tobytes())}')

    def _fgs_prof_wrap(self, rec: dict):
        """FGS_EVAL_PROF: wrap the evaluation's stages with synchronised timers and peak-memory readings. Returns an
        undo function. Nested stages: before a stage resets the peak, the peak so far is credited to every enclosing
        stage, and a stage's peak is credited to its enclosing stages when it ends."""
        _sys, _tm = __import__('sys'), __import__('time')
        import torchmetrics
        stack, undo = [], []
        def wrap(owner, name, label):
            orig = getattr(owner, name)
            def w(*a, **k):
                torch.cuda.synchronize()
                p0 = torch.cuda.max_memory_allocated()
                for f in stack:
                    f[1] = max(f[1], p0)
                torch.cuda.reset_peak_memory_stats()
                fr = [label, 0]
                stack.append(fr)
                t0 = _tm.perf_counter()
                try:
                    return orig(*a, **k)
                finally:
                    torch.cuda.synchronize()
                    stack.pop()
                    pk = max(fr[1], torch.cuda.max_memory_allocated())
                    for f in stack:
                        f[1] = max(f[1], pk)
                    r = rec.setdefault(label, [0, 0.0, 0])
                    r[0] += 1
                    r[1] += _tm.perf_counter() - t0
                    r[2] = max(r[2], pk)
            setattr(owner, name, w)
            undo.append((owner, name, orig))
        rmod = _sys.modules[type(self.renderer).compute_image_metrics.__module__]
        wrap(self.renderer, 'render_image', 'render_image')
        wrap(self.renderer, 'compute_image_metrics', 'compute_image_metrics')
        for fn in ('load_images', 'save_image'):
            if hasattr(rmod, fn):
                wrap(rmod, fn, fn)
        for cls, lab in ((torchmetrics.image.PeakSignalNoiseRatio, 'psnr'),
                         (torchmetrics.image.StructuralSimilarityIndexMeasure, 'ssim'),
                         (torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity, 'lpips')):
            wrap(cls, '__init__', f'{lab}_construct')
            wrap(cls, 'forward', f'{lab}_forward')
        def restore():
            for owner, name, orig in reversed(undo):
                if owner is self.renderer:
                    delattr(owner, name)   # instance attribute: the class method shows through again
                else:
                    setattr(owner, name, orig)
        return restore

    @training_callback(priority=10_000)
    @torch.no_grad()
    def fgs_eval(self, iteration: int, dataset: 'BaseDataset') -> None:
        """v7: evaluation is observational. The four random states (torch CPU, torch CUDA, Python, NumPy), the model's
        training flag and the dataset's mode are saved before rendering and restored afterwards, also on an exception.
        NeRFICG builds a new LPIPS-VGG metric at every evaluation, which draws from the CPU generator that the view
        sampler reshuffles with (adjudication job 4555954).
        FGS_EVAL_RNG_GUARD=0    restore nothing (the v6 behaviour; for the metric-path identity test only)
        FGS_EVAL_CHECK=1        print FGS_EVALCHECK: random-state digests before, after rendering and after the restore;
                                parameter / optimizer / densification digests before and after; training flag and mode
        FGS_EVAL_PROF=1         print FGS_EVALPROF: time, calls and peak allocated memory per stage, and in total
        FGS_EVAL_CUDNN_BENCH=0  cudnn.benchmark off during evaluation (restored afterwards)"""
        _os, _rd, _np, _tm = __import__('os'), __import__('random'), __import__('numpy'), __import__('time')
        _ee = int(_os.environ.get('FGS_EVAL_EVERY', '0'))
        if not (_ee and iteration % _ee == 0 and iteration >= int(_os.environ.get('FGS_EVAL_FROM', str(_ee)))):
            return
        guard = _os.environ.get('FGS_EVAL_RNG_GUARD', '1') != '0'
        check = _os.environ.get('FGS_EVAL_CHECK') == '1'
        prof = _os.environ.get('FGS_EVAL_PROF') == '1'
        nobench = _os.environ.get('FGS_EVAL_CUDNN_BENCH') == '0'
        out = getattr(self, '_fgs_log', None) or __import__('sys').stdout
        g = self.model.gaussians
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), _rd.getstate(), _np.random.get_state())
        was_training, mode, bench = self.model.training, dataset.mode, torch.backends.cudnn.benchmark
        if check:
            r0, s0 = self._fgs_rng_digest(), self._fgs_digest(g)
        rec, unwrap = {}, None
        if prof:
            torch.cuda.synchronize()
            a0 = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            t0 = _tm.perf_counter()
            unwrap = self._fgs_prof_wrap(rec)
        if nobench:
            torch.backends.cudnn.benchmark = False
        try:
            self._render_dataset(dataset=dataset, verbose=False)
        finally:
            torch.backends.cudnn.benchmark = bench
            if unwrap:
                unwrap()
            if prof:
                torch.cuda.synchronize()
                tot = _tm.perf_counter() - t0
                pk = max([torch.cuda.max_memory_allocated()] + [v[2] for v in rec.values()])
                for k, v in sorted(rec.items()):
                    print(f'FGS_EVALPROF step={iteration} stage={k} calls={v[0]} time_s={v[1]:.3f} peak_gib={v[2] / 2**30:.2f}',
                          file=out, flush=True)
                print(f'FGS_EVALPROF step={iteration} stage=total time_s={tot:.3f} alloc_before_gib={a0 / 2**30:.2f} '
                      f'peak_gib={pk / 2**30:.2f} alloc_after_gib={torch.cuda.memory_allocated() / 2**30:.2f} '
                      f'reserved_after_gib={torch.cuda.memory_reserved() / 2**30:.2f} cudnn_benchmark={int(bench and not nobench)}',
                      file=out, flush=True)
            r1 = self._fgs_rng_digest() if check else None
            t1, m1 = self.model.training, dataset.mode
            if guard:
                torch.set_rng_state(rng[0])
                torch.cuda.set_rng_state_all(rng[1])
                _rd.setstate(rng[2])
                _np.random.set_state(rng[3])
                self.model.train(was_training)
                dataset.set_mode(mode)
            if check:
                s1 = self._fgs_digest(g)
                print(f'FGS_EVALCHECK step={iteration} guard={int(guard)} rng_before={r0} rng_after_render={r1} '
                      f'rng_after_restore={self._fgs_rng_digest()} state_before={",".join(s0)} state_after={",".join(s1)} '
                      f'training_before={was_training} training_after_render={t1} training_after_restore={self.model.training} '
                      f'mode_before={mode} mode_after_render={m1} mode_after_restore={dataset.mode}', file=out, flush=True)
    # <<< fgs-resume (run)

''' + A_RUN
assert src.count(A_RUN) == 1, 'run anchor not found exactly once'
src = src.replace(A_RUN, RUN, 1)
path.write_text(src)
print('patched:', path)
