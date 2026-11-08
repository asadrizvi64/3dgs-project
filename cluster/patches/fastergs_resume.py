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
FGS_EVAL_EVERY=k    at every step i with i % k == 0 and i >= FGS_EVAL_FROM (default k), render the test set with
                    metrics into test_<i>/: the state after exactly i steps. It is its own training callback with the
                    highest priority, so it runs before that step's SH-degree, densification, ordering and
                    opacity-reset callbacks, and outside the clock hook's timer (not counted as training time).
"""
import sys
from pathlib import Path

MARK = '# >>> fgs-resume v1'
path = Path(sys.argv[1])
src = path.read_text()
if MARK in src:
    print('already patched:', path)
    sys.exit(0)
assert '# >>> fgs-sparse-from v5' in src and '        # <<< fgs-segclock\n' in src, 'apply fastergs_sparse_from.py v5 first'

# 2. save hook: after the clock hook has booked the step
A_SAVE = '        # <<< fgs-segclock\n'
SAVE = A_SAVE + '''        # >>> fgs-resume v1 (save)
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
                'fgs_cum': getattr(self, '_fgs_cum', 0.0),
            }, self.checkpoint_directory / f'state_{iteration + 1:07d}.pt')
            print(f'FGS_SAVE_STATE step={iteration + 1} N={_g._means.shape[0]} cum_s={getattr(self, "_fgs_cum", 0.0):.3f} '
                  f'path={self.checkpoint_directory / f"state_{iteration + 1:07d}.pt"}', file=getattr(self, '_fgs_log', None) or __import__('sys').stdout, flush=True)
        # <<< fgs-resume (save)
'''
assert src.count(A_SAVE) == 1
src = src.replace(A_SAVE, SAVE, 1)

# 3. run() override: pre-training callbacks with the current config, then the saved state, then the normal loop
A_RUN = '    @training_callback(priority=80)\n    def training_iteration('
RUN = '''    # >>> fgs-resume v1 (run)
    def run(self, dataset: 'BaseDataset') -> None:
        _os = __import__('os')
        _p = _os.environ.get('FGS_RESUME')
        if not _p:
            return super().run(dataset)
        for callback in self._gather_callbacks(-1):
            with callback.timer:
                callback(self, 0, dataset)
        st = torch.load(_p, map_location='cuda', weights_only=False)
        g = self.model.gaussians
        with torch.no_grad():
            for k, v in st['params'].items():
                setattr(g, k, torch.nn.Parameter(v.contiguous().requires_grad_(True)))
        g.training_setup(self, st['extent'])
        g.optimizer.load_state_dict(st['optimizer'])
        g.active_sh_degree = st['active_sh_degree']
        g.active_sh_bases = (g.active_sh_degree + 1) ** 2
        g._densification_info = None if (self.USE_MCMC or st['densification_info'] is None) else st['densification_info'].contiguous()
        self.train_sampler.id_sampler = st['id_sampler']
        torch.set_rng_state(st['rng_cpu'])
        torch.cuda.set_rng_state(st['rng_cuda'])
        self.model.num_iterations_trained = st['iteration']
        self._fgs_log = open(_os.environ['FGS_LOG'], 'a', buffering=1) if _os.environ.get('FGS_LOG') else __import__('sys').stdout
        self._fgs_evs, self._fgs_cum = [], float(st['fgs_cum'])
        print(f'FGS_RESUME step={st["iteration"]} N={g._means.shape[0]} cum_s={self._fgs_cum:.3f} sh_degree={g.active_sh_degree} '
              f'densification_info={"none" if g._densification_info is None else tuple(g._densification_info.shape)} '
              f'use_mcmc={self.USE_MCMC} max_primitives={self.MAX_PRIMITIVES} densification_end={self.DENSIFICATION_END_ITERATION} from={_p}',
              file=self._fgs_log, flush=True)
        return super().run(dataset)

    @training_callback(priority=10_000)
    @torch.no_grad()
    def fgs_eval(self, iteration: int, dataset: 'BaseDataset') -> None:
        _os = __import__('os')
        _ee = int(_os.environ.get('FGS_EVAL_EVERY', '0'))
        if _ee and iteration % _ee == 0 and iteration >= int(_os.environ.get('FGS_EVAL_FROM', str(_ee))):
            self._render_dataset(dataset=dataset, verbose=False)
    # <<< fgs-resume (run)

''' + A_RUN
assert src.count(A_RUN) == 1, 'run anchor not found exactly once'
src = src.replace(A_RUN, RUN, 1)
path.write_text(src)
print('patched:', path)
