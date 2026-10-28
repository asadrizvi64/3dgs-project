"""Adds the B switch (config key or env) and a training clock to Faster-GS's training_iteration (idempotent; upgrades older versions). Needs lean Adam.

    python cluster/patches/fastergs_sparse_from.py <nerficg>/src/Methods/FasterGS/Trainer.py

TRAINING.OPTIMIZER.SELECTIVE_FROM: n   (config key; default -1 = off)
                   stock (dense) Adam for iterations < n, then the FusedAdam's selective path (update only Gaussians
                   visible in the current view) from iteration n on.
FGS_SPARSE_FROM=n  environment override of SELECTIVE_FROM (takes precedence when set).
FGS_SEGCLOCK=k     pure training time: CUDA events around every training_iteration (nothing that runs between
                   iterations, such as testing or checkpointing, is counted); every k iterations the summed time is
                   written as 'SEGCLOCK step=.. seg_s=.. cum_s=..'.
                   v4: each SEGCLOCK line also has N (Gaussians now) and peak_alloc_gib (torch peak allocated so far).
                   The line at the last step is written inside the training step, before any final evaluation, so its
                   peak is a training-only peak when no intermediate evaluations run. Also written: 'FGS_N iteration=15000
                   N=..' (count right after densification ends) and, at the last step, 'FGS_TRAIN_END N=.. (before the
                   end-of-training opacity cleanup) peak_alloc_gib=.. peak_reserved_gib=..'.
FGS_WALL=a,b,..    v5: unprofiled elapsed time over whole windows. The device is synchronized only at these iteration
                   boundaries (at the start of iteration a, ..., and after the last training step if NUM_ITERATIONS is
                   listed); 'FGS_WALL from=a to=b s=.. views=..' gives the host wall time between consecutive
                   boundaries, including everything between steps (callbacks, loop overhead), plus a hash of the
                   sequence of training views used so far (global_frame_idx), to check that runs saw the same views.
FGS_LOG=path       file the hooks write to (stdout lines get lost in the progress-bar output); default stdout.
All off by default, so normal runs are unchanged.
"""
import re
import sys
from pathlib import Path

VERSION = '# >>> fgs-sparse-from v5'
path = Path(sys.argv[1])
src = path.read_text()
if VERSION in src:
    print('already patched:', path)
    sys.exit(0)
# remove an older version of these hooks
src = re.sub(r'        # >>> fgs-sparse-from.*?        # <<< fgs-sparse-from\n', '', src, flags=re.S)
src = re.sub(r'        # >>> fgs-segclock.*?        # <<< fgs-segclock\n', '', src, flags=re.S)

START = '        """Performs a training step."""\n'
START_HOOK = START + f'''        {VERSION}
        _os = __import__('os')
        _segk = int(_os.environ.get('FGS_SEGCLOCK', '0'))
        _sf = int(_os.environ.get('FGS_SPARSE_FROM') or getattr(self.OPTIMIZER, 'SELECTIVE_FROM', -1))
        if not hasattr(self, '_fgs_log'):
            self._fgs_log = open(_os.environ['FGS_LOG'], 'a', buffering=1) if _os.environ.get('FGS_LOG') else __import__('sys').stdout
            self._fgs_evs, self._fgs_cum = [], 0.0
        if _sf >= 0 and iteration == _sf:
            _opt = self.model.gaussians.optimizer
            __import__('sys').modules[type(_opt).__module__]._Rasterize.track_visibility = True
            _opt.sparse = True
            _opt.lean = True
            print(f'FGS_SPARSE_FROM: selective Adam from iteration {{iteration}}', file=self._fgs_log, flush=True)
        if _segk and iteration == 15000:
            print(f'FGS_N iteration=15000 N={{self.model.gaussians.means.shape[0]}}', file=self._fgs_log, flush=True)
        _wall = {{int(x) for x in _os.environ['FGS_WALL'].split(',')}} if _os.environ.get('FGS_WALL') else ()
        if not hasattr(self, '_fgs_vh'):
            self._fgs_vh, self._fgs_wall = 0, None
        if iteration in _wall:
            torch.cuda.synchronize()
            _tw = __import__('time').perf_counter()
            if self._fgs_wall is not None:
                print(f'FGS_WALL from={{self._fgs_wall[0]}} to={{iteration}} s={{_tw - self._fgs_wall[1]:.3f}} views={{self._fgs_vh:x}}', file=self._fgs_log, flush=True)
            self._fgs_wall = (iteration, _tw)
        if _segk:
            _fgs_e0 = torch.cuda.Event(enable_timing=True)
            _fgs_e0.record()
        # <<< fgs-sparse-from
'''
END = '''        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)
        if self.model.ppisp is not None:
            self.model.ppisp.step()
'''
END_HOOK = END + '''        # >>> fgs-segclock
        if _wall:
            self._fgs_vh = (self._fgs_vh * 1000003 + int(getattr(view, 'global_frame_idx', None) or getattr(view, 'frame_idx', 0) or 0) + 1) % (1 << 61)
            if iteration + 1 == self.NUM_ITERATIONS and self.NUM_ITERATIONS in _wall and self._fgs_wall is not None:
                torch.cuda.synchronize()
                _tw = __import__('time').perf_counter()
                print(f'FGS_WALL from={self._fgs_wall[0]} to={iteration + 1} s={_tw - self._fgs_wall[1]:.3f} views={self._fgs_vh:x}', file=self._fgs_log, flush=True)
        if _segk:
            _fgs_e1 = torch.cuda.Event(enable_timing=True)
            _fgs_e1.record()
            self._fgs_evs.append((_fgs_e0, _fgs_e1))
            if (iteration + 1) % _segk == 0:
                torch.cuda.synchronize()
                _dt = sum(a.elapsed_time(b) for a, b in self._fgs_evs) / 1000.0
                self._fgs_evs.clear()
                self._fgs_cum += _dt
                _vis = getattr(__import__('sys').modules[type(self.model.gaussians.optimizer).__module__]._Rasterize, 'last_visibility', None)
                _vf = f' visfrac={int(_vis.count_nonzero()) / max(_vis.numel(), 1):.3f}' if _vis is not None and self.model.gaussians.optimizer.sparse else ''
                _n = self.model.gaussians.means.shape[0]
                _pk = torch.cuda.max_memory_allocated() / 1024 ** 3
                print(f'SEGCLOCK step={iteration + 1} seg_s={_dt:.3f} cum_s={self._fgs_cum:.3f} sparse={int(self.model.gaussians.optimizer.sparse)}{_vf} N={_n} peak_alloc_gib={_pk:.2f}',
                      file=self._fgs_log, flush=True)
            if iteration + 1 == self.NUM_ITERATIONS:
                torch.cuda.synchronize()
                print(f'FGS_TRAIN_END N={self.model.gaussians.means.shape[0]} peak_alloc_gib={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} '
                      f'peak_reserved_gib={torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}', file=self._fgs_log, flush=True)
        # <<< fgs-segclock
'''
assert src.count(START) == 1, 'start anchor not found exactly once'
assert src.count(END) == 1, 'end anchor not found exactly once'
src = src.replace(START, START_HOOK, 1).replace(END, END_HOOK, 1)
path.write_text(src)
print('patched:', path)
