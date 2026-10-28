"""Adds two env-gated hooks to Faster-GS's training_iteration (idempotent). Needs the lean-Adam patch.

    python cluster/patches/fastergs_sparse_from.py <nerficg>/src/Methods/FasterGS/Trainer.py

FGS_SPARSE_FROM=n  stock (dense) Adam for iterations < n, then switch the FusedAdam to its selective path
                   (update only Gaussians visible in the current view) from iteration n on.
FGS_SEGCLOCK=k     time training iterations in blocks of k with a CUDA sync only at block boundaries, and print
                   'SEGCLOCK step=.. seg_s=.. cum_s=..' after every k steps. Testing and checkpointing happen
                   between blocks, so cum_s is pure training time.
Both are off by default, so normal runs are unchanged.
"""
import sys
from pathlib import Path

MARK = '# >>> fgs-sparse-from'
path = Path(sys.argv[1])
src = path.read_text()
if MARK in src:
    print('already patched:', path)
    sys.exit(0)

START = '        """Performs a training step."""\n'
START_HOOK = START + f'''        {MARK}
        _sf = __import__('os').environ.get('FGS_SPARSE_FROM')
        if _sf and iteration == int(_sf):
            _opt = self.model.gaussians.optimizer
            __import__('sys').modules[type(_opt).__module__]._Rasterize.track_visibility = True
            _opt.sparse = True
            _opt.lean = True
            print(f'FGS_SPARSE_FROM: selective Adam from iteration {{iteration}}', flush=True)
        _segk = int(__import__('os').environ.get('FGS_SEGCLOCK', '0'))
        if _segk and iteration % _segk == 0:
            torch.cuda.synchronize()
            self._fgs_seg_t0 = __import__('time').perf_counter()
        # <<< fgs-sparse-from
'''
END = '''        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)
        if self.model.ppisp is not None:
            self.model.ppisp.step()
'''
END_HOOK = END + '''        # >>> fgs-segclock
        if _segk and (iteration + 1) % _segk == 0:
            torch.cuda.synchronize()
            _dt = __import__('time').perf_counter() - self._fgs_seg_t0
            self._fgs_cum = getattr(self, '_fgs_cum', 0.0) + _dt
            print(f'SEGCLOCK step={iteration + 1} seg_s={_dt:.3f} cum_s={self._fgs_cum:.3f}', flush=True)
        # <<< fgs-segclock
'''
assert src.count(START) == 1, 'start anchor not found exactly once'
assert src.count(END) == 1, 'end anchor not found exactly once'
src = src.replace(START, START_HOOK, 1).replace(END, END_HOOK, 1)
path.write_text(src)
print('patched:', path)
