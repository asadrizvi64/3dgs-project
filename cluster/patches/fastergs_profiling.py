"""Adds env-gated profiling hooks to Faster-GS's training_iteration (idempotent).

    python cluster/patches/fastergs_profiling.py <nerficg>/src/Methods/FasterGS/Trainer.py

All hooks are off unless FGS_PROF=1, so normal baseline runs are unchanged
(cost when off: one dict lookup per step).

  FGS_PROF=1            STAGE lines (CUDA events: fwd = render + loss, bwd, opt = step + zero_grad
                        + post_optimizer_step) at FGS_STAGE_STEPS (default 500,5000,20000), and a
                        CURVE line every FGS_CURVE_EVERY steps (default 500): mean ms of
                        training_iteration over the window (host-synced at both ends) + Gaussian count.
  FGS_PROF_RANGE=a,b    cudaProfilerStart at step a, Stop at step b (nsys/ncu capture range),
                        with NVTX ranges fwd/bwd/opt inside it.
  FGS_PROF_EXIT=n       exit the process right after step n (for ncu runs that need only the window).
"""
import sys
from pathlib import Path

MARK = '# >>> fgs-prof'
path = Path(sys.argv[1])
src = path.read_text()
if MARK in src:
    print('already patched:', path)
    sys.exit(0)

HELPERS = f'''
{MARK} helpers
import os as _os, time as _time, sys as _sys
_P = _os.environ.get('FGS_PROF') == '1'
_STAGE = {{int(s) for s in _os.environ.get('FGS_STAGE_STEPS', '500,5000,20000').split(',')}}
_EVERY = int(_os.environ.get('FGS_CURVE_EVERY', '500'))
_RANGE = tuple(int(s) for s in _os.environ['FGS_PROF_RANGE'].split(',')) if _os.environ.get('FGS_PROF_RANGE') else None
_EXIT = int(_os.environ['FGS_PROF_EXIT']) if _os.environ.get('FGS_PROF_EXIT') else None
_win = [0.0, 0]
# <<< fgs-prof
'''

OLD_HEAD = '''        """Performs a training step."""
'''
NEW_HEAD = f'''        """Performs a training step."""
        {MARK}
        _st = _P and iteration in _STAGE
        _rng = _RANGE is not None and _RANGE[0] <= iteration < _RANGE[1]
        if _RANGE is not None and iteration == _RANGE[0]:
            torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart()
        if _P:
            torch.cuda.synchronize(); _t0 = _time.perf_counter()
        if _st:
            _ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]; _ev[0].record()
        if _rng: torch.cuda.nvtx.range_push('fwd')
        # <<< fgs-prof
'''

OLD_BWD = '''        # backward
        loss.backward()
        # optimizer step
'''
NEW_BWD = f'''        {MARK}
        if _st: _ev[1].record()
        if _rng: torch.cuda.nvtx.range_pop(); torch.cuda.nvtx.range_push('bwd')
        # <<< fgs-prof
        # backward
        loss.backward()
        {MARK}
        if _st: _ev[2].record()
        if _rng: torch.cuda.nvtx.range_pop(); torch.cuda.nvtx.range_push('opt')
        # <<< fgs-prof
        # optimizer step
'''

OLD_TAIL = '''        if self.model.ppisp is not None:
            self.model.ppisp.step()

    @training_callback(active='SPEEDYSPLAT_PRUNING.USE\''''
NEW_TAIL = f'''        if self.model.ppisp is not None:
            self.model.ppisp.step()
        {MARK}
        if _rng: torch.cuda.nvtx.range_pop()
        if _st:
            _ev[3].record(); torch.cuda.synchronize()
            print(f'STAGE step={{iteration}} fwd={{_ev[0].elapsed_time(_ev[1]):.3f}} bwd={{_ev[1].elapsed_time(_ev[2]):.3f}} '
                  f'opt={{_ev[2].elapsed_time(_ev[3]):.3f}} n_gauss={{self.model.gaussians._means.shape[0]}}', flush=True)
        if _P:
            torch.cuda.synchronize(); _win[0] += _time.perf_counter() - _t0; _win[1] += 1
            if (iteration + 1) % _EVERY == 0:
                print(f'CURVE step={{iteration + 1}} ms_per_step={{1e3 * _win[0] / _win[1]:.3f}} '
                      f'n_gauss={{self.model.gaussians._means.shape[0]}}', flush=True)
                _win[0], _win[1] = 0.0, 0
        if _RANGE is not None and iteration == _RANGE[1] - 1:
            torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
        if _EXIT is not None and iteration >= _EXIT:
            print(f'FGS_PROF_EXIT after step {{iteration}}', flush=True); _sys.stdout.flush(); _os._exit(0)
        # <<< fgs-prof

    @training_callback(active='SPEEDYSPLAT_PRUNING.USE\''''

OLD_IMPORT = 'import torch\n'
for old, new in [(OLD_IMPORT, OLD_IMPORT + HELPERS), (OLD_HEAD, NEW_HEAD), (OLD_BWD, NEW_BWD), (OLD_TAIL, NEW_TAIL)]:
    assert src.count(old) == 1, f'anchor not found exactly once:\n{old}'
    src = src.replace(old, new, 1)
path.write_text(src)
print('patched:', path)
