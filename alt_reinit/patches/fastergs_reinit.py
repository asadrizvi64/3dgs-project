"""Adds the alternative-track-R hooks to a Faster-GS variant's Trainer.py (idempotent; alt_reinit/PLAN.md).

    python alt_reinit/patches/fastergs_reinit.py <nerficg>/src/Methods/FasterGSReinit/Trainer.py

Apply after the rename and after cluster/patches/fastergs_sparse_from.py and fastergs_save_state.py
(alt_reinit/make_reinit_variant.sh does this). Every inserted block sits between '# >>> fgs-reinit <name>' and
'# <<< fgs-reinit <name>' markers; alt_reinit/tests/check_variant.py checks that removing them gives back the
unpatched file exactly. With FGS_REINIT and FGS_CBCLOCK unset the blocks do nothing:

FGS_REINIT=spec   reinitialization (reinit_core.parse_env), e.g. 'mode=apply,iters=2000,target=match' or
                  'mode=audit,iters=2000:5000,target=4166000/match'. Two callbacks are added, both inactive without it:
                  reinit_gaussians (priority 95: after densify and Morton ordering, before opacity reset and the
                  training step) at the configured iterations, and reinit_audit_final (audit only, priority 70: after the
                  last training step). The logic lives in reinit_trainer.py / reinit_core.py next to this Trainer.
FGS_CBCLOCK=1     per-call synchronized wall time of the callbacks that run between training steps (CBCLOCK lines),
                  installed at the first training step.
FGS_LOG=path      where both write (as the clock hooks do).
"""
import sys
from pathlib import Path

MARK = '# >>> fgs-reinit'
path = Path(sys.argv[1])
src = path.read_text()
if MARK in src:
    print('already patched:', path)
    sys.exit(0)

INIT_ANCHOR = '''        self.train_sampler = None
        self.loss = None
'''
INIT_HOOK = INIT_ANCHOR + '''        # >>> fgs-reinit init
        _fgs_spec = __import__('os').environ.get('FGS_REINIT', '')
        self._fgs_rcfg = __import__('importlib').import_module(__package__ + '.reinit_core').parse_env(_fgs_spec) if _fgs_spec else None
        _fgs_on = self._fgs_rcfg is not None and self._fgs_rcfg.mode != 'off'
        self._fgs_reinit_active = _fgs_on
        self._fgs_reinit_start = self._fgs_rcfg.start if _fgs_on else 0
        self._fgs_reinit_end = self._fgs_rcfg.end if _fgs_on else 0
        self._fgs_reinit_every = self._fgs_rcfg.every if _fgs_on else 1
        self._fgs_audit_final_active = _fgs_on and self._fgs_rcfg.mode == 'audit'
        self._fgs_audit_final_iteration = self.NUM_ITERATIONS - 1
        # <<< fgs-reinit init
'''

CB_ANCHOR = '''    @training_callback(priority=80)
    def training_iteration(self, iteration: int, dataset: 'BaseDataset') -> None:
'''
CB_HOOK = '''    # >>> fgs-reinit callbacks
    @training_callback(active='_fgs_reinit_active', priority=95, start_iteration='_fgs_reinit_start', end_iteration='_fgs_reinit_end', iteration_stride='_fgs_reinit_every')
    @torch.no_grad()
    def reinit_gaussians(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Alternative track R: depth reinitialization at this step (apply) or its dry run (audit)."""
        __import__('importlib').import_module(__package__ + '.reinit_trainer').run_reinit(self, iteration, dataset)

    @training_callback(active='_fgs_audit_final_active', priority=70, start_iteration='_fgs_audit_final_iteration', end_iteration='_fgs_audit_final_iteration')
    @torch.no_grad()
    def reinit_audit_final(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Alternative track R: audited depth against the final model's depth (audit runs only)."""
        __import__('importlib').import_module(__package__ + '.reinit_trainer').run_audit_final(self, iteration, dataset)

    # <<< fgs-reinit callbacks
''' + CB_ANCHOR

STEP_ANCHOR = '        """Performs a training step."""\n'
STEP_HOOK = STEP_ANCHOR + '''        # >>> fgs-reinit cbclock
        if not getattr(self, '_fgs_cbclock_done', False):
            self._fgs_cbclock_done = True
            if __import__('os').environ.get('FGS_CBCLOCK'):
                __import__('importlib').import_module(__package__ + '.reinit_trainer').install_cbclock(self)
        # <<< fgs-reinit cbclock
'''

for anchor in (INIT_ANCHOR, CB_ANCHOR, STEP_ANCHOR):
    assert src.count(anchor) == 1, f'anchor not found exactly once: {anchor!r}'
src = src.replace(INIT_ANCHOR, INIT_HOOK, 1).replace(CB_ANCHOR, CB_HOOK, 1).replace(STEP_ANCHOR, STEP_HOOK, 1)
path.write_text(src)
print('patched:', path)
