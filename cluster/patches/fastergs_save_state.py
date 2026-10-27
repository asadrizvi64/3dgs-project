"""Adds an env-gated hook to Faster-GS's training_iteration that saves parameters + Adam state (idempotent).

    python cluster/patches/fastergs_save_state.py <nerficg>/src/Methods/FasterGS/Trainer.py

FGS_SAVE_STATE=n   after the optimizer step that completes step n, write checkpoints/state_<n>.pt with the six
                   parameter tensors, the optimizer state_dict and the camera extent. NeRFICG's own
                   BACKUP.TRAINING_STATE cannot be used: it pickles the whole trainer, which holds lambdas.
Off by default, so normal runs are unchanged.
"""
import sys
from pathlib import Path

MARK = '# >>> fgs-state'
path = Path(sys.argv[1])
src = path.read_text()
if MARK in src:
    print('already patched:', path)
    sys.exit(0)

ANCHOR = '''        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)
        if self.model.ppisp is not None:
            self.model.ppisp.step()
'''
HOOK = ANCHOR + f'''        {MARK}
        if __import__('os').environ.get('FGS_SAVE_STATE') and iteration + 1 == int(__import__('os').environ['FGS_SAVE_STATE']):
            _g = self.model.gaussians
            torch.save({{
                'iteration': iteration + 1,
                'params': {{k: getattr(_g, k).detach().clone() for k in
                           ('_means', '_sh_coefficients_0', '_sh_coefficients_rest', '_opacities', '_scales', '_rotations')}},
                'optimizer': _g.optimizer.state_dict(),
                'extent': _g.training_cameras_extent,
                'active_sh_degree': getattr(_g, 'active_sh_degree', None),
            }}, self.checkpoint_directory / f'state_{{iteration + 1:07d}}.pt')
            print(f'FGS_SAVE_STATE wrote state at step {{iteration + 1}}', flush=True)
        # <<< fgs-state
'''
assert src.count(ANCHOR) == 1, 'anchor not found exactly once'
path.write_text(src.replace(ANCHOR, HOOK, 1))
print('patched:', path)
