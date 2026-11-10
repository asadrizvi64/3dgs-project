"""Experiment 1A switches for the MCMC policy's components (phase3/EXP1A_PLAN.md).

    python cluster/patches/fastergs_ablate.py <nerficg>/src/Methods/FasterGSUp/Trainer.py

Needs the clock hook (fastergs_sparse_from.py v5) first, for the end-of-training print. Idempotent. All env-gated:
with none of the variables below set, training is unchanged.

FGS_MCMC_RELOCATE=0   with USE_MCMC, the densification callback skips mcmc_densification: no relocation and, with the
                      cap at N, no growth either. Dead Gaussians stay.
FGS_MCMC_NOISE=0      with USE_MCMC, post_optimizer_step gets inject_noise=False.
FGS_OPACITY_RESET=0   without USE_MCMC, the periodic reset_opacities callback does nothing. The one extra reset at step
                      500 (non-black backgrounds only) is untouched.
FGS_ABL=1             count, and print one FGS_ABL line at the end of training (after FGS_TRAIN_END): the switch values,
                      relocation events and Gaussians relocated, dead Gaussians left in place when relocation is off,
                      noise steps, and the iterations at which opacities were reset. Counting dead Gaussians uses the
                      same test as Model.mcmc_densification and costs one device sync per densification step, in every
                      MCMC arm alike.
                      v2 (2026-11-10): the densify, morton_ordering and reset_opacities callbacks are timed with CUDA
                      events (no added synchronisation). They run between training steps, outside SEGCLOCK. One
                      FGS_ABL_CB line at the end gives their total time by callback and the cumulative time before every
                      1000-step point s (callbacks at iterations < s; a callback at iteration i runs after evaluation
                      point i and before training step i).
Upgrading from v1: restore the trainer saved before v1 was applied (~/Trainer.FasterGSUp.pre-ablate.bak on Capella)
and apply this version to it.
"""
import sys
from pathlib import Path

MARK = '# >>> fgs-ablate v2'
path = Path(sys.argv[1])
src = path.read_text()
if MARK in src:
    print('already patched:', path)
    sys.exit(0)
assert 'fgs-ablate' not in src, 'an older fgs-ablate version is applied: restore the pre-ablate trainer first'
assert '# >>> fgs-sparse-from v5' in src and 'FGS_TRAIN_END N=' in src, 'apply fastergs_sparse_from.py v5 first'


def sub(old, new):
    global src
    assert src.count(old) == 1, ('anchor not found exactly once', old)
    src = src.replace(old, new, 1)


# 0. the switch / counter state, one per trainer
A0 = 'from Optim.Samplers.DatasetSamplers import DatasetSampler\n'
sub(A0, A0 + '''# >>> fgs-ablate v2
def _fgs_abl(tr):
    a = getattr(tr, '_fgs_abl_st', None)
    if a is None:
        e = __import__('os').environ
        a = tr._fgs_abl_st = dict(relocate=e.get('FGS_MCMC_RELOCATE', '1') != '0', noise=e.get('FGS_MCMC_NOISE', '1') != '0',
                                  reset=e.get('FGS_OPACITY_RESET', '1') != '0', count=e.get('FGS_ABL') == '1',
                                  reloc_events=0, relocated=0, dead_left=0, noise_steps=0, resets=[], cb=[])
    return a


def _fgs_cbtime(fn):
    def w(self, *a, **k):
        st = _fgs_abl(self)
        if not st['count']:
            return fn(self, *a, **k)
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
        try:
            return fn(self, *a, **k)
        finally:
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()
            st['cb'].append((fn.__name__, int(a[0]) if a else -1, e0, e1))
    w.__name__, w.__doc__ = fn.__name__, fn.__doc__
    return w
# <<< fgs-ablate
''')

# 0b. time the three callbacks that run between training steps
for name in ('densify(self, iteration: int, dataset', 'morton_ordering(self, *_)', 'reset_opacities(self, *_)'):
    sub('    @torch.no_grad()' + chr(10) + '    def ' + name,
        '    @torch.no_grad()' + chr(10) + '    @_fgs_cbtime   # fgs-ablate (timing)' + chr(10) + '    def ' + name)

# 1. relocation
sub('''        if self.USE_MCMC:
            self.model.gaussians.mcmc_densification(min_opacity=0.005, cap_max=self.MAX_PRIMITIVES)
''', '''        if self.USE_MCMC:
            # >>> fgs-ablate (relocate)
            _a = _fgs_abl(self)
            _nd = 0
            if _a['count']:
                _g = self.model.gaussians
                _dm = _g.raw_opacities.flatten() <= __import__('math').log(0.005 / (1 - 0.005))
                _dm |= _g._rotations.mul(_g._rotations).sum(dim=1) < 1e-8
                _nd = int(_dm.sum().item())
            if _a['relocate']:
                self.model.gaussians.mcmc_densification(min_opacity=0.005, cap_max=self.MAX_PRIMITIVES)
                _a['reloc_events'] += int(_nd > 0)
                _a['relocated'] += _nd
            else:
                _a['dead_left'] = _nd
            # <<< fgs-ablate (relocate)
''')

# 2. opacity resets
sub('''    def reset_opacities(self, *_) -> None:
        """Reset opacities."""
        if not self.USE_MCMC:
            self.model.gaussians.reset_opacities()
''', '''    def reset_opacities(self, *_) -> None:
        """Reset opacities."""
        if not self.USE_MCMC and _fgs_abl(self)['reset']:   # fgs-ablate (reset)
            self.model.gaussians.reset_opacities()
            _fgs_abl(self)['resets'].append(int(_[0]) if _ else -1)
''')

# 3. noise injection
sub('''        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)
''', '''        _fgs_nz = self.USE_MCMC and _fgs_abl(self)['noise']   # fgs-ablate (noise)
        self.model.gaussians.post_optimizer_step(inject_noise=_fgs_nz)
        if _fgs_nz:
            _fgs_abl(self)['noise_steps'] += 1
''')

# 4. the FGS_ABL line, right after FGS_TRAIN_END
A4 = "                      f'peak_reserved_gib={torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}', file=self._fgs_log, flush=True)\n"
sub(A4, A4 + '''                # >>> fgs-ablate (print)
                _a = _fgs_abl(self)
                if _a['count']:
                    print(f"FGS_ABL relocate={int(_a['relocate'])} noise={int(_a['noise'])} reset={int(_a['reset'])} use_mcmc={self.USE_MCMC} "
                          f"relocation_events={_a['reloc_events']} relocated={_a['relocated']} dead_left={_a['dead_left']} "
                          f"noise_steps={_a['noise_steps']} resets={','.join(map(str, _a['resets'])) or 'none'}", file=self._fgs_log, flush=True)
                    _cbt = [(n, i, e0.elapsed_time(e1) / 1000.0) for n, i, e0, e1 in _a['cb']]
                    _by = {}
                    for n, i, d in _cbt:
                        _by[n] = _by.get(n, 0.0) + d
                    _cum = ','.join(f'{s_}:{sum(d for n, i, d in _cbt if i < s_):.4f}' for s_ in range(1000, self.NUM_ITERATIONS + 1, 1000))
                    print(f"FGS_ABL_CB total_s={sum(_by.values()):.4f} by={','.join(f'{n}:{v:.4f}' for n, v in sorted(_by.items())) or 'none'} "
                          f"calls={len(_cbt)} cum={_cum}", file=self._fgs_log, flush=True)
                # <<< fgs-ablate (print)
''')
path.write_text(src)
print('patched:', path)
