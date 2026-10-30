"""CPU tests of alt_reinit/tests/check_variant.py and alt_reinit/make_reinit_variant.sh (BUILD=0) on a synthetic
Faster-GS checkout that has the anchors the three trainer patches need:  python -m pytest -q alt_reinit/tests/test_check_variant.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHECK = [sys.executable, str(HERE / 'check_variant.py')]

STOCK_TRAINER = '''"""FasterGS/Trainer.py"""

import torch

from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.FasterGS.Loss import FasterGSLoss


class FasterGSTrainer:
    """Defines the trainer for the FasterGS variant."""

    def __init__(self, **kwargs) -> None:
        self.requires_empty_cache = True
        self.train_sampler = None
        self.loss = None

    @training_callback(priority=80)
    def training_iteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Performs a training step."""
        loss = self.loss(view)
        loss.backward()
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad()
        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)
        if self.model.ppisp is not None:
            self.model.ppisp.step()
'''


def git(cwd, *args):
    return subprocess.run(['git', '-C', str(cwd), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture()
def workspace(tmp_path):
    src = tmp_path / 'ws' / 'nerficg' / 'src' / 'Methods' / 'FasterGS'
    (src / 'FasterGSCudaBackend' / 'FasterGSCudaBackend').mkdir(parents=True)
    (src / 'Trainer.py').write_text(STOCK_TRAINER)
    (src / 'Model.py').write_text('from Methods.FasterGS.FasterGSCudaBackend import FusedAdam\n')
    (src / '__init__.py').write_text('')
    (src / 'fastergs_garden.yaml').write_text('METHOD_TYPE: FasterGS\nMODEL_NAME: fastergs_garden\n')
    (src / 'FasterGSCudaBackend' / 'FasterGSCudaBackend' / 'kernel.cu').write_text('// FasterGS kernel\n')
    git(src.parent, 'init', '-q', str(src))
    git(src, 'config', 'user.email', 't@t')
    git(src, 'config', 'user.name', 't')
    git(src, 'add', '-A')
    git(src, 'commit', '-qm', 'base')
    base = git(src, 'rev-parse', '--short', 'HEAD').strip()
    env = dict(os.environ, DATA=str(tmp_path / 'ws'), BASE=base, BUILD='0')
    r = subprocess.run(['bash', str(REPO / 'alt_reinit' / 'make_reinit_variant.sh')], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return dict(src=src, dst=src.parent / 'FasterGSReinit', base=base, tmp=tmp_path)


def test_make_variant_and_checks_pass(workspace):
    dst = workspace['dst']
    assert (dst / 'FasterGSReinitCudaBackend' / 'FasterGSReinitCudaBackend' / 'kernel.cu').exists()
    assert (dst / 'fastergsreinit_garden.yaml').read_text().startswith('METHOD_TYPE: FasterGSReinit')
    assert (dst / 'reinit_core.py').exists() and (dst / 'reinit_trainer.py').exists()
    r = subprocess.run(CHECK + ['tree', str(dst), '--base', workspace['base']], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    stock = workspace['tmp'] / 'stock_Trainer.py'
    stock.write_text(STOCK_TRAINER)
    r = subprocess.run(CHECK + ['trainer', str(stock), str(dst / 'Trainer.py')], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    compile((dst / 'Trainer.py').read_text(), 'Trainer.py', 'exec')
    # rebuilding over a variant made by the script is allowed; over anything else it is refused
    env = dict(os.environ, DATA=str(workspace['tmp'] / 'ws'), BASE=workspace['base'], BUILD='0')
    assert subprocess.run(['bash', str(REPO / 'alt_reinit' / 'make_reinit_variant.sh')], env=env, capture_output=True).returncode == 0
    (dst / '.alt-reinit-variant').unlink()
    assert subprocess.run(['bash', str(REPO / 'alt_reinit' / 'make_reinit_variant.sh')], env=env, capture_output=True).returncode != 0


def test_trainer_check_fails_on_an_edit_outside_the_blocks(workspace):
    dst = workspace['dst']
    t = (dst / 'Trainer.py').read_text().replace('        loss.backward()\n', '        loss.backward(retain_graph=True)\n')
    (dst / 'Trainer.py').write_text(t)
    stock = workspace['tmp'] / 'stock_Trainer.py'
    stock.write_text(STOCK_TRAINER)
    r = subprocess.run(CHECK + ['trainer', str(stock), str(dst / 'Trainer.py')], capture_output=True, text=True)
    assert r.returncode == 1 and 'FAIL variant Trainer.py minus marked blocks' in r.stdout


def test_trainer_check_fails_on_an_ungated_block(workspace):
    dst = workspace['dst']
    t = (dst / 'Trainer.py').read_text().replace("if __import__('os').environ.get('FGS_CBCLOCK'):", "if True:")
    (dst / 'Trainer.py').write_text(t)
    stock = workspace['tmp'] / 'stock_Trainer.py'
    stock.write_text(STOCK_TRAINER)
    r = subprocess.run(CHECK + ['trainer', str(stock), str(dst / 'Trainer.py')], capture_output=True, text=True)
    assert r.returncode == 1 and 'FAIL cbclock installs only with FGS_CBCLOCK' in r.stdout


def test_tree_check_fails_on_other_changes(workspace):
    dst = workspace['dst']
    (dst / 'Model.py').write_text((dst / 'Model.py').read_text() + '# changed\n')
    git(dst, 'commit', '-qam', 'touch Model.py')
    r = subprocess.run(CHECK + ['tree', str(dst), '--base', workspace['base']], capture_output=True, text=True)
    assert r.returncode == 1 and "FAIL after the rename only" in r.stdout
