"""Numerical motion conditioning must work without a simulator installation."""

import os
from pathlib import Path
import subprocess
import sys


def test_reference_conditioning_imports_without_simulation_or_training_libraries():
    source = (
        Path(__file__).resolve().parents[1] / "modules/controller-actor-training/src"
    )
    env = dict(os.environ, PYTHONPATH=str(source))
    code = """
import importlib.abc
import sys

blocked = {'warp', 'newton', 'torch', 'jax'}
class NoSimulation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, NoSimulation())

from flash_chord.runtime import actions
from flash_chord.runtime.actions.reference_denoising import (
    ReferenceDenoisingConfig, condition_reference,
)
from flash_chord.runtime.sim import SimConfig
assert SimConfig(fps=25, substeps=4).sim_dt == 0.01
import numpy as np
reference = np.zeros((20, 1))
result = condition_reference(reference, ('joint',), fps=50,
    config=ReferenceDenoisingConfig(('joint',)))
np.testing.assert_array_equal(result, reference)
assert result is not reference
assert 'SonicJointResidualAction' in dir(actions)
assert not blocked.intersection(sys.modules)
try:
    actions.nonexistent_action
except AttributeError:
    pass
else:
    raise AssertionError('Unknown actions must raise AttributeError')
"""
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=30)
