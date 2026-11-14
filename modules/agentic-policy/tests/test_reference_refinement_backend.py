from types import SimpleNamespace

import numpy as np
import pytest
from real2sim2real.agentic_policy.reference_refinement import ReferenceRefinementBackend


def backend():
    b = ReferenceRefinementBackend.__new__(ReferenceRefinementBackend)
    b.bounds = {}
    b.edits = None
    b.session = SimpleNamespace(
        started=False,
        finished=False,
        refinement=None,
        config={},
        ref={},
        model=SimpleNamespace(jnt_range=np.array([[-1, 1]])),
        hand={"joints": [0]},
        data=SimpleNamespace(
            qpos=np.zeros(1), qvel=np.zeros(1), ctrl=np.zeros(1), time=0.0
        ),
    )

    def run():
        b.session.started = True
        return {"task_completed": True}

    b.session.run = run
    return b


def test_empty_edit_keeps_original_loop_and_cannot_reset():
    b = backend()
    assert b.run({})["task_completed"]
    assert b.session.refinement is None
    with pytest.raises(RuntimeError, match="once"):
        b.run({})


def test_invalid_proposal_does_not_step_and_keeps_rejection_auditable(tmp_path):
    b = backend()
    with pytest.raises(ValueError, match="declared"):
        b.run({"scene": "replacement.xml"})
    assert b.session.started is False and b.session.refinement is None
    result = b.finish(tmp_path)
    assert not result["task_completed"]
    with np.load(tmp_path / "physics/states.npz") as saved:
        np.testing.assert_array_equal(saved["timestamps_s"], [0.0])
        assert len(saved["qpos"]) == 1
