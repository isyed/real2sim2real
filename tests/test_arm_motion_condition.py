from copy import deepcopy

import numpy as np
import pytest

from real2sim2real.controller_actor_training.arm_motion_condition import apply_condition


def source():
    return dict(
        records=[
            dict(
                phase=p,
                time_s=float(i),
                reference_position_m=[i, 0, 1],
                reference_rotation=np.eye(3).tolist(),
                object_position_m=[i, 0, 1],
                object_rotation=np.eye(3).tolist(),
            )
            for i, p in enumerate(["approach", "close", "lift", "hold", "hold"])
        ]
    )


def test_stationary_condition_freezes_only_hold_and_keeps_native_clock_and_source():
    parent = source()
    before = deepcopy(parent)
    targets = np.arange(10).reshape(5, 2)
    result, held = apply_condition(parent, targets, "stationary_hold")
    np.testing.assert_array_equal(held[:3], targets[:3])
    np.testing.assert_array_equal(held[3:], np.tile(targets[3], (2, 1)))
    assert result["records"][4]["reference_position_m"] == [3, 0, 1]
    assert result["records"][4]["time_s"] == 4
    assert parent == before
    original, commands = apply_condition(parent, targets, "demonstrated")
    assert original is parent and commands is targets


def test_stationary_condition_rejects_missing_or_disjoint_hold():
    parent = source()
    parent["records"][4]["phase"] = "lift"
    with pytest.raises(ValueError):
        apply_condition(parent, np.zeros((5, 2)), "stationary_hold")
    with pytest.raises(ValueError):
        apply_condition(source(), np.zeros((5, 2)), "unknown")
