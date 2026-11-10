from types import SimpleNamespace

import numpy as np
import pytest

from real2sim2real.controller_actor_training.bilateral_patch_replan import finger_bounds


def test_patch_posture_bounds_preserve_wrists_and_unselected_fingers():
    model = SimpleNamespace(
        jnt_range=np.array([[-0.3, 0.3], [-1, 1], [0, 1.5], [-1, 1]])
    )
    names = ["fixture_x", "fixture_rx", "thumb_IP", "index_DIP"]
    bounds = finger_bounds(
        model,
        dict(joints=np.arange(4)),
        names,
        [0, 0, 0, 0.3],
        ["thumb_IP"],
        0.25,
        "fixture_",
    )
    assert bounds["thumb_IP"] == [0, 0.25]
    for name in ["fixture_x", "fixture_rx", "index_DIP"]:
        assert bounds[name][1] - bounds[name][0] <= 2e-7 + 1e-10
    with pytest.raises(ValueError):
        finger_bounds(
            model,
            dict(joints=np.arange(4)),
            names,
            [0, 0, 0, 0.3],
            ["fixture_x"],
            0.25,
            "fixture_",
        )
    with pytest.raises(ValueError):
        finger_bounds(
            model,
            dict(joints=np.arange(4)),
            names,
            [0, 0, float("nan"), 0.3],
            ["thumb_IP"],
            0.25,
            "fixture_",
        )
