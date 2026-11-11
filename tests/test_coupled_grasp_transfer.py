import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.controller_actor_training.coupled_grasp_transfer import (
    object_relative_pose,
)


def test_loaded_grasp_transfer_preserves_object_relative_wrist_in_rotated_world():
    initial, loaded, wrist = [np.eye(4) for _ in range(3)]
    initial[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.5, 0.2]).as_matrix()
    initial[:3, 3] = [0.3, -0.1, 0.2]
    loaded[:3, :3] = Rotation.from_euler("xyz", [-0.2, 0.7, 0.1]).as_matrix()
    loaded[:3, 3] = [1, 2, 3]
    wrist[:3, :3] = Rotation.from_euler("xyz", [0.4, 0.1, -0.2]).as_matrix()
    wrist[:3, 3] = [1.1, 2.2, 3.1]
    before = [p.copy() for p in (initial, loaded, wrist)]
    result = object_relative_pose(initial, loaded, wrist)
    np.testing.assert_allclose(
        np.linalg.inv(initial) @ result, np.linalg.inv(loaded) @ wrist, atol=1e-12
    )
    for actual, original in zip((initial, loaded, wrist), before):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize(
    "bad",
    [np.eye(3), np.diag([-1, 1, 1, 1]), np.diag([2, 1, 1, 1]), np.full((4, 4), np.nan)],
)
def test_transfer_rejects_scale_reflection_and_invalid_poses(bad):
    with pytest.raises(ValueError):
        object_relative_pose(np.eye(4), bad, np.eye(4))


def test_coupled_bounds_unfreeze_only_requested_wrist_and_fingers_within_native_limits():
    from real2sim2real.controller_actor_training.loaded_finger_replan import (
        joint_search_bounds,
    )

    names = ["fixture_x", "fixture_ry", "finger", "other"]
    limits = np.array([[-0.1, 0.1], [-1, 1], [0, 0.8], [-1, 1]])
    seed = np.array([0, 0, 0.7, 0.2])
    bounds, active = joint_search_bounds(names, limits, seed, {"finger"}, 0.3)
    assert bounds["fixture_x"] == pytest.approx([-1e-7, 1e-7])
    assert bounds["finger"] == pytest.approx([0.4, 0.8])
    assert active == {"finger"}
    bounds, active = joint_search_bounds(
        names, limits, seed, {"finger"}, 0.7, dict(translation_m=0.02, rotation_rad=0.3)
    )
    assert bounds["fixture_x"] == pytest.approx([-0.02, 0.02])
    assert bounds["fixture_ry"] == pytest.approx([-0.3, 0.3])
    assert bounds["finger"] == pytest.approx([0, 0.8])
    assert bounds["other"] == pytest.approx([0.2 - 1e-7, 0.2 + 1e-7])
    assert active == {"fixture_x", "fixture_ry", "finger"}
    with pytest.raises(ValueError):
        joint_search_bounds(names, limits, seed, {"finger"}, 0.7)
    with pytest.raises(ValueError):
        joint_search_bounds(names, limits, seed, {"fixture_x"}, 0.1)
    with pytest.raises(ValueError):
        joint_search_bounds(
            names,
            limits,
            seed,
            {"finger"},
            0.2,
            dict(translation_m=0.05, rotation_rad=0.3),
        )
