from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.controller_actor_training.closing_wrist_offsets import (
    fraction,
    nonzero,
    transform,
    validate,
)
from real2sim2real.controller_actor_training.closing_grasp_plan import (
    command_bounds,
    offsets,
    response_deflection,
    environment_excess,
)


def settings():
    return dict(
        vectors_object_m=dict(right=[0.002, 0, 0], left=[-0.002, 0, 0]),
        rotation_vectors_object_rad=dict(right=[0.02, 0, 0], left=[0, 0, 0]),
    )


def test_preload_uses_closure_clock_and_recovers_before_original_release():
    rollout = dict(approach_s=1, close_s=0.8)
    original = deepcopy(rollout)
    assert fraction(0.5, rollout, 12, 13) == 0
    assert fraction(1, rollout, 12, 13) == 0
    assert fraction(1.4, rollout, 12, 13) == pytest.approx(0.5)
    assert fraction(1.8, rollout, 12, 13) == 1
    assert fraction(8, rollout, 12, 13) == 1
    assert fraction(12.5, rollout, 12, 13) == 0.5
    assert fraction(13, rollout, 12, 13) == 0
    assert fraction(16, rollout, 12, 13) == 0
    assert rollout == original
    with pytest.raises(ValueError):
        fraction(2, rollout, 13, 12)


def test_object_frame_preload_rotates_about_wrist_origin_without_orbit():
    wrist, obj = np.eye(4), np.eye(4)
    wrist[:3, 3] = [1, 2, 3]
    obj[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    original, obj_original = wrist.copy(), obj.copy()
    s = settings()
    result = transform(
        2, "right", wrist, obj, s, dict(approach_s=1, close_s=0.8), 12, 13
    )
    np.testing.assert_allclose(result[:3, 3], [1, 2.002, 3])
    np.testing.assert_allclose(
        result[:3, :3], Rotation.from_rotvec([0, 0.02, 0]).as_matrix(), atol=1e-14
    )
    delta = offsets(wrist, result, obj[:3, :3])
    np.testing.assert_allclose(
        delta["translation"], s["vectors_object_m"]["right"], atol=1e-14
    )
    np.testing.assert_allclose(
        delta["rotation"], s["rotation_vectors_object_rad"]["right"], atol=1e-14
    )
    np.testing.assert_array_equal(wrist, original)
    np.testing.assert_array_equal(obj, obj_original)


def test_zero_preload_is_exact_and_invalid_proposals_cannot_expand_budgets():
    s = settings()
    assert nonzero(s) and not nonzero(None)
    for key in s:
        for role in s[key]:
            s[key][role] = [0, 0, 0]
    assert not nonzero(s)
    wrist = np.eye(4)
    wrist[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.3, 0.1]).as_matrix()
    np.testing.assert_array_equal(
        transform(
            2, "left", wrist, np.eye(4), s, dict(approach_s=1, close_s=0.8), 12, 13
        ),
        wrist,
    )
    s["vectors_object_m"]["right"] = [0.005, 0.005, 0]
    with pytest.raises(ValueError):
        validate(s)
    s = settings()
    s["rotation_vectors_object_rad"]["left"] = [float("nan"), 0, 0]
    with pytest.raises(ValueError):
        validate(s)


def test_loaded_command_bounds_keep_both_command_and_predicted_pose_legal():
    nominal, delta = np.array([0.1, -0.1]), np.array([0.15, -0.15])
    limits = np.array([[-0.2, 0.2], [-0.2, 0.2]])
    lo, hi = command_bounds(nominal, delta, limits, 0.1)
    assert np.all(lo >= limits[:, 0]) and np.all(hi <= limits[:, 1])
    assert np.all(lo - delta >= limits[:, 0]) and np.all(hi - delta <= limits[:, 1])
    assert np.all(abs(lo - nominal) <= 0.1 + 1e-12)
    with pytest.raises(ValueError):
        command_bounds(nominal, delta * 100, limits, 0.1)


def test_supported_motor_deflection_removes_existing_bias_without_double_compensation():
    command = np.array([0.3, 0.4])
    actual = np.array([0.2, 0.25])
    bias = np.array([0, 0.02])
    before = [x.copy() for x in (command, actual, bias)]
    np.testing.assert_allclose(
        response_deflection(command, actual, bias, 1), [0.1, 0.13]
    )
    np.testing.assert_allclose(
        response_deflection(command, actual, bias, 0.5), [0.05, 0.065]
    )
    for x, y in zip(before, (command, actual, bias)):
        np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError):
        response_deflection(command, actual, [0], 1)


def test_a_clear_loaded_prediction_cannot_hide_commanded_self_collision():
    from types import SimpleNamespace

    loaded = SimpleNamespace(contact=[])
    commanded = SimpleNamespace(
        contact=[
            SimpleNamespace(geom1=1, geom2=2, dist=-0.012),
            SimpleNamespace(geom1=1, geom2=9, dist=-0.02),
        ]
    )
    np.testing.assert_array_equal(
        environment_excess(None, loaded, {1, 2}, set(), {9}), [0, 0]
    )
    # Intended pad/object contacts are excluded, hand/hand collision is not.
    np.testing.assert_allclose(
        environment_excess(None, commanded, {1, 2}, set(), {9}), [0.009, 0.009]
    )
