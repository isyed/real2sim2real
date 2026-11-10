import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.controller_actor_training.grasp_feedback import (
    contact_closure,
    orientation_correction,
    relative_translation_drift,
)


def test_contact_feedback_closes_on_missing_force_and_releases_excess():
    settings = dict(
        target_force_n=3.0,
        gain_rad_per_n_s=0.08,
        max_speed_rad_s=0.2,
        max_offset_rad=0.3,
    )
    assert contact_closure(0.0, 0.0, 0.1, settings) == pytest.approx(0.02)
    assert contact_closure(0.3, 0.0, 0.1, settings) == 0.3
    assert contact_closure(0.1, 20.0, 0.1, settings) == pytest.approx(0.08)
    assert contact_closure(0.0, 20.0, 0.1, settings) == 0.0
    with pytest.raises(ValueError):
        contact_closure(0.0, float("nan"), 0.1, settings)


def test_orientation_feedback_corrects_world_error_through_rotated_wrist():
    settings = dict(gain_s_inv=2.0, max_speed_rad_s=0.35, max_offset_rad=0.3)
    current = Rotation.from_rotvec([0.2, 0, 0]).as_matrix()
    jac = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    offset = orientation_correction(np.zeros(3), current, np.eye(3), jac, 0.1, settings)
    assert np.linalg.norm(offset) <= 0.035 + 1e-12
    assert (jac @ offset)[0] < 0
    np.testing.assert_allclose((jac @ offset)[1:], 0, atol=1e-12)
    for _ in range(100):
        offset = orientation_correction(offset, current, np.eye(3), jac, 0.1, settings)
    assert np.linalg.norm(offset) <= 0.3 + 1e-12


def test_singular_feedback_cannot_produce_unbounded_command():
    result = orientation_correction(
        np.zeros(3),
        Rotation.from_rotvec([1, 0, 0]).as_matrix(),
        np.eye(3),
        np.zeros((3, 3)),
        0.01,
        dict(gain_s_inv=2.0, max_speed_rad_s=0.35, max_offset_rad=0.3),
    )
    np.testing.assert_array_equal(result, np.zeros(3))


def test_rigid_grasp_rotation_is_not_slip_but_local_displacement_is():
    rotation = Rotation.from_euler("z", [[0], [90]], degrees=True).as_matrix()
    wrist = np.array([[0, 0, 0], [0.2, 0.3, 0.4]])
    local = np.array([[0.1, 0, 0], [0.1, 0, 0]])
    objects = wrist + np.einsum("nij,nj->ni", rotation, local)
    assert relative_translation_drift(objects, wrist, rotation) == pytest.approx(
        0.0, abs=1e-12
    )
    local[1, 0] += 0.012
    objects = wrist + np.einsum("nij,nj->ni", rotation, local)
    assert relative_translation_drift(objects, wrist, rotation) == pytest.approx(0.012)


def test_later_diagnostics_cannot_erase_failed_development_admission():
    from real2sim2real.controller_actor_training.grasp_robustness import (
        validation_decision,
    )

    assert (
        validation_decision(False, [dict(success=True)])
        == "diagnostic_only_development_failed"
    )
    assert validation_decision(True, [dict(success=False)]) == "validation_failed"
    assert validation_decision(True, []) == "validation_failed"
    assert (
        validation_decision(True, [dict(success=True)])
        == "passes_declared_controller_cases"
    )
