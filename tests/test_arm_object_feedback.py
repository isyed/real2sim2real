import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.controller_actor_training.arm_object_feedback import (
    ArmObjectFeedback,
    joint_offset,
    object_relative_error,
)


def settings():
    return dict(
        gain_s_inv=2,
        max_speed_m_s=0.008,
        max_angular_speed_rad_s=0.1,
        max_offset_m=0.008,
        max_angle_rad=0.05,
        minimum_contact_n=0.1,
        maximum_reference_error_m=0.02,
        maximum_reference_angle_rad=np.deg2rad(10),
        support_load=dict(
            target_weight_fraction=0.3,
            gain_m_per_n_s=0.003,
            max_speed_m_s=0.005,
            max_offset_m=0.008,
        ),
    )


def test_object_rotation_moves_wrist_around_object_not_its_own_origin():
    reference = np.eye(4)
    reference[:3, 3] = [1, 2, 3]
    actual = reference.copy()
    actual[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    wrist = reference.copy()
    wrist[0, 3] += 0.2
    error = object_relative_error(actual, reference, wrist)
    np.testing.assert_allclose(error[:3], [-0.2, 0.2, 0], atol=1e-12)
    np.testing.assert_allclose(error[3:], [0, 0, np.pi / 2], atol=1e-12)
    np.testing.assert_allclose(
        object_relative_error(reference, reference, wrist), 0, atol=1e-12
    )


def test_contact_loss_and_reference_loss_freeze_without_chasing_object():
    controller = ArmObjectFeedback(settings(), ["support"])
    nominal, actual = np.eye(4), np.eye(4)
    actual[0, 3] = 0.015
    previous = np.zeros(6)
    for _ in range(300):
        result = controller.update(
            "support", actual, nominal, nominal, 0.01, contact_n=1, active=True
        )
        assert np.linalg.norm(result[:3] - previous[:3]) <= 0.008 * 0.01 + 1e-12
        assert np.linalg.norm(result[:3]) <= 0.008 + 1e-12
        previous = result.copy()
    for bad_contact, offset in [(0, 0.015), (1, 0.021)]:
        actual[0, 3] = offset
        result = controller.update(
            "support",
            actual,
            nominal,
            nominal,
            0.01,
            contact_n=bad_contact,
            active=True,
        )
        np.testing.assert_array_equal(result, previous)
    assert controller.frozen["support"] == 2


def test_load_sharing_requires_contact_and_stays_bounded():
    controller = ArmObjectFeedback(settings(), ["support"])
    identity = np.eye(4)
    kwargs = dict(contact_n=0, active=True, support=True, weight_n=5, upward_force_n=0)
    for _ in range(10):
        controller.update("support", identity, identity, identity, 0.01, **kwargs)
    assert controller.load_offset == 0
    kwargs["contact_n"] = 1
    for _ in range(1000):
        result = controller.update(
            "support", identity, identity, identity, 0.01, **kwargs
        )
    assert 0 < result[2] <= 0.008
    held = controller.load_offset
    kwargs["upward_force_n"] = 3
    controller.update("support", identity, identity, identity, 0.01, **kwargs)
    assert controller.load_offset < held


def test_singular_jacobian_produces_finite_bounded_correction():
    assert not joint_offset(np.zeros((6, 7)), np.ones(6)).any()
    delta = joint_offset(np.eye(6), np.ones(6))
    assert np.linalg.norm(delta) <= 0.15 + 1e-12
    assert (delta > 0).all()
    with pytest.raises(ValueError):
        joint_offset(np.zeros((3, 6)), np.ones(6))


def test_nonrigid_and_unbounded_feedback_is_rejected():
    with pytest.raises(ValueError):
        ArmObjectFeedback({**settings(), "max_offset_m": 1}, ["hand"])
    bad = np.eye(4)
    bad[0, 0] = 2
    with pytest.raises(ValueError):
        object_relative_error(bad, np.eye(4), np.eye(4))


def test_precontact_support_alignment_uses_primary_retention_only_during_approach():
    from real2sim2real.controller_actor_training.arm_object_feedback import (
        support_contact_evidence,
    )

    assert support_contact_evidence("support_approach", 0, 2) == 0
    assert (
        support_contact_evidence("support_approach", 0, 2, allow_precontact=True) == 2
    )
    assert (
        support_contact_evidence("support_approach", 0, 0, allow_precontact=True) == 0
    )
    assert support_contact_evidence("hold", 0, 2, allow_precontact=True) == 0
    assert (
        support_contact_evidence("support_seat", 0.5, 2, allow_precontact=True) == 0.5
    )
    with pytest.raises(ValueError):
        support_contact_evidence(
            "support_approach", 0, float("nan"), allow_precontact=True
        )


def test_height_alignment_latches_once_and_keeps_speed_travel_and_retention_gates():
    from real2sim2real.controller_actor_training.arm_object_feedback import (
        SupportHeightAlignment,
    )

    s = dict(
        max_offset_m=0.01,
        max_speed_m_s=0.01,
        minimum_primary_contact_n=0.1,
        maximum_initial_error_m=0.02,
    )
    c = SupportHeightAlignment(s)
    assert c.update("lift", [0, 0, -0.007], 1, 0.01) == 0
    assert c.update("support_approach", [0, 0, -0.007], 1, 0.01) == pytest.approx(
        -0.0001
    )
    for _ in range(100):
        c.update("hold", [0, 0, 0.015], 0, 0.01)
    assert c.offset == pytest.approx(-0.007) and c.target_offset == pytest.approx(
        -0.007
    )
    rejected = SupportHeightAlignment(s)
    assert rejected.update("support_approach", [0, 0, -0.007], 0, 0.01) == 0
    assert rejected.update("support_approach", [0, 0, -0.007], 1, 0.01) == 0
    assert rejected.state == "rejected_observation"
