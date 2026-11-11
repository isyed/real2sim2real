import numpy as np
import pytest

from real2sim2real.controller_actor_training.arm_tracking_control import (
    ArmTrackingCompensator,
    InertialFeedforward,
)


def settings(**extra):
    return dict(
        velocity_fraction=0.5,
        filter_time_s=0.05,
        max_acceleration_rad_s2=2,
        integral_gain_s_inv=4,
        max_offset_rad=0.06,
        max_offset_speed_rad_s=0.1,
        **extra,
    )


def test_velocity_step_is_causal_rate_limited_and_does_not_retime_position():
    s = settings()
    s["integral_gain_s_inv"] = 0
    controller = ArmTrackingCompensator(s, [1])
    previous = 0
    for _ in range(100):
        command, velocity = controller.update(
            [0.2], [0.1], [10], [0], [5], 0.01, active=True
        )
        np.testing.assert_array_equal(command, [0.2])
        assert abs(velocity[0] - previous) <= 0.02 + 1e-12
        assert 0 <= velocity[0] <= 1
        previous = velocity[0]
    assert velocity[0] > 0.9


def test_integral_corrects_tracking_error_with_travel_and_rate_bounds():
    controller = ArmTrackingCompensator(settings(), [1])
    previous = 0
    for _ in range(100):
        command, _ = controller.update([0.2], [0.1], [0], [0], [5], 0.01, active=True)
        offset = command[0] - 0.2
        assert 0 <= offset <= 0.0600000001
        assert abs(offset - previous) <= 0.0010000001
        previous = offset
    assert offset == pytest.approx(0.06)


def test_torque_anti_windup_freezes_outward_but_allows_unwinding():
    controller = ArmTrackingCompensator(settings(), [1])
    command, _ = controller.update([0.2], [0], [0], [5], [5], 0.01, active=True)
    np.testing.assert_array_equal(command, [0.2])
    assert controller.frozen_joint_steps == 1
    command, _ = controller.update([0.2], [0.3], [0], [5], [5], 0.01, active=True)
    assert command[0] < 0.2


def test_inactive_phase_and_invalid_observations():
    controller = ArmTrackingCompensator(settings(), [1])
    command, velocity = controller.update([0.2], [0], [1], [0], [5], 0.01, active=False)
    np.testing.assert_array_equal(command, [0.2])
    assert velocity[0] == 0
    with pytest.raises(ValueError):
        controller.update([np.nan], [0], [0], [0], [5], 0.01, active=True)
    with pytest.raises(ValueError):
        ArmTrackingCompensator({**settings(), "max_offset_rad": 2}, [1])


def test_inertia_feedforward_preserves_coupling_and_torque_caps():
    controller = InertialFeedforward(
        dict(time_constant_s=0.02, torque_fraction=0.5), [10, 10]
    )
    mass = np.array([[2.0, 1.0], [1.0, 3.0]])
    for _ in range(500):
        torque = controller.update(mass, [1, 0], 0.001, active=True)
    np.testing.assert_allclose(torque, [2, 1], atol=1e-8)
    for _ in range(500):
        torque = controller.update(mass, [100, -100], 0.001, active=True)
        assert (abs(torque) <= 5).all()
    assert torque[0] > 4.99 and torque[1] < -4.99
    before = np.linalg.norm(torque)
    torque = controller.update(mass, [100, -100], 0.001, active=False)
    assert np.linalg.norm(torque) < before
