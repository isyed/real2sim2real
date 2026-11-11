import numpy as np
import pytest

from real2sim2real.controller_actor_training.finger_release import FingerRelease
from real2sim2real.controller_actor_training.arm_finger_feedback import finger_waypoint


def settings():
    return dict(
        start_s=3.0,
        duration_s=0.8,
        settle_s=0.8,
        support_dwell_s=0.2,
        maximum_clearance_m=0.005,
        maximum_speed_m_s=0.02,
    )


def observation():
    return dict(
        table_contacts=1, clearance_m=0.0, speed_m_s=0.0, table_load_fraction=1.0
    )


def test_release_waits_for_observed_support_then_opens_without_command_jump():
    release = FingerRelease(settings())
    held, opened = np.array([0.2, 0.3]), np.array([0.0, 0.1])
    args = (0.1, held, opened, held)
    blocked = observation()
    blocked["table_load_fraction"] = 0.1
    assert release.command(3.0, *args, **blocked) is None
    assert release.command(3.1, *args, **observation()) is None
    np.testing.assert_array_equal(release.command(3.2, *args, **observation()), held)
    np.testing.assert_allclose(
        release.command(3.6, *args, **observation()), (held + opened) / 2
    )
    lost = observation()
    lost["table_contacts"] = 0
    np.testing.assert_allclose(release.command(4.0, *args, **lost), opened)
    assert release.started_s == 3.2
    assert release.diagnostics()["maximum_open_error_rad"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "key,value",
    [
        ("clearance_m", 0.006),
        ("speed_m_s", 0.03),
        ("clearance_m", -0.004),
        ("table_contacts", 0),
    ],
)
def test_unsafe_observation_blocks_release(key, value):
    release = FingerRelease(settings())
    obs = observation()
    obs[key] = value
    for t in np.arange(3, 4, 0.1):
        assert release.command(t, 0.1, [1], [0], [1], **obs) is None


def test_kinematic_release_leaves_old_waypoint_schedule_exact():
    rollout = dict(approach_s=1.0, close_s=0.8)
    np.testing.assert_array_equal(
        finger_waypoint(3.4, [0.2], [0.0], [1.0], rollout), [1.0]
    )
    rollout["finger_release"] = settings()
    np.testing.assert_allclose(
        finger_waypoint(3.4, [0.2], [0.0], [1.0], rollout), [0.5]
    )
    with pytest.raises(ValueError):
        FingerRelease(dict(settings(), duration_s=0.1))
