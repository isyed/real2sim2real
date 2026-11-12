import numpy as np
import pytest

from real2sim2real.controller_actor_training.wrist_withdrawal import (
    validate_withdrawal,
    offset,
    release_ready,
)


def settings():
    return dict(
        delay_s=0.5,
        duration_s=0.8,
        vectors_world_m=dict(right=[0.02, 0, 0.01], left=[-0.018, 0.009, 0.01]),
    )


def test_withdrawal_stays_on_release_clock_and_requires_supported_unhooking():
    s = settings()
    release = dict(start_s=13.0, duration_s=1.5)
    validate_withdrawal(s, release)
    np.testing.assert_array_equal(offset(13.5, "right", s, release), [0, 0, 0])
    np.testing.assert_allclose(offset(13.9, "right", s, release), [0.01, 0, 0.005])
    np.testing.assert_allclose(
        offset(20, "left", s, release), s["vectors_world_m"]["left"]
    )
    assert release_ready(13.4, s, release, None)
    assert not release_ready(13.5, s, release, None)
    assert not release_ready(13.5, s, release, 13.1)
    assert release_ready(13.5, s, release, 13.001)
    for v in [[0.04, 0, 0], [0, 0, -0.001], [float("nan"), 0, 0]]:
        bad = settings()
        bad["vectors_world_m"]["right"] = v
        with pytest.raises(ValueError):
            validate_withdrawal(bad, release)
    with pytest.raises(ValueError):
        validate_withdrawal(dict(s, delay_s=1.0), release)
    with pytest.raises(ValueError, match="thumb"):
        validate_withdrawal(s, dict(release, thumb_duration_s=0.75))


def test_wrist_rotation_is_world_framed_and_does_not_orbit_the_object():
    from scipy.spatial.transform import Rotation
    from real2sim2real.controller_actor_training.wrist_withdrawal import transform

    s = settings()
    s["rotation_vectors_world_rad"] = dict(right=[0, 0, 0.15], left=[0, 0, 0])
    release = dict(start_s=13.0, duration_s=1.5)
    validate_withdrawal(s, release)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("x", 0.7).as_matrix()
    pose[:3, 3] = [1.0, 2.0, 3.0]
    before = pose.copy()
    actual = transform(15.0, "right", pose, s, release)
    np.testing.assert_allclose(actual[:3, 3], [1.02, 2.0, 3.01])
    np.testing.assert_allclose(
        actual[:3, :3], Rotation.from_rotvec([0, 0, 0.15]).as_matrix() @ pose[:3, :3]
    )
    np.testing.assert_array_equal(pose, before)
    np.testing.assert_array_equal(transform(13.5, "right", pose, s, release), pose)
    s["rotation_vectors_world_rad"]["right"] = [0, 0, 0.201]
    with pytest.raises(ValueError, match="rotation"):
        validate_withdrawal(s, release)


def test_rotation_before_translation_uses_its_own_clock_and_support_gate():
    from real2sim2real.controller_actor_training.wrist_withdrawal import transform

    s = settings()
    s.update(
        delay_s=0.65,
        rotation_vectors_world_rad=dict(right=[0, 0.15, 0], left=[0, 0, 0]),
        rotation_delay_s=0.1,
        rotation_duration_s=0.5,
    )
    release = dict(start_s=13.0, duration_s=1.5)
    validate_withdrawal(s, release)
    pose = transform(13.4, "right", np.eye(4), s, release)
    np.testing.assert_array_equal(pose[:3, 3], [0, 0, 0])
    assert not np.allclose(pose[:3, :3], np.eye(3))
    assert not release_ready(13.1, s, release, None)
    assert release_ready(13.1, s, release, 13.001)
    assert not release_ready(13.08, s, release, None, command_lead_s=0.02)
    assert release_ready(13.08, s, release, 13.001, command_lead_s=0.02)
    with pytest.raises(ValueError):
        release_ready(13.1, s, release, None, command_lead_s=0.1)
    with pytest.raises(ValueError):
        validate_withdrawal(dict(s, rotation_duration_s=2.0), release)
