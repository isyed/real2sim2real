import numpy as np
import pytest
from real2sim2real.egocentric.axis_motion import axis_frame, motion_summary


def test_axis_gauge_handles_inversion_and_preserves_proper_rotations():
    previous = np.eye(3)
    for theta in np.linspace(0, np.pi, 21):
        z = np.array([np.sin(theta), 0, np.cos(theta)])
        R = axis_frame(z, previous)
        np.testing.assert_allclose(R[:, 2], z, atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1)
        previous = R
    np.testing.assert_allclose(axis_frame([0, 0, -1])[:, 2], [0, 0, -1], atol=1e-12)


def test_source_clock_gaps_are_not_bridged_for_motion_rates():
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, 0, 3] = [0, 0.1, np.nan, 100]
    report = motion_summary(
        poses, [True, True, False, True], [10, 11, 12, 13], [5, 5.1, 5.2, 5.3]
    )
    assert report["maximum_adjacent_translation_speed_m_s"] == pytest.approx(1)
    assert report["gaps"][0]["first_frame"] == 12
    assert not report["gaps_filled"] and not report["time_warped"]


def test_nonrigid_pose_cannot_be_an_accepted_observation():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 0] = 2
    with pytest.raises(ValueError):
        motion_summary(poses, [True, True], [0, 1], [0.0, 0.1])
