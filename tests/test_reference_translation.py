import numpy as np
import pytest
from real2sim2real.controller_actor_training.reference_translation import (
    bounded_translation,
)


def test_zero_correction_reproduces_baseline_and_caps_only_translation():
    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    source = np.array([2.0, -3.0, 4.0])
    np.testing.assert_array_equal(
        bounded_translation(pose, source, gain=0, limit_m=0.02), pose
    )
    np.testing.assert_array_equal(
        bounded_translation(pose, source, gain=0.25, limit_m=0.02, blend=0), pose
    )
    candidate = bounded_translation(pose, source, gain=0.25, limit_m=0.02)
    assert np.linalg.norm(candidate[:3, 3] - pose[:3, 3]) == pytest.approx(0.02)
    np.testing.assert_array_equal(candidate[:3, :3], pose[:3, :3])
    np.testing.assert_array_equal(pose[:3, 3], [0.1, 0.2, 0.3])


def test_unbounded_or_nonrigid_inputs_are_rejected():
    with pytest.raises(ValueError):
        bounded_translation(np.eye(4), [0, 0, 1], gain=2, limit_m=0.02)
    pose = np.eye(4)
    pose[0, 0] = 2
    with pytest.raises(ValueError):
        bounded_translation(pose, [0, 0, 1], gain=0.25, limit_m=0.02)


def test_phase_window_preserves_excluded_controller_commands():
    from real2sim2real.controller_actor_training.reference_translation import (
        phase_window,
    )

    baseline = np.eye(4)
    source = np.array([0.2, -0.1, 0.3])
    for t in [-1, 0, 9.5, 12.5, 14.5, 17.5, 30]:
        proposal = bounded_translation(
            baseline,
            source,
            gain=0.05,
            limit_m=0.005,
            blend=phase_window(t, 12.5, 14.5, 0.5),
        )
        np.testing.assert_array_equal(proposal, baseline)
    assert phase_window(13.5, 12.5, 14.5, 0.5) == 1
    assert phase_window(12.75, 12.5, 14.5, 0.5) == pytest.approx(0.5)
    assert phase_window(14.25, 12.5, 14.5, 0.5) == pytest.approx(0.5)
    epsilon = 1e-6
    assert phase_window(12.5 + epsilon, 12.5, 14.5, 0.5) / epsilon < 1e-4
    with pytest.raises(ValueError):
        phase_window(13, 12.5, 14.5, 2)
