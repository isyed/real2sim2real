import numpy as np
from real2sim2real.egocentric.native_tracks import fit_camera


def test_camera_fit_uses_reserved_landmarks_only_for_validation():
    rng = np.random.default_rng(1)
    p = rng.uniform(-0.2, 0.2, (100, 3))
    p[:, 2] += 1
    k = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    t = np.array([0.03, -0.02, 0.01])
    project = (p + t) @ k.T
    q = project[:, :2] / project[:, 2:]
    train = np.arange(100) % 5 != 0
    pose, quality = fit_camera(p, q, k, train)
    assert quality["accepted"]
    np.testing.assert_allclose(pose[:3, 3], -t, atol=1e-6)
    bad = q.copy()
    bad[~train, 0] += 8
    rejected, quality = fit_camera(p, bad, k, train)
    assert not quality["accepted"] and np.isnan(rejected).all()
    assert quality["fit_inlier_median_px"] < 1e-4
    assert quality["validation_p90_px"] > 7.9
