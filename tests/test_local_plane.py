import numpy as np
import pytest
from real2sim2real.egocentric.local_plane import (
    assess_plane,
    fit_support_plane,
    plane_patch_pose,
)


def test_support_fit_and_heldout_outliers_are_not_trimmed():
    rng = np.random.default_rng(8)
    xy = rng.uniform(-0.1, 0.1, (200, 2))
    p = np.c_[xy, 0.02 + 0.1 * xy[:, 0]]
    dirty = p.copy()
    dirty[:20, 2] += 0.03
    fit = fit_support_plane(dirty, seed=9)
    assert assess_plane(p, fit)["p90_m"] < 1e-8
    check = p.copy()
    check[:50, 2] += 0.02
    audit = assess_plane(check, fit)
    assert audit["p90_m"] == pytest.approx(0.02 / np.sqrt(1.01))
    assert audit["within_tolerance_fraction"] == 0.75
    T = plane_patch_pose(fit, [0, 0], 0.004)
    top = T[:3, 3] + T[:3, 2] * 0.002
    assert top @ np.array(fit["normal"]) == pytest.approx(fit["offset_m"])
    np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)


def test_rejects_degenerate_or_nonfinite_support():
    with pytest.raises(ValueError):
        fit_support_plane(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        fit_support_plane([[0, 0, 0], [1, 0, 0], [0, np.nan, 0]])
