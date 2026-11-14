import numpy as np
import pytest
from real2sim2real.egocentric.correspondence_audit import (
    cycle_consistency,
    error_summary,
    spatial_groups,
)


def test_cycle_preserves_missing_and_exposes_inconsistent_identity():
    a = np.array([[10, 10], [20, 20], [30, 30.0]])
    c = np.array([[50, 50], [60, 60], [70, 70.0]])
    b = a[:2] + 2
    error = cycle_consistency(a, c, a[:2], b, b, c[:2] + [[0, 0], [3, 4]])
    assert np.allclose(error[:2], [0, 5])
    assert np.isnan(error[2])


def test_ambiguous_cycle_is_unavailable_not_best_selected():
    a = np.array([[10.0, 10]])
    assert np.isnan(
        cycle_consistency(a, a, a, a, np.repeat(a, 2, axis=0), [[10, 10], [20, 20]])[0]
    )


def test_spatial_cells_and_signed_summary():
    cells, radii = spatial_groups(
        [[0, 0], [99, 79], [50, 40]], [100, 80], [50, 40], columns=2, rows=2
    )
    assert cells.tolist() == [0, 3, 3]
    assert radii[0] == 1 and radii[2] == 0
    assert error_summary([-5, -3, 1])["signed_median_px"] == -3
    assert error_summary([])["p90_px"] is None
    with pytest.raises(ValueError):
        spatial_groups([[100, 2]], [100, 80], [50, 40])


def test_reciprocal_descriptor_support_rejects_false_saved_identity():
    from real2sim2real.egocentric.correspondence_audit import reciprocal_support

    p = np.array([[10.0, 10], [20, 20], [30, 30]])
    d = np.array([[0.0, 0], [10, 10], [20, 20]], np.float32)
    # An exact reciprocal pair, then a saved match with the wrong target ID.
    supported = reciprocal_support(
        p[:2], np.array([p[0] + 1, p[2] + 1]), (p, d), (p + 1, d)
    )
    assert supported.tolist() == [True, False]
