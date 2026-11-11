import numpy as np
import pytest

from real2sim2real.egocentric.surface_patch import fit_patch, normal_repeatability


def grid():
    x, y = np.meshgrid(np.linspace(-0.007, 0.007, 7), np.linspace(-0.007, 0.007, 7))
    return np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]


def test_metric_patch_coverage_and_normal_do_not_certify_thickness_or_insertion():
    fit = fit_patch(grid(), [0, 0, 0], [0, 0, 1])
    assert fit["accepted"]
    assert fit["center_in_hull_margin_m"] == pytest.approx(0.007)
    assert fit["normal_error_deg"] == pytest.approx(0)
    assert not fit["thickness_verified"] and not fit["insertion_clearance_verified"]
    assert not fit["finite_bearing_area_verified"]


def test_normal_fit_rejects_point_evidence_line_and_partial_coverage():
    assert not fit_patch(grid()[:3], [0, 0, 0], [0, 0, 1])["accepted"]
    line = np.c_[np.linspace(-0.01, 0.01, 15), np.zeros((15, 2))]
    assert (
        "insufficient_two_dimensional_coverage"
        in fit_patch(line, [0, 0, 0], [0, 0, 1])["rejected_for"]
    )
    points = grid()
    points[:, 0] += 0.006
    assert not fit_patch(points, [0, 0, 0], [0, 0, 1])["accepted"]


def test_noisy_or_disagreeing_surface_cannot_silently_repair_source_mesh():
    points = grid()
    points[:, 2] = points[:, 0]
    assert (
        "normal_disagrees_with_proposal"
        in fit_patch(points, [0, 0, 0], [0, 0, 1])["rejected_for"]
    )
    points = grid()
    points[:, 2] = np.random.default_rng(3).uniform(-0.003, 0.003, len(points))
    assert (
        "nonplanar_or_noisy_depth"
        in fit_patch(points, [0, 0, 0], [0, 0, 1])["rejected_for"]
    )
    with pytest.raises(ValueError):
        fit_patch(points, [0, 0, 0], [0, 0, 2])


def test_repeated_views_count_frames_and_keep_normal_disagreement():
    good = fit_patch(grid(), [0, 0, 0], [0, 0, 1])
    assert not normal_repeatability([good, good])["accepted"]
    assert normal_repeatability([good, good, good])["accepted"]
    tilted = dict(good, normal=[0.5, 0, np.sqrt(0.75)])
    assert not normal_repeatability([good, good, tilted])["accepted"]
    assert not normal_repeatability([dict(accepted=False)])["accepted"]
