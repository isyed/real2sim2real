import numpy as np
import pytest

from real2sim2real.reconstruction_repair.scale_depth_audit import audit


def sample(frame=0, hypothesis='predicted', observed=None):
    return dict(frame_id=frame, hypothesis=hypothesis,
                rendered_depth_m=np.full(100, 2.),
                observed_depth_m=np.full(100, 2.) if observed is None else observed,
                provenance={'model': 'synthetic-predictor', 'metric_truth': False})


def test_common_scale_is_diagnostic_not_repair():
    item = sample(observed=np.full(100, 1.8))
    original = item['observed_depth_m'].copy()
    result = audit([item], [.8, .9, 1., 1.1], error_limit_m=.01)
    assert result['common_passing_scales'] == [.9]
    assert not result['candidate_approved']
    assert not result['independent_metric_evidence_established']
    np.testing.assert_array_equal(item['observed_depth_m'], original)


def test_conflicting_frames_are_not_averaged_or_dropped():
    result = audit([sample(0, observed=np.full(100, 1.6)),
                    sample(1, observed=np.full(100, 2.4))], [.8, .9, 1., 1.1, 1.2])
    assert result['common_passing_scales'] == []
    assert result['minimax_diagnostic']['worst_error_m'] == pytest.approx(.4)


def test_hypotheses_with_same_model_remain_separate():
    result = audit([sample(hypothesis='full', observed=np.full(100, 1.6)),
                    sample(hypothesis='crop', observed=np.full(100, 2.4))], [.8, 1., 1.2])
    assert not result['common_passing_scales']
    assert len(result['rows']) == 2


def test_absolute_not_signed_median_and_even_sample_count():
    x = np.r_[np.full(50, 1.8), np.full(50, 2.2)]
    result = audit([sample(observed=x)], [1., 1.1])
    assert result['rows'][0]['error_m'][0] == pytest.approx(.2)


@pytest.mark.parametrize('grid', [[.8, .9], [1., .9], [1., 1.], [0., 1.], [1., np.nan]])
def test_grid_contract(grid):
    with pytest.raises(ValueError): audit([sample()], grid)


def test_missing_evidence_and_duplicate_identity_rejected():
    with pytest.raises(ValueError): audit([], [.9, 1.])
    with pytest.raises(ValueError): audit([sample(), sample()], [.9, 1.])
    item = sample(); item['provenance'] = {}
    with pytest.raises(ValueError): audit([item], [.9, 1.])
    item = sample(); item['observed_depth_m'] = np.zeros(100)
    with pytest.raises(ValueError): audit([item], [.9, 1.])


def test_scale_keeps_perspective_but_changes_physical_geometry():
    points = np.array([[.1, .2, 2.], [-.2, .5, 3.]])
    scaled = points * .8
    np.testing.assert_allclose(points[:, :2] / points[:, 2:], scaled[:, :2] / scaled[:, 2:])
    assert not np.allclose(points, scaled)
