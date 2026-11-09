import numpy as np
import pytest
from real2sim2real.reconstruction_repair.foreground_depth import spatial_floor_holdout


def check(z=None,excluded=None,rectangles=None):
    return spatial_floor_holdout(np.full((80,100),2.) if z is None else z,
        np.array([[100.,0,50],[0,100,40],[0,0,1]]),[0,0,1,-2],
        [[60,10,90,60]] if rectangles is None else rectangles,
        np.zeros((80,100),bool) if excluded is None else excluded,
        fit_rectangles=[[0,0,20,40],[25,0,45,40]])


def test_good_spatial_holdout_does_not_claim_metric_truth():
    r=check()
    assert r['accepted'] and r['regions'][0]['p90_residual_m']==0
    assert not r['scale_refitted'] and not r['metric_truth_known']
    assert not r['pixels_are_independent_samples']


def test_disagreement_outside_fit_rejected_without_refitting():
    z=np.full((80,100),2.);z[10:60,60:90]=2.2;before=z.copy()
    r=check(z)
    assert not r['accepted'] and 'spatial_holdout_residual_0' in r['reasons']
    np.testing.assert_array_equal(z,before)


def test_calibration_overlap_rejected():
    with pytest.raises(ValueError,match='overlap'):check(rectangles=[[10,10,70,60]])


def test_excluded_foreground_cannot_validate_floor():
    r=check(excluded=np.ones((80,100),bool))
    assert not r['accepted'] and r['regions'][0]['pixels']==0
    assert r['regions'][0]['p90_residual_m'] is None


@pytest.mark.parametrize('rect',[[-1,10,90,60],[60,10,101,60],[60.,10,90,60]])
def test_invalid_rectangle_rejected(rect):
    with pytest.raises(ValueError,match='integer rectangle'):check(rectangles=[rect])


def test_invalid_depth_rejected():
    z=np.full((80,100),2.);z[0,0]=float('nan')
    with pytest.raises(ValueError,match='Finite depth'):check(z)
