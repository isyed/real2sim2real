import numpy as np
import pytest
from real2sim2real.reconstruction.recovery_quality import depth_residual_summary

def test_mixed_signs_do_not_hide_large_absolute_error():
    x=np.array([-.2,-.12,.03,.11,.13]);r=depth_residual_summary(x)
    assert r['depth_shift_m']==pytest.approx(.03)
    assert r['depth_error_m']==pytest.approx(.12)
    assert r['depth_positive_fraction']==pytest.approx(.6)
    assert r['depth_negative_fraction']==pytest.approx(.4)
    assert r['depth_residual_quantiles_m'][2]==r['depth_shift_m']

def test_empty_evidence_is_unknown_not_a_zero_error_pass():
    r=depth_residual_summary([])
    assert r['depth_points']==0 and r['depth_error_m'] is None
    assert r['depth_shift_m'] is None and r['depth_residual_quantiles_m'] is None

@pytest.mark.parametrize('x',[[float('nan')],[float('inf')],[[.1,.2]]])
def test_invalid_residuals_rejected(x):
    with pytest.raises(ValueError):depth_residual_summary(x)
