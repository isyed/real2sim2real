import numpy as np
import pytest
from real2sim2real.retargeting.support_rim import measured_thickness

def test_visible_rim_thickness_is_translation_invariant():
    p=np.c_[np.linspace(-.2,.2,100),np.zeros(100),np.linspace(.77,.79,100)]
    a=measured_thickness(p,[0,0,1,-.8])
    b=measured_thickness(p+[2,3,4],[0,0,1,-4.8])
    assert a['thickness_m']==pytest.approx(.028)
    assert b['thickness_m']==pytest.approx(a['thickness_m'])

def test_top_noise_or_above_plane_is_not_rim_evidence():
    for z in [.8001,.7999,.6]:
        with pytest.raises(ValueError):measured_thickness(np.tile([0,0,z],(30,1)),[0,0,1,-.8])

def test_invalid_inputs_fail():
    with pytest.raises(ValueError):measured_thickness(np.zeros((30,3)),[0,0,2,-.8])
    with pytest.raises(ValueError):measured_thickness(np.full((30,3),np.nan),[0,0,1,-.8])
