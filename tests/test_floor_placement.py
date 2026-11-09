import numpy as np
import pytest
from real2sim2real.controller_actor_training.support_check import plane_placement_metrics


def test_small_gap_is_placement_only_and_penetration_is_reported():
    p=np.array([[0,0,.011],[.1,0,.05]])
    result=plane_placement_metrics(p,[0,0,1,0])
    assert result['passed'] and result['minimum_surface_gap_m']==pytest.approx(.011)
    assert not result['physical_execution_established']
    result=plane_placement_metrics(p-[0,0,.04],[0,0,1,0])
    assert not result['passed'] and result['maximum_sampled_penetration_m']==pytest.approx(.029)


def test_tilted_floor_has_explicit_unit_normal_and_missing_scale_fails():
    n=np.array([0,1,1.])/np.sqrt(2);p=np.array([[0,.1,.1]])
    assert plane_placement_metrics(p,np.r_[n,-.14])['passed']
    with pytest.raises(ValueError):plane_placement_metrics(p,[0,1,1,0])
