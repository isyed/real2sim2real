import numpy as np
import pytest
from real2sim2real.egocentric.tapered_container import outer_distance, surface, outline_samples


def test_open_mouth_does_not_become_solid_surface():
    dimensions = [.04,.06,.2]
    assert np.max(outer_distance(surface(*dimensions), dimensions)) < 1e-10
    assert outer_distance(np.array([[0,0,.2]]),dimensions)[0] > .05
    assert outer_distance(np.array([[0,0,0]]),dimensions)[0] == 0


def test_outline_sampling_is_uniform_in_arc_length():
    p=outline_samples([[0,0],[2,0],[2,1],[0,1]],6)
    assert np.allclose(p,[[0,0],[1,0],[2,0],[2,1],[1,1],[0,1]])


def test_negative_dimensions_rejected():
    with pytest.raises(ValueError):
        surface(.04,.06,-.2)


def test_resting_fit_recovers_synthetic_metric_dimensions():
    from scipy.spatial.transform import Rotation
    from real2sim2real.egocentric.tapered_container import fit_resting, projected_outline
    dimensions=[.04,.055,.18]
    p=surface(*dimensions)+[.1,.05,0]
    camera=np.eye(4)
    camera[:3,:3]=Rotation.from_euler('x',150,degrees=True).as_matrix()
    camera[:3,3]=[0,0,.7]
    k=np.array([[600.,0,640],[0,600,400],[0,0,1]])
    outline=projected_outline(p,camera,k)
    result=fit_resting(p[::3],outline,camera,k)
    assert result['optimizer_success']
    assert np.allclose(result['dimensions_m'],dimensions,atol=.002)
    assert np.allclose(result['center_xy_m'],[.1,.05],atol=.002)
