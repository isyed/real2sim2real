import numpy as np
import pytest
from real2sim2real.reconstruction_repair.support_observation import support_pixel_seed


def test_contact_lands_on_observed_ray_and_plane_without_changing_rotation():
    K=np.array([[400.,0,320],[0,400,240],[0,0,1]])
    r=np.eye(3);p=np.array([0.,.2,0.]);pose,receipt=support_pixel_seed(r,p,[320,280],K,[0,-1,0,.4],resting_evidence='Visible base on reviewed table')
    point=pose[:3,:3]@p+pose[:3,3]
    assert point[1]==pytest.approx(.4)
    assert np.allclose((K@point)[:2]/point[2],[320,280])
    assert np.array_equal(pose[:3,:3],r) and np.allclose(pose[:3,3],[0,.2,4])
    assert receipt['proposal_only'] and not receipt['contact_assumption_inferred']
    shifted,_=support_pixel_seed(r,p,[320,280],K,[0,-1,0,.401],resting_evidence='Same visible base; plane-offset sensitivity test')
    assert receipt['depth_sensitivity_to_plane_offset']==pytest.approx(10.)
    assert abs(shifted[2,3]-pose[2,3])/.001==pytest.approx(receipt['depth_sensitivity_to_plane_offset'])


@pytest.mark.parametrize('pixel,plane,evidence',[
    ([320,240],[0,-1,0,.4],'reviewed'),([320,200],[0,-1,0,.4],'reviewed'),
    ([320,280],[0,-2,0,.4],'reviewed'),([320,280],[0,-1,0,.4],''),
])
def test_ambiguous_or_unsupported_contact_is_rejected(pixel,plane,evidence):
    with pytest.raises(ValueError):support_pixel_seed(np.eye(3),[0,.2,0],pixel,np.array([[400.,0,320],[0,400,240],[0,0,1]]),plane,resting_evidence=evidence)
