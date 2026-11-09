import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.reconstruction_repair.pose_bounds import project_rotation_bound


def sequence():
    p=np.tile(np.eye(4),(3,1,1));p[:,:3,:3]=Rotation.from_euler('xyz',[[20,30,40],[-20,4,8],[1,2,3]],degrees=True).as_matrix()
    return p


def test_geodesic_bound_in_a_rotated_frame_preserves_translation_and_valid_proposals():
    original=sequence();proposal=original.copy()
    proposal[:,:3,:3]=(Rotation.from_rotvec(np.array([[1,1,1],[0,0,.1],[0,0,0]]))*Rotation.from_matrix(original[:,:3,:3])).as_matrix()
    proposal[:,:3,3]=[[1,2,3],[4,5,6],[7,8,9]]
    result,report=project_rotation_bound(original,proposal,maximum_degrees=20.)
    delta=np.rad2deg((Rotation.from_matrix(result[:,:3,:3])*Rotation.from_matrix(original[:,:3,:3]).inv()).magnitude())
    assert delta[0]==pytest.approx(20.)
    np.testing.assert_array_equal(result[:,:3,3],proposal[:,:3,3])
    np.testing.assert_array_equal(result[1:],proposal[1:])
    assert report['changed_indices']==[0] and report['requires_full_revalidation']
    assert not report['proposal_accepted']


@pytest.mark.parametrize('bound',[True,0,-1,float('nan'),float('inf'),181])
def test_invalid_limits_fail(bound):
    with pytest.raises(ValueError):project_rotation_bound(sequence(),sequence(),maximum_degrees=bound)


def test_invalid_rotation_fails_without_silently_normalizing():
    a=sequence();b=a.copy();b[0,0,0]+=1
    with pytest.raises(ValueError):project_rotation_bound(a,b,maximum_degrees=20)
