import numpy as np
import pytest
from real2sim2real.reconstruction.recovery_quality import pixel_translation_proposals,rejection_reasons


def test_bounded_pixel_grid_preserves_depth_rotation_and_camera_skew():
    pose=np.eye(4);pose[:3,3]=[.1,.2,2.]
    K=np.array([[800.,30.,500.],[0.,900.,400.],[0.,0.,1.]])
    q=pixel_translation_proposals(pose,K,radius_px=1.)
    assert q.shape==(25,4,4) and np.array_equal(q[12],pose)
    assert np.array_equal(q[:,:3,:3],np.repeat(pose[None,:3,:3],25,0))
    assert np.array_equal(q[:,2,3],np.repeat(pose[2,3],25))
    projected=q[:,:3,3]@K.T;xy=projected[:,:2]/projected[:,2,None]
    center=K@pose[:3,3];shift=xy-center[:2]/center[2]
    assert np.allclose(shift[0],[-1,-1]) and np.allclose(shift[-1],[1,1])
    # Proposals confer no temporal acceptance: a distant prior still vetoes them.
    previous=pose.copy();previous[0,3]-=.2
    metrics=dict(iou=.9,precision=.9,recall=.9);geometry=dict(depth_points=200,depth_error_m=.01,support_penetration_m=0.,behind_camera=False)
    assert 'translation_step' in rejection_reasons(q[12],previous,metrics,geometry,recovery=True)


@pytest.mark.parametrize('radius',[0.,-1.,3.,float('nan'),True])
def test_invalid_pixel_budget_is_rejected(radius):
    pose=np.eye(4);pose[2,3]=2
    with pytest.raises(ValueError):pixel_translation_proposals(pose,np.eye(3),radius_px=radius)
