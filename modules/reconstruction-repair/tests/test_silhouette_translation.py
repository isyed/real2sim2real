import numpy as np
import pytest
from real2sim2real.reconstruction_repair.silhouette_translation import propose,projected_box


def example():
    vertices=np.array([[x,y,z] for x in (-.2,.2) for y in (-.1,.1) for z in (-.1,.1)])
    return vertices,np.eye(3),np.array([.1,.05,3.]),np.array([[600.,0,320],[0,600,240],[0,0,1]])


def test_recovers_translation_without_changing_mesh_rotation_or_claiming_acceptance():
    v,R,t,K=example();target=t+np.array([.04,-.03,.1]);box=projected_box(v,R,target,K)
    original=(v.copy(),R.copy(),t.copy())
    candidate,report=propose(v,R,t,K,box)
    assert np.allclose(candidate,target,atol=1e-5)
    assert report['bounding_box_error_after_px']<report['bounding_box_error_before_px']*.01
    assert report['within_translation_bound'] and not report['reference_accepted']
    assert not report['depth_used_for_fitting'] and report['full_silhouette_validation_required']
    assert all(np.array_equal(a,b) for a,b in zip((v,R,t),original))


def test_partial_occlusion_box_is_only_a_proposal_even_if_it_fits():
    v,R,t,K=example();box=projected_box(v,R,t,K);box[0]+=10
    _,report=propose(v,R,t,K,box)
    assert report['proposal_only'] and report['independent_depth_validation_required']
    assert not report['reference_accepted'] and not report['physics_validated']


def test_rejects_invalid_image_and_camera_geometry():
    v,R,t,K=example()
    with pytest.raises(ValueError):propose(v,R,t,K,[10,10,5,5])
    with pytest.raises(ValueError):propose(v,R,[0,0,0],K,[10,10,20,20])
    with pytest.raises(ValueError):propose(v,R,t,K,[10,10,20,20],max_shift_m=True)


def test_declared_camera_depth_is_not_changed_by_box_size_disagreement():
    v,R,t,K=example();box=projected_box(v,R,t+np.array([.03,0,.15]),K)
    candidate,report=propose(v,R,t,K,box,preserve_camera_depth=True)
    assert candidate[2]==t[2] and report['camera_depth_preserved']
    assert not report['reference_accepted']
