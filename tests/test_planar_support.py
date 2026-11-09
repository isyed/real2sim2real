import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.reconstruction_repair.planar_support import propose
from real2sim2real.reconstruction_repair.source_pose_audit import projection_audit


def scene():
    import trimesh
    mesh=trimesh.creation.box(extents=[.5,.2,.8]);v,f=mesh.vertices,mesh.faces
    plane=np.array([0.,-1.,0.,.5]);r=Rotation.from_euler('y',15,degrees=True).as_matrix()
    t=np.array([.04,.399,2.]);k=np.array([[150.,0.,100.],[0.,150.,100.],[0.,0.,1.]])
    from real2sim2real.reconstruction_repair.surface_depth import front_surface_depth
    mask=np.isfinite(front_surface_depth(v@r.T+t,f,k,(200,200)))
    p=np.eye(4);p[:3,:3]=r;p[:3,3]=t+[.02,-.03,.02]
    return v,f,plane,p,k,mask


def test_planar_fit_respects_support_and_preserves_object_normal():
    v,f,plane,p,k,mask=scene()
    r,t,receipt=propose(v,np.eye(3),p,k,mask,plane,resting_evidence='Reviewed floor-supported synthetic object')
    np.testing.assert_allclose(r.T@plane[:3],plane[:3],atol=1e-10)
    assert receipt['minimum_support_height_m']==pytest.approx(.001,abs=1e-9)
    measured,_=projection_audit(v@r.T+t,f,k,mask,np.full(mask.shape,2.))
    assert measured['full_silhouette']['iou']>.9
    assert not receipt['full_reference_approved'] and receipt['independent_native_depth_motion_checks_required']


def test_unsupported_height_and_missing_phase_cannot_be_silently_projected():
    v,f,plane,p,k,mask=scene();p[1,3]-=.5
    with pytest.raises(ValueError,match='uncertainty'):
        propose(v,np.eye(3),p,k,mask,plane,resting_evidence='Reviewed phase')
    with pytest.raises(ValueError):propose(v,np.eye(3),p,k,mask,plane,resting_evidence='')


def test_nonunit_plane_rejected():
    v,f,plane,p,k,mask=scene()
    with pytest.raises(ValueError):propose(v,np.eye(3),p,k,mask,plane*2,resting_evidence='Reviewed phase')


def test_opt_in_tilt_remains_bounded_and_in_contact():
    v,f,plane,p,k,mask=scene()
    r,t,receipt=propose(v,np.eye(3),p,k,mask,plane,resting_evidence='Reviewed contact',allow_tilt=True)
    assert receipt['tilt_change_deg']<=10+1e-8
    assert receipt['yaw_change_deg'] is None
    assert receipt['minimum_support_height_m']==pytest.approx(.001,abs=1e-8)
    assert not receipt['full_reference_approved']


def test_tilt_cannot_be_enabled_implicitly():
    v,f,plane,p,k,mask=scene()
    with pytest.raises(ValueError):propose(v,np.eye(3),p,k,mask,plane,resting_evidence='Reviewed contact',allow_tilt='yes')
