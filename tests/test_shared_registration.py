import numpy as np
from scipy.spatial.transform import Rotation
from real2sim2real.egocentric.shared_registration import fit_shared_registration,fit_motion_from_flow,flow_errors
from real2sim2real.egocentric.temporal_alignment import project


def test_shared_static_correction_transfers_to_unused_view():
    rng=np.random.default_rng(1);k=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    truth=np.eye(4);truth[:3,:3]=Rotation.from_euler('y',.4,degrees=True).as_matrix();truth[:3,3]=[.003,-.002,.008]
    views=[]
    for _ in range(2):
        p=rng.uniform([-.3,-.2,.6],[.3,.2,3],(60,3));views.append(dict(points=p,pixels=project(p,truth,k)))
    t,report=fit_shared_registration(views,np.eye(4),k)
    held=rng.uniform([-.3,-.2,.6],[.3,.2,3],(40,3))
    assert report['optimizer_success'] and not report['bound_hit']
    assert np.max(np.linalg.norm(project(held,t,k)-project(held,truth,k),axis=1))<.01


def test_zero_flow_does_not_refit_fixed_registration():
    p=np.random.default_rng(2).uniform([-.3,-.2,.6],[.3,.2,3],(50,3));k=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    # An absolute 7px registration discrepancy must not be absorbed into motion.
    anchor=project(p,np.eye(4),k)+[7.,0]
    t,report=fit_motion_from_flow(p,anchor,anchor,k)
    assert report['optimizer_success']
    assert np.allclose(t,np.eye(4),atol=1e-10)


def test_flow_only_motion_recovers_shift_with_anchor_bias():
    p=np.random.default_rng(2).uniform([-.3,-.2,.6],[.3,.2,3],(80,3));k=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    truth=np.eye(4);truth[:3,3]=[.01,0,.005]
    base=project(p,np.eye(4),k);anchor=base+[7.,2.];target=anchor+project(p,truth,k)-base
    t,_=fit_motion_from_flow(p[:50],anchor[:50],target[:50],k)
    assert np.max(flow_errors(p[50:],anchor[50:],target[50:],k,t))<.01
