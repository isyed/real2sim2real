import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.controller_actor_training.reach_audit import summarize


def states():
    p=np.zeros((3,3,7));p[:,:,6]=1;p[:,1,0]=1;p[:,0,0]=1.3
    k=p.copy();p[:,1,0]=2;p[:,0,0]=2.4
    return p,k


def run(p,k):return summarize(p,k,palm_body_ids=[0],object_body_id=1,pelvis_body_id=2,contact_active=np.ones((3,1),bool))


def test_grasp_target_moves_with_actual_object_and_common_world_transform():
    p,k=states();r=run(p,k);assert r['median_target_error_m']==pytest.approx(.1)
    R=Rotation.from_euler('xyz',[.2,-.3,.7]);t=np.array([3,-2,.4])
    for x in (p,k):
        x[:,:,:3]=R.apply(x[:,:,:3].reshape(-1,3)).reshape(3,3,3)+t
        x[:,:,3:]=(R*Rotation.from_quat(x[:,:,3:].reshape(-1,4))).as_quat().reshape(3,3,4)
    assert run(p,k)['median_target_error_m']==pytest.approx(.1)
    assert not r['grasp_success_established']


def test_invalid_body_ids_and_float_phase_labels_fail_explicitly():
    p,k=states()
    with pytest.raises(ValueError,match='body IDs'):summarize(p,k,palm_body_ids=[-1],object_body_id=1,pelvis_body_id=2,contact_active=np.ones((3,1),bool))
    with pytest.raises(ValueError,match='boolean'):summarize(p,k,palm_body_ids=[0],object_body_id=1,pelvis_body_id=2,contact_active=np.ones((3,1)))
