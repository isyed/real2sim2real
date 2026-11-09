import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.reconstruction_repair.support_seed import local_rest_orientation,resting_orientation_hypotheses


def test_preserves_centre_and_later_motion_exactly():
    poses=np.tile(np.eye(4),(50,1,1));poses[:,:3,3]=np.arange(50)[:,None]*[.001,0,0]
    result,receipt=local_rest_orientation(poses,np.arange(50),Rotation.from_euler('x',8,degrees=True).as_matrix(),
        [.1,.02,.3],hold_end=10,fade_end=30,resting_evidence='Reviewed initial resting frames')
    assert np.array_equal(result[30:],poses[30:])
    assert receipt['maximum_center_motion_error_m']<1e-12
    assert receipt['changed_frame_ids']==list(range(30))
    assert not receipt['actor_training_approved'] and not receipt['geometry_modified']
    assert np.array_equal(poses[0],np.eye(4))


def test_requires_evidence_and_enforces_rotation_bound():
    p=np.tile(np.eye(4),(40,1,1));r=Rotation.from_euler('x',12,degrees=True).as_matrix()
    with pytest.raises(ValueError):local_rest_orientation(p,np.arange(40),r,[0,0,0],hold_end=10,fade_end=30,resting_evidence='reviewed')
    with pytest.raises(ValueError):local_rest_orientation(p,np.arange(40),np.eye(3),[0,0,0],hold_end=10,fade_end=30,resting_evidence='')


def test_resting_hypothesis_is_geometric_and_preserves_world_centre():
    vertices=np.array([[x,y,z] for x in [-.1,.1] for y in [-.1,.1] for z in [-.3,.3]])
    pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('x',8,degrees=True).as_matrix();pose[:3,3]=[0,0,.5]
    before=pose.copy()
    rows,report=resting_orientation_hypotheses(vertices,pose,[0,0,0],[0,0,1,-.1],resting_evidence='Reviewed resting object')
    assert rows and rows[0]['rotation_deg']==pytest.approx(8)
    assert np.allclose(rows[0]['pose'][:3,3],pose[:3,3])
    assert rows[0]['minimum_support_height_m']==pytest.approx(.1)
    assert np.array_equal(pose,before) and not report['physical_stability_established']


def test_resting_hypothesis_rejects_unphysical_centre():
    vertices=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    with pytest.raises(ValueError):resting_orientation_hypotheses(vertices,np.eye(4),[10,0,0],[0,0,1,0],resting_evidence='reviewed')
