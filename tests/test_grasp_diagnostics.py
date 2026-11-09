import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.controller_actor_training.grasp_diagnostics import palm_tracking,contact_intervals

def test_contact_gaps_are_not_reported_as_sustained_grasp():
    a=np.array([0,1,1,0,1],bool)
    rows=contact_intervals(a,fps=50)
    assert [r['frames'] for r in rows]==[2,1]
    assert rows[0]['duration_s']==.04

def test_object_motion_carries_target_without_creating_palm_error():
    obj=np.tile([0,0,0,0,0,0,1.],(5,1));palm=obj.copy();palm[:,0]=.1
    moved=obj.copy();moved[:,1]=2
    hand=palm.copy();hand[:,1]=2
    phase=np.ones(5,bool)
    result,arrays=palm_tracking(hand,palm,moved,obj,phase,phase,fps=50)
    assert result['median_position_error_m']==0
    assert result['longest_contact_s']==.1
    assert not result['grasp_success_established']

def test_metrics_are_invariant_to_common_rigid_world_transform():
    obj=np.tile([0,0,0,0,0,0,1.],(5,1));palm=obj.copy();palm[:,0]=.1
    actual=palm.copy();actual[:,0]+=.03;phase=np.ones(5,bool)
    a,_=palm_tracking(actual,palm,obj,obj,phase,phase,fps=50)
    rot=Rotation.from_euler('xyz',[.3,.8,.4]);offset=np.array([3.,-1.,2.])
    def move(p):return np.c_[rot.apply(p[:,:3])+offset,(rot*Rotation.from_quat(p[:,3:])).as_quat()]
    b,_=palm_tracking(*map(move,[actual,palm,obj,obj]),phase,phase,fps=50)
    assert b['median_position_error_m']==pytest.approx(a['median_position_error_m'])

def test_invalid_quaternion_is_rejected():
    p=np.zeros((3,7));phase=np.ones(3,bool)
    with pytest.raises(ValueError):palm_tracking(p,p,p,p,phase,phase,fps=50)

def test_phase_gap_does_not_split_actual_physical_contact():
    p=np.tile([0,0,0,0,0,0,1.],(5,1))
    phase=np.array([1,1,0,1,1],bool);contact=np.ones(5,bool)
    result,_=palm_tracking(p,p,p,p,phase,contact,fps=50)
    assert result['longest_contact_s']==.04
    assert result['longest_physical_contact_s']==.1
    assert len(result['physical_contact_intervals'])==1
