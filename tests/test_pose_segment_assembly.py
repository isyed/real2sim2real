import numpy as np
import pytest
from real2sim2real.reconstruction_repair.pose_segments import merge

def poses(n=8):
    p=np.tile(np.eye(4),(n,1,1));p[:,0,3]=np.arange(n)*.01;return p

def segment(ids,p):return dict(name='checked',source_frame_ids=ids,poses_camera=p,evidence_sha256='a'*64,native_validation_passed=True)

def test_backward_segment_recovers_gap_without_overwriting_baseline():
    base=poses();a=np.array([True,True,False,False,False,False,True,True])
    p,valid,r=merge(base,a,[segment([5,4,3,2],base[[5,4,3,2]])])
    np.testing.assert_array_equal(p,base);assert valid.all() and r['new_frames']==4
    assert r['baseline_preserved_exactly'] and not r['physical_reference_approved']

def test_join_conflicts_drop_new_boundary_not_trusted_poses():
    base=poses(6);a=np.array([True,True,False,False,True,True]);offered=base[[2,3]].copy();offered[:,0,3]+=.5
    p,valid,r=merge(base,a,[segment([2,3],offered)])
    assert valid.tolist()==a.tolist() and len(r['rejected_joins'])==2
    np.testing.assert_array_equal(p[a],base[a])

def test_unavailable_slots_are_never_interpolated():
    base=poses();a=np.array([True,True,False,False,False,False,False,False])
    _,valid,r=merge(base,a,[segment([6,7],base[[6,7]])])
    assert valid.tolist()==[True,True,False,False,False,False,True,True]
    assert not r['gaps_filled']

def test_overlap_keeps_earlier_checked_baseline():
    base=poses();a=np.ones(8,bool);other=base[[2,3]].copy();other[:,0,3]+=.1
    p,valid,r=merge(base,a,[segment([2,3],other)])
    np.testing.assert_array_equal(p,base);assert r['segments'][0]['overlap_preserved']==2

def test_unsupported_sparse_native_segment_rejected():
    base=poses();a=np.zeros(8,bool)
    with pytest.raises(ValueError,match='Contiguous'):merge(base,a,[segment([2,4],base[[2,4]])])

def test_claimed_valid_segment_with_bad_internal_motion_rejected():
    base=poses();a=np.zeros(8,bool);p=base[[2,3]].copy();p[1,0,3]=.5
    with pytest.raises(ValueError,match='itself'):merge(base,a,[segment([2,3],p)])
