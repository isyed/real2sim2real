import numpy as np
import pytest
from real2sim2real.reconstruction_repair.localized_pose import merge_windows


def fixture():
    poses=np.tile(np.eye(4),(21,1,1));poses[:,:,3][:,:3]=[0.,0.,2.]
    poses[10,0,3]=.10
    window=poses[7:14].copy();window[3,0,3]=.02
    return poses,np.arange(21),dict(poses=window,frame_ids=np.arange(7,14),decision=dict(accepted=True))


def test_local_repair_preserves_untouched_poses_and_checks_all_seams():
    parent,ids,window=fixture();candidate,audit=merge_windows(parent,ids,[window],preserve_rotations=True)
    assert audit['passed'] and audit['untouched_frames_preserved_exactly']
    assert np.array_equal(candidate[:7],parent[:7]) and np.array_equal(candidate[14:],parent[14:])
    assert np.array_equal(parent[10,:3,3],[.10,0.,2.])
    assert not audit['approved_for_use'] and not audit['full_reference_accepted']


@pytest.mark.parametrize('problem',['rejected','anchor','rotation','frame_ids','overlap','outside_jump'])
def test_invalid_window_assembly_stops(problem):
    parent,ids,w=fixture();windows=[w]
    if problem=='rejected':w['decision']['accepted']=False
    if problem=='anchor':w['poses'][0,0,3]+=.001
    if problem=='rotation':w['poses'][2,:3,:3]=np.diag([-1.,-1.,1.])
    if problem=='frame_ids':w['frame_ids']+=30
    if problem=='overlap':windows=[w,w]
    if problem=='outside_jump':parent[18,0,3]=.15
    with pytest.raises(ValueError):merge_windows(parent,ids,windows,preserve_rotations=True)
