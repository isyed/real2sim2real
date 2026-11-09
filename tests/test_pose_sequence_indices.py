import numpy as np
import pytest
from real2sim2real.reconstruction.validate_pose_sequence import local_indices,source_support,sequence_boundary


def test_native_zero_omits_history_without_inventing_negative_source_frame():
    p=np.tile(np.eye(4),(3,1,1))
    poses,previous,missing=sequence_boundary(dict(poses_camera=p),np.arange(3))
    assert missing and np.array_equal(previous,p[0]) and np.array_equal(poses,p)
    with pytest.raises(ValueError):sequence_boundary(dict(poses_camera=p),np.arange(5,8))


def test_nonzero_sequence_requires_real_adjacent_boundary():
    p=np.tile(np.eye(4),(3,1,1));row=dict(poses_camera=p,previous_pose_camera=np.eye(4),previous_source_frame=4)
    assert sequence_boundary(row,np.arange(5,8))[2] is False
    for bad in [-1,3,4.0]:
        with pytest.raises(ValueError):sequence_boundary(dict(row,previous_source_frame=bad),np.arange(5,8))
    with pytest.raises(ValueError):sequence_boundary(dict(poses_camera=p,previous_pose_camera=np.eye(4)),np.arange(3))


def test_native_window_subset_is_explicitly_mapped_to_local_indices():
    np.testing.assert_array_equal(local_indices([653,654,655],list(range(650,660))),[3,4,5])


@pytest.mark.parametrize('ids,window',[
    ([0,1],list(range(650,660))),([650,652],list(range(650,660))),
    ([650,651],[650,652]),([650.0,651.0],list(range(650,660))),([],list(range(650,660))),
])
def test_implicit_or_gapped_source_identity_is_rejected(ids,window):
    with pytest.raises(ValueError):local_indices(ids,window)


def test_support_uses_native_frame_identity_and_keeps_constant_support_compatible():
    base=dict(ground_truth_used=False,evidence='own predicted planes')
    first=dict(source_frame=308,plane_camera=[0,1,0,-1],center_camera_m=[0,1,0],radius_m=10)
    second=dict(first,source_frame=309,plane_camera=[0,1,0,-1.03])
    support=dict(base,frames=[first,second])
    assert source_support(support,309)==second
    assert source_support(dict(base,**first),309)['plane_camera']==first['plane_camera']
    assert support['frames'][0]==first
    with pytest.raises(ValueError):source_support(support,0)
    with pytest.raises(ValueError):source_support(support,310)


@pytest.mark.parametrize('ids',[[308,308],[309,308],[308.,309],[-1,0]])
def test_ambiguous_support_source_records_fail(ids):
    support=dict(ground_truth_used=False,evidence='predicted',frames=[dict(source_frame=i) for i in ids])
    with pytest.raises(ValueError):source_support(support,308)
