import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.reconstruction_repair.continuation import continuity,merge_continuation


def sample():
    p=np.tile(np.eye(4),(20,1,1));p[:,0,3]=np.arange(20)*.005;p[:,2,3]=3
    return p,np.arange(20)


def test_continuation_replaces_suffix_preserving_parent_anchor_and_prefix():
    p,ids=sample();p[10:,:3,:3]=Rotation.from_euler('z',127,degrees=True).as_matrix();p[10:,1,3]+=.15
    tail,_=sample();tail=tail[9:];assert not continuity(p,ids)['passed']
    merged,a=merge_continuation(p,ids,tail,ids[9:])
    assert a['passed'] and a['prefix_preserved_exactly'] and a['interpolated_frames']==0
    assert np.array_equal(merged[:10],p[:10]) and not a['full_reference_accepted']


@pytest.mark.parametrize('problem',['anchor','missing_tail','rotation_jump','gap','nan'])
def test_invalid_splices_are_rejected(problem):
    p,ids=sample();tail=p[9:].copy();tid=ids[9:].copy()
    if problem=='anchor':tail[0,0,3]+=.02
    if problem=='missing_tail':tail=tail[:-1];tid=tid[:-1]
    if problem=='rotation_jump':tail[1,:3,:3]=Rotation.from_euler('x',127,degrees=True).as_matrix()
    if problem=='gap':tid[2:]+=1
    if problem=='nan':tail[2,0,3]=np.nan
    with pytest.raises(ValueError):merge_continuation(p,ids,tail,tid)
