import numpy as np
import pytest
from test_temporal_pose import fixture
from real2sim2real.reconstruction_repair.anchor_pair_review import rank


def test_heldout_does_not_choose_orientation_pair():
    data=fixture();first=data['poses_camera'][[0]].copy();last=data['poses_camera'][[-1]].copy()
    bad=last.copy();bad[0,0,3]+=.3;last=np.concatenate([last,bad])
    kw=dict(fps=30,fit_ids=[4,8,12],heldout_ids=[6,10],max_seconds=10)
    a=rank(data,first,last,**kw)
    data['object_mask'][[6,10]]=np.roll(data['object_mask'][[6,10]],30,axis=2)
    b=rank(data,first,last,**kw)
    assert [r['last_index'] for r in a['ranked']]==[r['last_index'] for r in b['ranked']]
    assert a['ranked'][0]['heldout_mean_iou']>b['ranked'][0]['heldout_mean_iou']
    assert not a['depth_validated'] and not a['full_reference_accepted']
    with pytest.raises(ValueError):rank(data,first,last,**dict(kw,heldout_ids=[4,6]))
