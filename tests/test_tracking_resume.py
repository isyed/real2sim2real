import json
import numpy as np
import pytest
from real2sim2real.artifacts import digest
from real2sim2real.reconstruction_repair.tracking_resume import verified_prefix


def fixture(tmp_path):
    p=np.eye(4);q=p.copy();q[0,3]=.01
    seed=tmp_path/'seed.json';seed.write_text(json.dumps(dict(frame=0,pose_cam=p.tolist())))
    trace=tmp_path/'states.jsonl';rows=[dict(frame=1,accepted=True,previous_accepted_pose=p.tolist(),selected_pose=q.tolist()),
        dict(frame=2,accepted=False,previous_accepted_pose=q.tolist())]
    trace.write_text('\n'.join(map(json.dumps,rows)))
    return seed,trace,rows


def test_resume_retains_exact_prefix_and_never_fills_rejected_tail(tmp_path):
    seed,trace,rows=fixture(tmp_path)
    poses,receipt=verified_prefix(seed,trace,seed_sha256=digest(seed),trace_sha256=digest(trace),source_frames=10)
    assert poses.shape==(2,4,4) and np.array_equal(poses[1],rows[0]['selected_pose'])
    assert receipt['first_rejected_frame']==2 and receipt['last_source_frame']==1
    assert not receipt['poses_interpolated'] and not receipt['full_reference_accepted']


@pytest.mark.parametrize('bad',['hash','frame','previous','later_accept'])
def test_resume_rejects_tampering_and_broken_state_chains(tmp_path,bad):
    seed,trace,rows=fixture(tmp_path);expected=digest(trace)
    if bad=='frame':rows[0]['frame']=3
    if bad=='previous':rows[0]['previous_accepted_pose'][0][3]=1
    if bad=='later_accept':rows.append(dict(rows[0],frame=3))
    if bad!='hash':trace.write_text('\n'.join(map(json.dumps,rows)));expected=digest(trace)
    else:trace.write_text(trace.read_text()+'\n')
    with pytest.raises(ValueError):verified_prefix(seed,trace,seed_sha256=digest(seed),trace_sha256=expected,source_frames=10)


@pytest.mark.parametrize('ids',[[10,11,12,13,14,15,16,17,18,19],[20,19,18,17,16,15,14,13,12,11]])
def test_window_mapping_keeps_native_source_identity(tmp_path,ids):
    seed,trace,_=fixture(tmp_path)
    _,receipt=verified_prefix(seed,trace,seed_sha256=digest(seed),trace_sha256=digest(trace),source_frames=10,source_frame_ids=ids)
    assert receipt['source_frame_ids']==ids[:2] and receipt['last_source_frame']==ids[1]
    assert receipt['last_tracker_frame']==1 and receipt['first_rejected_source_frame']==ids[2]


def test_window_mapping_rejects_unexplained_frame_gap(tmp_path):
    seed,trace,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='source IDs'):
        verified_prefix(seed,trace,seed_sha256=digest(seed),trace_sha256=digest(trace),source_frames=10,source_frame_ids=[0,1,3,4,5,6,7,8,9,10])
