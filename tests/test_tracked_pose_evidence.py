import json
import numpy as np
import pytest
from real2sim2real.artifacts import digest
from real2sim2real.reconstruction.tracked_pose_evidence import recovery_flags


def evidence(tmp_path):
    pose=np.eye(4);pose[2,3]=3
    (tmp_path/'mapping.json').write_text(json.dumps(dict(source_video_sha256='video',source_frame_indices=[550,551,552])))
    states=[dict(frame=i,accepted=True,selected_pose=pose.tolist()) for i in [1,2]]
    updates=[dict(frame=i,method='tracked_update' if i==1 else 'local_recovery',recovery=i!=1,
        original=dict(rejected_for=[]),selected=dict(rejected_for=[]),proposed_pose_camera=pose.tolist()) for i in [1,2]]
    for name,rows in [('states',states),('updates',updates)]:(tmp_path/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    spec=dict(schema='real2sim2real.tracked-pose-evidence/1')
    for role,name in [('mapping','mapping.json'),('states','states.jsonl'),('updates','updates.jsonl')]:spec[role]=dict(path=name,sha256=digest(tmp_path/name))
    path=tmp_path/'evidence.json';path.write_text(json.dumps(spec))
    return path,np.tile(pose,(4,1,1))


def test_only_exact_unaltered_tracking_states_use_the_original_tracking_gate(tmp_path):
    path,poses=evidence(tmp_path)
    flags,receipt=recovery_flags(np.array([549,550,551,552]),poses,path,source_video_sha256='video')
    assert flags.tolist()==[True,True,False,True]
    assert receipt['tracking_gate_source_frame_ids']==[551]
    assert not receipt['shared_thresholds_changed']
    poses[2,0,3]+=.001
    flags,_=recovery_flags(np.array([549,550,551,552]),poses,path,source_video_sha256='video')
    assert flags.all()


def test_other_video_fails(tmp_path):
    path,poses=evidence(tmp_path)
    with pytest.raises(ValueError,match='another source video'):
        recovery_flags(np.array([549,550,551,552]),poses,path,source_video_sha256='other')


def test_modified_evidence_fails(tmp_path):
    path,poses=evidence(tmp_path);(tmp_path/'states.jsonl').write_text('')
    with pytest.raises(ValueError,match='member changed'):
        recovery_flags(np.array([549,550,551,552]),poses,path,source_video_sha256='video')


def test_backward_trace_keeps_exact_normal_and_recovery_modes(tmp_path):
    path,poses=evidence(tmp_path)
    mapping=tmp_path/'mapping.json'
    mapping.write_text(json.dumps(dict(source_video_sha256='video',source_frame_indices=[552,551,550])))
    spec=json.loads(path.read_text());spec['mapping']['sha256']=digest(mapping);path.write_text(json.dumps(spec))
    flags,receipt=recovery_flags(np.array([550,551,552]),poses[:3],path,source_video_sha256='video')
    assert flags.tolist()==[True,False,True]
    assert receipt['tracking_source_direction']=='backward' and not receipt['shared_thresholds_changed']


def test_backtracking_mapping_cannot_bounce_or_repeat_frames(tmp_path):
    path,poses=evidence(tmp_path);mapping=tmp_path/'mapping.json'
    mapping.write_text(json.dumps(dict(source_video_sha256='video',source_frame_indices=[550,551,550])))
    spec=json.loads(path.read_text());spec['mapping']['sha256']=digest(mapping);path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='monotone'):
        recovery_flags(np.array([550,551,552]),poses[:3],path,source_video_sha256='video')
