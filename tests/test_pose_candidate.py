import numpy as np
import pytest
from real2sim2real.reconstruction_repair.pose_candidate import compose


def geometry():
    return dict(frame_ids=np.arange(6),object_rotation=np.tile(np.eye(3),(6,1,1)),
        object_translation=np.tile([0.,0.,2.],(6,1)),vertices=np.ones((4,3)))


def segment(ids,translation,**evidence):
    poses=np.tile(np.eye(4),(len(ids),1,1));poses[:,2,3]=2.;poses[:,0,3]=translation
    return dict(source_frame_ids=np.array(ids),poses_camera=poses,
        evidence=dict(name='measured_interval',source_sha256='hash',**evidence))


def test_warned_interval_requires_explicit_decision_and_preserves_failed_gate():
    g=geometry();s=segment([2,3],.01,failed_gates=['image_preservation'])
    with pytest.raises(ValueError,match='warning acceptance'):compose(g,[s])
    s['evidence']['explicit_warning_acceptance']=True
    after,r=compose(g,[s]);assert r['sources'][1]['failed_gates']==['image_preservation']
    assert r['motion_bounds_passed'] and not r['physics_validated']
    np.testing.assert_array_equal(g['object_translation'][:,0],0)
    np.testing.assert_array_equal(after['vertices'],g['vertices'])


def test_individually_smooth_interval_cannot_hide_bad_joins():
    _,r=compose(geometry(),[segment([2,3],.3)])
    assert r['failed_motion_frame_ids']==[2,4] and not r['motion_bounds_passed']
    assert r['failed_join_frame_ids']==[2,4] and not r['joins_passed']
    assert r['parent_fallback_frames']==4 and r['replacement_frames']==2
    assert r['joins'][0]['previous_source']=='complete_parent'
    assert r['joins'][0]['next_source']=='measured_interval'
    assert r['joins'][1]['previous_source']=='measured_interval'
    assert r['joins'][1]['next_source']=='complete_parent'


def test_source_ids_cannot_be_missing_duplicated_or_outside_native_timeline():
    for ids in ([2,4],[2,2],[5,6]):
        with pytest.raises(ValueError):compose(geometry(),[segment(ids,.01)])


def test_composition_changes_only_declared_frames_without_interpolation():
    g=geometry();after,r=compose(g,[segment([2,3],.02)])
    np.testing.assert_array_equal(after['object_translation'][[0,1,4,5]],g['object_translation'][[0,1,4,5]])
    assert r['source_owner_indices']==[0,0,1,1,0,0] and not r['replacement_interpolation']
    assert r['joins_passed'] and r['failed_join_frame_ids']==[]


def test_parent_failures_stay_separate_from_new_join_failures():
    g=geometry();g['object_translation'][5,0]=.2
    _,r=compose(g,[segment([2,3],.01)])
    assert r['joins_passed'] and not r['motion_bounds_passed']
    assert r['failed_motion_frame_ids']==[5] and r['failed_join_frame_ids']==[]
    assert not r['complete_reference_approved']


def test_packaged_candidate_preserves_paired_human_and_separates_parent_quality(tmp_path):
    from real2sim2real.artifacts import publish,load,digest
    from real2sim2real.reconstruction_repair.pose_candidate import prepare
    parent=tmp_path/'parent';parent.mkdir();g=geometry()
    np.savez(parent/'geometry.npz',**g);np.savez(parent/'human.npz',pose=np.ones((6,3)))
    (parent/'quality.json').write_text('{"scope":"parent only"}')
    publish(parent,'reconstruction',{k:parent/p for k,p in [('geometry','geometry.npz'),('human','human.npz'),('quality','quality.json')]},metadata={})
    after,_=compose(g,[segment([2,3],.01)]);np.savez(tmp_path/'candidate.npz',**after)
    evidence=dict(clean_motion_used=False,decision='Explicit nomination with warnings',
        parent_artifact_sha256=digest(parent/'artifact.json'),geometry_sha256=digest(tmp_path/'candidate.npz'))
    prepare(parent,tmp_path/'candidate.npz',tmp_path/'out',evidence=evidence)
    artifact=load(tmp_path/'out',kinds={'reconstruction'})
    assert artifact['files']['human']['sha256']==digest(parent/'human.npz')
    assert 'quality' not in artifact['files'] and 'parent_quality' in artifact['files']
    assert not artifact['metadata']['physical_execution_validated']


def test_packaging_cannot_silently_change_paired_geometry(tmp_path):
    from real2sim2real.artifacts import publish,digest
    from real2sim2real.reconstruction_repair.pose_candidate import prepare
    parent=tmp_path/'parent';parent.mkdir();g=geometry();np.savez(parent/'geometry.npz',**g)
    publish(parent,'reconstruction',{'geometry':parent/'geometry.npz'},metadata={})
    g['vertices'][0,0]+=.1;np.savez(tmp_path/'candidate.npz',**g)
    with pytest.raises(ValueError,match='immutable'):
        prepare(parent,tmp_path/'candidate.npz',tmp_path/'out',evidence=dict(clean_motion_used=False,
            decision='Explicit nomination',parent_artifact_sha256=digest(parent/'artifact.json'),geometry_sha256=digest(tmp_path/'candidate.npz')))
    assert not (tmp_path/'out').exists()
