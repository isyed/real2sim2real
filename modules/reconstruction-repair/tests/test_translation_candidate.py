import numpy as np
import pytest
from real2sim2real.artifacts import publish,digest,load
from real2sim2real.reconstruction_repair.translation_candidate import prepare


def inputs(tmp_path):
    source=tmp_path/'parent';source.mkdir()
    g=dict(frame_ids=np.arange(3),timestamps_s=np.arange(3)/30,
        object_translation=np.zeros((3,3)),object_rotation=np.tile(np.eye(3),(3,1,1)),
        vertices=np.eye(3),faces=np.array([[0,1,2]]),object_scale=np.array(1.))
    np.savez(source/'geometry.npz',**g);np.savez(source/'human.npz',prediction=np.ones((3,2)))
    (source/'quality.json').write_text('{"mean_iou":0.9}')
    publish(source,'reconstruction',{k:source/v for k,v in dict(geometry='geometry.npz',human='human.npz',quality='quality.json').items()},metadata=dict(sequence_id='synthetic'))
    g['object_translation'][1,0]=.01;candidate=tmp_path/'candidate.npz';np.savez(candidate,**g)
    evidence=dict(diagnostic_only=True,clean_motion_used=False,parent_artifact_sha256=digest(source/'artifact.json'),geometry_sha256=digest(candidate))
    return source,candidate,evidence,g


def test_full_diagnostic_candidate_preserves_source_and_does_not_inherit_quality(tmp_path):
    source,candidate,evidence,g=inputs(tmp_path);old=digest(source/'artifact.json')
    result=prepare(source,candidate,tmp_path/'output',evidence=evidence,hardlink=True)
    artifact=load(tmp_path/'output');assert digest(source/'artifact.json')==old
    assert result['changed_frame_ids']==[1] and not result['full_reference_approved']
    assert 'quality' not in artifact['files'] and 'parent_quality' in artifact['files']
    assert artifact['files']['human']['sha256']==load(source)['files']['human']['sha256']


@pytest.mark.parametrize('field',['vertices','timestamps_s','object_rotation','frame_ids'])
def test_forbids_mesh_timeline_or_rotation_edits(tmp_path,field):
    source,candidate,evidence,g=inputs(tmp_path);g[field].flat[0]+=1;np.savez(candidate,**g)
    evidence['geometry_sha256']=digest(candidate)
    with pytest.raises(ValueError,match='immutable'):prepare(source,candidate,tmp_path/'output',evidence=evidence)


def test_rejects_missing_diagnostic_scope(tmp_path):
    source,candidate,evidence,g=inputs(tmp_path);evidence['diagnostic_only']=False
    with pytest.raises(ValueError,match='diagnostic'):prepare(source,candidate,tmp_path/'output',evidence=evidence)


def test_does_not_fill_missing_source_poses(tmp_path):
    source,candidate,evidence,g=inputs(tmp_path)
    broken=tmp_path/'broken_parent';broken.mkdir()
    g['object_translation'][1,0]=np.nan;np.savez(broken/'geometry.npz',**g)
    publish(broken,'reconstruction',dict(geometry=broken/'geometry.npz'))
    evidence['parent_artifact_sha256']=digest(broken/'artifact.json')
    with pytest.raises(ValueError,match='values'):
        prepare(broken,candidate,tmp_path/'output',evidence=evidence)
