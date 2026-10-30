import io
import numpy as np
import pytest
import h5py
from PIL import Image
from real2sim2real import artifacts
from real2sim2real.reconstruction_repair.foreground_visibility import artifact_audit


def test_artifact_audit_publishes_source_bound_semantic_veto(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();(source/'artifact.json').write_text('{}')
    np.savez(source/'geometry.npz',frame_ids=np.arange(2),timestamps_s=np.arange(2)/30)
    buf=io.BytesIO();Image.fromarray(np.full((16,16),2000,np.uint16)).save(buf,format='PNG')
    with h5py.File(source/'depth.h5','w') as h:
        d=h.create_dataset('aligned/front_stereo_camera_left',(2,),dtype=h5py.vlen_dtype(np.dtype('uint8')))
        for i in range(2):d[i]=np.frombuffer(buf.getvalue(),np.uint8)
    for name,fill in [('object',1),('human',0)]:
        with h5py.File(source/(name+'.h5'),'w') as h:h.create_dataset('frames',data=np.full((2,16,16),fill,np.uint8))
    files=dict(geometry='geometry.npz',depth='depth.h5',object_masks='object.h5',human_masks='human.h5')
    manifest=dict(metadata=dict(source_video_sha256='b'*64),files={k:dict(path=v,sha256=artifacts.digest(source/v)) for k,v in files.items()})
    monkeypatch.setattr(artifacts,'load',lambda *a,**k:manifest)
    review=dict(artifact_sha256=artifacts.digest(source/'artifact.json'),video_sha256='b'*64,passed=False,reviewed_frame_ids=[0,1],evidence='Mask selects background in RGB')
    result=artifact_audit(source,tmp_path/'audit',semantic_review=review)
    assert result['observed_frames']==2
    assert result['repair_plan']['pose_optimization_blocked_by_semantic_review']
    assert result['source_identity']['video_sha256']=='b'*64
    review['video_sha256']='c'*64
    with pytest.raises(ValueError,match='another source'):artifact_audit(source,tmp_path/'wrong',semantic_review=review)
    assert not (tmp_path/'wrong').exists()
