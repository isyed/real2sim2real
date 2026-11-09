import json,sys,types
import numpy as np
import pytest


def test_invalid_rgb_contract_is_rejected_before_gpu_estimator(tmp_path,monkeypatch):
    from real2sim2real.reconstruction import validate_pose_sequence as sequence
    from real2sim2real.reconstruction import tracking_recovery
    calls=[]
    up=types.SimpleNamespace(_pose_estimator=lambda *args:calls.append('GPU'),load_export_object_mesh=lambda *args:None)
    monkeypatch.setattr(tracking_recovery,'load_exporter',lambda:up)
    fake=types.ModuleType('prep.mhr_export_utils')
    def invalid(*args):raise ValueError("JPEG RGB dataset metadata is missing: ['optimize']")
    fake.frame_names=invalid;monkeypatch.setitem(sys.modules,'prep.mhr_export_utils',fake)
    export=tmp_path/'export';export.mkdir();identity='source-hash'
    (export/'wild_export.json').write_text(json.dumps(dict(source_video_sha256=identity,intrinsics=np.eye(3).tolist())))
    mapping=tmp_path/'mapping.json';mapping.write_text(json.dumps(dict(source_video_sha256=identity,source_frame_indices=[0])))
    support=tmp_path/'support.json';support.write_text(json.dumps(dict(ground_truth_used=False,evidence='source pixels',plane_camera=[0,0,1,0],center_camera_m=[0,0,0],radius_m=1)))
    proposals=tmp_path/'proposals.npz';np.savez(proposals,source_frame_ids=np.array([0]),poses_camera=np.eye(4)[None])
    with pytest.raises(ValueError,match='optimize'):sequence.validate(export,tmp_path/'depth.h5',support,proposals,mapping,tmp_path/'validation')
    assert calls==[]
