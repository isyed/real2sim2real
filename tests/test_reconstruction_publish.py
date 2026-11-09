"""The publish phase must work in a fresh process after a later-frame mesh run."""
import json
import pickle
import numpy as np


def test_publish_hashes_saved_outputs_without_executing_earlier_phases(tmp_path, monkeypatch):
    import trimesh
    from real2sim2real import human
    from real2sim2real.artifacts import load,digest
    from real2sim2real.reconstruction import backend
    root=tmp_path/'out';root.mkdir();weights=tmp_path/'weights';weights.mkdir()
    ckpt=weights/'sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt'
    model=ckpt.parent/'assets/mhr_model.pt';model.parent.mkdir(parents=True)
    ckpt.write_bytes(b'checkpoint');model.write_bytes(b'model')
    video=tmp_path/'source.mp4';video.write_bytes(b'video identity only')
    prompts=tmp_path/'prompts.json';prompts.write_text('{}')
    for name in ['sam2','sam3d']:(weights/name).mkdir()
    export=root/'export/source';(export/'object_mesh').mkdir(parents=True)
    trimesh.creation.box().export(export/'object_mesh/output_aligned.glb')
    (export/'wild_export.json').write_text(json.dumps(dict(intrinsics=np.eye(3).tolist())))
    for name in ['object_masks','human_masks']:
        (export/name).mkdir();(export/name/'front_stereo_camera_left.h5').write_bytes(b'opaque saved mask cache')
    (root/'depth').mkdir();(root/'depth/aligned.h5').write_bytes(b'opaque saved predicted depth')
    (root/'timeline.json').write_text(json.dumps(dict(frames=3,timestamps_s=[0,1/30,2/30])))
    (root/'metric_input_check.json').write_text(json.dumps(dict(mesh_frame_index=2)))
    poses=np.tile(np.eye(4),(3,1,1));poses[:,2,3]=2
    with (root/'object_motion.pkl').open('wb') as f:pickle.dump(dict(fp_poses_cam=poses[:,None],fp_reliable=np.ones(3,bool),fp_full_image_iou=np.ones(3)*.9),f)
    with (root/'human.pkl').open('wb') as f:pickle.dump({},f)
    monkeypatch.setattr(backend,'engine_paths',lambda:(tmp_path,tmp_path,tmp_path))
    monkeypatch.setattr(human,'load_mhr_buffers',lambda _: {})
    monkeypatch.setattr(human,'decode_initialization_mhr',lambda *_:dict(mhr_model_params=np.zeros((3,204))))
    request=dict(input=str(video),input_sha256=digest(video),output=str(root),config_base=str(tmp_path),
                 options=dict(prompts=str(prompts),weights=str(weights),sam2_weights=str(weights/'sam2'),sam3d_weights=str(weights/'sam3d')))
    backend.run(request,'publish')
    artifact=load(root,kinds={'reconstruction'})
    assert artifact['files']['geometry']['sha256']==digest(root/'geometry.npz')
    assert json.loads((root/'reconstruction.json').read_text())['metric_pointmap']['mesh_frame_index']==2
