import json
import numpy as np
import pytest
from real2sim2real.reconstruction.daid_recovery import sha, validate_request


def request(tmp_path):
    data=tmp_path/'data'; data.mkdir(); checkout=tmp_path/'upstream'
    fork=checkout/'reconstruction/modules/Fast-SAM3D'
    fork.mkdir(parents=True); (fork/'sam3d_objects/pipeline').mkdir(parents=True)
    names=['canonical_mesh.obj','shared_mesh.obj','shape_latent.pt','anchor_layout.json',
           '000002.png','000002_mask.png','000002_crop.npz']
    for name in names:(data/name).write_bytes(b'fixture')
    upstream=['track_object.py','sam3d_objects/pipeline/inference_pipeline_pointmap.py']
    for name in upstream:(fork/name).write_text('# fixture')
    state=tmp_path/'accepted.json';pose=np.eye(4).tolist()
    state.write_text(json.dumps(dict(source_frame=1,pose_camera=pose)))
    spec=dict(input_dir=str(data),output_dir=str(tmp_path/'out'),checkout=str(checkout),
              config_dir=str(tmp_path/'configs'),source_frame=2,previous_frame=1,
              previous_pose_camera=pose,depth_hypothesis='crop',accepted_state_path=str(state),
              accepted_state_sha256=sha(state),input_sha256={n:sha(data/n) for n in names},
              upstream_sha256={n:sha(fork/n) for n in upstream})
    path=tmp_path/'request.json';path.write_text(json.dumps(spec))
    return path,spec


def test_cpu_contract_needs_no_upstream_import_or_gpu(tmp_path):
    path,spec=request(tmp_path)
    assert validate_request(path)['source_frame']==2


def test_gpu_request_cannot_use_pending_cpu_anchor(tmp_path):
    path,spec=request(tmp_path)
    with pytest.raises(ValueError,match='independently accepted'):validate_request(path,require_accepted=True)


def test_conditioning_must_match_accepted_receipt(tmp_path):
    path,spec=request(tmp_path);spec['previous_pose_camera'][0][3]=.1;path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='Conditioning'):validate_request(path)


def test_changed_shape_cache_rejected(tmp_path):
    path,spec=request(tmp_path);(tmp_path/'data/shape_latent.pt').write_bytes(b'other')
    with pytest.raises(ValueError,match='Input changed'):validate_request(path)


def test_changed_upstream_requires_new_review(tmp_path):
    path,spec=request(tmp_path);(tmp_path/'upstream/reconstruction/modules/Fast-SAM3D/track_object.py').write_text('changed')
    with pytest.raises(ValueError,match='Upstream version changed'):validate_request(path)


def test_depth_filename_cannot_escape_input_folder(tmp_path):
    path,spec=request(tmp_path);spec['depth_hypothesis']='../../elsewhere';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):validate_request(path)
