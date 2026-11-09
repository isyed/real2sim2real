import json
import numpy as np
import pytest
imageio_ffmpeg=pytest.importorskip("imageio_ffmpeg")
from real2sim2real.comparison_video import render

def test_complete_video_cache_and_surface_identity(tmp_path):
 r=tmp_path;s=r/'surface';s.mkdir()
 v=np.array([[-.2,-.2,3.],[.2,-.2,3.],[0,.2,3.]])
 np.save(s/'vertices.npy',np.tile(v,(3,1,1)));np.save(s/'faces.npy',np.array([[0,1,2]]));(s/'surface.json').write_text(json.dumps(dict(frames=3)))
 np.savez(r/'g.npz',vertices=v,faces=np.array([[0,1,2]]),object_scale=np.array(1.),object_rotation=np.tile(np.eye(3),(3,1,1)),object_translation=np.zeros((3,3)),frame_ids=np.arange(3),timestamps_s=np.arange(3)/30,world_to_camera=np.tile(np.eye(4),(3,1,1)),intrinsics=np.tile([[100.,0,32],[0,100,24],[0,0,1]],(3,1,1)))
 w=imageio_ffmpeg.write_frames(str(r/'source.mp4'),(64,48),fps=30);w.send(None)
 for _ in range(3):w.send(np.zeros((48,64,3),np.uint8))
 w.close();c=[dict(label='Synthetic',geometry=r/'g.npz',surface=s)]
 result=render(r/'source.mp4',r/'video',c,workers=1);assert result['frames']==3
 assert render(r/'source.mp4',r/'video',c,workers=1)==result
 np.save(s/'faces.npy',np.array([[2,1,0]]))
 try:render(r/'source.mp4',r/'video',c,workers=1)
 except ValueError:pass
 else:raise AssertionError('Modified surface not detected')
 print('PASS: 3-frame render, complete decode, cache reuse, surface tamper rejection',r)


def partial_fixture(root):
 s=root/'surface';s.mkdir();v=np.array([[-.2,-.2,3.],[.2,-.2,3.],[0,.2,3.]])
 np.save(s/'vertices.npy',np.tile(v,(3,1,1)));np.save(s/'faces.npy',np.array([[0,1,2]]));(s/'surface.json').write_text(json.dumps(dict(frames=3)))
 np.savez(root/'g.npz',vertices=v,faces=np.array([[0,1,2]]),object_scale=np.array(1.),object_rotation=np.eye(3)[None],object_translation=np.zeros((1,3)),frame_ids=np.arange(1),timestamps_s=np.arange(1)/30,world_to_camera=np.eye(4)[None],intrinsics=np.array([[[100.,0,32],[0,100,24],[0,0,1]]]))
 w=imageio_ffmpeg.write_frames(str(root/'source.mp4'),(64,48),fps=30);w.send(None)
 for _ in range(3):w.send(np.zeros((48,64,3),np.uint8))
 w.close();return [dict(label='Rejected partial trial',geometry=root/'g.npz',surface=s)]


def test_partial_candidate_is_rejected_without_explicit_opt_in(tmp_path):
 candidates=partial_fixture(tmp_path)
 with pytest.raises(ValueError,match='timeline'):
  render(tmp_path/'source.mp4',tmp_path/'video',candidates,workers=1)


def test_partial_trial_video_marks_missing_poses_instead_of_carrying_geometry(tmp_path):
 from real2sim2real.comparison_video import _initialize,_render
 candidates=partial_fixture(tmp_path)
 result=render(tmp_path/'source.mp4',tmp_path/'video',candidates,workers=1,allow_partial=True)
 assert result['frames']==3 and result['candidate_coverage_frames']==[1]
 assert not result['complete_candidate_timelines']
 assert not result['unavailable_poses_carried_or_interpolated']
 _initialize(candidates,(64,48),30,'Diagnostic')
 _,first=_render((0,bytes(64*48*3)));_,missing=_render((1,bytes(64*48*3)))
 assert first[96:,512:].any()
 assert not missing[96:,512:].any()  # Real source RGB only, no stale overlay.
 assert missing[:45,512:].any()      # Explicit unavailable-pose title/notice.


def test_uncertain_existing_pose_is_labeled_and_bound_to_receipt(tmp_path):
 from real2sim2real.comparison_video import _initialize,_render
 candidates=partial_fixture(tmp_path);candidates[0]['uncertain_frame_ids']=[0]
 result=render(tmp_path/'source.mp4',tmp_path/'video',candidates,workers=1,allow_partial=True)
 assert result['candidate_uncertain_frame_ids']==[[0]] and result['uncertain_poses_labeled']
 _initialize(candidates,(64,48),30,'Diagnostic');_,uncertain=_render((0,bytes(64*48*3)))
 candidates[0]['uncertain_frame_ids']=[]
 _initialize(candidates,(64,48),30,'Diagnostic');_,ordinary=_render((0,bytes(64*48*3)))
 assert np.array_equal(uncertain[96:],ordinary[96:]) # Pose evidence is displayed, not replaced.
 assert not np.array_equal(uncertain[25:42,512:],ordinary[25:42,512:])
 with pytest.raises(ValueError,match='identity differs'):
  render(tmp_path/'source.mp4',tmp_path/'video',candidates,workers=1,allow_partial=True)


def test_uncertain_pose_ids_cannot_hide_unavailable_frames(tmp_path):
 candidates=partial_fixture(tmp_path);candidates[0]['uncertain_frame_ids']=[2]
 with pytest.raises(ValueError,match='outside candidate coverage'):
  render(tmp_path/'source.mp4',tmp_path/'video',candidates,workers=1,allow_partial=True)


def test_display_simplification_never_rewrites_source_geometry(tmp_path):
 import trimesh,hashlib
 from real2sim2real.comparison_video import _initialize
 candidates=partial_fixture(tmp_path);path=tmp_path/'g.npz'
 g=dict(np.load(path));mesh=trimesh.creation.icosphere(subdivisions=2,radius=.2)
 g.update(vertices=mesh.vertices,faces=mesh.faces);np.savez(path,**g)
 before=hashlib.sha256(path.read_bytes()).hexdigest()
 _initialize(candidates,(64,48),30,'Diagnostic',object_display_faces=100)
 from real2sim2real import comparison_video
 display=comparison_video._STATE[0][0][1]
 assert len(display['faces'])<=100
 assert np.array_equal(display['object_translation'],g['object_translation'])
 assert np.array_equal(display['object_rotation'],g['object_rotation'])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==before
 with pytest.raises(ValueError,match='face budget'):
  render(tmp_path/'source.mp4',tmp_path/'video',candidates,object_display_faces=True)


def test_missing_optional_simplifier_fails_before_creating_video(tmp_path,monkeypatch):
 import sys
 from real2sim2real.comparison_video import check_display_dependencies
 monkeypatch.setitem(sys.modules,'fast_simplification',None)
 check_display_dependencies(None)
 with pytest.raises(RuntimeError,match='rendering environment'):
  render(tmp_path/'missing.mp4',tmp_path/'video',[dict(label='Input')],object_display_faces=100)
 assert not (tmp_path/'video').exists()
