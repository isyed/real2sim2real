from pathlib import Path
import tempfile,json
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
