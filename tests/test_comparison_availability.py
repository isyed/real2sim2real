import numpy as np
import pytest
from real2sim2real import comparison_video as video

def test_default_and_disjoint_available_frames():
    assert video.available_pose_ids(None,10) is None
    assert video.available_pose_ids([0,1,7,8],10)==[0,1,7,8]
    assert video.available_pose_ids([],10)==[]

@pytest.mark.parametrize('ids',[[1,1],[2,1],[-1],[10],[True],[1.5]])
def test_invalid_or_ambiguous_available_ids_rejected(ids):
    with pytest.raises(ValueError,match='Available poses'):video.available_pose_ids(ids,10)

def test_unavailable_interval_draws_rgb_without_touching_geometry(monkeypatch):
    # Missing geometry fields deliberately fail if an unavailable pose is read.
    row=('Recovered intervals',dict(frame_ids=np.arange(5)),None,None,frozenset(),frozenset([0,1,4]))
    monkeypatch.setattr(video,'_STATE',([row],(30,20),30.,'Availability test',None),raising=False)
    rgb=np.zeros((20,30,3),np.uint8);rgb[:]=[20,40,60]
    i,out=video._render((2,rgb.tobytes()))
    assert i==2
    assert np.array_equal(out[96:,512:],np.broadcast_to([20,40,60],out[96:,512:].shape))

def test_disjoint_availability_requires_partial_mode(tmp_path):
    path=tmp_path/'geometry.npz';np.savez(path,frame_ids=np.arange(5))
    with pytest.raises(ValueError,match='allow_partial=True'):
        video.render(tmp_path/'unused.mp4',tmp_path/'out',[
            dict(label='Measured spans',geometry=path,surface=tmp_path/'unused_surface',available_frame_ids=[0,1,4])])
