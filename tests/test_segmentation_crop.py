import numpy as np
import pytest
from real2sim2real.reconstruction.segmentation_crop import crop_annotation,restore_mask


def test_prompt_coordinates_are_mapped_without_changing_source_or_labels():
    p=dict(frame_index=7,object_id=1,points=[dict(x=12,y=23),dict(x=17,y=26)],point_labels=[1,0])
    q=crop_annotation(p,[10,20,20,30],[40,50])
    assert q['points']==[dict(x=2,y=3),dict(x=7,y=6)] and q['point_labels']==[1,0]
    assert q['frame_index']==7 and p['points'][0]==dict(x=12,y=23)


def test_backprojection_preserves_pixel_identity_and_never_fills_outside_crop():
    m=np.zeros((10,10),bool);m[3,2]=True
    restored=restore_mask(m,[10,20,20,30],[40,50])
    assert restored.shape==(50,40) and restored[23,12]==255 and np.count_nonzero(restored)==1


def test_prompt_outside_crop_cannot_be_silently_clipped():
    with pytest.raises(ValueError):crop_annotation(dict(points=[dict(x=9,y=23)]),[10,20,20,30],[40,50])


def test_wrong_native_mask_shape_or_crop_is_rejected():
    with pytest.raises(ValueError):restore_mask(np.ones((9,10)),[10,20,20,30],[40,50])
    with pytest.raises(ValueError):restore_mask(np.ones((10,10)),[10,20,45,30],[40,50])
