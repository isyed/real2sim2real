import numpy as np
import pytest
from real2sim2real.reconstruction.mask_review import sample_overlay,review_masks


def test_overlays_expose_wrong_subject_without_semantic_approval():
    rgb=np.zeros((30,40,3),dtype=np.uint8);mask=np.zeros((30,40),dtype=np.uint8)
    mask[2:8,4:10]=255
    image,rows=sample_overlay(rgb,[mask,np.zeros_like(mask)],['human','object'])
    assert rows[0]['foreground_pixels']==36 and rows[0]['bbox_xyxy']==[4,2,10,8]
    assert rows[1]['foreground_pixels']==0 and rows[1]['bbox_xyxy'] is None
    assert np.asarray(image).max()>0


def test_review_rejects_other_pixel_grid():
    with pytest.raises(ValueError,match='different pixel grids'):
        sample_overlay(np.zeros((30,40,3),dtype=np.uint8),[np.zeros((15,20))],['human'])


def test_nonfinite_mask_cannot_be_reviewed():
    with pytest.raises(ValueError,match='Nonfinite'):
        sample_overlay(np.zeros((3,4,3),dtype=np.uint8),[np.full((3,4),np.nan)],['object'])
