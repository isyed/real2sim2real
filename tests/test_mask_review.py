import numpy as np
import pytest
from real2sim2real.reconstruction.mask_review import sample_overlay,audit_mask_timelines


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


def masks(tmp_path,counts):
    from PIL import Image
    for role in (0,1):
        folder=tmp_path/str(role);folder.mkdir()
        for i,count in enumerate(counts):
            a=np.zeros(100,np.uint8);a[:count if role==0 else 0]=255
            Image.fromarray(a.reshape(10,10)).save(folder/f'{i:06d}.png')


def test_all_frame_audit_catches_unsampled_human_dropout(tmp_path):
    masks(tmp_path,[50,19,20,50])
    report=audit_mask_timelines(tmp_path,4)
    assert report['sparse_human_frames']==[1]
    assert report['foreground_counts']['human']==[50,19,20,50]
    assert report['required_human_foreground_pixels']==20
    assert not report['model_prompt_inputs_valid']


def test_exact_existing_minimum_passes_and_object_invisibility_is_recorded(tmp_path):
    masks(tmp_path,[20,21])
    report=audit_mask_timelines(tmp_path,2)
    assert report['model_prompt_inputs_valid']
    assert report['foreground_counts']['object']==[0,0]


def test_audit_rejects_gapped_timeline(tmp_path):
    masks(tmp_path,[50,50])
    (tmp_path/'0/000001.png').unlink()
    with pytest.raises(ValueError,match='complete source timeline'):audit_mask_timelines(tmp_path,2)
