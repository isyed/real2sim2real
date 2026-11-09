import numpy as np
import pytest
from real2sim2real.reconstruction.mask_features import audit,check


def test_body_mask_alone_does_not_pass_a_required_thin_part():
    mask=np.zeros((100,100),dtype=bool);mask[50:90,25:75]=True
    parts=[dict(name='body',xyxy=[30,60,70,80],min_fraction=.9),
           dict(name='thin_rail',xyxy=[30,15,33,45],min_fraction=.5),
           dict(name='open_gap',xyxy=[40,15,50,45],max_fraction=.1)]
    assert not audit(mask,parts)['passed']
    mask[15:50,30:33]=True
    result=audit(mask,parts)
    assert result['passed'] and not result['semantics_verified']
    mask[15:45,40:50]=True
    assert not audit(mask,parts)['passed']


def test_native_frame_mask_receipt_changes_when_pixels_change(tmp_path):
    from PIL import Image
    folder=tmp_path/'masks/1';folder.mkdir(parents=True)
    path=folder/'000012.png';Image.fromarray(np.ones((10,10),dtype='uint8')*255).save(path)
    checks=[dict(frame_index=12,features=[dict(name='part',xyxy=[0,0,10,10],min_fraction=.5)])]
    first=check(tmp_path,checks,source_video_sha256='a'*64)
    Image.fromarray(np.zeros((10,10),dtype='uint8')).save(path)
    second=check(tmp_path,checks,source_video_sha256='a'*64)
    assert first['passed'] and not second['passed']
    assert first['checks'][0]['mask_sha256']!=second['checks'][0]['mask_sha256']


@pytest.mark.parametrize('box', [[-1,0,2,2],[0,0,200,2],[0,0,0,2],[0.,0,2,2]])
def test_rejects_ambiguous_or_out_of_grid_regions(box):
    with pytest.raises(ValueError):audit(np.ones((5,5)),[dict(name='part',xyxy=box)])
