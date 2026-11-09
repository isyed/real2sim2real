import numpy as np
import pytest
from real2sim2real.reconstruction_repair.crop_depth_sequence import compose,restore_outside_pixels


def test_crop_maps_to_native_pixels_and_keeps_outside_exact():
    original=np.arange(48,dtype=np.float32).reshape(6,8)/10
    patch=np.full((2,3),2.5,np.float32)
    output=compose(original,patch,np.ones_like(patch,bool),[2,1,5,3])
    expected=original.copy();expected[1:3,2:5]=2.5
    np.testing.assert_array_equal(output,expected)
    np.testing.assert_array_equal(original,np.arange(48,dtype=np.float32).reshape(6,8)/10)


def test_invalid_new_depth_does_not_silently_reuse_old_measurement():
    original=np.ones((4,4),np.float32);patch=np.full((2,2),3.,np.float32);valid=np.ones((2,2),bool);valid[0,1]=False
    output=compose(original,patch,valid,[1,1,3,3])
    assert output[1,2]==0 and output[1,1]==3 and output[0,0]==1


def test_native_integer_preservation_catches_legacy_float_codec_loss():
    original=np.full((4,5),1001,np.uint16);encoded=np.full((4,5),1000,np.uint16)
    encoded[1:3,2:4]=2500
    fixed=restore_outside_pixels(original,encoded,[2,1,4,3])
    expected=original.copy();expected[1:3,2:4]=2500
    np.testing.assert_array_equal(fixed,expected)
    np.testing.assert_array_equal(encoded[0],np.full(5,1000,np.uint16))


def test_integer_preservation_rejects_unmapped_or_float_grids():
    with pytest.raises(ValueError):restore_outside_pixels(np.ones((4,4)),np.ones((4,4),np.uint16),[1,1,3,3])


@pytest.mark.parametrize('box,patch,valid',[
    ([1,1,5,3],np.ones((2,4)),np.ones((2,4),bool)),
    ([1,1,3,3],np.ones((1,2)),np.ones((1,2),bool)),
    ([1,1,3,3],np.full((2,2),np.nan),np.ones((2,2),bool)),
    ([1,1,3,3],np.zeros((2,2)),np.ones((2,2),bool)),
])
def test_invalid_geometry_or_depth_cannot_be_materialized(box,patch,valid):
    with pytest.raises(ValueError):compose(np.ones((4,4),np.float32),patch,valid,box)
