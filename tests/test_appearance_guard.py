import numpy as np
import pytest
from real2sim2real.reconstruction.appearance_guard import Feature,feature_mask,assess_feature,angular_proposals


def blue():
    return Feature(differences=((2,0,20),(2,1,8)),lower=((2,40),),upper=((0,150),),min_pixels=2,tolerance_px=0)


def test_feature_uses_signed_rgb_and_rejects_wrong_visible_side():
    rgb=np.full((6,8,3),255,np.uint8);rgb[2:4,1:3]=[40,70,180]
    predicted=np.zeros((6,8),bool);predicted[2:4,5:7]=True
    obj=np.ones((6,8),bool);human=np.zeros_like(obj)
    report=assess_feature(predicted,rgb,obj,human,blue())
    assert report['rejected'] and report['precision']==0
    report=assess_feature(feature_mask(rgb,blue()),rgb,obj,human,blue())
    assert not report['rejected'] and report['precision']==1
    assert not feature_mask(np.full_like(rgb,[250,0,10]),blue()).any()


def test_hidden_human_feature_is_uncertain_not_approval():
    rgb=np.full((6,8,3),255,np.uint8);predicted=np.zeros((6,8),bool);predicted[2:4,1:3]=True
    obj=np.ones_like(predicted);human=predicted.copy()
    report=assess_feature(predicted,rgb,obj,human,blue())
    assert report['uncertain_hidden_feature'] and not report['rejected']
    assert report['accepted_yaw'] is False


def test_tolerance_is_bounded_and_zero_means_no_dilation():
    f=blue();rgb=np.zeros((6,8,3),np.uint8);rgb[2:4,1:3]=[40,70,180]
    predicted=np.zeros((6,8),bool);predicted[2:4,3]=True
    obj=np.ones_like(predicted);human=np.zeros_like(obj)
    assert assess_feature(predicted,rgb,obj,human,f)['rejected']
    from dataclasses import replace
    assert not assess_feature(predicted,rgb,obj,human,replace(f,tolerance_px=1))['rejected']
    with pytest.raises(ValueError): replace(f,tolerance_px=11).validate()
    with pytest.raises(ValueError): feature_mask(rgb.astype(float),f)


def test_angular_hypotheses_preserve_world_center_and_rigidity():
    v=np.array([[1,0,0],[2,1,0],[0,1,2]],float);p=np.eye(4);p[:3,3]=[0,0,3]
    result=angular_proposals([p],v,[0,1,0])
    assert len(result)==9
    for q in result:
        np.testing.assert_allclose(q[:3,:3]@v.mean(0)+q[:3,3],v.mean(0)+p[:3,3])
        np.testing.assert_allclose(q[:3,:3]@q[:3,:3].T,np.eye(3),atol=1e-12)
    with pytest.raises(ValueError): angular_proposals([p],v,[0,2,0])
