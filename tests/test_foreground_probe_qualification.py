import pytest
from real2sim2real.reconstruction_repair.foreground_depth import qualify_for_pose_validation

def probe():
    return dict(rows=[dict(source_frame=i,variants=dict(crop=dict(accepted=True,
        object_pixels=300,valid_object_pixels=298,object_depth_quantiles_m=[2.8,3.,3.2]))) for i in [5,6,20]])

def test_pass_allows_only_native_pose_validation():
    r=qualify_for_pose_validation(probe(),[5,6,20])
    assert r['passed'] and r['requires_native_gpu_pose_checks']
    assert not r['reference_approved'] and not r['metric_truth_known']
    assert r['audit']['unsampled_intervals']==[[7,19]]
    assert not r['audit']['continuous_motion_established']

def test_floor_and_foreground_are_independent():
    p=probe();p['rows'][1]['variants']['crop']['valid_object_pixels']=150
    assert 'coverage_6' in qualify_for_pose_validation(p,[5,6,20])['reasons']

def test_adjacent_jump_is_not_hidden_by_good_floor():
    p=probe();p['rows'][1]['variants']['crop']['object_depth_quantiles_m']=[3.5,4.,4.5]
    assert 'adjacent_depth_jump' in qualify_for_pose_validation(p,[5,6,20])['reasons']

def test_native_sample_identity_is_required():
    with pytest.raises(ValueError,match='declared native'):qualify_for_pose_validation(probe(),[5,6,21])

def test_inconsistent_pixel_counts_rejected():
    p=probe();p['rows'][1]['variants']['crop']['valid_object_pixels']=301
    with pytest.raises(ValueError,match='foreground counts'):qualify_for_pose_validation(p,[5,6,20])

def test_variant_must_exist():
    with pytest.raises(ValueError,match='variant'):qualify_for_pose_validation(probe(),[5,6,20],variant='absent')

@pytest.mark.parametrize('limit',[0.,1.1,float('nan')])
def test_invalid_coverage_limit_rejected(limit):
    with pytest.raises(ValueError,match='qualification limits'):
        qualify_for_pose_validation(probe(),[5,6,20],min_visible_coverage=limit)
