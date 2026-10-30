import pytest
from real2sim2real.reconstruction_repair.observation_plan import plan


def audit(reasons):
    return dict(source_frames=len(reasons),rows=[dict(frame_id=i,reasons=r,observation_available=not r) for i,r in enumerate(reasons)])


def test_missing_masks_route_to_rgb_review_without_claiming_segmentation_error():
    report=plan(audit([['object_mask_missing']]*8+[[],[]]))
    assert report['next_action']=='review_source_rgb_and_object_masks'
    assert report['observed_frames']==2
    assert not report['reference_approved'] and report['source_rgb_review_required']


def test_depth_failure_is_separate_from_missing_masks():
    report=plan(audit([['incoherent_foreground_depth']]*8+[[],[]]))
    assert report['next_action']=='review_foreground_visibility_and_depth'


def test_good_observations_do_not_approve_geometry():
    report=plan(audit([[]]*8+[['object_mask_missing']]*2))
    assert report['next_action']=='inspect_reported_gaps_before_pose_repair'
    assert not report['reference_approved']


def test_incomplete_audit_cannot_supply_a_whole_clip_recommendation():
    value=audit([[],[]]);value['source_frames']=3
    with pytest.raises(ValueError,match='Complete'):plan(value)


def test_source_bound_semantic_rejection_vetoes_even_available_depth():
    value=audit([[]]*10);value['source_identity']=dict(artifact_sha256='a'*64,video_sha256='b'*64)
    review=dict(value['source_identity'],passed=False,reviewed_frame_ids=[0,5,9],evidence='Masks stay on background while the object moves in RGB')
    result=plan(value,semantic_review=review)
    assert result['pose_optimization_blocked_by_semantic_review']
    assert result['next_action']=='repair_foreground_segmentation_before_pose'
    review['video_sha256']='c'*64
    with pytest.raises(ValueError,match='another source'):plan(value,semantic_review=review)
