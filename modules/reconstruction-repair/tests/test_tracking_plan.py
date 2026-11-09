from real2sim2real.reconstruction_repair.tracking_plan import plan,assess_replacement


def test_rejected_calibration_does_not_resize_or_authorize_full_tracking():
    p=plan(dict(scale_multiplier=2.5,candidate_ready=False,reasons=['heldout_extent_disagreement']))
    assert p['mesh_scale_multiplier']==1
    assert not p['full_tracking_permitted']
    assert not p['fine_repair_permitted']


def test_valid_resize_keeps_pose_reuse_and_fine_repair_blocked():
    p=plan(dict(scale_multiplier=2.5,candidate_ready=True,reasons=[]))
    assert p['mesh_scale_multiplier']==2.5 and p['full_tracking_permitted']
    assert not p['old_pose_sequence_reusable'] and not p['fine_repair_permitted']


def test_a_mean_gain_cannot_hide_a_good_frame_regression_or_partial_track():
    kw=dict(expected_frames=4,accepted_state_frames=[1,2,3],continuity_passed=True,native_views_passed=True,metric_passed=True)
    result=assess_replacement([.9,.1,.1,.1],[.8,.8,.8,.8],**kw)
    assert not result['accepted'] and result['previous_good_regression_frames']==[0]
    kw['accepted_state_frames']=[1,2]
    result=assess_replacement([.3]*4,[.8]*4,**kw)
    assert 'incomplete_accepted_state_trace' in result['reasons']


def test_complete_measured_gain_still_needs_current_metric_evidence():
    kw=dict(expected_frames=4,accepted_state_frames=[1,2,3],continuity_passed=True,native_views_passed=True,metric_passed=False)
    result=assess_replacement([.3]*4,[.8]*4,**kw)
    assert not result['accepted'] and result['reasons']==['current_mesh_metric_evidence']
    kw['metric_passed']=True
    assert assess_replacement([.3]*4,[.8]*4,**kw)['accepted']


def test_missing_calibration_depth_records_no_scale_and_only_allows_diagnosis():
    p=plan(dict(scale_multiplier=None,candidate_ready=False,reasons=['Insufficient reliable depth']))
    assert not p['calibration_scale_available']
    assert p['mesh_scale_multiplier']==1 and not p['full_tracking_permitted']


def test_explicit_diagnostic_tracking_cannot_resize_or_promote_a_failed_calibration():
    p=plan(dict(scale_multiplier=2.5,candidate_ready=False,reasons=['heldout_extent_disagreement']),
           allow_unchanged_mesh_diagnostic_tracking=True)
    assert p['full_tracking_permitted'] and p['unchanged_mesh_diagnostic_opt_in']
    assert p['mesh_scale_multiplier']==1 and not p['resize_hypothesis_retained']
    assert not p['reference_promotion_permitted'] and not p['fine_repair_permitted']
