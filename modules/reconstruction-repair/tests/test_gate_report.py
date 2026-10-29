import json

from real2sim2real.reconstruction_repair.gate_report import explain, markdown, explain_file


def test_minor_quality_failure_is_distinct_from_required_validity():
    report = dict(schema='r2s2r_engines.repair-report/1',metric_size_preflight=dict(passed=True,status='passed_current_size'),
        stages=[dict(stage='pose_recovery',outcome='rejected',validation=dict(accepted=False,
            reasons=['per_frame_silhouette_regressed'],checked_gates=[
                dict(id='immutable_frame_ids_changed',status='passed',explanation='Frame identity preserved'),
                dict(id='per_frame_silhouette_regressed',status='failed',observed=-.0201,threshold=-.02)]))])
    result=explain(report)
    failed=[r for r in result['gates'] if r['status']=='failed']
    assert len(failed)==1 and failed[0]['importance']=='reference_quality'
    assert result['required_checks_unresolved']==[]
    assert result['component_outcomes'][0]['fallback']=='retained parent'
    assert not result['actor_training_authorized'] and not result['acceptance_policy_changed']
    assert '-0.0201' in markdown(result)


def test_unknown_metric_check_is_not_silently_passed():
    report=dict(schema='r2s2r_engines.repair-report/1',stages=[])
    result=explain(report)
    assert 'metric_size_current_mesh' in result['required_checks_unresolved']
    assert not result['actor_training_authorized']


def test_controller_failure_can_be_a_training_target_not_an_input_failure(tmp_path):
    report=dict(schema='r2s2r_engines.external-comparison-gates/1',gates=[
        dict(id='metric_size_current_mesh',stage='metric_size',status='passed',importance='required_validity',explanation='Example supplied checked evidence'),
        dict(id='balance',stage='physics',status='failed',importance='controller_execution',explanation='Frozen controller fell')])
    result=explain(report)
    assert result['required_checks_unresolved']==[]
    assert 'bounded actor pilot' in result['actor_training_recommendation']
    assert not result['actor_training_authorized']
    source=tmp_path/'report.json';source.write_text(json.dumps(report))
    explain_file(source,tmp_path/'review')
    assert (tmp_path/'review/GATES.md').is_file()


def test_window_report_lists_passing_depth_and_failing_image_separately():
    from test_temporal_pose import fixture,CFG
    from real2sim2real.reconstruction_repair.temporal_pose import refine_window
    data=fixture();_,report=refine_window(data,CFG)
    # Use a real report, then an explicit failed image-review decision.
    report['decision']['reasons']=['previous_good_image_regressed'];report['decision']['accepted']=False
    result=explain(report);by_id={r['id']:r for r in result['gates']}
    assert by_id['depth_gate']['status']=='passed'
    assert by_id['previous_good_image_regressed']['status']=='failed'
    assert by_id['previous_good_image_regressed']['importance']=='reference_quality'
    assert by_id['physics']['status']=='not_checked'


def test_rejected_proposal_failure_is_not_attributed_to_retained_parent():
    report=dict(schema='r2s2r_engines.repair-report/1',metric_size_preflight=dict(passed=True,status='passed_current_size'),
        stages=[dict(stage='pose_recovery',outcome='rejected',validation=dict(accepted=False,
            reasons=['locked_frames_changed'],checked_gates=[dict(id='locked_frames_changed',status='failed')]))])
    result=explain(report)
    assert result['required_checks_unresolved']==[]
    failed=next(r for r in result['gates'] if r['id']=='locked_frames_changed')
    assert failed['status']=='failed' and not failed['applies_to_selected_candidate']
    assert 'unselected proposal' in markdown(result)
