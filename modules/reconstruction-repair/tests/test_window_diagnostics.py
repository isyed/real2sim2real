import copy
import pytest
from real2sim2real.reconstruction_repair.window_diagnostics import explain


def example():
    row=dict(frame=10,iou=.9,precision=.95,recall=.95,depth_points=1000,
             depth_error_m=.01,support_penetration_m=.03,behind_camera=False)
    return dict(config=dict(min_iou=.6,min_precision=.6,min_recall=.6,
        min_depth_points=100,max_depth_error_m=.1,max_support_penetration_m=.02,
        max_frame_iou_drop=.02),before=[row],after=[copy.deepcopy(row)],decision=dict(accepted=False))


def test_unchanged_bad_locked_support_cannot_be_fixed_by_other_frames():
    report=example();original=copy.deepcopy(report)
    result=explain(report,locked_frame_ids=[10])
    assert result['locked_anchor_blockers']==[dict(frame=10,reasons=['support_gate'])]
    assert result['frames'][0]['inherited']==['support_gate']
    assert result['strict_pass_requires_anchor_or_input_revision']
    assert not result['candidate_accepted'] and report==original


def test_new_image_loss_is_not_hidden_by_resolved_support():
    report=example();report['after'][0].update(support_penetration_m=0,iou=.86)
    result=explain(report)
    assert result['frames'][0]['resolved']==['support_gate']
    assert result['frames'][0]['introduced']==['previous_good_image_regressed']
    assert not result['strict_pass_requires_anchor_or_input_revision']


def test_unavailable_depth_at_locked_anchor_is_explicit():
    report=example();report['before'][0]['depth_error_m']=None
    report['after'][0]['depth_error_m']=None
    result=explain(report,locked_frame_ids=[10])
    assert 'depth_gate' in result['locked_anchor_blockers'][0]['reasons']


def test_rejects_different_timelines_and_unmeasured_anchors():
    report=example()
    with pytest.raises(ValueError):explain(report,locked_frame_ids=[11])
    report['after'][0]['frame']=11
    with pytest.raises(ValueError):explain(report)


def test_saved_window_result_includes_immutable_support_blockers(tmp_path):
    import json,numpy as np
    from test_temporal_pose import fixture,CFG
    from real2sim2real.reconstruction_repair.bundle import file_sha256
    from real2sim2real.reconstruction_repair.temporal_pose import run_window
    data=fixture();data['support_plane_camera'][:,3]=-.5
    path=tmp_path/'input.npz';np.savez_compressed(path,**data)
    provenance=tmp_path/'provenance.json'
    provenance.write_text(json.dumps(dict(schema='r2s2r_engines.temporal-object-window/1',
        source='Synthetic fixed-anchor support regression',clean_trajectory_used=False,input_sha256=file_sha256(path))))
    result=run_window(path,provenance,tmp_path/'result',CFG,diagnostic_only=True)
    assert not result['decision']['accepted']
    assert result['failure_explanation']['strict_pass_requires_anchor_or_input_revision']
    saved=json.loads((tmp_path/'result/report.json').read_text())
    assert saved['failure_explanation']==result['failure_explanation']
