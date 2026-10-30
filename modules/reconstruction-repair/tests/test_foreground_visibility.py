import numpy as np
import pytest
from real2sim2real.reconstruction_repair.foreground_visibility import measure,audit,overlay,ForegroundConfig


def frame(z=2.,missing=False):
    mask=np.zeros((40,40),bool);mask[5:35,5:35]=not missing
    return np.full(mask.shape,z),mask,np.zeros_like(mask)


def test_stable_evidence_does_not_approve_metric_accuracy_or_geometry():
    rows=[measure(*frame(),frame_id=i) for i in range(3)];result=audit(rows,np.arange(3)/30)
    assert result['full_observation_workflow_ready']
    assert not result['reference_approved'] and not result['depth_modified']
    assert not result['rows'][0]['metric_accuracy_verified']


def test_hidden_interval_requires_reacquisition_without_inventing_poses():
    rows=[measure(*frame(missing=i in (1,2)),frame_id=i) for i in range(5)]
    result=audit(rows,np.arange(5)/30);gap=result['unavailable_observation_spans'][0]
    assert (gap['first_frame'],gap['last_frame'],gap['reacquisition_frame'])==(1,2,3)
    assert not result['full_observation_workflow_ready'] and not result['hidden_frames_filled']
    assert result['fresh_registration_needed_after_gaps']==[3]


def test_coherent_background_jump_is_flagged_without_rewriting_depth():
    rows=[measure(*frame(z),frame_id=i) for i,z in enumerate([2.,2.02,4.])]
    result=audit(rows,np.arange(3)/30)
    assert result['rows'][2]['observation_available']  # narrow within-frame IQR alone was insufficient
    assert result['temporal_depth_warning_frames']==[2] and not result['depth_modified']
    assert rows[2]['median_depth_m']==4. and 'temporal_depth_warning' not in rows[2]


def test_overlay_preserves_unobserved_pixels_and_missing_objects():
    z,m,h=frame();row=measure(z,m,h,frame_id=0);rgb=np.full((*m.shape,3),100,np.uint8);out=overlay(rgb,m,row)
    assert np.array_equal(out[~m],rgb[~m]) and not np.array_equal(out[m],rgb[m])
    _,empty,_=frame(missing=True)
    assert np.array_equal(overlay(rgb,empty,row),rgb)


def test_invalid_grid_or_timeline_is_rejected():
    z,m,h=frame()
    with pytest.raises(ValueError):measure(z[:-1],m,h,frame_id=0)
    rows=[measure(z,m,h,frame_id=i) for i in range(2)]
    with pytest.raises(ValueError):audit(rows,[0,0])
    with pytest.raises(ValueError):ForegroundConfig(min_depth_pixels=True).validate()


def test_depth_origin_separates_raw_prediction_spikes_from_alignment_warnings():
    from real2sim2real.reconstruction_repair.foreground_visibility import depth_origin
    raw=[measure(*frame(z),frame_id=i) for i,z in enumerate([2.,2.,4.,2.])]
    aligned=[measure(*frame(z),frame_id=i) for i,z in enumerate([2.,3.,6.,3.])]
    result=depth_origin(raw,aligned,np.arange(4)/30,alignment_scales=[1.,1.5,1.5,1.5])
    assert result['warnings_already_in_raw']==[2,3]
    assert result['warnings_only_after_alignment']==[1]
    assert not result['depth_modified'] and not result['true_depth_established']
    assert 'temporal_depth_warning' not in raw[0]


@pytest.mark.parametrize('scales',[[1.],[1.,0.],[1.,float('nan')]])
def test_depth_origin_rejects_invalid_alignment_provenance(scales):
    from real2sim2real.reconstruction_repair.foreground_visibility import depth_origin
    rows=[measure(*frame(),frame_id=i) for i in range(2)]
    with pytest.raises(ValueError):depth_origin(rows,rows,[0.,1/30],alignment_scales=scales)
