import pytest
from real2sim2real.reconstruction_repair.depth_preview import warning_frames


def test_context_includes_onset_and_sustained_bias_rather_than_only_adjacent_frames():
    report=dict(source_frames=870,temporal_depth_warning_frames=[230,231,235])
    assert warning_frames(report,30)==[229,230,235,260,280,320]


def test_endpoints_and_short_clips_preserve_unique_source_ids():
    assert warning_frames(dict(source_frames=3,temporal_depth_warning_frames=[2]),30)==[1,2]
    assert warning_frames(dict(source_frames=1,temporal_depth_warning_frames=[]),30)==[0]


@pytest.mark.parametrize('report,fps',[(dict(source_frames=8,temporal_depth_warning_frames=[8]),30),
    (dict(source_frames=8,temporal_depth_warning_frames=[True]),30),
    (dict(source_frames=8,temporal_depth_warning_frames=[]),0)])
def test_invalid_coverage_or_sampling_rate_is_rejected(report,fps):
    with pytest.raises(ValueError):warning_frames(report,fps)
