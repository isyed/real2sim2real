import pytest

from real2sim2real.artifacts import digest,write_json
from real2sim2real.reconstruction.tracking_proposals import verify_seed


def test_registration_evidence_is_hash_bound_and_explicitly_accepted(tmp_path):
    report=tmp_path/'report.json';seed=tmp_path/'seed.json'
    write_json(report,dict(schema='real2sim2real.initial-registration/1',accepted=True,
        selected_index=0,frame=0,thresholds_relaxed=False))
    write_json(seed,dict(source='observed_rgb_silhouette_depth_fit',frame=0,
        clean_trajectory_used=False,probe_report_sha256=digest(report)))
    verify_seed(seed,report)
    write_json(report,dict(accepted=False,selected_index=None))
    with pytest.raises(ValueError,match='accepted registration'):verify_seed(seed,report)
