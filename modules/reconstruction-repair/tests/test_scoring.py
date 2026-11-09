import numpy as np
import pytest

from real2sim2real.reconstruction_repair.cli import main
from real2sim2real.reconstruction_repair.scoring import geometry_scores, reference_scores, scores_compare

from conftest import box_geometry, observations


def test_score_identity_has_zero_deltas_and_no_fake_pen(make_bundle):
    b = make_bundle()
    s = geometry_scores(b)
    comparison = scores_compare(s, s)
    assert comparison["after_minus_before"]["silhouette_iou"]["mean"] == 0
    assert comparison["camera_center_depth_change_m"]["max"] == 0
    refs = reference_scores(b, b, samples=128)
    assert refs["symmetric_surface_chamfer_object_cm"]["max"] == 0
    assert refs["human_vertex_chamfer_cm"] is None
    assert refs["penetration_cm"] is None


def test_reference_object_and_decoded_human_error(make_bundle):
    g = box_geometry()
    hv = np.zeros((9, 4, 3))
    reference = make_bundle(g=g, human={"vertices_world": hv})
    shifted = {k: v.copy() for k, v in g.items()}
    shifted["object_translation"][:, 2] += .1
    predicted = make_bundle(g=shifted, human={"vertices_world": hv + [0, 0, .1]})
    s = reference_scores(predicted, reference, samples=128)
    assert s["object_center_error_cm"]["mean"] == pytest.approx(10)
    assert s["human_vertex_chamfer_cm"]["mean"] == pytest.approx(10)
    assert 0 < s["symmetric_surface_chamfer_object_cm"]["mean"] <= 10


def test_score_comparison_does_not_change_observation_set(make_bundle, tmp_path):
    a = make_bundle()
    o = observations(a.geometry)
    o["object_mask"] = ~o["object_mask"]
    b = make_bundle(o=o)
    assert main(["score", "--before", str(a.root), "--after", str(b.root), "--output", str(tmp_path / "bad.json")]) == 2
    assert not (tmp_path / "bad.json").exists()


def test_reference_rejects_mismatched_world_or_frames(make_bundle):
    a, b = make_bundle(), make_bundle()
    b.manifest["world_frame_id"] = "another-world"
    with pytest.raises(ValueError, match="world_frame"):
        reference_scores(a, b)
    b.manifest["world_frame_id"] = a.manifest["world_frame_id"]
    b.geometry["frame_ids"][0] -= 1
    with pytest.raises(ValueError, match="frame IDs"):
        reference_scores(a, b)


def test_motion_scores_use_seconds_not_assumed_30fps(make_bundle):
    g = box_geometry()
    g["object_translation"][:, 0] = g["timestamps_s"] * .3
    b = make_bundle(g=g)
    s = geometry_scores(b)
    assert s.summary["center_speed_m_s"]["mean"] == pytest.approx(.3)
    assert s.summary["center_acceleration_m_s2"]["max"] < 1e-12
