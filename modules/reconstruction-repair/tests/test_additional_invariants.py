import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.reconstruction_repair.geometry import valid_segments
from real2sim2real.reconstruction_repair.repairs import temporal_translation_propose
from real2sim2real.reconstruction_repair.scoring import geometry_scores

from conftest import box_geometry, observations, only


def test_uint8_locks_and_timestamp_gaps_are_real_boundaries(make_bundle):
    g = box_geometry(12)
    g["timestamps_s"][6:] += 2
    o = observations(g)
    o["locked"] = np.zeros(12, np.uint8)
    o["locked"][3] = 1
    segments = valid_segments(g, o, .2)
    assert [a.tolist() for a in segments] == [[0, 1, 2], [4, 5], [6, 7, 8, 9, 10, 11]]


def test_temporal_filter_preserves_depth_with_moving_camera(make_bundle):
    g = box_geometry(15)
    g["object_translation"][1:-1, 0] = .012 * (-1.) ** np.arange(13)
    g["object_translation"][1:-1, 2] += .005 * (-1.) ** np.arange(13)
    g["world_to_camera"][:, :3, :3] = Rotation.from_euler("y", np.linspace(-10, 10, 15)[:, None], degrees=True).as_matrix()
    b = make_bundle(g=g)
    p = temporal_translation_propose(b, g, only("temporal_translation"))
    delta = p.geometry["object_translation"] - g["object_translation"]
    np.testing.assert_allclose(delta.sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(np.einsum("ti,ti->t", g["world_to_camera"][:, 2, :3], delta), 0, atol=1e-12)


def test_one_frame_scores_do_not_invent_motion(make_bundle):
    b = make_bundle(g=box_geometry(1))
    s = geometry_scores(b)
    assert s.summary["center_acceleration_m_s2"]["mean"] is None
    assert s.summary["angular_speed_deg_s"]["mean"] is None
    assert s.summary["silhouette_iou"]["mean"] == 1
