import itertools
import numpy as np
import pytest
from real2sim2real.retargeting.contact_uncertainty import (
    convex_pad_samples,
    tapered_sample_evidence,
)


def test_convex_samples_follow_real_face_and_normal():
    v = np.array(list(itertools.product([-0.01, 0.01], repeat=3)))
    p, n = convex_pad_samples(v, [0, 0.01, 0], [0, 1, 0])
    assert p.shape == (5, 3) and np.allclose(p[:, 1], 0.01)
    assert np.allclose(n, [0, 1, 0])


def test_low_penetration_alone_does_not_establish_support():
    p = np.array([[0.042, 0, 0.05], [0.04, 0, 0.05], [0.036, 0, 0.05], [0.04, 0, 0.2]])
    n = np.tile([-1.0, 0, 0], (4, 1))
    r = tapered_sample_evidence(p, n, [0, 0, 0], [0.04, 0.04, 0.1])
    assert r["supported"] == [False, True, False, False]
    assert not r["physical_contact_validated"]


def test_wrong_facing_sample_is_not_contact_support():
    r = tapered_sample_evidence(
        [[0.04, 0, 0.05]], [[1.0, 0, 0]], [0, 0, 0], [0.04, 0.04, 0.1]
    )
    assert r["supported_samples"] == 0


def test_axis_sample_is_finite_but_not_wall_support():
    r = tapered_sample_evidence(
        [[0, 0, 0.05]], [[-1.0, 0, 0]], [0, 0, 0], [0.04, 0.04, 0.1]
    )
    assert r["supported_samples"] == 0 and np.isfinite(r["normal_error_deg"]).all()


@pytest.mark.parametrize(
    "counts,depth,expected",
    [
        ([1, 1, 1], 0.001, "stop_on_support"),
        ([1, 1, 0], 0.001, "continue"),
        ([1, 1, 1], 0.004, "abort_geometry"),
        ([1, 1], 0.001, "abort_invalid_observation"),
        ([1, 1, 1], float("nan"), "abort_invalid_observation"),
    ],
)
def test_stop_gate_never_accepts_unsafe_or_missing_support(counts, depth, expected):
    from real2sim2real.retargeting.contact_uncertainty import closure_stop_decision

    assert closure_stop_decision(counts, depth, True) == expected


def test_closure_frame_orientation_aborts_before_support():
    from real2sim2real.retargeting.contact_uncertainty import closure_stop_decision

    assert (
        closure_stop_decision(
            [1, 1, 1], 0.001, True, frame_normal_errors_deg=[10, 20, 25]
        )
        == "stop_on_support"
    )
    assert (
        closure_stop_decision(
            [1, 1, 1], 0.001, True, frame_normal_errors_deg=[10, 20, 25.01]
        )
        == "abort_orientation"
    )
    assert (
        closure_stop_decision(
            [0, 0, 0], 0.001, True, frame_normal_errors_deg=[26, 20, 20]
        )
        == "abort_orientation"
    )
    for angles in [[10, 20], [10, 20, float("nan")], [-1, 20, 20]]:
        assert (
            closure_stop_decision(
                [1, 1, 1], 0.001, True, frame_normal_errors_deg=angles
            )
            == "abort_invalid_observation"
        )
    assert closure_stop_decision([1, 1, 1], 0.001, True) == "stop_on_support"
