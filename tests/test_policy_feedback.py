from copy import deepcopy

import numpy as np
import pytest
from real2sim2real.controller_actor_training.policy_feedback import (
    first_breach,
    pad_statistics,
    retention_series,
    statistics,
)


def record(time, phase, x, wrist=0, rotation=None):
    rotation = np.eye(3) if rotation is None else rotation
    return {
        "time_s": time,
        "phase": phase,
        "object_position_m": [x, 0, 0],
        "object_rotation": np.eye(3).tolist(),
        "wrist_position_m": [wrist, 0, 0],
        "support_wrist_position_m": [wrist, 0, 0],
        "wrist_rotation": rotation.tolist(),
        "support_wrist_rotation": rotation.tolist(),
    }


def test_retention_does_not_reset_at_hold_or_count_intended_release():
    source = {
        "continuous_protocol": {
            "acquisition_end_s": 1.8,
            "finger_release": {"start_s": 5},
        },
        "records": [
            record(1, "close", 0),
            record(1.82, "lift", 0),
            record(3, "lift", 0.011),
            record(4, "hold", 0.012),
            record(5, "release", 0.2),
        ],
    }
    original = deepcopy(source)
    values = retention_series(source)
    assert [r["time_s"] for r in values] == [1.82, 3, 4]
    assert values[-1]["primary_drift_m"] == pytest.approx(0.012)
    assert first_breach(values, "primary_drift_m", 0.01)["time_s"] == 3
    assert source == original


def test_common_wrist_object_translation_is_not_slip():
    source = {
        "continuous_protocol": {
            "acquisition_end_s": 0,
            "finger_release": {"start_s": 3},
        },
        "records": [record(0, "lift", 0.1), record(1, "hold", 1.1, wrist=1)],
    }
    assert retention_series(source)[1]["primary_drift_m"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("times", [[1, 1], [2, 1], [0, float("nan")]])
def test_invalid_observation_clock_cannot_produce_feedback(times):
    source = {
        "continuous_protocol": {
            "acquisition_end_s": 0,
            "finger_release": {"start_s": 3},
        },
        "records": [record(t, "lift", 0) for t in times],
    }
    with pytest.raises(ValueError):
        retention_series(source)


def test_minimum_and_maximum_gate_direction_and_exact_limits():
    rows = [
        {"time_s": i, "phase": "hold", "value": v}
        for i, v in enumerate([0.05, 0.049, 0.06])
    ]
    assert first_breach(rows, "value", 0.05, minimum=True)["time_s"] == 1
    assert first_breach(rows, "value", 0.05)["time_s"] == 2
    assert first_breach(rows, "value", 1) is None
    with pytest.raises(ValueError):
        statistics([float("nan")])


def test_no_loaded_pad_is_missing_sliding_evidence_not_zero_slip():
    row = {
        "contact_patch_motion": {
            "pads": {"primary:thumb": {"loaded": False, "force_n": 0}}
        },
        "contact_force_feedback": {
            "primary:thumb": {
                "target_n": 8,
                "joint_offset_rad": [0],
                "normal_gap_m": 0.001,
                "tangent_gap_m": 0.002,
            }
        },
    }
    result = pad_statistics([row], "primary:thumb")
    assert result["loaded_sample_fraction"] == 0
    assert result["coherent_force_n"]["mean"] == 0
    assert result["sliding_fraction_of_loaded_samples"] is None
    assert result["loaded_tangent_speed_m_s"]["samples"] == 0
    assert result["loaded_mean_object_local_centroid_m"] is None
