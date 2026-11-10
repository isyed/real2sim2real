from copy import deepcopy

import numpy as np
import pytest

from real2sim2real.controller_actor_training.grasp_hold import (
    measure_hold,
    hold_rejections,
)


def records():
    row = dict(
        phase="hold",
        minimum_object_height_m=0.065,
        thumb_force_n=2.0,
        finger_force_n=1.0,
        rotation_change_deg=3.0,
        object_position_m=[0.0, 0.0, 0.065],
        wrist_position_m=[0.0, 0.0, 0.0],
        wrist_rotation=np.eye(3).tolist(),
        normal_contact_force_by_group_n={
            "thumb": 2.0,
            "index": 0.0,
            "pinky": 1.0,
            "palm_or_other": 5.0,
        },
    )
    return [row, deepcopy(row)]


def reject(metrics, **overrides):
    settings = dict(
        minimum_clearance_m=0.05,
        observed_max_penetration_m=0.001,
        allowed_max_penetration_m=0.003,
    )
    return hold_rejections(metrics, **{**settings, **overrides})


def test_actual_support_does_not_claim_all_planned_fingers_carry_load():
    data = records()
    data[1]["normal_contact_force_by_group_n"]["pinky"] = 0.0
    data[1]["normal_contact_force_by_group_n"]["middle"] = 1.0
    measured = measure_hold(data, table_top_m=0.0)
    assert measured["hold_contact_support"]["index"]["active_sample_fraction"] == 0.0
    assert measured["hold_contact_support"]["pinky"]["active_sample_fraction"] == 0.5
    assert (
        measured["hold_contact_support"]["palm_or_other"]["mean_normal_force_n"] == 5.0
    )
    assert not reject(measured)


def test_later_hold_failure_cannot_be_hidden_by_good_initial_samples():
    data = records()
    data[-1]["object_position_m"][0] = 0.025
    data[-1]["minimum_object_height_m"] = 0.02
    assert reject(measure_hold(data, table_top_m=0.0)) == [
        "insufficient_lift_hold",
        "hold_slip",
    ]


def test_nonfinite_observations_and_missing_hold_cannot_pass():
    data = records()
    data[0]["rotation_change_deg"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        measure_hold(data, table_top_m=0.0)
    assert reject(measure_hold([], table_top_m=0.0)) == [
        "insufficient_lift_hold",
        "missing_opposed_finger_contact",
        "hold_slip",
        "excess_object_rotation",
    ]


def test_opposed_contact_and_penetration_remain_independent_gates():
    data = records()
    data[1]["finger_force_n"] = 0.0
    assert reject(
        measure_hold(data, table_top_m=0.0), observed_max_penetration_m=0.004
    ) == ["missing_opposed_finger_contact", "robot_penetration"]
