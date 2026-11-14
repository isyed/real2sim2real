from copy import deepcopy

import numpy as np
from real2sim2real.controller_actor_training.placement_outcome import placement_outcome


def fixture():
    records = [
        {
            "time_s": float(t),
            "joints_rad": {"arm": 0.0, "finger": 0.0},
            "wrist_position_m": [0.0, 0.0, 1.0],
            "wrist_rotation": np.eye(3).tolist(),
            "object_com_m": [0.2, 0.1, 0.05],
            "object_rotation": np.diag([1.0, -1.0, -1.0]).tolist(),
            "support_fraction": 1.0,
            "hand_force_n": 0.0,
            "object_speed_m_s": 0.0,
        }
        for t in np.arange(0.0, 1.001, 0.02)
    ]
    task = {
        "bay_xy_m": [0.2, 0.1],
        "bay_radius_m": 0.035,
        "inverted_angle_deg": 30,
        "minimum_support_fraction": 0.5,
        "maximum_hand_force_n": 0.1,
        "maximum_object_speed_m_s": 0.02,
        "stable_window_s": 0.5,
        "home_limits": {
            "arm_rad": 0.01,
            "finger_rad": 0.05,
            "wrist_m": 0.003,
            "wrist_deg": 1.0,
        },
    }
    return records, {"horizon_s": 1.0, "task": task, "arm_joints": ["arm"]}


def test_released_placement_and_same_home_are_separate_from_graph_status():
    records, args = fixture()
    result = placement_outcome(records, **args)
    assert result["task_completed"]
    assert not result["control_accepted"]
    for key, value in [
        ("hand_force_n", 0.2),
        ("support_fraction", 0.1),
        ("object_speed_m_s", 0.03),
        ("object_com_m", [1, 1, 1]),
        ("object_rotation", np.eye(3).tolist()),
    ]:
        changed = deepcopy(records)
        changed[-5][key] = value
        assert not placement_outcome(changed, **args)["task_completed"]
    records[-1]["joints_rad"]["arm"] = 0.02
    result = placement_outcome(records, **args)
    assert result["placement_completed"] and not result["task_completed"]


def test_partial_sparse_or_faulted_records_cannot_pass():
    records, args = fixture()
    assert not placement_outcome(records[:-1], **args)["task_completed"]
    assert not placement_outcome(records[::10], **args)["task_completed"]
    assert not placement_outcome(records[:30] + records[34:], **args)["task_completed"]
    assert not placement_outcome(records, fault="physical_penetration", **args)[
        "task_completed"
    ]
