"""Independent placement/home verdict from recorded physics, never graph status."""

import numpy as np


def placement_state_checks(record, task):
    """Apply the frozen placement conditions to one physical observation."""
    return {
        "in_bay": bool(
            np.linalg.norm(np.array(record["object_com_m"])[:2] - task["bay_xy_m"])
            < task["bay_radius_m"]
        ),
        "inverted": bool(
            np.array(record["object_rotation"])[2, 2]
            < -np.cos(np.deg2rad(task["inverted_angle_deg"]))
        ),
        "supported": bool(
            record["support_fraction"] >= task["minimum_support_fraction"]
        ),
        "released": bool(record["hand_force_n"] < task["maximum_hand_force_n"]),
        "stationary": bool(
            record["object_speed_m_s"] < task["maximum_object_speed_m_s"]
        ),
    }


def placement_outcome(records, *, horizon_s, task, arm_joints, fault=None):
    """Evaluate a stable released placement and return to the initial home.

    Task thresholds are supplied by the frozen downstream protocol. This is an
    outcome gate, not a grasp-quality, reconstruction or control certification.
    """
    if not records:
        return {"task_completed": False, "reason": "no physics records"}
    first, last = records[0], records[-1]
    times = np.array([r["time_s"] for r in records], float)
    complete = bool(
        times[0] <= 1e-8
        and abs(times[-1] - horizon_s) < 1e-6
        and np.all(np.diff(times) > 0)
        and fault is None
    )
    window = [
        r for r in records if r["time_s"] >= horizon_s - task["stable_window_s"] - 1e-8
    ]
    maximum_gap = task.get("maximum_sample_gap_s", 0.020001)
    covered = bool(
        window
        and window[0]["time_s"] <= horizon_s - task["stable_window_s"] + maximum_gap
        and len(window) >= task.get("minimum_window_samples", 20)
        and np.all(np.diff([r["time_s"] for r in window]) <= maximum_gap)
    )
    checks = {
        "in_bay": [],
        "inverted": [],
        "supported": [],
        "released": [],
        "stationary": [],
    }
    for r in window:
        for name, value in placement_state_checks(r, task).items():
            checks[name].append(value)
    checks = {k: bool(v and all(v)) for k, v in checks.items()}
    errors = {n: abs(v - last["joints_rad"][n]) for n, v in first["joints_rad"].items()}
    arm = max((v for n, v in errors.items() if n in arm_joints), default=0.0)
    finger = max((v for n, v in errors.items() if n not in arm_joints), default=0.0)
    position = float(
        np.linalg.norm(np.array(last["wrist_position_m"]) - first["wrist_position_m"])
    )
    relative = np.array(first["wrist_rotation"]).T @ np.array(last["wrist_rotation"])
    angle = float(np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
    limits = task["home_limits"]
    home = bool(
        arm < limits["arm_rad"]
        and finger < limits["finger_rad"]
        and position < limits["wrist_m"]
        and angle < limits["wrist_deg"]
    )
    placed = complete and covered and all(checks.values())
    return {
        "complete_horizon": complete,
        "final_window_covered": covered,
        "placement_checks": checks,
        "placement_completed": placed,
        "home": {
            "passed": home,
            "arm_error_rad": arm,
            "finger_error_rad": finger,
            "wrist_error_m": position,
            "wrist_error_deg": angle,
        },
        "task_completed": bool(placed and home),
        "fault": fault,
        "control_accepted": False,
        "evaluation_scope": "physical placement and same-home outcome; strict quality is separate",
    }
