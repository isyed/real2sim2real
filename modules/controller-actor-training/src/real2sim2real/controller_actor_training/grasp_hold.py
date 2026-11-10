"""Shared grasp evidence: actual support, object clearance and wrist-frame drift."""

import numpy as np

from .grasp_feedback import relative_translation_drift


def contact_support(records, *, threshold_n=0.1):
    """Describe observed normal forces; planned contact sites are not evidence."""
    if not records or not all("normal_contact_force_by_group_n" in r for r in records):
        return {}
    groups = sorted(
        set().union(*(r["normal_contact_force_by_group_n"] for r in records))
    )
    result = {}
    for group in groups:
        values = np.array(
            [r["normal_contact_force_by_group_n"].get(group, 0.0) for r in records]
        )
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Contact evidence must be finite and nonnegative")
        result[group] = dict(
            mean_normal_force_n=float(values.mean()),
            active_sample_fraction=float(np.mean(values > threshold_n)),
        )
    return result


def measure_hold(records, *, table_top_m):
    """Measure retained hold samples without claiming continuous-time guarantees."""
    hold = [r for r in records if r["phase"] == "hold"]
    if not hold:
        return dict(
            hold_minimum_clearance_m=-1.0,
            hold_opposed_contact_fraction=0.0,
            hold_slip_m=None,
            hold_relative_world_displacement_m=None,
            hold_max_rotation_deg=None,
            hold_contact_support={},
        )
    evidence = np.array(
        [
            [
                r["minimum_object_height_m"],
                r["thumb_force_n"],
                r["finger_force_n"],
                r["rotation_change_deg"],
            ]
            for r in hold
        ]
    )
    if not np.isfinite(evidence).all() or not np.isfinite(table_top_m):
        raise ValueError("Nonfinite hold evidence cannot establish a grasp")
    relative = np.array(
        [np.asarray(r["object_position_m"]) - r["wrist_position_m"] for r in hold]
    )
    return dict(
        hold_minimum_clearance_m=float(evidence[:, 0].min() - table_top_m),
        hold_opposed_contact_fraction=float(
            np.mean((evidence[:, 1] > 0.1) & (evidence[:, 2] > 0.1))
        ),
        hold_slip_m=relative_translation_drift(
            [r["object_position_m"] for r in hold],
            [r["wrist_position_m"] for r in hold],
            [r["wrist_rotation"] for r in hold],
        ),
        hold_relative_world_displacement_m=float(
            np.linalg.norm(relative - relative[0], axis=1).max()
        ),
        hold_max_rotation_deg=float(evidence[:, 3].max()),
        hold_contact_support=contact_support(hold),
    )


def hold_rejections(
    metrics,
    *,
    minimum_clearance_m,
    observed_max_penetration_m,
    allowed_max_penetration_m,
    aborted=None,
):
    """Keep nominal task gates separate from later robustness/arm admission."""
    failures = [] if aborted is None else [aborted]
    if metrics["hold_minimum_clearance_m"] < minimum_clearance_m:
        failures.append("insufficient_lift_hold")
    if metrics["hold_opposed_contact_fraction"] < 0.8:
        failures.append("missing_opposed_finger_contact")
    if observed_max_penetration_m > allowed_max_penetration_m:
        failures.append("robot_penetration")
    if metrics["hold_slip_m"] is None or metrics["hold_slip_m"] > 0.01:
        failures.append("hold_slip")
    if (
        metrics["hold_max_rotation_deg"] is None
        or metrics["hold_max_rotation_deg"] > 30
    ):
        failures.append("excess_object_rotation")
    return failures
