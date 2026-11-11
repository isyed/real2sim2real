"""Small world-frame retreat after table-supported finger unhooking."""

import numpy as np


def validate_withdrawal(settings, release):
    if set(settings) != {"delay_s", "duration_s", "vectors_world_m"}:
        raise ValueError("Explicit withdrawal clock and world vectors required")
    if not np.isfinite([settings["delay_s"], settings["duration_s"]]).all() or not (
        0.25 <= settings["delay_s"] <= 1
        and 0.5 <= settings["duration_s"] <= 1.5
        and settings["delay_s"] + settings["duration_s"] <= release["duration_s"]
    ):
        raise ValueError("Withdrawal must follow unhooking and finish within release")
    if (
        release.get("thumb_duration_s") is not None
        and settings["delay_s"] < release["thumb_duration_s"] + 0.05
    ):
        raise ValueError(
            "Finish thumb unhooking before wrist retreat, including latch tolerance"
        )
    if set(settings["vectors_world_m"]) != {"right", "left"}:
        raise ValueError("Both wrist withdrawal vectors required")
    for vector in settings["vectors_world_m"].values():
        v = np.asarray(vector, float)
        if (
            v.shape != (3,)
            or not np.isfinite(v).all()
            or not 0 < np.linalg.norm(v) <= 0.03
            or v[2] < 0
        ):
            raise ValueError("Withdrawal is limited to 30 mm with no downward command")


def offset(time, role, settings, release):
    from .object_grasp_pilot import smoothstep

    start = release["start_s"] + settings["delay_s"]
    fraction = smoothstep(np.clip((time - start) / settings["duration_s"], 0, 1))
    return fraction * np.asarray(settings["vectors_world_m"][role], float)


def release_ready(time, settings, release, observed_start_s):
    """Do not move a wrist if supported opening did not latch on schedule."""
    if time < release["start_s"] + settings["delay_s"]:
        return True
    return (
        observed_start_s is not None
        and release["start_s"] <= observed_start_s <= release["start_s"] + 0.05
    )
