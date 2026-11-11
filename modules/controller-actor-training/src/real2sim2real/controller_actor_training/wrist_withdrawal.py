"""Small world-frame retreat after table-supported finger unhooking."""

import numpy as np


def validate_withdrawal(settings, release):
    if set(settings) - {
        "rotation_vectors_world_rad",
        "rotation_delay_s",
        "rotation_duration_s",
    } != {
        "delay_s",
        "duration_s",
        "vectors_world_m",
    }:
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
    rotations = settings.get("rotation_vectors_world_rad")
    if "rotation_delay_s" in settings or "rotation_duration_s" in settings:
        if rotations is None or not {"rotation_delay_s", "rotation_duration_s"} <= set(
            settings
        ):
            raise ValueError(
                "Separate wrist rotation needs explicit rotation vectors and both clock fields"
            )
        delay, duration = settings["rotation_delay_s"], settings["rotation_duration_s"]
        if (
            not np.isfinite([delay, duration]).all()
            or not 0.1 <= delay <= 1
            or not 0.25 <= duration <= 1.5
            or delay + duration > release["duration_s"]
        ):
            raise ValueError("Rotation clock must fit inside supported finger release")
    if rotations is not None:
        if set(rotations) != {"right", "left"}:
            raise ValueError("Both world-frame wrist rotation vectors required")
        for vector in rotations.values():
            value = np.asarray(vector, float)
            if (
                value.shape != (3,)
                or not np.isfinite(value).all()
                or np.linalg.norm(value) > 0.2
            ):
                raise ValueError(
                    "Wrist rotation is limited to 0.2 rad about the wrist origin"
                )


def offset(time, role, settings, release):
    from .object_grasp_pilot import smoothstep

    start = release["start_s"] + settings["delay_s"]
    fraction = smoothstep(np.clip((time - start) / settings["duration_s"], 0, 1))
    return fraction * np.asarray(settings["vectors_world_m"][role], float)


def transform(time, role, pose, settings, release):
    """World-axis rotation about the wrist origin plus bounded translation."""
    from scipy.spatial.transform import Rotation
    from .object_grasp_pilot import smoothstep

    result = np.array(pose, float, copy=True)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError("Finite world-frame wrist pose required")
    result[:3, 3] += offset(time, role, settings, release)
    if settings.get("rotation_vectors_world_rad"):
        start = release["start_s"] + settings.get(
            "rotation_delay_s", settings["delay_s"]
        )
        duration = settings.get("rotation_duration_s", settings["duration_s"])
        fraction = smoothstep(np.clip((time - start) / duration, 0, 1))
        vector = fraction * np.asarray(
            settings["rotation_vectors_world_rad"][role], float
        )
        result[:3, :3] = Rotation.from_rotvec(vector).as_matrix() @ result[:3, :3]
    return result


def release_ready(time, settings, release, observed_start_s, *, command_lead_s=0.0):
    """Do not move a wrist if supported opening did not latch on schedule."""
    if not np.isfinite(command_lead_s) or not 0 <= command_lead_s <= 0.025 + 1e-9:
        raise ValueError("Withdrawal requires a dense command grid up to 25 ms")
    delay = settings["delay_s"]
    if any(
        np.linalg.norm(v) > 0
        for v in settings.get("rotation_vectors_world_rad", {}).values()
    ):
        delay = min(delay, settings.get("rotation_delay_s", delay))
    # Joint interpolation can begin motion on the segment preceding the analytic
    # start. Check one full command interval early, before that segment begins.
    if time < release["start_s"] + delay - command_lead_s:
        return True
    return (
        observed_start_s is not None
        and release["start_s"] <= observed_start_s <= release["start_s"] + 0.05
    )
