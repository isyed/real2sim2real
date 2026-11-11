"""Bounded robot-only closure schedules on an unchanged object-reference clock.

The original open/closed targets are preserved. Profiles coordinate wrists,
thumbs, opposing fingers and explicitly declared owned joint overrides within
the existing acquisition interval. Every candidate still needs path screening.
"""

from copy import deepcopy

import numpy as np


def window(value):
    v = np.asarray(value, float)
    if (
        v.shape != (2,)
        or not np.isfinite(v).all()
        or not 0 <= v[0] <= 0.5
        or not 0.5 <= v[1] <= 1
        or v[1] - v[0] < 0.4 - 1e-12
    ):
        raise ValueError("Closure windows need start 0–0.5, end 0.5–1, width >=0.4")
    return v


def validate(profile):
    if set(profile) - {"joint_windows", "pressure_activation_fraction"} != {
        "wrist_windows",
        "finger_windows",
    }:
        raise ValueError(
            "Explicit bilateral wrist and finger acquisition windows required"
        )
    if set(profile["wrist_windows"]) != {"right", "left"} or set(
        profile["finger_windows"]
    ) != {"primary", "secondary"}:
        raise ValueError("Both arm and hand identities required")
    for value in profile["wrist_windows"].values():
        window(value)
    for hand in profile["finger_windows"].values():
        if set(hand) != {"thumb", "fingers"}:
            raise ValueError("Separate thumb and opposing-finger windows required")
        for value in hand.values():
            window(value)
    for value in profile.get("joint_windows", {}).values():
        window(value)
    if "pressure_activation_fraction" in profile:
        value = profile["pressure_activation_fraction"]
        if isinstance(value, bool) or not np.isfinite(value) or not 0.5 <= value <= 1:
            raise ValueError(
                "Pressure activation must remain within 50–100% of closure"
            )


def identity():
    return dict(
        wrist_windows={r: [0.0, 1.0] for r in ("right", "left")},
        finger_windows={
            r: {g: [0.0, 1.0] for g in ("thumb", "fingers")}
            for r in ("primary", "secondary")
        },
    )


def warp(time_s, approach_s, close_s, value):
    """Robot command sampling time only; no raw object reference is resampled."""
    start, end = window(value)
    t = np.asarray(time_s, float)
    if (
        not np.isfinite(t).all()
        or not np.isfinite([approach_s, close_s]).all()
        or approach_s <= 0
        or close_s <= 0
        or np.any(t < 0)
    ):
        raise ValueError(
            "Finite nonnegative robot clock and positive acquisition durations required"
        )
    if start == 0 and end == 1:
        return t.copy()
    f = (t - approach_s) / close_s
    retimed = approach_s + close_s * np.clip((f - start) / (end - start), 0, 1)
    return np.where((t >= approach_s) & (t < approach_s + close_s), retimed, t)


def arm_targets(times, targets, arms, rollout, profile):
    validate(profile)
    times, targets = np.asarray(times, float), np.asarray(targets, float)
    if (
        times.ndim != 1
        or len(times) < 2
        or targets.shape != (len(times), sum(len(a["joints"]) for a in arms))
        or not np.isfinite(times).all()
        or not np.isfinite(targets).all()
        or np.any(np.diff(times) <= 0)
        or {a["name"] for a in arms} != {"left", "right"}
    ):
        raise ValueError(
            "Complete increasing acquisition clock and matching bilateral arm commands required"
        )
    result = targets.copy()
    cursor = 0
    for arm in arms:
        value = profile["wrist_windows"][arm["name"]]
        sampled = warp(times, rollout["approach_s"], rollout["close_s"], value)
        if not np.array_equal(sampled, times):
            for j in range(cursor, cursor + len(arm["joints"])):
                result[:, j] = np.interp(sampled, times, targets[:, j])
        cursor += len(arm["joints"])
    return result


def bind_fingers(live, model, profile):
    validate(profile)
    if live.rollout.get("finger_close_fraction", 1) != 1:
        raise ValueError(
            "Acquisition profile must exclusively own finger closure timing"
        )
    by_q = {
        int(q): (p["role"], "thumb" if p["group"] == "thumb" else "fingers")
        for p in live.servo.pads
        for q in p["qids"]
    }
    names = {
        model.joint(int(model.dof_jntid[d])).name: int(q)
        for p in live.servo.pads
        for q, d in zip(p["qids"], p["dofs"])
    }
    overrides = profile.get("joint_windows", {})
    if not set(overrides) <= set(names):
        raise ValueError("Acquisition joint overrides may own declared fingers only")
    if hasattr(live, "acquisition_profile"):
        raise ValueError("Bind acquisition once to a fresh controller")
    if "pressure_activation_fraction" in profile:
        from .contact_force_control import validate_settings

        force = deepcopy(live.settings)
        force["close_activation_fraction"] = profile["pressure_activation_fraction"]
        validate_settings(force)
    by_override = {names[n]: value for n, value in overrides.items()}
    mapped = []
    for hand in live.hands:
        values = np.array(
            [
                by_override.get(
                    int(q), profile["finger_windows"][by_q[int(q)][0]][by_q[int(q)][1]]
                )
                for q in hand["qids"]
            ],
            float,
        )
        mapped.append(None if np.all(values == [0, 1]) else values)
    for hand, values in zip(live.hands, mapped):
        if values is not None:
            hand["closing_windows"] = values.copy()
    if "pressure_activation_fraction" in profile:
        live.settings["close_activation_fraction"] = float(
            profile["pressure_activation_fraction"]
        )
    live.acquisition_profile = deepcopy(profile)


def screen(model, qids, targets, times, velocity_limits, margin):
    from .arm_path_audit import interval_speed

    indices = [np.flatnonzero(model.jnt_qposadr == q) for q in qids]
    if any(len(v) != 1 for v in indices) or len(set(map(int, qids))) != len(qids):
        raise ValueError("Distinct scalar arm coordinates required")
    joints = np.array([int(v[0]) for v in indices])
    ranges = model.jnt_range[joints]
    speed = interval_speed(targets, times, velocity_limits)
    remaining = float(
        np.min(np.minimum(targets - ranges[:, 0], ranges[:, 1] - targets))
    )
    ratio = float(max(speed))
    return dict(
        accepted=bool(remaining >= margin and ratio <= 1),
        minimum_command_joint_margin_rad=remaining,
        maximum_command_velocity_ratio=ratio,
        samples=len(times),
        object_reference_retimed=False,
        acquisition_anchor_changed=False,
        requires_full_collision_screen=True,
    )
