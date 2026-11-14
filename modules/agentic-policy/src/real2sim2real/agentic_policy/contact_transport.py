"""Opt-in transport screening for an already acquired articulated grasp."""

import numpy as np

from .transport_screen_audit import audit_transport_screen, contact_groups


def contact_evidence(skills, observation):
    forces = np.asarray(observation["pad_forces_n"], float)
    s = skills.session
    loaded = [
        g
        for g, f in zip(s.pad_ids, forces)
        if f >= skills.checks["minimum_pad_force_n"]
    ]
    return {
        "eligible": bool(
            skills.anchor is not None
            and not s.fault
            and len(forces) == len(s.pad_ids)
            and np.isfinite(forces).all()
            and len(loaded) >= skills.checks["minimum_contact_pads"]
        ),
        "loaded_pad_ids": loaded,
        "scope": "Current contact evidence, not force closure or complete hold acceptance",
    }


def screen_transport(skills, positions, durations_s, observation):
    s = skills.session
    s._validate_path(positions, durations_s)
    evidence = contact_evidence(skills, observation)
    if not evidence["eligible"]:
        return {
            "admitted": False,
            "reason": "acquired pad-contact evidence required",
            "evidence": evidence,
        }
    audit = audit_transport_screen(
        s,
        positions,
        durations_s,
        stationary_pad_ids=evidence["loaded_pad_ids"],
        carried_pad_ids=evidence["loaded_pad_ids"],
    )
    checked = [
        value
        for groups in audit["groups"].values()
        for name, value in groups.items()
        if name != "stationary_pad_contacts"
    ]
    worst = max(checked, key=lambda value: value["peak_penetration_m"])
    result = {
        "admitted": worst["peak_penetration_m"] <= s.config["penetration_limit_m"],
        "peak_penetration_m": worst["peak_penetration_m"],
        "witness": worst["witness"],
        "evidence": evidence,
        "audit": audit,
        "assumption": "Only loaded pad/cup pairs are exempt from the stationary-object hypothesis; all carried contacts and other pairs remain checked.",
    }
    # A short loaded forecast may resolve intended-pad-only kinematic overlap.
    # External/non-pad collision screens never use this exception.
    if (
        not result["admitted"]
        and getattr(skills, "transport_mode", None) == "contact_aware_preview"
    ):
        others = [
            value
            for groups in audit["groups"].values()
            for name, value in groups.items()
            if name not in {"stationary_pad_contacts", "carried_pad_contacts"}
        ]
        if all(
            v["peak_penetration_m"] <= s.config["penetration_limit_m"] for v in others
        ):
            preview = preview_transport(skills, positions, durations_s)
            result["dynamic_preview"] = preview
            result["admitted"] = preview["passed"]
    return result


def preview_transport(skills, positions, durations_s):
    """Bounded closed-loop forecast in an independent copy, never a live retry."""
    from types import SimpleNamespace

    from .manipulation import relative_pose

    if sum(durations_s) > 0.5 + 1e-9:
        return {"passed": False, "reason": "preview limited to 0.5 s segments"}
    if skills.transport_previews >= 64:
        return {"passed": False, "reason": "transport preview budget exhausted"}
    skills.transport_previews += 1
    fork = skills.session._fork_for_preview()
    copy_skills = SimpleNamespace(
        session=fork,
        checks=skills.checks.copy(),
        anchor=skills.anchor.copy(),
        events=[],
    )
    # observe is performed on the independent copy; it cannot warmstart live data.
    guard = retention_guard(copy_skills, relative_pose(fork.observe()))
    start = fork.data.time
    error = None
    try:
        result = fork._execute_screened_path(
            positions, durations_s, feedback_guard=guard
        )
        error = guard(result["observation"])
    except (RuntimeError, ValueError) as exception:
        error = str(exception)
    peak = max((row["penetration_m"] for row in fork.history), default=0.0)
    report = {
        "passed": error is None and fork.fault is None,
        "reason": error or fork.fault,
        "preview_id": skills.transport_previews,
        "simulated_s": float(fork.data.time - start),
        "live_physics_steps": 0,
        "peak_sampled_penetration_m": peak,
        "quality_warnings": copy_skills.events,
        "final_state": {
            "qpos": fork.data.qpos.tolist(),
            "qvel": fork.data.qvel.tolist(),
            "ctrl": fork.data.ctrl.tolist(),
            "time_s": fork.data.time,
        },
        "scope": "Same-model loaded forecast; not independent evidence or task success",
    }
    skills.events.append({"tool": "transport_dynamic_preview", "result": report})
    return report


def retention_guard(skills, reference):
    """Read-only check on existing feedback observations; no extra forward call."""
    from .manipulation import relative_error, relative_pose

    s = skills.session

    def check(observation):
        # Include object contacts against fixed obstacle bodies, not just world geoms.
        if any(
            v["peak_penetration_m"] > s.config["penetration_limit_m"]
            for v in contact_groups(s, s.data).values()
        ):
            return "transport_physical_penetration"
        if not contact_evidence(skills, observation)["eligible"]:
            return "transport_contact_loss"
        translation, rotation = relative_error(reference, relative_pose(observation))
        if not np.isfinite([translation, rotation]).all():
            return "transport_nonfinite_retention"
        if (
            translation > skills.checks["maximum_grasp_translation_m"]
            or rotation > skills.checks["maximum_grasp_rotation_deg"]
        ):
            # Strict hold quality is separate from nominal task completion.
            # Keep the candidate's phase-dependent loading/carry abort checks.
            event = {
                "tool": "transport_retention_warning",
                "result": {
                    "time_s": observation["time_s"],
                    "translation_m": translation,
                    "rotation_deg": rotation,
                    "scope": "Segment-relative strict hold quality; not task acceptance",
                },
            }
            if (
                not skills.events
                or skills.events[-1]["tool"] != "transport_retention_warning"
            ):
                skills.events.append(event)
            else:
                previous = skills.events[-1]["result"]
                previous["last_time_s"] = observation["time_s"]
                previous["translation_m"] = max(previous["translation_m"], translation)
                previous["rotation_deg"] = max(previous["rotation_deg"], rotation)
        return None

    return check
