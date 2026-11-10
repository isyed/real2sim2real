"""Plan and audit pad contact, opening and preload before physical execution."""

import numpy as np
import trimesh
import json
from pathlib import Path

from real2sim2real.retargeting.pad_contacts import (
    pad_frame,
    fit_pad_contacts,
    gravity_wrench_screen,
)
from real2sim2real.retargeting.grasp_kinematics import contact_penetrations


def pad_closure_geometry(model, data, indices, candidate, closed):
    import mujoco
    from real2sim2real.retargeting.pad_contacts import observe_frames

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    frames = candidate["pad_plan"]["frames"]
    points = []
    for q in (candidate["q_grasp"], closed):
        scratch.qpos[indices["q"]] = q
        mujoco.mj_fwdPosition(model, scratch)
        points.append(observe_frames(model, scratch, frames, indices)[0])
    targets = np.asarray(candidate["targets_m"])
    return dict(
        sites=[f["geom"] for f in frames],
        contact_reference="collision_pad",
        planned_target_error_m=np.linalg.norm(points[0] - targets, axis=1).tolist(),
        closed_command_target_error_m=np.linalg.norm(
            points[1] - targets, axis=1
        ).tolist(),
        command_tip_displacement_m=np.linalg.norm(
            points[1] - points[0], axis=1
        ).tolist(),
        physical_contact_validated=False,
    )


def reassess_patches(config):
    """Keep every fitted path fixed; reassess force balance over real pad samples."""
    import mujoco
    from real2sim2real.artifacts import digest, write_json
    from real2sim2real.retargeting.pad_contacts import contact_patch_samples
    from real2sim2real.retargeting.grasp_kinematics import robot_indices
    from .object_grasp_pilot import load_scene, set_wrist_frame

    options = config["patch_reassessment"]
    parent = Path(options["parent_plan"])
    if digest(parent) != options["parent_sha256"]:
        raise ValueError("Source pad plan changed")
    report = json.loads(parent.read_text())
    root = Path(config["output"])
    if report["scene_receipt_sha256"] != digest(root / "scene/receipt.json"):
        raise ValueError("Pad plan scene changed")
    if report["settling"] != json.loads((root / "settling.json").read_text()):
        raise ValueError("Pad plan settled start changed")
    output = root / config["planning_directory"]
    output.mkdir(exist_ok=False)
    for candidate in report["candidates"]:
        model, data, mesh = load_scene(config)
        set_wrist_frame(
            model,
            data,
            config,
            candidate["base_position_m"],
            candidate["base_rotation"],
        )
        idx = robot_indices(model, config["robot"]["root_body"])
        qid = model.jnt_qposadr[model.joint("task_object_free").id]
        data.qpos[qid : qid + 7] = report["settling"]["object_qpos"]
        data.qpos[idx["q"]] = candidate["q_grasp"]
        mujoco.mj_fwdPosition(model, data)
        mesh.apply_transform(report["settling"]["settled_world_from_object"])
        pad = candidate["pad_plan"]
        old = pad["gravity_wrench"]
        if (
            old["mass_kg"] != config["physics"]["mass_kg"]
            or old["friction"] != config["physics"]["friction"]
            or old["max_normal_force_n"] != config["pad_planning"]["max_normal_force_n"]
        ):
            raise ValueError("Patch reassessment cannot retune force, mass or friction")
        patches = contact_patch_samples(
            model,
            data,
            pad["frames"],
            mesh,
            radius_m=options["radius_m"],
            max_surface_gap_m=options["max_surface_gap_m"],
        )
        balance = dict(feasible=False, reason="missing_nearby_opposed_pad_samples")
        if patches["all_pads_observed"]:
            balance = gravity_wrench_screen(
                patches["points_m"],
                patches["normals"],
                mesh.center_mass,
                old["mass_kg"],
                old["friction"],
                old["max_normal_force_n"],
                contact_groups=patches["groups"],
            )
        pad["point_wrench_baseline"] = old
        pad["sampled_patches"] = patches
        pad["gravity_wrench"] = balance
        for record in (pad, candidate):
            record["rejected_for"] = [
                x for x in record["rejected_for"] if x != "pad_static_gravity_balance"
            ]
            if not balance["feasible"]:
                record["rejected_for"].append("pad_static_gravity_balance")
        candidate["accepted"] = not candidate["rejected_for"]
        print(
            f"Pad patches {candidate['candidate_id']}: samples={patches['samples_per_pad']} accepted={candidate['accepted']}",
            flush=True,
        )
    report.update(
        accepted=sum(c["accepted"] for c in report["candidates"]),
        parent_plan_sha256=digest(parent),
        patch_reassessment=options,
        geometry_and_closing_paths_changed=False,
    )
    write_json(output / "plan.json", report)
    return report


def closing_target(config, candidate, legacy_target):
    if "pad_plan" not in candidate:
        return legacy_target
    if (
        config["rollout"]["close_offset_rad"] != 0
        or config.get("finger_closure")
        or config.get("finger_surface_feedback")
        or config.get("contact_feedback")
    ):
        raise ValueError(
            "Pad closing path cannot be altered by unplanned closure feedback"
        )
    result = np.asarray(candidate["q_closed"], float)
    if result.shape != np.asarray(legacy_target).shape or not np.isfinite(result).all():
        raise ValueError("Finite planned pad closure required")
    return result.copy()


def path_audit(model, data, indices, waypoints, *, max_step_rad=0.02, max_step_m=0.002):
    """Sample every segment, bounding each joint step; not continuous certification."""
    import mujoco

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    bounds = model.jnt_range[indices["joints"]]
    q = np.asarray(waypoints, float)
    if (
        q.ndim != 2
        or q.shape[1] != len(bounds)
        or len(q) < 2
        or not np.isfinite(q).all()
        or not 0 < max_step_rad <= 0.02
        or not 0 < max_step_m <= 0.002
    ):
        raise ValueError("Finite path and bounded sampling steps required")
    step = np.where(
        model.jnt_type[indices["joints"]] == mujoco.mjtJoint.mjJNT_SLIDE,
        max_step_m,
        max_step_rad,
    )
    max_pen, samples = 0.0, 0
    for a, b in zip(q[:-1], q[1:]):
        count = max(2, int(np.ceil(np.max(abs(b - a) / step))) + 1)
        if count > 512:
            raise ValueError("Path exceeds bounded sampling budget")
        for value in np.linspace(a, b, count):
            scratch.qpos[indices["q"]] = value
            mujoco.mj_fwdPosition(model, scratch)
            max_pen = max(
                max_pen,
                float(contact_penetrations(model, scratch, indices).max(initial=0)),
            )
            samples += 1
    return dict(
        max_penetration_m=max_pen,
        samples=samples,
        within_joint_limits=bool(((q >= bounds[:, 0]) & (q <= bounds[:, 1])).all()),
        continuous_collision_certified=False,
    )


def fit_candidate(model, data, config, world, targets, seed, idx, z_index):
    settings = config["pad_planning"]
    if len(settings["pads"]) != len(targets):
        raise ValueError("One declared pad per contact required")
    if (
        len({p["geom"] for p in settings["pads"]}) != len(targets)
        or not 0 < settings["max_normal_error_deg"] <= 45
        or not 0 < settings["max_normal_force_n"] <= 50
    ):
        raise ValueError("Distinct pads and bounded alignment/force settings required")
    gap, preload = settings["opening_gap_m"], settings["preload_m"]
    if not 0 < gap <= 0.02 or not 0 <= preload <= 0.002:
        raise ValueError("Bounded opening and preload required")
    frames = [
        pad_frame(
            model,
            config["robot"]["root_body"],
            p["geom"],
            p["outward_body"],
            point_body_m=p.get("point_body_m"),
        )
        for p in settings["pads"]
    ]
    raw_targets = np.asarray(targets).copy()
    targets, distance, faces = trimesh.proximity.closest_point(world, raw_targets)
    if distance.max() > 0.0005:
        raise ValueError("Ray proposals exceed the 0.5mm surface projection budget")
    outward = world.face_normals[faces]
    kwargs = dict(
        max_nfev=config["planning"]["max_nfev"],
        joint_bounds=settings.get("joint_bounds"),
    )
    root = config["robot"]["root_body"]
    fit = fit_pad_contacts(model, data, root, frames, targets, -outward, seed, **kwargs)
    opened = fit_pad_contacts(
        model, data, root, frames, targets + gap * outward, -outward, fit["q"], **kwargs
    )
    closed = fit_pad_contacts(
        model,
        data,
        root,
        frames,
        targets - preload * outward,
        -outward,
        fit["q"],
        **kwargs,
    )
    pre = opened["q"].copy()
    pre[z_index] += config["rollout"]["approach_offset_m"]
    path = path_audit(model, data, idx, [pre, opened["q"], closed["q"]])
    wrench = gravity_wrench_screen(
        targets,
        outward,
        world.center_mass,
        config["physics"]["mass_kg"],
        config["physics"]["friction"],
        settings["max_normal_force_n"],
    )
    reasons = []
    if (
        max(x["normal_errors_deg"].max() for x in (fit, opened, closed))
        > settings["max_normal_error_deg"]
    ):
        reasons.append("pad_normal_alignment")
    if (
        max(x["site_errors_m"].max() for x in (opened, closed))
        > config["planning"]["contact_error_m"]
    ):
        reasons.append("pad_open_or_close_target_error")
    if path["max_penetration_m"] > config["planning"]["max_penetration_m"]:
        reasons.append("pad_closing_path_penetration")
    if not path["within_joint_limits"]:
        reasons.append("pad_path_joint_limits")
    if not wrench["feasible"]:
        reasons.append("pad_static_gravity_balance")
    report = dict(
        frames=frames,
        object_outward_normals=outward.tolist(),
        raw_ray_targets_m=raw_targets.tolist(),
        surface_projection_m=distance.tolist(),
        opening_gap_m=gap,
        preload_m=preload,
        grasp_normal_errors_deg=fit["normal_errors_deg"].tolist(),
        open_normal_errors_deg=opened["normal_errors_deg"].tolist(),
        closed_normal_errors_deg=closed["normal_errors_deg"].tolist(),
        open_target_errors_m=opened["site_errors_m"].tolist(),
        closed_target_errors_m=closed["site_errors_m"].tolist(),
        path=path,
        gravity_wrench=wrench,
        rejected_for=reasons,
        physical_success_validated=False,
    )
    return (
        fit,
        opened["q"],
        pre,
        dict(pad_plan=report, q_closed=closed["q"].tolist()),
        targets,
    )
