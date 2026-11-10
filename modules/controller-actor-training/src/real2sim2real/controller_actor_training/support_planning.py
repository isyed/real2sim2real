"""Plan an articulated underbody support against a pinned lifted object state."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .object_grasp_pilot import (
    load_scene,
    set_wrist_frame,
    set_root_frame,
    object_transform,
)


def copy_named_state(source_model, qpos, model, data):
    """Copy joint coordinates by name, preserving added robots' coordinates."""
    import mujoco

    for j in range(source_model.njnt):
        name = mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if target < 0 or model.jnt_type[target] != source_model.jnt_type[j]:
            raise ValueError("Baseline joint missing or type changed")
        count = (
            7
            if source_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            else 4
            if source_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL
            else 1
        )
        a, b = source_model.jnt_qposadr[j], model.jnt_qposadr[target]
        data.qpos[b : b + count] = qpos[a : a + count]


def underside_targets(mesh, center_xy, along, offsets):
    """First upward intersections are underside targets, never cavity surfaces."""
    origins = np.array(
        [np.r_[center_xy, mesh.bounds[0, 2] - 0.1] + along * o for o in offsets]
    )
    p, rays, _ = mesh.ray.intersects_location(
        origins, np.tile([0.0, 0.0, 1.0], (len(origins), 1)), multiple_hits=True
    )
    if len(np.unique(rays)) != len(origins):
        return None
    return np.array(
        [p[rays == i][np.argmin(p[rays == i, 2])] for i in range(len(origins))]
    )


def plan_support(config):
    import mujoco
    from real2sim2real.retargeting.grasp_kinematics import (
        robot_indices,
        fit_contacts,
        contact_penetrations,
    )

    path = Path(config["parent"]["trial"])
    if digest(path) != config["parent"]["trial_sha256"]:
        raise ValueError("Baseline trial changed")
    parent = json.loads(path.read_text())
    states_path = path.parent / "states.npz"
    if digest(states_path) != parent["states_sha256"]:
        raise ValueError("Baseline states changed")
    states = np.load(states_path)
    baseline_model, _, _ = load_scene(parent["configuration"])
    model, data, mesh = load_scene(config)
    primary = parent["candidate"]
    set_wrist_frame(
        model, data, config, primary["base_position_m"], primary["base_rotation"]
    )
    selected = next(i for i, r in enumerate(parent["records"]) if r["phase"] == "hold")
    copy_named_state(baseline_model, states["qpos"][selected], model, data)
    mujoco.mj_fwdPosition(model, data)
    world = mesh.copy()
    reference_transform = object_transform(model, data)
    world.apply_transform(reference_transform)
    settings, robot = config["support_planning"], config["support_robot"]
    if not (
        type(settings["yaw_samples"]) is int
        and type(settings["max_nfev"]) is int
        and np.isfinite(
            [
                settings["approach_m"],
                settings["opening_gap_m"],
                settings["seed_flexion_rad"],
            ]
        ).all()
        and 0 <= settings["seed_flexion_rad"] <= 1.0
        and 1 <= settings["yaw_samples"] <= 12
        and 0 < settings["approach_m"] <= 0.2
        and 0 < settings["opening_gap_m"] <= 0.01
        and 1 <= settings["max_nfev"] <= 400
    ):
        raise ValueError("Bounded support search required")
    idx = robot_indices(model, robot["root_body"])
    names = [model.joint(int(j)).name for j in idx["joints"]]
    xyz = [names.index(robot["fixture_prefix"] + a) for a in "xyz"]
    sites = [model.site(n).id for n in robot["sites"]]
    offsets = np.asarray(settings["finger_offsets_m"], float)
    if (
        offsets.shape != (len(sites),)
        or not np.isfinite(offsets).all()
        or (np.abs(offsets) > 0.1).any()
        or any(model.site_bodyid[site] not in idx["bodies"] for site in sites)
    ):
        raise ValueError(
            "Ordered support-owned sites and finite bounded offsets required"
        )
    output = Path(config["output"]) / config.get("planning_directory", "planning")
    output.mkdir(exist_ok=False)
    rows = []
    for n, angle in enumerate(
        np.linspace(0, 2 * np.pi, settings["yaw_samples"], endpoint=False)
    ):
        forward = np.array([np.cos(angle), np.sin(angle), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        along = np.cross(forward, up)
        rotation = np.stack([up, along, forward], axis=1)
        targets = underside_targets(world, world.center_mass[:2], along, offsets)
        if targets is None:
            continue
        set_root_frame(
            model,
            data,
            robot["root_body"],
            robot["fixture_prefix"],
            [0, 0, 0],
            np.eye(3),
        )
        mujoco.mj_resetData(model, data)
        seed = np.zeros(len(names))
        for i, name in enumerate(names):
            if name.endswith(("MCP_FE", "PIP", "DIP", "IP")):
                seed[i] = settings["seed_flexion_rad"]
            if name.endswith("thumb_CMC_FE"):
                seed[i] = 1.2
        data.qpos[idx["q"]] = seed
        mujoco.mj_fwdPosition(model, data)
        position = targets.mean(0) - rotation @ data.site_xpos[sites].mean(0)
        set_root_frame(
            model, data, robot["root_body"], robot["fixture_prefix"], position, rotation
        )
        copy_named_state(baseline_model, states["qpos"][selected], model, data)
        fit = fit_contacts(
            model,
            data,
            robot["root_body"],
            robot["sites"],
            targets,
            seed,
            max_nfev=settings["max_nfev"],
            fixture_prefix=robot["fixture_prefix"],
        )
        opened = fit["q"].copy()
        opened[xyz[2]] -= settings["opening_gap_m"]
        pre = opened.copy()
        pre[xyz] -= forward * settings["approach_m"]
        maximum = 0.0
        # Insertion and final vertical seating include every other body's contacts.
        for start, end in ((pre, opened), (opened, fit["q"])):
            count = int(np.ceil(np.linalg.norm(end[xyz] - start[xyz]) / 0.002)) + 1
            for fraction in np.linspace(0, 1, count):
                data.qpos[idx["q"]] = (1 - fraction) * start + fraction * end
                mujoco.mj_fwdPosition(model, data)
                maximum = max(
                    maximum,
                    float(contact_penetrations(model, data, idx).max(initial=0)),
                )
        parked_maximum = 0.0
        # The parked hand must also be clear during the baseline's initial lift.
        for i in range(selected + 1):
            copy_named_state(baseline_model, states["qpos"][i], model, data)
            data.qpos[idx["q"]] = pre
            mujoco.mj_fwdPosition(model, data)
            parked_maximum = max(
                parked_maximum,
                float(contact_penetrations(model, data, idx).max(initial=0)),
            )
        reasons = []
        if fit["site_errors_m"].max() > config["planning"]["contact_error_m"]:
            reasons.append("support_contact_target_error")
        if (
            fit["penetration_m"].max(initial=0)
            > config["planning"]["max_penetration_m"]
        ):
            reasons.append("support_grasp_penetration")
        if maximum > config["planning"]["max_penetration_m"]:
            reasons.append("support_approach_penetration")
        if parked_maximum > config["planning"]["max_penetration_m"]:
            reasons.append("support_parked_collision")
        support = dict(
            root_body=robot["root_body"],
            fixture_prefix=robot["fixture_prefix"],
            base_position_m=position.tolist(),
            base_rotation=rotation.tolist(),
            joint_names=names,
            q_pregrasp=pre.tolist(),
            q_open=opened.tolist(),
            q_grasp=fit["q"].tolist(),
            targets_m=targets.tolist(),
            site_errors_m=fit["site_errors_m"].tolist(),
            max_penetration_m=float(fit["penetration_m"].max(initial=0)),
            approach_max_penetration_m=maximum,
            parked_max_penetration_m=parked_maximum,
            reference_world_from_object=reference_transform.tolist(),
            nfev=fit["nfev"],
            reference_primary_q=states["qpos"][selected][
                [
                    baseline_model.jnt_qposadr[baseline_model.joint(name).id]
                    for name in primary["joint_names"]
                ]
            ].tolist(),
        )
        row = deepcopy(primary)
        row.update(
            candidate_id=n,
            baseline_primary_candidate_id=primary["candidate_id"],
            support=support,
            accepted=not reasons,
            rejected_for=reasons,
            score=float(max(fit["site_errors_m"]) + maximum),
        )
        rows.append(row)
        print(
            f"Support {n}: error={max(fit['site_errors_m']) * 1000:.1f} mm approach={maximum * 1000:.1f} mm parked={parked_maximum * 1000:.1f} mm {reasons}",
            flush=True,
        )
        write_json(output / "progress.json", dict(candidates=rows))
    report = dict(
        schema="real2sim2real.bimanual-grasp-plan/1",
        candidates=rows,
        accepted=sum(r["accepted"] for r in rows),
        settling=json.loads((Path(config["output"]) / "settling.json").read_text()),
        scene_receipt_sha256=digest(Path(config["output"]) / "scene/receipt.json"),
        parent_trial_sha256=config["parent"]["trial_sha256"],
        reference_sample_index=selected,
        reference_time_s=float(states["timestamps_s"][selected]),
        human_finger_reference_used=False,
        physical_success_validated=False,
    )
    write_json(output / "plan.json", report)
    return report
