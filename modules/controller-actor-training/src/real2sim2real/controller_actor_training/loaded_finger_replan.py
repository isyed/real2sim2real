"""Fit opposing collision pads at a pinned loaded pose with wrists held fixed.

The resulting offsets are proposals for a fresh grasp trial. A fit at one saved
state cannot establish contact retention or justify commanding a real robot.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import trimesh
from real2sim2real.artifacts import digest, write_json


def rebase_wrist(model, data, base, idx, names):
    """Express a world-parented loaded wrist at zero fixture coordinates.

    Keeps all world body poses and object coordinates unchanged. Intended for
    scratch IK data, not an in-flight simulation or a dynamics restart.
    """
    import mujoco
    from .object_grasp_pilot import set_wrist_frame

    if model.body_parentid[idx["root"]] != 0:
        raise ValueError("World-parented Cartesian fixture required")
    # Rebase the fixture at the observed wrist so initial-grasp search bounds
    # do not clamp a lifted wrist. Preserve every body's world pose exactly.
    fixed_wrist = np.eye(4)
    fixed_wrist[:3, 3] = data.xpos[idx["root"]]
    fixed_wrist[:3, :3] = data.xmat[idx["root"]].reshape(3, 3)
    saved_qpos = data.qpos.copy()
    set_wrist_frame(model, data, base, fixed_wrist[:3, 3], fixed_wrist[:3, :3])
    data.qpos[:] = saved_qpos
    for i, name in enumerate(names):
        if name.startswith("fixture_"):
            data.qpos[idx["q"][i]] = 0.0
    mujoco.mj_fwdPosition(model, data)
    return fixed_wrist


def joint_search_bounds(names, limits, seed, active, maximum_offset_rad, wrist=None):
    """Intersect bounded coupled planning changes with native physical limits."""
    limits, seed, active = (
        np.asarray(limits, float),
        np.asarray(seed, float),
        set(active),
    )
    if (
        len(set(names)) != len(names)
        or limits.shape != (len(names), 2)
        or seed.shape != (len(names),)
        or not np.isfinite(limits).all()
        or not np.isfinite(seed).all()
        or not active
        or not active.issubset(names)
        or any(n.startswith("fixture_") for n in active)
        or (limits[:, 0] >= limits[:, 1]).any()
        or (seed < limits[:, 0]).any()
        or (seed > limits[:, 1]).any()
        or not 0 < maximum_offset_rad <= (0.7 if wrist is not None else 0.35)
    ):
        raise ValueError(
            "Finite legal seed, distinct owned fingers and bounded search required"
        )
    if wrist is not None and (
        not np.isfinite([wrist["translation_m"], wrist["rotation_rad"]]).all()
        or not 0 < wrist["translation_m"] <= 0.03
        or not 0 < wrist["rotation_rad"] <= 0.35
    ):
        raise ValueError(
            "Wrist search limited to 30 mm translation and 0.35 rad per axis"
        )
    movable = set(active)
    bounds = {}
    for i, name in enumerate(names):
        radius = maximum_offset_rad if name in active else 1e-7
        if wrist is not None and name.startswith("fixture_"):
            radius = (
                wrist["rotation_rad"]
                if name.startswith("fixture_r")
                else wrist["translation_m"]
            )
            movable.add(name)
        bounds[name] = [
            max(limits[i, 0], seed[i] - radius),
            min(limits[i, 1], seed[i] + radius),
        ]
    return bounds, movable


def run(config):
    import mujoco
    from .object_grasp_pilot import (
        load_scene,
        load_plan,
        set_wrist_frame,
        set_root_frame,
        object_transform,
    )
    from real2sim2real.retargeting.grasp_kinematics import robot_indices
    from real2sim2real.retargeting.pad_contacts import pad_frame, fit_pad_contacts

    source = Path(config["parent_trial"])
    if digest(source) != config["parent_trial_sha256"]:
        raise ValueError("Pinned loaded trial changed")
    parent = json.loads(source.read_text())
    if (
        not parent["success"]
        or digest(source.with_name("states.npz")) != parent["states_sha256"]
    ):
        raise ValueError("Successful unchanged loaded trial and states required")
    sample = config["sample_index"]
    if type(sample) is not int or not 0 <= sample < len(parent["records"]):
        raise ValueError("Explicit valid saved sample required")
    if (
        not 0
        < config["maximum_offset_rad"]
        <= (0.7 if config.get("wrist_search") else 0.35)
    ):
        raise ValueError(
            "Finger offsets bounded to 0.35 rad, or 0.7 rad for coupled search, required"
        )
    if not 1 <= len(config["pads"]) <= 5:
        raise ValueError("One to five declared pads required")
    base, candidate = parent["configuration"], parent["candidate"]
    model, data, mesh = load_scene(base)
    set_wrist_frame(
        model, data, base, candidate["base_position_m"], candidate["base_rotation"]
    )
    support = candidate["support"]
    set_root_frame(
        model,
        data,
        support["root_body"],
        support["fixture_prefix"],
        support["base_position_m"],
        support["base_rotation"],
    )
    with np.load(source.with_name("states.npz"), allow_pickle=False) as archive:
        data.qpos[:] = archive["qpos"][sample]
    mujoco.mj_fwdPosition(model, data)
    idx = robot_indices(model, base["robot"]["root_body"])
    names = [model.joint(int(j)).name for j in idx["joints"]]
    active = set(config["joints"])
    if len(active) != len(config["joints"]):
        raise ValueError("Distinct finger joints required")
    if (
        not active
        or not active.issubset(names)
        or any(n.startswith("fixture_") for n in active)
    ):
        raise ValueError("Owned nonfixture finger joints required")
    settled = np.asarray(load_plan(base)["settling"]["settled_world_from_object"])
    pose = object_transform(model, data)
    transform = pose @ np.linalg.inv(settled)
    if any(
        type(p["target_index"]) is not int
        or not 0 <= p["target_index"] < len(candidate["targets_m"])
        for p in config["pads"]
    ):
        raise ValueError("Explicit valid target indices required")
    points = np.asarray(
        [candidate["targets_m"][p["target_index"]] for p in config["pads"]]
    )
    targets = trimesh.transform_points(points, transform)
    mesh.apply_transform(pose)
    targets, _, faces = trimesh.proximity.closest_point(mesh, targets)
    normals = -mesh.face_normals[faces]
    frames, anchors = [], []
    for i, pad in enumerate(config["pads"]):
        direction = pad["outward_body"]
        point_body_m = None
        mode = pad.get("target_mode", "retained_target")
        if mode == "simulated_contact_anchor":
            gid = model.geom(pad["geom"]).id
            obj = model.body("task_object").id
            contacts = [
                c
                for c in data.contact
                if gid in (c.geom1, c.geom2)
                and model.geom_bodyid[c.geom2 if c.geom1 == gid else c.geom1] == obj
                and -base["planning"]["max_penetration_m"] <= c.dist <= 0
            ]
            if not contacts:
                raise ValueError(
                    "Requested collision pad has no admissible object contact anchor"
                )
            contact = min(contacts, key=lambda c: c.dist)
            hand_normal = np.asarray(contact.frame).reshape(3, 3)[0] * (
                1 if contact.geom1 == gid else -1
            )
            body = model.geom_bodyid[gid]
            point_body_m = data.xmat[body].reshape(3, 3).T @ (
                contact.pos - data.xpos[body]
            )
            direction = (data.xmat[body].reshape(3, 3).T @ hand_normal).tolist()
            near, distance, face = trimesh.proximity.closest_point(mesh, [contact.pos])
            if distance[0] > 0.003:
                raise ValueError(
                    "Contact anchor exceeds 3 mm reconstructed-surface agreement"
                )
            targets[i] = near[0]
            normals[i] = -mesh.face_normals[face[0]]
            anchors.append(
                dict(
                    pad=pad["geom"],
                    source="saved-pose collision witness, not independent observation",
                    contact_position_world_m=contact.pos.tolist(),
                    contact_distance_m=float(contact.dist),
                    source_mesh_projection_m=float(distance[0]),
                )
            )
        elif mode != "retained_target":
            raise ValueError("Unknown pad target mode")
        frames.append(
            pad_frame(
                model,
                base["robot"]["root_body"],
                pad["geom"],
                direction,
                point_body_m=point_body_m,
            )
        )
    shift = np.asarray(config.get("target_shift_world_m", [0, 0, 0]), float)
    if (
        shift.shape != (3,)
        or not np.isfinite(shift).all()
        or np.linalg.norm(shift) > 0.03
    ):
        raise ValueError("Explicit contact-layout shift bounded to 30 mm required")
    if np.any(shift):
        targets, projection, face = trimesh.proximity.closest_point(
            mesh, targets + shift
        )
        if projection.max() > 0.005:
            raise ValueError("Shifted contact proposal exceeds 5 mm surface projection")
        normals = -mesh.face_normals[face]
    fixed_wrist = rebase_wrist(model, data, base, idx, names)
    limits = model.jnt_range[idx["joints"]].copy()
    # Match existing pad fitter's fixture planning envelope, while freezing it.
    for i, name in enumerate(names):
        if name.startswith("fixture_"):
            bound = 0.35 if name.startswith("fixture_r") else 0.06
            limits[i] = max(limits[i, 0], -bound), min(limits[i, 1], bound)
    # Soft simulated stops can slightly exceed legal planning limits. Project
    # at most 0.01 rad, disclose every change, and require a new physical trial.
    measured = data.qpos[idx["q"]].copy()
    if (measured < limits[:, 0] - 0.01).any() or (measured > limits[:, 1] + 0.01).any():
        raise ValueError("Loaded pose exceeds allowed joint envelope")
    seed = np.clip(measured, limits[:, 0] + 1e-6, limits[:, 1] - 1e-6)
    wrist = config.get("wrist_search")
    bounds, movable = joint_search_bounds(
        names, limits, seed, active, config["maximum_offset_rad"], wrist
    )
    fit = fit_pad_contacts(
        model,
        data,
        base["robot"]["root_body"],
        frames,
        targets,
        normals,
        seed,
        joint_bounds=bounds,
        max_nfev=200,
    )
    offsets = {
        name: float(fit["q"][names.index(name)] - seed[names.index(name)])
        for name in config["joints"]
    }
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    rejected = []
    if fit["site_errors_m"].max() > base["planning"]["contact_error_m"]:
        rejected.append("pad_position_error")
    if fit["normal_errors_deg"].max() > 25:
        rejected.append("pad_normal_error")
    if fit["penetration_m"].max(initial=0) > base["planning"]["max_penetration_m"]:
        rejected.append("pad_penetration")
    report = dict(
        geometry_accepted=not rejected,
        rejected_for=rejected,
        schema="real2sim2real.loaded-finger-replan/1",
        config=config,
        joint_offsets_rad=offsets,
        joint_names=names,
        fitted_q=fit["q"].tolist(),
        seed_q=seed.tolist(),
        wrist_search=wrist,
        target_world_m=targets.tolist(),
        fixed_wrist_world=fixed_wrist.tolist(),
        target_pad_normals=normals.tolist(),
        pad_frames=frames,
        simulated_contact_anchors=anchors,
        target_shift_world_m=shift.tolist(),
        shifted_targets_are_proposals=bool(np.any(shift)),
        sample_time_s=parent["records"][sample]["time_s"],
        site_errors_m=fit["site_errors_m"].tolist(),
        normal_errors_deg=fit["normal_errors_deg"].tolist(),
        maximum_penetration_m=float(fit["penetration_m"].max(initial=0)),
        nfev=fit["nfev"],
        maximum_frozen_joint_change_rad=float(
            max(
                (
                    abs(fit["q"][i] - seed[i])
                    for i, name in enumerate(names)
                    if name not in movable
                ),
                default=0.0,
            )
        ),
        loaded_seed_projection_rad=(seed - measured).tolist(),
        physical_contact_validated=False,
        arm_control_validated=False,
        implementation_sha256=digest(Path(__file__)),
    )
    write_json(output / "fit.json", report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not args.worker:
        from real2sim2real.bounded_process import run as bounded

        return bounded(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(args.config.resolve()),
                "--worker",
            ],
            timeout=120,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only fitting required")
    run(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
