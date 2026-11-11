"""Bounded body/rim grasp planning for two articulated hands in one scene."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from real2sim2real.artifacts import digest, write_json


def wall_targets(mesh, center_xy, radial, offsets_m, heights_m, family):
    """Cast from outside toward a body; rim thumb uses the opposing inner wall.

    Return actual source-mesh hits and normals. The family is a geometric
    hypothesis; hidden geometry and insertion are not validated by this function.
    """
    radial = np.asarray(radial, float)
    center = np.asarray(center_xy, float)
    offsets, heights = np.asarray(offsets_m, float), np.asarray(heights_m, float)
    if (
        radial.shape != (3,)
        or center.shape != (2,)
        or not np.isfinite(np.r_[radial, center, offsets, heights]).all()
        or not np.isclose(np.linalg.norm(radial), 1)
        or abs(radial[2]) > 1e-8
        or offsets.shape != heights.shape
        or offsets.ndim != 1
        or not 2 <= len(offsets) <= 5
        or family not in ("body_cup", "rim_pinch", "rim_outer_thumb")
    ):
        raise ValueError(
            "Finite horizontal unit direction and paired pad offsets/heights required"
        )
    tangent = np.cross([0, 0, 1], radial)
    reach = np.linalg.norm(mesh.extents) + 0.1
    origins = np.array(
        [
            np.r_[center, z] + tangent * offset + radial * reach
            for offset, z in zip(offsets, heights)
        ]
    )
    hits, rays, faces = mesh.ray.intersects_location(
        origins, np.tile(-radial, (len(origins), 1)), multiple_hits=True
    )
    targets, normals = [], []
    for i in range(len(origins)):
        ids = np.flatnonzero(rays == i)
        if not len(ids):
            return None
        ids = ids[np.argsort(np.linalg.norm(hits[ids] - origins[i], axis=1))]
        index = ids[0]
        inner_contact = (family == "rim_pinch" and i == 0) or (
            family == "rim_outer_thumb" and i > 0
        )
        if inner_contact:
            outer = hits[index]
            inner = [
                j
                for j in ids[1:]
                if np.linalg.norm(hits[j] - outer) <= 0.03
                and mesh.face_normals[faces[j]] @ -radial > 0.5
            ]
            if not inner:
                return None
            index = inner[0]
        normal = mesh.face_normals[faces[index]]
        expected = -radial if inner_contact else radial
        if normal @ expected < 0.5:
            return None
        targets.append(hits[index])
        normals.append(normal)
    return np.array(targets), np.array(normals)


def fit_hand(
    base, world_mesh, settled, robot, namespace, angle, height, family, settings
):
    import mujoco
    from .object_grasp_pilot import load_scene, set_root_frame
    from .pad_grasp_planning import path_audit
    from real2sim2real.retargeting.grasp_kinematics import robot_indices
    from real2sim2real.retargeting.pad_contacts import (
        pad_frame,
        observe_frames,
        fit_pad_contacts,
        contact_patch_samples,
    )

    model, data, _ = load_scene(base)
    # Park the other hand away while fitting; pair collision is checked later.
    for other, prefix in [
        (base["robot"], "fixture_"),
        (base["support_robot"], base["support_robot"]["fixture_prefix"]),
    ]:
        set_root_frame(
            model, data, other["root_body"], prefix, [0.8, 0.8, 0.8], np.eye(3)
        )
    prefix = robot.get("fixture_prefix", "fixture_")
    root = robot["root_body"]
    set_root_frame(model, data, root, prefix, [0, 0, 0], np.eye(3))
    idx = robot_indices(model, root)
    names = [model.joint(int(j)).name for j in idx["joints"]]
    seed = np.zeros(len(names))
    for i, name in enumerate(names):
        if name.endswith(("MCP_FE", "PIP", "DIP", "IP")):
            seed[i] = settings["seed_flexion_rad"]
        if name.endswith("thumb_CMC_FE"):
            seed[i] = 1.2
    seed = np.clip(
        seed,
        model.jnt_range[idx["joints"], 0] + 1e-6,
        model.jnt_range[idx["joints"], 1] - 1e-6,
    )
    frames = [
        pad_frame(
            model,
            root,
            namespace + f"{robot['side']}_{finger}_elastomer",
            settings.get("thumb_outward_body", [0, 1, 0])
            if finger == "thumb"
            else [0, 1, 0],
        )
        for finger in ("thumb", "index", "middle", "ring", "pinky")
    ]
    data.qpos[idx["q"]] = seed
    mujoco.mj_fwdPosition(model, data)
    local, normals = observe_frames(model, data, frames, idx)
    radial = np.array([np.cos(angle), np.sin(angle), 0])
    # Root axes are embodiment declarations, not estimated from already-curled
    # distal pad normals: that can put the palms inside the vessel.
    rotation = Rotation.align_vectors(
        np.array([[0, 0, -1], -radial]),
        np.array([settings["palm_axis_body"], settings["hand_long_axis_body"]]),
    )[0].as_matrix()
    tangent = np.cross([0, 0, 1], radial)
    offsets = (local @ rotation.T) @ tangent
    offsets -= offsets.mean()
    offsets *= min(1, settings["maximum_tangent_span_m"] / np.ptp(offsets))
    if family == "body_cup":
        z = (local @ rotation.T)[:, 2]
        heights = np.clip(height + z - z.mean(), 0.015, world_mesh.bounds[1, 2] - 0.004)
    else:
        heights = np.full(5, height)
    proposal = wall_targets(
        world_mesh, world_mesh.center_mass[:2], radial, offsets, heights, family
    )
    if proposal is None:
        return dict(accepted=False, rejected_for=["missing_body_or_rim_surface"])
    targets, outward = proposal
    if (
        np.linalg.norm(targets[:, :2] - world_mesh.center_mass[:2], axis=1).max()
        > settings["maximum_contact_radius_m"]
    ):
        return dict(accepted=False, rejected_for=["contact_outside_body_region"])
    position = targets.mean(0) - rotation @ local.mean(0)
    set_root_frame(model, data, root, prefix, position, rotation)
    data.qpos[
        model.joint("task_object_free").qposadr[0] : model.joint(
            "task_object_free"
        ).qposadr[0]
        + 7
    ] = settled["object_qpos"]
    kwargs = dict(
        max_nfev=settings["max_nfev"],
        fixture_prefix=prefix,
        position_scale_m=settings["position_scale_m"],
    )
    fit = fit_pad_contacts(model, data, root, frames, targets, -outward, seed, **kwargs)
    opened = fit_pad_contacts(
        model,
        data,
        root,
        frames,
        targets + settings["opening_gap_m"] * outward,
        -outward,
        fit["q"],
        **kwargs,
    )
    closed = fit_pad_contacts(
        model,
        data,
        root,
        frames,
        targets - settings["preload_m"] * outward,
        -outward,
        fit["q"],
        **kwargs,
    )
    xyz = [names.index(prefix + a) for a in "xyz"]
    paths = []
    for offset in [
        radial * settings["approach_m"],
        np.array([0, 0, settings["approach_m"]]),
    ]:
        pre = opened["q"].copy()
        pre[xyz] += offset
        audit = path_audit(model, data, idx, [pre, opened["q"], closed["q"]])
        paths.append(dict(q_pregrasp=pre.tolist(), offset_m=offset.tolist(), **audit))
    path = min(
        paths, key=lambda p: (not p["within_joint_limits"], p["max_penetration_m"])
    )
    data.qpos[idx["q"]] = closed["q"]
    mujoco.mj_fwdPosition(model, data)
    patches = contact_patch_samples(
        model, data, frames, world_mesh, radius_m=0.004, max_surface_gap_m=0.0005
    )
    reasons = []
    if (
        max(f["site_errors_m"].max() for f in (fit, opened, closed))
        > base["planning"]["contact_error_m"]
    ):
        reasons.append("pad_position_error")
    if max(f["normal_errors_deg"].max() for f in (fit, opened, closed)) > 25:
        reasons.append("pad_normal_error")
    if path["max_penetration_m"] > base["planning"]["max_penetration_m"]:
        reasons.append("hand_path_penetration")
    if not path["within_joint_limits"]:
        reasons.append("hand_joint_limits")
    if not patches["all_pads_observed"]:
        reasons.append("missing_finite_pad_contact")
    return dict(
        root_body=root,
        fixture_prefix=prefix,
        base_position_m=position.tolist(),
        base_rotation=rotation.tolist(),
        joint_names=names,
        q_pregrasp=path["q_pregrasp"],
        q_open=opened["q"].tolist(),
        q_grasp=fit["q"].tolist(),
        q_closed=closed["q"].tolist(),
        targets_m=targets.tolist(),
        outward_normals=outward.tolist(),
        site_errors_m=fit["site_errors_m"].tolist(),
        normal_errors_deg=fit["normal_errors_deg"].tolist(),
        closed_target_errors_m=closed["site_errors_m"].tolist(),
        maximum_normal_error_deg=float(
            max(f["normal_errors_deg"].max() for f in (fit, opened, closed))
        ),
        patches=patches,
        pad_plan=dict(frames=frames),
        paths=paths,
        selected_path=path,
        reference_world_from_object=settled["settled_world_from_object"],
        accepted=not reasons,
        rejected_for=reasons,
    )


def run(config):
    from .object_grasp_pilot import load_scene, load_plan, set_root_frame
    from real2sim2real.retargeting.paired_contacts import paired_wrench_screen
    from real2sim2real.retargeting.grasp_kinematics import (
        robot_indices,
    )

    parent_path = Path(config["parent_trial"])
    if digest(parent_path) != config["parent_trial_sha256"]:
        raise ValueError("Pinned parent changed")
    parent = json.loads(parent_path.read_text())
    base = parent["configuration"]
    plan = load_plan(base)
    if (
        not parent["success"]
        or digest(parent_path.with_name("states.npz")) != parent["states_sha256"]
        or parent["candidate"] not in plan["candidates"]
    ):
        raise ValueError("Unchanged passing parent and plan required")
    settings = config["search"]
    if (
        not 1 <= len(config["proposals"]) <= 12
        or not 1 <= settings["max_nfev"] <= 200
        or not 0.0005 <= settings["position_scale_m"] <= 0.006
        or not 0 < settings["opening_gap_m"] <= 0.02
        or not 0 <= settings["preload_m"] <= 0.002
        or not 0 < settings["approach_m"] <= 0.12
        or not 0 < settings["maximum_tangent_span_m"] <= 0.1
        or not 0 < settings["maximum_contact_radius_m"] <= 0.15
        or not 0 < settings["seed_flexion_rad"] <= 1
        or np.asarray(settings["hand_long_axis_body"]).shape != (3,)
        or not np.isclose(np.linalg.norm(settings["hand_long_axis_body"]), 1)
        or np.asarray(settings["palm_axis_body"]).shape != (3,)
        or not np.isclose(np.linalg.norm(settings["palm_axis_body"]), 1)
        or abs(np.dot(settings["palm_axis_body"], settings["hand_long_axis_body"]))
        > 1e-6
    ):
        raise ValueError("Bounded bilateral search required")
    model, data, mesh = load_scene(base)
    mesh.apply_transform(plan["settling"]["settled_world_from_object"])
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", config)
    (out / "implementation").mkdir()
    for path in (Path(__file__), Path(__file__).with_name("pad_grasp_planning.py")):
        (out / "implementation" / path.name).write_bytes(path.read_bytes())
    from real2sim2real.retargeting import pad_contacts

    (out / "implementation/pad_contacts.py").write_bytes(
        Path(pad_contacts.__file__).read_bytes()
    )
    rows = []
    for number, p in enumerate(config["proposals"]):
        if not np.isfinite([p["angle_rad"], p["height_m"]]).all():
            raise ValueError("Finite declared proposal required")
        hands = []
        for side, robot, namespace, angle in [
            ("right", base["robot"], "", p["angle_rad"]),
            (
                "left",
                base["support_robot"],
                base["support_robot"]["namespace"],
                p["angle_rad"] + np.pi,
            ),
        ]:
            hand = fit_hand(
                base,
                mesh,
                plan["settling"],
                dict(robot, side=side),
                namespace,
                angle,
                p["height_m"],
                p["family"],
                settings,
            )
            hands.append(hand)
        reasons = [
            side + ":" + r
            for side, h in zip(["right", "left"], hands)
            for r in h["rejected_for"]
        ]
        wrench = dict(feasible=False, reason="missing_hand_geometry")
        penetration = None
        if all("patches" in h for h in hands):
            wrench = paired_wrench_screen(
                hands[0]["patches"],
                hands[1]["patches"],
                mesh.center_mass,
                base["physics"]["mass_kg"],
                base["physics"]["friction"],
                config["maximum_normal_force_n"],
            )
            if not wrench["feasible"]:
                reasons.append("paired_static_gravity_balance")
            # Check both articulated hands together throughout synchronous phases.
            model, data, _ = load_scene(base)
            for h in hands:
                set_root_frame(
                    model,
                    data,
                    h["root_body"],
                    h["fixture_prefix"],
                    h["base_position_m"],
                    h["base_rotation"],
                )
            objq = model.joint("task_object_free").qposadr[0]
            data.qpos[objq : objq + 7] = plan["settling"]["object_qpos"]
            from .pad_grasp_planning import path_audit

            idx = [robot_indices(model, h["root_body"]) for h in hands]
            combined = {
                key: np.concatenate([ids[key] for ids in idx])
                for key in ("joints", "q", "dofs", "geoms")
            }
            pair_path = path_audit(
                model,
                data,
                combined,
                [
                    np.concatenate([h[key] for h in hands])
                    for key in ("q_pregrasp", "q_open", "q_closed")
                ],
            )
            penetration = pair_path["max_penetration_m"]
            if penetration > base["planning"]["max_penetration_m"]:
                reasons.append("paired_path_collision")
        row = dict(
            candidate_id=number,
            proposal=p,
            hands=hands,
            gravity_wrench=wrench,
            paired_path_max_penetration_m=penetration,
            accepted=not reasons,
            rejected_for=reasons,
            physical_success_validated=False,
        )
        rows.append(row)
        write_json(out / "progress.json", dict(candidates=rows))
        print(number, p, "accepted", row["accepted"], reasons, flush=True)
    report = dict(
        schema="real2sim2real.bilateral-grasp-plan/1",
        config=config,
        candidates=rows,
        accepted=sum(r["accepted"] for r in rows),
        parent_sha256=config["parent_trial_sha256"],
        scene_receipt_sha256=plan["scene_receipt_sha256"],
        settling=plan["settling"],
        baseline_replaced=False,
        physical_success_validated=False,
        implementation_sha256=digest(out / "implementation/bilateral_grasp.py"),
    )
    write_json(out / "plan.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    if not a.worker:
        from real2sim2real.bounded_process import run as bounded

        return bounded(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(a.config.resolve()),
                "--worker",
            ],
            timeout=900,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only planning required")
    run(json.loads(a.config.read_text()))


def rebase_closed_wrist(model, data, hand):
    """Re-express grasp waypoints around the closed wrist without moving them.

    Scalar XYZ translations are world-aligned; the three local hinge axes use
    MuJoCo intrinsic XYZ order. Verify every reconstructed wrist pose by FK.
    This changes interpolation coordinates, so the path must be checked again.
    """
    import mujoco
    from .object_grasp_pilot import set_root_frame
    from real2sim2real.retargeting.grasp_kinematics import robot_indices

    saved_qpos = data.qpos.copy()
    hand = dict(hand)
    idx = robot_indices(model, hand["root_body"])
    if model.body_parentid[idx["root"]] != 0:
        raise ValueError("World-parented fixture required")
    xyz = [hand["joint_names"].index(hand["fixture_prefix"] + a) for a in "xyz"]
    rot = [hand["joint_names"].index(hand["fixture_prefix"] + "r" + a) for a in "xyz"]
    set_root_frame(
        model,
        data,
        hand["root_body"],
        hand["fixture_prefix"],
        hand["base_position_m"],
        hand["base_rotation"],
    )
    data.qpos[:] = saved_qpos
    poses = {}
    for key in ["q_pregrasp", "q_open", "q_grasp", "q_closed"]:
        data.qpos[idx["q"]] = hand[key]
        mujoco.mj_fwdPosition(model, data)
        poses[key] = (
            data.xpos[idx["root"]].copy(),
            data.xmat[idx["root"]].reshape(3, 3).copy(),
        )
    position, rotation = poses["q_closed"]
    set_root_frame(
        model, data, hand["root_body"], hand["fixture_prefix"], position, rotation
    )
    data.qpos[:] = saved_qpos
    errors = []
    for key, (p, r) in poses.items():
        q = np.array(hand[key])
        q[xyz] = p - position
        q[rot] = Rotation.from_matrix(rotation.T @ r).as_euler("XYZ")
        data.qpos[idx["q"]] = q
        mujoco.mj_fwdPosition(model, data)
        error = max(
            np.linalg.norm(data.xpos[idx["root"]] - p),
            np.linalg.norm(data.xmat[idx["root"]].reshape(3, 3) - r),
        )
        if error > 1e-8:
            raise ValueError("Rebased fixture does not preserve wrist pose")
        hand[key] = q.tolist()
        errors.append(float(error))
    hand["base_position_m"] = position.tolist()
    hand["base_rotation"] = rotation.tolist()
    hand["rebase_maximum_pose_residual"] = max(errors)
    return hand


if __name__ == "__main__":
    main()
