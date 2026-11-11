"""Bounded, geometry-only planning of an opposing top/underside support grasp.

First exterior ray hits exclude cavity floors. Source surface geometry is a
hypothesis: this stage neither verifies observation support nor admits control.
"""

import argparse
import json
import inspect
import shutil
import os
from pathlib import Path
import sys

import numpy as np

from real2sim2real.artifacts import digest, write_json


def exterior_targets(
    mesh,
    xy,
    sides,
    *,
    patch_radius_m=0.004,
    maximum_height_variation_m=0.004,
    minimum_normal_z=0.7,
):
    """First hits from above/below, with a five-ray finite patch per contact.

    ``sides`` is +1 for an upper face, -1 for a lower face. A source normal must
    face the ray origin. Nearby hidden/internal faces cannot substitute for a
    missing or wrong-facing exterior patch.
    """
    xy, sides = np.asarray(xy, float), np.asarray(sides, float)
    if (
        xy.ndim != 2
        or xy.shape[1] != 2
        or not 1 <= len(xy) <= 5
        or sides.shape != (len(xy),)
        or not np.isin(sides, [-1, 1]).all()
        or not np.isfinite(xy).all()
        or not 0 < patch_radius_m <= 0.005
        or not 0 < maximum_height_variation_m <= 0.01
        or not 0.5 <= minimum_normal_z <= 1
    ):
        raise ValueError("Bounded metric patch positions and top/bottom sides required")
    stencil = patch_radius_m * np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    positions = (xy[:, None, :] + stencil).reshape(-1, 2)
    signs = np.repeat(sides, 5)
    origins = np.c_[
        positions, np.where(signs > 0, mesh.bounds[1, 2] + 0.1, mesh.bounds[0, 2] - 0.1)
    ]
    directions = np.c_[np.zeros((len(origins), 2)), -signs]
    hits, rays, faces = mesh.ray.intersects_location(
        origins, directions, multiple_hits=True
    )
    points, normals, face_ids = [], [], []
    for i in range(len(origins)):
        ids = np.flatnonzero(rays == i)
        if not len(ids):
            return dict(accepted=False, rejected_for=["missing_exterior_patch"])
        pick = ids[np.argmin(np.linalg.norm(hits[ids] - origins[i], axis=1))]
        normal = mesh.face_normals[faces[pick]]
        if normal[2] * signs[i] < minimum_normal_z:
            return dict(accepted=False, rejected_for=["wrong_facing_or_steep_exterior"])
        points.append(hits[pick])
        normals.append(normal)
        face_ids.append(int(faces[pick]))
    points, normals = (
        np.asarray(points).reshape(-1, 5, 3),
        np.asarray(normals).reshape(-1, 5, 3),
    )
    if np.ptp(points[:, :, 2], axis=1).max() > maximum_height_variation_m:
        return dict(accepted=False, rejected_for=["discontinuous_exterior_patch"])
    return dict(
        accepted=True,
        rejected_for=[],
        targets_m=points[:, 0].tolist(),
        outward_normals=normals[:, 0].tolist(),
        patch_points_m=points.tolist(),
        face_indices=np.asarray(face_ids).reshape(-1, 5).tolist(),
        patch_radius_m=patch_radius_m,
        independent_observation_validated=False,
    )


def survey_underside(mesh, xy_m, *, table_top_z_m, minimum_gap_m=0.015):
    """Screen at most 4096 declared grid positions; do not infer observation support."""
    xy = np.asarray(xy_m, float)
    if (
        xy.ndim != 2
        or xy.shape[1] != 2
        or not 1 <= len(xy) <= 4096
        or not np.isfinite(xy).all()
        or not np.isfinite(table_top_z_m)
        or not 0 < minimum_gap_m <= 0.1
    ):
        raise ValueError("Bounded metric survey and explicit table height required")
    sites = []
    for point in xy:
        result = exterior_targets(mesh, [point], [-1])
        if result["accepted"]:
            z = result["targets_m"][0][2]
            if z - table_top_z_m >= minimum_gap_m:
                sites.append(result)
    return dict(
        sampled_positions=len(xy),
        sites=sites,
        table_top_z_m=float(table_top_z_m),
        minimum_gap_m=minimum_gap_m,
        independent_observation_validated=False,
        control_admitted=False,
    )


def validate(config):
    proposals = config["proposals"]
    if (
        not 1 <= len(proposals) <= 12
        or type(config["maximum_evaluations"]) is not int
        or not 1 <= config["maximum_evaluations"] <= 200
    ):
        raise ValueError("At most twelve bounded support fits required")
    if not 0 < config["maximum_normal_force_n"] <= 10:
        raise ValueError("Explicit conservative force ceiling required")
    if not 0 < config["deadline_s"] <= 1800:
        raise ValueError("Finite support planning deadline required")
    for p in proposals:
        xy = np.asarray(p["xy_m"], float)
        if (
            xy.shape != (5, 2)
            or not np.isfinite(xy).all()
            or not np.isfinite(p["yaw_rad"])
            or p.get("orientation", "palm_up") not in ("palm_up", "vertical_wrap")
            or np.abs(xy).max() > 2
        ):
            raise ValueError("Five metric contact positions and finite yaw required")


def run(config):
    import mujoco
    from .object_grasp_pilot import load_scene, load_plan, set_root_frame
    from .pad_grasp_planning import path_audit
    from real2sim2real.retargeting.grasp_kinematics import robot_indices
    from real2sim2real.retargeting.pad_contacts import (
        observe_frames,
        fit_pad_contacts,
        contact_patch_samples,
        gravity_wrench_screen,
    )

    validate(config)
    source_path = Path(config["source_trial"]["path"])
    if digest(source_path) != config["source_trial"]["sha256"]:
        raise ValueError("Pinned fixture grasp source changed")
    parent = json.loads(source_path.read_text())
    base = parent["configuration"]
    model, data, mesh = load_scene(base)
    plan = load_plan(base)
    settled = plan["settling"]
    mesh.apply_transform(np.asarray(settled["settled_world_from_object"]))
    source_state = settled["object_qpos"]
    task = json.loads(Path(base["task"]).read_text())
    if task["up_axis"] not in ("z", "+Z") or task["units"] not in ("m", "metres"):
        raise ValueError("Metric upright tabletop required")
    table_z = float(task["table"]["top_z_m"])
    primary = parent["candidate"]
    support = primary["support"]
    primary_idx = robot_indices(model, base["robot"]["root_body"])
    robot = base["support_robot"]
    idx = robot_indices(model, robot["root_body"])
    names = [model.joint(int(j)).name for j in idx["joints"]]
    if names != support["joint_names"]:
        raise ValueError("Support joint ownership/order changed")
    frames = support["pad_plan"]["frames"]
    if len(frames) != 5:
        raise ValueError("Five admitted physical pad frames required")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    helper_paths = [
        Path(__file__),
        Path(inspect.getfile(load_scene)),
        Path(inspect.getfile(path_audit)),
        Path(inspect.getfile(robot_indices)),
        Path(inspect.getfile(fit_pad_contacts)),
    ]
    implementations = {str(path): digest(path) for path in helper_paths}
    snapshot = output / "implementation"
    snapshot.mkdir()
    for path in helper_paths:
        shutil.copyfile(path, snapshot / path.name)
    implementation_sha = implementations[str(Path(__file__))]
    if "survey_xy_m" in config:
        write_json(
            output / "surface-survey.json",
            survey_underside(mesh, config["survey_xy_m"], table_top_z_m=table_z),
        )
    rows = []
    for i, proposal in enumerate(config["proposals"]):
        row = dict(
            index=i,
            proposal=proposal,
            control_admitted=False,
            independent_observation_validated=False,
            geometry_candidate=False,
        )
        surface = exterior_targets(mesh, proposal["xy_m"], [1, -1, -1, -1, -1])
        row["surface"] = surface
        if not surface["accepted"]:
            row["rejected_for"] = surface["rejected_for"]
            rows.append(row)
            write_json(output / f"candidate-{i:02d}.json", row)
            continue
        targets, outward = (
            np.asarray(surface["targets_m"]),
            np.asarray(surface["outward_normals"]),
        )
        row["underside_table_gap_m"] = float(targets[1:, 2].min() - table_z)
        if row["underside_table_gap_m"] < 0.015:
            row["rejected_for"] = ["insufficient_initial_insertion_gap"]
            rows.append(row)
            write_json(output / f"candidate-{i:02d}.json", row)
            continue
        mujoco.mj_resetData(model, data)
        j = model.joint("task_object_free").qposadr[0]
        data.qpos[j : j + 7] = source_state
        set_root_frame(
            model,
            data,
            base["robot"]["root_body"],
            "fixture_",
            primary["base_position_m"],
            primary["base_rotation"],
        )
        data.qpos[primary_idx["q"]] = primary["q_closed"]
        set_root_frame(
            model,
            data,
            robot["root_body"],
            robot["fixture_prefix"],
            [0, 0, 0],
            np.eye(3),
        )
        seed = np.zeros(len(names))
        for k, name in enumerate(names):
            if name.endswith(("MCP_FE", "PIP", "DIP", "IP")):
                seed[k] = 0.8
            if name.endswith("thumb_CMC_FE"):
                seed[k] = 1.2
        seed = np.clip(
            seed,
            model.jnt_range[idx["joints"], 0] + 1e-6,
            model.jnt_range[idx["joints"], 1] - 1e-6,
        )
        if proposal.get("orientation", "palm_up") == "vertical_wrap":
            seed = np.asarray(support["q_closed"], float).copy()
            for k, name in enumerate(names):
                if name.startswith(robot["fixture_prefix"]):
                    seed[k] = 0
        data.qpos[idx["q"]] = seed
        mujoco.mj_fwdPosition(model, data)
        local, _ = observe_frames(model, data, frames, idx)
        yaw = proposal["yaw_rad"]
        forward = np.array([np.cos(yaw), np.sin(yaw), 0])
        up = np.array([0, 0, 1])
        rotation = np.stack([up, np.cross(forward, up), forward], axis=1)
        if proposal.get("orientation", "palm_up") == "vertical_wrap":
            rotation = np.stack([forward, np.cross(-up, forward), -up], axis=1)
        position = targets.mean(0) - rotation @ local.mean(0)
        set_root_frame(
            model, data, robot["root_body"], robot["fixture_prefix"], position, rotation
        )
        kwargs = dict(
            max_nfev=config["maximum_evaluations"],
            fixture_prefix=robot["fixture_prefix"],
            position_scale_m=0.001,
        )
        fitted = fit_pad_contacts(
            model, data, robot["root_body"], frames, targets, -outward, seed, **kwargs
        )
        opened = fit_pad_contacts(
            model,
            data,
            robot["root_body"],
            frames,
            targets + 0.004 * outward,
            -outward,
            fitted["q"],
            **kwargs,
        )
        closed = fit_pad_contacts(
            model,
            data,
            robot["root_body"],
            frames,
            targets - 0.0005 * outward,
            -outward,
            fitted["q"],
            **kwargs,
        )
        pre = opened["q"].copy()
        xyz = [names.index(robot["fixture_prefix"] + a) for a in "xyz"]
        pre[xyz] -= 0.08 * forward
        path = path_audit(model, data, idx, [pre, opened["q"], closed["q"]])
        data.qpos[idx["q"]] = closed["q"]
        mujoco.mj_fwdPosition(model, data)
        patches = contact_patch_samples(model, data, frames, mesh)
        primary_targets = np.asarray(primary["targets_m"])
        import trimesh

        _, _, faces = trimesh.proximity.closest_point(mesh, primary_targets)
        points = np.vstack([primary_targets, targets])
        normals = np.vstack([mesh.face_normals[faces], outward])
        physics = dict(
            mass_kg=base["physics"]["mass_kg"],
            max_normal_force_n=config["maximum_normal_force_n"],
        )
        force = {
            str(mu): gravity_wrench_screen(
                points, normals, mesh.center_mass, friction=mu, **physics
            )
            for mu in (0, base["physics"]["friction"])
        }
        errors = [fitted, opened, closed]
        reasons = []
        if (
            max(float(f["site_errors_m"].max()) for f in errors)
            > base["planning"]["contact_error_m"]
        ):
            reasons.append("pad_position_error")
        if max(float(f["normal_errors_deg"].max()) for f in errors) > 25:
            reasons.append("pad_normal_error")
        if path["max_penetration_m"] > base["planning"]["max_penetration_m"]:
            reasons.append("insertion_or_seating_penetration")
        if not path["within_joint_limits"]:
            reasons.append("native_joint_limits")
        if not patches["all_pads_observed"]:
            reasons.append("missing_finite_pad_contact")
        if not force[str(base["physics"]["friction"])]["feasible"]:
            reasons.append("static_gravity_wrench")
        row.update(
            rejected_for=reasons,
            geometry_candidate=not reasons,
            base_position_m=position.tolist(),
            base_rotation=rotation.tolist(),
            root_body=robot["root_body"],
            fixture_prefix=robot["fixture_prefix"],
            joint_names=names,
            q_pregrasp=pre.tolist(),
            q_open=opened["q"].tolist(),
            q_grasp=fitted["q"].tolist(),
            q_closed=closed["q"].tolist(),
            targets_m=targets.tolist(),
            outward_normals=outward.tolist(),
            pad_plan=dict(frames=frames),
            path=path,
            patches=patches,
            site_errors_m=fitted["site_errors_m"].tolist(),
            normal_errors_deg=fitted["normal_errors_deg"].tolist(),
            maximum_all_pose_error_m=max(
                float(f["site_errors_m"].max()) for f in errors
            ),
            gravity_screens=force,
            force_screen_uses_hypothesized_targets=True,
        )
        rows.append(row)
        write_json(output / f"candidate-{i:02d}.json", row)
    if any(digest(Path(path)) != sha for path, sha in implementations.items()):
        raise ValueError("Planning implementation changed during the bounded search")
    result = dict(
        implementation_sources=implementations,
        runtime=dict(mujoco=mujoco.__version__, numpy=np.__version__),
        schema="bearing-support-plan/1",
        config=config,
        implementation_sha256=implementation_sha,
        implementation_unchanged=digest(Path(__file__)) == implementation_sha,
        candidates=rows,
        geometry_candidates=sum(r["geometry_candidate"] for r in rows),
        control_admitted=False,
        arm_reachability_validated=False,
        acquisition_timing_validated=False,
        dynamic_success_validated=False,
    )
    write_json(output / "plan.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate(config)
    if args.worker:
        run(config)
    else:
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
            timeout=config["deadline_s"],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )


if __name__ == "__main__":
    main()
