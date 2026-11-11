"""Audit handle openings and idealized insertion envelopes before robot IK."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from real2sim2real.artifacts import digest, write_json
from real2sim2real.retargeting import insertion_geometry
from .object_grasp_pilot import verify_scene_assets, load_plan
from .support_planning import underside_targets


def tool_traces(skeleton, translation, step):
    points, cover = insertion_geometry.polyline_samples(skeleton, step)
    return [np.asarray(translation) + p for p in points], cover


def audit(config):
    path = Path(config["source_config"])
    if digest(path) != config["source_config_sha256"]:
        raise ValueError("Planning config changed")
    source = json.loads(path.read_text())
    root, receipt = verify_scene_assets(source)
    plan = load_plan(source)
    plan_path = Path(source["output"]) / source["planning_directory"] / "plan.json"
    if digest(plan_path) != config["plan_sha256"]:
        raise ValueError("Plan changed")
    task = json.loads(Path(source["task"]).read_text())
    table = trimesh.load(root / "table.obj", force="mesh", process=False)
    mesh = trimesh.load(root / "object.obj", force="mesh", process=False)
    pose = np.asarray(plan["settling"]["settled_world_from_object"])
    mesh.apply_transform(pose)
    pieces = [
        trimesh.load(root / name, force="mesh", process=False)
        for name in receipt["assets"]
        if name.startswith("collision_") and name.endswith(".obj")
    ]
    for piece in pieces:
        piece.apply_transform(pose)
    if not mesh.is_watertight or not table.is_watertight:
        raise ValueError("Closed prepared geometry required")
    proxy = trimesh.util.concatenate(pieces)
    if (
        not 1 <= len(config["radii_m"]) <= 5
        or not 1 <= len(config["depth_levels_m"]) <= 4
        or config["headings"] not in [4, 8]
    ):
        raise ValueError("Bounded probe sweep required")
    if (
        not np.isfinite(
            [
                *config["radii_m"],
                *config["depth_levels_m"],
                config["margin_m"],
                config["tool_step_m"],
                config["path_step_m"],
            ]
        ).all()
        or min(config["radii_m"]) <= 0
        or min(config["depth_levels_m"]) <= 0
        or config["margin_m"] < 0
    ):
        raise ValueError("Finite positive probe dimensions required")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    grid = insertion_geometry.vertical_grid(mesh, **config["grid"])
    xx, yy = np.meshgrid(grid["x"], grid["y"])
    xy = np.c_[xx.ravel(), yy.ravel()]
    occupied = insertion_geometry.union_columns(pieces, xy).reshape(xx.shape)
    openings = [
        o for o in grid["openings"] if o["area_m2"] >= config["minimum_opening_area_m2"]
    ]
    for opening in openings:
        selected = grid["labels"] == opening["component"]
        opening["proxy_blocked_fraction"] = float(occupied[selected].mean())
    np.savez_compressed(
        out / "columns.npz",
        x=grid["x"],
        y=grid["y"],
        source_occupied=grid["occupied"],
        proxy_occupied=occupied,
        source_low=grid["low"],
        source_high=grid["high"],
        hole_labels=grid["labels"],
    )
    write_json(out / "openings.json", dict(openings=openings, config=config))
    print("Openings:", openings, flush=True)
    obstacles = [
        ("source_object", mesh, None),
        ("collision_proxy", proxy, pieces),
        ("table", table, None),
    ]
    rows = []
    base_radius = min(config["radii_m"])
    top = float(mesh.bounds[1, 2] + 0.04)

    def check(kind, label, translation, skeleton, **metadata):
        traces, cover = tool_traces(skeleton, translation, config["tool_step_m"])
        base = insertion_geometry.sweep_clearance(
            obstacles,
            traces,
            radius_m=base_radius,
            margin_m=config["margin_m"] + cover,
            maximum_step_m=config["path_step_m"],
        )
        variants = []
        for radius in config["radii_m"]:
            checks = [
                dict(
                    r,
                    sampled_clearance_m=r["sampled_clearance_m"] + base_radius - radius,
                    conservative_clearance_m=r["conservative_clearance_m"]
                    + base_radius
                    - radius,
                    accepted=r["conservative_clearance_m"] + base_radius - radius >= 0,
                )
                for r in base["obstacles"]
            ]
            lower = min(r["conservative_clearance_m"] for r in checks)
            variants.append(
                dict(
                    radius_m=radius,
                    accepted=all(r["accepted"] for r in checks),
                    obstacles=checks,
                    survives_geometry_inflation_m={
                        str(v): lower >= v for v in config["inflation_m"]
                    },
                )
            )
        row = dict(
            kind=kind,
            label=label,
            translation_path_m=np.asarray(translation).tolist(),
            skeleton_m=np.asarray(skeleton).tolist(),
            tool_skeleton_cover_m=cover,
            path_cover_m=base["centerline_cover_radius_m"],
            variants=variants,
            robot_geometry_validated=False,
            physical_success_validated=False,
            **metadata,
        )
        rows.append(row)
        print(label, [(v["radius_m"], v["accepted"]) for v in variants], flush=True)

    for opening in openings[:1]:
        center = np.asarray(opening["center_xy_m"])
        zmid = float(np.mean(mesh.bounds[:, 2]))
        for z in config["depth_levels_m"]:
            check(
                "straight_probe",
                f"straight-z{z:g}",
                [np.r_[center, top], np.r_[center, z]],
                [[0, 0, 0], [0, 0, config["shank_length_m"]]],
            )
        for yaw in np.arange(config["headings"]) * 2 * np.pi / config["headings"]:
            direction = np.array([np.cos(yaw), np.sin(yaw)])
            radii = np.arange(grid["pitch_m"], max(mesh.extents[:2]), grid["pitch_m"])
            radial = center + radii[:, None] * direction
            col = np.rint((radial[:, 0] - grid["x"][0]) / grid["pitch_m"]).astype(int)
            row = np.rint((radial[:, 1] - grid["y"][0]) / grid["pitch_m"]).astype(int)
            valid = (
                (row >= 0)
                & (row < len(grid["y"]))
                & (col >= 0)
                & (col < len(grid["x"]))
            )
            hits = np.flatnonzero(
                valid
                & grid["occupied"][
                    np.clip(row, 0, len(grid["y"]) - 1),
                    np.clip(col, 0, len(grid["x"]) - 1),
                ]
            )
            if not len(hits):
                continue
            toe = center + (radii[hits[0]] + config["engagement_overlap_m"]) * direction
            under = underside_targets(mesh, toe, np.array([1.0, 0, 0]), [0])
            start_xy = center - direction * config["toe_length_m"] / 2
            end_xy = toe - direction * config["toe_length_m"]
            skeleton = [
                np.array([0, 0, config["shank_length_m"]]),
                np.zeros(3),
                np.r_[direction * config["toe_length_m"], 0],
            ]
            for z in config["depth_levels_m"]:
                check(
                    "hook_envelope",
                    f"hook-yaw{round(np.rad2deg(yaw))}-z{z:g}",
                    [np.r_[start_xy, top], np.r_[start_xy, z], np.r_[end_xy, z]],
                    skeleton,
                    intended_underside_point_m=None
                    if under is None
                    else under[0].tolist(),
                    final_toe_center_m=np.r_[toe, z].tolist(),
                    bearing_surface_observation_validated=False,
                )
        opening["display_center_task_m"] = [*center.tolist(), zmid]
    com = np.asarray(plan["candidates"][0]["pair"]["center_of_mass_m"])
    for yaw in [0.0, np.pi / 2]:
        approach = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        along = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        start = com - approach * (np.linalg.norm(mesh.extents[:2]) + 0.04)
        skeleton = [
            -along * config["bar_length_m"] / 2,
            along * config["bar_length_m"] / 2,
        ]
        for z in config["depth_levels_m"]:
            check(
                "broad_support_bar",
                f"bar-yaw{round(np.rad2deg(yaw))}-z{z:g}",
                [np.r_[start[:2], z], np.r_[com[:2], z]],
                skeleton,
            )
    gap_requirements = []
    for row in rows:
        under = row.get("intended_underside_point_m")
        if under is None or not row["label"].endswith(
            f"z{config['depth_levels_m'][0]:g}"
        ):
            continue
        gap = under[2] - task["table"]["top_z_m"]
        gap_requirements.append(
            dict(
                label=row["label"],
                current_gap_m=gap,
                radii=[
                    dict(
                        radius_m=r,
                        required_gap_m=2 * r + 2 * config["margin_m"],
                        minimum_additional_gap_m=max(
                            0.0, 2 * r + 2 * config["margin_m"] - gap
                        ),
                    )
                    for r in config["radii_m"]
                ],
                bound_ignores_robot_shape_and_pose_uncertainty=True,
            )
        )
    report = dict(
        schema="real2sim2real.handle-access-audit/1",
        config=config,
        openings=openings,
        paths=rows,
        hook_gap_requirements=gap_requirements,
        scene_receipt_sha256=digest(root / "receipt.json"),
        columns_sha256=digest(out / "columns.npz"),
        worker_source_hashes={
            str(Path(p)): digest(Path(p))
            for p in [__file__, insertion_geometry.__file__]
        },
        world_from_object=pose.tolist(),
        source_mesh_changed=False,
        table_changed=False,
        hand_ik_run=False,
        observation_support_established=False,
        control_ready=False,
    )
    write_json(out / "audit.json", report)
    for name, mask in [("source", grid["occupied"]), ("proxy", occupied)]:
        pixels = np.full((*mask.shape, 3), 238, np.uint8)
        pixels[mask] = [83, 105, 120]
        pixels[grid["labels"] > 0] = [60, 190, 200]
        if name == "proxy":
            pixels[(grid["labels"] > 0) & mask] = [220, 90, 60]
        picture = Image.fromarray(pixels[::-1])
        picture = picture.resize(
            (picture.width * 4, picture.height * 4), resample=Image.Resampling.NEAREST
        )
        ImageDraw.Draw(picture).text(
            (10, 10),
            f"{name}: sampled top-down columns; cyan=source opening",
            fill="black",
        )
        picture.save(out / f"{name}-opening.png")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.worker:
        audit(config)
    else:
        from real2sim2real.bounded_process import run

        return run(
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
