"""Bounded observation and local collision audit before contact-plan promotion."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from real2sim2real.artifacts import digest, write_json


def audit(config):
    from real2sim2real.egocentric.interaction_io import Reconstruction
    from real2sim2real.egocentric.camera import transform
    from real2sim2real.egocentric.contact_observations import (
        observe_contacts,
        summarize_contact,
    )
    from real2sim2real.retargeting.contact_proxy import local_span_audit
    from .object_grasp_pilot import load_scene, load_plan

    source_config = Path(config["source_config"])
    if digest(source_config) != config["source_config_sha256"]:
        raise ValueError("Pinned planning configuration changed")
    settings = json.loads(source_config.read_text())
    plan_path = Path(settings["output"]) / settings["planning_directory"] / "plan.json"
    if digest(plan_path) != config["plan_sha256"]:
        raise ValueError("Pinned contact plan changed")
    plan = load_plan(settings)
    _, _, mesh = load_scene(settings)
    frames = config["frames"]
    candidates = [
        c for c in plan["candidates"] if c["candidate_id"] in config["candidate_ids"]
    ]
    if (
        not 1 <= len(candidates) <= 16
        or len(candidates) != len(set(config["candidate_ids"]))
        or not 1 <= len(frames) <= 40
        or len(set(frames)) != len(frames)
        or any(type(i) is not int or i < 0 for i in frames)
    ):
        raise ValueError("Bounded distinct candidates and capture frames required")
    source = Reconstruction(config["reconstruction"])
    task = json.loads(Path(settings["task"]).read_text())
    if task["object"]["sha256"] != digest(source.path("mesh/metric.glb")):
        raise ValueError(
            "Contact geometry and observations belong to different objects"
        )
    for name in ["geometry.npz", "timeline.json"]:
        if digest(source.path(name)) not in task["sources"].values():
            raise ValueError(
                "Contact scene and observation poses have different provenance"
            )
    if (
        not np.isfinite(config["self_occlusion_tolerance_m"])
        or config["self_occlusion_tolerance_m"] < 0
    ):
        raise ValueError("Finite nonnegative self-occlusion tolerance required")
    if max(frames) >= source.count:
        raise ValueError("Audit frame outside capture")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    settled = np.asarray(plan["settling"]["settled_world_from_object"])
    object_from_task = np.linalg.inv(settled)
    scene_root = Path(settings["output"]) / "scene"
    proxies = [
        trimesh.load(p, force="mesh", process=True)
        for p in sorted(scene_root.glob("collision_*.obj"))
    ]
    points, normals, owners = [], [], []
    reports = []
    for c in candidates:
        p = transform(np.asarray(c["targets_m"]), object_from_task)
        n = np.asarray(c["pad_plan"]["object_outward_normals"]) @ settled[:3, :3]
        local_center = transform(
            np.asarray(c["pair"]["center_m"])[None], object_from_task
        )[0]
        axis = object_from_task[:3, :3] @ np.asarray(c["pair"]["close_axis"])
        proxy = local_span_audit(
            mesh, proxies, local_center, axis, **config["proxy_limits"]
        )
        reports.append(
            dict(candidate_id=c["candidate_id"], local_proxy=proxy, contacts=[])
        )
        for j, (point, normal) in enumerate(zip(p, n)):
            points.append(point)
            normals.append(normal)
            owners.append((len(reports) - 1, j))
    points, normals = np.asarray(points), np.asarray(normals)
    evidence = [[] for _ in points]
    hashes = {
        str(source_config): digest(source_config),
        str(plan_path): digest(plan_path),
    }
    for name in ["geometry.npz", "timeline.json", "mesh/metric.glb", "artifact.json"]:
        hashes[str(source.path(name))] = digest(source.path(name))
    for i in frames:
        camera_from_object = (
            np.linalg.inv(source.world_from_camera[i]) @ source.world_from_object[i]
        )
        depth = source.depth(i)
        mask = source.mask(i, config.get("object_mask_id", 1))
        occluder = np.zeros(mask.shape, bool)
        for identifier in config.get("occluder_ids", [2, 3]):
            occluder |= source.mask(i, identifier)
        rows = observe_contacts(
            points,
            normals,
            camera_from_object,
            source.k,
            depth,
            mask,
            occluder,
            **config["image_sampling"],
        )
        camera_object = np.linalg.inv(camera_from_object)[:3, 3]
        vectors = points - camera_object
        distances = np.linalg.norm(vectors, axis=1)
        hits, rays, _ = mesh.ray.intersects_location(
            np.tile(camera_object, (len(points), 1)),
            vectors / distances[:, None],
            multiple_hits=True,
        )
        first = np.full(len(points), np.inf)
        if len(hits):
            np.minimum.at(first, rays, np.linalg.norm(hits - camera_object, axis=1))
        rgb = Image.fromarray(source.rgb(i))
        draw = ImageDraw.Draw(rgb)
        for j, row in enumerate(rows):
            if (
                row["observed"]
                and first[j] < distances[j] - config["self_occlusion_tolerance_m"]
            ):
                row.update(
                    observed=False, reason="mesh_self_occlusion", residual_m=None
                )
            row.update(
                frame_index=i,
                frame_id=int(source.frame_ids[i]),
                timestamp_s=float(source.times[i]),
            )
            evidence[j].append(row)
            if row["pixel"] is not None:
                x, y = row["pixel"]
                if 0 <= x < rgb.width and 0 <= y < rgb.height:
                    color = "#00ddff" if row["observed"] else "#ff8800"
                    draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
                    draw.text(
                        (x + 5, y),
                        f"{candidates[owners[j][0]]['candidate_id']}:{owners[j][1]}",
                        fill=color,
                    )
        rgb.save(out / f"frame-{i:06d}.jpg", quality=88)
        identifiers = [
            config.get("object_mask_id", 1),
            *config.get("occluder_ids", [2, 3]),
        ]
        for name in [
            f"rgb/{i:06d}.png",
            f"depth/{i:06d}.npy",
            *[f"masks/{k}/{i:06d}.png" for k in identifiers],
        ]:
            hashes[str(source.path(name))] = digest(source.path(name))
    for j, ((owner, contact), rows) in enumerate(zip(owners, evidence)):
        reports[owner]["contacts"].append(
            dict(
                contact_index=contact,
                point_object_m=points[j].tolist(),
                normal_object=normals[j].tolist(),
                summary=summarize_contact(rows, **config["observation_limits"]),
                observations=rows,
            )
        )
    for row in reports:
        row["contact_region_supported"] = row["local_proxy"]["accepted"] and all(
            c["summary"]["evidence"] == "supported" for c in row["contacts"]
        )
    report = dict(
        schema="real2sim2real.contact-region-audit/1",
        config=config,
        candidates=reports,
        source_hashes=hashes,
        ground_truth_used=False,
        object_and_camera_poses_refitted=False,
        reconstruction_inputs_reused=True,
        observations_are_consistency_checks_not_independent_truth=True,
        control_ready=False,
    )
    from real2sim2real.egocentric import contact_observations
    from real2sim2real.retargeting import contact_proxy

    report["worker_source_hashes"] = {
        str(p): digest(p)
        for p in [
            Path(__file__),
            Path(contact_observations.__file__),
            Path(contact_proxy.__file__),
        ]
    }
    write_json(out / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.worker:
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
    audit(config)


if __name__ == "__main__":
    main()
