"""Bounded evidence-first surface survey, before robot-hand fitting."""

import argparse
from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import trimesh

from real2sim2real.artifacts import digest, write_json
from real2sim2real.egocentric import capture, stereo_contact_evidence
from real2sim2real.egocentric.camera import transform
from real2sim2real.egocentric.contact_observations import (
    observe_contacts,
    summarize_contact,
)
from real2sim2real.egocentric.interaction_io import Reconstruction
from real2sim2real.egocentric.stereo import Stereo
from real2sim2real.retargeting import contact_site_search
from real2sim2real.retargeting.pad_contacts import gravity_wrench_screen
from .stereo_contact_audit import project, visible_surface


def search(config):
    for key in [
        "capture",
        "calibration",
        "plan",
        "scene_receipt",
        "reconstruction_artifact",
    ]:
        if digest(Path(config[key])) != config[key + "_sha256"]:
            raise ValueError("Pinned surface survey input changed")
    source = Reconstruction(config["reconstruction"])
    if (
        source.path("artifact.json").resolve()
        != Path(config["reconstruction_artifact"]).resolve()
    ):
        raise ValueError("Pinned reconstruction differs")
    cap = capture.load(config["capture"])
    artifact = json.loads(source.path("artifact.json").read_text())
    if (
        artifact["metadata"]["capture_sha256"] != cap["manifest_sha256"]
        or not np.array_equal(source.frame_ids, cap["frame_ids"])
        or not np.allclose(source.times, cap["timestamps_s"], atol=1e-8, rtol=0)
    ):
        raise ValueError("Capture provenance or timeline differs")
    plan = json.loads(Path(config["plan"]).read_text())
    receipt = json.loads(Path(config["scene_receipt"]).read_text())
    if plan["scene_receipt_sha256"] != config["scene_receipt_sha256"]:
        raise ValueError("Plan scene differs")
    task_path = Path(receipt["config"]["task"])
    if digest(task_path) != receipt["sources"][str(task_path)]:
        raise ValueError("Task changed")
    task = json.loads(task_path.read_text())
    if task["object"]["sha256"] != digest(source.path("mesh/metric.glb")) or any(
        digest(source.path(k)) not in task["sources"].values()
        for k in ["geometry.npz", "timeline.json"]
    ):
        raise ValueError("Scene and reconstructed source object/poses differ")
    discovery, validation = config["discovery_frames"], config["validation_frames"]
    ids = discovery + validation
    if (
        not discovery
        or not validation
        or len(ids) > 40
        or len(ids) != len(set(ids))
        or any(type(i) is not int or not 0 <= i < source.count for i in ids)
    ):
        raise ValueError("Disjoint bounded discovery and validation frames required")
    calibration = json.loads(Path(config["calibration"]).read_text())
    if not calibration["accepted"]:
        raise ValueError("Accepted calibration required")
    left, right = [cap["streams"][name] for name in cap["stereo_pair"]]
    stereo = Stereo(
        left,
        right,
        cap["streams"][cap["target_stream"]],
        right_from_left=calibration["candidate_right_from_left"],
    )
    mesh = trimesh.load(source.path("mesh/metric.glb"), force="mesh", process=False)
    world_from_object = np.asarray(plan["settling"]["settled_world_from_object"])
    physics = plan["candidates"][0]["pad_plan"]["point_wrench_baseline"]
    mode = config.get("sampling_mode", "whole_surface")
    if mode not in ["whole_surface", "upward_force", "inner_wall", "regions"]:
        raise ValueError("Unknown surface sampling mode")
    capacity = contact_site_search.upward_force_capacity(
        mesh.face_normals @ world_from_object[:3, :3].T, physics["friction"]
    )
    eligible = capacity > 0 if mode == "upward_force" else None
    opening_source = None
    if mode == "inner_wall":
        region = config["opening_region"]
        opening_source = Path(region["report"])
        if digest(opening_source) != region["sha256"]:
            raise ValueError("Pinned opening geometry changed")
        opening = json.loads(opening_source.read_text())
        if opening["config"]["plan_sha256"] != config["plan_sha256"]:
            raise ValueError("Opening region belongs to another task/settled world")
        index = region["index"]
        if type(index) is not int or not 0 <= index < len(opening["openings"]):
            raise ValueError("Explicit valid opening index required")
        eligible = contact_site_search.inner_wall_faces(
            mesh,
            world_from_object,
            opening["openings"][index]["center_xy_m"],
            **region["parameters"],
        )
    region_counts = None
    if mode == "regions":
        region_masks = contact_site_search.region_masks(
            mesh, world_from_object, config["surface_regions"]
        )
        region_counts = {name: int(mask.sum()) for name, mask in region_masks.items()}
        sites = contact_site_search.sample_regions(
            mesh, world_from_object, config["surface_regions"], **config["sampling"]
        )
    else:
        sites = contact_site_search.sample_sites(
            mesh, eligible_faces=eligible, **config["sampling"]
        )
    points = np.array([s["point_object_m"] for s in sites])
    normals = np.array([s["normal_object"] for s in sites])
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "sites.json",
        dict(sites=sites, config=config, robot_contacts_validated=False),
    )
    evidence = [[] for _ in sites]
    hashes = {
        str(source.path(k)): digest(source.path(k))
        for k in ["geometry.npz", "timeline.json", "mesh/metric.glb", "artifact.json"]
    }
    if opening_source is not None:
        hashes[str(opening_source)] = digest(opening_source)
    shortlist = []
    for split, frames in [("discovery", discovery), ("validation", validation)]:
        indices = list(range(len(sites))) if split == "discovery" else shortlist
        if not indices:
            continue
        p, n = points[indices], normals[indices]
        videos = [
            {
                i: rgb
                for i, rgb in enumerate(capture.frames(stream, max(frames) + 1))
                if i in frames
            }
            for stream in [left, right]
        ]
        if any(len(v) != len(frames) for v in videos):
            raise ValueError("Missing capture frames")
        for frame in sorted(frames):
            a_from_object = (
                np.linalg.inv(source.world_from_camera[frame])
                @ source.world_from_object[frame]
            )
            l_from_object = np.linalg.inv(stereo.target_from_left) @ a_from_object
            r_from_object = stereo.right_from_left @ l_from_object
            rectpoints = transform(p, l_from_object) @ stereo.rotations[0].T
            uv = project(rectpoints, stereo.projections[0])
            rect = [
                cv2.remap(
                    cv2.cvtColor(v[frame], cv2.COLOR_RGB2GRAY),
                    *stereo.maps[k],
                    interpolation=cv2.INTER_LINEAR,
                )
                for k, v in enumerate(videos)
            ]
            rgb_rows = observe_contacts(
                p,
                n,
                a_from_object,
                source.k,
                source.depth(frame),
                source.mask(frame, 1),
                source.mask(frame, 2) | source.mask(frame, 3),
                radius_pixels=0,
                minimum_depth_pixels=1,
            )
            visible = visible_surface(
                mesh, p, n, np.linalg.inv(l_from_object), 0.001
            ) & visible_surface(mesh, p, n, np.linalg.inv(r_from_object), 0.001)
            for j, index in enumerate(indices):
                row = dict(
                    split=split,
                    frame_index=frame,
                    frame_id=int(source.frame_ids[frame]),
                    timestamp_s=float(source.times[frame]),
                    observed=False,
                    residual_m=None,
                )
                if (
                    rgb_rows[j]["reason"] not in ["observed", "insufficient_depth"]
                    or not visible[j]
                    or rectpoints[j, 2] <= 0
                ):
                    row.update(
                        reason="visibility_rejected",
                        rgb_visibility_reason=rgb_rows[j]["reason"],
                    )
                else:
                    match = stereo_contact_evidence.match_rectified(
                        *rect, uv[j], **config["matching"]
                    )
                    row.update(reason=match["reason"], match=match)
                    if match["accepted"]:
                        xyz = stereo_contact_evidence.triangulate_match(
                            match, stereo.projections
                        )
                        target = transform(
                            (xyz @ stereo.rotations[0])[None], stereo.target_from_left
                        )[0]
                        row.update(
                            observed=True,
                            residual_m=float(
                                target[2] - transform(p[j : j + 1], a_from_object)[0, 2]
                            ),
                            point_error_m=float(
                                np.linalg.norm(
                                    transform(
                                        target[None], np.linalg.inv(a_from_object)
                                    )[0]
                                    - p[j]
                                )
                            ),
                        )
                evidence[index].append(row)
            for name in [
                f"depth/{frame:06d}.npy",
                *[f"masks/{i}/{frame:06d}.png" for i in [1, 2, 3]],
            ]:
                hashes[str(source.path(name))] = digest(source.path(name))
            print(
                f"{split} frame {frame}: {sum(evidence[i][-1]['observed'] for i in indices)} / {len(indices)} matched",
                flush=True,
            )
        if split == "discovery":
            summaries = [
                dict(
                    split="discovery",
                    **summarize_contact(rows, **config["observation_limits"]),
                )
                for rows in evidence
            ]
            shortlist = contact_site_search.shortlist_sites(
                sites, summaries, **config["shortlist"]
            )
            write_json(
                out / "shortlist.json",
                dict(
                    site_ids=shortlist,
                    discovery_summaries=summaries,
                    selection_used_validation=False,
                ),
            )
            print(f"Frozen shortlist: {shortlist}", flush=True)
    records = []
    supported = []
    for index, site in enumerate(sites):
        record = dict(
            site,
            discovery=summarize_contact(
                [r for r in evidence[index] if r["split"] == "discovery"],
                **config["observation_limits"],
            ),
            validation=None,
            shortlisted=index in shortlist,
            reasons=dict(Counter(r["reason"] for r in evidence[index])),
        )
        if index in shortlist:
            record["validation"] = summarize_contact(
                [r for r in evidence[index] if r["split"] == "validation"],
                **config["observation_limits"],
            )
            if record["validation"]["evidence"] == "supported":
                supported.append(index)
        records.append(record)
    world_from_object = np.asarray(plan["settling"]["settled_world_from_object"])
    wp = transform(points, world_from_object)
    wn = normals @ world_from_object[:3, :3].T
    com = np.asarray(plan["candidates"][0]["pair"]["center_of_mass_m"])
    physics = plan["candidates"][0]["pad_plan"]["point_wrench_baseline"]
    keys = ["mass_kg", "friction", "max_normal_force_n"]
    if any(
        any(c["pad_plan"]["point_wrench_baseline"][k] != physics[k] for k in keys)
        for c in plan["candidates"]
    ):
        raise ValueError("Plan assumptions differ across candidates")
    pairs = []
    for a, b in combinations(supported, 2):
        if mode == "regions" and sites[a]["region"] == sites[b]["region"]:
            continue
        if np.linalg.norm(wp[a] - wp[b]) < config["shortlist"]["minimum_spacing_m"]:
            continue
        pairs.append(
            dict(
                site_ids=[a, b],
                gravity_wrench=gravity_wrench_screen(
                    wp[[a, b]], wn[[a, b]], com, **{k: physics[k] for k in keys}
                ),
                heights_above_table_m=(
                    wp[[a, b], 2] - task["table"]["top_z_m"]
                ).tolist(),
                finite_patch_validated=False,
                robot_pair_validated=False,
                control_ready=False,
            )
        )
    for record in records:
        index = record["site_id"]
        record.update(
            point_task_world_m=wp[index].tolist(),
            normal_task_world=wn[index].tolist(),
            height_above_table_m=float(wp[index, 2] - task["table"]["top_z_m"]),
            upward_force_per_unit_normal=float(
                contact_site_search.upward_force_capacity(
                    wn[index : index + 1], physics["friction"]
                )[0]
            ),
        )
    report = dict(
        schema="real2sim2real.surface-support-survey/1",
        config=config,
        sites=records,
        supported_site_ids=supported,
        region_face_counts=region_counts,
        supported_regions=sorted({sites[i]["region"] for i in supported})
        if mode == "regions"
        else None,
        all_declared_regions_supported=(
            len({sites[i]["region"] for i in supported})
            == len(config["surface_regions"])
        )
        if mode == "regions"
        else None,
        pairs=pairs,
        shortlist_sha256=digest(out / "shortlist.json"),
        source_hashes=hashes,
        worker_source_hashes={
            str(Path(p)): digest(Path(p))
            for p in [
                __file__,
                contact_site_search.__file__,
                stereo_contact_evidence.__file__,
            ]
        },
        control_ready=False,
        geometry_refitted=False,
        hand_fitting_run=False,
        independent_ground_truth=False,
    )
    write_json(
        out / "observations.json",
        dict(
            sites=[
                dict(site_id=i, observations=rows) for i, rows in enumerate(evidence)
            ]
        ),
    )
    report["observations_sha256"] = digest(out / "observations.json")
    write_json(out / "survey.json", report)
    # Stable camera views of shortlisted sites (or the most observed failures).
    shown = (
        shortlist
        or sorted(
            range(len(sites)), key=lambda i: -records[i]["discovery"]["observed_frames"]
        )[:16]
    )
    review_frames = config.get(
        "review_frames",
        sorted(set([0, ids[len(ids) // 3], ids[2 * len(ids) // 3], max(ids)])),
    )
    for frame in review_frames:
        if frame >= source.count:
            continue
        image = Image.fromarray(source.rgb(frame))
        draw = ImageDraw.Draw(image)
        cam = (
            np.linalg.inv(source.world_from_camera[frame])
            @ source.world_from_object[frame]
        )
        coords = project(transform(points[shown], cam), np.c_[source.k, np.zeros(3)])
        for index, (x, y) in zip(shown, coords):
            if 0 <= x < image.width and 0 <= y < image.height:
                color = "#00ddff" if index in supported else "#ff8800"
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
                draw.text((x + 5, y), str(index), fill=color)
        image.save(out / f"frame-{frame:06d}.jpg", quality=90)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.worker:
        cv2.setNumThreads(1)
        search(config)
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
