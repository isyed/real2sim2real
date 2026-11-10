"""Direct stereo evidence for model-seeded contact regions, without refitting."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import trimesh

from real2sim2real.artifacts import digest, write_json
from real2sim2real.egocentric.camera import transform
from real2sim2real.egocentric.contact_observations import (
    observe_contacts,
    summarize_contact,
)
from real2sim2real.egocentric.interaction_io import Reconstruction
from real2sim2real.egocentric.stereo import Stereo
from real2sim2real.egocentric import capture, stereo_contact_evidence


def project(points, matrix):
    p = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    return p[:, :2] / p[:, 2:]


def visible_surface(mesh, points, normals, object_from_camera, tolerance):
    center = object_from_camera[:3, 3]
    vectors = points - center
    distances = np.linalg.norm(vectors, axis=1)
    direction = vectors / distances[:, None]
    hits, rays, _ = mesh.ray.intersects_location(
        np.tile(center, (len(points), 1)), direction, multiple_hits=True
    )
    nearest = np.full(len(points), np.inf)
    if len(hits):
        np.minimum.at(nearest, rays, np.linalg.norm(hits - center, axis=1))
    return ((normals * -direction).sum(1) >= 0.25) & (nearest >= distances - tolerance)


def audit(config):
    for key in ["capture", "calibration", "contact_audit"]:
        if digest(Path(config[key])) != config[key + "_sha256"]:
            raise ValueError("Pinned stereo audit input changed")
    source = Reconstruction(config["reconstruction"])
    original = json.loads(Path(config["contact_audit"]).read_text())
    planning_path = Path(original["config"]["source_config"])
    if digest(planning_path) != original["config"]["source_config_sha256"]:
        raise ValueError("Planning configuration changed")
    planning = json.loads(planning_path.read_text())
    plan_path = Path(planning["output"]) / planning["planning_directory"] / "plan.json"
    if digest(plan_path) != original["config"]["plan_sha256"]:
        raise ValueError("Contact plan changed")
    plan = json.loads(plan_path.read_text())
    for name in ["geometry.npz", "timeline.json", "mesh/metric.glb"]:
        path = source.path(name)
        if original["source_hashes"].get(str(path)) != digest(path):
            raise ValueError("Contact audit and reconstruction differ")
    cap = capture.load(config["capture"])
    artifact = json.loads(source.path("artifact.json").read_text())
    if artifact["metadata"]["capture_sha256"] != cap["manifest_sha256"]:
        raise ValueError("Stereo capture and reconstruction differ")
    if not np.array_equal(source.frame_ids, cap["frame_ids"]) or not np.allclose(
        source.times, cap["timestamps_s"], rtol=0, atol=1e-8
    ):
        raise ValueError("Capture timeline mismatch")
    calibration = json.loads(Path(config["calibration"]).read_text())
    if not calibration["accepted"]:
        raise ValueError("An accepted calibration is required")
    fit, validation = config["fit_frames"], config["validation_frames"]
    selected = sorted(fit + validation)
    if (
        not fit
        or not validation
        or len(set(selected)) != len(selected)
        or len(selected) > 40
        or any(type(i) is not int or not 0 <= i < source.count for i in selected)
    ):
        raise ValueError("Bounded disjoint capture frame splits required")
    contacts = [
        (c["candidate_id"], p) for c in original["candidates"] for p in c["contacts"]
    ]
    if not 1 <= len(contacts) <= 48:
        raise ValueError("At most 48 contact points required")
    points = np.array([c["point_object_m"] for _, c in contacts])
    normals = np.array([c["normal_object"] for _, c in contacts])
    plan_candidates = {c["candidate_id"]: c for c in plan["candidates"]}
    regions = {c["candidate_id"]: c for c in original["candidates"]}
    object_from_task = np.linalg.inv(plan["settling"]["settled_world_from_object"])
    half_spans = np.array(
        [
            object_from_task[:3, :3]
            @ np.asarray(plan_candidates[i]["pair"]["close_axis"])
            * regions[i]["local_proxy"]["source_width_m"]
            / 2
            for i, _ in contacts
        ]
    )
    left, right = [cap["streams"][key] for key in cap["stereo_pair"]]
    stereo = Stereo(
        left,
        right,
        cap["streams"][cap["target_stream"]],
        right_from_left=calibration["candidate_right_from_left"],
    )
    mesh = trimesh.load(source.path("mesh/metric.glb"), force="mesh", process=False)
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    evidence = [[] for _ in contacts]
    controls = []
    hashes = {
        str(Path(config[k])): digest(Path(config[k]))
        for k in ["capture", "calibration", "contact_audit"]
    }
    for stream in cap["streams"].values():
        hashes[stream["video"]] = stream["sha256"]
    images = {}
    # Decode each video once; do not infer timestamps from encoded playback FPS.
    for name, stream in zip(["left", "right"], [left, right]):
        images[name] = {
            i: im
            for i, im in enumerate(capture.frames(stream, max(selected) + 1))
            if i in selected
        }
        if len(images[name]) != len(selected):
            raise ValueError("Missing synchronized video frame")
    for i in selected:
        a_from_object = (
            np.linalg.inv(source.world_from_camera[i]) @ source.world_from_object[i]
        )
        l_from_object = np.linalg.inv(stereo.target_from_left) @ a_from_object
        r_from_object = stereo.right_from_left @ l_from_object
        lpoints = transform(points, l_from_object)
        rectpoints = lpoints @ stereo.rotations[0].T
        uvs = [project(rectpoints, p) for p in stereo.projections]
        ends = [
            transform(points + sign * half_spans, l_from_object) @ stereo.rotations[0].T
            for sign in [-1, 1]
        ]
        wall_width = np.linalg.norm(
            project(ends[0], stereo.projections[0])
            - project(ends[1], stereo.projections[0]),
            axis=1,
        )
        rect = [
            cv2.remap(
                cv2.cvtColor(images[name][i], cv2.COLOR_RGB2GRAY),
                *stereo.maps[j],
                interpolation=cv2.INTER_LINEAR,
            )
            for j, name in enumerate(["left", "right"])
        ]
        feature_mask = np.zeros(rect[0].shape, np.uint8)
        margin = (
            config["matching"]["radius"] + config["matching"]["vertical_radius"] + 1
        )
        feature_mask[
            margin:-margin, config["matching"]["maximum_disparity"] + margin : -margin
        ] = 255
        corners = cv2.goodFeaturesToTrack(
            rect[0], maxCorners=12, qualityLevel=0.02, minDistance=40, mask=feature_mask
        )
        for uv in [] if corners is None else corners[:, 0]:
            controls.append(
                dict(
                    frame_index=i,
                    source_pixel=uv.tolist(),
                    match=stereo_contact_evidence.match_rectified(
                        *rect, uv, **config["matching"]
                    ),
                )
            )
        masks = source.mask(i, 2) | source.mask(i, 3)
        rgb_visibility = observe_contacts(
            points,
            normals,
            a_from_object,
            source.k,
            source.depth(i),
            source.mask(i, 1),
            masks,
            radius_pixels=0,
            minimum_depth_pixels=1,
        )
        visible = visible_surface(
            mesh, points, normals, np.linalg.inv(l_from_object), 0.001
        ) & visible_surface(mesh, points, normals, np.linalg.inv(r_from_object), 0.001)
        panels = [
            Image.fromarray(source.rgb(i)),
            *[Image.fromarray(im).convert("RGB") for im in rect],
        ]
        panel_uv = [
            np.array(
                [
                    r["pixel"] if r["pixel"] is not None else [-1, -1]
                    for r in rgb_visibility
                ]
            ),
            *uvs,
        ]
        for j, (identifier, contact) in enumerate(contacts):
            row = dict(
                frame_index=i,
                frame_id=int(source.frame_ids[i]),
                timestamp_s=float(source.times[i]),
                split="fit" if i in fit else "validation",
                observed=False,
                residual_m=None,
                projected_wall_width_px=float(wall_width[j]),
            )
            if (
                rgb_visibility[j]["reason"] not in ["observed", "insufficient_depth"]
                or not visible[j]
                or rectpoints[j, 2] <= 0
            ):
                row["reason"] = "visibility_rejected"
                row["rgb_visibility_reason"] = rgb_visibility[j]["reason"]
                row["stereo_mesh_visible"] = bool(visible[j])
            else:
                match = stereo_contact_evidence.match_rectified(
                    *rect, uvs[0][j], **config["matching"]
                )
                row.update(
                    reason=match["reason"],
                    match=match,
                    predicted_disparity_px=float(uvs[0][j, 0] - uvs[1][j, 0]),
                )
                # Sensitivity at predicted depth; not a calibrated uncertainty estimate.
                d = row["predicted_disparity_px"]
                row["depth_change_per_disparity_pixel_m"] = (
                    float(rectpoints[j, 2] / d) if d > 0 else None
                )
                if match["accepted"]:
                    xyz = stereo_contact_evidence.triangulate_match(
                        match, stereo.projections
                    )
                    target = transform(
                        (xyz @ stereo.rotations[0])[None], stereo.target_from_left
                    )[0]
                    observed_object = transform(
                        target[None], np.linalg.inv(a_from_object)
                    )[0]
                    displacement = observed_object - points[j]
                    row.update(
                        observed=True,
                        residual_m=float(
                            target[2]
                            - transform(points[j : j + 1], a_from_object)[0, 2]
                        ),
                        observed_point_object_m=observed_object.tolist(),
                        normal_displacement_m=float(displacement @ normals[j]),
                        point_error_m=float(np.linalg.norm(displacement)),
                    )
            evidence[j].append(row)
            for k, panel in enumerate(panels):
                x, y = panel_uv[k][j]
                if 0 <= x < panel.width and 0 <= y < panel.height:
                    draw = ImageDraw.Draw(panel)
                    color = "#00ddff" if row["observed"] else "#ff8800"
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
                    draw.text(
                        (x + 4, y),
                        f"{identifier}:{contact['contact_index']}",
                        fill=color,
                    )
                    if k == 2 and row["observed"]:
                        mx, my = row["match"]["target_pixel"]
                        draw.line((x, y, mx, my), fill="#00ff66", width=2)
                        draw.ellipse(
                            (mx - 3, my - 3, mx + 3, my + 3), outline="#00ff66", width=2
                        )
        crops = []
        for panel, uv in zip(panels, panel_uv):
            valid = uv[
                (uv[:, 0] >= 0)
                & (uv[:, 0] < panel.width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < panel.height)
            ]
            if len(valid):
                lo = np.maximum(valid.min(0).astype(int) - 65, 0)
                hi = np.minimum(valid.max(0).astype(int) + 65, panel.size)
                panel = panel.crop((*lo, *hi))
            panel.thumbnail((500, 400))
            # Expand small contact crops for inspection; this creates no new detail.
            scale = min(500 / panel.width, 400 / panel.height)
            if scale > 1:
                panel = panel.resize(
                    (round(panel.width * scale), round(panel.height * scale))
                )
            crops.append(panel)
        sheet = Image.new("RGB", (1500, 430), "#17202b")
        draw = ImageDraw.Draw(sheet)
        for k, panel in enumerate(crops):
            sheet.paste(panel, (500 * k, 30))
            draw.text(
                (500 * k + 5, 5),
                ["RGB target", "Rectified left", "Rectified right; green = matched"][k],
                fill="white",
            )
        sheet.save(out / f"frame-{i:06d}.jpg", quality=90)
        for name in [
            f"depth/{i:06d}.npy",
            f"rgb/{i:06d}.png",
            *[f"masks/{k}/{i:06d}.png" for k in [1, 2, 3]],
        ]:
            hashes[str(source.path(name))] = digest(source.path(name))
        print(f"Stereo contact audit frame {i}", flush=True)
    reports = []
    for (identifier, contact), rows in zip(contacts, evidence):
        summaries = {
            split: summarize_contact(
                [r for r in rows if r["split"] == split],
                **original["config"]["observation_limits"],
            )
            for split in ["fit", "validation"]
        }
        reports.append(
            dict(
                candidate_id=identifier,
                contact_index=contact["contact_index"],
                summaries=summaries,
                reasons=dict(Counter(r["reason"] for r in rows)),
                observations=rows,
            )
        )
    report = dict(
        schema="real2sim2real.stereo-contact-audit/1",
        config=config,
        contacts=reports,
        textured_image_controls=controls,
        control_scope="Left-image corners; checks matching availability, not contact geometry accuracy.",
        source_hashes=hashes,
        worker_source_hashes={
            str(Path(p)): digest(Path(p))
            for p in [__file__, stereo_contact_evidence.__file__]
        },
        calibration_refitted=False,
        geometry_refitted=False,
        model_seeded_image_locations=True,
        independent_ground_truth=False,
        control_ready=False,
    )
    write_json(out / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.worker:
        cv2.setNumThreads(1)
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
