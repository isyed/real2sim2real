"""Read-only multiview approach/lift evidence and reconstructed motion audit."""

import argparse
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import trimesh

from real2sim2real.artifacts import digest, write_json
from . import capture
from .camera import transform
from .interaction_io import Reconstruction, fresh_directory


def motion_rows(poses, vertices, times, frame_ids, table_top_m=0):
    poses, vertices, times, frame_ids = map(
        np.asarray, (poses, vertices, times, frame_ids)
    )
    if (
        len(poses) != len(times)
        or len(times) != len(frame_ids)
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("Aligned increasing motion timeline required")
    rows = []
    for i, pose in enumerate(poses):
        heights = vertices @ pose[2, :3] + pose[2, 3] - table_top_m
        rows.append(
            dict(
                frame_id=int(frame_ids[i]),
                time_s=float(times[i]),
                minimum_clearance_m=float(heights.min()),
                lower_decile_height_m=float(np.quantile(heights, 0.1)),
                upper_decile_height_m=float(np.quantile(heights, 0.9)),
                object_position_m=pose[:3, 3].tolist(),
                translation_from_start_m=float(
                    np.linalg.norm(pose[:3, 3] - poses[0, :3, 3])
                ),
                rotation_from_start_deg=float(
                    np.rad2deg(
                        Rotation.from_matrix(
                            pose[:3, :3] @ poses[0, :3, :3].T
                        ).magnitude()
                    )
                ),
                contact_observed=False,
            )
        )
    return rows


def audit(config):
    import cv2

    for p, sha in config["sources"].items():
        if digest(Path(p)) != sha:
            raise ValueError("Strategy-audit source changed")
    source = Reconstruction(config["reconstruction"], config["frame_count"])
    cap = capture.load(config["capture"])
    support = json.loads(Path(config["support"]).read_text())
    if not support["accepted"] or support["source_geometry_sha256"] != digest(
        source.path("geometry.npz")
    ):
        raise ValueError("Accepted support must bind the reconstructed geometry")
    if not np.array_equal(
        source.frame_ids, np.asarray(cap["frame_ids"])[: source.count]
    ) or not np.allclose(
        source.times, cap["timestamps_s"][: source.count], rtol=0, atol=1e-8
    ):
        raise ValueError("Capture and reconstruction timeline mismatch")
    u = np.asarray(support["upright_from_world"])
    poses = u @ source.world_from_object
    mesh = trimesh.load(source.path("mesh/metric.glb"), force="mesh")
    rows = motion_rows(poses, mesh.vertices, source.times, source.frame_ids)
    probe_rows = []
    for probe in config.get("object_local_probes", []):
        p = np.asarray(probe["point_m"])
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("Finite object-local probe required")
        heights = np.einsum("tij,j->ti", poses[:, :3, :3], p) + poses[:, :3, 3]
        probe_rows.append(
            dict(
                name=probe["name"],
                original_start_height_m=float(heights[0, 2]),
                settled_height_m=probe["settled_height_m"],
                frame_heights_m=heights[:, 2].tolist(),
                observed_contact=False,
            )
        )
    out = fresh_directory(config["output"])
    chosen = config["review_frame_ids"]
    if (
        not 1 <= len(chosen) <= 24
        or len(set(chosen)) != len(chosen)
        or any(i not in source.frame_ids for i in chosen)
    ):
        raise ValueError("At most 24 available review frames required")
    indices = {int(fid): i for i, fid in enumerate(source.frame_ids) if fid in chosen}
    panels = {i: Image.new("RGB", (1440, 520), "#e8edf2") for i in chosen}
    images, sources = {}, dict(config["sources"])
    rig_a = np.asarray(cap["streams"][cap["target_stream"]]["rig_from_camera"])
    for column, (name, stream) in enumerate(cap["streams"].items()):
        if column >= 3:
            raise ValueError("Review supports three declared views")
        images[name] = {}
        sources[stream["video"]] = stream["sha256"]
        view_from_a = np.linalg.inv(stream["rig_from_camera"]) @ rig_a
        for i, pixels in enumerate(capture.frames(stream, max(indices.values()) + 1)):
            fid = int(source.frame_ids[i])
            if fid not in chosen:
                continue
            origin_a = transform(
                source.world_from_object[i, :3, 3][None],
                np.linalg.inv(source.world_from_camera[i]),
            )
            center = transform(origin_a, view_from_a)
            if center[0, 2] <= 0:
                raise ValueError("Object center behind review camera")
            k = np.asarray(stream["intrinsics"], float)
            uv, _ = cv2.projectPoints(
                center, np.zeros(3), np.zeros(3), k, np.array(stream["distortion"])
            )
            x, y = uv.reshape(2)
            radius = (
                config["crop_radius_m"] * float((k[0, 0] + k[1, 1]) / 2) / center[0, 2]
            )
            box = [
                max(0, int(x - radius)),
                max(0, int(y - radius)),
                min(pixels.shape[1], int(x + radius)),
                min(pixels.shape[0], int(y + radius)),
            ]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Empty review crop")
            image = Image.fromarray(pixels).crop(box)
            image.thumbnail((480, 480))
            panels[fid].paste(image, (480 * column, 30))
            ImageDraw.Draw(panels[fid]).text(
                (480 * column + 10, 8),
                f"Frame {fid} | camera {name} | {source.times[i]:.3f}s",
                fill="#101820",
            )
            images[name][str(fid)] = dict(
                source_pixel_box=box,
                source_frame_id=int(source.frame_ids[i]),
                source_time_s=float(source.times[i]),
            )
    for i, panel in panels.items():
        ImageDraw.Draw(panel).text(
            (10, 505),
            "Source crops. Contact identity and force remain unverified.",
            fill="#101820",
        )
        panel.save(out / f"frame-{i:06d}.jpg", quality=94)
    for i in chosen:
        sources[str(source.path(f"rgb/{indices[i]:06d}.png"))] = digest(
            source.path(f"rgb/{indices[i]:06d}.png")
        )
    report = dict(
        schema="real2sim2real.demonstration-strategy/1",
        config=config,
        motion=rows,
        images=images,
        sources=sources,
        object_local_probes=probe_rows,
        source_mesh_sha256=digest(source.path("mesh/metric.glb")),
        gravity_alignment_assumption="observed table is level",
        contact_or_hand_identity_validated=False,
        reference_ground_truth_used=False,
        inference_run=False,
        physics_run=False,
        implementation_sha256=digest(Path(__file__)),
    )
    write_json(out / "audit.json", report)
    motion_table = "".join(
        f"<tr><td>{r['frame_id']}</td><td>{1000 * r['minimum_clearance_m']:.1f}</td><td>{1000 * r['translation_from_start_m']:.1f}</td><td>{r['rotation_from_start_deg']:.1f}</td></tr>"
        for r in rows
        if r["frame_id"] in chosen
    )
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Demonstration strategy audit</title><style>body{font:17px system-ui;max-width:1440px;margin:30px auto;background:#18202b;color:#eee}img{width:100%}td,th{padding:8px}a{color:#8df}</style><h1>Approach and initial lift: source observations</h1><p>Read-only synchronized camera crops. Reconstructed motion is estimated; minimum clearance does not establish contact. Hidden finger identity, forces and contact normals remain unknown.</p><p><a href="audit.json">Metrics, source hashes and crop coordinates</a></p><table><tr><th>Frame</th><th>Reconstructed minimum clearance mm</th><th>Translation from start mm</th><th>Rotation from start degrees</th></tr>'
        + motion_table
        + "</table>"
        + "".join(
            f'<h2>Frame {i}</h2><img src="frame-{i:06d}.jpg" alt="{html.escape(str(i))}: synchronized capture views">'
            for i in chosen
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
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
    return audit(config)


if __name__ == "__main__":
    main()
