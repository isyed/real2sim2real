"""Audit observed stereo neighborhoods around pinned surface proposals on CPU."""

import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import sys

import numpy as np

from real2sim2real.artifacts import digest, write_json


def audit(config):
    import cv2
    from PIL import Image, ImageDraw
    import trimesh
    from real2sim2real.egocentric import capture, stereo_contact_evidence, surface_patch
    from real2sim2real.egocentric.camera import transform
    from real2sim2real.egocentric.interaction_io import Reconstruction
    from real2sim2real.egocentric.stereo import Stereo
    from .stereo_contact_audit import project as rect_project, visible_surface

    def project(points, intrinsics):
        projected = np.asarray(points) @ intrinsics.T
        return projected[:, :2] / projected[:, 2:]

    sources = {}
    proposals = {}
    for item in config["surveys"]:
        path = Path(item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError("Pinned survey changed")
        report = json.loads(path.read_text())
        for name, sha in report["source_hashes"].items():
            if digest(Path(name)) != sha:
                raise ValueError("Survey source changed")
            sources[name] = sha
        sources[str(path)] = digest(path)
        for site in report["sites"]:
            if site["site_id"] in item["site_ids"]:
                if (
                    not site["validation"]
                    or site["validation"]["evidence"] != "supported"
                ):
                    raise ValueError("Each patch must start at a supported point")
                key = str(site["face_index"])
                if key in proposals:
                    raise ValueError("Distinct source faces required")
                proposals[key] = site
    if not 1 <= len(proposals) <= 8:
        raise ValueError("At most eight pinned patches required")
    first = json.loads(Path(config["surveys"][0]["path"]).read_text())["config"]
    source = Reconstruction(first["reconstruction"])
    cap = capture.load(first["capture"])
    artifact = json.loads(source.path("artifact.json").read_text())
    if (
        artifact["metadata"]["capture_sha256"] != cap["manifest_sha256"]
        or not np.array_equal(source.frame_ids, cap["frame_ids"])
        or not np.allclose(source.times, cap["timestamps_s"], atol=1e-8, rtol=0)
    ):
        raise ValueError("Capture provenance/timing differs")
    for item in config["surveys"]:
        other = json.loads(Path(item["path"]).read_text())["config"]
        for name in (
            "capture_sha256",
            "calibration_sha256",
            "plan_sha256",
            "reconstruction_artifact_sha256",
        ):
            if other[name] != first[name]:
                raise ValueError("Patch proposals belong to different scenes")
    frames = config["frames"]
    grid = config["pixel_grid"]
    if (
        not 1 <= len(frames) <= 40
        or len(set(frames)) != len(frames)
        or any(type(i) is not int or not 0 <= i < source.count for i in frames)
        or set(grid) != {"radius_px", "step_px"}
        or any(type(v) is not int for v in grid.values())
        or not 3 <= grid["radius_px"] <= 24
        or not 2 <= grid["step_px"] <= 8
    ):
        raise ValueError("Bounded frames and local pixel grid required")
    offsets = np.arange(-grid["radius_px"], grid["radius_px"] + 1, grid["step_px"])
    if len(offsets) ** 2 > 81:
        raise ValueError("At most 81 local rays per patch/frame required")
    left, right = [cap["streams"][k] for k in cap["stereo_pair"]]
    calibration = json.loads(Path(first["calibration"]).read_text())
    if not calibration["accepted"]:
        raise ValueError("Accepted calibration required")
    stereo = Stereo(
        left,
        right,
        cap["streams"][cap["target_stream"]],
        right_from_left=calibration["candidate_right_from_left"],
    )
    mesh = trimesh.load(source.path("mesh/metric.glb"), force="mesh", process=False)
    plan = json.loads(Path(first["plan"]).read_text())
    receipt = json.loads(Path(first["scene_receipt"]).read_text())
    task_path = Path(receipt["config"]["task"])
    if digest(task_path) != receipt["sources"][str(task_path)]:
        raise ValueError("Original task changed")
    sources[str(task_path)] = digest(task_path)
    task = json.loads(task_path.read_text())
    initial_pose = np.asarray(task["object"]["world_from_object"])
    settled_pose = np.asarray(plan["settling"]["settled_world_from_object"])
    videos = [
        {
            i: im
            for i, im in enumerate(capture.frames(stream, max(frames) + 1))
            if i in frames
        }
        for stream in (left, right)
    ]
    if any(len(v) != len(frames) for v in videos):
        raise ValueError("Missing stereo frames")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    points = np.array([s["point_object_m"] for s in proposals.values()])
    normals = np.array([s["normal_object"] for s in proposals.values()])
    rows = {key: [] for key in proposals}
    all_observations = []
    for frame in frames:
        a_from_object = (
            np.linalg.inv(source.world_from_camera[frame])
            @ source.world_from_object[frame]
        )
        l_from_object = np.linalg.inv(stereo.target_from_left) @ a_from_object
        r_from_object = stereo.right_from_left @ l_from_object
        rect_points = transform(points, l_from_object) @ stereo.rotations[0].T
        centers = rect_project(rect_points, stereo.projections[0])
        visible = visible_surface(
            mesh, points, normals, np.linalg.inv(l_from_object), 0.001
        )
        visible &= visible_surface(
            mesh, points, normals, np.linalg.inv(r_from_object), 0.001
        )
        rect = [
            cv2.remap(
                cv2.cvtColor(v[frame], cv2.COLOR_RGB2GRAY),
                *stereo.maps[k],
                interpolation=cv2.INTER_LINEAR,
            )
            for k, v in enumerate(videos)
        ]
        mask = source.mask(frame, 1)
        occluder = source.mask(frame, 2) | source.mask(frame, 3)
        preview = Image.fromarray(source.rgb(frame))
        draw = ImageDraw.Draw(preview)
        for j, (key, site) in enumerate(proposals.items()):
            observed, reasons, evidence = [], Counter(), []
            for dy in offsets if visible[j] and rect_points[j, 2] > 0 else []:
                for dx in offsets:
                    uv = centers[j] + [dx, dy]
                    match = stereo_contact_evidence.match_rectified(
                        *rect, uv, **config["matching"]
                    )
                    entry = dict(source_pixel=uv.tolist(), match=match)
                    if match["accepted"]:
                        xyz = stereo_contact_evidence.triangulate_match(
                            match, stereo.projections
                        )
                        left_xyz = xyz @ stereo.rotations[0]
                        target = transform(left_xyz[None], stereo.target_from_left)[0]
                        local = transform(left_xyz[None], np.linalg.inv(l_from_object))[
                            0
                        ]
                        pixel = project(target[None], source.k)[0]
                        x, y = np.rint(pixel).astype(int)
                        valid = (
                            0 <= x < mask.shape[1]
                            and 0 <= y < mask.shape[0]
                            and mask[y, x]
                            and not occluder[y, x]
                            and np.linalg.norm(local - points[j])
                            <= config["patch_limits"]["radius_m"]
                        )
                        if valid:
                            observed.append(local)
                            entry["point_object_m"] = local.tolist()
                            entry["accepted"] = True
                            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="#00ddff")
                        else:
                            entry["accepted"] = False
                            reasons["outside_local_object_or_hand_occluded"] += 1
                    else:
                        reasons[match["reason"]] += 1
                    evidence.append(entry)
            fit = surface_patch.fit_patch(
                np.asarray(observed).reshape(-1, 3),
                points[j],
                normals[j],
                **config["patch_limits"],
            )
            row = dict(
                frame_index=frame,
                frame_id=int(source.frame_ids[frame]),
                timestamp_s=float(source.times[frame]),
                center_visible=bool(visible[j]),
                fit=fit,
                rejection_counts=dict(reasons),
            )
            rows[key].append(row)
            all_observations.append(
                dict(face_index=int(key), **row, correspondences=evidence)
            )
            center_pixel = project(
                transform(points[j : j + 1], a_from_object), source.k
            )[0]
            if np.isfinite(center_pixel).all():
                x, y = center_pixel
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline="#ffbb33", width=2)
                draw.text(
                    (x + 5, y), f"{site['region']} / {site['site_id']}", fill="#ffbb33"
                )
        preview.save(out / f"frame-{frame:06d}.jpg", quality=88)
        for name in [
            f"rgb/{frame:06d}.png",
            *[f"masks/{k}/{frame:06d}.png" for k in (1, 2, 3)],
        ]:
            sources[str(source.path(name))] = digest(source.path(name))
        print(
            f"frame {frame}: "
            + str([rows[k][-1]["fit"]["points_in_region"] for k in rows]),
            flush=True,
        )
    reports = []
    for j, (key, site) in enumerate(proposals.items()):
        initial_height = float(
            transform(points[j : j + 1], initial_pose)[0, 2] - task["table"]["top_z_m"]
        )
        reports.append(
            dict(
                face_index=int(key),
                region=site["region"],
                site_id=site["site_id"],
                point_object_m=points[j].tolist(),
                proposed_normal_object=normals[j].tolist(),
                initial_reconstructed_point_height_m=initial_height,
                settled_model_point_height_m=float(
                    transform(points[j : j + 1], settled_pose)[0, 2]
                    - task["table"]["top_z_m"]
                ),
                original_insertion_gap_verified=False,
                thickness_verified=False,
                repeated_plane=surface_patch.normal_repeatability(
                    [r["fit"] for r in rows[key]], **config["repeatability_limits"]
                ),
                frames=rows[key],
            )
        )
    implementation = out / "implementation"
    implementation.mkdir()
    impl_hashes = {}
    for module in (
        Path(__file__),
        Path(surface_patch.__file__),
        Path(stereo_contact_evidence.__file__),
    ):
        target = implementation / module.name
        target.write_bytes(module.read_bytes())
        impl_hashes[module.name] = digest(target)
    write_json(out / "observations.json", all_observations)
    report = dict(
        schema="real2sim2real.stereo-patch-audit/1",
        config=config,
        patches=reports,
        source_hashes=sources,
        implementation_sha256=impl_hashes,
        observations_sha256=digest(out / "observations.json"),
        geometry_refitted=False,
        observations_model_seeded=True,
        independent_ground_truth=False,
        control_ready=False,
        force_balance_recomputed=False,
        conclusion="One-sided stereo neighborhoods cannot certify thickness, initial insertion or a grasp; preserve the prior wrench rejection.",
    )
    write_json(out / "audit.json", report)
    cards = "".join(
        f'<figure><img src="frame-{i:06d}.jpg" width="720"><figcaption>Frame {i}</figcaption></figure>'
        for i in frames
    )
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Stereo contact patches</title>'
        "<h1>Observed neighborhoods at proposed contact patches</h1><p>Cyan: accepted local stereo points. Orange: model-seeded patch centers. "
        "Sampled surface evidence; thickness and insertion remain unverified.</p><pre>"
        + html.escape(
            json.dumps(
                [{k: v for k, v in r.items() if k != "frames"} for r in reports],
                indent=2,
            )
        )
        + "</pre>"
        + cards
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
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
            timeout=min(config["deadline_s"], 900),
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only audit required")
    audit(config)


if __name__ == "__main__":
    main()
