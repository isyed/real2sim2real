"""CPU audit of disputed hand observations in calibrated secondary images.

Consumes review views and existing hypotheses without altering either. Sparse
appearance triangulation is diagnostic surface evidence, not joint truth.
"""

import argparse
from collections import Counter
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from real2sim2real.artifacts import digest, write_json
from .camera import transform
from .hand_review_inputs import annotation_errors
from .hand_point_tracks import close_fingertip_pairs
from .interaction_io import fresh_directory
from .multiview_points import (
    reciprocal_match,
    triangulate,
    pixel_sensitivity,
    project_points,
)


def audit(config):
    views_dir = Path(config["views"])
    manifest = json.loads((views_dir / "views.json").read_text())
    views = manifest["views"]
    names = config.get("secondary_views", [n for n in views if n != "target"])
    if len(names) != 2 or len(set(names)) != 2 or "target" in names:
        raise ValueError("Two distinct secondary views required")
    hashes = {str(views_dir / "views.json"): digest(views_dir / "views.json")}
    implementation = {
        name: digest(Path(__file__).with_name(name))
        for name in [
            "hand_anchor_audit.py",
            "multiview_points.py",
            "hand_point_tracks.py",
            "hand_review_inputs.py",
            "camera.py",
        ]
    }
    tracks = dict(np.load(config["observations"]))
    hashes[config["observations"]] = digest(config["observations"])
    if not np.array_equal(np.unique(tracks["frame_ids"]), tracks["frame_ids"]):
        raise ValueError("Ordered unique observation frame IDs required")
    variants = {}
    for name, spec in config["variants"].items():
        data = dict(np.load(spec["path"]))
        hashes[spec["path"]] = digest(spec["path"])
        if not np.array_equal(
            data["frame_ids"], tracks["frame_ids"]
        ) or not np.allclose(
            data["timestamps_s"], tracks["timestamps_s"], atol=1e-8, rtol=0
        ):
            raise ValueError("Candidate and observation timelines differ")
        variants[name] = data[spec.get("joints_key", "camera_joints")]
    annotations = (
        json.loads(Path(config["annotations"]).read_text())
        if config.get("annotations")
        else {"points": []}
    )
    if config.get("annotations"):
        hashes[config["annotations"]] = digest(config["annotations"])
    out = fresh_directory(config["output"])
    rows, panels, identity_warnings = [], [], []
    for frame in config.get("frame_ids", manifest["frame_ids"]):
        matches = np.flatnonzero(tracks["frame_ids"] == frame)
        if len(matches) != 1:
            raise ValueError("Requested audit frame absent from observations")
        i = int(matches[0])
        images = {}
        for name in ["target", *names]:
            info = views[name]["images"][str(frame)]
            p = views_dir / info["path"]
            if digest(p) != info["sha256"]:
                raise ValueError("Review image changed")
            hashes[str(p)] = info["sha256"]
            images[name] = np.asarray(Image.open(p).convert("RGB"))
        k = np.asarray(views["target"]["intrinsics"])
        for hand in range(2):
            for pair in close_fingertip_pairs(
                tracks["keypoints_px"][i, hand], tracks["weights"][i, hand]
            ):
                identity_warnings.append(dict(frame_id=int(frame), hand=hand, **pair))
            for joint in np.flatnonzero(tracks["weights"][i, hand] > 0):
                uv = tracks["keypoints_px"][i, hand, joint]
                row = dict(
                    frame_id=int(frame),
                    timestamp_s=float(tracks["timestamps_s"][i]),
                    hand=hand,
                    joint=int(joint),
                    target_px=uv.tolist(),
                    supporting_seeds=int(tracks["supporting_seeds"][i, hand, joint]),
                    seed_spread_px=float(tracks["disagreement_px"][i, hand, joint]),
                    matches={},
                    metric_surface_supported=False,
                    anatomical_identity_verified=False,
                    reasons=[],
                    candidate_errors={},
                )
                points, cameras = [uv], [(k, np.eye(4))]
                pair_depths = []
                for name in names:
                    view = views[name]
                    camera = (
                        np.asarray(view["intrinsics"]),
                        np.asarray(view["view_from_target"]),
                    )
                    m = reciprocal_match(
                        images["target"],
                        images[name],
                        uv,
                        k,
                        *camera,
                        **config.get("matching", {}),
                    )
                    row["matches"][name] = m
                    if m["accepted"]:
                        points.append(m["match_px"])
                        cameras.append(camera)
                        pair_depths.append(
                            triangulate([uv, m["match_px"]], [cameras[0], camera])[
                                "xyz_reference_m"
                            ][2]
                        )
                if len(points) == 3:
                    t = triangulate(points, cameras)
                    sensitivity = pixel_sensitivity(points, cameras)
                    row.update(
                        triangulation=t,
                        pixel_sensitivity=sensitivity,
                        pair_depth_difference_m=float(
                            abs(pair_depths[0] - pair_depths[1])
                        ),
                    )
                    if not t["positive_depth"] or t["max_ray_angle_deg"] < 1:
                        row["reasons"].append("degenerate_geometry")
                    if max(t["reprojection_errors_px"]) > 2:
                        row["reasons"].append("multiview_reprojection")
                    if row["pair_depth_difference_m"] > 0.03:
                        row["reasons"].append("secondary_depth_disagreement")
                    if sensitivity["max_depth_change_m"] > 0.02:
                        row["reasons"].append("depth_sensitive_to_pixels")
                    row["metric_surface_supported"] = not row["reasons"]
                else:
                    row["reasons"].append("two_reciprocal_secondary_matches_required")
                for name, joints in variants.items():
                    xyz = joints[i, hand, joint]
                    if np.isfinite(xyz).all() and xyz[2] > 0:
                        errors = {
                            "target": float(np.linalg.norm(project_points(xyz, k) - uv))
                        }
                        for view_name in names:
                            match = row["matches"][view_name]
                            if match["accepted"]:
                                view = views[view_name]
                                p = transform(
                                    xyz[None], np.asarray(view["view_from_target"])
                                )[0]
                                if p[2] > 0:
                                    errors[view_name] = float(
                                        np.linalg.norm(
                                            project_points(p, view["intrinsics"])
                                            - match["match_px"]
                                        )
                                    )
                        row["candidate_errors"][name] = errors
                rows.append(row)
        for view_name in ["target", *names]:
            rgb = images[view_name]
            view = views[view_name]
            # Every variant uses the same scene crop, determined from observed
            # points transformed at a broad nominal depth solely for display.
            uv = tracks["keypoints_px"][i].reshape(-1, 2)
            uv = uv[np.isfinite(uv).all(1)]
            rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(k).T * 0.55
            uv_crop = project_points(
                transform(rays, np.asarray(view["view_from_target"])),
                view["intrinsics"],
            )
            lo = np.maximum(np.floor(uv_crop.min(0) - 65).astype(int), 0)
            hi = np.minimum(np.ceil(uv_crop.max(0) + 65).astype(int), rgb.shape[1::-1])
            tiles = []
            for name, joints in {"observations": None, **variants}.items():
                im = rgb.copy()
                if joints is not None:
                    for h in range(2):
                        xyz = joints[i, h]
                        if not np.isfinite(xyz).all():
                            continue
                        uv_model = project_points(
                            transform(xyz, np.asarray(view["view_from_target"])),
                            view["intrinsics"],
                        )
                        for j, point in enumerate(uv_model):
                            xy = tuple(np.rint(point).astype(int))
                            cv2.circle(im, xy, 3, (40, 200, 255), -1)
                            if j in (4, 8, 12, 16, 20):
                                cv2.putText(
                                    im,
                                    str(j),
                                    xy,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (40, 200, 255),
                                    1,
                                )
                for row in rows:
                    if row["frame_id"] != frame:
                        continue
                    m = row["matches"].get(view_name)
                    point = (
                        row["target_px"] if view_name == "target" else m.get("match_px")
                    )
                    if point is None:
                        continue
                    color = (
                        (50, 240, 70)
                        if view_name == "target" or m["accepted"]
                        else (245, 120, 40)
                    )
                    xy = tuple(np.rint(point).astype(int))
                    cv2.circle(im, xy, 4, color, 1)
                    if name == "observations":
                        cv2.putText(
                            im,
                            str(row["joint"]),
                            xy,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            color,
                            1,
                        )
                for label in annotations.get("points", []):
                    if label["frame_id"] == frame and label["view"] == view_name:
                        cv2.drawMarker(
                            im,
                            tuple(np.rint(label["xy_px"]).astype(int)),
                            (255, 50, 220),
                            cv2.MARKER_CROSS,
                            12,
                            2,
                        )
                tile = im[lo[1] : hi[1], lo[0] : hi[0]].copy()
                cv2.putText(
                    tile,
                    name,
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                tiles.append(tile)
            filename = f"{view_name}_{frame:06d}.jpg"
            Image.fromarray(np.concatenate(tiles, axis=1)).save(
                out / filename, quality=90
            )
            panels.append(
                f'<details><summary>Frame {frame}, {html.escape(view_name)}</summary><a href="{filename}"><img loading="lazy" src="{filename}"></a></details>'
            )
    manual = {
        name: annotation_errors(annotations, joints, tracks["frame_ids"], views)
        for name, joints in variants.items()
    }
    summary = dict(
        schema="real2sim2real.hand-anchor-audit/1",
        frame_ids=config.get("frame_ids", manifest["frame_ids"]),
        tracked_points=len(rows),
        secondary_matches={
            n: sum(r["matches"][n]["accepted"] for r in rows) for n in names
        },
        metric_surface_supported=sum(r["metric_surface_supported"] for r in rows),
        failures=dict(Counter(reason for row in rows for reason in row["reasons"])),
        manual_annotation_reviewer=annotations.get("reviewer"),
        manual_annotations_human_verified=annotations.get("human_verified", False),
        control_ready=False,
        promoted=False,
        poses_modified=False,
        limitations=[
            "Appearance correspondence does not establish anatomical identity.",
            "Secondary images and calibration overlap with the existing stereo-depth source; this is a consistency audit, not independent 3D ground truth.",
            "Pixel perturbation sensitivity is not a confidence interval and excludes calibration uncertainty.",
            "Manual image annotations are reserved for evaluation and remain approximate.",
        ],
    )
    summary["close_fingertip_pairs"] = identity_warnings
    write_json(
        out / "audit.json",
        dict(
            summary,
            observations=rows,
            manual_errors=manual,
            implementation_sha256=implementation,
            source_hashes=hashes,
            config=config,
        ),
    )
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Hand anchor audit</title><style>body{font:16px system-ui;background:#17202b;color:#eee;margin:25px}img{max-width:100%}summary{padding:12px;cursor:pointer}a{color:#8cf}pre{white-space:pre-wrap}</style><h1>Hand anchor audit</h1><p>Green: observed target tracks or accepted secondary appearance matches. Orange: rejected secondary matches. Blue: model joint projections. Magenta: approximate visual evaluation labels. Candidate columns include rejected hypotheses; this is not a control trajectory.</p><a href="audit.json">Full diagnostics and provenance</a><pre>'
        + html.escape(json.dumps(summary, indent=2))
        + "</pre>"
        + "".join(panels)
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.config.read_text())), indent=2))


if __name__ == "__main__":
    main()
