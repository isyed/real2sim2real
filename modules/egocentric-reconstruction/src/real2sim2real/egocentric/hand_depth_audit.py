"""Post-fit reserved-depth residuals near named image landmarks.

Regions are fixed from image observations, shared across candidates, and are
not anatomical segmentation or a replacement for whole-hand acceptance gates.
"""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import trimesh
import cv2
from PIL import Image

from real2sim2real.artifacts import digest, write_json
from .hand_observation_fit import measured_surface
from .interaction_io import Reconstruction, fresh_directory
from .mesh_surface import MeshSurfaceQuery
from .metric_hands import project


def image_regions(points, k, landmarks, regions, *, radius_px=20):
    if not np.isfinite(radius_px) or radius_px <= 0:
        raise ValueError("Positive finite image radius required")
    points = np.asarray(points)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or (points[:, 2] <= 0).any()
    ):
        raise ValueError("Finite positive-depth Nx3 points required")
    uv = project(points, k)
    masks = {}
    for name, ids in regions.items():
        if name in ("all", "remaining"):
            raise ValueError("Reserved region name")
        labels = np.asarray(landmarks)[ids]
        labels = labels[np.isfinite(labels).all(1)]
        masks[name] = (
            np.linalg.norm(uv[:, None] - labels[None], axis=-1).min(1) <= radius_px
            if len(labels)
            else np.zeros(len(uv), bool)
        )
    covered = np.zeros(len(uv), bool)
    for mask in masks.values():
        covered |= mask
    return dict(all=np.ones(len(uv), bool), **masks, remaining=~covered)


def residual_summary(distances, regions):
    distances = np.asarray(distances)
    if distances.ndim != 1 or not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Finite nonnegative surface residuals required")
    out = {}
    for name, mask in regions.items():
        if mask.shape != distances.shape or mask.dtype != bool:
            raise ValueError("Boolean region mask must match residuals")
        d = distances[mask]
        out[name] = dict(
            count=len(d),
            median_m=float(np.median(d)) if len(d) else None,
            p90_m=float(np.quantile(d, 0.9)) if len(d) else None,
            over_30mm=int((d > 0.03).sum()),
        )
    return out


def wrist_halfspace(points, k, landmarks, *, margin_px=10):
    """Diagnostic 2D wrist boundary; never an automatic depth-rejection rule."""
    if not np.isfinite(margin_px) or margin_px < 0:
        raise ValueError("Nonnegative finite wrist margin required")
    landmarks = np.asarray(landmarks)
    wrist, knuckles = landmarks[0], landmarks[[5, 9, 13, 17]]
    if not np.isfinite(wrist).all() or not np.isfinite(knuckles).all():
        raise ValueError("Observed wrist and palm landmarks required")
    direction = knuckles.mean(0) - wrist
    length = np.linalg.norm(direction)
    if length < 5:
        raise ValueError("Unresolved wrist/palm direction")
    beyond = (project(points, k) - wrist) @ (direction / length) < -margin_px
    return dict(beyond_detected_wrist=beyond, within_detected_hand_side=~beyond)


def audit(config):
    source = Reconstruction(
        config["reconstruction"], config["frame_count"], start=config["frame_start"]
    )
    obs = dict(np.load(config["observations"]))
    crops = dict(np.load(config["crops"]))
    variants = {
        name: dict(np.load(spec["path"])) for name, spec in config["variants"].items()
    }
    wrist_obs = (
        dict(np.load(config["wrist_observations"]))
        if config.get("wrist_observations")
        else None
    )
    for data in [
        obs,
        crops,
        *variants.values(),
        *([] if wrist_obs is None else [wrist_obs]),
    ]:
        if not np.array_equal(data["frame_ids"], source.frame_ids) or not np.allclose(
            data["timestamps_s"], source.times, atol=1e-8, rtol=0
        ):
            raise ValueError("Depth audit input timelines differ")
    paths = [
        config["observations"],
        config["crops"],
        *[s["path"] for s in config["variants"].values()],
    ]
    if wrist_obs is not None:
        paths.append(config["wrist_observations"])
    paths += [
        str(source.path(n)) for n in ["artifact.json", "geometry.npz", "timeline.json"]
    ]
    rows, arrays, panels = [], {}, []
    out = fresh_directory(config["output"])
    for case in config["cases"]:
        frame, h = case["frame_id"], case["hand"]
        ids = np.flatnonzero(source.frame_ids == frame)
        if len(ids) != 1 or h not in (0, 1):
            raise ValueError("Invalid case frame or hand")
        i = int(ids[0])
        for n in [
            f"depth/{i + source.start:06d}.npy",
            f"masks/{config['mask_ids'][h]}/{i + source.start:06d}.png",
        ]:
            paths.append(str(source.path(n)))
        _, points = measured_surface(
            source.depth(i),
            source.mask(i, config["mask_ids"][h]),
            crops["boxes_xyxy"][i, h],
            source.k,
        )
        masks = image_regions(
            points,
            source.k,
            obs["keypoints_px"][i, h],
            config["regions"],
            radius_px=config.get("radius_px", 20),
        )
        wrist_available = wrist_obs is not None and bool(wrist_obs["accepted"][i, h])
        if wrist_available:
            masks.update(
                wrist_halfspace(
                    points,
                    source.k,
                    wrist_obs["keypoints_px"][i, h],
                    margin_px=config.get("wrist_margin_px", 10),
                )
            )
        key = f"frame_{frame}_hand_{h}"
        arrays[key + "_points"] = points
        for name, mask in masks.items():
            arrays[key + "_region_" + name] = mask
        row = dict(
            frame_id=frame, hand=h, variants={}, wrist_region_available=wrist_available
        )
        rgb = source.rgb(i)
        paths.append(str(source.path(f"rgb/{i + source.start:06d}.png")))
        pixels = project(points, source.k)
        lo = (
            np.maximum(np.floor(pixels.min(0) - 20).astype(int), 0)
            if len(points)
            else np.zeros(2, int)
        )
        hi = (
            np.minimum(np.ceil(pixels.max(0) + 20).astype(int), rgb.shape[1::-1])
            if len(points)
            else np.array(rgb.shape[1::-1])
        )
        tiles = []
        for name, data in variants.items():
            spec = config["variants"][name]
            vertices = data[spec.get("vertices_key", "camera_vertices")][i, h]
            if not np.isfinite(vertices).all():
                row["variants"][name] = dict(unavailable=True)
                continue
            mesh = trimesh.Trimesh(vertices, data["faces"][h], process=False)
            distances = MeshSurfaceQuery(mesh).closest(points)[1]
            arrays[key + "_" + name + "_distances"] = distances
            row["variants"][name] = residual_summary(distances, masks)
            image = rgb.copy()
            for uv, distance in zip(pixels, distances):
                color = (
                    (240, 50, 40)
                    if distance > 0.03
                    else (255, 180, 20)
                    if distance > 0.02
                    else (40, 200, 100)
                )
                cv2.circle(image, tuple(np.rint(uv).astype(int)), 2, color, -1)
            tile = image[lo[1] : hi[1], lo[0] : hi[0]].copy()
            cv2.putText(
                tile, name, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2
            )
            tiles.append(tile)
        rows.append(row)
        if tiles:
            filename = key + ".jpg"
            Image.fromarray(np.concatenate(tiles, axis=1)).save(
                out / filename, quality=90
            )
            panels.append(
                f'<h2>Frame {frame}, hand {h}</h2><a href="{filename}"><img src="{filename}"></a>'
            )
    np.savez_compressed(out / "residuals.npz", **arrays)
    write_json(
        out / "audit.json",
        dict(
            schema="real2sim2real.hand-depth-audit/1",
            rows=rows,
            config=config,
            source_sha256={p: digest(p) for p in sorted(set(paths))},
            implementation_sha256={
                n: digest(Path(__file__).with_name(n))
                for n in [
                    "hand_depth_audit.py",
                    "hand_observation_fit.py",
                    "mesh_surface.py",
                ]
            },
            reserved_depth_only=True,
            used_for_fitting=False,
            changes_acceptance=False,
            region_semantics="Image neighborhoods, not anatomical segmentation; regions may overlap, remaining is their complement.",
        ),
    )
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Reserved hand depth audit</title><style>body{font:16px system-ui;background:#17202b;color:#eee;margin:25px}img{max-width:100%}a{color:#8cf}pre{white-space:pre-wrap}</style><h1>Reserved hand depth audit</h1><p>Identical reserved stereo samples in every candidate. Green: distance ≤20 mm; orange: 20–30 mm; red: >30 mm. Distances are to the complete hand mesh, not anatomical point correspondences. These diagnostics do not change acceptance gates.</p><a href="audit.json">Metrics and provenance</a>'
        + "".join(panels)
        + "<pre>"
        + html.escape(json.dumps(rows, indent=2))
        + "</pre>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    audit(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
