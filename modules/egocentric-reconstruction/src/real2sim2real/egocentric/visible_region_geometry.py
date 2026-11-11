"""Observation-defined surface regions versus immutable pose hypotheses."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import trimesh

from real2sim2real.artifacts import digest, write_json
from .camera import transform, unproject
from .interaction_io import Reconstruction, fresh_directory


def image_regions(mask, occluder, lower_fraction=0.6, border=2, context=18):
    """Focus on a declared lower image region, never on favorable 3D residuals."""
    mask, occluder = np.asarray(mask, bool), np.asarray(occluder, bool)
    if (
        mask.shape != occluder.shape
        or mask.ndim != 2
        or not mask.any()
        or not 0 < lower_fraction < 1
        or not 0 < border < context
    ):
        raise ValueError(
            "Explicit nonempty mask and bounded image-region settings required"
        )
    y, x = np.nonzero(mask)
    yy, _ = np.indices(mask.shape)
    lower = yy >= y.min() + lower_fraction * (y.max() - y.min())
    safe = ~ndimage.binary_dilation(occluder, iterations=border)
    solid = ndimage.binary_erosion(mask, iterations=border) & safe
    hole = ndimage.binary_fill_holes(mask) & ~mask
    return dict(
        focus=solid & lower,
        body=solid & ~lower,
        hole=ndimage.binary_erosion(hole, iterations=border) & safe & lower,
        nearby_table=ndimage.binary_dilation(mask & lower, iterations=context)
        & ~ndimage.binary_dilation(mask, iterations=border)
        & ~hole
        & safe
        & lower,
    )


def ray_depth(mesh, camera_from_object, uv, k):
    """First visible triangle depth for specified pinhole image rays."""
    uv = np.asarray(uv, float)
    t = np.asarray(camera_from_object)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("Pixel coordinate pairs required")
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(k).T
    object_from_camera = np.linalg.inv(t)
    origins = np.tile(object_from_camera[:3, 3], (len(uv), 1))
    directions = rays @ object_from_camera[:3, :3].T
    hit, ids, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
    depth = np.full(len(uv), np.inf)
    if len(hit):
        z = transform(hit, t)[:, 2]
        ok = z > 0
        np.minimum.at(depth, ids[ok], z[ok])
    depth[~np.isfinite(depth)] = np.nan
    return depth


def distribution(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(
        count=len(values),
        q10_m=None if not len(values) else float(np.quantile(values, 0.1)),
        median_m=None if not len(values) else float(np.median(values)),
        q90_m=None if not len(values) else float(np.quantile(values, 0.9)),
    )


def audit(config):
    import cv2
    from . import capture, stereo_contact_evidence
    from .stereo import Stereo

    for p, sha in config["sources"].items():
        if digest(Path(p)) != sha:
            raise ValueError("Pinned geometry-audit source changed")
    source = Reconstruction(config["reconstruction"])
    frames = config["fit_frames"] + config["validation_frames"]
    if (
        not config["fit_frames"]
        or not config["validation_frames"]
        or len(set(frames)) != len(frames)
        or len(frames) > 20
        or min(frames) < 0
        or max(frames) >= source.count
    ):
        raise ValueError("Disjoint bounded capture frames required")
    support = json.loads(Path(config["support"]).read_text())
    if (
        not support["accepted"]
        or support["units"] != "metres"
        or support["source_geometry_sha256"] != digest(source.path("geometry.npz"))
    ):
        raise ValueError("Accepted matching metric support required")
    cap = capture.load(config["capture"])
    cal = json.loads(Path(config["calibration"]).read_text())
    if (
        not cal["accepted"]
        or not np.array_equal(source.frame_ids, cap["frame_ids"])
        or not np.allclose(source.times, cap["timestamps_s"], rtol=0, atol=1e-8)
    ):
        raise ValueError("Accepted calibration and matching capture times required")
    left, right = [cap["streams"][n] for n in cap["stereo_pair"]]
    stereo = Stereo(
        left,
        right,
        cap["streams"][cap["target_stream"]],
        right_from_left=cal["candidate_right_from_left"],
    )
    videos = [
        {
            i: im
            for i, im in enumerate(capture.frames(stream, max(frames) + 1))
            if i in frames
        }
        for stream in [left, right]
    ]
    mesh = trimesh.load(source.path("mesh/metric.glb"), force="mesh", process=False)
    u = np.asarray(support["upright_from_world"])
    original = u @ source.world_from_object
    settling = json.loads(Path(config["settling"]).read_text())
    if not settling["accepted"] or not np.allclose(
        settling["initial_world_from_object"], original[0], atol=1e-6, rtol=0
    ):
        raise ValueError("Settled hypothesis must start from this reconstruction")
    settled = np.asarray(settling["settled_world_from_object"])
    correction = settled @ np.linalg.inv(original[0])
    out = fresh_directory(config["output"])
    rows = []
    hashes = dict(config["sources"])
    for frame in sorted(frames):
        d = source.depth(frame)
        mask = source.mask(frame, 1)
        occ = source.mask(frame, 2) | source.mask(frame, 3)
        regions = image_regions(mask, occ, **config["regions"])
        world_from_camera = u @ source.world_from_camera[frame]
        points = transform(
            unproject(d, source.k).reshape(-1, 3), world_from_camera
        ).reshape(*d.shape, 3)
        display = Image.fromarray(source.rgb(frame))
        draw = ImageDraw.Draw(display)
        records = {}
        for name, region in regions.items():
            y, x = np.nonzero(region & np.isfinite(d) & (d > 0))
            # Span the whole region even when its size is just below twice the
            # sample budget; truncated floor-strides would select only its top.
            indices = np.linspace(
                0, len(x) - 1, min(len(x), config["maximum_region_samples"]), dtype=int
            )
            y, x = y[indices], x[indices]
            observed = d[y, x]
            uv = np.c_[x, y]
            record = dict(
                region_pixels=int(region.sum()),
                depth_pixels=int((region & (d > 0)).sum()),
                observed_heights=distribution(points[y, x, 2]),
                hypotheses={},
            )
            if len(uv) and name in ["focus", "body"]:
                predicted = {}
                for label, pose in [
                    ("original", original[frame]),
                    ("settled_offset", correction @ original[frame]),
                ]:
                    pred = ray_depth(
                        mesh, np.linalg.inv(world_from_camera) @ pose, uv, source.k
                    )
                    predicted[label] = pred
                    valid = np.isfinite(pred)
                    record["hypotheses"][label] = dict(
                        ray_hit_fraction=float(valid.mean()),
                        signed_depth_error=distribution(pred[valid] - observed[valid]),
                        absolute_depth_error=distribution(
                            abs(pred[valid] - observed[valid])
                        ),
                    )
                common = np.logical_and.reduce(
                    [np.isfinite(p) for p in predicted.values()]
                )
                record["common_ray_count"] = int(common.sum())
                record["paired_absolute_depth_error"] = {
                    label: distribution(abs(p[common] - observed[common]))
                    for label, p in predicted.items()
                }
            records[name] = record
            color = {
                "focus": "cyan",
                "body": "yellow",
                "hole": "magenta",
                "nearby_table": "orange",
            }[name]
            for px, py in uv[:: max(1, len(uv) // 100)]:
                draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=color)
        # Detect image corners in a rectified ROI. Geometry proposes only the ROI;
        # disparity search and final region membership do not use model normals.
        rect = [
            cv2.remap(
                cv2.cvtColor(v[frame], cv2.COLOR_RGB2GRAY),
                *stereo.maps[j],
                interpolation=cv2.INTER_LINEAR,
            )
            for j, v in enumerate(videos)
        ]
        focus_y, focus_x = np.nonzero(regions["focus"] & (d > 0))
        samples = unproject(d, source.k)[focus_y, focus_x]
        in_left = (
            transform(samples, np.linalg.inv(stereo.target_from_left))
            @ stereo.rotations[0].T
        )
        projection = in_left @ stereo.projections[0][:, :3].T
        uv = projection[:, :2] / projection[:, 2:]
        roi = np.zeros(rect[0].shape, np.uint8)
        if len(uv):
            lo = np.maximum(np.floor(np.quantile(uv, 0.05, axis=0) - 15).astype(int), 0)
            hi = np.minimum(
                np.ceil(np.quantile(uv, 0.95, axis=0) + 15).astype(int),
                np.array(roi.shape[::-1]) - 1,
            )
            roi[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1] = 255
        corners = cv2.goodFeaturesToTrack(
            rect[0],
            maxCorners=config["maximum_corners"],
            qualityLevel=0.01,
            minDistance=4,
            mask=roi,
        )
        matches = []
        reasons = Counter()
        for px in [] if corners is None else corners[:, 0]:
            match = stereo_contact_evidence.match_rectified(
                *rect, px, **config["matching"]
            )
            reasons[match["reason"]] += 1
            if not match["accepted"]:
                continue
            xyz = stereo_contact_evidence.triangulate_match(match, stereo.projections)
            target = transform(
                (xyz @ stereo.rotations[0])[None], stereo.target_from_left
            )[0]
            p = source.k @ target
            p = np.rint(p[:2] / p[2]).astype(int)
            x, y = p
            names = [
                n
                for n, m in regions.items()
                if 0 <= x < m.shape[1] and 0 <= y < m.shape[0] and m[y, x]
            ]
            world = transform(target[None], world_from_camera)[0]
            matches.append(
                dict(
                    pixel_rectified=px.tolist(),
                    pixel_target=p.tolist(),
                    region=names[0] if names else "outside_regions",
                    point_upright_m=world.tolist(),
                    match=match,
                )
            )
            if names:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="red", width=1)
        row = dict(
            frame_id=int(source.frame_ids[frame]),
            timestamp_s=float(source.times[frame]),
            split="fit" if frame in config["fit_frames"] else "validation",
            regions=records,
            direct_stereo=matches,
            matching_reasons=dict(reasons),
        )
        rows.append(row)
        display.save(out / f"frame-{frame:06d}.jpg", quality=92)
        for f in [
            f"depth/{frame:06d}.npy",
            f"rgb/{frame:06d}.png",
            *[f"masks/{i}/{frame:06d}.png" for i in [1, 2, 3]],
        ]:
            hashes[str(source.path(f))] = digest(source.path(f))
        print(
            frame,
            {n: r["observed_heights"]["median_m"] for n, r in records.items()},
            "matches",
            len(matches),
            flush=True,
        )
    report = dict(
        schema="real2sim2real.visible-region-geometry/1",
        config=config,
        frames=rows,
        sources=hashes,
        pose_correction_fitted=False,
        geometry_changed=False,
        contacts_validated=False,
        robot_fitting_run=False,
        interpretation="Original and settled-offset hypotheses scored on the same observation-defined pixels. Direct stereo is a consistency check, not independent ground truth.",
        implementation_sha256=digest(Path(__file__)),
    )
    write_json(out / "audit.json", report)
    table = "".join(
        f"<tr><td>{r['frame_id']}</td>"
        + "".join(
            f"<td>{1000 * r['regions'][n]['observed_heights']['median_m']:.1f}</td>"
            if r["regions"][n]["observed_heights"]["median_m"] is not None
            else "<td>missing</td>"
            for n in ["focus", "body", "hole", "nearby_table"]
        )
        + "</tr>"
        for r in rows
    )
    paired = []
    for row in rows:
        region = row["regions"]["focus"]
        values = [row["frame_id"], row["split"], region.get("common_ray_count", 0)]
        for hypothesis in ["original", "settled_offset"]:
            error = (
                region.get("paired_absolute_depth_error", {})
                .get(hypothesis, {})
                .get("median_m")
            )
            values.append("missing" if error is None else f"{error * 1000:.1f}")
            coverage = region["hypotheses"].get(hypothesis, {}).get("ray_hit_fraction")
            values.append("missing" if coverage is None else f"{100 * coverage:.1f}%")
        paired.append("<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>")
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Visible-region geometry audit</title><style>body{font:17px system-ui;max-width:1100px;margin:30px auto;background:#18202b;color:#eee}img{width:100%}td,th{padding:8px}a{color:#8df}</style><h1>Observed region geometry and local table checks</h1><p>Cyan: lower object region; yellow: body; magenta: enclosed hole; orange: nearby background; red: accepted reciprocal stereo feature assigned to a region. Region names are image proxies, not verified semantic surfaces. Source depth and direct stereo share the same cameras. Hidden bearing contacts remain unknown.</p><a href="audit.json">Full evidence and source hashes</a><table><tr><th>Frame</th><th>Focus height mm</th><th>Body height mm</th><th>Hole height mm</th><th>Nearby background height mm</th></tr>'
        + table
        + "</table>"
        + "<p>Lower-region comparison: median absolute camera-depth errors use only the identical pixels hit by both mesh hypotheses. Coverage uses all sampled valid depth pixels. No pose or shape is fitted. Original and settled-offset are competing fixed hypotheses, not ground truth.</p>"
        + "<table><tr><th>Frame</th><th>Split</th><th>Common rays</th><th>Original error mm</th><th>Original coverage</th><th>Settled error mm</th><th>Settled coverage</th></tr>"
        + "".join(paired)
        + "</table>"
        + "".join(
            f'<h2>Frame {r["frame_id"]}</h2><img src="frame-{r["frame_id"]:06d}.jpg">'
            for r in rows
        )
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    if not a.worker:
        from real2sim2real.bounded_process import run

        return run(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(a.config.resolve()),
                "--worker",
            ],
            timeout=c["deadline_s"],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    audit(c)


if __name__ == "__main__":
    main()
