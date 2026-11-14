"""CPU observation-only scene screening with explicit partial geometry.

Run with an external finite timeout. Configuration owns image annotations,
static-scene frame selection, source pins and thresholds. No reference poses,
scans, inferred hidden surfaces, robot fitting or simulation are consumed.
"""

import argparse
from collections import Counter
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree
import trimesh

from real2sim2real.artifacts import digest, write_json
from . import capture
from .camera import rigid, transform
from .observed_scene import (
    finger_probe,
    polygon_mask,
    propagate_mask,
    sample_visibility,
    stereo_observations,
    surface_mesh,
    voxel_points,
)
from .stereo import Stereo
from .stereo_contact_evidence import match_rectified
from .support import fit_plane, prism_mesh, upright_transform
from .surface_patch import fit_patch, normal_repeatability


def _quantiles(values):
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    return None if not len(x) else np.quantile(x, [0.1, 0.5, 0.9]).tolist()


def run(config):
    implementation = {
        name: digest(Path(__file__).with_name(name))
        for name in [
            "scene_screening.py",
            "observed_scene.py",
            "stereo.py",
            "camera.py",
            "capture.py",
            "support.py",
            "surface_patch.py",
            "stereo_contact_evidence.py",
        ]
    }
    sources = config["sources"]
    for path, sha in sources.items():
        if digest(Path(path)) != sha:
            raise ValueError("Pinned scene-screening source changed")
    for name in ["capture", "calibration", "camera_motion"]:
        if config[name] not in sources:
            raise ValueError("Capture, calibration and motion must be pinned")
    cap = capture.load(config["capture"])
    cal = json.loads(Path(config["calibration"]).read_text())
    motion = np.load(config["camera_motion"])
    fit_ids, holdout = config["fit_frames"], config["validation_frames"]
    ids = sorted(fit_ids + holdout)
    if (
        not fit_ids
        or not holdout
        or len(ids) != len(set(ids))
        or len(ids) > 12
        or ids[0] < 0
        or ids[0] != config["annotation_frame"]
        or ids[-1] >= len(motion["accepted"])
        or not motion["accepted"][ids].all()
        or not cal["accepted"]
        or not np.array_equal(
            motion["frame_ids"], cap["frame_ids"][: len(motion["frame_ids"])]
        )
        or not np.allclose(
            motion["timestamps_s"],
            cap["timestamps_s"][: len(motion["frame_ids"])],
            atol=1e-8,
            rtol=0,
        )
    ):
        raise ValueError(
            "Disjoint valid static frames, accepted calibration and matching motion required"
        )
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", config)
    left, right = [cap["streams"][s] for s in cap["stereo_pair"]]
    model = Stereo(
        left,
        right,
        cap["streams"][cap["target_stream"]],
        right_from_left=cal["candidate_right_from_left"],
    )
    videos = [
        {i: im for i, im in enumerate(capture.frames(s, max(ids) + 1)) if i in ids}
        for s in (left, right)
    ]
    from_rect = np.eye(4)
    from_rect[:3, :3] = model.rotations[0].T
    evidence, poses, points, masks, rows = {}, {}, {}, {}, []
    colors = {name: region["color"] for name, region in config["regions"].items()}
    for i in ids:
        e = stereo_observations(
            model, videos[0][i], videos[1][i], **config["stereo_checks"]
        )
        evidence[i] = e
        pose = (
            rigid(motion["world_from_camera"][i]) @ model.target_from_left @ from_rect
        )
        poses[i] = pose
        # Do not pass invalid disparity infinities through rigid arithmetic.
        finite = np.where(e["raw_valid"][..., None], e["points"], 0)
        points[i] = transform(finite.reshape(-1, 3), pose).reshape(*e["valid"].shape, 3)
        if i == ids[0]:
            initial = {
                name: polygon_mask(e["left"].shape, region["polygons_px"])
                for name, region in config["regions"].items()
            }
        masks[i] = {}
        overlay = cv2.cvtColor(e["left"], cv2.COLOR_GRAY2RGB)
        records = {}
        for name, region in config["regions"].items():
            m = (
                initial[name].copy()
                if i == ids[0]
                else propagate_mask(evidence[ids[0]]["left"], e["left"], initial[name])
            )
            erode = config["mask_erosion_px"]
            m = cv2.erode(
                m.astype(np.uint8), np.ones((2 * erode + 1,) * 2, np.uint8)
            ).astype(bool)
            masks[i][name] = m
            selected = m & e["valid"]
            records[name] = dict(
                mask_pixels=int(m.sum()),
                raw_depth_pixels=int((m & e["raw_valid"]).sum()),
                reliable_depth_pixels=int(selected.sum()),
                reliable_fraction=float(selected.sum() / max(1, m.sum())),
                texture_q10_q50_q90=_quantiles(e["texture_std"][m]),
                half_pixel_depth_sensitivity_q10_q50_q90_m=_quantiles(
                    e["half_pixel_depth_sensitivity_m"][selected]
                ),
            )
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, colors[name], 2)
            overlay[selected] = (
                0.4 * overlay[selected] + 0.6 * np.array(colors[name])
            ).astype(np.uint8)
        Image.fromarray(overlay).save(out / f"observations-{i:04d}.jpg")
        Image.fromarray(e["left"]).save(out / f"left-{i:04d}.jpg")
        Image.fromarray(e["right"]).save(out / f"right-{i:04d}.jpg")
        np.savez_compressed(
            out / f"observations-{i:04d}.npz",
            **e,
            world_from_rectified=pose,
            **{"mask_" + n: m for n, m in masks[i].items()},
        )
        rows.append(
            dict(
                frame=i,
                timestamp_s=cap["timestamps_s"][i],
                split="fit" if i in fit_ids else "validation",
                regions=records,
            )
        )
        print(
            f"observed frame {i}: "
            + str({n: r["reliable_depth_pixels"] for n, r in records.items()}),
            flush=True,
        )
    table = config["table_region"]
    plane_points = np.concatenate(
        [points[i][masks[i][table] & evidence[i]["valid"]] for i in fit_ids]
    )
    n, d, inliers = fit_plane(
        plane_points, threshold=config["table_checks"]["inlier_m"]
    )
    upright = upright_transform(n, d, np.median(plane_points[inliers], axis=0))
    checks = []
    for i in holdout:
        p = points[i][masks[i][table] & evidence[i]["valid"]]
        residual = abs(transform(p, upright)[:, 2])
        q = _quantiles(residual)
        checks.append(
            dict(
                frame=i,
                points=len(p),
                q10_q50_q90_m=q,
                accepted=bool(
                    len(p) >= 100
                    and q[1] <= config["table_checks"]["median_m"]
                    and q[2] <= config["table_checks"]["p90_m"]
                ),
            )
        )
    support_accepted = all(r["accepted"] for r in checks)
    xy = transform(plane_points[inliers], upright)[:, :2]
    low, high = np.quantile(xy, [0.005, 0.995], axis=0)
    xy = xy[((xy >= low) & (xy <= high)).all(1)]
    boundary = xy[ConvexHull(xy).vertices]
    slab = prism_mesh(boundary, config["table_checks"]["slab_thickness_m"])
    slab.export(out / "table-slab-hypothesis.ply")
    write_json(
        out / "support.json",
        dict(
            accepted=support_accepted,
            validation=checks,
            fit_inlier_fraction=float(inliers.mean()),
            upright_from_world=upright.tolist(),
            observed_boundary_xy=boundary.tolist(),
            assumption="selected support is level; convex hull spans occluded tabletop; slab thickness assumed",
            thickness_m=config["table_checks"]["slab_thickness_m"],
        ),
    )
    cloud, meshes, geometry = {}, {}, {}
    for name in config["regions"]:
        parts, samples = [], []
        for i in ids:
            p = transform(points[i].reshape(-1, 3), upright).reshape(points[i].shape)
            selected = masks[i][name] & evidence[i]["valid"]
            samples.append(p[selected])
            parts.append(
                surface_mesh(p, selected, maximum_edge_m=config["maximum_mesh_edge_m"])
            )
        cloud[name] = voxel_points(np.concatenate(samples), config["voxel_m"])
        meshes[name] = trimesh.util.concatenate(parts)
        meshes[name].visual.vertex_colors = [*colors[name], 255]
        meshes[name].export(out / f"{name}-observed.ply")
        trimesh.points.PointCloud(cloud[name], colors=colors[name]).export(
            out / f"{name}-points.ply"
        )
        fit_cloud = np.concatenate(
            [
                transform(points[i][masks[i][name] & evidence[i]["valid"]], upright)
                for i in fit_ids
            ]
        )
        held_cloud = np.concatenate(
            [
                transform(points[i][masks[i][name] & evidence[i]["valid"]], upright)
                for i in holdout
            ]
        )
        repeat_distance = (
            cKDTree(fit_cloud).query(held_cloud)[0]
            if len(fit_cloud) and len(held_cloud)
            else []
        )
        geometry[name] = dict(
            points=len(cloud[name]),
            faces=len(meshes[name].faces),
            observed_bounds_m=None
            if not len(cloud[name])
            else [cloud[name].min(0).tolist(), cloud[name].max(0).tolist()],
            below_table_5mm_fraction=float((cloud[name][:, 2] < -0.005).mean())
            if len(cloud[name])
            else None,
            heldout_to_fit_nearest_q10_q50_q90_m=_quantiles(repeat_distance),
            nearest_distance_is_not_surface_accuracy=True,
            closed_solid=False,
            hidden_surfaces_completed=False,
            independent_frame_resampling=False,
        )
    # Separate sparse NCC diagnostic with full disparity search. Select image
    # pixels uniformly before consulting dense depth; no favorable depth seed.
    sparse = []
    for i in ids:
        for name in config["regions"]:
            yy, xx = np.nonzero(masks[i][name])
            chosen = np.linspace(
                0,
                len(xx) - 1,
                min(len(xx), config["sparse_samples_per_region"]),
                dtype=int,
            )
            results = [
                match_rectified(
                    evidence[i]["left"],
                    evidence[i]["right"],
                    [xx[j], yy[j]],
                    **config["sparse_checks"],
                )
                for j in chosen
            ]
            sparse.append(
                dict(
                    frame=i,
                    region=name,
                    samples=len(results),
                    accepted=sum(r["accepted"] for r in results),
                    reasons=dict(Counter(r["reason"] for r in results)),
                    results=results,
                )
            )
        print(f"sparse stereo frame {i} complete", flush=True)
    write_json(out / "sparse-matches.json", sparse)
    # Local pad patches and straight-cylinder access probe. No opposing grasp or
    # robot collision admission follows automatically from this screen.
    patch_name = config["contact_region"]
    patch_rows = []
    base = ids[0]
    base_world = transform(points[base].reshape(-1, 3), upright).reshape(
        points[base].shape
    )
    yy, xx = np.nonzero(masks[base][patch_name] & evidence[base]["valid"])
    selected = np.linspace(
        0, len(xx) - 1, min(len(xx), config["maximum_patch_candidates"]), dtype=int
    )
    per_frame = {
        i: transform(points[i][masks[i][patch_name] & evidence[i]["valid"]], upright)
        for i in ids
    }
    trees = {i: cKDTree(p) for i, p in per_frame.items()}
    for j in selected:
        center = base_world[yy[j], xx[j]]
        local = per_frame[base][
            trees[base].query_ball_point(center, config["patch_checks"]["radius_m"])
        ]
        if len(local) < 9:
            patch_rows.append(
                dict(
                    pixel=[int(xx[j]), int(yy[j])],
                    accepted=False,
                    reason="insufficient_metric_points",
                )
            )
            continue
        _, _, vh = np.linalg.svd(local - local.mean(0), full_matrices=False)
        hint = vh[-1]
        view = transform(poses[base][:3, 3][None], upright)[0] - center
        hint *= 1 if hint @ view >= 0 else -1
        fits = [
            dict(
                frame=i,
                **fit_patch(per_frame[i], center, hint, **config["patch_checks"]),
            )
            for i in ids
        ]
        repeated = normal_repeatability(fits, **config["repeatability_checks"])
        # A prior-free local normal has no independent ground-truth normal.
        # The check asks whether other observed frames corroborate it.
        probe = finger_probe(center, hint, **config["finger_probe"])
        free = np.zeros(len(probe), bool)
        near = np.zeros(len(probe), bool)
        for i in ids:
            labels = sample_visibility(
                probe,
                upright @ poses[i],
                model.projections[0],
                evidence[i]["points"][..., 2],
                evidence[i]["valid"],
                margin_m=config["visibility_margin_m"],
            )
            free |= labels == 1
            near |= labels == 2
        clearance = float(probe[:, 2].min())
        patch_rows.append(
            dict(
                pixel=[int(xx[j]), int(yy[j])],
                center_m=center.tolist(),
                normal=hint.tolist(),
                accepted=bool(repeated["accepted"]),
                fits=fits,
                repeatability=repeated,
                access=dict(
                    sampled_free_fraction=float(free.mean()),
                    never_observed_free_fraction=float((~free).mean()),
                    surface_or_occluded_fraction=float(near.mean()),
                    minimum_table_clearance_m=clearance,
                    sampled_access_clear=bool(
                        support_accepted
                        and free.all()
                        and not near.any()
                        and clearance >= config["minimum_table_clearance_m"]
                    ),
                    robot_insertion_verified=False,
                ),
            )
        )
    write_json(out / "contact-patches.json", patch_rows)
    summary = dict(
        schema="real2sim2real.observed-scene-screen/1",
        units="metres",
        up_axis="+Z",
        config_sha256=digest(out / "config.json"),
        source_hashes=sources,
        implementation_sha256=implementation,
        reference_poses_or_scans_used=False,
        static_scene_assumption=config["static_scene_assumption"],
        calibration_accepted=cal["accepted"],
        baseline_m=cal["baseline_m"],
        calibration_parameter_bound_hits=int(
            np.count_nonzero(
                np.isclose(abs(np.asarray(cal["fit_parameters"])), 0.035, atol=1e-6)
            )
        ),
        support_accepted=support_accepted,
        support_validation=checks,
        frames=rows,
        geometry=geometry,
        contact=dict(
            candidates=len(patch_rows),
            repeated_patches=sum(r["accepted"] for r in patch_rows),
            sampled_clear_probes=sum(
                r.get("access", {}).get("sampled_access_clear", False) and r["accepted"]
                for r in patch_rows
            ),
            opposing_grasp_verified=False,
            robot_insertion_verified=False,
        ),
        control_ready=False,
        physics_run=False,
        limitations=[
            "partial visible surfaces; holes remain missing",
            "SGBM/NCC share images and calibration",
            "half-pixel sensitivity excludes calibration/systematic error",
            "normal from local points, not independent ground truth",
            "table thickness assumed; no object mass/thickness/hidden-side recovery",
            "straight finger probe is not articulated-hand or arm collision checking",
        ],
    )
    for path, sha in sources.items():
        if digest(Path(path)) != sha:
            raise ValueError("Source changed during scene screening")
    for name, sha in implementation.items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError("Implementation changed during scene screening")
    write_json(out / "summary.json", summary)
    _review(out, summary, config, cloud)
    return summary


def _review(out, summary, config, cloud):
    # Offline point viewer uses only inline data; media remains local.
    points = []
    for name, p in cloud.items():
        indices = np.linspace(0, len(p) - 1, min(len(p), 5000), dtype=int)
        points.extend(
            [*np.round(q, 5).tolist(), config["regions"][name]["color"]]
            for q in p[indices]
        )
    rows = "".join(
        f'<tr><td style="color:rgb({",".join(map(str, config["regions"][n]["color"]))})">{html.escape(n)}</td><td>{g["points"]}</td><td>{g["faces"]}</td><td><a href="{n}-observed.ply">partial surface</a> / <a href="{n}-points.ply">points</a></td></tr>'
        for n, g in summary["geometry"].items()
    )
    images = "".join(
        f'<figure><img src="observations-{r["frame"]:04d}.jpg"><figcaption>Frame {r["frame"]} ({r["timestamp_s"]:.3f} s); outlines: image regions; colored pixels: filtered stereo. <a href="left-{r["frame"]:04d}.jpg">Left</a> / <a href="right-{r["frame"]:04d}.jpg">right</a></figcaption></figure>'
        for r in summary["frames"]
    )
    text = """<!doctype html><meta charset="utf-8"><title>Observed metric scene</title>
<style>body{font:16px system-ui;max-width:1200px;margin:24px auto;background:#161b22;color:#e6edf3}a{color:#79c0ff}img{width:100%}canvas{width:100%;height:600px;background:#080e18;touch-action:none}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #555}video{max-width:100%}</style>
<h1>Observed metric scene: stereo screening</h1><p>Partial measured geometry; not a closed collision scene or control acceptance. No physics or training.</p>
<p>Drag to rotate the point cloud; wheel to zoom. Colors match the image outlines. World units are metres; table plane is z=0. Empty space in this viewer is not proof of clearance.</p><canvas id="v" width="1200" height="600"></canvas>
<table><tr><th>Region</th><th>Observed voxel samples</th><th>Observed triangles</th><th>Downloads</th></tr>ROWS</table>
<p>DETAILS</p><p><a href="summary.json">Full diagnostics</a> · <a href="support.json">Table plane</a> · <a href="contact-patches.json">Contact/access screening</a> · <a href="sparse-matches.json">Separate NCC matches</a></p>
<p>Interior and exterior regions are not interchangeable contact surfaces. Hidden surfaces and holes remain missing. Table slab thickness is assumed. Held-out epipolar agreement is not an independent metric-scale measurement; inspect the recorded calibration parameter bound hits.</p>
IMAGES
<script>const pts=POINTS,c=document.getElementById('v'),ctx=c.getContext('2d');let az=0.3,el=0.8,zoom=850,last=null;
const ctr=[0,1,2].map(i=>pts.reduce((s,p)=>s+p[i],0)/Math.max(1,pts.length));
function draw(){ctx.clearRect(0,0,c.width,c.height);const projected=pts.map(p=>{let x=p[0]-ctr[0],y=p[1]-ctr[1],z=p[2]-ctr[2];let a=Math.cos(az)*x-Math.sin(az)*y,b=Math.sin(az)*x+Math.cos(az)*y;return [600+zoom*a,300+zoom*(Math.cos(el)*b-Math.sin(el)*z),Math.sin(el)*b+Math.cos(el)*z,p[3]]});projected.sort((a,b)=>a[2]-b[2]);for(const p of projected){ctx.fillStyle='rgb('+p[3].join(',')+')';ctx.fillRect(p[0],p[1],2,2)}}
c.onpointerdown=e=>{last=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>last=null;c.onpointermove=e=>{if(last){az+=(e.clientX-last[0])*0.008;el+=(e.clientY-last[1])*0.008;last=[e.clientX,e.clientY];draw()}};c.onwheel=e=>{e.preventDefault();zoom=Math.max(100,Math.min(5000,zoom*Math.exp(-e.deltaY*0.001)));draw()};draw();</script>"""
    details = html.escape(
        f"Table validation accepted: {summary['support_accepted']}. Repeated local patches: {summary['contact']['repeated_patches']}/{summary['contact']['candidates']}; sampled-clear probes: {summary['contact']['sampled_clear_probes']}. Opposing grasp and robot insertion remain unverified."
    )
    (out / "review.html").write_text(
        text.replace("ROWS", rows)
        .replace("DETAILS", details)
        .replace("IMAGES", images)
        .replace("POINTS", json.dumps(points))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
