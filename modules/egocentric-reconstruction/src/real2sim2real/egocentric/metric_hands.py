"""Align articulated hand priors to calibrated RGB and measured stereo surfaces.

The hand's shape and scale are constant within a sequence. No object contact is
used to force a fit. Missing/rejected estimates remain NaN in accepted outputs.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, write_json
from .camera import transform, unproject
from .interaction_io import Reconstruction, fresh_directory


def project(points, k):
    xyz = np.asarray(points) @ np.asarray(k).T
    return xyz[..., :2] / np.maximum(xyz[..., 2:], 1e-6)


def observed_hand_surface(depth, mask, keypoints, k, *, max_points=800):
    """Only observed skin within the hand's image footprint, excluding forearm."""
    hull = cv2.convexHull(np.rint(keypoints).astype(np.int32))
    footprint = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(footprint, hull, 1)
    footprint = cv2.dilate(footprint, np.ones((13, 13), np.uint8)) > 0
    skin = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    valid = footprint & skin & np.isfinite(depth) & (depth > 0.15) & (depth < 1.5)
    y, x = np.nonzero(valid)
    points = unproject(depth, k)[valid]
    if len(points):
        median = np.median(points[:, 2])
        keep = np.abs(points[:, 2] - median) < 0.12
        points, y, x = points[keep], y[keep], x[keep]
    # Spatially interleaved fit/validation measurements are never shared.
    parity = ((x // 3 + y // 3) % 2).astype(bool)
    fit, validation = points[~parity], points[parity]
    rng = np.random.default_rng(0)
    if len(fit) > max_points:
        fit = fit[rng.choice(len(fit), max_points, replace=False)]
    if len(validation) > max_points:
        validation = validation[rng.choice(len(validation), max_points, replace=False)]
    return fit, validation


def fit_hand(
    vertices,
    faces,
    joints,
    keypoints,
    points,
    k,
    *,
    fixed_scale=None,
    initial=None,
    max_rotation_deg=35,
):
    """Bounded rigid/similarity fit, using point-to-triangle correspondences."""
    import trimesh

    if len(points) < 80 or not np.isfinite(keypoints).all():
        raise ValueError("Insufficient measured hand surface or image landmarks")
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    if initial is None:
        z = np.median(points[:, 2])
        centre = np.mean(keypoints, axis=0)
        t = (np.linalg.inv(k) @ [*centre, 1]) * z - np.mean(joints, axis=0)
        x = np.r_[np.zeros(3), t, 0.0]
    else:
        x = np.asarray(initial).copy()
    fixed = fixed_scale is not None
    if fixed:
        x[6] = np.log(fixed_scale)
    angle = np.deg2rad(max_rotation_deg)
    low = np.array([-angle] * 3 + [-2, -2, 0.1] + [np.log(0.6)])
    high = np.array([angle] * 3 + [2, 2, 1.6] + [np.log(1.6)])
    x = np.clip(x, low + 1e-7, high - 1e-7)

    def unpack(parameters):
        if fixed:
            parameters = np.r_[parameters, np.log(fixed_scale)]
        return (
            Rotation.from_rotvec(parameters[:3]).as_matrix(),
            parameters[3:6],
            np.exp(parameters[6]),
        )

    parameter = x[:6] if fixed else x
    for iteration in range(20):
        r, t, scale = unpack(parameter)
        local_points = (points - t) @ r / scale
        closest, distances, _ = trimesh.proximity.closest_point(mesh, local_points)
        threshold = max(0.01, np.quantile(distances, 0.8))
        usable = distances <= min(threshold, 0.08 / scale)
        if usable.sum() < 50:
            raise ValueError("Too few consistent hand-surface correspondences")
        a, b = closest[usable], points[usable]

        def residual(parameters):
            r, t, scale = unpack(parameters)
            uv = project(joints * scale @ r.T + t, k)
            image_error = ((uv - keypoints) / 5).ravel() / np.sqrt(len(joints))
            surface_error = ((a * scale @ r.T + t - b) / 0.006).ravel() / np.sqrt(
                len(a)
            )
            # Mild priors constrain the unobserved side of the hand.
            priors = parameters[:3] / np.deg2rad(25)
            if not fixed:
                priors = np.r_[priors, parameters[6] / 0.25]
            return np.r_[image_error, surface_error, 0.15 * priors]

        result = least_squares(
            residual,
            parameter,
            bounds=(low[: len(parameter)], high[: len(parameter)]),
            loss="soft_l1",
            f_scale=0.5,
            max_nfev=50,
        )
        change = np.linalg.norm(result.x - parameter)
        parameter = result.x
        if change < 1e-5:
            break
    if fixed:
        parameter = np.r_[parameter, np.log(fixed_scale)]
    return parameter


def apply_fit(vertices, parameters):
    r = Rotation.from_rotvec(parameters[:3]).as_matrix()
    return np.asarray(vertices) * np.exp(parameters[6]) @ r.T + parameters[3:6]


def reconstruct_hands(config):
    import trimesh

    source = Reconstruction(config["reconstruction"], config.get("frame_count"))
    root = Path(config["output"])
    support = json.loads((root / "support/support.json").read_text())
    if not support["accepted"]:
        raise ValueError("Support must pass validation before hand alignment")
    observations = np.load(root / "hands2d/landmarks.npz")
    raw = np.load(root / "hands_raw/predictions.npz")
    if not np.array_equal(raw["frame_ids"], source.frame_ids) or not np.array_equal(
        observations["frame_ids"], source.frame_ids
    ):
        raise ValueError("Hand and scene frame IDs must agree")
    options = config.get("metric_hands", {})
    mask_ids = config["hand_landmarks"].get("mask_ids", [2, 3])
    out = fresh_directory(root / "hands_metric")
    n = source.count
    candidates = np.full((n, 2, 7), np.nan)
    records = [
        [
            dict(
                frame=i,
                hand=["left", "right"][h],
                accepted=False,
                reason="missing_observation",
            )
            for h in range(2)
        ]
        for i in range(n)
    ]
    surfaces = {}
    for i, h in np.argwhere(raw["accepted"] & observations["accepted"]):
        points, validation = observed_hand_surface(
            source.depth(i),
            source.mask(i, mask_ids[h]),
            observations["keypoints_px"][i, h],
            source.k,
        )
        surfaces[i, h] = (points, validation)
        try:
            parameters = fit_hand(
                raw["vertices"][i, h],
                raw["faces"][h],
                raw["joints"][i, h],
                observations["keypoints_px"][i, h],
                points,
                source.k,
            )
            candidates[i, h] = parameters
            records[i][h]["reason"] = "candidate"
        except ValueError as error:
            records[i][h]["reason"] = str(error)
    scales = []
    for h in range(2):
        values = np.exp(candidates[np.isfinite(candidates[:, h]).all(1), h, 6])
        if len(values) < options.get("minimum_scale_frames", 10):
            write_json(
                out / "quality.json",
                dict(
                    accepted=False,
                    reason="insufficient_scale_observations",
                    records=records,
                ),
            )
            raise ValueError(
                f"Too few observations to estimate a constant scale for hand {h}"
            )
        scales.append(float(np.median(values)))
    if options.get("fixed_scale_from"):
        scales = json.loads(Path(options["fixed_scale_from"]).read_text())[
            "scale_per_hand"
        ]
    accepted = np.zeros((n, 2), bool)
    camera_vertices = np.full(raw["vertices"].shape, np.nan, np.float32)
    camera_joints = np.full(raw["joints"].shape, np.nan, np.float32)
    proposed_vertices = camera_vertices.copy()
    proposed_joints = camera_joints.copy()
    fits = candidates.copy()
    limits = dict(
        max_reprojection_median_px=12.0,
        max_reprojection_p90_px=25.0,
        max_validation_surface_median_m=0.012,
        max_validation_surface_p90_m=0.03,
        min_validation_points=80,
        max_wrist_speed_m_s=3.0,
    )
    limits.update(options.get("limits", {}))
    upright_from_camera = (
        np.asarray(support["upright_from_world"]) @ source.world_from_camera
    )
    previous = [None, None]
    for i, h in np.argwhere(np.isfinite(candidates).all(2)):
        points, validation = surfaces[i, h]
        try:
            p = fit_hand(
                raw["vertices"][i, h],
                raw["faces"][h],
                raw["joints"][i, h],
                observations["keypoints_px"][i, h],
                points,
                source.k,
                fixed_scale=scales[h],
                initial=candidates[i, h],
            )
            fits[i, h] = p
            v, j = (
                apply_fit(raw["vertices"][i, h], p),
                apply_fit(raw["joints"][i, h], p),
            )
            proposed_vertices[i, h], proposed_joints[i, h] = v, j
            reprojection = np.linalg.norm(
                project(j, source.k) - observations["keypoints_px"][i, h], axis=1
            )
            mesh = trimesh.Trimesh(v, raw["faces"][h], process=False)
            distances = (
                trimesh.proximity.closest_point(mesh, validation)[1]
                if len(validation)
                else np.array([])
            )
            row = dict(
                frame=int(i),
                hand=["left", "right"][h],
                scale=scales[h],
                reprojection_median_px=float(np.median(reprojection)),
                reprojection_p90_px=float(np.quantile(reprojection, 0.9)),
                validation_points=len(validation),
                validation_surface_median_m=float(np.median(distances))
                if len(distances)
                else None,
                validation_surface_p90_m=float(np.quantile(distances, 0.9))
                if len(distances)
                else None,
            )
            failed = []
            for name in (
                "reprojection_median_px",
                "reprojection_p90_px",
                "validation_surface_median_m",
                "validation_surface_p90_m",
            ):
                if row[name] is None or row[name] > limits["max_" + name]:
                    failed.append(name)
            if len(validation) < limits["min_validation_points"]:
                failed.append("validation_coverage")
            wrist = transform(j[:1], upright_from_camera[i])[0]
            if previous[h] is not None:
                old_i, old_wrist = previous[h]
                speed = float(
                    np.linalg.norm(wrist - old_wrist)
                    / (source.times[i] - source.times[old_i])
                )
                row["wrist_speed_m_s"] = speed
                if speed > limits["max_wrist_speed_m_s"]:
                    failed.append("wrist_speed")
            row.update(accepted=not failed, rejected_for=failed)
            records[i][h] = row
            if not failed:
                accepted[i, h] = True
                camera_vertices[i, h], camera_joints[i, h] = v, j
                previous[h] = (i, wrist)
        except ValueError as error:
            records[i][h]["reason"] = str(error)
        if i % 20 == 0:
            print(f"metric hands {i + 1}/{n}, hand {h}", flush=True)
    world_vertices, world_joints = camera_vertices.copy(), camera_joints.copy()
    for i, h in np.argwhere(accepted):
        world_vertices[i, h] = transform(camera_vertices[i, h], upright_from_camera[i])
        world_joints[i, h] = transform(camera_joints[i, h], upright_from_camera[i])
    np.savez_compressed(
        out / "hands.npz",
        world_vertices=world_vertices,
        world_joints=world_joints,
        camera_vertices=camera_vertices,
        camera_joints=camera_joints,
        proposed_camera_vertices=proposed_vertices,
        proposed_camera_joints=proposed_joints,
        fits=fits,
        scale_per_hand=scales,
        faces=raw["faces"],
        accepted=accepted,
        frame_ids=source.frame_ids,
        timestamps_s=source.times,
    )
    write_json(
        out / "quality.json",
        dict(
            schema="real2sim2real.metric-hands/1",
            units="metres",
            hand_order=["left", "right"],
            scale_per_hand=scales,
            scale_source="calibrated_stereo_surface_fit",
            total_frames=n,
            accepted_per_hand=accepted.sum(0).tolist(),
            complete=bool(accepted.all()),
            limits=limits,
            records=records,
            raw_sha256=digest(root / "hands_raw/predictions.npz"),
            support_sha256=digest(root / "support/support.json"),
            contact_used_for_fitting=False,
            interpolation_used=False,
            ground_truth_used=False,
        ),
    )
