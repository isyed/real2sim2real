"""Common fixed-scale fitting for priors with partial or missing 2D landmarks.

Only calibrated fit pixels and visible automatic landmarks drive optimisation.
Silhouettes also constrain fitting. Held-out depth and contacts assess hypotheses;
manual annotations never enter this module. Geometry support is not joint truth.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, write_json
from .camera import transform, unproject
from .contact_diagnostics import support_clearance, surface_contact, ray_intersector
from .hand_tracks import load_tracks
from .interaction_io import Reconstruction, fresh_directory
from .metric_hands import project, apply_fit


def crop_region(shape, box):
    region = np.zeros(shape, bool)
    if not np.isfinite(box).all():
        return region
    x0, y0, x1, y1 = np.rint(box).astype(int)
    x0, x1 = np.clip([x0, x1], 0, shape[1])
    y0, y1 = np.clip([y0, y1], 0, shape[0])
    region[y0:y1, x0:x1] = True
    return region


def measured_surface(depth, mask, box, k, *, max_points=600, support_mask=None):
    skin = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    y, x = np.indices(depth.shape)
    region = skin & crop_region(depth.shape, box)
    if support_mask is not None:
        support_mask = np.asarray(support_mask)
        if support_mask.shape != depth.shape or support_mask.dtype != bool:
            raise ValueError("Boolean hand depth support must match the image")
        region &= support_mask
    parity = (x // 3 + y // 3) % 2 == 0
    valid = region & np.isfinite(depth) & (depth > 0.15) & (depth < 1.5)
    xyz = unproject(depth, k)
    fit = xyz[valid & parity]
    validation = xyz[valid & ~parity]
    if len(fit):
        # Validation values cannot affect initialisation or fit-point selection.
        center = np.median(fit[:, 2])
        fit = fit[np.abs(fit[:, 2] - center) < 0.12]
        validation = validation[np.abs(validation[:, 2] - center) < 0.12]
    rng = np.random.default_rng(0)
    if len(fit) > max_points:
        fit = fit[rng.choice(len(fit), max_points, replace=False)]
    # A separate RNG ensures validation size cannot affect fit sampling.
    rng = np.random.default_rng(1)
    if len(validation) > max_points:
        validation = validation[rng.choice(len(validation), max_points, replace=False)]
    return fit, validation


def visible_landmarks(points, mask):
    points = np.asarray(points)
    finite = np.isfinite(points).all(1)
    indices = np.rint(np.nan_to_num(points)).astype(int)
    inside = (
        finite
        & (indices[:, 0] >= 0)
        & (indices[:, 0] < mask.shape[1])
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < mask.shape[0])
    )
    expanded = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    result = np.zeros(len(points), bool)
    result[inside] = expanded[indices[inside, 1], indices[inside, 0]]
    return result


def fit_prior(vertices, faces, joints, landmarks, points, mask, occluder, k, scale):
    import trimesh

    if len(points) < 80:
        raise ValueError("insufficient_fit_depth")
    visible = visible_landmarks(landmarks, mask)
    if visible.any():
        z = np.median(points[:, 2])
        target = (np.linalg.inv(k) @ [*landmarks[visible].mean(0), 1]) * z
        translation = target - joints[visible].mean(0) * scale
    else:
        translation = points.mean(0) - vertices.mean(0) * scale
    x = np.r_[np.zeros(3), translation]
    angle = np.deg2rad(35)
    low, high = np.r_[[-angle] * 3, -2, -2, 0.1], np.r_[[angle] * 3, 2, 2, 1.6]
    x = np.clip(x, low + 1e-6, high - 1e-6)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    distance_image = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)

    def position(p, local):
        return local * scale @ Rotation.from_rotvec(p[:3]).as_matrix().T + p[3:]

    for _ in range(10):
        rotation = Rotation.from_rotvec(x[:3]).as_matrix()
        closest, distances, _ = trimesh.proximity.closest_point(
            mesh, (points - x[3:]) @ rotation / scale
        )
        keep = distances <= min(max(0.01, np.quantile(distances, 0.8)), 0.08 / scale)
        if keep.sum() < 50:
            raise ValueError("insufficient_surface_correspondences")
        local, target = closest[keep], points[keep]

        def residual(p):
            surface = ((position(p, local) - target) / 0.006).ravel() / np.sqrt(
                len(local)
            )
            image = (
                (project(position(p, joints[visible]), k) - landmarks[visible]) / 5
            ).ravel() / np.sqrt(max(1, visible.sum()))
            uv = project(position(p, vertices[::8]), k)
            coords = np.array([uv[:, 1], uv[:, 0]])
            blocked = map_coordinates(
                occluder.astype(float), coords, order=0, mode="constant", cval=0
            )
            silhouette = (
                map_coordinates(
                    distance_image, coords, order=1, mode="constant", cval=100
                )
                * (1 - blocked)
                / 8
            )
            return np.r_[
                surface,
                image,
                0.25 * silhouette / np.sqrt(len(uv)),
                0.15 * p[:3] / np.deg2rad(25),
            ]

        result = least_squares(
            residual, x, bounds=(low, high), loss="soft_l1", f_scale=0.5, max_nfev=50
        )
        change = np.linalg.norm(result.x - x)
        x = result.x
        if change < 1e-5:
            break
    return np.r_[x, np.log(scale)], visible


def projected_mask(vertices, faces, k, shape):
    result = np.zeros(shape, np.uint8)
    if not np.isfinite(vertices).all() or (vertices[:, 2] <= 0).any():
        return result.astype(bool)
    uv = np.rint(np.clip(project(vertices, k), -10000, 10000)).astype(np.int32)
    for triangle in uv[faces]:
        cv2.fillConvexPoly(result, triangle, 1)
    return result.astype(bool)


def assess_geometry(
    vertices,
    faces,
    joints,
    landmarks,
    validation,
    mask,
    other_mask,
    object_mask,
    box,
    k,
):
    import trimesh

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    distances = (
        trimesh.proximity.closest_point(mesh, validation)[1]
        if len(validation)
        else np.array([])
    )
    visible = visible_landmarks(landmarks, mask)
    errors = np.linalg.norm(project(joints[visible], k) - landmarks[visible], axis=1)
    rendered = projected_mask(vertices, faces, k, mask.shape)
    assessable = ~object_mask & ~other_mask
    intersection = rendered & mask & assessable
    union = (rendered | mask) & assessable & crop_region(mask.shape, box)
    visible_render = rendered & assessable
    ownership = [int((rendered & m & ~object_mask).sum()) for m in [mask, other_mask]]
    return dict(
        visible_landmarks=int(visible.sum()),
        reprojection_median_px=float(np.median(errors)) if len(errors) else None,
        reprojection_p90_px=float(np.quantile(errors, 0.9)) if len(errors) else None,
        validation_points=len(validation),
        validation_surface_median_m=float(np.median(distances))
        if len(distances)
        else None,
        validation_surface_p90_m=float(np.quantile(distances, 0.9))
        if len(distances)
        else None,
        silhouette_precision=float(intersection.sum() / max(1, visible_render.sum())),
        silhouette_iou=float(
            (intersection & crop_region(mask.shape, box)).sum() / max(1, union.sum())
        ),
        hand_identity_purity=float(ownership[0] / max(1, sum(ownership))),
        landmark_validation_available=len(errors) >= 4,
    )


def geometry_rejections(row, limits):
    failed = []
    for name in ["validation_surface_median_m", "validation_surface_p90_m"]:
        if (
            row[name] is None
            or not np.isfinite(row[name])
            or row[name] > limits["max_" + name]
        ):
            failed.append(name)
    if row["validation_points"] < limits["min_validation_points"]:
        failed.append("validation_coverage")
    if row["visible_landmarks"] >= 4:
        for name in ["reprojection_median_px", "reprojection_p90_px"]:
            if row[name] is None or row[name] > limits["max_" + name]:
                failed.append(name)
    for name in ["silhouette_precision", "silhouette_iou", "hand_identity_purity"]:
        if not np.isfinite(row[name]) or row[name] < limits["min_" + name]:
            failed.append(name)
    return failed


def fit_sequence(config, variant):
    import trimesh
    from .mesh_surface import MeshSurfaceQuery

    root = Path(config["output"])
    source = Reconstruction(
        config["reconstruction"], config["frame_count"], start=config["frame_start"]
    )
    raw_path = root / variant / "hands_raw/predictions.npz"
    raw = np.load(raw_path)
    if not np.array_equal(raw["frame_ids"], source.frame_ids) or not np.allclose(
        raw["timestamps_s"], source.times, rtol=0, atol=1e-8
    ):
        raise ValueError("Raw predictions must match the reconstruction window")
    tracks = load_tracks(root / "inputs/hand_tracks/tracks.npz", source)
    observations = np.load(root / "inputs/landmarks.npz")
    support = json.loads((root / "inputs/support.json").read_text())
    if not support["accepted"]:
        raise ValueError("Support must be accepted")
    scales = json.loads((root / "inputs/quality.json").read_text())["scale_per_hand"]
    out = fresh_directory(root / variant / "hands_metric")
    upright = np.asarray(support["upright_from_world"])
    transforms = upright @ source.world_from_camera
    mesh = trimesh.load(source.path("mesh/metric.glb"), force="mesh")
    proximity = MeshSurfaceQuery(mesh)
    intersector = ray_intersector(
        mesh, config.get("contacts", {}).get("ray_backend", "auto")
    )
    accepted = np.zeros((source.count, 2), bool)
    v = np.full_like(raw["vertices"], np.nan)
    j = np.full_like(raw["joints"], np.nan)
    world_v, world_j = v.copy(), j.copy()
    parameters = np.full((source.count, 2, 7), np.nan)
    limits = dict(
        max_validation_surface_median_m=0.012,
        max_validation_surface_p90_m=0.03,
        min_validation_points=80,
        max_reprojection_median_px=12,
        max_reprojection_p90_px=25,
        min_silhouette_precision=0.7,
        min_silhouette_iou=0.35,
        min_hand_identity_purity=0.7,
        max_penetration_m=0.003,
        max_wrist_speed_m_s=3.0,
    )
    limits.update(config.get("fit_limits", {}))
    records = []
    for i in range(source.count):
        masks = [source.mask(i, h) for h in config.get("mask_ids", [2, 3])]
        obj = source.mask(i, config.get("object_id", 1))
        row = []
        for h in range(2):
            record = dict(
                frame_id=int(source.frame_ids[i]),
                hand=["left", "right"][h],
                geometry_supported=False,
            )
            try:
                if not raw["accepted"][i, h] or not tracks["crop_valid"][i, h]:
                    raise ValueError("missing_image_inference")
                points, validation = measured_surface(
                    source.depth(i), masks[h], tracks["boxes_xyxy"][i, h], source.k
                )
                landmarks = observations["keypoints_px"][i, h].copy()
                if not observations["accepted"][i, h]:
                    landmarks[:] = np.nan
                p, _ = fit_prior(
                    raw["vertices"][i, h],
                    raw["faces"][h],
                    raw["joints"][i, h],
                    landmarks,
                    points,
                    masks[h],
                    obj | masks[1 - h],
                    source.k,
                    scales[h],
                )
                parameters[i, h] = p
                v[i, h], j[i, h] = (
                    apply_fit(raw["vertices"][i, h], p),
                    apply_fit(raw["joints"][i, h], p),
                )
                world_v[i, h], world_j[i, h] = (
                    transform(v[i, h], transforms[i]),
                    transform(j[i, h], transforms[i]),
                )
                metrics = assess_geometry(
                    v[i, h],
                    raw["faces"][h],
                    j[i, h],
                    landmarks,
                    validation,
                    masks[h],
                    masks[1 - h],
                    obj,
                    tracks["boxes_xyxy"][i, h],
                    source.k,
                )
                failed = geometry_rejections(metrics, limits)
                table = support_clearance(
                    world_v[i, h], support["observed_boundary_xy"]
                )
                contact = surface_contact(
                    mesh,
                    transform(
                        world_v[i, h],
                        np.linalg.inv(upright @ source.world_from_object[i]),
                    ),
                    proximity=proximity,
                    intersector=intersector,
                )
                for name, info in [("table", table), ("object", contact)]:
                    if (
                        info["penetration_depth_m"] is not None
                        and info["penetration_depth_m"] > limits["max_penetration_m"]
                    ):
                        failed.append(name + "_penetration")
                if i and np.isfinite(world_j[i - 1, h]).all():
                    speed = float(
                        np.linalg.norm(world_j[i, h, 0] - world_j[i - 1, h, 0])
                        / (source.times[i] - source.times[i - 1])
                    )
                    metrics["wrist_speed_m_s"] = speed
                    if speed > limits["max_wrist_speed_m_s"]:
                        failed.append("wrist_speed")
                accepted[i, h] = not failed
                record.update(
                    metrics,
                    geometry_supported=not failed,
                    rejected_for=failed,
                    table=table,
                    object=contact,
                    image_evidence="automatic_landmarks"
                    if metrics["landmark_validation_available"]
                    else "mask_and_depth_only",
                )
            except ValueError as error:
                record["rejected_for"] = [str(error)]
            row.append(record)
        records.append(row)
        print(
            f"{variant} fitted source frame {source.frame_ids[i]}: {accepted[i].tolist()}",
            flush=True,
        )
    np.savez_compressed(
        out / "hypotheses.npz",
        camera_vertices=v,
        camera_joints=j,
        world_vertices=world_v,
        world_joints=world_j,
        geometry_supported=accepted,
        fits=parameters,
        faces=raw["faces"],
        frame_ids=source.frame_ids,
        timestamps_s=source.times,
    )
    write_json(
        out / "quality.json",
        dict(
            schema="real2sim2real.hand-observation-fit/1",
            variant=variant,
            units="metres",
            records=records,
            limits=limits,
            geometry_supported_per_hand=accepted.sum(0).tolist(),
            support_is_not_joint_ground_truth=True,
            promoted=False,
            control_ready=False,
            raw_sha256=digest(raw_path),
            tracks_sha256=digest(root / "inputs/hand_tracks/tracks.npz"),
            manual_landmarks_used_for_fitting=False,
            contact_used_for_fitting=False,
            silhouette_used_for_fitting=True,
            heldout_depth_used_for_fitting=False,
        ),
    )
