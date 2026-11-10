"""Fit bounded finger articulation only in rejected baseline intervals.

No evaluation annotations or held-out detector frames are read here. A new
rigid initialization uses tracked points and fitting depth only; the previous
common fitter's detector-informed parameters are deliberately not reused.
"""

import json
from pathlib import Path

import numpy as np
import trimesh

from real2sim2real.artifacts import digest, write_json
from .camera import transform
from .contact_diagnostics import support_clearance, surface_contact, ray_intersector
from .hand_observation_fit import (
    measured_surface,
    fit_prior,
    assess_geometry,
    geometry_rejections,
)
from .hand_refinement import admit_temporal
from .interaction_io import Reconstruction, fresh_directory
from .mesh_surface import MeshSurfaceQuery
from .metric_hands import apply_fit


def run(request, check=False):
    from .mano_refinement import ManoHand, refine, damped_parameters

    model_path = Path(request["weights"]["hamer"]) / "_DATA/data/mano/MANO_RIGHT.pkl"
    model = ManoHand(model_path)
    if check:
        ray_intersector(
            trimesh.creation.box(),
            request.get("contacts", {}).get("ray_backend", "auto"),
        )
        return
    inputs = Path(request["fit_inputs"])
    for name, sha in json.loads((inputs / "receipt.json").read_text()).items():
        if digest(inputs / name) != sha:
            raise ValueError("Fitting input changed: " + name)
    provenance = json.loads((inputs / "raw_provenance.json").read_text())
    if provenance.get("mano_sha256", digest(model_path)) != digest(model_path):
        raise ValueError("MANO model differs from retained HaWoR prior")
    source = Reconstruction(
        request["reconstruction"], request["frame_count"], start=request["frame_start"]
    )
    for name, sha in json.loads((inputs / "inputs.json").read_text())["files"].items():
        if digest(source.path(name)) != sha:
            raise ValueError("Reconstruction observation changed: " + name)
    baseline = dict(np.load(inputs / "baseline.npz"))
    raw = dict(np.load(inputs / "raw.npz"))
    crops = dict(np.load(inputs / "crops.npz"))
    observations = dict(np.load(Path(request["point_tracks"]) / "observations.npz"))
    for data in (baseline, raw, crops, observations):
        if not np.array_equal(data["frame_ids"], source.frame_ids) or not np.allclose(
            data["timestamps_s"], source.times, atol=1e-8, rtol=0
        ):
            raise ValueError("Point refinement requires matching frame IDs and times")
    support = json.loads((inputs / "support.json").read_text())
    if not support["accepted"]:
        raise ValueError("An accepted upright support is required")
    upright = np.asarray(support["upright_from_world"])
    transforms = upright @ source.world_from_camera
    mesh = trimesh.load(source.path("mesh/metric.glb"), force="mesh")
    proximity = MeshSurfaceQuery(mesh)
    rays = ray_intersector(mesh, request.get("contacts", {}).get("ray_backend", "auto"))
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
    out = fresh_directory(Path(request["output"]) / "hands_metric")
    vertices, joints = (
        baseline["camera_vertices"].copy(),
        baseline["camera_joints"].copy(),
    )
    world_v, world_j = (
        baseline["world_vertices"].copy(),
        baseline["world_joints"].copy(),
    )
    prior_v, prior_j = vertices.copy(), joints.copy()
    fitted = np.zeros_like(baseline["accepted"])
    proposed = fitted.copy()
    local_delta = np.zeros((source.count, 2, 15, 3))
    fits = baseline["fits"].copy()
    records = []
    for i, frame in enumerate(source.frame_ids):
        masks = [source.mask(i, h) for h in request["mask_ids"]]
        obj = source.mask(i, request["object_id"])
        row = []
        for h in range(2):
            record = dict(
                frame_id=int(frame),
                hand=["left", "right"][h],
                baseline_preserved=bool(baseline["accepted"][i, h]),
                geometry_supported=bool(baseline["accepted"][i, h]),
                rejected_for=[],
            )
            diagnostic = baseline["accepted"][i, h] and frame in request.get(
                "diagnostic_frame_ids", []
            )
            record["diagnostic_only"] = bool(diagnostic)
            if baseline["accepted"][i, h] and not diagnostic:
                row.append(record)
                continue
            record["geometry_supported"] = False
            try:
                uv, weights = (
                    observations["keypoints_px"][i, h],
                    observations["weights"][i, h],
                )
                record["visible_tracked_points"] = int((weights > 0).sum())
                if not raw["accepted"][i, h] or not crops["crop_valid"][i, h]:
                    raise ValueError("missing_image_prior")
                if (weights > 0).sum() < 4:
                    raise ValueError("insufficient_visible_tracked_points")
                points, validation = measured_surface(
                    source.depth(i), masks[h], crops["boxes_xyxy"][i, h], source.k
                )
                p, _ = fit_prior(
                    raw["vertices"][i, h],
                    raw["faces"][h],
                    raw["joints"][i, h],
                    uv,
                    points,
                    masks[h],
                    obj | masks[1 - h],
                    source.k,
                    baseline["scale_per_hand"][h],
                )
                prior_v[i, h], prior_j[i, h] = (
                    apply_fit(raw["vertices"][i, h], p),
                    apply_fit(raw["joints"][i, h], p),
                )
                prior_metrics = assess_geometry(
                    prior_v[i, h],
                    raw["faces"][h],
                    prior_j[i, h],
                    uv,
                    validation,
                    masks[h],
                    masks[1 - h],
                    obj,
                    crops["boxes_xyxy"][i, h],
                    source.k,
                )
                record["before_refinement"] = prior_metrics
                prior = dict(
                    vertices=raw["vertices"][i, h],
                    joints=raw["joints"][i, h],
                    betas=raw["fixed_betas"][h],
                    pose=raw["hand_pose"][i, h],
                    orient=raw["global_orient"][i, h],
                    faces=raw["faces"][h],
                )
                v, j, details = refine(
                    model,
                    prior,
                    p,
                    uv,
                    points,
                    source.k,
                    h,
                    options=request.get("optimizer", {}),
                    joint_weights=weights,
                )
                fitted[i, h] = True
                vertices[i, h], joints[i, h] = v, j
                fits[i, h], local_delta[i, h] = damped_parameters(
                    p, np.zeros((15, 3)), details, 1.0
                )
                world_v[i, h], world_j[i, h] = (
                    transform(v, transforms[i]),
                    transform(j, transforms[i]),
                )
                metrics = assess_geometry(
                    v,
                    raw["faces"][h],
                    j,
                    uv,
                    validation,
                    masks[h],
                    masks[1 - h],
                    obj,
                    crops["boxes_xyxy"][i, h],
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
                    intersector=rays,
                )
                for name, value in [("table", table), ("object", contact)]:
                    if (
                        value["penetration_depth_m"] is not None
                        and value["penetration_depth_m"] > limits["max_penetration_m"]
                    ):
                        failed.append(name + "_penetration")
                proposed[i, h] = not failed
                record.update(
                    metrics,
                    refinement=details,
                    geometry_supported=not failed,
                    rejected_for=failed,
                    table=table,
                    object=contact,
                )
            except ValueError as error:
                record["rejected_for"] = [str(error)]
            row.append(record)
        records.append(row)
        print(
            f"Point refinement frame {frame}: fitted {fitted[i].tolist()}, proposed {proposed[i].tolist()}",
            flush=True,
        )
    diagnostic_v, diagnostic_j = vertices.copy(), joints.copy()
    diagnostic_supported = proposed.copy()
    # Diagnostic fits at held-out frames never alter accepted baseline anchors.
    keep = baseline["accepted"]
    vertices[keep], joints[keep] = (
        baseline["camera_vertices"][keep],
        baseline["camera_joints"][keep],
    )
    world_v[keep], world_j[keep] = (
        baseline["world_vertices"][keep],
        baseline["world_joints"][keep],
    )
    fits[keep], local_delta[keep] = baseline["fits"][keep], 0
    proposed &= ~keep
    accepted = admit_temporal(
        baseline["accepted"],
        proposed,
        world_j[:, :, 0],
        source.times,
        max_speed=limits["max_wrist_speed_m_s"],
    )
    for i, h in np.argwhere(proposed & ~accepted):
        records[i][h]["rejected_for"].append("baseline_transition_wrist_speed")
        records[i][h]["geometry_supported"] = False
    # This combined candidate is diagnostic; it does not overwrite the baseline.
    np.savez_compressed(
        out / "hypotheses.npz",
        camera_vertices=vertices,
        camera_joints=joints,
        world_vertices=world_v,
        world_joints=world_j,
        prior_camera_vertices=prior_v,
        prior_camera_joints=prior_j,
        fits=fits,
        local_delta_rad=local_delta,
        diagnostic_camera_vertices=diagnostic_v,
        diagnostic_camera_joints=diagnostic_j,
        diagnostic_geometry_supported=diagnostic_supported,
        geometry_supported=accepted,
        baseline_preserved=baseline["accepted"],
        refinement_attempted=fitted,
        faces=raw["faces"],
        frame_ids=source.frame_ids,
        timestamps_s=source.times,
    )
    write_json(
        out / "quality.json",
        dict(
            schema="real2sim2real.point-hand-refinement/1",
            units="metres",
            world_frame="fixed_upright_reconstruction",
            records=records,
            limits=limits,
            geometry_supported_per_hand=accepted.sum(0).tolist(),
            recovered_per_hand=(accepted & ~baseline["accepted"]).sum(0).tolist(),
            refined_per_hand=fitted.sum(0).tolist(),
            promoted=False,
            control_ready=False,
            evaluation_labels_used=False,
            heldout_depth_used_for_fitting=False,
            contact_used_for_fitting=False,
            baseline_sha256=digest(inputs / "baseline.npz"),
            observations_sha256=digest(
                Path(request["point_tracks"]) / "observations.npz"
            ),
            mano_sha256=digest(model_path),
            legacy_prior_has_mano_digest="mano_sha256" in provenance,
            mano_geometry_reconstruction_checked=True,
            support_is_not_joint_ground_truth=True,
        ),
    )
