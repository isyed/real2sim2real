"""Bounded CPU sequence candidates; complete components are accepted or rejected."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from real2sim2real.artifacts import digest, write_json
from .camera import transform
from .contact_diagnostics import support_clearance, surface_contact, ray_intersector
from .hand_observation_fit import measured_surface, assess_geometry, geometry_rejections
from .hand_sequence import (
    sequence_options,
    select_components,
    compare_motion,
    motion_rejections,
)
from .interaction_io import Reconstruction, fresh_directory
from .mesh_surface import MeshSurfaceQuery
from .hand_observation_policy import observation_reliability, policy_options


def run(request, check=False):
    from .mano_refinement import ManoHand
    from .mano_sequence import fit_sequence

    options = sequence_options(request.get("sequence_optimizer", {}))
    policy_config = request.get("observation_policy")
    if policy_config is not None:
        policy_options(policy_config)
    model_path = Path(request["weights"]["hamer"]) / "_DATA/data/mano/MANO_RIGHT.pkl"
    model = ManoHand(model_path)
    if check:
        ray_intersector(
            trimesh.creation.box(),
            request.get("contacts", {}).get("ray_backend", "auto"),
        )
        return
    inputs, parent_dir = Path(request["fit_inputs"]), Path(request["sequence_parent"])
    source = Reconstruction(
        request["reconstruction"], request["frame_count"], start=request["frame_start"]
    )
    for name, sha in json.loads((inputs / "receipt.json").read_text()).items():
        if digest(inputs / name) != sha:
            raise ValueError("Sequence fitting input changed: " + name)
    for name, sha in json.loads((inputs / "inputs.json").read_text())["files"].items():
        if digest(source.path(name)) != sha:
            raise ValueError("Sequence reconstruction changed: " + name)
    parent = dict(np.load(parent_dir / "hypotheses.npz"))
    baseline = dict(np.load(inputs / "baseline.npz"))
    raw = dict(np.load(inputs / "raw.npz"))
    anchors = dict(np.load(request["anchor_priors"]))
    observations = dict(np.load(Path(request["point_tracks"]) / "observations.npz"))
    crops = dict(np.load(inputs / "crops.npz"))
    quality = json.loads((parent_dir / "quality.json").read_text())
    if quality.get("sequence_fitting"):
        raise ValueError(
            "Sequence candidates must start from the original pointwise parent"
        )
    if (
        digest(model_path) != quality["mano_sha256"]
        or digest(Path(request["point_tracks"]) / "observations.npz")
        != quality["observations_sha256"]
    ):
        raise ValueError("Sequence model/observations differ from the retained parent")
    for data in (parent, baseline, raw, anchors, observations, crops):
        if not np.array_equal(data["frame_ids"], source.frame_ids) or not np.allclose(
            data["timestamps_s"], source.times, atol=1e-8, rtol=0
        ):
            raise ValueError("Sequence inputs require the identical capture window")
    support = json.loads((inputs / "support.json").read_text())
    if not support["accepted"]:
        raise ValueError("Accepted support required")
    upright = np.asarray(support["upright_from_world"])
    transforms = upright @ source.world_from_camera
    mesh = trimesh.load(source.path("mesh/metric.glb"), force="mesh")
    proximity = MeshSurfaceQuery(mesh)
    rays = ray_intersector(mesh, request.get("contacts", {}).get("ray_backend", "auto"))
    selected, components = select_components(
        parent["geometry_supported"] & ~baseline["accepted"],
        parent["geometry_supported"],
        source.times,
        context_frames=options["context_frames"],
        max_step_s=options["max_step_s"],
    )
    reliability = None
    if policy_config is not None:
        reliability = observation_reliability(
            observations["keypoints_px"],
            observations["weights"],
            parent["camera_joints"],
            source.k,
            baseline["accepted"],
            options=policy_config,
        )
    out = fresh_directory(Path(request["output"]) / "hands_metric")
    result = {k: v.copy() for k, v in parent.items()}
    # Export complete MANO parameters, including the mixed HaMeR/HaWoR origins.
    # A stale pointwise delta plus a new rigid fit would not reproduce the mesh.
    complete_pose = np.full_like(raw["hand_pose"], np.nan)
    complete_orient = np.full_like(raw["global_orient"], np.nan)
    complete_betas = np.full_like(raw["betas"], np.nan)
    for i, h in np.argwhere(parent["geometry_supported"]):
        origin = anchors if baseline["accepted"][i, h] else raw
        delta = (
            baseline["articulation_delta_rad"][i, h]
            if baseline["accepted"][i, h]
            else parent["local_delta_rad"][i, h]
        )
        complete_pose[i, h] = (
            Rotation.from_rotvec(delta).as_matrix() @ origin["hand_pose"][i, h]
        )
        complete_orient[i, h] = origin["global_orient"][i, h]
        complete_betas[i, h] = origin["fixed_betas"][h]
    proposed_v, proposed_j = (
        parent["camera_vertices"].copy(),
        parent["camera_joints"].copy(),
    )
    proposed_wv, proposed_wj = (
        parent["world_vertices"].copy(),
        parent["world_joints"].copy(),
    )
    proposed_supported = parent["geometry_supported"].copy()
    admitted = np.zeros_like(selected)
    records, component_records = deepcopy(quality["records"]), []
    limits = quality["limits"]

    def assess(i, h, v, j, uv, validation, masks, obj):
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
        world_v = transform(v, transforms[i])
        table = support_clearance(world_v, support["observed_boundary_xy"])
        contact = surface_contact(
            mesh,
            transform(world_v, np.linalg.inv(upright @ source.world_from_object[i])),
            proximity=proximity,
            intersector=rays,
        )
        failed = geometry_rejections(metrics, limits)
        for name, value in [("table", table), ("object", contact)]:
            if (
                value["penetration_depth_m"] is not None
                and value["penetration_depth_m"] > limits["max_penetration_m"]
            ):
                failed.append(name + "_penetration")
        return dict(metrics, table=table, object=contact), failed

    for component in components:
        h, ids, window = component["hand"], component["selected"], component["window"]
        row = dict(
            hand=int(h),
            frame_ids=source.frame_ids[ids].tolist(),
            anchor_frame_ids=source.frame_ids[np.setdiff1d(window, ids)].tolist(),
            accepted=False,
        )
        if reliability is not None:
            row["weak_baseline_frame_ids"] = source.frame_ids[
                ids[
                    baseline["accepted"][ids, h]
                    & ~reliability["reliable_baseline"][ids, h]
                ]
            ].tolist()
            fixed = np.setdiff1d(window, ids)
            row["weak_fixed_anchor_frame_ids"] = source.frame_ids[
                fixed[
                    baseline["accepted"][fixed, h]
                    & ~reliability["reliable_baseline"][fixed, h]
                ]
            ].tolist()
        try:
            pose, orient, betas, fit_points, heldout_points, masks_all, objects = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for i in ids:
                anchor = bool(baseline["accepted"][i, h])
                origin = anchors if anchor else raw
                delta = (
                    baseline["articulation_delta_rad"][i, h]
                    if anchor
                    else parent["local_delta_rad"][i, h]
                )
                pose.append(
                    Rotation.from_rotvec(delta).as_matrix() @ origin["hand_pose"][i, h]
                )
                orient.append(origin["global_orient"][i, h])
                betas.append(origin["fixed_betas"][h])
                masks = [source.mask(i, x) for x in request["mask_ids"]]
                points, validation = measured_surface(
                    source.depth(i), masks[h], crops["boxes_xyxy"][i, h], source.k
                )
                fit_points.append(points)
                heldout_points.append(validation)
                masks_all.append(masks)
                objects.append(source.mask(i, request["object_id"]))
            prior = dict(
                pose=np.asarray(pose),
                orient=np.asarray(orient),
                betas=np.asarray(betas),
                camera_vertices=parent["camera_vertices"][ids, h],
                camera_joints=parent["camera_joints"][ids, h],
                faces=raw["faces"][h],
            )
            policy_kwargs = {}
            image_weights = observations["weights"][ids, h]
            if reliability is not None:
                policy_kwargs["normalization_weights"] = image_weights
                image_weights = reliability["weights"][ids, h]
                if reliability["options"]["reassess_anchors"]:
                    policy_kwargs.update(
                        reliable_anchors=reliability["reliable_baseline"][window, h],
                        weak_anchor_prior_weight=reliability["options"][
                            "weak_anchor_prior_weight"
                        ],
                    )
            fit = fit_sequence(
                model,
                prior,
                parent["fits"][ids, h],
                observations["keypoints_px"][ids, h],
                image_weights,
                fit_points,
                source.k,
                h,
                transforms[window],
                parent["world_joints"][window, h],
                selected[window, h],
                source.times[window],
                options=options,
                baseline_mask=baseline["accepted"][window, h],
                **policy_kwargs,
            )
            proposed_v[ids, h], proposed_j[ids, h] = (
                fit["camera_vertices"],
                fit["camera_joints"],
            )
            proposed_wv[ids, h] = np.asarray(
                [
                    transform(v, transforms[i])
                    for v, i in zip(fit["camera_vertices"], ids)
                ]
            )
            proposed_wj[window, h] = fit["world_joints"]
            # Use the caller's exact immutable anchor values, not float32 copies.
            fixed = np.setdiff1d(window, ids)
            proposed_wj[fixed, h] = parent["world_joints"][fixed, h]
            failures = []
            for local, i in enumerate(ids):
                uv = observations["keypoints_px"][i, h]
                before, _ = assess(
                    i,
                    h,
                    parent["camera_vertices"][i, h],
                    parent["camera_joints"][i, h],
                    uv,
                    heldout_points[local],
                    masks_all[local],
                    objects[local],
                )
                after, rejected = assess(
                    i,
                    h,
                    fit["camera_vertices"][local],
                    fit["camera_joints"][local],
                    uv,
                    heldout_points[local],
                    masks_all[local],
                    objects[local],
                )
                # Predeclared tolerances; held-out evidence only admits/rejects a
                # completed component and is never fed back to optimization.
                for key, tolerance in [
                    ("validation_surface_p90_m", 0.003),
                    ("reprojection_median_px", 2),
                    ("reprojection_p90_px", 5),
                ]:
                    if (
                        before[key] is not None
                        and after[key] is not None
                        and after[key] > before[key] + tolerance
                    ):
                        rejected.append(key + "_regression")
                proposed_supported[i, h] = not rejected
                records[i][h] = dict(
                    after,
                    frame_id=int(source.frame_ids[i]),
                    hand=["left", "right"][h],
                    baseline_preserved=bool(baseline["accepted"][i, h]),
                    geometry_supported=not rejected,
                    rejected_for=rejected,
                    before_sequence=before,
                    sequence_proposed=True,
                )
                if rejected:
                    failures.append(
                        dict(frame_id=int(source.frame_ids[i]), reasons=rejected)
                    )
            motion = compare_motion(
                parent["world_joints"][window, h],
                proposed_wj[window, h],
                selected[window, h],
                source.times[window],
            )
            failed_motion = motion_rejections(motion)
            okay = not failures and not failed_motion
            row.update(
                accepted=okay,
                motion=motion,
                geometry_failures=failures,
                temporal_failures=failed_motion,
                objective=fit["objective"],
                max_joint_delta_deg=float(
                    np.rad2deg(np.linalg.norm(fit["local_delta_rad"], axis=-1)).max()
                ),
                max_translation_delta_m=float(
                    np.linalg.norm(fit["translation_delta_m"], axis=-1).max()
                ),
            )
            if okay:
                admitted[ids, h] = True
                for key, value in [
                    ("camera_vertices", proposed_v),
                    ("camera_joints", proposed_j),
                    ("world_vertices", proposed_wv),
                    ("world_joints", proposed_wj),
                ]:
                    result[key][ids, h] = value[ids, h]
                result["fits"][ids, h] = fit["fits"]
                complete_pose[ids, h] = fit["pose"]
        except ValueError as error:
            row["error"] = str(error)
            proposed_supported[ids, h] = False
        component_records.append(row)
        print(
            f"Sequence component hand {h}, frames {row['frame_ids']}: accepted={row['accepted']}",
            flush=True,
        )
    # Preserve both unaccepted hypotheses and the atomic fallback result.
    import torch

    with torch.no_grad():
        tensor = lambda x: torch.as_tensor(np.asarray(x).copy(), dtype=torch.float32)
        for h in range(2):
            ids = np.flatnonzero(parent["geometry_supported"][:, h])
            if not len(ids):
                continue
            v, j = model.batch(
                tensor(complete_betas[ids, h]),
                tensor(complete_pose[ids, h]),
                tensor(complete_orient[ids, h]),
                h,
            )
            fits = result["fits"][ids, h]
            rotation = Rotation.from_rotvec(fits[:, :3]).as_matrix()
            for actual, key in [
                (v.numpy(), "camera_vertices"),
                (j.numpy(), "camera_joints"),
            ]:
                actual = (
                    actual
                    * np.exp(fits[:, 6])[:, None, None]
                    @ rotation.transpose(0, 2, 1)
                    + fits[:, None, 3:6]
                )
                if not np.allclose(actual, result[key][ids, h], rtol=0, atol=2e-5):
                    raise ValueError(
                        "Exported sequence MANO parameters do not reproduce the retained result"
                    )
    for i, h in np.argwhere(selected):
        records[i][h]["baseline_was_accepted"] = bool(baseline["accepted"][i, h])
        records[i][h]["baseline_preserved"] = bool(
            baseline["accepted"][i, h] and not admitted[i, h]
        )
    result.update(
        hand_pose=complete_pose,
        global_orient=complete_orient,
        betas=complete_betas,
        parent_local_delta_rad=result.pop("local_delta_rad"),
        sequence_selected=selected,
        sequence_admitted=admitted,
        diagnostic_camera_vertices=proposed_v,
        diagnostic_camera_joints=proposed_j,
        diagnostic_geometry_supported=proposed_supported,
        proposed_world_vertices=proposed_wv,
        proposed_world_joints=proposed_wj,
    )
    if reliability is not None:
        result.update(
            original_observation_weights=observations["weights"],
            effective_observation_weights=reliability["weights"],
            observation_identity_ambiguous=reliability["ambiguous"],
            baseline_anchor_reliable=reliability["reliable_baseline"],
        )
    np.savez_compressed(out / "hypotheses.npz", **result)
    quality.update(
        schema="real2sim2real.point-hand-sequence/1",
        sequence_fitting=True,
        baseline_anchors_may_encode_detector_evaluation_labels=True,
        heldout_detector_labels_loaded=False,
        manual_annotations_loaded=False,
        exported_mano_geometry_checked=True,
        sequence_optimizer=options,
        observation_policy=None if reliability is None else reliability["options"],
        observation_reliability_records=None
        if reliability is None
        else reliability["records"],
        validation_uses_original_observations=True,
        components=component_records,
        records=records,
        sequence_admitted_per_hand=admitted.sum(0).tolist(),
        sequence_selected_per_hand=selected.sum(0).tolist(),
        accepted_baseline_changes_per_hand=(admitted & baseline["accepted"])
        .sum(0)
        .tolist(),
        parent_sha256=digest(parent_dir / "hypotheses.npz"),
        anchor_priors_sha256=digest(request["anchor_priors"]),
        promoted=False,
        control_ready=False,
        units="metres",
    )
    write_json(out / "quality.json", quality)
