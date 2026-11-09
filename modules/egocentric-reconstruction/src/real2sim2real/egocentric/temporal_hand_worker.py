"""Conservative temporal refinement of newly recovered hands, on CPU."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, load, write_json
from .camera import transform
from .hand_refinement import refinement_rejections
from .hand_temporal import component_admission, neighbor_target, visibility_weights
from .interaction_io import Reconstruction, fresh_directory
from .metric_hands import hand_metrics, observed_hand_surface


def run(request, check=False):
    import torch
    import trimesh
    from .mano_refinement import ManoHand, refine, validate_options, damped_parameters
    from .contact_diagnostics import surface_contact, support_clearance
    from .mesh_surface import MeshSurfaceQuery

    options = request["hand_temporal"]
    if set(options) - {"parent", "optimizer", "temporal_weight", "max_step_s"}:
        raise ValueError("Unknown temporal hand option")
    optimizer_options = validate_options(
        options.get(
            "optimizer",
            dict(
                max_joint_correction_deg=8,
                max_rigid_correction_deg=3,
                max_translation_correction_m=0.005,
                correspondence_iterations=5,
            ),
        )
    )
    weight, max_step = (
        options.get("temporal_weight", 1.0),
        options.get("max_step_s", 0.1),
    )
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not np.isfinite(weight)
        or not 0 < weight <= 10
        or isinstance(max_step, bool)
        or not isinstance(max_step, (int, float))
        or not np.isfinite(max_step)
        or not 0 < max_step <= 0.5
    ):
        raise ValueError("Temporal fitting requires bounded weight and neighbor time")
    parent, root = Path(options["parent"]), Path(request["output"])
    model_path = Path(request["weights"]["hamer"]) / "_DATA/data/mano/MANO_RIGHT.pkl"
    model = ManoHand(model_path)
    if check:
        return
    load(parent, kinds={"metric-interaction-reconstruction"})
    source = Reconstruction(request["reconstruction"], request.get("frame_count"))
    quality = json.loads((parent / "hands_metric/quality.json").read_text())
    if not quality.get("articulation_refinement") or quality.get("temporal_refinement"):
        raise ValueError("Use an independently refined parent, not chained smoothing")
    for name, sha in json.loads((parent / "inputs.json").read_text())["files"].items():
        if digest(source.path(name)) != sha:
            raise ValueError(
                "Temporal fitting observations differ from parent: " + name
            )
    with np.load(parent / "hands_metric/hands.npz") as archive:
        baseline = {key: archive[key] for key in archive.files}
    raw = np.load(parent / "hands_raw/predictions.npz")
    observations = np.load(parent / "hands2d/landmarks.npz")
    for data in (baseline, raw, observations):
        if not np.array_equal(
            data["frame_ids"], source.frame_ids
        ) or not np.array_equal(data["timestamps_s"], source.times):
            raise ValueError("Temporal fitting requires the exact parent timeline")
    provenance = json.loads((parent / "hands_raw/provenance.json").read_text())
    if (
        provenance["mano_sha256"] != digest(model_path)
        or provenance["observations_sha256"] != digest(parent / "hands2d/landmarks.npz")
        or quality["raw_sha256"] != digest(parent / "hands_raw/predictions.npz")
    ):
        raise ValueError("Temporal model and observations must match parent provenance")
    for directory in ("support", "hands_raw"):
        shutil.copytree(parent / directory, root / directory)
    (root / "hands2d").mkdir()
    for name in ("landmarks.npz", "quality.json"):
        shutil.copyfile(parent / "hands2d" / name, root / "hands2d" / name)
    out = fresh_directory(root / "hands_metric")
    write_json(root / "baseline_hand_quality.json", quality)
    result = {key: value.copy() for key, value in baseline.items()}
    records = deepcopy(quality["records"])
    scene = np.load(root / "support/scene.npz")
    support = json.loads((root / "support/support.json").read_text())
    mesh = trimesh.load(source.path("mesh/metric.glb"), force="mesh")
    proximity = MeshSurfaceQuery(mesh)
    selected = baseline["refinement_accepted"] & baseline["accepted"]
    proposed = np.zeros_like(selected)
    attempts, visibility = [], np.full((*selected.shape, 21), np.nan, np.float32)
    mask_ids = request["hand_landmarks"].get("mask_ids", [2, 3])
    for i, h in np.argwhere(selected):
        started = time.monotonic()
        record = dict(
            frame=int(source.frame_ids[i]), index=int(i), hand=int(h), accepted=False
        )
        target_world = neighbor_target(
            baseline["world_joints"],
            baseline["accepted"],
            source.times,
            i,
            h,
            max_step_s=max_step,
        )
        if target_world is None:
            record["rejected_for"] = ["no_two_observed_neighbors"]
            attempts.append(record)
            continue
        depth, mask = source.depth(i), source.mask(i, mask_ids[h])
        uv = observations["keypoints_px"][i, h]
        weights, labels = visibility_weights(
            uv,
            baseline["camera_joints"][i, h],
            depth,
            mask,
            source.mask(i, request.get("object_id", 1)),
        )
        visibility[i, h] = weights
        record.update(visibility_weights=weights.tolist(), visibility_labels=labels)
        try:
            points, heldout = observed_hand_surface(depth, mask, uv, source.k)
            prior = dict(
                betas=raw["fixed_betas"][h],
                pose=raw["hand_pose"][i, h],
                orient=raw["global_orient"][i, h],
                vertices=raw["vertices"][i, h],
                joints=raw["joints"][i, h],
                faces=raw["faces"][h],
            )
            initial = baseline["articulation_delta_rad"][i, h]
            # Verify that saved cumulative corrections reconstruct the parent,
            # including reflection of physical left hands and calibrated scale.
            with torch.no_grad():
                tensor = lambda a: torch.as_tensor(
                    np.asarray(a).copy(), dtype=torch.float32
                )
                pose = tensor(Rotation.from_rotvec(initial).as_matrix()) @ tensor(
                    prior["pose"]
                )
                vv, jj = model(tensor(prior["betas"]), pose, tensor(prior["orient"]), h)
                fit = baseline["fits"][i, h]
                for actual, expected in (
                    (vv, baseline["camera_vertices"][i, h]),
                    (jj, baseline["camera_joints"][i, h]),
                ):
                    actual = (
                        actual.numpy()
                        * np.exp(fit[6])
                        @ Rotation.from_rotvec(fit[:3]).as_matrix().T
                        + fit[3:6]
                    )
                    if not np.allclose(actual, expected, rtol=0, atol=2e-5):
                        raise ValueError(
                            "Cumulative MANO parameters do not reconstruct parent"
                        )
            v, j, details = refine(
                model,
                prior,
                baseline["fits"][i, h],
                uv,
                points,
                source.k,
                h,
                options=optimizer_options,
                initial_local=initial,
                joint_weights=weights,
                temporal_target=transform(
                    target_world, np.linalg.inv(scene["world_from_camera"][i])
                ),
                temporal_weight=weight,
            )
            steps = []
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                fitted, cumulative = damped_parameters(
                    baseline["fits"][i, h], initial, details, fraction
                )
                if fraction != 1:
                    with torch.no_grad():
                        vv, jj = model(
                            tensor(prior["betas"]),
                            tensor(Rotation.from_rotvec(cumulative).as_matrix())
                            @ tensor(prior["pose"]),
                            tensor(prior["orient"]),
                            h,
                        )
                        v, j = [
                            x.numpy()
                            * np.exp(fitted[6])
                            @ Rotation.from_rotvec(fitted[:3]).as_matrix().T
                            + fitted[3:6]
                            for x in (vv, jj)
                        ]
                metrics = hand_metrics(v, raw["faces"][h], j, uv, heldout, source.k)
                failed = refinement_rejections(
                    quality["records"][i][h],
                    metrics,
                    quality["limits"],
                    require_image_improvement=False,
                )
                world_v, world_j = [
                    transform(x, scene["world_from_camera"][i]) for x in (v, j)
                ]
                contact = surface_contact(
                    mesh,
                    transform(world_v, np.linalg.inv(scene["world_from_object"][i])),
                    proximity=proximity,
                )
                table = support_clearance(world_v, support["observed_boundary_xy"])
                for label, measurement in (("object", contact), ("table", table)):
                    if (
                        measurement["penetration_depth_m"] is not None
                        and measurement["penetration_depth_m"] > 0.003
                    ):
                        failed.append(label + "_penetration")
                steps.append(
                    dict(fraction=fraction, metrics=metrics, rejected_for=failed)
                )
                if not failed:
                    break
            record.update(
                metrics=metrics,
                optimizer=details,
                rejected_for=failed,
                object_contact=contact,
                table_contact=table,
                backtracking=steps,
                selected_fraction=fraction,
            )
            if not failed:
                proposed[i, h] = True
                for key, value in (
                    ("camera_vertices", v),
                    ("camera_joints", j),
                    ("world_vertices", world_v),
                    ("world_joints", world_j),
                ):
                    result[key][i, h] = value
                (
                    result["proposed_camera_vertices"][i, h],
                    result["proposed_camera_joints"][i, h],
                ) = v, j
                result["articulation_delta_rad"][i, h] = cumulative
                result["fits"][i, h] = fitted
                records[i][h].update(metrics, temporal_refined=True)
        except ValueError as error:
            record["rejected_for"] = [str(error)]
        record["elapsed_s"] = time.monotonic() - started
        attempts.append(record)
        write_json(out / "temporal_attempts.json", attempts)
        print(f"temporal frame {i}, hand {h}: {record.get('rejected_for')}", flush=True)
    admitted, components = component_admission(
        baseline["world_joints"],
        result["world_joints"],
        baseline["accepted"],
        proposed,
        source.times,
        max_step_s=max_step,
    )
    for key in (
        "camera_vertices",
        "camera_joints",
        "world_vertices",
        "world_joints",
        "proposed_camera_vertices",
        "proposed_camera_joints",
        "articulation_delta_rad",
        "fits",
    ):
        result[key][~admitted] = baseline[key][~admitted]
    for row in attempts:
        i, h = row["index"], row["hand"]
        row["accepted"] = bool(admitted[i, h])
        if not admitted[i, h]:
            records[i][h] = quality["records"][i][h]
            if proposed[i, h]:
                row["rejected_for"] = ["temporal_component_gate"]
    np.savez_compressed(
        out / "hands.npz",
        **result,
        temporal_refinement_accepted=admitted,
        image_support_weights=visibility,
    )
    quality.update(
        records=records,
        temporal_refinement=True,
        temporal_parent_sha256=digest(parent / "artifact.json"),
        temporal_changed_per_hand=admitted.sum(0).tolist(),
        temporal_options=dict(
            optimizer=optimizer_options, temporal_weight=weight, max_step_s=max_step
        ),
        accepted_parent_hands_unchanged=not bool(admitted.any()),
        original_unrefined_anchors_unchanged=True,
        coverage_unchanged=True,
        interpolation_used=False,
        temporal_note="Only previously recovered hands may change; unsupported proposals fall back to the parent. Targets are priors, not observations.",
    )
    write_json(out / "quality.json", quality)
    write_json(out / "temporal_attempts.json", attempts)
    write_json(out / "temporal_components.json", components)
    print(
        json.dumps(
            dict(
                accepted=result["accepted"].sum(0).tolist(),
                changed=admitted.sum(0).tolist(),
            )
        ),
        flush=True,
    )
