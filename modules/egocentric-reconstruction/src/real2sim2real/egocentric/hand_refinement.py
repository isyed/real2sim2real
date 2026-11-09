"""Refine rejected observations in a new, independently reviewable interaction.

Accepted parent hands are immutable anchors. No missing detection is filled.
Image fitting uses the same observations and only the fit half of stereo depth;
the parent's absolute gates and additional non-regression checks decide admission.
"""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, load, write_json
from .camera import transform
from .interaction_io import Reconstruction, fresh_directory
from .metric_hands import hand_metrics, hand_rejections, observed_hand_surface


def refinement_rejections(before, after, limits):
    keys = (
        "reprojection_median_px",
        "reprojection_p90_px",
        "validation_surface_median_m",
        "validation_surface_p90_m",
    )
    if any(before.get(key) is None or not np.isfinite(before[key]) for key in keys):
        raise ValueError("Refinement requires scored finite parent evidence")
    failed = hand_rejections(after, limits)
    image = ("reprojection_median_px", "reprojection_p90_px")
    if any(after[k] is None or not np.isfinite(after[k]) for k in keys):
        return failed
    if any(after[k] > before[k] + 0.5 for k in image):
        failed.append("image_regression")
    if not any(after[k] <= 0.9 * before[k] for k in image):
        failed.append("insufficient_image_improvement")
    for key, tolerance in (
        ("validation_surface_median_m", 0.001),
        ("validation_surface_p90_m", 0.003),
    ):
        if (
            after[key] is None
            or before[key] is None
            or after[key] > before[key] + tolerance
        ):
            failed.append("depth_regression:" + key)
    return failed


def admit_temporal(parent, proposed, wrists, times, max_speed):
    """Discard proposals causing a fast transition; never discard parent anchors."""
    parent, proposed = np.asarray(parent, bool), np.asarray(proposed, bool)
    wrists, times = np.asarray(wrists), np.asarray(times)
    if parent.shape != proposed.shape or parent.ndim != 2 or parent.shape[1] != 2:
        raise ValueError("Matching Nx2 parent and proposal masks required")
    if (
        wrists.shape != (*parent.shape, 3)
        or times.shape != (len(parent),)
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
        or not np.isfinite(max_speed)
        or max_speed <= 0
    ):
        raise ValueError(
            "Finite increasing timeline, wrists and positive speed limit required"
        )
    accepted = parent | proposed
    if not np.isfinite(wrists[accepted]).all():
        raise ValueError("Accepted wrists must be finite")
    # Removing a proposal can expose another edge; repeat until anchors/gaps settle.
    for _ in range(len(parent) + 1):
        remove = np.zeros_like(parent)
        for h in range(2):
            ids = np.flatnonzero(accepted[:, h])
            for a, b in zip(ids[:-1], ids[1:]):
                speed = np.linalg.norm(wrists[b, h] - wrists[a, h]) / (
                    times[b] - times[a]
                )
                if speed > max_speed:
                    for i in (a, b):
                        if not parent[i, h]:
                            remove[i, h] = True
        if not remove.any():
            return accepted
        accepted[remove] = False
    raise RuntimeError("Temporal admission did not converge")


def run(request, check=False):
    from .mano_refinement import ManoHand, refine, validate_options

    options = request["hand_refinement"]
    optimizer_options = validate_options(options.get("optimizer", {}))
    parent, root = Path(options["parent"]), Path(request["output"])
    model_path = Path(request["weights"]["hamer"]) / "_DATA/data/mano/MANO_RIGHT.pkl"
    model = ManoHand(model_path)
    if check:
        return
    load(parent, kinds={"metric-interaction-reconstruction"})
    source = Reconstruction(request["reconstruction"], request.get("frame_count"))
    parent_quality = json.loads((parent / "hands_metric/quality.json").read_text())
    if parent_quality.get("articulation_refinement"):
        raise ValueError(
            "Start from the retained unrefined prior, not a chained refinement"
        )
    parent_inputs = json.loads((parent / "inputs.json").read_text())
    for name, sha in parent_inputs["files"].items():
        if digest(source.path(name)) != sha:
            raise ValueError("Refinement observations differ from parent: " + name)
    baseline = np.load(parent / "hands_metric/hands.npz")
    observations = np.load(parent / "hands2d/landmarks.npz")
    raw = np.load(parent / "hands_raw/predictions.npz")
    provenance = json.loads((parent / "hands_raw/provenance.json").read_text())
    for data in (baseline, observations, raw):
        if not np.array_equal(data["frame_ids"], source.frame_ids) or not np.allclose(
            data["timestamps_s"], source.times, rtol=0, atol=1e-8
        ):
            raise ValueError("Refinement requires the exact parent timeline")
    if (
        parent_quality["raw_sha256"] != digest(parent / "hands_raw/predictions.npz")
        or provenance["observations_sha256"] != digest(parent / "hands2d/landmarks.npz")
        or provenance["mano_sha256"] != digest(model_path)
    ):
        raise ValueError(
            "Model or observation provenance differs from the retained prior"
        )
    for directory in ("support", "hands_raw"):
        shutil.copytree(parent / directory, root / directory)
    (root / "hands2d").mkdir()
    for name in ("landmarks.npz", "quality.json"):
        shutil.copyfile(parent / "hands2d" / name, root / "hands2d" / name)
    out = fresh_directory(root / "hands_metric")
    write_json(root / "baseline_hand_quality.json", parent_quality)
    hands = {name: baseline[name].copy() for name in baseline.files}
    before = hands["accepted"].copy()
    quality = deepcopy(parent_quality)
    records, attempts = quality["records"], []
    candidates = np.zeros_like(before)
    local_delta = np.zeros((*before.shape, 15, 3), np.float32)
    scene = np.load(root / "support/scene.npz")
    support = json.loads((root / "support/support.json").read_text())
    mask_ids = request["hand_landmarks"].get("mask_ids", [2, 3])
    selected = (
        np.isfinite(hands["fits"]).all(2)
        & raw["accepted"]
        & observations["accepted"]
        & ~before
    )
    if options.get("frame_ids") is not None:
        frame_ids = np.asarray(options["frame_ids"])
        if (
            frame_ids.ndim != 1
            or frame_ids.dtype.kind not in "iu"
            or not len(frame_ids)
            or len(np.unique(frame_ids)) != len(frame_ids)
            or not np.isin(frame_ids, source.frame_ids).all()
        ):
            raise ValueError("Select unique frame IDs from the parent timeline")
        selected &= np.isin(source.frame_ids, options["frame_ids"])[:, None]
    import trimesh
    from .contact_diagnostics import surface_contact, support_clearance
    from .mesh_surface import MeshSurfaceQuery

    object_mesh = trimesh.load(source.path("mesh/metric.glb"), force="mesh")
    proximity = MeshSurfaceQuery(object_mesh)
    for i, h in np.argwhere(selected):
        start = time.monotonic()
        record = dict(frame=int(source.frame_ids[i]), hand=("left", "right")[h])
        try:
            points, heldout = observed_hand_surface(
                source.depth(i),
                source.mask(i, mask_ids[h]),
                observations["keypoints_px"][i, h],
                source.k,
            )
            prior = dict(
                betas=raw["fixed_betas"][h],
                pose=raw["hand_pose"][i, h],
                orient=raw["global_orient"][i, h],
                vertices=raw["vertices"][i, h],
                joints=raw["joints"][i, h],
                faces=raw["faces"][h],
            )
            v, j, details = refine(
                model,
                prior,
                hands["fits"][i, h],
                observations["keypoints_px"][i, h],
                points,
                source.k,
                h,
                options=optimizer_options,
            )
            metrics = hand_metrics(
                v,
                raw["faces"][h],
                j,
                observations["keypoints_px"][i, h],
                heldout,
                source.k,
            )
            failed = refinement_rejections(
                parent_quality["records"][i][h], metrics, quality["limits"]
            )
            world_v = transform(v, scene["world_from_camera"][i])
            world_j = transform(j, scene["world_from_camera"][i])
            contact = surface_contact(
                object_mesh,
                transform(world_v, np.linalg.inv(scene["world_from_object"][i])),
                proximity=proximity,
            )
            table = support_clearance(world_v, support["observed_boundary_xy"])
            if (
                contact["penetration_depth_m"] is not None
                and contact["penetration_depth_m"] > 0.003
            ):
                failed.append("object_penetration")
            if (
                table["penetration_depth_m"] is not None
                and table["penetration_depth_m"] > 0.003
            ):
                failed.append("table_penetration")
            hands["proposed_camera_vertices"][i, h] = v
            hands["proposed_camera_joints"][i, h] = j
            local_delta[i, h] = details["local_rotation_delta_rad"]
            record.update(
                metrics,
                accepted=not failed,
                rejected_for=failed,
                optimizer=details,
                object_contact=contact,
                table_contact=table,
            )
            records[i][h] = dict(
                metrics,
                frame=int(source.frame_ids[i]),
                hand=record["hand"],
                scale=quality["scale_per_hand"][h],
                accepted=not failed,
                rejected_for=failed,
                articulation_refined=True,
            )
            if not failed:
                candidates[i, h] = True
                hands["camera_vertices"][i, h], hands["camera_joints"][i, h] = v, j
                hands["world_vertices"][i, h], hands["world_joints"][i, h] = (
                    world_v,
                    world_j,
                )
                r = Rotation.from_rotvec(
                    details["rigid_rotation_delta_rad"]
                ) * Rotation.from_rotvec(hands["fits"][i, h, :3])
                hands["fits"][i, h, :3] = r.as_rotvec()
                hands["fits"][i, h, 3:6] += details["translation_delta_m"]
        except ValueError as error:
            record.update(accepted=False, reason=str(error))
        record["elapsed_s"] = time.monotonic() - start
        attempts.append(record)
        write_json(out / "refinement_attempts.json", attempts)
        print(
            f"refine frame {i}, hand {h}: {record.get('rejected_for', record.get('reason', 'accepted'))}, {record['elapsed_s']:.1f}s",
            flush=True,
        )
    accepted = admit_temporal(
        before,
        candidates,
        hands["world_joints"][:, :, 0],
        source.times,
        quality["limits"]["max_wrist_speed_m_s"],
    )
    for i, h in np.argwhere(candidates & ~accepted):
        records[i][h]["accepted"] = False
        records[i][h]["rejected_for"].append("wrist_speed")
    frame_index = {int(frame): i for i, frame in enumerate(source.frame_ids)}
    for attempt in attempts:
        i, h = frame_index[attempt["frame"]], ("left", "right").index(attempt["hand"])
        attempt["accepted"] = bool(accepted[i, h])
        if candidates[i, h] and not accepted[i, h]:
            attempt["rejected_for"] = records[i][h]["rejected_for"].copy()
    write_json(out / "refinement_attempts.json", attempts)
    for key in ("camera_vertices", "camera_joints", "world_vertices", "world_joints"):
        hands[key][~accepted] = np.nan
        if not np.array_equal(hands[key][before], baseline[key][before]):
            raise AssertionError("Refinement changed an accepted parent hand")
    hands.update(
        accepted=accepted,
        articulation_delta_rad=local_delta,
        refinement_accepted=accepted & ~before,
    )
    np.savez_compressed(out / "hands.npz", **hands)
    quality.update(
        accepted_per_hand=accepted.sum(0).tolist(),
        complete=bool(accepted.all()),
        articulation_refinement=True,
        refinement_parent_sha256=digest(parent / "artifact.json"),
        baseline_accepted_per_hand=before.sum(0).tolist(),
        recovered_per_hand=(accepted & ~before).sum(0).tolist(),
        refinement_options=optimizer_options,
        refinement_admission=dict(
            image_min_fractional_improvement=0.1,
            max_image_regression_px=0.5,
            max_depth_median_regression_m=0.001,
            max_depth_p90_regression_m=0.003,
            max_new_penetration_m=0.003,
        ),
        accepted_parent_hands_unchanged=True,
        shape_and_scale_fixed=True,
        contact_used_for_fitting=False,
        contact_used_for_admission=True,
        support_sha256=digest(root / "support/support.json"),
        interpolation_used=False,
        refinement_note="fits compose the saved local MANO articulation deltas with the raw prior; all geometry is explicit",
    )
    write_json(out / "quality.json", quality)
    print(
        json.dumps(
            dict(
                accepted=accepted.sum(0).tolist(),
                recovered=(accepted & ~before).sum(0).tolist(),
            )
        ),
        flush=True,
    )
