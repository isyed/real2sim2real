"""Refit named collision-pad patches with fixed wrists and fresh path admission.

Patch rays are bounded anatomical hypotheses on the existing compiled pad hull.
Source contact telemetry guides proposals, not independent geometry validation.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
from real2sim2real.artifacts import digest, write_json


def shifted_contacts(mesh, targets, normals, frames, shifts):
    """Project bounded named contact proposals onto the unchanged source mesh."""
    import trimesh

    points, outward = (
        np.array(targets, float, copy=True),
        np.array(normals, float, copy=True),
    )
    for group, delta in shifts.items():
        if group not in ("thumb", "index", "middle", "ring", "pinky"):
            raise ValueError("Explicit anatomical contact group required")
        indices = [i for i, f in enumerate(frames) if group in f["geom"]]
        delta = np.asarray(delta, float)
        if (
            len(indices) != 1
            or delta.shape != (3,)
            or not np.isfinite(delta).all()
            or np.linalg.norm(delta) > 0.006
        ):
            raise ValueError("One owned contact and a shift within 6 mm required")
        i = indices[0]
        # Thin walls can put the opposite surface closer than the intended one.
        # Preserve the proposed side rather than silently flipping its normal.
        target = points[i] + delta
        eligible = np.linalg.norm(mesh.triangles_center - target, axis=1) <= 0.02
        eligible &= mesh.face_normals @ outward[i] >= np.cos(np.deg2rad(35))
        if not eligible.any():
            raise ValueError("No same-side source surface near shifted contact")
        local = mesh.submesh([np.flatnonzero(eligible)], append=True, repair=False)
        near, gap, face = trimesh.proximity.closest_point(local, [target])
        if gap[0] > 0.003 or np.linalg.norm(near[0] - points[i]) > 0.006:
            raise ValueError(
                "Shifted contact exceeds source-surface projection/travel bounds"
            )
        points[i], outward[i] = near[0], local.face_normals[face[0]]
    return points, outward


def posture_seed(names, original, offsets, bounds):
    seed = np.asarray(original, float).copy()
    for name, delta in offsets.items():
        if name not in names or not np.isfinite(delta) or abs(delta) > 0.2:
            raise ValueError("Owned joint seed changes within 0.2 rad required")
        i = names.index(name)
        if bounds[name][1] - bounds[name][0] < 1e-6:
            raise ValueError("Cannot seed a frozen joint")
        seed[i] = np.clip(seed[i] + delta, *bounds[name])
    return seed


def finger_bounds(model, indices, names, seed, active, radius, prefix):
    seed = np.asarray(seed, float)
    if (
        not 0 < radius <= 0.35
        or len(names) != len(indices["joints"])
        or seed.shape != (len(names),)
        or not np.isfinite(seed).all()
        or len(set(names)) != len(names)
        or not active
        or not set(active) <= set(names)
        or any(n.startswith(prefix) for n in active)
    ):
        raise ValueError("Owned fingers and a bounded posture radius required")
    result = {}
    for j, name, value in zip(indices["joints"], names, seed):
        lo, hi = model.jnt_range[j].copy()
        if name.startswith(prefix):
            bound = 0.35 if name.startswith(prefix + "r") else 0.06
            lo, hi = max(lo, -bound), min(hi, bound)
        delta = radius if name in active else 1e-7
        result[name] = [max(lo, value - delta), min(hi, value + delta)]
        if result[name][0] >= result[name][1]:
            raise ValueError("Source posture has no legal planning interval")
    return result


def run(config):
    import mujoco
    from .object_grasp_pilot import load_scene, set_root_frame
    from .bilateral_grasp import rebase_closed_wrist
    from .pad_grasp_planning import path_audit
    from real2sim2real.retargeting.grasp_kinematics import robot_indices
    from real2sim2real.retargeting.pad_contacts import (
        pad_frame,
        fit_pad_contacts,
        contact_patch_samples,
        unit,
    )
    from real2sim2real.retargeting.paired_contacts import paired_wrench_screen

    plan_path, audit_path = Path(config["plan"]), Path(config["audit"])
    if (
        digest(plan_path) != config["plan_sha256"]
        or digest(audit_path) != config["audit_sha256"]
    ):
        raise ValueError("Pinned patch-plan inputs changed")
    plan, audit = json.loads(plan_path.read_text()), json.loads(audit_path.read_text())
    audited = Path(audit["config"]["source_result"])
    if (
        not audit["states_identical"]
        or digest(audited) != audit["config"]["source_sha256"]
        or json.loads(audited.read_text())["candidate"]["bilateral_plan_sha256"]
        != config["plan_sha256"]
    ):
        raise ValueError(
            "Contact audit must reproduce a trial of this exact grasp plan"
        )
    selected = next(
        c for c in plan["candidates"] if c["candidate_id"] == config["candidate_id"]
    )
    if not selected["accepted"] or not 1 <= len(config["proposals"]) <= 4:
        raise ValueError("An admitted source and one to four proposals required")
    parent_path = Path(plan.get("parent_trial", plan["config"]["parent_trial"]))
    if digest(parent_path) != plan.get(
        "parent_trial_sha256", plan["config"]["parent_trial_sha256"]
    ):
        raise ValueError("Source task/robot parent changed")
    base = json.loads(parent_path.read_text())["configuration"]
    for path, sha in config.get("diagnostic_inputs", {}).items():
        if digest(Path(path)) != sha:
            raise ValueError("Pinned posture diagnostic changed")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", config)
    (out / "implementation").mkdir()
    for name in (
        "bilateral_patch_replan.py",
        "bilateral_grasp.py",
        "pad_grasp_planning.py",
    ):
        (out / "implementation" / name).write_bytes(
            Path(__file__).with_name(name).read_bytes()
        )
    rows = []
    for number, proposal in enumerate(config["proposals"]):
        model, data, mesh = load_scene(base)
        hands = [
            rebase_closed_wrist(model, data, deepcopy(h)) for h in selected["hands"]
        ]
        for h in hands:
            set_root_frame(
                model,
                data,
                h["root_body"],
                h["fixture_prefix"],
                h["base_position_m"],
                h["base_rotation"],
            )
        ids = [robot_indices(model, h["root_body"]) for h in hands]
        objq = int(model.joint("task_object_free").qposadr[0])
        data.qpos[objq : objq + 7] = plan["settling"]["object_qpos"]
        mesh.apply_transform(plan["settling"]["settled_world_from_object"])
        for role, h in zip(("primary", "secondary"), hands):
            for group, direction in proposal.get(role, {}).items():
                if group not in ("thumb", "index", "middle", "ring", "pinky"):
                    raise ValueError("Explicit anatomical patch group required")
                index = next(
                    i
                    for i, f in enumerate(h["pad_plan"]["frames"])
                    if group in f["geom"]
                )
                old = h["pad_plan"]["frames"][index]
                if unit(direction) @ unit(old["declared_outward_body"]) < np.cos(
                    np.deg2rad(35)
                ):
                    raise ValueError(
                        "Patch-ray change exceeds 35 degree proposal bound"
                    )
                h["pad_plan"]["frames"][index] = pad_frame(
                    model, h["root_body"], old["geom"], direction
                )
            h["targets_m"], h["outward_normals"] = shifted_contacts(
                mesh,
                h["targets_m"],
                h["outward_normals"],
                h["pad_plan"]["frames"],
                proposal.get("surface_shifts_m", {}).get(role, {}),
            )
            h["targets_m"] = h["targets_m"].tolist()
            h["outward_normals"] = h["outward_normals"].tolist()
        frames = [f for h in hands for f in h["pad_plan"]["frames"]]
        targets = np.concatenate([h["targets_m"] for h in hands])
        outward = np.concatenate([h["outward_normals"] for h in hands])
        qids = np.concatenate([i["q"] for i in ids])
        fits = {}
        frozen_wrist_change = 0.0
        frozen_translation_m = frozen_rotation_rad = 0.0
        for key, inset in [("q_grasp", 0.0), ("q_closed", 0.0005)]:
            seed = np.concatenate([h[key] for h in hands])
            data.qpos[qids] = seed
            bounds = {}
            for role, h, i in zip(("primary", "secondary"), hands, ids):
                groups = set(proposal.get(role, {})) | set(
                    proposal.get("surface_shifts_m", {}).get(role, {})
                )
                active = [
                    n
                    for n in h["joint_names"]
                    if any(g in n for g in groups)
                    and not n.startswith(h["fixture_prefix"])
                ]
                active.extend(
                    n
                    for n in proposal.get("seed_offsets_rad", {}).get(role, {})
                    if n not in active
                )
                if active:
                    bounds.update(
                        finger_bounds(
                            model,
                            i,
                            h["joint_names"],
                            h[key],
                            active,
                            config["maximum_joint_change_rad"],
                            h["fixture_prefix"],
                        )
                    )
                else:
                    for j, n, v in zip(i["joints"], h["joint_names"], h[key]):
                        lo, hi = model.jnt_range[j]
                        bounds[n] = [max(lo, v - 1e-7), min(hi, v + 1e-7)]
            names = [n for h in hands for n in h["joint_names"]]
            offsets = {
                n: v
                for role in ("primary", "secondary")
                for n, v in proposal.get("seed_offsets_rad", {}).get(role, {}).items()
            }
            fit_seed = posture_seed(names, seed, offsets, bounds)
            fits[key] = fit_pad_contacts(
                model,
                data,
                [h["root_body"] for h in hands],
                frames,
                targets - inset * outward,
                -outward,
                fit_seed,
                joint_bounds=bounds,
                fixture_prefix=[h["fixture_prefix"] for h in hands],
                max_nfev=200,
                position_scale_m=0.001,
            )
            names = [n for h in hands for n in h["joint_names"]]
            mask = [
                n.startswith(tuple(h["fixture_prefix"] for h in hands)) for n in names
            ]
            frozen_wrist_change = max(
                frozen_wrist_change,
                float(np.max(np.abs(fits[key]["q"][mask] - seed[mask]))),
            )
            joints = np.concatenate([i["joints"] for i in ids])
            change = np.abs(fits[key]["q"] - seed)
            fixture = np.asarray(mask)
            frozen_translation_m = max(
                frozen_translation_m,
                float(
                    change[
                        fixture
                        & (model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_SLIDE)
                    ].max(initial=0)
                ),
            )
            frozen_rotation_rad = max(
                frozen_rotation_rad,
                float(
                    change[
                        fixture
                        & (model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE)
                    ].max(initial=0)
                ),
            )
        cursor = 0
        pad_cursor = 0
        for h, i in zip(hands, ids):
            n = len(i["q"])
            for key in fits:
                h[key] = fits[key]["q"][cursor : cursor + n].tolist()
            count = len(h["pad_plan"]["frames"])
            h["site_errors_m"] = fits["q_grasp"]["site_errors_m"][
                pad_cursor : pad_cursor + count
            ].tolist()
            h["normal_errors_deg"] = fits["q_grasp"]["normal_errors_deg"][
                pad_cursor : pad_cursor + count
            ].tolist()
            h["closed_target_errors_m"] = fits["q_closed"]["site_errors_m"][
                pad_cursor : pad_cursor + count
            ].tolist()
            cursor += n
            pad_cursor += count
        data.qpos[qids] = fits["q_closed"]["q"]
        mujoco.mj_fwdPosition(model, data)
        patches = [
            contact_patch_samples(model, data, h["pad_plan"]["frames"], mesh)
            for h in hands
        ]
        for h, patch in zip(hands, patches):
            h["patches"] = patch
        combined = {
            k: np.concatenate([i[k] for i in ids])
            for k in ("joints", "q", "dofs", "geoms")
        }
        path = path_audit(
            model,
            data,
            combined,
            [
                np.concatenate([h[k] for h in hands])
                for k in ("q_pregrasp", "q_open", "q_closed")
            ],
        )
        wrench = paired_wrench_screen(
            *patches,
            mesh.center_mass,
            base["physics"]["mass_kg"],
            base["physics"]["friction"],
            plan["config"]["maximum_normal_force_n"],
        )
        reasons = []
        if (
            max(v["site_errors_m"].max() for v in fits.values())
            > base["planning"]["contact_error_m"]
        ):
            reasons.append("pad_position_error")
        if max(v["normal_errors_deg"].max() for v in fits.values()) > 25:
            reasons.append("pad_normal_error")
        if path["max_penetration_m"] > base["planning"]["max_penetration_m"]:
            reasons.append("paired_path_collision")
        if not path["within_joint_limits"]:
            reasons.append("joint_limits")
        if not all(p["all_pads_observed"] for p in patches):
            reasons.append("missing_finite_pad_contact")
        if not wrench["feasible"]:
            reasons.append("paired_static_gravity_balance")
        if frozen_wrist_change > 1e-6:
            reasons.append("frozen_wrist_changed")
        row = dict(
            candidate_id=9000 + number,
            hands=hands,
            accepted=not reasons,
            rejected_for=reasons,
            paired_path_max_penetration_m=path["max_penetration_m"],
            gravity_wrench=wrench,
            proposal=proposal,
            path=path,
            patches=patches,
            wrists_frozen=True,
            maximum_frozen_wrist_change=frozen_wrist_change,
            maximum_frozen_translation_m=frozen_translation_m,
            maximum_frozen_rotation_rad=frozen_rotation_rad,
            maximum_site_error_m=float(
                max(v["site_errors_m"].max() for v in fits.values())
            ),
            maximum_normal_error_deg=float(
                max(v["normal_errors_deg"].max() for v in fits.values())
            ),
            physical_success_validated=False,
        )
        rows.append(row)
        write_json(out / "progress.json", dict(candidates=rows))
        print(
            row["candidate_id"],
            reasons,
            row["maximum_site_error_m"],
            row["maximum_normal_error_deg"],
            flush=True,
        )
    result = dict(
        plan,
        candidates=rows,
        accepted=sum(r["accepted"] for r in rows),
        patch_replan_configuration=config,
        parent_trial=str(parent_path),
        parent_trial_sha256=digest(parent_path),
        source_plan_sha256=config["plan_sha256"],
    )
    write_json(out / "plan.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    if not a.worker:
        from real2sim2real.bounded_process import run as bounded

        return bounded(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(a.config.resolve()),
                "--worker",
            ],
            timeout=600,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only patch replanning required")
    run(json.loads(a.config.read_text()))


if __name__ == "__main__":
    main()
