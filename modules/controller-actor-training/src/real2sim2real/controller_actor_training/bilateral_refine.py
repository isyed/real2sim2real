"""Joint two-hand contact IK with mutual collision and fresh path admission."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
from real2sim2real.artifacts import digest, write_json


def run(config):
    import mujoco
    from .object_grasp_pilot import load_scene, set_root_frame
    from .pad_grasp_planning import path_audit
    from real2sim2real.retargeting.grasp_kinematics import robot_indices
    from real2sim2real.retargeting.pad_contacts import (
        pad_frame,
        fit_pad_contacts,
        contact_patch_samples,
    )
    from real2sim2real.retargeting.paired_contacts import paired_wrench_screen

    path = Path(config["plan"])
    if digest(path) != config["plan_sha256"]:
        raise ValueError("Pinned pair plan changed")
    report = json.loads(path.read_text())
    parent_path = Path(report["config"]["parent_trial"])
    if digest(parent_path) != report["config"]["parent_trial_sha256"]:
        raise ValueError("Parent changed")
    parent = json.loads(parent_path.read_text())
    base = parent["configuration"]
    selected = next(
        c for c in report["candidates"] if c["candidate_id"] == config["candidate_id"]
    )
    if not all("q_grasp" in h for h in selected["hands"]):
        raise ValueError("Both initial hand fits required")
    if not 1 <= len(config["seed_offsets_m"]) <= 4:
        raise ValueError("One to four bounded joint seeds required")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", config)
    rows = []
    for number, offsets in enumerate(config["seed_offsets_m"]):
        offsets = np.asarray(offsets, float)
        if (
            offsets.shape != (2, 3)
            or not np.isfinite(offsets).all()
            or (np.linalg.norm(offsets, axis=1) > 0.05).any()
        ):
            raise ValueError("Two seed offsets within 50 mm required")
        hands = deepcopy(selected["hands"])
        model, data, mesh = load_scene(base)
        mesh.apply_transform(report["settling"]["settled_world_from_object"])
        idx = [robot_indices(model, h["root_body"]) for h in hands]
        for h in hands:
            set_root_frame(
                model,
                data,
                h["root_body"],
                h["fixture_prefix"],
                h["base_position_m"],
                h["base_rotation"],
            )
        objq = model.joint("task_object_free").qposadr[0]
        data.qpos[objq : objq + 7] = report["settling"]["object_qpos"]
        for h, ids in zip(hands, idx):
            data.qpos[ids["q"]] = h["q_grasp"]
        mujoco.mj_fwdPosition(model, data)
        # Rebase at fitted world wrists, retaining every world pose and freeing
        # equal planning travel for both namespaced fixtures.
        poses = [
            (data.xpos[ids["root"]].copy(), data.xmat[ids["root"]].reshape(3, 3).copy())
            for ids in idx
        ]
        saved = data.qpos.copy()
        for h, (position, rotation) in zip(hands, poses):
            set_root_frame(
                model, data, h["root_body"], h["fixture_prefix"], position, rotation
            )
            h["base_position_m"] = position.tolist()
            h["base_rotation"] = rotation.tolist()
        data.qpos[:] = saved
        for h, ids in zip(hands, idx):
            for i, name in enumerate(h["joint_names"]):
                if name.startswith(h["fixture_prefix"]):
                    data.qpos[ids["q"][i]] = 0
        frames = [
            pad_frame(
                model,
                h["root_body"],
                f["geom"],
                f["declared_outward_body"],
                point_body_m=f["point_body_m"],
            )
            for h in hands
            for f in h["pad_plan"]["frames"]
        ]
        targets = np.concatenate([h["targets_m"] for h in hands])
        outward = np.concatenate([h["outward_normals"] for h in hands])
        qids = np.concatenate([ids["q"] for ids in idx])
        seed = data.qpos[qids].copy()
        cursor = 0
        for h, ids, offset in zip(hands, idx, offsets):
            xyz = [h["joint_names"].index(h["fixture_prefix"] + a) for a in "xyz"]
            seed[cursor + np.array(xyz)] += offset
            cursor += len(ids["q"])
        kwargs = dict(
            root_name=[h["root_body"] for h in hands],
            frames=frames,
            fixture_prefix=[h["fixture_prefix"] for h in hands],
            max_nfev=200,
            position_scale_m=0.001,
        )
        fit = fit_pad_contacts(
            model, data, targets=targets, normals=-outward, initial=seed, **kwargs
        )
        opened = fit_pad_contacts(
            model,
            data,
            targets=targets + 0.008 * outward,
            normals=-outward,
            initial=fit["q"],
            **kwargs,
        )
        closed = fit_pad_contacts(
            model,
            data,
            targets=targets - 0.0005 * outward,
            normals=-outward,
            initial=fit["q"],
            **kwargs,
        )
        data.qpos[qids] = closed["q"]
        mujoco.mj_fwdPosition(model, data)
        patches = []
        cursor = 0
        pad_cursor = 0
        for h, ids in zip(hands, idx):
            n = len(ids["q"])
            pad_count = len(h["pad_plan"]["frames"])
            h.update(
                site_errors_m=fit["site_errors_m"][
                    pad_cursor : pad_cursor + pad_count
                ].tolist(),
                normal_errors_deg=fit["normal_errors_deg"][
                    pad_cursor : pad_cursor + pad_count
                ].tolist(),
                q_grasp=fit["q"][cursor : cursor + n].tolist(),
                q_open=opened["q"][cursor : cursor + n].tolist(),
                q_closed=closed["q"][cursor : cursor + n].tolist(),
            )
            pre = np.array(h["q_open"])
            pre[h["joint_names"].index(h["fixture_prefix"] + "z")] += 0.08
            h["q_pregrasp"] = pre.tolist()
            # Use the exact stored frames; no center-ray patch substitution.
            h["patches"] = contact_patch_samples(
                model,
                data,
                h["pad_plan"]["frames"],
                mesh,
                radius_m=0.004,
                max_surface_gap_m=0.0005,
            )
            patches.append(h["patches"])
            cursor += n
            pad_cursor += pad_count
        combined = dict(
            joints=np.concatenate([i["joints"] for i in idx]),
            q=qids,
            dofs=np.concatenate([i["dofs"] for i in idx]),
            geoms=np.concatenate([i["geoms"] for i in idx]),
        )
        waypoints = [
            np.concatenate([h[k] for h in hands])
            for k in ["q_pregrasp", "q_open", "q_closed"]
        ]
        path_a = path_audit(model, data, combined, waypoints)
        wrench = paired_wrench_screen(
            *patches,
            mesh.center_mass,
            base["physics"]["mass_kg"],
            base["physics"]["friction"],
            report["config"]["maximum_normal_force_n"],
        )
        reasons = []
        if (
            max(f["site_errors_m"].max() for f in (fit, opened, closed))
            > base["planning"]["contact_error_m"]
        ):
            reasons.append("pad_position_error")
        if max(f["normal_errors_deg"].max() for f in (fit, opened, closed)) > 25:
            reasons.append("pad_normal_error")
        if path_a["max_penetration_m"] > base["planning"]["max_penetration_m"]:
            reasons.append("paired_path_collision")
        if not path_a["within_joint_limits"]:
            reasons.append("joint_limits")
        if not wrench["feasible"]:
            reasons.append("paired_static_gravity_balance")
        row = dict(
            candidate_id=number,
            hands=hands,
            accepted=not reasons,
            rejected_for=reasons,
            path=path_a,
            paired_path_max_penetration_m=path_a["max_penetration_m"],
            gravity_wrench=wrench,
            fit_position_errors_m=fit["site_errors_m"].tolist(),
            fit_normal_errors_deg=fit["normal_errors_deg"].tolist(),
            maximum_normal_error_deg=float(
                max(f["normal_errors_deg"].max() for f in (fit, opened, closed))
            ),
            physical_success_validated=False,
        )
        rows.append(row)
        write_json(out / "progress.json", dict(candidates=rows))
        print(
            number,
            "accepted",
            row["accepted"],
            reasons,
            "path",
            path_a["max_penetration_m"],
            "samples",
            [p["samples_per_pad"] for p in patches],
            flush=True,
        )
    result = dict(
        schema="real2sim2real.bilateral-grasp-refinement/1",
        config=config,
        parent_trial=str(parent_path),
        parent_trial_sha256=report["config"]["parent_trial_sha256"],
        candidates=rows,
        accepted=sum(c["accepted"] for c in rows),
        settling=report["settling"],
        scene_receipt_sha256=report["scene_receipt_sha256"],
        physical_success_validated=False,
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
            timeout=900,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only refinement required")
    run(json.loads(a.config.read_text()))


if __name__ == "__main__":
    main()
