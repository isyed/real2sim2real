"""Bounded geometric grasp proposals, followed by unchanged free-object trials.

Rotate a Cartesian support fixture about a declared world pivot, preserving its
articulation and preserving the world approach and seating directions. Optional named finger
changes are explicit posture hypotheses, not inferred contact observations.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from real2sim2real.artifacts import digest, write_json
from .arm_clearance_audit import table_gap


def propose(
    candidate,
    rotation_vector_rad,
    pivot_world_m,
    joint_offsets_rad,
    additional_opening_gap_m=0.0,
):
    result = deepcopy(candidate)
    vector, pivot = (
        np.asarray(rotation_vector_rad, float),
        np.asarray(pivot_world_m, float),
    )
    if (
        vector.shape != (3,)
        or pivot.shape != (3,)
        or not np.isfinite(vector).all()
        or not np.isfinite(pivot).all()
        or np.linalg.norm(vector) > np.deg2rad(20)
    ):
        raise ValueError(
            "Finite world pivot and support rotation at most 20 degrees required"
        )
    if (
        not np.isfinite(additional_opening_gap_m)
        or not 0 <= additional_opening_gap_m <= 0.02
    ):
        raise ValueError("Additional vertical insertion gap must be within 20 mm")
    support = result["support"]
    rotation = Rotation.from_rotvec(vector).as_matrix()
    support["base_position_m"] = (
        pivot + rotation @ (np.asarray(support["base_position_m"]) - pivot)
    ).tolist()
    support["base_rotation"] = (
        rotation @ np.asarray(support["base_rotation"])
    ).tolist()
    xyz = [support["joint_names"].index(support["fixture_prefix"] + a) for a in "xyz"]
    closed_xyz = np.asarray(support["q_grasp"])[xyz]
    shift = rotation @ closed_xyz - closed_xyz
    for key in ("q_pregrasp", "q_open", "q_grasp"):
        q = np.asarray(support[key]).copy()
        q[xyz] += shift
        support[key] = q.tolist()
    for key in ("q_pregrasp", "q_open"):
        support[key][xyz[2]] -= additional_opening_gap_m
    # These are proposal targets, not a claim of a new surface observation.
    support["targets_m"] = (
        pivot + (np.asarray(support["targets_m"]) - pivot) @ rotation.T
    ).tolist()
    for name, offset in joint_offsets_rad.items():
        if (
            name not in result["joint_names"]
            or name.startswith("fixture_")
            or not np.isfinite(offset)
            or abs(offset) > 0.35
        ):
            raise ValueError("Owned finger offsets of at most 0.35 rad required")
        index = result["joint_names"].index(name)
        for key in ("q_pregrasp", "q_open", "q_grasp"):
            result[key][index] += offset
    # Source-fit diagnostics do not describe changed geometry.
    for group in (result, support):
        for key in (
            "site_errors_m",
            "max_penetration_m",
            "approach_max_penetration_m",
            "parked_max_penetration_m",
            "nfev",
            "converged",
            "approach_search",
        ):
            group.pop(key, None)
    result["contact_replan"] = dict(
        rotation_vector_world_rad=vector.tolist(),
        pivot_world_m=pivot.tolist(),
        joint_offsets_rad=joint_offsets_rad,
        additional_opening_gap_m=additional_opening_gap_m,
        source_targets_revalidated=False,
        physical_contact_validated=False,
    )
    return result


def screen(
    base, candidate, parent, states, minimum_gap_m, height_offsets_m=(0.0, -0.008)
):
    """Replay support commands against saved object poses; no force inference.

    Include an 8 mm lower insertion scenario by translating both object and
    support commands. This is a clearance stress test, not a payload simulation.
    """
    import mujoco
    from .object_grasp_pilot import load_scene, set_wrist_frame, object_transform
    from .support_control import SupportMotion
    from real2sim2real.retargeting.grasp_kinematics import (
        robot_indices,
        contact_penetrations,
    )

    model, data, _ = load_scene(base)
    times = np.asarray([r["time_s"] for r in parent["records"]], float)
    qpos = np.asarray(states["qpos"], float)
    if (
        not 2 <= len(times) <= 1000
        or qpos.shape != (len(times), model.nq)
        or not np.isfinite(qpos).all()
        or not np.isfinite(times).all()
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError(
            "Bounded, aligned, finite saved states and increasing clock required"
        )
    if "timestamps_s" in states and not np.allclose(
        states["timestamps_s"], times, atol=1e-8, rtol=0
    ):
        raise ValueError("Saved state and record clocks disagree")
    set_wrist_frame(
        model, data, base, candidate["base_position_m"], candidate["base_rotation"]
    )
    motion = SupportMotion(model, data, base, candidate)
    idx = robot_indices(model, base["robot"]["root_body"])
    limits_ok = all(
        np.all(
            (np.asarray(candidate[k]) >= model.jnt_range[idx["joints"], 0])
            & (np.asarray(candidate[k]) <= model.jnt_range[idx["joints"], 1])
        )
        for k in ("q_pregrasp", "q_open", "q_grasp")
    )
    table = model.geom("table").id
    object_q = model.joint("task_object_free").qposadr[0]
    z = model.joint("fixture_z").qposadr[0]
    offsets = np.asarray(height_offsets_m, float)
    if (
        offsets.ndim != 1
        or not 1 <= len(offsets) <= 4
        or not np.isfinite(offsets).all()
        or 0 not in offsets
        or (offsets > 0).any()
        or (offsets < -0.02).any()
    ):
        raise ValueError(
            "One to four height stress cases within -20 to 0 mm, including nominal, required"
        )
    for key in ("q_pregrasp", "q_open", "q_grasp"):
        q = np.asarray(candidate["support"][key])
        limits = model.jnt_range[motion.idx["joints"]]
        limits_ok = limits_ok and bool(
            ((q >= limits[:, 0]) & (q <= limits[:, 1])).all()
        )
    rows = []
    for lowering in offsets:
        motion.alignment = None
        motion.rejection = None
        minimum, peak, count = 0.1, 0.0, 0
        for i, record in enumerate(parent["records"]):
            if record["phase"] == "hold":
                break
            data.qpos[:] = states["qpos"][i]
            if record["time_s"] >= motion.start:
                data.qpos[object_q + 2] += lowering
            mujoco.mj_fwdPosition(model, data)
            target = motion.target(
                record["time_s"], object_transform(model, data), data.qpos[z]
            )
            data.qpos[motion.idx["q"]] = target
            mujoco.mj_fwdPosition(model, data)
            minimum = min(minimum, table_gap(model, data, motion.idx["geoms"], table))
            peak = max(
                peak,
                float(contact_penetrations(model, data, motion.idx).max(initial=0)),
            )
            count += 1
        rows.append(
            dict(
                object_height_offset_m=lowering,
                minimum_support_table_gap_m=minimum,
                maximum_support_penetration_m=peak,
                samples=count,
                rejection=motion.rejection,
            )
        )
    reasons = []
    if not limits_ok:
        reasons.append("planned_joint_limits")
    if any(r["minimum_support_table_gap_m"] < minimum_gap_m for r in rows):
        reasons.append("support_table_clearance_margin")
    if any(
        r["maximum_support_penetration_m"] > base["planning"]["max_penetration_m"]
        for r in rows
    ):
        reasons.append("support_path_penetration")
    if any(r["rejection"] for r in rows):
        reasons.append("support_command_limits_or_alignment")
    return dict(
        cases=rows,
        rejected_for=reasons,
        sampled_clearance_only=True,
        forces_observed=False,
        continuous_collision_certified=False,
    )


def run(config):
    from .object_grasp_pilot import load_plan, trial
    from .grasp_posture_search import local_name
    from .retention_experiment import assess

    parent_path = Path(config["parent_trial"])
    if digest(parent_path) != config["parent_trial_sha256"]:
        raise ValueError("Pinned parent changed")
    parent = json.loads(parent_path.read_text())
    if (
        not parent["success"]
        or digest(parent_path.with_name("states.npz")) != parent["states_sha256"]
    ):
        raise ValueError("Successful pinned parent and states required")
    base = parent["configuration"]
    source = load_plan(base)
    if parent["candidate"] not in source["candidates"]:
        raise ValueError("Retained candidate absent from source plan")
    variants = config["variants"]
    names = [local_name(v["name"]) for v in variants]
    if not 1 <= len(names) <= 8 or len(set(names)) != len(names) or "baseline" in names:
        raise ValueError("One to eight distinct bounded proposals required")
    if not 0 <= config["minimum_table_gap_m"] <= 0.02:
        raise ValueError("Bounded positive-clearance screening margin required")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", config)
    (out / "implementation").mkdir()
    for filename in ("contact_replan.py", "arm_clearance_audit.py"):
        (out / "implementation" / filename).write_bytes(
            Path(__file__).with_name(filename).read_bytes()
        )
    with np.load(parent_path.with_name("states.npz"), allow_pickle=False) as archive:
        states = {key: archive[key] for key in archive.files}
    candidates = [deepcopy(parent["candidate"])]
    rows = []
    for i, variant in enumerate(variants):
        candidate = propose(
            parent["candidate"],
            variant["rotation_vector_world_rad"],
            variant["pivot_world_m"],
            variant.get("joint_offsets_rad", {}),
            variant.get("additional_opening_gap_m", 0.0),
        )
        candidate.update(candidate_id=3000 + i, proposal_name=variant["name"])
        diagnostic = screen(
            base,
            candidate,
            parent,
            states,
            config["minimum_table_gap_m"],
            config.get("height_offsets_m", [0.0, -0.008]),
        )
        candidate.update(
            clearance_screen=diagnostic,
            accepted=not diagnostic["rejected_for"],
            rejected_for=diagnostic["rejected_for"],
        )
        candidates.append(candidate)
        print(variant["name"], diagnostic, flush=True)
        write_json(out / "planning-progress.json", dict(candidates=candidates))
    (out / "planning").mkdir()
    write_json(out / "planning/plan.json", dict(source, candidates=candidates))
    settings = deepcopy(base)
    settings["planning_directory"] = str(out / "planning")
    for candidate in candidates:
        name = candidate.get("proposal_name", "baseline")
        if not candidate["accepted"]:
            continue
        result = trial(settings, candidate, out / name)
        if name == "baseline" and result["states_sha256"] != parent["states_sha256"]:
            raise ValueError("Baseline state regression")
        score = assess(result, config["fidelity_limits"])
        row = dict(
            name=name,
            success=result["success"],
            rejected_for=result["rejected_for"],
            hold_slip_m=result["hold_slip_m"],
            support_wrist_drift_m=result["support_wrist_drift_m"],
            hold_opposed_contact_fraction=result["hold_opposed_contact_fraction"],
            **score,
        )
        rows.append(row)
        print(row, flush=True)
        write_json(out / "progress.json", dict(trials=rows))
    report = dict(
        schema="real2sim2real.contact-replan/1",
        config=config,
        trials=rows,
        candidates=[
            dict(
                name=c.get("proposal_name", "baseline"),
                accepted=c["accepted"],
                rejected_for=c["rejected_for"],
            )
            for c in candidates
        ],
        arm_control_validated=False,
        baseline_replaced=False,
        implementation_sha256=digest(out / "implementation/contact_replan.py"),
    )
    write_json(out / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.worker:
        from real2sim2real.bounded_process import run as bounded

        if not 0 < config["deadline_s"] <= 900:
            raise ValueError("Deadline at most 900 seconds required")
        return bounded(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(args.config.resolve()),
                "--worker",
            ],
            timeout=config["deadline_s"],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only experiment required")
    run(config)


if __name__ == "__main__":
    main()
