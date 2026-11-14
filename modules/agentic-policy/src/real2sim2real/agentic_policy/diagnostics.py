"""Shared post-rollout audit, independent of the manual or authored controller.

No dynamics are stepped. Contact forces are recomputed from saved states with
zero solver warmstart and are explicitly estimates, not recorded sensor values.
The same calculation is used for both controllers. Ground truth is never read.
"""

import json
import time
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest, write_json


def retention(records, protocol):
    """Apply a controller-independent airborne interval and retention anchor."""
    initial_z = records[0]["object_com_m"][2]
    airborne = [
        r
        for r in records
        if r["object_com_m"][2] - initial_z >= protocol["airborne_rise_m"]
        and r["support_fraction"] < 0.5
    ]
    result = {
        "carry_samples": len(airborne),
        "carry_drift_m": None,
        "carry_rotation_deg": None,
        "two_pad_contact_fraction": None,
        "pad_alignment_deg": None,
    }
    if not airborne:
        return result

    def relative(r):
        wrist = np.asarray(r["wrist_rotation"])
        return (
            wrist.T @ (np.asarray(r["object_com_m"]) - r["wrist_position_m"]),
            wrist.T @ np.asarray(r["object_rotation"]),
        )

    p0, r0 = relative(airborne[0])
    drift, angles = [], []
    for r in airborne:
        p, rotation = relative(r)
        drift.append(float(np.linalg.norm(p - p0)))
        angles.append(
            float(
                np.rad2deg(
                    np.arccos(np.clip((np.trace(r0.T @ rotation) - 1) / 2, -1, 1))
                )
            )
        )
    alignments = [a for r in airborne for a in r["pad_alignment_deg"] if a is not None]
    result.update(
        carry_drift_m=max(drift),
        carry_rotation_deg=max(angles),
        two_pad_contact_fraction=float(
            np.mean(
                [
                    sum(f >= protocol["minimum_pad_force_n"] for f in r["pad_forces_n"])
                    >= 2
                    for r in airborne
                ]
            )
        ),
        pad_alignment_deg=max(alignments) if alignments else None,
    )
    return result


def audit(bundle, runtime_path, states_path, protocol, output):
    from real2sim2real.controller_actor_training.continuous_policy import (
        ContinuousPolicy,
    )
    from real2sim2real.controller_actor_training.placement_outcome import (
        placement_outcome,
    )

    bundle, states_path, output = (
        Path(bundle).resolve(),
        Path(states_path),
        Path(output),
    )
    runtime = (bundle / runtime_path).resolve()
    if not runtime.is_relative_to(bundle):
        raise ValueError("Audit runtime must be in the frozen bundle")
    config = json.loads(runtime.read_text())
    for key in ["scene", "initial"]:
        path = (bundle / config[key]).resolve()
        if not path.is_relative_to(bundle):
            raise ValueError("Audit assets must be in the frozen bundle")
        config[key] = str(path)
    session = ContinuousPolicy(config)
    m, d, mj = session.model, session.data, session.mj
    with np.load(states_path, allow_pickle=False) as saved:
        q, v, u, times = (
            saved[k].copy() for k in ["qpos", "qvel", "commands", "timestamps_s"]
        )
    if (
        q.shape != (len(times), m.nq)
        or v.shape != (len(times), m.nv)
        or u.shape != (len(times), m.nu)
        or not len(times)
        or not all(np.isfinite(x).all() for x in [q, v, u, times])
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "Finite matching sampled states with a strictly increasing clock required"
        )
    started = time.monotonic()
    rows, poses = [], []
    for i, t in enumerate(times):
        if time.monotonic() - started > protocol.get("wall_deadline_s", 180):
            raise TimeoutError("Post-rollout audit deadline")
        d.qpos[:], d.qvel[:], d.ctrl[:], d.time = q[i], v[i], u[i], t
        d.qacc_warmstart[:] = 0
        r = session.observe()
        alignment = []
        for pad, normal in zip(session.pad_ids, r["pad_normals_world"]):
            values = []
            for co in d.contact:
                a, b = int(co.geom1), int(co.geom2)
                if a == pad and m.geom_bodyid[b] == session.object:
                    desired = np.asarray(co.frame[:3])
                elif b == pad and m.geom_bodyid[a] == session.object:
                    desired = -np.asarray(co.frame[:3])
                else:
                    continue
                if co.dist <= 0:
                    values.append(
                        float(
                            np.rad2deg(
                                np.arccos(np.clip(np.dot(normal, desired), -1, 1))
                            )
                        )
                    )
            alignment.append(min(values) if values else None)
        overlap = 0.0
        for name in config.get("exclusion_geoms", []):
            for gid in session.hand["geoms"]:
                distance = mj.mj_geomDistance(
                    m, d, m.geom(name).id, int(gid), 0.1, np.zeros(6)
                )
                overlap = max(overlap, -float(distance))
        r.update(pad_alignment_deg=alignment, exclusion_overlap_m=overlap)
        rows.append(r)
        poses.append(
            np.r_[d.xpos[session.object], d.xquat[session.object][[1, 2, 3, 0]]]
        )
    outcome = placement_outcome(
        rows,
        horizon_s=config["horizon_s"],
        task=config["task"],
        arm_joints=config["arm_joints"],
    )
    rise = max(r["object_com_m"][2] - rows[0]["object_com_m"][2] for r in rows)
    lifted = rise >= protocol["minimum_lift_m"]
    inverted = any(
        r["object_rotation"][2][2]
        < -np.cos(np.deg2rad(config["task"]["inverted_angle_deg"]))
        and r["object_com_m"][2] - rows[0]["object_com_m"][2]
        >= protocol["airborne_rise_m"]
        and r["support_fraction"] < 0.5
        for r in rows
    )
    metrics = {
        **retention(rows, protocol),
        "maximum_rise_m": float(rise),
        "peak_penetration_m": max(r["penetration_m"] for r in rows),
        "exclusion_overlap_m": max(r["exclusion_overlap_m"] for r in rows),
        "peak_hand_force_n": max(r["hand_force_n"] for r in rows),
        "final_object_speed_m_s": rows[-1]["object_speed_m_s"],
        "home_arm_error_rad": outcome["home"]["arm_error_rad"],
        "home_finger_error_rad": outcome["home"]["finger_error_rad"],
        "finite_states": 1,
        "duration_s": float(times[-1]),
    }
    result = {
        "schema": "real2sim2real.matched-rollout-audit/1",
        "metrics": metrics,
        "outcome": outcome,
        "lift_completed": bool(lifted),
        "airborne_inversion": bool(inverted),
        "task_completed": bool(outcome["task_completed"] and lifted and inverted),
        "source_states_sha256": digest(states_path),
        "runtime_sha256": digest(runtime),
        "protocol": protocol,
        "physics_stepped": False,
        "contact_evidence": "posthoc force estimates, zero solver warmstart; not sensors or original controller feedback",
        "geometry_status": "conditional; scene verification remains an external prerequisite",
    }
    output.mkdir(exist_ok=False, parents=True)
    np.savez_compressed(
        output / "object-poses.npz", timestamps_s=times, poses_xyzw=poses
    )
    write_json(output / "observations.json", rows)
    write_json(output / "diagnostics.json", result)
    return result
