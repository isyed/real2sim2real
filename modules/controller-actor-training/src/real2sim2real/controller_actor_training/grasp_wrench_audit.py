"""Read-only CPU audit of saved grasp load capacity; never replay control or train."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, write_json

from .paired_acquisition_audit import pinned_json
from .wrench_feasibility import build, contact_basis, solve


def reference_kinematics(times, poses, query, body_com, window=7):
    """Declared local cubic derivatives in world coordinates; no pose/time edits."""
    t = np.asarray(times, float)
    poses = np.asarray(poses, float)
    if (
        window < 5
        or window % 2 != 1
        or len(t) < window
        or not t[0] <= query <= t[-1]
        or not (np.diff(t) > 0).all()
    ):
        raise ValueError(
            "Complete reference and odd local derivative window >=5 required"
        )
    k = int(np.argmin(abs(t - query)))
    start = int(np.clip(k - window // 2, 0, len(t) - window))
    ids = np.arange(start, start + window)
    rotation = poses[k, :3, :3]
    com = poses[ids, :3, 3] + np.einsum("tij,j->ti", poses[ids, :3, :3], body_com)
    V = np.vander(t[ids] - query, 4, increasing=True)
    c = np.linalg.lstsq(V, com, rcond=None)[0]
    rotvec = Rotation.from_matrix(poses[ids, :3, :3] @ rotation.T).as_rotvec()
    angle = np.linalg.lstsq(V, rotvec, rcond=None)[0]
    # The local log origin is the nearest native pose; correct its small fitted
    # offset when deriving world rates by differentiating fitted rotation matrices.
    h = 1e-4
    phi = lambda x: np.array([1, x, x * x, x * x * x]) @ angle
    R = lambda x: Rotation.from_rotvec(phi(x)).as_matrix() @ rotation
    omega = lambda x: Rotation.from_matrix(R(x + h) @ R(x - h).T).as_rotvec() / (2 * h)
    return {
        "acceleration_world_m_s2": 2 * c[2],
        "omega_world_rad_s": omega(0),
        "alpha_world_rad_s2": (omega(h) - omega(-h)) / (2 * h),
        "window_frames": window,
    }


def subtree(model, body):
    result = set()
    for b in range(model.nbody):
        p = b
        while p and p != body:
            p = int(model.body_parentid[p])
        if p == body:
            result.add(b)
    return result


def actuator_layout(model, roots):
    import mujoco

    ids = []
    dofs = []
    joints = []
    groups = []
    members = {}
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        body = int(model.jnt_bodyid[j])
        if (
            model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT
            or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE
            or not np.array_equal(model.actuator_gear[a], [1, 0, 0, 0, 0, 0])
            or not model.actuator_forcelimited[a]
            or model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED
            or model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE
            or model.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE
        ):
            raise ValueError(
                "Adapter requires unique unit-gear force-limited affine hinge actuators"
            )
        group = None
        for role, bodies in roots.items():
            if body in bodies:
                name = model.joint(j).name
                g = next(
                    (
                        g
                        for g in ("thumb", "index", "middle", "ring", "pinky")
                        if g in name
                    ),
                    None,
                )
                if g:
                    group = role + ":" + g
        ids.append(a)
        dofs.append(int(model.jnt_dofadr[j]))
        joints.append(j)
        groups.append(group)
        if group:
            members.setdefault(group, []).append(len(ids) - 1)
    if len(set(dofs)) != len(dofs):
        raise ValueError("One actuator per audited scalar joint required")
    for name, values in members.items():
        members[name] = sorted(values, key=lambda i: joints[i])
    return {
        "ids": np.array(ids),
        "dofs": np.array(dofs),
        "joints": np.array(joints),
        "groups": groups,
        "members": members,
    }


def torque_bounds(m, d, layout, record, scope, budget):
    a, j = layout["ids"], layout["joints"]
    q = d.qpos[m.jnt_qposadr[j]]
    v = d.qvel[layout["dofs"]]
    gain = m.actuator_gainprm[a, 0]
    if not np.isfinite(gain).all() or (gain <= 0).any():
        raise ValueError("Positive finite affine servo gains required")
    bias = (
        m.actuator_biasprm[a, 0]
        + m.actuator_biasprm[a, 1] * q
        + m.actuator_biasprm[a, 2] * v
    )
    lo, hi = m.actuator_forcerange[a].T.copy()
    for i, jid in enumerate(j):
        if m.jnt_actfrclimited[jid]:
            lo[i] = max(lo[i], m.jnt_actfrcrange[jid, 0])
            hi[i] = min(hi[i], m.jnt_actfrcrange[jid, 1])
    control_lo = np.where(
        m.actuator_ctrllimited[a], m.actuator_ctrlrange[a, 0], -np.inf
    )
    control_hi = np.where(m.actuator_ctrllimited[a], m.actuator_ctrlrange[a, 1], np.inf)
    nominal = d.ctrl[a].copy()
    if scope.startswith("feedback"):
        for label, idx in layout["members"].items():
            old = np.asarray(
                record["contact_force_feedback"][label]["joint_offset_rad"]
            )
            if len(old) != len(idx):
                raise ValueError("Pressure-servo joint order changed")
            nominal[idx] -= old
            radius = budget / (np.sqrt(len(idx)) if scope == "feedback_inner" else 1.0)
            control_lo[idx] = np.maximum(control_lo[idx], nominal[idx] - radius)
            control_hi[idx] = np.minimum(control_hi[idx], nominal[idx] + radius)
    hard_lo, hard_hi = lo.copy(), hi.copy()
    lo = np.clip(gain * control_lo + bias, hard_lo, hard_hi)
    hi = np.clip(gain * control_hi + bias, hard_lo, hard_hi)
    return lo, hi, nominal, gain, bias


def snapshot(m, d, layout, roots, object_id):
    import mujoco

    from .support_control import contact_wrench_on_body

    contacts = []
    observed = []
    external = np.zeros(6)
    center = d.xipos[object_id]
    for cid, c in enumerate(d.contact):
        geoms = [int(c.geom1), int(c.geom2)]
        bodies = [int(m.geom_bodyid[g]) for g in geoms]
        if object_id not in bodies:
            continue
        side = bodies.index(object_id)
        hand = 1 - side
        role = next((r for r, bs in roots.items() if bodies[hand] in bs), None)
        raw = np.zeros(6)
        mujoco.mj_contactForce(m, d, cid, raw)
        if role is None:
            f, t = contact_wrench_on_body(m, d, cid, object_id)
            external += np.r_[f, t]
            continue
        if c.efc_address < 0:
            continue
        dim = int(c.dim)
        frame = (1 if side == 1 else -1) * np.asarray(c.frame).reshape(3, 3).T
        G = contact_basis(frame, c.pos, center, dim)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, c.pos, bodies[hand])
        local = np.zeros((6, dim))
        local[: min(3, dim), : min(3, dim)] = np.eye(min(3, dim))
        if dim > 3:
            local[3:dim, 3:dim] = np.eye(dim - 3)
        F = frame @ local[:3]
        M = frame @ local[3:]
        # Contact reaction on the hand is negative. The actuator required to
        # counter it therefore adds +J_hand^T f_object to the dynamics offset.
        T = jp[:, layout["dofs"]].T @ F + jr[:, layout["dofs"]].T @ M
        name = m.geom(geoms[hand]).name or str(geoms[hand])
        group = next(
            (g for g in ("thumb", "index", "middle", "ring", "pinky") if g in name),
            "other",
        )
        contacts.append(
            {
                "wrench": G,
                "torque": T,
                "friction": np.asarray(c.friction[: dim - 1]).copy(),
                "group": role + ":" + group,
                "geom": name,
                "point_world_m": c.pos.copy(),
                "normal_force_n": float(raw[0]),
            }
        )
        observed.append(raw[:dim])
    if not contacts:
        raise ValueError("No active hand/object contacts at this sample")
    actual = np.concatenate(observed)
    G = np.hstack([c["wrench"] for c in contacts])
    T = np.hstack([c["torque"] for c in contacts])
    full = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, full, d.qM)
    base = full @ d.qacc + d.qfrc_bias - d.qfrc_passive - d.qfrc_applied
    bg = d.qfrc_constraint[layout["dofs"]] + T @ actual
    offset = base[layout["dofs"]] - bg
    residual = offset + T @ actual - d.qfrc_actuator[layout["dofs"]]
    jfree = m.body_jntadr[object_id]
    if (
        m.jnt_type[jfree] != mujoco.mjtJoint.mjJNT_FREE
        or m.body_parentid[object_id] != 0
    ):
        raise ValueError("A separate unassisted free object body is required")
    od = np.arange(m.jnt_dofadr[jfree], m.jnt_dofadr[jfree] + 6)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacBodyCom(m, d, jp, jr, object_id)
    world_required = np.linalg.solve(np.vstack([jp, jr])[:, od].T, base[od])
    error = G @ actual + external - world_required
    if (
        np.max(abs(residual)) > 1e-5
        or np.linalg.norm(error[:3]) > 1e-5
        or np.linalg.norm(error[3:]) > 1e-6
    ):
        raise ValueError(
            "Contact action/reaction or dynamics reconstruction did not close"
        )
    return (
        contacts,
        offset,
        {
            "robot_torque_residual_nm": float(np.max(abs(residual))),
            "object_force_residual_n": float(np.linalg.norm(error[:3])),
            "object_moment_residual_nm": float(np.linalg.norm(error[3:])),
            "other_robot_constraint_peak_nm": float(np.max(abs(bg))),
            "external_object_wrench": external.tolist(),
            "measured_hand_wrench": (G @ actual).tolist(),
            "actual_required_wrench": world_required.tolist(),
        },
    )


def run(config):
    import mujoco

    from .continuous_reference import reference_goal
    from .guided_motion import load_reference

    started = time.monotonic()
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    report = pinned_json(config["result"])
    path = Path(config["result"]["path"])
    for name, key in [("scene.xml", "scene_sha256"), ("states.npz", "states_sha256")]:
        if digest(path.with_name(name)) != report[key]:
            raise ValueError("Pinned physical archive changed")
    if (
        not report.get("physics_run")
        or report.get("motion_condition") != "demonstrated"
        or "incomplete_pilot" in report["rejected_for"]
    ):
        raise ValueError("Complete saved continuous physics required")
    m = mujoco.MjModel.from_xml_path(str(path.with_name("scene.xml")))
    if m.opt.cone != mujoco.mjtCone.mjCONE_ELLIPTIC:
        raise ValueError("This adapter requires elliptic model contacts")
    d = mujoco.MjData(m)
    qs = mujoco.MjData(m)
    obj = m.body(config["object_body"]).id
    roots = {r: subtree(m, m.body(n).id) for r, n in config["hand_roots"].items()}
    layout = actuator_layout(m, roots)
    audit = pinned_json(
        {"path": report["config"]["audit"], "sha256": report["config"]["audit_sha256"]}
    )
    for source, sha in audit["sources"].items():
        if digest(Path(source)) != sha:
            raise ValueError("Pinned source geometry/configuration changed")
    parent_path = Path(audit["config"]["trial"])
    if digest(parent_path) != audit["sources"][str(parent_path)]:
        raise ValueError("Source command ancestry changed")
    parent = json.loads(parent_path.read_text())
    ref = load_reference(parent["configuration"]["reference_guidance"])
    refposes = np.array(
        [
            reference_goal(
                ref, t, report["continuous_protocol"]["initial_world_from_object"]
            )
            for t in ref.elapsed
        ]
    )
    with np.load(path.with_name("states.npz"), allow_pickle=False) as f:
        states = {k: f[k].copy() for k in f.files}
    times = states["timestamps_s"]
    grid = np.r_[
        np.arange(config["start_s"], config["end_s"] + 1e-8, config["sample_period_s"]),
        config["critical_times_s"],
    ]
    indices = sorted({int(np.argmin(abs(times - t))) for t in grid})
    if any(
        abs(times[i] - t) > 0.03 for t in grid for i in [int(np.argmin(abs(times - t)))]
    ):
        raise ValueError("Complete saved sample coverage required")
    rows = []
    witnesses = {}
    modes = ("motor_25N", "goal_10N", "feedback_inner", "feedback_outer")
    for order, i in enumerate(indices):
        if time.monotonic() - started > config["deadline_s"] - 5:
            raise TimeoutError("Audit deadline reached")
        d.qpos[:] = states["qpos"][i]
        d.qvel[:] = states["qvel"][i]
        d.ctrl[:] = states["ctrl"][i]
        d.qacc_warmstart[:] = 0
        d.time = float(times[i])
        mujoco.mj_forward(m, d)
        contacts, dynamic_offset, validation = snapshot(m, d, layout, roots, obj)
        row = {
            "time_s": float(times[i]),
            "record_index": i,
            "phase": report["records"][i]["phase"],
            "contact_count": len(contacts),
            "validation": validation,
        }
        row["contact_stage"] = (
            "external_support"
            if np.linalg.norm(validation["external_object_wrench"]) > 1e-5
            else "hands_only"
        )
        row["source_table_contacts"] = report["records"][i]["table_contacts"]
        qs.qpos[:] = d.qpos
        qs.qvel[:] = 0
        qs.ctrl[:] = d.ctrl
        qs.qacc_warmstart[:] = 0
        mujoco.mj_forward(m, qs)
        static_offset = (qs.qfrc_bias - qs.qfrc_passive)[layout["dofs"]]
        kin = reference_kinematics(
            ref.elapsed,
            refposes,
            times[i],
            m.body_ipos[obj],
            config["derivative_window_frames"],
        )
        kin2 = reference_kinematics(
            ref.elapsed,
            refposes,
            times[i],
            m.body_ipos[obj],
            config["sensitivity_window_frames"],
        )
        I = (
            d.ximat[obj].reshape(3, 3)
            @ np.diag(m.body_inertia[obj])
            @ d.ximat[obj].reshape(3, 3).T
        )

        def load(k, I=I):
            return np.r_[
                m.body_mass[obj] * (k["acceleration_world_m_s2"] - m.opt.gravity),
                I @ k["alpha_world_rad_s2"]
                + np.cross(k["omega_world_rad_s"], I @ k["omega_world_rad_s"]),
            ]

        desired = load(kin)
        gravity = np.r_[-m.body_mass[obj] * m.opt.gravity, np.zeros(3)]
        row.update(
            reference_required_wrench=desired.tolist(),
            gravity_required_wrench=gravity.tolist(),
            derivative_sensitivity_wrench_difference=(load(kin2) - desired).tolist(),
            envelopes={},
        )
        for mode in modes:
            scope = (
                "feedback_inner"
                if mode == "feedback_inner"
                else "feedback_outer"
                if mode == "feedback_outer"
                else "motor"
            )
            caps = {
                c["group"]: (25.0 if mode == "motor_25N" else 10.0) for c in contacts
            }
            for demand, target, offset, state in [
                ("gravity", gravity, static_offset, qs),
                ("reference", desired, dynamic_offset, d),
            ]:
                lo, hi, nom, gain, bias = torque_bounds(
                    m,
                    state,
                    layout,
                    report["records"][i],
                    scope,
                    report["live_finger_feedback_settings"]["max_joint_offset_rad"],
                )
                entry = {}
                for cone in ("inner", "outer"):
                    problem = build(contacts, offset, lo, hi, caps, cone)
                    minimum = solve(
                        problem, target, time_limit_s=config["lp_time_limit_s"]
                    )
                    capacity = solve(
                        problem,
                        target,
                        capacity=True,
                        time_limit_s=config["lp_time_limit_s"],
                    )
                    if minimum.get("verified"):
                        tau = np.array(minimum["actuator_torque_nm"])
                        control = (tau - bias) / gain
                        norms = {
                            label: float(np.linalg.norm(control[idx] - nom[idx]))
                            for label, idx in layout["members"].items()
                        }
                        minimum["feedback_offset_norms_rad"] = norms
                        minimum["actual_feedback_ball_satisfied"] = (
                            max(norms.values())
                            <= report["live_finger_feedback_settings"][
                                "max_joint_offset_rad"
                            ]
                            + 1e-6
                        )
                        hard = np.maximum(
                            abs(m.actuator_forcerange[layout["ids"], 0]),
                            abs(m.actuator_forcerange[layout["ids"], 1]),
                        )
                        minimum["peak_hard_torque_utilization"] = float(
                            np.max(abs(tau) / hard)
                        )
                    key = f"{mode}/{demand}/{cone}"
                    if abs(times[i] - config["witness_time_s"]) < 0.011:
                        witnesses[key] = {"minimum": minimum, "capacity": capacity}

                    def compact(x):
                        return {
                            k: v
                            for k, v in x.items()
                            if k
                            not in ("actuator_torque_nm", "local_contact_components")
                        }

                    entry[cone] = {
                        "minimum": compact(minimum),
                        "capacity": compact(capacity),
                    }
                row["envelopes"][mode + "/" + demand] = entry
        rows.append(row)
        if order % 10 == 0:
            print(json.dumps({"samples": order + 1, "time_s": times[i]}), flush=True)
    sources = {
        str(p): digest(p)
        for p in [
            path,
            path.with_name("scene.xml"),
            path.with_name("states.npz"),
            Path(report["config"]["audit"]),
            parent_path,
            Path(parent["configuration"]["reference_guidance"]["path"]),
            Path(__file__),
            Path(__file__).with_name("wrench_feasibility.py"),
        ]
    }
    result = {
        "schema": "grasp-wrench-audit/1",
        "config": config,
        "rows": rows,
        "sources": sources,
        "verified_ancestry_sources": audit["sources"],
        "actuator_names": [m.actuator(a).name for a in layout["ids"]],
        "object_mass_kg": float(m.body_mass[obj]),
        "source_success": report["success"],
        "source_rejected_for": report["rejected_for"],
        "frame": "world force N and moment Nm about actual object COM",
        "dynamic_robot_background": "measured non-hand constraints held fixed; instantaneous conditional capacity",
        "gravity_robot_background": "quasistatic robot gravity/passive only, without other constraint assistance",
        "contact_model": "inner/outer polyhedral bounds of model elliptic contact cone, including torsional/rolling components",
        "normal_force_caps_are_physical_hardware_limits": False,
        "capacity_with_external_support": "counterfactual hand-only capacity at the observed contact geometry; external wrench is not granted to the LP",
        "trajectory_feasibility_proven": False,
        "controller_success_proven": False,
        "reference_retimed": False,
        "physics_replayed": False,
        "training_run": False,
    }
    write_json(out / "audit.json", result)
    write_json(out / "witnesses.json", witnesses)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.worker:
        run(config)
    else:
        from real2sim2real.bounded_process import run as bounded

        bounded(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--config",
                str(args.config.resolve()),
                "--worker",
            ],
            timeout=min(900, config["deadline_s"]),
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
