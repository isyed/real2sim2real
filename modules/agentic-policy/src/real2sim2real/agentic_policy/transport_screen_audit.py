"""Read-only comparison of stationary versus carried-object collision assumptions."""

import numpy as np


def contact_groups(session, data, *, stationary_pad_ids=()):
    """Classify cached contacts, separating explicitly identified pad contacts."""
    from real2sim2real.controller_actor_training.arm_path_audit import joint_neighbors
    from real2sim2real.controller_actor_training.mujoco_step_trace import (
        audit_contact_geometry,
    )

    s, m = session, session.model
    if audit_contact_geometry(m, data)["anomalies"]:
        raise ValueError("Implausible contact geometry in transport screen")
    groups = {
        name: {"peak_penetration_m": 0.0, "witness": None}
        for name in [
            "robot_object",
            "object_environment",
            "robot_environment",
            "robot_self",
        ]
    }
    if stationary_pad_ids:
        groups["stationary_pad_contacts"] = {"peak_penetration_m": 0.0, "witness": None}
    for contact in data.contact:
        a, b = int(contact.geom1), int(contact.geom2)
        if min(a, b) < 0:
            raise ValueError("Rigid geom contacts required")
        ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
        ra, rb = ba in s.robot, bb in s.robot
        if ba == bb or (ra and rb and joint_neighbors(m, ba, bb)):
            continue
        if s.object in [ba, bb]:
            other = b if ba == s.object else a
            group = "robot_object" if ra or rb else "object_environment"
            if group == "robot_object" and other in stationary_pad_ids:
                group = "stationary_pad_contacts"
        elif ra and rb:
            group = "robot_self"
        elif ra or rb:
            group = "robot_environment"
        else:
            continue
        depth = max(0.0, -float(contact.dist))
        if not np.isfinite(contact.dist):
            raise ValueError("Nonfinite contact distance")
        if depth > groups[group]["peak_penetration_m"]:
            groups[group] = {
                "peak_penetration_m": depth,
                "witness": [m.geom(a).name, m.geom(b).name],
            }
    return groups


def audit_transport_screen(
    session, positions, durations_s, *, stationary_pad_ids=(), carried_pad_ids=()
):
    """Report collision groups along the same sampled joint path in scratch data.

    This does not admit or execute a path. A rigidly carried object is a
    hypothesis; contact retention still requires separate loaded evidence.
    """
    s, m, mj = session, session.model, session.mj
    if not positions or len(positions) != len(durations_s):
        raise ValueError("Matching nonempty positions and durations required")
    durations = np.asarray(durations_s, float)
    if not np.isfinite(durations).all() or (durations <= 0).any():
        raise ValueError("Positive finite durations required")
    if (
        not np.isfinite(s.feedback)
        or s.feedback <= 0
        or not np.allclose(
            durations / s.feedback, np.rint(durations / s.feedback), rtol=0, atol=1e-8
        )
    ):
        raise ValueError("Durations must align to positive feedback ticks")
    goals = [s._vector(q) for q in positions]
    if (set(stationary_pad_ids) | set(carried_pad_ids)) - set(
        getattr(s, "pad_ids", [])
    ):
        raise ValueError("Only configured pads can be classified as intended contacts")
    object_joint = int(m.body_jntadr[s.object])
    if (
        object_joint < 0
        or m.jnt_type[object_joint] != mj.mjtJoint.mjJNT_FREE
        or m.body_parentid[s.object] != 0
    ):
        raise ValueError("A free object is required")
    address = int(m.jnt_qposadr[object_joint])
    origin = mj.MjData(m)
    origin.qpos[:] = s.data.qpos
    mj.mj_fwdPosition(m, origin)

    def pose(data, body):
        value = np.eye(4)
        value[:3, :3] = data.xmat[body].reshape(3, 3)
        value[:3, 3] = data.xpos[body]
        return value

    relative = np.linalg.inv(pose(origin, s.wrist)) @ pose(origin, s.object)
    result = {}
    for mode in ["stationary_object", "rigidly_carried_object"]:
        scratch = mj.MjData(m)
        scratch.qpos[:] = s.data.qpos
        excluded = (
            tuple(stationary_pad_ids)
            if mode == "stationary_object"
            else tuple(carried_pad_ids)
        )
        groups = contact_groups(s, scratch, stationary_pad_ids=excluded)
        start = s.nominal.copy()
        elapsed = 0.0
        for goal, duration in zip(goals, durations):
            for u in np.linspace(0, 1, round(duration / s.feedback) + 1):
                scratch.qpos[s.qids] = start + (goal - start) * u * u * (3 - 2 * u)
                mj.mj_fwdPosition(m, scratch)
                if mode == "rigidly_carried_object":
                    obj = pose(scratch, s.wrist) @ relative
                    scratch.qpos[address : address + 3] = obj[:3, 3]
                    mj.mju_mat2Quat(
                        scratch.qpos[address + 3 : address + 7], obj[:3, :3].flatten()
                    )
                    mj.mj_fwdPosition(m, scratch)
                sample = contact_groups(s, scratch, stationary_pad_ids=excluded)
                for group, value in sample.items():
                    if (
                        value["peak_penetration_m"]
                        > groups[group]["peak_penetration_m"]
                    ):
                        groups[group] = value | {
                            "path_time_s": elapsed + float(u * duration)
                        }
            start = goal
            elapsed += duration
        if mode == "rigidly_carried_object" and carried_pad_ids:
            groups["carried_pad_contacts"] = groups.pop("stationary_pad_contacts")
        result[mode] = groups
    return {
        "groups": result,
        "sample_s": s.feedback,
        "live_physics_steps": 0,
        "path_admitted": False,
        "interpretation": "Diagnostic geometry only; carried pose assumes retained contact and is not a certificate of motion safety.",
    }
