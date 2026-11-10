"""Staged second-hand insertion and measured load sharing, without object aids."""

import numpy as np

from .object_grasp_pilot import set_root_frame, smoothstep


def bias_compensated_target(model, data, target, indices):
    """Feed robot bias through position actuators, retaining their force limits.

    The shift supplies gravity/Coriolis bias via the existing actuator only;
    it never writes applied forces or modifies the object's dynamics.
    """
    result = np.asarray(target, float).copy()
    if (
        not np.isfinite(result).all()
        or not np.isfinite(data.qfrc_bias[indices["dofs"]]).all()
    ):
        raise ValueError("Finite robot targets and bias required")
    joints = set(indices["joints"])
    for actuator, joint in enumerate(model.actuator_trnid[:, 0]):
        if joint not in joints:
            continue
        gain = model.actuator_gainprm[actuator, 0]
        if gain <= 0 or not np.isclose(model.actuator_biasprm[actuator, 1], -gain):
            raise ValueError("Bias compensation requires a position actuator")
        if not model.actuator_forcelimited[actuator]:
            raise ValueError("Bias compensation requires finite actuator force limits")
        if not model.actuator_ctrllimited[actuator]:
            raise ValueError("Bias compensation requires bounded position commands")
        qid, dof = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        value = result[qid] + data.qfrc_bias[dof] / gain
        result[qid] = np.clip(value, *model.actuator_ctrlrange[actuator])
    return result


def contact_wrench_on_body(model, data, contact_id, body_id):
    """World force/torque on one body at a contact, with MuJoCo side convention."""
    import mujoco

    contact = data.contact[contact_id]
    sides = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
    if body_id not in sides or sides[0] == sides[1]:
        return np.zeros(3), np.zeros(3)
    local = np.zeros(6)
    mujoco.mj_contactForce(model, data, contact_id, local)
    sign = 1.0 if sides[1] == body_id else -1.0
    frame = np.asarray(contact.frame).reshape(3, 3)
    force = sign * (frame.T @ local[:3])
    torque = sign * (frame.T @ local[3:]) + np.cross(
        contact.pos - data.xipos[body_id], force
    )
    return force, torque


def support_load_offset(offset, upward_force_n, weight_n, dt, settings):
    """Bounded vertical robot command from signed measured object support."""
    values = [
        offset,
        upward_force_n,
        weight_n,
        dt,
        settings["target_weight_fraction"],
        settings["gain_m_per_n_s"],
        settings["max_speed_m_s"],
        settings["max_offset_m"],
    ]
    if (
        not np.isfinite(values).all()
        or weight_n <= 0
        or dt <= 0
        or offset < 0
        or not 0 < settings["target_weight_fraction"] <= 1
        or not 0 < settings["max_offset_m"] <= 0.02
        or settings["gain_m_per_n_s"] <= 0
        or not 0 < settings["max_speed_m_s"] <= 0.02
    ):
        raise ValueError("Finite bounded support force feedback required")
    error = settings["target_weight_fraction"] * weight_n - upward_force_n
    speed = np.clip(
        settings["gain_m_per_n_s"] * error,
        -settings["max_speed_m_s"],
        settings["max_speed_m_s"],
    )
    return float(np.clip(offset + dt * speed, 0, settings["max_offset_m"]))


def simultaneous_target(
    time, pre, opened, closed, xyz, primary_z, primary_closed_z, rollout
):
    """Close both hands on the same clock and share the primary's lift command."""
    from .object_grasp_pilot import target_at

    q = [np.asarray(v, float) for v in (pre, opened, closed)]
    if (
        not np.isfinite([time, primary_z, primary_closed_z]).all()
        or time < 0
        or any(v.shape != q[0].shape or not np.isfinite(v).all() for v in q)
        or len(xyz) != 3
        or len(set(xyz)) != 3
    ):
        raise ValueError(
            "Finite synchronous commands and distinct world translation indices required"
        )
    target, phase = target_at(time, *q, xyz[2], rollout)
    if phase in ("lift", "hold"):
        target[xyz[2]] = q[2][xyz[2]] + primary_z - primary_closed_z
    return target, phase


class SupportMotion:
    """Park during the initial lift, insert, seat, then share vertical commands.

    A bounded translation aligns the support plan once to the observed lifted
    object. It is frozen after entry; the hand does not chase a falling object.
    Thereafter both wrists share changes in the primary's vertical command.
    """

    def __init__(self, model, data, config, candidate):
        from real2sim2real.retargeting.grasp_kinematics import robot_indices

        self.plan = candidate["support"]
        robot = config["support_robot"]
        if (
            self.plan["root_body"] != robot["root_body"]
            or self.plan["fixture_prefix"] != robot["fixture_prefix"]
        ):
            raise ValueError("Support plan/robot mismatch")
        set_root_frame(
            model,
            data,
            robot["root_body"],
            robot["fixture_prefix"],
            self.plan["base_position_m"],
            self.plan["base_rotation"],
        )
        self.idx = robot_indices(model, robot["root_body"])
        self.settings = config["support_controller"]
        values = [self.settings[k] for k in ("approach_s", "seat_s", "max_alignment_m")]
        if (
            not np.isfinite(values).all()
            or min(values) <= 0
            or self.settings["max_alignment_m"] > 0.05
        ):
            raise ValueError("Positive bounded support timings/alignment required")
        self.enabled = self.settings.get("enabled", True)
        self.start = sum(
            config["rollout"][k] for k in ("approach_s", "close_s", "lift_s")
        )
        self.duration = self.settings["approach_s"] + self.settings["seat_s"]
        self.mode = self.settings.get("mode", "lift_then_insert")
        if self.mode not in ("lift_then_insert", "simultaneous_grasp"):
            raise ValueError("Unknown second-hand motion mode")
        self.rollout = config["rollout"]
        self.load_start = self.start + self.settings["approach_s"]
        if getattr(self, "mode", "lift_then_insert") == "simultaneous_grasp":
            self.primary_closed_z = candidate.get("q_closed", candidate["q_grasp"])[
                candidate["joint_names"].index("fixture_z")
            ]
            if not self.enabled:
                raise ValueError("Simultaneous grasp requires two enabled hands")
            self.start, self.duration = 0.0, 0.0
            self.load_start = sum(
                self.rollout[k] for k in ("approach_s", "close_s", "lift_s")
            )

        self.xyz = [
            self.plan["joint_names"].index(robot["fixture_prefix"] + a) for a in "xyz"
        ]
        self.limits = model.jnt_range[self.idx["joints"]]
        self.alignment = None
        self.anchor_z = None
        self.rejection = None
        self.load_offset = 0.0

    def update_load(self, time, upward_force_n, weight_n, dt):
        feedback = self.settings.get("load_feedback")
        if (
            feedback
            and self.enabled
            and not self.rejection
            and time
            >= getattr(self, "load_start", self.start + self.settings["approach_s"])
        ):
            self.load_offset = support_load_offset(
                self.load_offset, upward_force_n, weight_n, dt, feedback
            )

    def phase(self, time):
        if getattr(self, "mode", "lift_then_insert") == "simultaneous_grasp":
            if time < self.rollout["approach_s"]:
                return "approach"
            if time < self.rollout["approach_s"] + self.rollout["close_s"]:
                return "close"
            if time < self.load_start:
                return "lift"
            return "hold"
        if time < self.start:
            return None
        if time < self.start + self.settings["approach_s"]:
            return "support_approach"
        if time < self.start + self.duration:
            return "support_seat"
        return "hold"

    def target(self, time, object_transform, primary_z):
        pre, opened, closed = [
            np.asarray(self.plan[k], float).copy()
            for k in ("q_pregrasp", "q_open", "q_grasp")
        ]
        if time < self.start or not self.enabled or self.rejection:
            return pre
        if getattr(self, "mode", "lift_then_insert") == "simultaneous_grasp":
            closed = np.asarray(
                self.plan.get("q_closed", self.plan["q_grasp"]), float
            ).copy()
            target, phase = simultaneous_target(
                time,
                pre,
                opened,
                closed,
                self.xyz,
                primary_z,
                self.primary_closed_z,
                self.rollout,
            )
            if phase == "hold":
                target[self.xyz[2]] += self.load_offset
            if (target < self.limits[:, 0]).any() or (target > self.limits[:, 1]).any():
                self.rejection = "support_joint_command_limit"
            return np.clip(target, self.limits[:, 0], self.limits[:, 1])
        if self.alignment is None:
            self.alignment = (
                object_transform[:3, 3]
                - np.asarray(self.plan["reference_world_from_object"])[:3, 3]
            )
            self.anchor_z = primary_z
            if np.linalg.norm(self.alignment) > self.settings["max_alignment_m"]:
                self.rejection = "support_alignment_out_of_bounds"
                return pre
        shift = self.alignment.copy()
        shift[2] += primary_z - self.anchor_z
        shift[2] += getattr(self, "load_offset", 0.0)
        opened[self.xyz] += shift
        closed[self.xyz] += shift
        t = time - self.start
        if t < self.settings["approach_s"]:
            a = smoothstep(t / self.settings["approach_s"])
            target = (1 - a) * pre + a * opened
        else:
            a = smoothstep((t - self.settings["approach_s"]) / self.settings["seat_s"])
            target = (1 - a) * opened + a * closed
        if (target < self.limits[:, 0]).any() or (target > self.limits[:, 1]).any():
            self.rejection = "support_joint_command_limit"
        return np.clip(target, self.limits[:, 0], self.limits[:, 1])


def support_hold_metrics(records, object_weight_n):
    """Actual upward object support, not normal-force sums or target counts."""
    if not np.isfinite(object_weight_n) or object_weight_n <= 0:
        raise ValueError("Positive finite object weight required")
    hold = [r for r in records if r["phase"] == "hold"]
    if not hold:
        return dict(
            support_load_fraction=0.0,
            support_mean_upward_force_n=0.0,
            support_wrist_drift_m=None,
        )
    forces = np.asarray([r["support_object_force_world_n"][2] for r in hold])
    if not np.isfinite(forces).all():
        raise ValueError("Finite measured support forces required")
    from .grasp_feedback import relative_translation_drift

    drift = relative_translation_drift(
        [r["object_position_m"] for r in hold],
        [r["support_wrist_position_m"] for r in hold],
        [r["support_wrist_rotation"] for r in hold],
    )
    from .grasp_hold import contact_support

    groups = (
        contact_support(
            [
                dict(
                    normal_contact_force_by_group_n=r[
                        "support_normal_contact_force_by_group_n"
                    ]
                )
                for r in hold
            ]
        )
        if all("support_normal_contact_force_by_group_n" in r for r in hold)
        else {}
    )
    return dict(
        support_load_fraction=float((forces >= 0.1 * object_weight_n).mean()),
        support_mean_upward_force_n=float(forces.mean()),
        support_wrist_drift_m=drift,
        support_minimum_upward_force_n=float(forces.min()),
        support_object_weight_n=float(object_weight_n),
        support_contact_support=groups,
    )


def support_rejections(metrics, rejection=None):
    if (
        not np.isfinite(metrics["support_load_fraction"])
        or not 0 <= metrics["support_load_fraction"] <= 1
        or (
            metrics["support_wrist_drift_m"] is not None
            and not np.isfinite(metrics["support_wrist_drift_m"])
        )
    ):
        return ["invalid_second_hand_evidence"]
    reasons = []
    if metrics["support_load_fraction"] < 0.8:
        reasons.append("insufficient_second_hand_load_sharing")
    if (
        metrics["support_wrist_drift_m"] is None
        or metrics["support_wrist_drift_m"] > 0.01
    ):
        reasons.append("support_wrist_drift")
    if rejection:
        reasons.append(rejection)
    return reasons
