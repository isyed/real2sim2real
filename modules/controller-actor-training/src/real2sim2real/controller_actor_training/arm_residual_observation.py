"""Causal, explicitly privileged observations from a separate MuJoCo data object."""

import numpy as np
from scipy.spatial.transform import Rotation


def relative_position(position, wrist_position, wrist_rotation):
    return np.asarray(wrist_rotation).T @ (np.asarray(position) - wrist_position)


class ArmResidualObservation:
    def __init__(self, model, data, fingers, roots, parent, times):
        import mujoco

        self.model, self.live = model, data
        self.scratch = mujoco.MjData(model)
        self.fingers, self.roots = fingers, roots
        self.parent, self.times = parent, times
        self.object_id = model.body("task_object").id
        self.anchors = None
        self.phases = list(dict.fromkeys(r["phase"] for r in parent["records"]))
        self.joints = [
            j
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self.qids = model.jnt_qposadr[self.joints]
        self.vids = model.jnt_dofadr[self.joints]
        self.schema = None

    def record(self, record):
        # Identical hold-entry samples to the independent frozen evaluator.
        if self.anchors is None and record["phase"] == "hold":
            self.anchors = [
                relative_position(
                    record["object_position_m"],
                    record[p + "wrist_position_m"],
                    record[p + "wrist_rotation"],
                )
                for p in ("", "support_")
            ]

    def capture(self, time, i, alpha, residual):
        import mujoco
        from .contact_force_control import object_pad_observations
        from .support_control import contact_wrench_on_body

        m, d, live = self.model, self.scratch, self.live
        for field in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
            getattr(d, field)[:] = getattr(live, field)
        d.time = live.time
        mujoco.mj_forward(m, d)
        obj = self.object_id
        position, rotation = d.xpos[obj].copy(), d.xmat[obj].reshape(3, 3).copy()
        source, following = self.parent["records"][i : i + 2]
        valid_reference = all("reference_position_m" in r for r in (source, following))
        position_error, rotation_error = np.zeros(3), np.zeros(3)
        if valid_reference:
            goal = (1 - alpha) * np.asarray(
                source["reference_position_m"]
            ) + alpha * np.asarray(following["reference_position_m"])
            first = np.asarray(source["reference_rotation"])
            delta = Rotation.from_matrix(
                np.asarray(following["reference_rotation"]) @ first.T
            ).as_rotvec()
            goal_rotation = Rotation.from_rotvec(alpha * delta).as_matrix() @ first
            position_error = position - goal
            rotation_error = Rotation.from_matrix(
                rotation @ goal_rotation.T
            ).as_rotvec()
        relative, drift = [], []
        for k, role in enumerate(("right", "left")):
            root = self.roots[role]
            value = relative_position(
                position, d.xpos[root], d.xmat[root].reshape(3, 3)
            )
            relative.extend(value)
            drift.extend(
                value - self.anchors[k] if self.anchors is not None else np.zeros(3)
            )
        pads = self.fingers.servo.pads
        observed = object_pad_observations(m, d, obj, [p["gid"] for p in pads])
        force = np.array([observed[p["gid"]]["force_n"] for p in pads])
        upward = np.zeros(2)
        for cid, contact in enumerate(d.contact):
            pair = {int(contact.geom1), int(contact.geom2)}
            if not any(m.geom_bodyid[g] == obj for g in pair):
                continue
            for k, role in enumerate(("primary", "secondary")):
                if pair & self.fingers.servo.role_geoms[role]:
                    upward[k] += contact_wrench_on_body(m, d, cid, obj)[0][2]
        payload = self.fingers.servo.payload
        payload_state = (
            np.r_[
                payload.weight_n,
                payload.secondary_fraction,
                payload.scales,
                payload.upward_n,
            ]
            if payload
            else np.zeros(6)
        )
        parts = {
            "elapsed_s": [float(live.time) - self.times[0]],
            "scheduled_reference_time_s": [time],
            "phase_one_hot": [float(source["phase"] == p) for p in self.phases],
            "joint_position_rad": d.qpos[self.qids],
            "joint_velocity_rad_s": d.qvel[self.vids],
            "previous_actuator_command": live.ctrl.copy(),
            "object_position_error_m": position_error,
            "object_rotation_error_rad": rotation_error,
            "reference_valid": [float(valid_reference)],
            "object_position_in_wrists_m": relative,
            "grasp_drift_in_wrists_m": drift,
            "grasp_anchor_valid": [float(self.anchors is not None)],
            "pad_normal_force_n": force,
            "previous_pad_target_n": [
                p.get("target_n", self.fingers.settings["target_force_n"][p["group"]])
                for p in pads
            ],
            "finger_servo_offset_rad": np.concatenate([p["offset"] for p in pads]),
            "upward_hand_force_n": upward,
            "payload_filter_state": payload_state,
            "previous_requested_action": residual.requested,
            "applied_pressure_offset_n": residual.offset,
        }
        schema, offset = {}, 0
        for name, values in parts.items():
            schema[name] = [offset, offset + len(values)]
            offset += len(values)
        self.schema = schema
        vector = np.concatenate([np.asarray(v, float) for v in parts.values()])
        if not np.isfinite(vector).all():
            raise ValueError("Nonfinite residual observation")
        hold = source["phase"] == "hold"
        contact_cost = 0.0
        for role in ("primary", "secondary"):
            thumb = sum(
                force[j]
                for j, p in enumerate(pads)
                if p["role"] == role and p["group"] == "thumb"
            )
            other = sum(
                force[j]
                for j, p in enumerate(pads)
                if p["role"] == role and p["group"] != "thumb"
            )
            contact_cost += float(thumb <= 0.1) + float(other <= 0.1)
        costs = dict(
            position=float(np.dot(position_error, position_error) / 0.02**2)
            if hold and valid_reference
            else 0.0,
            rotation=float(np.dot(rotation_error, rotation_error) / np.deg2rad(10) ** 2)
            if hold and valid_reference
            else 0.0,
            grasp=float(np.dot(drift, drift) / 0.01**2)
            if hold and self.anchors is not None
            else 0.0,
            contact=contact_cost if hold else 0.0,
            excess_force=float(np.sum(np.maximum(force - 10, 0) ** 2) / 10**2),
            residual=float(
                np.mean((residual.offset / residual.settings["max_offset_n"]) ** 2)
            ),
        )
        return dict(
            observation=vector,
            time_s=float(live.time),
            costs=costs,
            phase=source["phase"],
        )


def transition_reward(previous, current, *, horizon_s, success=None, failure=False):
    """Time-normalized bounded cost; hard acceptance always comes from replay."""
    dt = current["time_s"] - previous["time_s"]
    if not np.isfinite([dt, horizon_s]).all() or dt <= 0 or horizon_s <= 0:
        if failure:
            return -110.0
        raise ValueError("Positive transition and horizon durations required")
    weights = dict(
        position=4, rotation=2, grasp=6, contact=2, excess_force=1, residual=0.1
    )
    cost = sum(weights[k] * current["costs"][k] for k in weights)
    if not np.isfinite(cost) or cost < 0:
        raise ValueError("Finite nonnegative learning costs required")
    # Dense cost is bounded by 100 over a full horizon; a failed complete trial
    # also costs 10. Cover both so early failure cannot escape remaining cost.
    reward = -min(cost, 100.0) * dt / horizon_s
    if failure:
        reward -= 110.0
    elif success is not None:
        reward += 10.0 if success else -10.0
    return float(reward)
