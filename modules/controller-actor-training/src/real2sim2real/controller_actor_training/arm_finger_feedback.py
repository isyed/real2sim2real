"""Bind an admitted bilateral finger-pressure rule to assembled robot arms.

The arm controller keeps wrist ownership. Finger waypoints are reconstructed
before pressure correction, rather than adding feedback to archived corrections.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest


def feedback_settings(original, overrides=None):
    from .contact_force_control import validate_settings

    if overrides and set(overrides) - {
        "normal_observation",
        "minimum_observed_force_n",
        "contact_anchor",
        "loaded_tangent_gap_m",
        "close_activation_fraction",
        "gain_m_per_n_s",
        "max_normal_speed_m_s",
    }:
        raise ValueError(
            "Arm feedback overrides may change observations, seating rate or retention rules only"
        )
    result = {**deepcopy(original), **deepcopy(overrides or {})}
    validate_settings(result)
    return result


def finger_waypoint(time, pre, opened, closed, rollout, *, release_open=None):
    from .object_grasp_pilot import smoothstep

    values = [np.asarray(v, float) for v in (pre, opened, closed)]
    approach, close = rollout["approach_s"], rollout["close_s"]
    if (
        not np.isfinite([time, approach, close]).all()
        or time < 0
        or approach <= 0
        or close <= 0
        or any(v.shape != values[0].shape or not np.isfinite(v).all() for v in values)
    ):
        raise ValueError(
            "Finite common finger waypoints and positive phase durations required"
        )
    if time < approach:
        a = smoothstep(time / approach)
        return (1 - a) * values[0] + a * values[1]
    if time < approach + close:
        a = smoothstep((time - approach) / close)
        return (1 - a) * values[1] + a * values[2]
    if rollout.get("finger_release"):
        from .finger_release import release_fraction

        a = release_fraction(time, rollout["finger_release"])
        released = (
            values[1] if release_open is None else np.asarray(release_open, float)
        )
        if released.shape != values[1].shape or not np.isfinite(released).all():
            raise ValueError("Finite matching release posture required")
        return (1 - a) * values[2] + a * released
    return values[2].copy()


class ArmFingerFeedback:
    def __init__(
        self, model, data, parent, roots, mesh, finger_actuators, *, overrides=None
    ):
        import mujoco
        from real2sim2real.retargeting.grasp_kinematics import robot_indices
        from .object_grasp_pilot import load_plan
        from .contact_force_control import BilateralContactForce

        config = parent["configuration"]
        force = config.get("contact_force_feedback")
        if (
            not force
            or force.get("posture_transition")
            or config.get("finger_closure")
            or config.get("finger_surface_feedback")
            or config.get("contact_feedback")
            or config["rollout"]["close_offset_rad"] != 0
            or config["support_controller"].get("mode") != "simultaneous_grasp"
        ):
            raise ValueError(
                "Live arm binding requires the admitted simultaneous pad-pressure rule without extra posture/closure controllers"
            )
        force = feedback_settings(force, overrides)
        plan = load_plan(config)
        if (
            parent["candidate"] not in plan["candidates"]
            or not parent["candidate"]["accepted"]
        ):
            raise ValueError(
                "Live finger controller must use the unchanged admitted candidate"
            )
        self.rollout = config["rollout"]
        from .finger_release import FingerRelease

        self.release = (
            FingerRelease(self.rollout["finger_release"])
            if self.rollout.get("finger_release")
            else None
        )
        self.last_target = None
        self.settings = force
        candidate = deepcopy(parent["candidate"])
        candidate["root_body"] = model.body(roots["right"]).name
        if candidate["support"]["root_body"] != model.body(roots["left"]).name:
            raise ValueError("Mounted support-hand identity differs")
        pose = np.asarray(plan["settling"]["settled_world_from_object"], float)
        world_mesh = mesh.copy()
        world_mesh.apply_transform(pose)
        self.servo = BilateralContactForce(
            model, data, force, candidate, pose, world_mesh
        )
        self.hands = []
        owned = {int(q) for p in self.servo.pads for q in p["qids"]}
        covered = set()
        for role, hand in [("primary", candidate), ("secondary", candidate["support"])]:
            columns, qids = [], []
            for i, name in enumerate(hand["joint_names"]):
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if j >= 0 and int(model.jnt_qposadr[j]) in owned:
                    columns.append(i)
                    qids.append(int(model.jnt_qposadr[j]))
            if not qids or covered & set(qids):
                raise ValueError("Disjoint owned finger waypoint coordinates required")
            covered.update(qids)
            self.hands.append(
                dict(
                    role=role,
                    qids=np.array(qids),
                    waypoints=[
                        np.asarray(hand[key], float)[columns]
                        for key in ("q_pregrasp", "q_open", "q_closed")
                    ],
                )
            )
        if covered != owned:
            raise ValueError("Source waypoints must cover every controlled finger")
        self.actuators = np.asarray(sorted(finger_actuators), int)
        self.qids = np.array(
            [
                int(model.jnt_qposadr[int(model.actuator_trnid[a, 0])])
                for a in self.actuators
            ]
        )
        if set(self.qids) != owned or len(self.qids) != len(owned):
            raise ValueError("Live feedback may own existing finger actuators only")
        self.support_bias = config["support_controller"].get(
            "actuator_bias_compensation", False
        )
        self.support_indices = robot_indices(model, model.body(roots["left"]).name)
        self.source_hashes = {
            str(Path(config["output"]) / "settling.json"): digest(
                Path(config["output"]) / "settling.json"
            ),
            str(
                Path(config["output"])
                / config.get("planning_directory", "planning")
                / "plan.json"
            ): digest(
                Path(config["output"])
                / config.get("planning_directory", "planning")
                / "plan.json"
            ),
        }

    def command(
        self, model, data, time, dt, *, pressure_offsets=None, table_observation=None
    ):
        from .support_control import bias_compensated_target

        nominal = data.qpos.copy()
        schedule = {k: v for k, v in self.rollout.items() if k != "finger_release"}
        for hand in self.hands:
            nominal[hand["qids"]] = finger_waypoint(time, *hand["waypoints"], schedule)
        activation = (
            self.rollout["approach_s"]
            + self.settings["close_activation_fraction"] * self.rollout["close_s"]
        )
        extra = {}
        if pressure_offsets is not None:
            extra["pressure_offsets"] = pressure_offsets
        if self.settings.get("contact_anchor"):
            extra["retention_active"] = bool(
                time
                >= self.rollout["approach_s"]
                + self.rollout["close_s"]
                + self.rollout["lift_s"]
            )
        release = getattr(self, "release", None)
        opening = None
        if release and table_observation is not None:
            opened = nominal.copy()
            for hand in self.hands:
                opened[hand["qids"]] = hand.get("release_open", hand["waypoints"][1])
            held = self.last_target if self.last_target is not None else nominal
            opening = release.command(
                time,
                dt,
                held[self.qids],
                opened[self.qids],
                data.qpos[self.qids],
                **table_observation,
            )
        target = self.servo.target(
            model,
            data,
            nominal,
            active=time >= activation and opening is None,
            dt=dt,
            **extra,
        )
        if opening is not None:
            target[self.qids] = opening
            for pad in self.servo.pads:
                pad["target_n"] = 0.0
                pad["offset"][:] = 0.0
        self.last_target = target.copy()
        if self.support_bias:
            target = bias_compensated_target(model, data, target, self.support_indices)
        data.ctrl[self.actuators] = target[self.qids]

    def diagnostics(self):
        result = self.servo.diagnostics()
        if self.release:
            result["finger_release"] = self.release.diagnostics()
        return result
