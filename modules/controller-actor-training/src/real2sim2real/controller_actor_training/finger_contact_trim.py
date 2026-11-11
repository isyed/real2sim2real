"""Bounded recovery of a declared finger contact through owned joint targets."""

import numpy as np
from .finger_closure import FLEXION_SUFFIXES


class FingerContactTrim:
    """Integrate force deficit only while the opposing contact retains the object.

    The direction is an explicit robot-specific hypothesis, not inferred from
    force magnitude. Existing joint/actuator force limits and task gates remain
    unchanged. Measurements in the simulation adapter are privileged.
    """

    def __init__(self, settings, names, ranges):
        required = {
            "joint_directions",
            "contact_group",
            "target_force_n",
            "minimum_anchor_force_n",
            "gain_rad_per_n_s",
            "max_speed_rad_s",
            "max_offset_rad",
            "maximum_reference_error_m",
        }
        if set(settings) != required or settings["contact_group"] not in {
            "index",
            "middle",
            "ring",
            "pinky",
        }:
            raise ValueError(
                "Explicit finger contact and bounded recovery settings required"
            )
        bounds = {
            "target_force_n": 5,
            "minimum_anchor_force_n": 10,
            "gain_rad_per_n_s": 0.1,
            "max_speed_rad_s": 0.2,
            "max_offset_rad": 0.2,
            "maximum_reference_error_m": 0.02,
        }
        if any(
            not np.isfinite(settings[k]) or not 0 < settings[k] <= limit
            for k, limit in bounds.items()
        ):
            raise ValueError("Finger recovery exceeds finite control bounds")
        directions = settings["joint_directions"]
        if (
            not directions
            or set(directions) - set(names)
            or any(not n.endswith(FLEXION_SUFFIXES) for n in directions)
        ):
            raise ValueError("Only declared owned finger flexion joints may be trimmed")
        self.names = list(directions)
        self.direction = np.asarray(list(directions.values()), float)
        if (
            not np.isfinite(self.direction).all()
            or np.linalg.norm(self.direction) < 1e-8
        ):
            raise ValueError("Finite nonzero closure direction required")
        self.direction /= np.linalg.norm(self.direction)
        self.ranges = np.asarray([ranges[n] for n in self.names], float)
        if (
            self.ranges.shape != (len(self.names), 2)
            or not np.isfinite(self.ranges).all()
            or (self.ranges[:, 1] <= self.ranges[:, 0]).any()
        ):
            raise ValueError("Finite ordered finger joint limits required")
        self.settings = settings
        self.offset = np.zeros(len(self.names))
        self.updated = self.frozen = 0

    def target(
        self, nominal, *, force_n, anchor_force_n, reference_error_m, active, dt
    ):
        q = np.asarray(nominal, float)
        values = np.array([force_n, anchor_force_n, reference_error_m, dt])
        if (
            q.shape != self.offset.shape
            or not np.isfinite(q).all()
            or not np.isfinite(values).all()
            or (values[:3] < 0).any()
            or dt <= 0
        ):
            raise ValueError(
                "Finite finger command, forces, reference error and positive clock required"
            )
        if (q < self.ranges[:, 0] - 1e-12).any() or (
            q > self.ranges[:, 1] + 1e-12
        ).any():
            raise ValueError(
                "Nominal finger commands must already satisfy the joint limits"
            )
        s = self.settings
        if (
            active
            and anchor_force_n >= s["minimum_anchor_force_n"]
            and reference_error_m <= s["maximum_reference_error_m"]
        ):
            speed = min(
                s["max_speed_rad_s"],
                s["gain_rad_per_n_s"] * max(0, s["target_force_n"] - force_n),
            )
            self.offset += self.direction * dt * speed
            norm = np.linalg.norm(self.offset)
            self.offset *= min(1, s["max_offset_rad"] / max(norm, 1e-12))
            self.updated += 1
        else:
            self.frozen += 1
        result = np.clip(q + self.offset, *self.ranges.T)
        self.offset = result - q
        return result

    def diagnostics(self):
        return dict(
            joint_offsets_rad=dict(zip(self.names, self.offset.tolist())),
            updated_steps=self.updated,
            frozen_steps=self.frozen,
        )


def bind(model, settings, owned_actuator_ids, primary_body):
    """Admit only unit-gear position actuators in the primary hand subtree."""
    import mujoco

    names, ids, ranges = [], {}, {}
    for aid in owned_actuator_ids:
        if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jid = int(model.actuator_trnid[aid, 0])
        body = int(model.jnt_bodyid[jid])
        while body and body != primary_body:
            body = int(model.body_parentid[body])
        if body != primary_body:
            continue
        name = model.joint(jid).name
        if name not in settings["joint_directions"]:
            continue
        if (
            model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE
            or not model.jnt_limited[jid]
            or not np.array_equal(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0])
            or model.actuator_gainprm[aid, 0] <= 0
            or model.actuator_biasprm[aid, 1] != -model.actuator_gainprm[aid, 0]
            or name in ids
        ):
            raise ValueError(
                "Finger trim requires unique unit-gear bounded position actuators"
            )
        limits = model.jnt_range[jid].copy()
        if model.actuator_ctrllimited[aid]:
            limits = [
                max(limits[0], model.actuator_ctrlrange[aid, 0]),
                min(limits[1], model.actuator_ctrlrange[aid, 1]),
            ]
        names.append(name)
        ids[name] = aid
        ranges[name] = limits
    controller = FingerContactTrim(settings, names, ranges)
    return controller, np.array([ids[n] for n in controller.names], int)
