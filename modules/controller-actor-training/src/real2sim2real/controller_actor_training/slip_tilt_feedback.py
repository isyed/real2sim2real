"""Fixed-anchor bimanual slip/tilt feedback; all outputs are robot target offsets."""

import numpy as np
from scipy.spatial.transform import Rotation

from .arm_object_feedback import rigid_pose
from .wrist_coordination import bounded_vector


class SlipTiltFeedback:
    def __init__(self, settings, protocol):
        bounds = dict(
            slip_gain_s_inv=(0, 3),
            tilt_gain=(0, 1),
            height_gain=(0, 1),
            filter_gain_s_inv=(0, 5),
            max_offset_m=(0, 0.006),
            max_angle_rad=(0, 0.06),
            max_speed_m_s=(0, 0.02),
            max_angular_speed_rad_s=(0, 0.2),
            minimum_contact_n=(0, 10),
            maximum_reference_error_m=(0, 0.02),
            maximum_reference_angle_rad=(0, np.deg2rad(10)),
            observation_period_s=(0.01, 0.1),
        )
        if set(settings) != set(bounds) or any(
            isinstance(settings[k], bool)
            or not np.isfinite(settings[k])
            or not lo <= settings[k] <= hi
            or (
                settings[k] == 0
                and k not in ("slip_gain_s_inv", "tilt_gain", "height_gain")
            )
            for k, (lo, hi) in bounds.items()
        ):
            raise ValueError("Explicit finite bounded slip/tilt settings required")
        gap = protocol["finger_release"]["start_s"] - protocol["capture_duration_s"]
        if gap <= 0 or (
            1.875 * settings["max_offset_m"] / gap > settings["max_speed_m_s"]
            or 1.875 * settings["max_angle_rad"] / gap
            > settings["max_angular_speed_rad_s"]
        ):
            raise ValueError(
                "Return interval must admit bounded smooth recovery before release"
            )
        self.settings, self.protocol = dict(settings), protocol
        self.anchors = None
        self.offsets = np.zeros((2, 6))
        self.integral = np.zeros((2, 3))
        self.recovery = None
        self.updated = np.zeros(2, dtype=int)
        self.frozen = np.zeros(2, dtype=int)
        self.blocked_downward = 0
        self.peak = np.zeros((2, 2))
        self.last_drift = np.zeros((2, 3))
        self.last_tilt = np.zeros(3)
        self.last_reference_valid = False

    def record(self, record):
        if (
            self.anchors is None
            and record["time_s"] >= self.protocol["acquisition_end_s"]
        ):
            obj = np.asarray(record["object_position_m"])
            self.anchors = np.array(
                [
                    np.asarray(record[p + "wrist_rotation"]).T
                    @ (obj - np.asarray(record[p + "wrist_position_m"]))
                    for p in ("", "support_")
                ]
            )

    def update(
        self,
        time_s,
        dt,
        actual_object,
        reference_object,
        actual_wrists,
        nominal_wrists,
        forces_n,
    ):
        from .object_grasp_pilot import smoothstep

        actual, reference = rigid_pose(actual_object), rigid_pose(reference_object)
        wrists = [rigid_pose(p) for p in actual_wrists]
        nominal = [rigid_pose(p) for p in nominal_wrists]
        forces = np.asarray(forces_n, float)
        if (
            len(wrists) != 2
            or len(nominal) != 2
            or forces.shape != (2,)
            or not np.isfinite(forces).all()
            or (forces < 0).any()
            or not np.isfinite([time_s, dt]).all()
            or dt <= 0
        ):
            raise ValueError(
                "Finite bilateral poses, nonnegative forces and positive clock required"
            )
        s, p = self.settings, self.protocol
        self.last_tilt = Rotation.from_matrix(
            actual[:3, :3] @ reference[:3, :3].T
        ).as_rotvec()
        error = actual[:3, 3] - reference[:3, 3]
        valid = (
            np.linalg.norm(error) <= s["maximum_reference_error_m"]
            and np.linalg.norm(self.last_tilt) <= s["maximum_reference_angle_rad"]
        )
        self.last_reference_valid = bool(valid)
        if (
            not p.get("release_disabled_for_stationary_diagnostic")
            and time_s >= p["capture_duration_s"]
        ):
            if self.recovery is None:
                self.recovery = self.offsets.copy()
            a = 1 - float(
                smoothstep(
                    np.clip(
                        (time_s - p["capture_duration_s"])
                        / (p["finger_release"]["start_s"] - p["capture_duration_s"]),
                        0,
                        1,
                    )
                )
            )
            self.offsets = self.recovery * a
            return self.offsets.copy()
        if self.anchors is None:
            return self.offsets.copy()
        both_contact = bool((forces >= s["minimum_contact_n"]).all())
        rotation_vector = (
            bounded_vector(-s["tilt_gain"] * self.last_tilt, s["max_angle_rad"])
            if both_contact
            else np.zeros(3)
        )
        correction_rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
        for k, wrist in enumerate(wrists):
            drift = wrist[:3, :3].T @ (actual[:3, 3] - wrist[:3, 3]) - self.anchors[k]
            self.last_drift[k] = drift
            if not valid or forces[k] < s["minimum_contact_n"]:
                self.frozen[k] += 1
                continue
            slip_world = wrist[:3, :3] @ drift
            if error[2] < 0 and slip_world[2] < 0:
                slip_world[2] = 0
                self.blocked_downward += 1
            self.integral[k] = bounded_vector(
                self.integral[k] + dt * s["slip_gain_s_inv"] * slip_world,
                s["max_offset_m"],
            )
            lever = nominal[k][:3, 3] - actual[:3, 3]
            desired = np.r_[
                self.integral[k]
                + (correction_rotation @ lever - lever)
                + [0, 0, s["height_gain"] * max(0, -error[2])],
                rotation_vector,
            ]
            for sl, cap, speed in (
                (slice(0, 3), s["max_offset_m"], s["max_speed_m_s"]),
                (slice(3, 6), s["max_angle_rad"], s["max_angular_speed_rad_s"]),
            ):
                goal = bounded_vector(desired[sl], cap)
                velocity = bounded_vector(
                    s["filter_gain_s_inv"] * (goal - self.offsets[k, sl]), speed
                )
                self.offsets[k, sl] = bounded_vector(
                    self.offsets[k, sl] + dt * velocity, cap
                )
            self.updated[k] += 1
        self.peak = np.maximum(
            self.peak,
            np.c_[
                np.linalg.norm(self.offsets[:, :3], axis=1),
                np.linalg.norm(self.offsets[:, 3:], axis=1),
            ],
        )
        return self.offsets.copy()

    def diagnostics(self):
        return dict(
            anchor_valid=self.anchors is not None,
            fixed_anchor_in_wrists_m=None
            if self.anchors is None
            else self.anchors.tolist(),
            offsets_world=self.offsets.tolist(),
            last_drift_in_wrists_m=self.last_drift.tolist(),
            last_tilt_error_world_rad=self.last_tilt.tolist(),
            reference_valid=self.last_reference_valid,
            updated_steps=self.updated.tolist(),
            frozen_steps=self.frozen.tolist(),
            blocked_downward_steps=self.blocked_downward,
            peak_translation_m=self.peak[:, 0].tolist(),
            peak_rotation_rad=self.peak[:, 1].tolist(),
            recovery_started=self.recovery is not None,
        )
