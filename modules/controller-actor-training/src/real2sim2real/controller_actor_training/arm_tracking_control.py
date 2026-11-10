"""Causal, bounded lag compensation without retiming the reference trajectory."""

import numpy as np


class InertialFeedforward:
    """Filtered M(q) qdd torque through robot actuators only, with explicit caps."""

    def __init__(self, settings, torque_limits):
        if (
            set(settings) != {"time_constant_s", "torque_fraction"}
            or not np.isfinite(list(settings.values())).all()
            or not 0.005 <= settings["time_constant_s"] <= 0.2
            or not 0 < settings["torque_fraction"] <= 1
        ):
            raise ValueError("Bounded inertial feedforward settings required")
        self.limits = np.asarray(torque_limits, float) * settings["torque_fraction"]
        if (
            self.limits.ndim != 1
            or not np.isfinite(self.limits).all()
            or (self.limits <= 0).any()
        ):
            raise ValueError("Positive torque limits required")
        self.tau = settings["time_constant_s"]
        self.torque = np.zeros_like(self.limits)
        self.peak = self.torque.copy()

    def update(self, mass_matrix, acceleration, dt, *, active):
        matrix, acceleration = np.asarray(mass_matrix), np.asarray(acceleration)
        n = len(self.limits)
        if (
            matrix.shape != (n, n)
            or acceleration.shape != (n,)
            or not np.isfinite(matrix).all()
            or not np.isfinite(acceleration).all()
            or not np.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError("Finite aligned inertia and acceleration required")
        desired = (
            np.clip(matrix @ acceleration, -self.limits, self.limits)
            if active
            else np.zeros(n)
        )
        self.torque += (1 - np.exp(-dt / self.tau)) * (desired - self.torque)
        self.peak = np.maximum(self.peak, abs(self.torque))
        return self.torque.copy()


class ArmTrackingCompensator:
    """Filtered velocity feedforward and optional anti-windup position integral.

    These are robot actuator targets, not edits to the scored object reference.
    No object state or human hand observation is required.
    """

    def __init__(self, settings, velocity_limits):
        required = {
            "velocity_fraction",
            "filter_time_s",
            "max_acceleration_rad_s2",
            "integral_gain_s_inv",
            "max_offset_rad",
            "max_offset_speed_rad_s",
        }
        if set(settings) != required or any(
            not np.isfinite(settings[k]) for k in required
        ):
            raise ValueError("Explicit finite tracking compensation settings required")
        if not (
            0 <= settings["velocity_fraction"] <= 1
            and 0.01 <= settings["filter_time_s"] <= 0.5
            and 0 < settings["max_acceleration_rad_s2"] <= 10
            and 0 <= settings["integral_gain_s_inv"] <= 20
            and 0 < settings["max_offset_rad"] <= 0.1
            and 0 < settings["max_offset_speed_rad_s"] <= 0.5
        ):
            raise ValueError("Tracking compensation exceeds bounded settings")
        self.limits = np.asarray(velocity_limits, float)
        if (
            self.limits.ndim != 1
            or not len(self.limits)
            or not np.isfinite(self.limits).all()
            or (self.limits <= 0).any()
        ):
            raise ValueError("Positive native joint velocity limits required")
        self.settings = settings
        self.velocity = np.zeros_like(self.limits)
        self.offset = np.zeros_like(self.limits)
        self.peak_offset = np.zeros_like(self.limits)
        self.frozen_joint_steps = 0

    def update(
        self, target, position, reference_velocity, torque, torque_limits, dt, *, active
    ):
        values = [
            np.asarray(v, float)
            for v in (target, position, reference_velocity, torque, torque_limits)
        ]
        if (
            any(
                v.shape != self.limits.shape or not np.isfinite(v).all() for v in values
            )
            or not np.isfinite(dt)
            or dt <= 0
            or (values[-1] <= 0).any()
        ):
            raise ValueError("Aligned finite joint states and a positive step required")
        target, position, reference_velocity, torque, torque_limits = values
        s = self.settings
        wanted = (
            np.clip(
                reference_velocity * s["velocity_fraction"], -self.limits, self.limits
            )
            if active
            else np.zeros_like(self.limits)
        )
        filtered_delta = (1 - np.exp(-dt / s["filter_time_s"])) * (
            wanted - self.velocity
        )
        self.velocity += np.clip(
            filtered_delta,
            -s["max_acceleration_rad_s2"] * dt,
            s["max_acceleration_rad_s2"] * dt,
        )
        if active:
            error = target - position
            speed = np.clip(
                s["integral_gain_s_inv"] * error,
                -s["max_offset_speed_rad_s"],
                s["max_offset_speed_rad_s"],
            )
            saturated = (abs(torque) >= 0.98 * torque_limits) & (speed * torque > 0)
            self.frozen_joint_steps += int(saturated.sum())
            speed[saturated] = 0
            self.offset = np.clip(
                self.offset + dt * speed, -s["max_offset_rad"], s["max_offset_rad"]
            )
            self.peak_offset = np.maximum(self.peak_offset, abs(self.offset))
        return target + self.offset, self.velocity.copy()

    def diagnostics(self):
        return dict(
            offset_rad=self.offset.tolist(),
            peak_absolute_offset_rad=self.peak_offset.tolist(),
            filtered_velocity_rad_s=self.velocity.tolist(),
            anti_windup_joint_steps=self.frozen_joint_steps,
        )
