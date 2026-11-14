"""Bounded contact-loss posture corrections around an immutable reference.

Simulation forces are privileged observations, not validated tactile sensors.
The host owns release and all dynamics; no state reset or arm command is exposed.
"""

import numpy as np


class ContactConditionedPosture:
    def __init__(self, specification, config, reference, limits, bounds):
        allowed = {"gain", "phases", "loss_threshold_n", "loss_dwell_s", "offsets_rad"}
        if not isinstance(specification, dict) or set(specification) != allowed:
            raise ValueError("Explicit contact feedback specification required")
        self.gain = float(specification["gain"])
        self.threshold = float(specification["loss_threshold_n"])
        self.dwell = float(specification["loss_dwell_s"])
        self.phases = specification["phases"]
        if (
            not isinstance(self.phases, list)
            or not self.phases
            or any(not isinstance(p, str) for p in self.phases)
            or not set(self.phases) <= set(reference["phases"])
        ):
            raise ValueError("Known phase labels required")
        self.offsets = np.asarray(specification["offsets_rad"], float)
        joints, pads = len(config["finger_joints"]), len(config["pressure_groups"])
        if self.offsets.shape != (pads, joints) or not np.isfinite(self.offsets).all():
            raise ValueError("Finite pad by joint corrections required")
        if not (
            0 <= self.gain <= 1
            and 0 < self.threshold <= 0.5
            and 0.01 <= self.dwell <= 0.5
        ):
            raise ValueError("Contact feedback gain/threshold/dwell outside bounds")
        for pad, group in enumerate(config["pressure_groups"]):
            if any(self.offsets[pad, j] != 0 for j in range(joints) if j not in group):
                raise ValueError(
                    "Each pad correction must own its pressure-group joints"
                )
        self.bound = float(bounds["maximum_finger_offset_rad"])
        if np.any(np.sum(np.abs(self.offsets), axis=0) > self.bound):
            raise ValueError("Combined contact corrections exceed the offset bound")
        self.rate = float(bounds["maximum_offset_speed_rad_s"])
        self.dt = float(config["feedback_s"])
        self.limits = np.asarray(limits, float)
        self.release_time = float(config["schedule"]["release_reference_s"])
        self.start_time = float(config["schedule"]["pressure_start_reference_s"])
        self.weight = float(config["object_weight_n"])
        if not (self.dt > 0 and self.weight > 0 and self.rate > 0 and self.bound > 0):
            raise ValueError("Positive feedback, weight and correction bounds required")
        self.loss_time = np.zeros(pads)
        self.correction = np.zeros(joints)
        self.previous_total = np.zeros(joints)

    def command(
        self,
        nominal,
        profiled,
        *,
        reference_time,
        phase,
        forces_n,
        support_n,
        releasing,
    ):
        # Explicit zero bypass: no arithmetic or clipping changes the baseline.
        if self.gain == 0:
            return profiled
        nominal, profiled = np.asarray(nominal, float), np.asarray(profiled, float)
        try:
            force = np.asarray(forces_n, float)
            support_n = float(support_n)
            valid = (
                force.shape == self.loss_time.shape
                and np.isfinite(force).all()
                and np.all(force >= 0)
                and np.isfinite(support_n)
            )
        except (TypeError, ValueError):
            valid = False
            force = np.zeros_like(self.loss_time)
        active = (
            valid
            and not releasing
            and self.start_time <= reference_time < self.release_time
            and phase in self.phases
            and support_n < 0.5 * self.weight
        )
        if not active:
            self.loss_time[:] = 0
            target = np.zeros_like(self.correction)
        else:
            self.loss_time = np.where(
                force < self.threshold, self.loss_time + self.dt, 0
            )
            target = self.gain * (
                (self.loss_time >= self.dwell).astype(float) @ self.offsets
            )
        if (
            releasing
            or reference_time >= self.release_time
            or reference_time < self.start_time
        ):
            self.correction[:] = 0  # Host release has authority over the correction.
            self.previous_total[:] = 0
            return profiled
        delta = np.clip(
            target - self.correction, -self.rate * self.dt, self.rate * self.dt
        )
        self.correction += delta
        combined = np.clip(
            profiled - nominal + self.correction, -self.bound, self.bound
        )
        combined = self.previous_total + np.clip(
            combined - self.previous_total, -self.rate * self.dt, self.rate * self.dt
        )
        # Clip only edited commands; leave untouched reference entries bit-exact.
        edited = combined != 0
        result = nominal.copy()
        result[edited] = np.clip(
            (nominal + combined)[edited], self.limits[edited, 0], self.limits[edited, 1]
        )
        realized = result - nominal
        if np.any(np.abs(realized - self.previous_total) > self.rate * self.dt + 1e-10):
            raise ValueError("Native joint limits conflict with correction-rate limit")
        self.previous_total = realized.copy()
        return result
