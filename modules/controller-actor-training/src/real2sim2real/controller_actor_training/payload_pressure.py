"""Bounded pressure scheduling from supported load and bilateral force sharing.

The core consumes observations, never model mass or a robustness-case identity.
Its current adapter supplies privileged simulator contact forces/acceleration.
"""

import numpy as np


def validate_settings(settings):
    caps = dict(
        reference_weight_n=100,
        weight_filter_s=5,
        share_filter_s=5,
        minimum_weight_ratio=1,
        maximum_weight_ratio=2,
        target_secondary_fraction=1,
        share_gain=2,
        maximum_share_correction=0.3,
        maximum_scale_rate_s_inv=5,
        maximum_acceleration_m_s2=5,
    )
    if (
        set(settings) - {"minimum_group_ratio"} != set(caps)
        or any(
            not np.isfinite(settings[k]) or settings[k] < 0 or settings[k] > cap
            for k, cap in caps.items()
        )
        or any(
            settings[k] <= 0
            for k in (
                "reference_weight_n",
                "weight_filter_s",
                "share_filter_s",
                "minimum_weight_ratio",
                "maximum_scale_rate_s_inv",
                "maximum_acceleration_m_s2",
            )
        )
        or not (
            settings["minimum_weight_ratio"] <= 1 <= settings["maximum_weight_ratio"]
            and 0 < settings["target_secondary_fraction"] < 1
        )
    ):
        raise ValueError("Explicit bounded payload/pressure sharing settings required")
    if "minimum_group_ratio" in settings:
        floors = settings["minimum_group_ratio"]
        if set(floors) != {"thumb", "index", "middle", "ring", "pinky"} or any(
            not np.isfinite(v) or not 0.1 <= v <= 1 for v in floors.values()
        ):
            raise ValueError("Explicit bounded per-group pressure floors required")


class PayloadPressure:
    def __init__(self, settings):
        validate_settings(settings)
        self.settings = settings
        self.weight_n = settings["reference_weight_n"]
        self.secondary_fraction = settings["target_secondary_fraction"]
        self.scales = np.ones(2)
        self.upward_n = np.zeros(2)
        self.measurement_n = None
        self.valid_updates = 0
        self.frozen_updates = 0
        self.saturated_targets = 0

    def update(self, upward_n, acceleration_z, gravity, dt, *, active, isolated):
        forces = np.asarray(upward_n, float)
        if (
            forces.shape != (2,)
            or not np.isfinite(forces).all()
            or not np.isfinite([acceleration_z, gravity, dt]).all()
            or gravity <= 0
            or dt <= 0
        ):
            raise ValueError("Finite bilateral force, acceleration and clock required")
        self.upward_n = forces.copy()
        s = self.settings
        total = float(forces.sum())
        valid = (
            active
            and isolated
            and total > 0.1
            and abs(acceleration_z) <= s["maximum_acceleration_m_s2"]
            and gravity + acceleration_z >= 0.5 * gravity
        )
        if not valid:
            self.frozen_updates += int(active)
            return
        self.measurement_n = total * gravity / (gravity + acceleration_z)
        a = -np.expm1(-dt / s["weight_filter_s"])
        self.weight_n += a * (self.measurement_n - self.weight_n)
        a = -np.expm1(-dt / s["share_filter_s"])
        self.secondary_fraction += a * (
            float(forces[1] / total) - self.secondary_fraction
        )
        load = np.clip(
            self.weight_n / s["reference_weight_n"],
            s["minimum_weight_ratio"],
            s["maximum_weight_ratio"],
        )
        # Raise requested pressure on the underloaded hand. Keep the unconstrained
        # pair sum fixed; per-pad force caps can subsequently limit either side.
        share = np.clip(
            s["share_gain"]
            * (s["target_secondary_fraction"] - self.secondary_fraction),
            -s["maximum_share_correction"],
            s["maximum_share_correction"],
        )
        desired = load * np.array([1 - share, 1 + share])
        self.scales += np.clip(
            desired - self.scales,
            -dt * s["maximum_scale_rate_s_inv"],
            dt * s["maximum_scale_rate_s_inv"],
        )
        self.valid_updates += 1

    def target(self, role, nominal_n, *, group=None):
        if (
            role not in ("primary", "secondary")
            or not np.isfinite(nominal_n)
            or not 0 < nominal_n <= 10
        ):
            raise ValueError(
                "Explicit hand role and bounded nominal pad target required"
            )
        scale = self.scales[int(role == "secondary")]
        floors = self.settings.get("minimum_group_ratio")
        if floors is not None:
            if group not in floors:
                raise ValueError(
                    "Explicit anatomical group required with pressure floors"
                )
            scale = max(scale, floors[group])
        value = nominal_n * scale
        self.saturated_targets += int(value > 10)
        return float(min(value, 10))

    def diagnostics(self):
        return dict(
            estimated_weight_n=float(self.weight_n),
            observed_weight_n=self.measurement_n,
            secondary_upward_fraction=float(self.secondary_fraction),
            upward_force_n=self.upward_n.tolist(),
            pressure_scales=self.scales.tolist(),
            valid_updates=self.valid_updates,
            frozen_updates=self.frozen_updates,
            capped_targets=self.saturated_targets,
        )
