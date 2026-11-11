"""Small pressure-action interface sharing the existing finger actuator owner."""

import numpy as np

CHANNELS = ("primary:thumb", "primary:fingers", "secondary:thumb", "secondary:fingers")


def validate_action(action):
    value = np.asarray(action, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all() or (abs(value) > 1).any():
        raise ValueError("Four finite normalized pressure actions in [-1, 1] required")
    return value.copy()


def validate_offsets(offsets):
    if set(offsets) != set(CHANNELS) or any(
        not np.isfinite(v) or abs(v) > 2 for v in offsets.values()
    ):
        raise ValueError("Four named pressure offsets bounded by 2 N required")


def pressure_goal(nominal_n, offset_n):
    if (
        not np.isfinite([nominal_n, offset_n]).all()
        or not 0 < nominal_n <= 10
        or abs(offset_n) > 2
    ):
        raise ValueError(
            "Finite positive baseline pressure <=10 N and offset <=2 N required"
        )
    # Avoid even clipping the original goal on the exact zero path.
    return nominal_n if offset_n == 0 else float(np.clip(nominal_n + offset_n, 0.1, 10))


class PressureResidual:
    def __init__(self, settings):
        if set(settings) != {
            "policy_period_s",
            "max_offset_n",
            "max_rate_n_s",
            "active_phases",
        }:
            raise ValueError(
                "Explicit pressure policy clock, bounds and phases required"
            )
        for name, low, high in [
            ("policy_period_s", 0.01, 0.1),
            ("max_offset_n", 0, 2),
            ("max_rate_n_s", 0, 10),
        ]:
            v = settings[name]
            if (
                isinstance(v, bool)
                or not np.isfinite(v)
                or not low <= v <= high
                or v == 0
            ):
                raise ValueError("Pressure policy clock or correction bounds invalid")
        phases = settings["active_phases"]
        if (
            not isinstance(phases, list)
            or not phases
            or len(set(phases)) != len(phases)
            or set(phases) - {"close", "lift", "hold"}
        ):
            raise ValueError("Explicit close/lift/hold phases required")
        self.settings = dict(settings, active_phases=list(phases))
        self.requested = np.zeros(4)
        self.offset = np.zeros(4)
        self.peak = np.zeros(4)

    def accept(self, action):
        self.requested = validate_action(action)

    def advance(self, dt, phase):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite control interval required")
        desired = (
            self.requested * self.settings["max_offset_n"]
            if phase in self.settings["active_phases"]
            else np.zeros(4)
        )
        self.offset += np.clip(
            desired - self.offset,
            -dt * self.settings["max_rate_n_s"],
            dt * self.settings["max_rate_n_s"],
        )
        self.peak = np.maximum(self.peak, abs(self.offset))
        if not self.offset.any():
            return None
        return dict(zip(CHANNELS, self.offset.tolist()))
