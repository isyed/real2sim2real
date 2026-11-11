"""One-way, table-supported release of owned finger commands after return."""

import numpy as np


def validate_release(settings):
    bounds = dict(
        start_s=(0, 120),
        duration_s=(0.5, 2),
        settle_s=(0.5, 2),
        support_dwell_s=(0.1, 0.5),
        maximum_clearance_m=(0, 0.005),
        maximum_speed_m_s=(0, 0.02),
    )
    if (
        set(settings) != set(bounds)
        or any(
            not np.isfinite(settings[k]) or not lo <= settings[k] <= hi
            for k, (lo, hi) in bounds.items()
        )
        or min(settings["start_s"], settings["maximum_speed_m_s"]) <= 0
    ):
        raise ValueError("Explicit bounded table-supported finger release required")


def release_fraction(time, settings):
    from .object_grasp_pilot import smoothstep

    return float(
        smoothstep(np.clip((time - settings["start_s"]) / settings["duration_s"], 0, 1))
    )


class FingerRelease:
    """Latch only after observed table support; never re-close automatically."""

    def __init__(self, settings):
        validate_release(settings)
        self.settings = settings
        self.support_s = 0.0
        self.started_s = None
        self.initial = None
        self.fraction = 0.0
        self.open_error = None

    def command(
        self,
        time,
        dt,
        held,
        opened,
        actual,
        *,
        table_contacts,
        clearance_m,
        speed_m_s,
        table_load_fraction,
    ):
        values = [np.asarray(x, float) for x in (held, opened, actual)]
        if any(
            x.shape != values[0].shape or not np.isfinite(x).all() for x in values
        ) or (
            not np.isfinite(
                [time, dt, clearance_m, speed_m_s, table_load_fraction]
            ).all()
            or dt <= 0
            or speed_m_s < 0
        ):
            raise ValueError("Finite finger commands and table observations required")
        s = self.settings
        supported = (
            table_contacts > 0
            and -0.003 <= clearance_m <= s["maximum_clearance_m"]
            and speed_m_s <= s["maximum_speed_m_s"]
            and table_load_fraction >= 0.5
        )
        self.support_s = self.support_s + dt if supported else 0.0
        if (
            self.started_s is None
            and time >= s["start_s"]
            and self.support_s >= s["support_dwell_s"]
        ):
            self.started_s = time
            self.initial = values[0].copy()
        if self.started_s is None:
            return None
        self.fraction = release_fraction(time, dict(s, start_s=self.started_s))
        self.open_error = float(np.max(np.abs(values[2] - values[1]), initial=0))
        return (1 - self.fraction) * self.initial + self.fraction * values[1]

    def diagnostics(self):
        return dict(
            started_s=self.started_s,
            fraction=self.fraction,
            supported_duration_s=self.support_s,
            maximum_open_error_rad=self.open_error,
        )
