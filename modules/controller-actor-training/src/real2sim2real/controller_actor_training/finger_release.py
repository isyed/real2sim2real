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
        set(settings) - {"thumb_duration_s"} != set(bounds)
        or any(
            not np.isfinite(settings[k]) or not lo <= settings[k] <= hi
            for k, (lo, hi) in bounds.items()
        )
        or min(settings["start_s"], settings["maximum_speed_m_s"]) <= 0
    ):
        raise ValueError("Explicit bounded table-supported finger release required")
    if "thumb_duration_s" in settings and (
        not np.isfinite(settings["thumb_duration_s"])
        or not 0.5 <= settings["thumb_duration_s"] <= settings["duration_s"]
    ):
        raise ValueError("Thumb unhooking must finish within the release window")


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
        self.unhook_mask = None
        self.thumb_fraction = 0.0

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
        result = (1 - self.fraction) * self.initial + self.fraction * values[1]
        if self.unhook_mask is not None and "thumb_duration_s" in s:
            mask = np.asarray(self.unhook_mask, bool)
            if mask.shape != values[0].shape:
                raise ValueError("Thumb mask must match owned finger commands")
            self.thumb_fraction = release_fraction(
                time, dict(s, start_s=self.started_s, duration_s=s["thumb_duration_s"])
            )
            result[mask] = (1 - self.thumb_fraction) * self.initial[
                mask
            ] + self.thumb_fraction * values[1][mask]
        else:
            self.thumb_fraction = self.fraction
        return result

    def diagnostics(self):
        return dict(
            started_s=self.started_s,
            fraction=self.fraction,
            supported_duration_s=self.support_s,
            maximum_open_error_rad=self.open_error,
            thumb_fraction=self.thumb_fraction,
        )
