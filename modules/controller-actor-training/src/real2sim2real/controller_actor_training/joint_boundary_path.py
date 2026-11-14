"""Smooth trial-boundary joint commands, separate from the task controller.

These interpolants do not certify collision clearance. Callers must screen the
whole robot and gate return motion on supported placement and cleared contact.
Joint values are unwrapped radians; no shortest-angle wrapping is applied.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointBoundaryPath:
    times_s: np.ndarray
    positions_rad: np.ndarray

    def __post_init__(self):
        t = np.array(self.times_s, dtype=float, copy=True)
        q = np.array(self.positions_rad, dtype=float, copy=True)
        if (
            t.ndim != 1
            or len(t) < 2
            or q.ndim != 2
            or q.shape[0] != len(t)
            or q.shape[1] == 0
            or not np.isfinite(t).all()
            or not np.isfinite(q).all()
            or t[0] != 0
            or (np.diff(t) <= 0).any()
        ):
            raise ValueError(
                "Finite joint waypoints and increasing times starting at zero required"
            )
        t.setflags(write=False)
        q.setflags(write=False)
        object.__setattr__(self, "times_s", t)
        object.__setattr__(self, "positions_rad", q)

    @property
    def duration_s(self):
        return float(self.times_s[-1])

    def command(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError("Finite time required")
        if time_s <= 0:
            return self.positions_rad[0].copy(), np.zeros(self.positions_rad.shape[1])
        if time_s >= self.duration_s:
            return self.positions_rad[-1].copy(), np.zeros(self.positions_rad.shape[1])
        i = int(np.searchsorted(self.times_s, time_s, side="right") - 1)
        dt = self.times_s[i + 1] - self.times_s[i]
        u = (time_s - self.times_s[i]) / dt
        blend = u**3 * (10 + u * (-15 + 6 * u))
        rate = 30 * u**2 * (1 - u) ** 2 / dt
        delta = self.positions_rad[i + 1] - self.positions_rad[i]
        return self.positions_rad[i] + blend * delta, rate * delta

    def maximum_speed_rad_s(self):
        # Maximum derivative of quintic smoothstep is 1.875 at u=0.5.
        return np.max(
            1.875
            * np.abs(np.diff(self.positions_rad, axis=0))
            / np.diff(self.times_s)[:, None],
            axis=0,
        )


def return_admitted(
    *,
    task_completed,
    release_fraction,
    hand_force_n,
    support_fraction,
    object_speed_m_s,
):
    """Contact/placement gate; collision screening remains independently required."""
    values = [release_fraction, hand_force_n, support_fraction, object_speed_m_s]
    if not np.isfinite(values).all() or min(values) < 0:
        return False
    return bool(
        task_completed
        and release_fraction >= 0.99
        and hand_force_n < 0.1
        and support_fraction >= 0.5
        and object_speed_m_s < 0.02
    )
