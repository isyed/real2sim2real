"""Bounded common-wrist admittance after confirmed object/table support.

No object assistance, finger commands, reference edits or early release. The
caller composes these offsets inside its existing Cartesian authority budget.
"""

from pathlib import Path

import numpy as np

from real2sim2real.artifacts import digest

from .paired_acquisition_audit import pinned_json


def validate_source(spec, config):
    if set(spec) != {"baseline", "controller"}:
        raise ValueError("Pinned table-transfer comparison and controller required")
    baseline = pinned_json(spec["baseline"])
    from .continuous_residual import validate_baseline

    validate_baseline(baseline)
    if (
        not baseline.get("physics_run")
        or not baseline.get("closed_loop_grasp_feedback")
        or baseline.get("table_load_transfer")
        or baseline.get("residual_adapter")
        or "incomplete_pilot" in baseline["rejected_for"]
    ):
        raise ValueError("Complete original wrist-feedback comparison required")
    for name, key in [("scene.xml", "scene_sha256"), ("states.npz", "states_sha256")]:
        if digest(Path(spec["baseline"]["path"]).with_name(name)) != baseline[key]:
            raise ValueError("Table-transfer comparison archive changed")
    ignored = {"output", "purpose", "motion_condition", "table_load_transfer", "record_arm_commands"}
    strip = lambda c: {k: v for k, v in c.items() if k not in ignored}
    if strip(config) != strip(baseline["config"]) or config.get(
        "motion_condition"
    ) not in ("demonstrated", "stationary_hold"):
        raise ValueError(
            "Table transfer must preserve controller, physics and reference"
        )
    TableLoadTransfer(spec["controller"], baseline["continuous_protocol"])


class TableLoadTransfer:
    def __init__(self, settings, protocol):
        bounds = {
            "gain_m_per_n_s": (0, 0.003),
            "max_offset_m": (0, 0.006),
            "max_speed_m_s": (0, 0.012),
            "confirmation_s": (0.02, 0.2),
            "support_fraction": (0.25, 1.0),
            "deadband_fraction": (0.02, 0.25),
        }
        if set(settings) != set(bounds) or any(
            isinstance(settings[k], bool)
            or not np.isfinite(settings[k])
            or not lo <= settings[k] <= hi
            or (settings[k] == 0 and k != "gain_m_per_n_s")
            for k, (lo, hi) in bounds.items()
        ):
            raise ValueError("Explicit bounded table-transfer settings required")
        gap = protocol["finger_release"]["start_s"] - protocol["capture_duration_s"]
        if (
            gap <= 0
            or 1.875 * settings["max_offset_m"] / gap > settings["max_speed_m_s"]
        ):
            raise ValueError("Table correction must recover before scheduled release")
        self.settings, self.protocol = dict(settings), protocol
        self.offset = self.peak = self.support_duration = 0.0
        self.recovery = None
        self.active = False
        self.first_active_s = None
        self.budget_limited_steps = 0
        self.table_force = self.weight = 0.0

    def advance(self, time_s, dt, table_force_n, weight_n, *, table_contacts, phase):
        if (
            not np.isfinite([time_s, dt, table_force_n, weight_n]).all()
            or dt <= 0
            or weight_n <= 0
            or table_contacts < 0
        ):
            raise ValueError(
                "Finite signed table force, payload and increasing control step required"
            )
        s, p = self.settings, self.protocol
        self.table_force, self.weight = float(table_force_n), float(weight_n)
        if time_s >= p["capture_duration_s"] and not p.get(
            "release_disabled_for_stationary_diagnostic"
        ):
            from .object_grasp_pilot import smoothstep

            if self.recovery is None:
                self.recovery = self.offset
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
            self.offset = self.recovery * a
            self.active = False
            return self.offset
        supported = (
            phase == "return"
            and table_contacts > 0
            and table_force_n >= s["support_fraction"] * weight_n
        )
        self.support_duration = self.support_duration + dt if supported else 0.0
        self.active = bool(
            self.support_duration + 1e-12 >= s["confirmation_s"]
            and s["gain_m_per_n_s"] > 0
        )
        if self.active:
            if self.first_active_s is None:
                self.first_active_s = float(time_s)
            error = table_force_n - weight_n
            error = np.sign(error) * max(
                0.0, abs(error) - s["deadband_fraction"] * weight_n
            )
            speed = np.clip(
                s["gain_m_per_n_s"] * error, -s["max_speed_m_s"], s["max_speed_m_s"]
            )
            self.offset = float(np.clip(self.offset + dt * speed, 0, s["max_offset_m"]))
        else:
            self.offset = max(0.0, self.offset - dt * s["max_speed_m_s"])
        return self.offset

    def compose(self, original, cap_m):
        """Share a common upward offset without consuming lateral/rotation authority."""
        offsets = np.asarray(original, float)
        if (
            offsets.shape != (2, 6)
            or not np.isfinite(offsets).all()
            or not 0 < cap_m <= 0.006
            or (np.linalg.norm(offsets[:, :3], axis=1) > cap_m + 1e-10).any()
        ):
            raise ValueError(
                "Existing bilateral offsets must satisfy the shared translation cap"
            )
        if self.offset == 0:
            return offsets.copy()
        headroom = (
            np.sqrt(np.maximum(0, cap_m**2 - np.sum(offsets[:, :2] ** 2, axis=1)))
            - offsets[:, 2]
        )
        applied = min(self.offset, max(0.0, float(headroom.min())))
        self.budget_limited_steps += int(applied < self.offset - 1e-12)
        self.offset = applied  # anti-windup against the shared authority boundary
        result = offsets.copy()
        result[:, 2] += applied
        self.peak = max(self.peak, applied)
        return result

    def diagnostics(self):
        return {
            "active": self.active,
            "first_active_s": self.first_active_s,
            "common_upward_offset_m": self.offset,
            "peak_offset_m": self.peak,
            "confirmed_support_duration_s": self.support_duration,
            "table_vertical_force_n": self.table_force,
            "weight_n": self.weight,
            "budget_limited_steps": self.budget_limited_steps,
            "recovery_started": self.recovery is not None,
        }


def observe_table_force(model, data, object_geoms, table_geom, object_body):
    """Signed upward object load from table contacts only, excluding all hands."""
    from .support_control import contact_wrench_on_body

    force, count = 0.0, 0
    for cid, c in enumerate(data.contact):
        pair = {int(c.geom1), int(c.geom2)}
        if table_geom in pair and pair & object_geoms:
            force += contact_wrench_on_body(model, data, cid, object_body)[0][2]
            count += int(c.efc_address >= 0)
    return float(force), count
