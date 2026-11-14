"""Validated phase profiles around a retained reference controller.

Profiles change commands, never physics, observations or acceptance criteria.
The existing contact-pressure feedback remains responsible for closed-loop force
correction. These profiles are not a learned policy or a control certificate.
"""

import numpy as np


class ReferenceRefinement:
    def __init__(self, edits, config, reference, limits, bounds):
        allowed = {
            "reference_times_s",
            "pressure_targets_n",
            "gain_scale",
            "finger_offsets_rad",
        }
        if not isinstance(edits, dict) or set(edits) - allowed:
            raise ValueError("Only declared reference refinement fields are allowed")
        self.base_goals = np.asarray(config["pressure_targets_n"], float)
        self.base_gain = float(config["pressure"]["gain_m_per_n_s"])
        self.times = np.asarray(edits.get("reference_times_s"), float)
        start = config["schedule"]["pressure_start_reference_s"]
        end = config["schedule"]["release_reference_s"]
        if (
            self.times.ndim != 1
            or not 2 <= len(self.times) <= bounds["maximum_knots"]
            or not np.isfinite(self.times).all()
            or np.any(np.diff(self.times) <= 0)
            or self.times[0] != start
            or self.times[-1] != end
        ):
            raise ValueError(
                "Increasing reference-time knots must cover pressure start through release"
            )
        n, joints = len(self.times), len(config["finger_joints"])
        self.goals = np.asarray(
            edits.get("pressure_targets_n", np.tile(self.base_goals, (n, 1))), float
        )
        self.gain = np.asarray(edits.get("gain_scale", np.ones(n)), float)
        self.offset = np.asarray(
            edits.get("finger_offsets_rad", np.zeros((n, joints))), float
        )
        if (
            self.goals.shape != (n, len(self.base_goals))
            or self.gain.shape != (n,)
            or self.offset.shape != (n, joints)
        ):
            raise ValueError("Profile dimensions do not match the retained controller")
        if not np.isfinite(
            np.r_[self.goals.ravel(), self.gain, self.offset.ravel()]
        ).all():
            raise ValueError("Finite profile values required")
        if np.any(self.goals <= 0) or np.any(self.goals > 5):
            raise ValueError("Pressure targets must remain in (0, 5] N")
        if (
            np.any(self.gain < bounds["minimum_gain_scale"])
            or np.any(self.gain > bounds["maximum_gain_scale"])
            or np.any(self.gain * self.base_gain > 0.01)
        ):
            raise ValueError("Pressure response exceeds the trusted bounds")
        if np.max(np.abs(self.offset)) > bounds["maximum_finger_offset_rad"]:
            raise ValueError("Finger correction exceeds the trusted offset bound")
        owned = {i for group in config["pressure_groups"] for i in group}
        if any(np.any(self.offset[:, i]) for i in range(joints) if i not in owned):
            raise ValueError("Only configured pressure-group finger joints may change")
        if (
            np.any(self.offset[[0, -1]])
            or not np.array_equal(self.goals[[0, -1]], np.tile(self.base_goals, (2, 1)))
            or np.any(self.gain[[0, -1]] != 1)
        ):
            raise ValueError("Profile endpoints must return to the retained command")
        times = np.asarray(reference["reference_times_s"])
        offsets = np.column_stack(
            [
                np.interp(times, self.times, self.offset[:, j], left=0, right=0)
                for j in range(joints)
            ]
        )
        rates = np.diff(offsets, axis=0) / config["feedback_s"]
        if (
            np.max(np.abs(rates), initial=0)
            > bounds["maximum_offset_speed_rad_s"] + 1e-10
        ):
            raise ValueError(
                "Correction rate exceeds the bound on the actual feedback clock"
            )
        edited = np.asarray(reference["finger_targets_rad"]) + offsets
        changed = np.any(offsets != 0, axis=1)
        limits = np.asarray(limits)
        if np.any(edited[changed] < limits[:, 0] - 1e-8) or np.any(
            edited[changed] > limits[:, 1] + 1e-8
        ):
            raise ValueError("Refined finger command exceeds native joint limits")

    def command(self, reference_time, nominal):
        if reference_time <= self.times[0] or reference_time >= self.times[-1]:
            return nominal, self.base_goals.copy(), self.base_gain
        offset = np.array(
            [np.interp(reference_time, self.times, values) for values in self.offset.T]
        )
        goals = np.array(
            [np.interp(reference_time, self.times, values) for values in self.goals.T]
        )
        gain = self.base_gain * float(np.interp(reference_time, self.times, self.gain))
        return nominal + offset, goals, gain
