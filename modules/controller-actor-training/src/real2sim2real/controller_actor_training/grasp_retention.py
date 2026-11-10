"""Bounded wrist translation to retain an established, loaded grasp."""

import numpy as np

from .wrist_coordination import bounded_vector


class GraspRetention:
    """Keep the object-to-wrist vector near its fixed hold-entry value.

    Updates require opposed contact and a small independent reference error.
    No contact labels, target reference, force limits or object state are changed.
    This observes privileged simulator poses/forces and is not a tactile estimator.
    """

    def __init__(self, settings):
        required = {
            "gain_s_inv",
            "max_speed_m_s",
            "max_offset_m",
            "minimum_contact_n",
            "maximum_reference_error_m",
        }
        if set(settings) != required:
            raise ValueError("Explicit bounded retention settings required")
        bounds = {
            "gain_s_inv": 3,
            "max_speed_m_s": 0.01,
            "max_offset_m": 0.01,
            "minimum_contact_n": 10,
            "maximum_reference_error_m": 0.02,
        }
        if any(
            not np.isfinite(settings[k]) or not 0 < settings[k] <= bounds[k]
            for k in required
        ):
            raise ValueError("Retention settings exceed finite safety bounds")
        self.settings = settings
        self.anchor = None
        self.offset = np.zeros(3)
        self.updated = self.frozen = self.saturated = 0

    def update(
        self,
        object_position,
        wrist_position,
        wrist_rotation,
        dt,
        *,
        thumb_force_n,
        opposing_force_n,
        reference_error_m,
    ):
        p, w, r = map(np.asarray, (object_position, wrist_position, wrist_rotation))
        if (
            p.shape != (3,)
            or w.shape != (3,)
            or r.shape != (3, 3)
            or not np.isfinite(
                [
                    *p,
                    *w,
                    *r.ravel(),
                    dt,
                    thumb_force_n,
                    opposing_force_n,
                    reference_error_m,
                ]
            ).all()
            or dt <= 0
            or min(thumb_force_n, opposing_force_n, reference_error_m) < 0
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(r), 1)
        ):
            raise ValueError("Finite rigid observations and a positive step required")
        if self.anchor is None:
            self.anchor = r.T @ (p - w)
        s = self.settings
        if (
            min(thumb_force_n, opposing_force_n) < s["minimum_contact_n"]
            or reference_error_m > s["maximum_reference_error_m"]
        ):
            self.frozen += 1
            return self.offset.copy()
        error = p - w - r @ self.anchor
        velocity = bounded_vector(s["gain_s_inv"] * error, s["max_speed_m_s"])
        proposed = self.offset + dt * velocity
        if np.linalg.norm(proposed) > s["max_offset_m"]:
            self.saturated += 1
        self.offset = bounded_vector(proposed, s["max_offset_m"])
        self.updated += 1
        return self.offset.copy()

    def apply(self, target, q_indices, limits):
        result = np.asarray(target).copy()
        ids = np.asarray(q_indices)
        limits = np.asarray(limits)
        if ids.shape != (3,) or len(set(ids.tolist())) != 3 or limits.shape != (3, 2):
            raise ValueError(
                "Three distinct checked Cartesian slide coordinates required"
            )
        proposed = result[ids] + self.offset
        if ((proposed < limits[:, 0]) | (proposed > limits[:, 1])).any():
            raise ValueError("retention_wrist_joint_limit")
        result[ids] = proposed
        return result

    def diagnostics(self):
        return dict(
            retention_offset_world_m=self.offset.tolist(),
            retention_anchor_wrist_m=None
            if self.anchor is None
            else self.anchor.tolist(),
            retention_updated_steps=self.updated,
            retention_frozen_steps=self.frozen,
            retention_saturated_steps=self.saturated,
        )
