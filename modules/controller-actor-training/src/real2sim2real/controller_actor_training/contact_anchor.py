"""Latch observed pad material points for bounded tangential retention.

Anchors are never moved to hide later slip. This privileged-state simulator
adapter changes a finger task velocity, not the object or scored reference.
"""

import numpy as np

from .arm_object_feedback import rigid_pose
from .wrist_coordination import bounded_vector


def validate_settings(settings):
    caps = dict(
        gain_s_inv=5, max_speed_m_s=0.005, minimum_force_n=5, maximum_error_m=0.015
    )
    if set(settings) != set(caps) or any(
        not np.isfinite(settings[k]) or not 0 < settings[k] <= cap
        for k, cap in caps.items()
    ):
        raise ValueError("Explicit bounded contact-anchor settings required")


class ContactAnchors:
    def __init__(self, settings):
        validate_settings(settings)
        self.settings = settings
        self.anchors = {}
        self.updated = self.frozen = 0

    def task(
        self, name, body_pose, object_pose, observed_point, normal, force_n, *, active
    ):
        body, obj = map(rigid_pose, (body_pose, object_pose))
        normal = np.asarray(normal, float)
        if (
            normal.shape != (3,)
            or not np.isfinite(normal).all()
            or not np.isclose(np.linalg.norm(normal), 1)
            or not np.isfinite(force_n)
            or force_n < 0
            or type(active) is not bool
        ):
            raise ValueError(
                "Unit normal, finite nonnegative force and explicit activation required"
            )
        if (
            not active
            or force_n < self.settings["minimum_force_n"]
            or observed_point is None
        ):
            self.frozen += int(active)
            return None
        point = np.asarray(observed_point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Finite observed world point required")
        if name not in self.anchors:
            self.anchors[name] = dict(
                body=body[:3, :3].T @ (point - body[:3, 3]),
                object=obj[:3, :3].T @ (point - obj[:3, 3]),
            )
        anchor = self.anchors[name]
        current = body[:3, 3] + body[:3, :3] @ anchor["body"]
        desired = obj[:3, 3] + obj[:3, :3] @ anchor["object"]
        error = current - desired
        anchor["error_m"] = float(np.linalg.norm(error))
        if np.linalg.norm(error) > self.settings["maximum_error_m"]:
            self.frozen += 1
            return None
        projection = np.eye(3) - np.outer(normal, normal)
        velocity = bounded_vector(
            -self.settings["gain_s_inv"] * (projection @ error),
            self.settings["max_speed_m_s"],
        )
        self.updated += 1
        return current, projection, velocity

    def diagnostics(self):
        return dict(
            latched_pads=len(self.anchors),
            updated_pad_steps=self.updated,
            frozen_pad_steps=self.frozen,
            errors_m={k: v.get("error_m", 0.0) for k, v in self.anchors.items()},
            anchors_relatched=False,
        )
