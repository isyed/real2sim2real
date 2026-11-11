"""Bounded object-relative wrist targets and contact-gated support admittance.

This controller consumes world-frame poses and measured forces. Simulation
experiments must declare privileged state access; it is not a state estimator.
It changes robot targets only, never the object, reference or acceptance gates.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .support_control import support_load_offset
from .wrist_coordination import bounded_vector


class SupportHeightAlignment:
    """Latch one observed height discrepancy before insertion; never chase it."""

    def __init__(self, settings):
        bounds = {
            "max_offset_m": 0.01,
            "max_speed_m_s": 0.02,
            "minimum_primary_contact_n": 10,
            "maximum_initial_error_m": 0.02,
        }
        if set(settings) != set(bounds) or any(
            not np.isfinite(settings[k]) or not 0 < settings[k] <= v
            for k, v in bounds.items()
        ):
            raise ValueError("Explicit bounded support-height alignment required")
        self.settings = settings
        self.target_offset = self.offset = 0.0
        self.state = "waiting"

    def update(self, phase, delta_world_m, primary_opposed_n, dt):
        delta = np.asarray(delta_world_m, float)
        if (
            delta.shape != (3,)
            or not np.isfinite(delta).all()
            or not np.isfinite([primary_opposed_n, dt]).all()
            or primary_opposed_n < 0
            or dt <= 0
        ):
            raise ValueError(
                "Finite object discrepancy, primary contact and positive clock required"
            )
        s = self.settings
        if self.state == "waiting" and phase == "support_approach":
            if (
                primary_opposed_n < s["minimum_primary_contact_n"]
                or np.linalg.norm(delta) > s["maximum_initial_error_m"]
            ):
                self.state = "rejected_observation"
            else:
                self.target_offset = float(
                    np.clip(delta[2], -s["max_offset_m"], s["max_offset_m"])
                )
                self.state = "latched"
        self.offset += float(
            np.clip(
                self.target_offset - self.offset,
                -s["max_speed_m_s"] * dt,
                s["max_speed_m_s"] * dt,
            )
        )
        return self.offset

    def diagnostics(self):
        return dict(
            state=self.state,
            target_offset_m=self.target_offset,
            applied_offset_m=self.offset,
        )


def support_contact_evidence(
    phase, support_force_n, primary_opposed_n, *, allow_precontact=False
):
    """Gate precontact alignment on the primary grasp, never invent support force.

    Before support insertion there cannot yet be support contact. With explicit
    opt-in, primary opposed contact can enable the existing bounded pose servo
    during support_approach. Seating and hold still require support contact.
    """
    if (
        type(allow_precontact) is not bool
        or not np.isfinite([support_force_n, primary_opposed_n]).all()
        or min(support_force_n, primary_opposed_n) < 0
    ):
        raise ValueError(
            "Finite nonnegative observed forces and explicit alignment flag required"
        )
    return (
        primary_opposed_n
        if allow_precontact and phase == "support_approach"
        else support_force_n
    )


def rigid_pose(value):
    value = np.asarray(value, float)
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1])
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(value[:3, :3]), 1)
    ):
        raise ValueError("A finite rigid world-frame pose is required")
    return value


def object_relative_error(actual_object, reference_object, reference_wrist):
    """World-frame twist from nominal wrist to the same grasp on actual object."""
    actual, reference, wrist = map(
        rigid_pose, (actual_object, reference_object, reference_wrist)
    )
    desired = actual @ np.linalg.inv(reference) @ wrist
    return np.r_[
        desired[:3, 3] - wrist[:3, 3],
        Rotation.from_matrix(desired[:3, :3] @ wrist[:3, :3].T).as_rotvec(),
    ]


def joint_offset(jacobian, twist, *, damping=0.02, max_offset_rad=0.15):
    """Damped local IK with a norm bound; rotation rows use a 0.1 m scale."""
    j, error = np.asarray(jacobian, float), np.asarray(twist, float)
    if (
        j.ndim != 2
        or j.shape[0] != 6
        or error.shape != (6,)
        or not np.isfinite(j).all()
        or not np.isfinite(error).all()
        or not np.isfinite([damping, max_offset_rad]).all()
        or damping <= 0
        or not 0 < max_offset_rad <= 0.2
    ):
        raise ValueError("Finite six-row Jacobian and bounded IK required")
    scale = np.array([1, 1, 1, 0.1, 0.1, 0.1])
    weighted = scale[:, None] * j
    result = weighted.T @ np.linalg.solve(
        weighted @ weighted.T + damping**2 * np.eye(6), scale * error
    )
    return bounded_vector(result, max_offset_rad)


class ArmObjectFeedback:
    """Rate/travel limited grasp correction; freeze on contact/reference loss."""

    def __init__(self, settings, names):
        required = {
            "gain_s_inv",
            "max_speed_m_s",
            "max_angular_speed_rad_s",
            "max_offset_m",
            "max_angle_rad",
            "minimum_contact_n",
            "maximum_reference_error_m",
            "maximum_reference_angle_rad",
        }
        if set(settings) - {"support_load"} != required:
            raise ValueError("Explicit bounded wrist feedback settings required")
        bounds = dict(
            gain_s_inv=5,
            max_speed_m_s=0.02,
            max_angular_speed_rad_s=0.2,
            max_offset_m=0.01,
            max_angle_rad=0.1,
            minimum_contact_n=10,
            maximum_reference_error_m=0.02,
            maximum_reference_angle_rad=np.deg2rad(10),
        )
        if any(
            not np.isfinite(settings[k]) or not 0 < settings[k] <= bounds[k]
            for k in required
        ):
            raise ValueError("Feedback settings exceed finite bounds")
        if not names or len(set(names)) != len(names):
            raise ValueError("Distinct wrist names required")
        if settings.get("support_load"):
            support_load_offset(0, 0, 1, 0.001, settings["support_load"])
        self.settings = settings
        self.offsets = {n: np.zeros(6) for n in names}
        self.updated = dict.fromkeys(names, 0)
        self.frozen = dict.fromkeys(names, 0)
        self.load_offset = 0.0
        self.load_updated = 0

    def update(
        self,
        name,
        actual_object,
        reference_object,
        reference_wrist,
        dt,
        *,
        contact_n,
        active,
        support=False,
        upward_force_n=0,
        weight_n=1,
    ):
        actual, reference, wrist = map(
            rigid_pose, (actual_object, reference_object, reference_wrist)
        )
        if (
            name not in self.offsets
            or not np.isfinite([dt, contact_n, upward_force_n, weight_n]).all()
            or dt <= 0
            or contact_n < 0
            or weight_n <= 0
        ):
            raise ValueError("Finite force observations and positive timestep required")
        s = self.settings
        error = object_relative_error(actual, reference, wrist)
        angle = Rotation.from_matrix(actual[:3, :3] @ reference[:3, :3].T).magnitude()
        allowed = (
            active
            and contact_n >= s["minimum_contact_n"]
            and np.linalg.norm(actual[:3, 3] - reference[:3, 3])
            <= s["maximum_reference_error_m"]
            and angle <= s["maximum_reference_angle_rad"]
        )
        offset = self.offsets[name]
        if allowed:
            for sl, speed, travel in (
                (slice(0, 3), s["max_speed_m_s"], s["max_offset_m"]),
                (slice(3, 6), s["max_angular_speed_rad_s"], s["max_angle_rad"]),
            ):
                velocity = bounded_vector(
                    s["gain_s_inv"] * (error[sl] - offset[sl]), speed
                )
                offset[sl] = bounded_vector(offset[sl] + dt * velocity, travel)
            self.updated[name] += 1
            if support and s.get("support_load"):
                self.load_offset = support_load_offset(
                    self.load_offset, upward_force_n, weight_n, dt, s["support_load"]
                )
                self.load_updated += 1
        else:
            self.frozen[name] += 1
        result = offset.copy()
        if support:
            result[2] += self.load_offset
            result[:3] = bounded_vector(result[:3], s["max_offset_m"])
        return result

    def diagnostics(self):
        return dict(
            offsets_world={k: v.tolist() for k, v in self.offsets.items()},
            updated_steps=self.updated.copy(),
            frozen_steps=self.frozen.copy(),
            support_load_offset_m=self.load_offset,
            support_load_updated_steps=self.load_updated,
        )
