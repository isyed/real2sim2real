"""Bounded robot corrections from observed contact and object orientation.

The simulator supplies these observations in the current pilot. A real robot
needs corresponding sensing; none of these corrections act on the object.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def relative_translation_drift(object_positions, wrist_positions, wrist_rotations):
    """Object-origin displacement in the moving wrist frame, in metres."""
    relative = np.asarray(object_positions) - np.asarray(wrist_positions)
    rotations = np.asarray(wrist_rotations)
    if (
        relative.ndim != 2
        or relative.shape[1] != 3
        or rotations.shape != (len(relative), 3, 3)
    ):
        raise ValueError("Matching object/wrist positions and wrist rotations required")
    if (
        not len(relative)
        or not np.isfinite(relative).all()
        or not np.isfinite(rotations).all()
    ):
        raise ValueError("Nonempty finite hold observations required")
    local = np.einsum("nji,nj->ni", rotations, relative)
    return float(np.linalg.norm(local - local[0], axis=1).max())


def contact_closure(offset, force_n, dt, settings):
    values = [
        offset,
        force_n,
        dt,
        settings["target_force_n"],
        settings["gain_rad_per_n_s"],
        settings["max_speed_rad_s"],
        settings["max_offset_rad"],
    ]
    if not np.isfinite(values).all() or min(values) < 0 or dt == 0:
        raise ValueError(
            "Contact feedback requires finite nonnegative bounds and positive dt"
        )
    speed = np.clip(
        settings["gain_rad_per_n_s"] * (settings["target_force_n"] - force_n),
        -settings["max_speed_rad_s"],
        settings["max_speed_rad_s"],
    )
    return float(np.clip(offset + dt * speed, 0, settings["max_offset_rad"]))


def orientation_correction(offset, current, target, wrist_jacobian, dt, settings):
    values = [
        dt,
        settings["gain_s_inv"],
        settings["max_speed_rad_s"],
        settings["max_offset_rad"],
    ]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Orientation feedback requires finite positive bounds")
    offset, current, target, jac = map(
        np.asarray, (offset, current, target, wrist_jacobian)
    )
    if not all(np.isfinite(a).all() for a in (offset, current, target, jac)):
        raise ValueError("Nonfinite orientation feedback observations")
    error = Rotation.from_matrix(target @ current.T).as_rotvec()
    # Damped least squares avoids unbounded commands near a fixture singularity.
    speed = jac.T @ np.linalg.solve(
        jac @ jac.T + 1e-6 * np.eye(3), error * settings["gain_s_inv"]
    )
    speed *= min(1.0, settings["max_speed_rad_s"] / max(np.linalg.norm(speed), 1e-12))
    result = offset + dt * speed
    result *= min(1.0, settings["max_offset_rad"] / max(np.linalg.norm(result), 1e-12))
    return result
