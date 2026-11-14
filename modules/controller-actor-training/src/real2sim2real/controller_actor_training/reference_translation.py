"""Bounded source translation proposals preserving a baseline orientation."""

import numpy as np


def phase_window(reference_s, start_s, end_s, ramp_s):
    """Smooth correction envelope with exact zero outside a declared phase.

    Times belong to the caller's reference clock, not simulation timestep.
    This preserves a frozen controller through excluded acquisition/release
    phases and avoids applying an inversion correction during another phase.
    """
    if (
        not np.isfinite([reference_s, start_s, end_s, ramp_s]).all()
        or not start_s < end_s
        or not 0 < ramp_s <= (end_s - start_s) / 2
    ):
        raise ValueError("Finite increasing phase interval and bounded ramp required")
    x = np.clip(
        min((reference_s - start_s) / ramp_s, (end_s - reference_s) / ramp_s), 0.0, 1.0
    )
    return float(x * x * (3 - 2 * x))


def bounded_translation(baseline_pose, source_position_m, *, gain, limit_m, blend=1.0):
    """Return a proposal, not a collision or loaded-control acceptance.

    Source/world alignment and timing are caller-owned explicit contracts. Zero
    gain/blend exactly preserves the baseline. This cannot resolve ambiguous yaw.
    """
    pose = np.asarray(baseline_pose, float)
    position = np.asarray(source_position_m, float)
    if (
        pose.shape != (4, 4)
        or position.shape != (3,)
        or not np.isfinite(pose).all()
        or not np.isfinite(position).all()
        or not np.isfinite([gain, limit_m, blend]).all()
        or not 0 <= gain <= 1
        or not 0 <= blend <= 1
        or limit_m <= 0
        or not np.allclose(pose[3], [0, 0, 0, 1])
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        or np.linalg.det(pose[:3, :3]) < 0
    ):
        raise ValueError(
            "Rigid metric pose, finite position and bounded gains required"
        )
    correction = gain * (position - pose[:3, 3])
    correction *= min(1.0, limit_m / max(np.linalg.norm(correction), 1e-12))
    result = pose.copy()
    result[:3, 3] += blend * correction
    return result
