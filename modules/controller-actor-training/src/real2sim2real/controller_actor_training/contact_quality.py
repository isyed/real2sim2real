"""Sampled contact quality evidence; never a hardware-readiness verdict."""

import numpy as np


def contact_quality(
    forces_n,
    drift_m,
    angles_deg,
    *,
    minimum_force_n,
    maximum_drift_m,
    maximum_angle_deg,
):
    """Summarize one explicitly selected loaded phase without hiding pad loss.

    Fractions count saved samples, not time, independent trials or probability.
    Callers must retain the sample clock, phase and source hashes separately.
    """
    forces = np.asarray(forces_n, float)
    drift = np.asarray(drift_m, float)
    angles = np.asarray(angles_deg, float)
    limits = np.array([minimum_force_n, maximum_drift_m, maximum_angle_deg])
    if (
        forces.ndim != 2
        or min(forces.shape, default=0) == 0
        or drift.shape != (len(forces),)
        or angles.shape != drift.shape
        or not all(np.isfinite(x).all() for x in [forces, drift, angles, limits])
        or any((x < 0).any() for x in [forces, drift, angles])
        or (limits <= 0).any()
    ):
        raise ValueError(
            "Nonempty aligned finite contact samples and positive limits required"
        )
    loss = forces < minimum_force_n
    return {
        "samples": len(forces),
        "minimum_pad_force_n": forces.min(axis=0).tolist(),
        "any_pad_loss_fraction": float(loss.any(axis=1).mean()),
        "all_pad_loss_fraction": float(loss.all(axis=1).mean()),
        "peak_drift_m": float(drift.max()),
        "drift_exceedance_fraction": float((drift > maximum_drift_m).mean()),
        "peak_angle_deg": float(angles.max()),
        "angle_exceedance_fraction": float((angles > maximum_angle_deg).mean()),
        "sampled_limits_pass": bool(
            not loss.any()
            and drift.max() <= maximum_drift_m
            and angles.max() <= maximum_angle_deg
        ),
    }
