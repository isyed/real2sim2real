"""Phase semantics for continuous pressure residuals, independent of physics writes."""

import numpy as np

STAGES = ("approach", "acquisition", "carry", "placement", "release", "settled")


def validate_baseline(result):
    p = result["continuous_protocol"]
    from .finger_release import validate_release

    if (
        p.get("schema") != "real2sim2real.continuous-reference-protocol/1"
        or result.get("motion_condition") != "demonstrated"
        or not result.get("demonstrated_motion_executed")
        or not p.get("native_timing_preserved")
        or not p.get("raw_reference_preserved")
        or not p.get("release_planned")
        or p.get("release_disabled_for_stationary_diagnostic")
    ):
        raise ValueError("Complete native-clock continuous motion and release required")
    validate_release(p["finger_release"])
    clock = np.array(
        [
            p["acquisition_end_s"],
            p["capture_duration_s"],
            p["finger_release"]["start_s"],
            p["duration_s"],
        ],
        float,
    )
    if not np.isfinite(clock).all() or not (np.diff(clock) > 0).all():
        raise ValueError(
            "Ordered finite acquisition/capture/release/end clock required"
        )
    dt = result["physics_timestep_s"]
    times = np.array([r["time_s"] for r in result["records"]])
    if (
        not np.isfinite(dt)
        or not 0 < dt <= 0.01
        or len(times) < 2
        or not np.isfinite(times).all()
        or not (np.diff(times) > 0).all()
        or times[0] > dt + 1e-9
        or abs(times[-1] - clock[-1]) > dt / 2 + 1e-8
    ):
        raise ValueError("Complete continuous physical clock required")


def stage_at(time_s, phase, protocol, release_started_s, release_fraction):
    # Observe the actual supported-release latch, not just the command schedule.
    if release_started_s is not None:
        return "settled" if release_fraction >= 1.0 else "release"
    if phase == "approach":
        return "approach"
    if time_s < protocol["acquisition_end_s"]:
        return "acquisition"
    if time_s >= protocol["capture_duration_s"]:
        return "placement"
    return "carry"


def phase_costs(
    costs,
    *,
    stage,
    scheduled_phase,
    position_error,
    rotation_error,
    drift,
    anchor_valid,
    reference_valid,
    contact_cost,
    clearance_m,
    minimum_clearance_m,
    table_load_fraction,
    hand_force_n,
    release_fraction,
    open_error_rad,
):
    """Provisional costs: retain during carry, unload only after supported release."""
    c = dict(costs)
    tracking = stage != "approach" and reference_valid
    c.update(
        position=float(np.dot(position_error, position_error) / 0.02**2)
        if tracking
        else 0.0,
        rotation=float(np.dot(rotation_error, rotation_error) / np.deg2rad(10) ** 2)
        if tracking
        else 0.0,
        grasp=float(np.dot(drift, drift) / 0.01**2)
        if stage == "carry" and anchor_valid
        else 0.0,
        contact=float(contact_cost) if stage == "carry" else 0.0,
        clearance=float(max(0, minimum_clearance_m - clearance_m) ** 2 / 0.01**2)
        if stage == "carry" and scheduled_phase == "hold"
        else 0.0,
        placement=float(
            (1 - np.clip(table_load_fraction, 0, 1)) ** 2
            + max(0, clearance_m - 0.005) ** 2 / 0.005**2
        )
        if stage in ("placement", "release", "settled")
        else 0.0,
        release=float(
            release_fraction * (hand_force_n / 10) ** 2
            + ((open_error_rad or 0) / 0.1) ** 2 * (stage == "settled")
        )
        if stage in ("release", "settled")
        else 0.0,
        table_penetration=float(max(0, -clearance_m - 0.003) ** 2 / 0.003**2),
    )
    return c
