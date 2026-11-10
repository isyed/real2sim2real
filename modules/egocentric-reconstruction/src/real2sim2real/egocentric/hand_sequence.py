"""Sequence selection and admission for observed hand articulation.

All derivatives use capture time. Missing observations split components; fixed
anchors may constrain motion but never create measured landmarks or depth.
"""

import numpy as np


def sequence_options(options):
    defaults = dict(
        context_frames=0,
        max_joint_delta_deg=30.0,
        max_rigid_delta_deg=10.0,
        max_translation_delta_m=0.015,
        temporal_weight=1.0,
        acceleration_scale_m_s2=5.0,
        velocity_scale_m_s=0.6,
        correspondence_iterations=4,
        max_step_s=0.1,
        anchor_data_weight=1.0,
        boundary_velocity_weight=0.0,
        fit_all_supported=False,
    )
    if set(options) - set(defaults):
        raise ValueError("Unknown sequence fitting option")
    result = dict(defaults, **options)
    if type(result["fit_all_supported"]) is not bool:
        raise ValueError("fit_all_supported must be a boolean")
    for key, upper in [("context_frames", 10), ("correspondence_iterations", 10)]:
        minimum = 0 if key == "context_frames" else 1
        if type(result[key]) is not int or not minimum <= result[key] <= upper:
            raise ValueError("Invalid sequence integer option: " + key)
    for key, upper in [
        ("max_joint_delta_deg", 45),
        ("max_rigid_delta_deg", 20),
        ("max_translation_delta_m", 0.03),
        ("temporal_weight", 10),
        ("acceleration_scale_m_s2", 20),
        ("velocity_scale_m_s", 3),
        ("max_step_s", 0.5),
        ("anchor_data_weight", 5),
    ]:
        value = result[key]
        if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= upper:
            raise ValueError("Invalid bounded sequence option: " + key)
    boundary = result["boundary_velocity_weight"]
    if (
        isinstance(boundary, bool)
        or not np.isfinite(boundary)
        or not 0 <= boundary <= 10
    ):
        raise ValueError("Invalid boundary velocity weight")
    return result


def protected_boundary_edges(selected, baseline):
    """Only baseline/baseline seams supply a measured boundary velocity target."""
    selected, baseline = np.asarray(selected, bool), np.asarray(baseline, bool)
    if selected.ndim != 1 or selected.shape != baseline.shape:
        raise ValueError("Matching one-hand masks required")
    return (selected[1:] != selected[:-1]) & baseline[1:] & baseline[:-1]


def select_components(recovered, observed, times, *, context_frames=0, max_step_s=0.1):
    if isinstance(max_step_s, bool) or not np.isfinite(max_step_s) or max_step_s <= 0:
        raise ValueError("Positive finite maximum capture step required")
    recovered, observed, times = (
        np.asarray(recovered, bool),
        np.asarray(observed, bool),
        np.asarray(times),
    )
    if (
        recovered.shape != observed.shape
        or recovered.ndim != 2
        or recovered.shape[1] != 2
    ):
        raise ValueError("Require matching Nx2 recovery and observation masks")
    if (
        times.shape != (len(observed),)
        or not np.isfinite(times).all()
        or not (np.diff(times) > 0).all()
    ):
        raise ValueError("Require finite increasing capture timestamps")
    if (recovered & ~observed).any():
        raise ValueError("Every recovered pose must have measured support")
    if type(context_frames) is not int or not 0 <= context_frames <= 10:
        raise ValueError("Invalid context frame count")
    selected = recovered.copy()
    for h in range(2):
        for i in np.flatnonzero(recovered[:, h]):
            for direction in [-1, 1]:
                for offset in range(1, context_frames + 1):
                    j, previous = i + direction * offset, i + direction * (offset - 1)
                    if (
                        not 0 <= j < len(times)
                        or not observed[j, h]
                        or abs(times[j] - times[previous]) > max_step_s
                    ):
                        break
                    selected[j, h] = True
    components = []
    for h in range(2):
        ids = np.flatnonzero(selected[:, h])
        if not len(ids):
            continue
        cuts = np.asarray(
            [
                b - a > 3
                or not observed[a : b + 1, h].all()
                or (np.diff(times[a : b + 1]) > max_step_s).any()
                for a, b in zip(ids[:-1], ids[1:])
            ]
        )
        for group in np.split(ids, np.flatnonzero(cuts) + 1):
            lo, hi = int(group[0]), int(group[-1]) + 1
            # Two measured neighbors constrain boundary velocity/acceleration.
            start, end = lo, hi
            for _ in range(2):
                if (
                    start
                    and observed[start - 1, h]
                    and times[start] - times[start - 1] <= max_step_s
                ):
                    start -= 1
                if (
                    end < len(times)
                    and observed[end, h]
                    and times[end] - times[end - 1] <= max_step_s
                ):
                    end += 1
            components.append(
                dict(hand=h, selected=group, window=np.arange(start, end))
            )
    return selected, components


def affected_edges(selected):
    selected = np.asarray(selected, bool)
    if selected.ndim != 1:
        raise ValueError("One hand's selection mask required")
    return selected[:-1] | selected[1:], selected[:-2] | selected[1:-1] | selected[2:]


def derivatives(joints, times):
    dt = np.diff(times)
    if (
        not np.isfinite(times).all()
        or not (dt > 0).all()
        or not np.isfinite(joints).all()
    ):
        raise ValueError("Finite joints and increasing times required")
    velocity = np.diff(joints, axis=0) / dt[:, None, None]
    acceleration = 2 * np.diff(velocity, axis=0) / (dt[1:] + dt[:-1])[:, None, None]
    return velocity, acceleration


def compare_motion(before, after, selected, times):
    edges, triples = affected_edges(selected)
    if not triples.any():
        raise ValueError("Sequence fitting requires an observed temporal triple")
    metrics = []
    for joints in [before, after]:
        v, a = derivatives(joints, times)
        speed, accel = (
            np.linalg.norm(v[edges], axis=-1),
            np.linalg.norm(a[triples], axis=-1),
        )
        # Boundaries mean transitions from fixed to variable poses.
        seams = np.diff(np.asarray(selected, int)) != 0
        seam_speed = np.linalg.norm(v[seams], axis=-1)
        metrics.append(
            dict(
                acceleration_rms_m_s2=float(np.sqrt(np.mean(accel**2))),
                acceleration_max_m_s2=float(accel.max()),
                speed_p90_m_s=float(np.quantile(speed, 0.9)),
                speed_max_m_s=float(speed.max()),
                boundary_speed_max_m_s=float(seam_speed.max())
                if seam_speed.size
                else None,
            )
        )
    return dict(before=metrics[0], after=metrics[1])


def motion_rejections(comparison):
    before, after = comparison["before"], comparison["after"]
    reasons = []
    if after["acceleration_rms_m_s2"] > 0.9 * before["acceleration_rms_m_s2"]:
        reasons.append("insufficient_sequence_acceleration_improvement")
    if after["acceleration_max_m_s2"] > before["acceleration_max_m_s2"] + 1e-6:
        reasons.append("sequence_acceleration_peak_regression")
    if after["speed_max_m_s"] > before["speed_max_m_s"] + 0.05:
        reasons.append("sequence_speed_peak_regression")
    if (
        before["boundary_speed_max_m_s"] is not None
        and after["boundary_speed_max_m_s"] > before["boundary_speed_max_m_s"] + 0.05
    ):
        reasons.append("sequence_boundary_regression")
    return reasons
