"""One-shot unloaded-pad seeking within the existing acquisition clock.

This changes the normal servo's rate, not its force goals, geometry guards,
joint limits or offset budget. A first solved pad contact permanently ends
fast seeking for that pad; loss of contact does not restart acquisition.
"""

import numpy as np


def validate(settings, activation_fraction):
    if set(settings) != {
        "start_fraction",
        "end_after_close_s",
        "gain_m_per_n_s",
        "max_normal_speed_m_s",
    } or any(isinstance(v, bool) or not np.isfinite(v) for v in settings.values()):
        raise ValueError("Explicit finite unloaded acquisition settings required")
    if not (
        0 <= settings["start_fraction"] <= activation_fraction
        and 0 <= settings["end_after_close_s"] <= 1
        and 0 < settings["gain_m_per_n_s"] <= 0.003
        and 0 < settings["max_normal_speed_m_s"] <= 0.006
    ):
        raise ValueError("Unloaded acquisition exceeds bounded clock/rate limits")


def active(time_s, rollout, settings):
    start = rollout["approach_s"] + settings["start_fraction"] * rollout["close_s"]
    end = rollout["approach_s"] + rollout["close_s"] + settings["end_after_close_s"]
    return start <= time_s < end


def seeking(pad, force_n, *, active, threshold_n):
    """Observe first contact even outside the seeking window; never rearm."""
    if not np.isfinite(force_n) or force_n < 0:
        raise ValueError("Finite nonnegative solved pad force required")
    pad["acquisition_contact_seen"] = (
        pad.get("acquisition_contact_seen", False) or force_n >= threshold_n
    )
    return bool(
        active and pad["group"] != "thumb" and not pad["acquisition_contact_seen"]
    )
