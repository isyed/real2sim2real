"""Fitting reliability distinct from tracker confidence and validation labels.

Nearby fingertip identities can be occluded or ambiguous, not necessarily
incorrect. Reduce their influence symmetrically without guessing a new identity.
Anchor reliability uses only retained fitting observations and initial poses.
"""

import numpy as np

from .hand_point_tracks import close_fingertip_pairs
from .metric_hands import project


def policy_options(options):
    defaults = dict(
        ambiguity_radius_px=8.0,
        ambiguous_weight=0.1,
        propagate_to_finger=True,
        reassess_anchors=False,
        anchor_max_median_px=12.0,
        anchor_min_landmarks=3,
        weak_anchor_prior_weight=0.25,
    )
    if set(options) - set(defaults):
        raise ValueError("Unknown hand observation policy option")
    out = dict(defaults, **options)
    for name in ("propagate_to_finger", "reassess_anchors"):
        if type(out[name]) is not bool:
            raise ValueError("Policy switches must be booleans")
    for name, upper in [
        ("ambiguity_radius_px", 20),
        ("ambiguous_weight", 1),
        ("anchor_max_median_px", 50),
        ("weak_anchor_prior_weight", 1),
    ]:
        value = out[name]
        if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= upper:
            raise ValueError("Invalid bounded observation policy option: " + name)
    if (
        type(out["anchor_min_landmarks"]) is not int
        or not 3 <= out["anchor_min_landmarks"] <= 21
    ):
        raise ValueError("Require 3–21 landmarks to assess an anchor")
    return out


def observation_reliability(points, weights, joints, k, baseline, *, options):
    """Return copied fitting weights and anchor flags; never change observations.

    The minimum landmark count is an evidence floor, not proof that the full
    hand is observed. Weak anchors outside the selected fitting window remain
    immutable and must be reported as such by the caller.
    """
    options = policy_options(options)
    points, weights, joints, baseline = map(
        np.asarray, (points, weights, joints, baseline)
    )
    if (
        weights.ndim != 3
        or weights.shape[1:] != (2, 21)
        or points.shape != (*weights.shape, 2)
        or joints.shape != (*weights.shape, 3)
        or baseline.shape != weights.shape[:2]
    ):
        raise ValueError("Matching Nx2x21 hand observations and poses required")
    if (
        not np.isfinite(weights).all()
        or not ((weights >= 0) & (weights <= 1)).all()
        or not np.isfinite(points[weights > 0]).all()
    ):
        raise ValueError("Finite observed coordinates and weights in [0,1] required")
    factors = np.ones_like(weights, dtype=float)
    ambiguous = np.zeros_like(weights, dtype=bool)
    reliable = np.zeros_like(baseline, dtype=bool)
    rows = []
    for i in range(len(weights)):
        per_hand = []
        for h in range(2):
            pairs = close_fingertip_pairs(
                points[i, h],
                weights[i, h],
                separation_px=options["ambiguity_radius_px"],
            )
            for pair in pairs:
                for tip in pair["joints"]:
                    ids = (
                        np.arange(tip - 3, tip + 1)
                        if options["propagate_to_finger"]
                        else [tip]
                    )
                    ambiguous[i, h, ids] = True
            factors[i, h, ambiguous[i, h]] = options["ambiguous_weight"]
            usable = (weights[i, h] > 0) & ~ambiguous[i, h]
            usable &= np.isfinite(joints[i, h]).all(-1) & (joints[i, h, :, 2] > 0)
            errors = np.linalg.norm(
                project(joints[i, h, usable], k) - points[i, h, usable], axis=-1
            )
            median = float(np.median(errors)) if len(errors) else None
            evidence_ok = (
                len(errors) >= options["anchor_min_landmarks"]
                and median <= options["anchor_max_median_px"]
            )
            reliable[i, h] = bool(baseline[i, h] and evidence_ok)
            per_hand.append(
                dict(
                    close_fingertip_pairs=pairs,
                    reduced_observed_joints=np.flatnonzero(
                        ambiguous[i, h] & (weights[i, h] > 0)
                    ).tolist(),
                    unambiguous_observed_joints=int(usable.sum()),
                    initial_unambiguous_median_px=median,
                    baseline_was_accepted=bool(baseline[i, h]),
                    reliable_baseline_anchor=bool(reliable[i, h]),
                )
            )
        rows.append(per_hand)
    return dict(
        weights=weights * factors,
        factors=factors,
        ambiguous=ambiguous,
        reliable_baseline=reliable,
        records=rows,
        options=options,
    )


def anchor_prior_weights(baseline, reliable, weak_weight):
    """Only weak, previously accepted poses lose correction-prior strength."""
    baseline, reliable = np.asarray(baseline, bool), np.asarray(reliable, bool)
    if baseline.shape != reliable.shape or not 0 < weak_weight <= 1:
        raise ValueError("Matching masks and prior weight in (0,1] required")
    return np.where(baseline & ~reliable, weak_weight, 1.0)


def image_normalization_mass(weights, original=None):
    """Preserve original image/depth balance when reducing observation weights."""
    weights = np.asarray(weights)
    reference = weights if original is None else np.asarray(original)
    if (
        weights.ndim != 2
        or weights.shape[1] != 21
        or reference.shape != weights.shape
        or not np.isfinite(weights).all()
        or not np.isfinite(reference).all()
        or (weights < 0).any()
        or (reference < weights).any()
        or (reference > 1).any()
        or (reference.sum(1) <= 0).any()
    ):
        raise ValueError("Finite original weights must bound fitted Nx21 weights")
    return reference.sum(1)
