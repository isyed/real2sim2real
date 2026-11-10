"""Visibility-aware image point observations, independent of a 3D hand model.

Point identities originate in the seed detector. Tracking skin appearance does
not establish anatomical accuracy; all outputs remain diagnostic observations.
"""

import numpy as np

from .hand_observation_fit import visible_landmarks


def close_fingertip_pairs(points, weights, *, separation_px=8.0):
    """Flag distinct fingertip identities sharing a small image neighborhood.

    This is an observability warning, not proof of a mislabeled point: distinct
    fingers can overlap in projection. Such pairs need extra identity evidence
    before being treated as two independently observed anatomical targets.
    """
    points, weights = np.asarray(points), np.asarray(weights)
    if points.shape != (21, 2) or weights.shape != (21,):
        raise ValueError("One 21-joint hand observation required")
    if not np.isfinite(separation_px) or separation_px <= 0:
        raise ValueError("Positive finite image separation required")
    tips = [
        j for j in (4, 8, 12, 16, 20) if weights[j] > 0 and np.isfinite(points[j]).all()
    ]
    pairs = []
    for i, a in enumerate(tips):
        for b in tips[i + 1 :]:
            distance = float(np.linalg.norm(points[a] - points[b]))
            if distance <= separation_px:
                pairs.append(
                    dict(
                        joints=[a, b],
                        separation_px=distance,
                        requires_identity_review=True,
                    )
                )
    return pairs


def select_seeds(observations, masks, frame_ids, seed_ids, heldout_ids):
    """Use only explicitly selected detector frames, never evaluation frames."""
    if set(seed_ids) & set(heldout_ids):
        raise ValueError("Seed and evaluation frames must be disjoint")
    if not set(seed_ids).issubset(set(map(int, frame_ids))):
        raise ValueError("Seed frame IDs must belong to the input window")
    seeds, hands, joints = [], [], []
    for i, frame in enumerate(frame_ids):
        if frame not in seed_ids:
            continue
        for h in range(2):
            if not observations["accepted"][i, h]:
                continue
            uv = observations["keypoints_px"][i, h]
            visible = visible_landmarks(uv, masks[i, h])
            for j in np.flatnonzero(visible):
                seeds.append([i, uv[j, 1], uv[j, 0]])
                hands.append(h)
                joints.append(j)
    if not seeds:
        raise ValueError("No visible independent detector seeds")
    return dict(
        query_tyx=np.asarray(seeds, np.float32),
        hand=np.asarray(hands, np.int64),
        joint=np.asarray(joints, np.int64),
    )


def confidence(occlusion, expected_dist):
    """Official TAPIR visible-and-accurate score from two prediction logits."""
    sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -60, 60)))
    return (1 - sigmoid(occlusion)) * (1 - sigmoid(expected_dist))


def mask_support(xy, masks, hands):
    result = np.zeros(xy.shape[:2], bool)
    for q, hand in enumerate(hands):
        for t, point in enumerate(xy[q]):
            if np.isfinite(point).all():
                x, y = np.rint(point).astype(int)
                mask = masks[t, hand]
                result[q, t] = (
                    0 <= x < mask.shape[1]
                    and 0 <= y < mask.shape[0]
                    and mask[y, x]
                    and not masks[t, 1 - hand, y, x]
                )
    return result


def aggregate_tracks(
    tracks,
    reverse_tracks,
    scores,
    reverse_scores,
    seeds,
    masks,
    *,
    min_score=0.8,
    max_cycle_px=5.0,
    max_agreement_px=8.0,
    min_seeds=2,
):
    """Retain only visible, independently re-queried, multi-seed agreement.

    Reverse tracks are re-seeded at a different observed time, then aligned to
    the forward timeline. Reversing the same query alone is not a cycle test.
    No 3D model projection enters selection or weighting.
    """
    if not np.isfinite(min_score) or not 0 < min_score <= 1:
        raise ValueError("Tracking confidence threshold must be in (0, 1]")
    if any(not np.isfinite(x) or x <= 0 for x in (max_cycle_px, max_agreement_px)):
        raise ValueError("Tracking pixel limits must be finite and positive")
    if type(min_seeds) is not int or min_seeds < 2:
        raise ValueError("Require at least two distinct seed frames")
    tracks, reverse_tracks = np.asarray(tracks), np.asarray(reverse_tracks)
    if (
        tracks.shape != reverse_tracks.shape
        or tracks.ndim != 3
        or tracks.shape[-1] != 2
    ):
        raise ValueError("Require matching Q,T,2 point trajectories")
    if scores.shape != tracks.shape[:2] or reverse_scores.shape != scores.shape:
        raise ValueError("Point confidence shape differs")
    cycle = np.linalg.norm(tracks - reverse_tracks, axis=-1)
    usable = (
        np.isfinite(tracks).all(-1)
        & np.isfinite(reverse_tracks).all(-1)
        & (scores >= min_score)
        & (reverse_scores >= min_score)
        & (cycle <= max_cycle_px)
        & mask_support(tracks, masks, seeds["hand"])
        & mask_support(reverse_tracks, masks, seeds["hand"])
    )
    n = tracks.shape[1]
    points = np.full((n, 2, 21, 2), np.nan, np.float32)
    weights = np.zeros((n, 2, 21), np.float32)
    counts = np.zeros((n, 2, 21), np.int32)
    spread = np.full((n, 2, 21), np.nan, np.float32)
    for t in range(n):
        for h in range(2):
            for j in range(21):
                ids = np.flatnonzero(
                    usable[:, t] & (seeds["hand"] == h) & (seeds["joint"] == j)
                )
                distinct = np.unique(seeds["query_tyx"][ids, 0])
                counts[t, h, j] = len(distinct)
                if len(distinct) < min_seeds:
                    continue
                observed = tracks[ids, t]
                center = np.median(observed, axis=0)
                deviation = np.linalg.norm(observed - center, axis=-1).max()
                spread[t, h, j] = deviation
                if deviation > max_agreement_px:
                    continue
                points[t, h, j] = center
                weights[t, h, j] = np.minimum(
                    scores[ids, t], reverse_scores[ids, t]
                ).min()
    return dict(
        keypoints_px=points,
        weights=weights,
        supporting_seeds=counts,
        disagreement_px=spread,
        query_usable=usable,
        cycle_error_px=cycle,
    )
