"""Source-clock motion contracts for objects whose axial spin is unobserved."""

import numpy as np
from scipy.spatial.transform import Rotation

from .camera import rigid


def axis_frame(direction, previous=None):
    """Choose a continuous orientation gauge without claiming measured spin."""
    z = np.asarray(direction, float)
    if z.shape != (3,) or not np.isfinite(z).all() or np.linalg.norm(z) < 1e-10:
        raise ValueError("Finite nonzero directed axis required")
    z = z / np.linalg.norm(z)
    old = np.eye(3) if previous is None else np.asarray(previous, float)
    if (
        old.shape != (3, 3)
        or not np.isfinite(old).all()
        or not np.allclose(old.T @ old, np.eye(3), atol=1e-6)
        or np.linalg.det(old) < 0
    ):
        raise ValueError("Previous frame must be a proper rotation")
    cross = np.cross(old[:, 2], z)
    cos = float(np.clip(old[:, 2] @ z, -1, 1))
    if np.linalg.norm(cross) < 1e-10:
        rotation = (
            np.eye(3)
            if cos > 0
            else Rotation.from_rotvec(np.pi * old[:, 0]).as_matrix()
        )
    else:
        rotation = Rotation.from_rotvec(
            cross / np.linalg.norm(cross) * np.arctan2(np.linalg.norm(cross), cos)
        ).as_matrix()
    return rotation @ old


def motion_summary(poses, accepted, frame_ids, timestamps_s):
    p = np.asarray(poses, float)
    ok = np.asarray(accepted, bool)
    ids = np.asarray(frame_ids)
    t = np.asarray(timestamps_s, float)
    if (
        p.shape != (len(t), 4, 4)
        or ok.shape != t.shape
        or ids.shape != t.shape
        or len(t) < 2
        or not np.isfinite(t).all()
        or (np.diff(t) <= 0).any()
        or (np.diff(ids) <= 0).any()
        or not np.isfinite(p[ok]).all()
        or not np.issubdtype(ids.dtype, np.integer)
    ):
        raise ValueError(
            "Increasing source clock/IDs and finite accepted poses required"
        )
    for pose in p[ok]:
        rigid(pose)
    adjacent = ok[:-1] & ok[1:] & (np.diff(ids) == 1)
    speed = np.linalg.norm(np.diff(p[:, :3, 3], axis=0), axis=1) / np.diff(t)
    cosine = np.einsum("ij,ij->i", p[:-1, :3, 2], p[1:, :3, 2])
    rate = np.rad2deg(np.arccos(np.clip(cosine, -1, 1))) / np.diff(t)
    gaps = []
    start = None
    for i, v in enumerate(np.r_[ok, True]):
        if not v and start is None:
            start = i
        if v and start is not None:
            gaps.append(
                {
                    "first_frame": int(ids[start]),
                    "last_frame": int(ids[i - 1]),
                    "duration_between_bracketing_observations_s": float(
                        t[min(i, len(t) - 1)] - t[max(0, start - 1)]
                    ),
                }
            )
            start = None
    return {
        "frames": len(t),
        "accepted_frames": int(ok.sum()),
        "capture_start_s": float(t[0]),
        "capture_end_s": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "maximum_adjacent_translation_speed_m_s": float(speed[adjacent].max())
        if adjacent.any()
        else None,
        "maximum_adjacent_axis_rate_deg_s": float(rate[adjacent].max())
        if adjacent.any()
        else None,
        "gaps": gaps,
        "gaps_filled": False,
        "axial_spin_observed": False,
        "time_warped": False,
    }
