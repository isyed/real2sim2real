"""Report reconstruction warnings separately from tracker execution validity.

These configurable per-frame limits are review heuristics, not universal physical
constraints. The report does not approve a reference or modify predictions.
"""

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class QualityLimits:
    min_image_iou: float = 0.4
    max_low_iou_fraction: float = 0.1
    max_translation_step_m: float = 0.08
    max_rotation_step_deg: float = 30.0

    def validate(self):
        values = asdict(self)
        if any(isinstance(x, bool) or not np.isfinite(x) for x in values.values()):
            raise ValueError("Quality limits must be finite numeric values")
        if not 0 <= self.min_image_iou <= 1 or not 0 <= self.max_low_iou_fraction <= 1:
            raise ValueError("IoU/fraction limits must be in [0,1]")
        if (
            self.max_translation_step_m <= 0
            or not 0 < self.max_rotation_step_deg <= 180
        ):
            raise ValueError("Motion review limits must be positive")


def quality_report(
    poses,
    timestamps_s,
    *,
    image_iou=None,
    tracker_reliable=None,
    limits=QualityLimits(),
):
    limits.validate()
    p = np.asarray(poses, float)
    t = np.asarray(timestamps_s, float)
    if (
        p.ndim != 3
        or p.shape[1:] != (4, 4)
        or not len(p)
        or not np.isfinite(p).all()
        or t.shape != (len(p),)
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
        or not np.allclose(p[:, 3], [0, 0, 0, 1], atol=1e-7, rtol=0)
    ):
        raise ValueError("Expected complete finite poses and increasing timestamps")
    rotation = p[:, :3, :3]
    if not np.allclose(
        rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-4, rtol=0
    ) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0):
        raise ValueError("Object rotations must be proper rigid rotations")
    n = len(p)
    flags = []
    if image_iou is None:
        score = np.full(n, np.nan)
    else:
        score = np.asarray(image_iou, float).reshape(-1)
        if score.shape != (n,):
            raise ValueError("Image-score timeline differs from poses")
    finite = np.isfinite(score)
    if np.any((score[finite] < 0) | (score[finite] > 1)):
        raise ValueError("Image IoU must be in [0,1]")
    missing = np.flatnonzero(~finite)
    low = np.flatnonzero(finite & (score < limits.min_image_iou))
    if len(missing):
        flags.append("image_agreement_unassessed_frames")
    if len(low) / n > limits.max_low_iou_fraction:
        flags.append("poor_image_agreement")
    steps = np.linalg.norm(np.diff(p[:, :3, 3], axis=0), axis=1)
    relative = rotation[1:] @ rotation[:-1].transpose(0, 2, 1)
    angles = np.rad2deg(
        np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
    )
    position_frames = (
        np.flatnonzero(steps > limits.max_translation_step_m) + 1
    ).tolist()
    rotation_frames = (
        np.flatnonzero(angles > limits.max_rotation_step_deg) + 1
    ).tolist()
    if position_frames:
        flags.append("translation_discontinuities")
    if rotation_frames:
        flags.append("rotation_discontinuities")
    if tracker_reliable is None:
        reliable_count = None
        reliability_conflicts = []
    else:
        reliable = np.asarray(tracker_reliable).reshape(-1)
        if reliable.shape != (n,) or not np.isin(reliable, [0, 1]).all():
            raise ValueError("Invalid tracker reliability timeline")
        reliable_count = int(reliable.sum())
        reliability_conflicts = np.flatnonzero(reliable.astype(bool) & finite
                                               & (score < limits.min_image_iou)).tolist()
        if reliability_conflicts:
            flags.append('tracker_reliability_conflicts_with_image')
    return dict(
        schema="real2sim2real.reconstruction-quality/1",
        frames=n,
        status="review_required" if flags else "no_configured_warnings",
        flags=flags,
        limits=asdict(limits),
        tracker_reliable_frames=reliable_count,
        tracker_reliability_scope='Producer-reported flag; not independent image, depth or physical validation',
        image_agreement=dict(
            mean_iou=float(score[finite].mean()) if finite.any() else None,
            missing_frames=missing.tolist(),
            low_iou_frames=low.tolist(),
            producer_reliable_but_low_iou_frames=reliability_conflicts,
            scope="producer-reported full-image mask IoU",
        ),
        continuity=dict(
            max_translation_step_m=float(steps.max()) if len(steps) else 0.0,
            max_rotation_step_deg=float(angles.max()) if len(angles) else 0.0,
            translation_warning_frames=position_frames,
            rotation_warning_frames=rotation_frames,
        ),
        median_frame_interval_s=float(np.median(np.diff(t))) if len(t) > 1 else None,
        unassessed=[
            "absolute_metric_accuracy",
            "human_surface_alignment",
            "support_placement",
            "physical_contact",
            "robot_execution",
        ],
        approved_for_retargeting=False,
        approved_for_actor_training=False,
        predictions_modified=False,
        limitations=[
            "Per-frame motion limits must be interpreted with the source sampling rate.",
            "Mask agreement uses the producer segmentation and is not independent ground truth.",
            "No configured warnings does not establish physical correctness or replace comparison with a retained candidate.",
        ],
    )
