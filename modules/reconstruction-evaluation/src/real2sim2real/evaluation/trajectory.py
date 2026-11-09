"""Single-object trajectory errors after body registration and ONE world alignment."""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.integrate import trapezoid


def score(predicted, reference, registration, reference_points, *, accepted=None):
    predicted = np.asarray(predicted, float)
    reference = np.asarray(reference, float)
    if (
        predicted.shape != reference.shape
        or predicted.ndim != 3
        or predicted.shape[1:] != (4, 4)
    ):
        raise ValueError("Matching Tx4x4 trajectories required")
    valid = np.isfinite(predicted).all((1, 2)) & np.isfinite(reference).all((1, 2))
    if accepted is not None:
        valid &= np.asarray(accepted, bool)
    if not valid[0]:
        raise ValueError(
            "An observed initial pose is required for fixed world alignment"
        )
    # Register body orientation and anchor only. Never rescale trajectory translation.
    r = np.asarray(registration["rotation"])
    t = np.asarray(registration["translation"])
    scale = registration["scale"]
    center = np.asarray(registration["reference_center"])
    anchor = r.T @ (center - t) / scale
    candidate = predicted.copy()
    truth = reference.copy()
    candidate[:, :3, 3] += np.einsum("tij,j->ti", candidate[:, :3, :3], anchor)
    candidate[:, :3, :3] = candidate[:, :3, :3] @ r.T
    truth[:, :3, 3] += np.einsum("tij,j->ti", truth[:, :3, :3], center)
    align = truth[0] @ np.linalg.inv(candidate[0])
    candidate = align[None] @ candidate
    p = candidate[valid]
    q = truth[valid]
    position = np.linalg.norm(p[:, :3, 3] - q[:, :3, 3], axis=1)
    angle = Rotation.from_matrix(
        np.transpose(q[:, :3, :3], (0, 2, 1)) @ p[:, :3, :3]
    ).magnitude()
    points = np.asarray(reference_points) - center
    add = []
    for pp, qq in zip(p, q):
        add.append(
            np.linalg.norm(
                points @ pp[:3, :3].T + pp[:3, 3] - points @ qq[:3, :3].T - qq[:3, 3],
                axis=1,
            ).mean()
        )
    add = np.asarray(add)
    keypoints = np.concatenate([np.eye(3), -np.eye(3)]) * 0.05
    mppe = np.linalg.norm(
        np.einsum("tij,kj->tki", p[:, :3, :3], keypoints)
        + p[:, :3, 3, None].transpose(0, 2, 1)
        - np.einsum("tij,kj->tki", q[:, :3, :3], keypoints)
        - q[:, :3, 3, None].transpose(0, 2, 1),
        axis=2,
    ).mean(1)
    thresholds = np.arange(0.01, 0.10, 0.01)
    auc = float(
        trapezoid(
            (add[:, None] < thresholds).mean(0), np.linspace(0.0, 1.0, len(thresholds))
        )
    )
    complete = bool(valid.all())
    coverage = float(valid.mean())
    metrics = dict(
        coverage=coverage,
        scored_frames=int(valid.sum()),
        total_frames=len(valid),
        complete=complete,
        mean_position_cm=float(position.mean() * 100),
        mean_orientation_deg=float(np.rad2deg(angle).mean()),
        mppe_cm=float(mppe.mean() * 100),
        add_cm=float(add.mean() * 100),
        auc_observed=auc,
        auc_missing_as_failure=auc * coverage,
        sp_sr=bool(complete and position.mean() <= 0.1 and angle.mean() <= 0.5),
        mp_sr=bool(
            complete and position.mean() < 0.03 and angle.mean() < np.deg2rad(30)
        ),
        rpe_cm=0.0,
        rpe_note="Single-object relative pose error is uninformative",
        trajectory_rescaled=False,
        alignment="one_initial_rigid_transform",
        world_alignment=align.tolist(),
    )
    metrics["auc_thresholds_m"] = thresholds.tolist()
    return metrics, dict(
        frame_mask=valid,
        position_error_m=position,
        orientation_error_rad=angle,
        add_m=add,
        mppe_m=mppe,
        aligned_candidate=candidate,
        reference=truth,
    )
