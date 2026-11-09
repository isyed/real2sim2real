"""Calibrated stereo with explicit held-out epipolar validation."""

from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .camera import rigid, transform, project_depth
from .capture import frame


def matches(left, right):
    detector = cv2.SIFT_create(nfeatures=4000)
    features = [
        detector.detectAndCompute(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), None)
        for x in (left, right)
    ]
    if any(x[1] is None for x in features):
        raise ValueError("Insufficient image texture for calibration")
    pairs = cv2.BFMatcher().knnMatch(features[0][1], features[1][1], k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.7 * p[1].distance]
    if len(good) < 30:
        raise ValueError("Insufficient stereo correspondences")
    a, b = [
        np.array(
            [features[i][0][m.queryIdx if i == 0 else m.trainIdx].pt for m in good],
            np.float32,
        )
        for i in (0, 1)
    ]
    _, inliers = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.5, 0.999)
    if inliers is None or inliers.sum() < 30:
        raise ValueError("No reliable stereo geometry")
    return a[inliers.ravel() != 0], b[inliers.ravel() != 0]


class Stereo:
    def __init__(self, left, right, target, *, right_from_left=None, disparities=128):
        self.left, self.right, self.target = left, right, target
        self.size = tuple(left["size"])
        if self.size != tuple(right["size"]):
            raise ValueError("Stereo camera resolutions differ")
        t = rigid(
            right_from_left
            if right_from_left is not None
            else np.linalg.inv(rigid(right["rig_from_camera"]))
            @ rigid(left["rig_from_camera"])
        )
        self.right_from_left = t
        self.K = [np.asarray(x["intrinsics"], float) for x in (left, right)]
        self.D = [np.asarray(x["distortion"], float) for x in (left, right)]
        r1, r2, p1, p2, self.Q, _, _ = cv2.stereoRectify(
            *[
                self.K[0],
                self.D[0],
                self.K[1],
                self.D[1],
                self.size,
                np.ascontiguousarray(t[:3, :3]),
                np.ascontiguousarray(t[:3, 3]).reshape(3, 1),
            ],
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        if p2[0, 3] >= 0:
            raise ValueError("Stereo ordering must produce positive disparity")
        self.rotations, self.projections = (r1, r2), (p1, p2)
        self.maps = [
            cv2.initUndistortRectifyMap(
                self.K[i],
                self.D[i],
                self.rotations[i],
                self.projections[i],
                self.size,
                cv2.CV_32FC1,
            )
            for i in (0, 1)
        ]
        self.target_from_left = np.linalg.inv(rigid(target["rig_from_camera"])) @ rigid(
            left["rig_from_camera"]
        )
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=disparities,
            blockSize=5,
            P1=8 * 25,
            P2=32 * 25,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def epipolar_errors(self, correspondences):
        rect = [
            cv2.undistortPoints(
                np.asarray(correspondences[i])[:, None],
                self.K[i],
                self.D[i],
                R=self.rotations[i],
                P=self.projections[i],
            ).reshape(-1, 2)
            for i in (0, 1)
        ]
        return np.abs(rect[0][:, 1] - rect[1][:, 1])

    def depth(self, left, right, target_k, target_size, *, min_depth=0.15, max_depth=8):
        rect = [
            cv2.remap(
                cv2.cvtColor(x, cv2.COLOR_RGB2GRAY),
                *self.maps[i],
                interpolation=cv2.INTER_LINEAR,
            )
            for i, x in enumerate((left, right))
        ]
        disparity = self.matcher.compute(*rect).astype(np.float32) / 16
        xyz = cv2.reprojectImageTo3D(disparity, self.Q)
        valid = (
            (disparity > 0)
            & np.isfinite(xyz).all(-1)
            & (xyz[..., 2] > min_depth)
            & (xyz[..., 2] < max_depth)
        )
        left_xyz = xyz[valid] @ self.rotations[0]
        target_xyz = transform(left_xyz, self.target_from_left)
        return project_depth(target_xyz, target_k, target_size), {
            "stereo_valid_fraction": float(valid.mean()),
            "median_disparity_px": float(np.median(disparity[valid]))
            if valid.any()
            else None,
        }


def refine_calibration(capture, fit_ids, validation_ids):
    """Refine relative pose within a small trust region; preserve metric baseline."""
    if set(fit_ids) & set(validation_ids) or not fit_ids or not validation_ids:
        raise ValueError("Disjoint nonempty calibration/validation frames required")
    left, right = [capture["streams"][n] for n in capture["stereo_pair"]]
    target = capture["streams"][capture["target_stream"]]
    original = Stereo(left, right, target)
    observed = {
        i: matches(frame(left, i), frame(right, i)) for i in fit_ids + validation_ids
    }
    points = [np.concatenate([observed[i][j] for i in fit_ids]) for j in (0, 1)]
    norm = [
        cv2.undistortPoints(p[:, None], original.K[i], original.D[i]).reshape(-1, 2)
        for i, p in enumerate(points)
    ]
    a, b = [np.column_stack([p, np.ones(len(p))]) for p in norm]
    base = original.right_from_left
    length = np.linalg.norm(base[:3, 3])

    def matrix(parameters):
        result = base.copy()
        result[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix() @ base[:3, :3]
        direction = base[:3, 3] / length + parameters[3:]
        result[:3, 3] = length * direction / np.linalg.norm(direction)
        return result

    def residual(parameters):
        t = matrix(parameters)
        x, y, z = t[:3, 3]
        essential = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) @ t[:3, :3]
        line = a @ essential.T
        return (
            (b * line).sum(1)
            / np.maximum(np.linalg.norm(line[:, :2], axis=1), 1e-12)
            * original.K[0][0, 0]
        )

    solution = least_squares(
        residual,
        np.zeros(6),
        bounds=(-0.035, 0.035),
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=100,
    )
    candidate = Stereo(left, right, target, right_from_left=matrix(solution.x))
    rows = []
    for i, pair in observed.items():
        old, new = original.epipolar_errors(pair), candidate.epipolar_errors(pair)
        rows.append(
            dict(
                frame_id=i,
                split="fit" if i in fit_ids else "validation",
                matches=len(old),
                original_median_px=float(np.median(old)),
                candidate_median_px=float(np.median(new)),
                candidate_p90_px=float(np.quantile(new, 0.9)),
            )
        )
    held = [x for x in rows if x["split"] == "validation"]
    accepted = bool(
        solution.success
        and all(
            x["candidate_median_px"] < 1
            and x["candidate_p90_px"] < 2
            and x["candidate_median_px"] < x["original_median_px"]
            for x in held
        )
    )
    return dict(
        accepted=accepted,
        original_right_from_left=base.tolist(),
        candidate_right_from_left=candidate.right_from_left.tolist(),
        baseline_m=float(length),
        frames=rows,
        fit_parameters=solution.x.tolist(),
        method="bounded_relative_pose_fixed_baseline",
    )
