"""Transfer reconstructed object-motion increments through robot wrist commands."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from real2sim2real.artifacts import digest
from .wrist_coordination import ObjectFrameCoordinator, rigid_wrist_targets


class MotionReference:
    """Immutable metric capture interval; no rescaling, retiming or gap filling."""

    def __init__(self, times, poses, frame_ids):
        self.times = np.asarray(times, float)
        self.poses = np.asarray(poses, float)
        self.frame_ids = np.asarray(frame_ids)
        n = len(self.times)
        if (
            n < 2
            or self.times.shape != (n,)
            or self.poses.shape != (n, 4, 4)
            or self.frame_ids.shape != (n,)
            or not np.isfinite(self.times).all()
            or not np.isfinite(self.poses).all()
            or not np.all(np.diff(self.times) > 0)
            or np.max(np.diff(self.times)) > 0.1
            or not np.all(np.diff(self.frame_ids) == 1)
            or not np.allclose(self.poses[:, 3], [0, 0, 0, 1])
            or not np.allclose(
                self.poses[:, :3, :3].transpose(0, 2, 1) @ self.poses[:, :3, :3],
                np.eye(3),
                atol=1e-6,
            )
            or not np.allclose(np.linalg.det(self.poses[:, :3, :3]), 1, atol=1e-6)
        ):
            raise ValueError(
                "Contiguous finite metric poses and capture times required"
            )
        self.elapsed = self.times - self.times[0]
        self.positions = self.poses[:, :3, 3] - self.poses[0, :3, 3]
        self.rotations = self.poses[:, :3, :3] @ self.poses[0, :3, :3].T
        self.slerp = Slerp(self.elapsed, Rotation.from_matrix(self.rotations))

    def sample(self, elapsed):
        if not np.isfinite(elapsed):
            raise ValueError("Finite reference time required")
        t = np.clip(elapsed, 0, self.elapsed[-1])
        position = np.array([np.interp(t, self.elapsed, v) for v in self.positions.T])
        return position, self.slerp(t).as_matrix()

    def bounds(self):
        dt = np.diff(self.times)
        angular_steps = Rotation.from_matrix(
            self.rotations[1:] @ self.rotations[:-1].transpose(0, 2, 1)
        ).magnitude()
        return dict(
            maximum_translation_m=float(np.linalg.norm(self.positions, axis=1).max()),
            maximum_rotation_rad=float(
                Rotation.from_matrix(self.rotations).magnitude().max()
            ),
            maximum_speed_m_s=float(
                (np.linalg.norm(np.diff(self.positions, axis=0), axis=1) / dt).max()
            ),
            maximum_angular_speed_rad_s=float((angular_steps / dt).max()),
            duration_s=float(self.elapsed[-1]),
        )


def load_reference(settings):
    path = Path(settings["path"])
    if digest(path) != settings["sha256"]:
        raise ValueError("Motion reference changed")
    with np.load(path, allow_pickle=False) as z:
        reference = MotionReference(
            z["timestamps_s"], z["world_from_object"], z["frame_ids"]
        )
    bounds = reference.bounds()
    for name in (
        "maximum_translation_m",
        "maximum_rotation_rad",
        "maximum_speed_m_s",
        "maximum_angular_speed_rad_s",
    ):
        if not 0 < settings["limits"][name] or not np.isfinite(
            settings["limits"][name]
        ):
            raise ValueError("Finite explicit motion bounds required")
        if bounds[name] > settings["limits"][name]:
            raise ValueError(f"Reference exceeds {name}")
    if settings["mode"] != "world_increment_after_grasp":
        raise ValueError("Explicit post-grasp motion-transfer mode required")
    return reference


class GuidedMotion:
    """Move both wrists by capture-world increments after support establishment.

    Existing finger targets and force-limited actuators remain active. The
    object is observed but never assigned poses. A fixture limit rejects the
    motion; independently clipping hands is not a successful transfer.
    """

    def __init__(self, model, hands, settings):
        self.reference = load_reference(settings)
        # Reuse the checked Cartesian-fixture contract, not its feedback law.
        fixture = ObjectFrameCoordinator(
            model,
            hands,
            dict(
                translation_gain_s_inv=0,
                rotation_gain_s_inv=0,
                max_translation_speed_m_s=0.02,
                max_rotation_speed_rad_s=0.2,
                max_translation_m=0.025,
                max_rotation_rad=0.2,
            ),
            np.eye(3),
        )
        self.hands, self.scratch = fixture.hands, fixture.scratch
        self.start = None
        self.anchor = None
        self.goal = None
        self.rejection = None
        self.max_command_error_m = 0.0

    def enter(self, time, object_pose):
        if self.start is None:
            self.start = float(time)
            self.anchor = np.asarray(object_pose, float).copy()

    def delta(self, time):
        if self.start is None:
            return np.zeros(3), np.eye(3)
        return self.reference.sample(time - self.start)

    def feedback_pose(self, object_pose, time):
        """Express tracking error in the baseline frame before its feedback law."""
        if self.start is None:
            return np.asarray(object_pose).copy()
        p, r = self.delta(time)
        result = np.asarray(object_pose).copy()
        result[:3, 3] = self.anchor[:3, 3] + r.T @ (
            result[:3, 3] - self.anchor[:3, 3] - p
        )
        result[:3, :3] = r.T @ result[:3, :3]
        return result

    def target(self, model, target, time):
        import mujoco

        translation, rotation = self.delta(time)
        self.goal = self.anchor.copy()
        self.goal[:3, 3] += translation
        self.goal[:3, :3] = rotation @ self.anchor[:3, :3]
        self.scratch.qpos[:] = target
        mujoco.mj_kinematics(model, self.scratch)
        poses = np.tile(np.eye(4), (len(self.hands), 1, 1))
        for i, h in enumerate(self.hands):
            poses[i, :3, 3] = self.scratch.xpos[h["root"]]
            poses[i, :3, :3] = self.scratch.xmat[h["root"]].reshape(3, 3)
        goals = rigid_wrist_targets(poses, self.anchor[:3, 3], rotation, translation)
        result = np.asarray(target, float).copy()
        for h, pose, goal in zip(self.hands, poses, goals):
            q = h["q"]
            result[q[:3]] += goal[:3, 3] - pose[:3, 3]
            result[q[3:]] = Rotation.from_matrix(h["base"].T @ goal[:3, :3]).as_euler(
                "XYZ"
            )
            if (
                (result[q] < h["limits"][:, 0]) | (result[q] > h["limits"][:, 1])
            ).any():
                self.rejection = "reference_wrist_joint_limit"
                return np.asarray(target).copy()
        self.scratch.qpos[:] = result
        mujoco.mj_kinematics(model, self.scratch)
        for h, goal in zip(self.hands, goals):
            self.max_command_error_m = max(
                self.max_command_error_m,
                float(np.linalg.norm(self.scratch.xpos[h["root"]] - goal[:3, 3])),
            )
        return result

    def observation(self, object_pose, time):
        if self.start is None:
            return {}
        translation, rotation = self.delta(time)
        p = self.anchor[:3, 3] + translation
        r = rotation @ self.anchor[:3, :3]
        return dict(
            reference_elapsed_s=float(time - self.start),
            reference_position_m=p.tolist(),
            reference_rotation=r.tolist(),
            reference_position_error_m=float(np.linalg.norm(object_pose[:3, 3] - p)),
            reference_rotation_error_deg=float(
                np.rad2deg(Rotation.from_matrix(object_pose[:3, :3] @ r.T).magnitude())
            ),
        )
