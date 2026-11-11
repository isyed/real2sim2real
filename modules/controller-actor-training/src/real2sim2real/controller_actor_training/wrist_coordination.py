"""Bounded common wrist corrections around a fixed object hold target."""

import numpy as np
from scipy.spatial.transform import Rotation


def bounded_vector(value, limit):
    value = np.asarray(value, float)
    if not np.isfinite(value).all() or not np.isfinite(limit) or limit < 0:
        raise ValueError("Finite vector and nonnegative bound required")
    return value * min(1.0, limit / max(np.linalg.norm(value), 1e-12))


def rigid_wrist_targets(poses, pivot, rotation, translation):
    """Apply one rigid motion to all wrists, preserving their relative frames."""
    result = np.asarray(poses, float).copy()
    pivot, rotation, translation = map(np.asarray, (pivot, rotation, translation))
    if (
        result.ndim != 3
        or result.shape[1:] != (4, 4)
        or pivot.shape != (3,)
        or rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not all(np.isfinite(x).all() for x in (result, pivot, rotation, translation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        or not np.isclose(np.linalg.det(rotation), 1.0)
    ):
        raise ValueError("Finite rigid wrist frames and common transform required")
    result[:, :3, 3] = (result[:, :3, 3] - pivot) @ rotation.T + pivot + translation
    result[:, :3, :3] = rotation @ result[:, :3, :3]
    return result


def hold_world_motion(records):
    """Audit object translation independently of moving wrist-frame drift."""
    rows = [r for r in records if r["phase"] == "hold"]
    if not rows:
        return dict(hold_world_horizontal_drift_m=None)
    positions = np.asarray([r["object_position_m"] for r in rows], float)
    if positions.shape != (len(rows), 3) or not np.isfinite(positions).all():
        raise ValueError("Finite object observations required")
    return dict(
        hold_world_horizontal_drift_m=float(
            np.linalg.norm(positions[:, :2] - positions[0, :2], axis=1).max()
        )
    )


class ObjectFrameCoordinator:
    """Correct both wrist commands about the object, never its simulated pose.

    The XY target and pivot are frozen at hold entry. Orientation targets the
    original settled orientation. Vertical clearance and support load control
    remain separate. Only world-aligned slide plus intrinsic XYZ wrist fixtures
    are supported; this is not an arm inverse-kinematics solver.

    With an explicit desired_pose, its position becomes the moving pivot and all
    three translation axes are corrected. The caller must freeze the stationary
    goal or supply an independent reference, and disable competing hold servos.
    """

    def __init__(self, model, hands, settings, target_rotation):
        import mujoco

        self.settings = settings
        required = (
            "translation_gain_s_inv",
            "rotation_gain_s_inv",
            "max_translation_speed_m_s",
            "max_rotation_speed_rad_s",
            "max_translation_m",
            "max_rotation_rad",
        )
        if set(settings) - {"joint_limit_policy"} != set(required):
            raise ValueError(
                "Explicit coordination gains, speeds and travel bounds required"
            )
        if settings.get("joint_limit_policy", "reject") not in (
            "reject",
            "project_common",
        ):
            raise ValueError("Unknown coordinated joint-limit policy")
        values = np.asarray([settings[k] for k in required], float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or max(values[:2]) > 10
            or not 0 < settings["max_translation_speed_m_s"] <= 0.05
            or not 0 < settings["max_rotation_speed_rad_s"] <= 0.5
            or not 0 < settings["max_translation_m"] <= 0.05
            or not 0 < settings["max_rotation_rad"] <= 0.45
        ):
            raise ValueError("Finite bounded coordination settings required")
        self.hands = []
        for root_name, prefix in hands:
            root = model.body(root_name).id
            joints = [
                model.joint(prefix + a).id for a in ("x", "y", "z", "rx", "ry", "rz")
            ]
            base = Rotation.from_quat(model.body_quat[root][[1, 2, 3, 0]]).as_matrix()
            if (
                model.body_parentid[root] != 0
                or model.body_jntnum[root] != 6
                or any(model.jnt_bodyid[j] != root for j in joints)
                or list(joints)
                != list(range(model.body_jntadr[root], model.body_jntadr[root] + 6))
                or not np.all(model.jnt_limited[joints])
                or not np.allclose(model.qpos0[model.jnt_qposadr[joints]], 0)
                or list(model.jnt_type[joints])
                != [mujoco.mjtJoint.mjJNT_SLIDE] * 3 + [mujoco.mjtJoint.mjJNT_HINGE] * 3
                or not np.allclose(model.jnt_axis[joints[:3]], base, atol=1e-7)
                or not np.allclose(model.jnt_axis[joints[3:]], np.eye(3), atol=1e-7)
                or not np.allclose(model.jnt_pos[joints], 0)
            ):
                raise ValueError("World-aligned Cartesian/XYZ wrist fixture required")
            self.hands.append(
                dict(
                    root=root,
                    q=model.jnt_qposadr[joints],
                    base=base,
                    limits=model.jnt_range[joints],
                )
            )
        if len(self.hands) < 2 or len({h["root"] for h in self.hands}) != len(
            self.hands
        ):
            raise ValueError("At least two distinct wrists required")
        self.scratch = mujoco.MjData(model)
        self.target_rotation = np.asarray(target_rotation, float)
        if (
            self.target_rotation.shape != (3, 3)
            or not np.allclose(
                self.target_rotation.T @ self.target_rotation, np.eye(3), atol=1e-7
            )
            or not np.isclose(np.linalg.det(self.target_rotation), 1.0)
        ):
            raise ValueError("Proper target rotation required")
        self.pivot = None
        self.translation = np.zeros(3)
        self.rotation = np.eye(3)
        self.rejection = None
        self.max_position_residual_m = 0.0
        self.max_rotation_residual_rad = 0.0
        self.projected_steps = 0
        self.update_steps = 0

    def update(self, object_pose, dt, desired_pose=None):
        object_pose = np.asarray(object_pose, float)
        if (
            object_pose.shape != (4, 4)
            or not np.isfinite(object_pose).all()
            or not np.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError("Finite metric object pose and positive step required")
        if self.pivot is None:
            self.pivot = object_pose[:3, 3].copy()
        if desired_pose is not None:
            desired_pose = np.asarray(desired_pose, float)
            if (
                desired_pose.shape != (4, 4)
                or not np.isfinite(desired_pose).all()
                or not np.allclose(desired_pose[3], [0, 0, 0, 1])
                or not np.allclose(
                    desired_pose[:3, :3].T @ desired_pose[:3, :3], np.eye(3), atol=1e-7
                )
                or not np.isclose(np.linalg.det(desired_pose[:3, :3]), 1)
            ):
                raise ValueError(
                    "Proper independently supplied desired object pose required"
                )
            self.pivot = desired_pose[:3, 3].copy()
            self.target_rotation = desired_pose[:3, :3].copy()
        error = self.pivot - object_pose[:3, 3]
        if desired_pose is None:
            error[2] = 0.0
        s = self.settings
        velocity = bounded_vector(
            error * s["translation_gain_s_inv"], s["max_translation_speed_m_s"]
        )
        self.translation = bounded_vector(
            self.translation + dt * velocity, s["max_translation_m"]
        )
        angle = Rotation.from_matrix(
            self.target_rotation @ object_pose[:3, :3].T
        ).as_rotvec()
        omega = bounded_vector(
            angle * s["rotation_gain_s_inv"], s["max_rotation_speed_rad_s"]
        )
        rotation = Rotation.from_rotvec(dt * omega).as_matrix() @ self.rotation
        self.rotation = Rotation.from_rotvec(
            bounded_vector(
                Rotation.from_matrix(rotation).as_rotvec(), s["max_rotation_rad"]
            )
        ).as_matrix()

    def target(self, model, target, object_pose, dt, *, desired_pose=None):
        import mujoco

        previous_rotation, previous_translation = (
            self.rotation.copy(),
            self.translation.copy(),
        )
        self.update(object_pose, dt, desired_pose)
        result = np.asarray(target, float).copy()
        self.scratch.qpos[:] = result
        mujoco.mj_kinematics(model, self.scratch)
        poses = np.tile(np.eye(4), (len(self.hands), 1, 1))
        for i, hand in enumerate(self.hands):
            poses[i, :3, 3] = self.scratch.xpos[hand["root"]]
            poses[i, :3, :3] = self.scratch.xmat[hand["root"]].reshape(3, 3)

        def solve(rotation, translation):
            desired = rigid_wrist_targets(poses, self.pivot, rotation, translation)
            commands = np.asarray(target, float).copy()
            feasible = True
            for hand, original, goal in zip(self.hands, poses, desired):
                q = hand["q"]
                commands[q[:3]] += goal[:3, 3] - original[:3, 3]
                commands[q[3:]] = Rotation.from_matrix(
                    hand["base"].T @ goal[:3, :3]
                ).as_euler("XYZ")
                lo, hi = hand["limits"].T
                feasible = feasible and not (
                    (commands[q] < lo).any() or (commands[q] > hi).any()
                )
            return commands, desired, feasible

        result, desired, feasible = solve(self.rotation, self.translation)
        self.update_steps += 1
        if (
            not feasible
            and self.settings.get("joint_limit_policy", "reject") == "project_common"
        ):
            # Preserve feasible lateral correction even if rotation is limited.
            anchor_translation = self.translation.copy()
            _, _, anchor_ok = solve(previous_rotation, anchor_translation)
            if not anchor_ok:
                anchor_translation = previous_translation
            anchor, anchor_desired, anchor_ok = solve(
                previous_rotation, anchor_translation
            )
            if anchor_ok:
                requested_rotation = self.rotation.copy()
                requested_translation = self.translation.copy()
                delta = Rotation.from_matrix(
                    requested_rotation @ previous_rotation.T
                ).as_rotvec()
                lower, upper = 0.0, 1.0
                result, desired = anchor, anchor_desired
                self.rotation, self.translation = previous_rotation, anchor_translation
                for _ in range(12):
                    fraction = (lower + upper) / 2
                    rotation = (
                        Rotation.from_rotvec(delta * fraction).as_matrix()
                        @ previous_rotation
                    )
                    translation = anchor_translation + fraction * (
                        requested_translation - anchor_translation
                    )
                    proposal, goal, ok = solve(rotation, translation)
                    if ok:
                        lower = fraction
                        result, desired = proposal, goal
                        self.rotation, self.translation = rotation, translation
                    else:
                        upper = fraction
                feasible = True
                self.projected_steps += 1
        if not feasible:
            self.rejection = "coordinated_wrist_joint_limit"
            for hand in self.hands:
                q = hand["q"]
                lo, hi = hand["limits"].T
                result[q] = np.clip(result[q], lo, hi)
        # Check realized command kinematics; never change physical qpos here.
        self.scratch.qpos[:] = result
        mujoco.mj_kinematics(model, self.scratch)
        for hand, goal in zip(self.hands, desired):
            dp = np.linalg.norm(self.scratch.xpos[hand["root"]] - goal[:3, 3])
            dr = Rotation.from_matrix(
                self.scratch.xmat[hand["root"]].reshape(3, 3) @ goal[:3, :3].T
            ).magnitude()
            self.max_position_residual_m = max(self.max_position_residual_m, float(dp))
            self.max_rotation_residual_rad = max(
                self.max_rotation_residual_rad, float(dr)
            )
        return result

    def diagnostics(self):
        return dict(
            coordination_translation_m=self.translation.tolist(),
            coordination_rotation_vector_rad=Rotation.from_matrix(self.rotation)
            .as_rotvec()
            .tolist(),
            coordination_pivot_m=None if self.pivot is None else self.pivot.tolist(),
            coordination_projected_steps=self.projected_steps,
            coordination_update_steps=self.update_steps,
        )
