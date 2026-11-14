"""Additional selected-link collision costs in explicit uncertainty scenes."""

import numpy as np

from .grasp_kinematics import contact_penetrations, robot_indices


class CollisionReserve:
    """A scratch-state planning cost; never removes primary or evaluation collisions.

    The caller places the same embodiment in the same metric frame in each
    context. Only the named links contribute this extra cost. Complete-hand
    acceptance must still run separately in every required scene.
    """

    def __init__(
        self, model, data, roots, geom_names, *, threshold_m=0.001, scale_m=0.002
    ):
        import mujoco

        roots = [roots] if isinstance(roots, str) else list(roots)
        if (
            not roots
            or len(set(roots)) != len(roots)
            or not 0 <= threshold_m <= 0.003
            or not 0.0005 <= scale_m <= 0.01
        ):
            raise ValueError("Distinct roots and bounded collision cost required")
        groups = [robot_indices(model, r) for r in roots]
        joints = np.concatenate([g["joints"] for g in groups])
        q = np.concatenate([g["q"] for g in groups])
        dofs = np.concatenate([g["dofs"] for g in groups])
        if len(set(joints.tolist())) != len(joints):
            raise ValueError("Reserve roots must have disjoint joints")
        owned = {int(g) for group in groups for g in group["geoms"]}
        names = list(geom_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Distinct named collision links required")
        geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
        if any(g not in owned for g in geoms):
            raise ValueError("Reserve geometry must be owned and collidable")
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = data.qpos
        self.indices = {
            "joints": joints,
            "q": q,
            "dofs": dofs,
            "geoms": np.asarray(geoms),
            "bodies": set.union(*(g["bodies"] for g in groups)),
        }
        self.joint_names = [model.joint(int(j)).name for j in joints]
        self.threshold_m = float(threshold_m)
        self.scale_m = float(scale_m)

    def evaluate(self, q):
        import mujoco

        q = np.asarray(q, float)
        if q.shape != self.indices["q"].shape or not np.isfinite(q).all():
            raise ValueError("Finite owned joint vector required")
        self.data.qpos[self.indices["q"]] = q
        mujoco.mj_fwdPosition(self.model, self.data)
        depth, jac = contact_penetrations(
            self.model, self.data, self.indices, jacobian=True
        )
        active = depth > self.threshold_m
        return np.maximum(depth - self.threshold_m, 0) / self.scale_m, jac * active[
            :, None
        ] / self.scale_m
