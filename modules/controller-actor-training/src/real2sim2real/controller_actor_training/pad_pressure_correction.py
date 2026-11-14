"""Small finger-only force corrections around a retained grasp command."""

import numpy as np

from .contact_force_control import normal_force_step


class PadPressureCorrection:
    """Reuse bounded normal admittance for explicitly selected, disjoint pad joints.

    Force and Jacobian observations may be privileged simulator evidence. Callers
    must restrict the declared groups to finger joints. This helper only changes
    those command offsets; it does not alter geometry or actuators.
    Collision, orientation and sustained-contact gates remain caller obligations.
    """

    def __init__(self, groups, limits, settings):
        raw = [np.asarray(g) for g in groups]
        if any(g.dtype.kind not in "iu" for g in raw):
            raise ValueError("Integer joint indices required")
        self.groups = [g.astype(int) for g in raw]
        self.limits = np.asarray(limits, float)
        required = {
            "target_force_n",
            "gain_m_per_n_s",
            "max_normal_speed_m_s",
            "max_joint_speed_rad_s",
            "max_joint_offset_rad",
            "damping_m2",
        }
        if set(settings) != required:
            raise ValueError("Explicit bounded pressure settings required")
        bounds = {
            "target_force_n": (0, 5),
            "gain_m_per_n_s": (0, 0.01),
            "max_normal_speed_m_s": (0, 0.01),
            "max_joint_speed_rad_s": (0, 1),
            "max_joint_offset_rad": (0, 0.15),
            "damping_m2": (0, 0.0001),
        }
        if any(
            not np.isfinite(settings[k]) or not lo < settings[k] <= hi
            for k, (lo, hi) in bounds.items()
        ):
            raise ValueError("Pressure setting outside bounded range")
        if (
            self.limits.ndim != 2
            or self.limits.shape[1] != 2
            or not np.isfinite(self.limits).all()
            or (self.limits[:, 0] >= self.limits[:, 1]).any()
        ):
            raise ValueError("Finite joint limits required")
        all_ids = np.concatenate(self.groups) if self.groups else np.array([], int)
        if (
            not self.groups
            or any(g.ndim != 1 or not len(g) for g in self.groups)
            or len(set(all_ids.tolist())) != len(all_ids)
            or (all_ids < 0).any()
            or (all_ids >= len(self.limits)).any()
        ):
            raise ValueError("Disjoint nonempty owned joint groups required")
        self.settings = dict(settings)
        self.offset = np.zeros(len(self.limits))

    def update(
        self,
        nominal,
        normal_jacobians,
        forces_n,
        dt,
        *,
        targets_n=None,
        tangent_tasks=None,
    ):
        """Optionally share pressure by pad and resist observed material-point slip.

        ``tangent_tasks`` contains None or (3-by-N projected Jacobian, 3-vector
        velocity) for each pad. The velocity cap is 5 mm/s; all tasks share the
        original joint-speed/offset envelope. Omitted tasks preserve the scalar
        normal-only controller exactly. Callers own contact validity and anchors.
        """
        q = np.asarray(nominal, float)
        j = np.asarray(normal_jacobians, float)
        f = np.asarray(forces_n, float)
        if (
            q.shape != self.offset.shape
            or j.shape != (len(self.groups), len(q))
            or f.shape != (len(self.groups),)
            or not np.isfinite(np.r_[q, j.ravel(), f, dt]).all()
            or (f < 0).any()
            or not 0 < dt <= 0.01
        ):
            raise ValueError(
                "Finite commands, pad observations and bounded update period required"
            )
        goals = (
            np.full(len(self.groups), self.settings["target_force_n"])
            if targets_n is None
            else np.asarray(targets_n, float)
        )
        if (
            goals.shape != f.shape
            or not np.isfinite(goals).all()
            or (goals <= 0).any()
            or (goals > 5).any()
        ):
            raise ValueError("One finite pressure target in (0, 5] N per pad required")
        tasks = [None] * len(self.groups) if tangent_tasks is None else tangent_tasks
        if len(tasks) != len(self.groups):
            raise ValueError("One optional tangent task per pad required")
        normalized = []
        for task in tasks:
            if task is None:
                normalized.append(None)
                continue
            jac, velocity = (np.asarray(x, float) for x in task)
            if (
                jac.shape != (3, len(q))
                or velocity.shape != (3,)
                or not np.isfinite(np.r_[jac.ravel(), velocity]).all()
                or np.linalg.norm(velocity) > 0.005 + 1e-12
            ):
                raise ValueError(
                    "Finite owned tangent task with speed <= 5 mm/s required"
                )
            normalized.append((jac, velocity))
        # Compute atomically: a malformed later pad must not leave partial state.
        new = self.offset.copy()
        result = q.copy()
        for i, g in enumerate(self.groups):
            tangent = (
                {}
                if normalized[i] is None
                else {
                    "tangent_jacobian": normalized[i][0][:, g],
                    "tangent_velocity": normalized[i][1],
                }
            )
            result[g], new[g] = normal_force_step(
                q[g],
                self.offset[g],
                j[i, g],
                f[i],
                goals[i],
                dt,
                self.limits[g],
                self.settings,
                **tangent,
            )
        self.offset = new
        return result
