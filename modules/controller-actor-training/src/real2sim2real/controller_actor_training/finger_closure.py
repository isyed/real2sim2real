"""Explicit bounded finger closing commands; no object or wrist actuation."""

import numpy as np


FLEXION_SUFFIXES = ("MCP_FE", "PIP", "DIP", "IP", "thumb_CMC_FE")


def closure_target(joint_names, grasp, default_offset_rad, settings=None):
    """Override named finger flexion offsets while preserving the legacy default."""
    result = np.asarray(grasp, float).copy()
    if result.shape != (len(joint_names),) or not np.isfinite(result).all():
        raise ValueError("Finite grasp coordinates matching joint names required")
    if not np.isfinite(default_offset_rad) or not 0 <= default_offset_rad <= 0.6:
        raise ValueError("Bounded nonnegative default closure required")
    if settings is not None and (
        not isinstance(settings, dict) or set(settings) != {"offsets_rad"}
    ):
        raise ValueError("Explicit named closure offsets required")
    overrides = {} if settings is None else settings["offsets_rad"]
    if not isinstance(overrides, dict):
        raise ValueError("Named closure offsets must be a mapping")
    allowed = {n for n in joint_names if n.endswith(FLEXION_SUFFIXES)}
    if set(overrides) - allowed:
        raise ValueError("Closure can only name owned finger flexion joints")
    values = np.asarray(list(overrides.values()), float)
    if (
        values.shape != (len(overrides),)
        or not np.isfinite(values).all()
        or (np.abs(values) > 0.6).any()
    ):
        raise ValueError("Finite finger offsets bounded to 0.6 radians required")
    for i, name in enumerate(joint_names):
        if name in allowed:
            result[i] += overrides.get(name, default_offset_rad)
    return result


def closure_geometry(model, data, indices, site_names, grasp, closed, targets):
    """Audit command-only FK in scratch data; not a contact/physical prediction."""
    import mujoco

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    sites = [model.site(name).id for name in site_names]
    values = []
    for q in (grasp, closed):
        scratch.qpos[indices["q"]] = q
        mujoco.mj_kinematics(model, scratch)
        values.append(scratch.site_xpos[sites].copy())
    return dict(
        sites=site_names,
        planned_target_error_m=np.linalg.norm(values[0] - targets, axis=1).tolist(),
        closed_command_target_error_m=np.linalg.norm(
            values[1] - targets, axis=1
        ).tolist(),
        command_tip_displacement_m=np.linalg.norm(
            values[1] - values[0], axis=1
        ).tolist(),
        physical_contact_validated=False,
    )


class FingerSurfaceController:
    """Bounded finger-only servo to object-local contact targets and normal force.

    Contact force and object pose are privileged simulator measurements. The
    controller does not move the wrist, object, mesh, or physical joint limits.
    """

    def __init__(self, model, config, candidate, indices, world_from_object):
        self.settings = config["finger_surface_feedback"]
        s = self.settings
        required = {
            "groups",
            "gain_s_inv",
            "max_speed_rad_s",
            "max_offset_rad",
            "target_force_n",
            "force_gain_m_per_n_s",
            "max_inset_m",
            "max_outset_m",
        }
        if set(s) != required or not 1 <= len(s["groups"]) <= 5:
            raise ValueError("Explicit bounded finger surface controller required")
        vals = np.array([s[k] for k in sorted(required - {"groups"})], float)
        if (
            not np.isfinite(vals).all()
            or (vals <= 0).any()
            or s["gain_s_inv"] > 10
            or s["max_speed_rad_s"] > 2
            or s["max_offset_rad"] > 0.8
            or s["target_force_n"] > 10
            or s["force_gain_m_per_n_s"] > 0.01
            or max(s["max_inset_m"], s["max_outset_m"]) > 0.005
        ):
            raise ValueError("Finite bounded finger feedback settings required")
        self.names = candidate["joint_names"]
        self.offset = np.zeros(len(self.names))
        self.groups = []
        self.object_id = model.body("task_object").id
        transform = np.asarray(world_from_object)
        normal = (
            np.asarray(candidate["pair"]["close_axis"])
            * candidate["thumb_side_direction"]
        )
        used_joints, used_groups, used_sites = set(), set(), set()
        for group in s["groups"]:
            name, site = group["name"], model.site(group["site"]).id
            if (
                name not in ("thumb", "index", "middle", "ring", "pinky")
                or name in used_groups
                or group["site"] not in config["robot"]["sites"]
                or site in used_sites
            ):
                raise ValueError("Distinct finger groups and declared sites required")
            target_index = config["robot"]["sites"].index(group["site"])
            if (target_index == 0) != (name == "thumb"):
                raise ValueError(
                    "Thumb group must use the first declared contact target"
                )
            ancestors = set()
            body = model.site_bodyid[site]
            while body:
                ancestors.add(int(body))
                body = model.body_parentid[body]
            chosen = [self.names.index(n) for n in group["joints"]]
            if not chosen or len(chosen) != len(set(chosen)):
                raise ValueError("Distinct finger joints required")
            joints = indices["joints"][chosen]
            if any(
                j in used_joints
                or model.jnt_bodyid[j] not in ancestors
                or model.jnt_bodyid[j] == indices["root"]
                or not model.jnt_limited[j]
                for j in joints
            ):
                raise ValueError("Only independent bounded finger ancestors may move")
            used_joints.update(joints)
            used_groups.add(name)
            used_sites.add(site)
            point = np.asarray(candidate["targets_m"][target_index])
            self.groups.append(
                dict(
                    name=name,
                    site=site,
                    indices=np.asarray(chosen),
                    dofs=indices["dofs"][chosen],
                    limits=model.jnt_range[joints],
                    point=transform[:3, :3].T @ (point - transform[:3, 3]),
                    normal=transform[:3, :3].T
                    @ (normal * (-1 if name == "thumb" else 1)),
                    inset=0.0,
                    error_m=0.0,
                )
            )

    def target(self, model, data, q, forces, dt):
        import mujoco

        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite controller step required")
        # Refresh kinematic fields only, never assign physical coordinates.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        r = data.xmat[self.object_id].reshape(3, 3)
        t = data.xpos[self.object_id]
        output = np.asarray(q).copy()
        s = self.settings
        for g in self.groups:
            force = forces.get(g["name"], 0.0)
            if not np.isfinite(force) or force < 0:
                raise ValueError("Finite nonnegative finger force required")
            g["inset"] = float(
                np.clip(
                    g["inset"]
                    + dt * s["force_gain_m_per_n_s"] * (s["target_force_n"] - force),
                    -s["max_outset_m"],
                    s["max_inset_m"],
                )
            )
            target = r @ (g["point"] - g["normal"] * g["inset"]) + t
            error = target - data.site_xpos[g["site"]]
            g["error_m"] = float(np.linalg.norm(error))
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac, None, g["site"])
            j = jac[:, g["dofs"]]
            speed = j.T @ np.linalg.solve(
                j @ j.T + 1e-5 * np.eye(3), s["gain_s_inv"] * error
            )
            speed *= min(1.0, s["max_speed_rad_s"] / max(np.linalg.norm(speed), 1e-12))
            ids = g["indices"]
            delta = np.clip(
                self.offset[ids] + dt * speed, -s["max_offset_rad"], s["max_offset_rad"]
            )
            output[ids] = np.clip(q[ids] + delta, *g["limits"].T)
            self.offset[ids] = output[ids] - q[ids]
        return output

    def diagnostics(self):
        return dict(
            finger_surface_offset_rad=self.offset.tolist(),
            finger_surface={
                g["name"]: dict(inset_m=g["inset"], target_error_m=g["error_m"])
                for g in self.groups
            },
        )
