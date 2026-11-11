"""Bounded finger-only admittance from solved per-pad object contact forces.

Simulator contact forces and object pose are privileged observations. Corrections
use existing position actuators; no object state, wrench or joint limit is changed.
"""

import numpy as np


def contact_geometry_allowed(normal_gap_m, tangent_gap_m, force_n, settings):
    if (
        not np.isfinite([normal_gap_m, tangent_gap_m, force_n]).all()
        or tangent_gap_m < 0
        or force_n < 0
    ):
        raise ValueError("Finite contact gaps and nonnegative force required")
    tangent_limit = settings["maximum_tangent_gap_m"]
    if force_n >= settings.get("minimum_observed_force_n", 0.1):
        tangent_limit = settings.get("loaded_tangent_gap_m", tangent_limit)
    return (
        abs(normal_gap_m) <= settings["maximum_normal_gap_m"]
        and tangent_gap_m <= tangent_limit
    )


def normal_force_step(
    nominal,
    offset,
    normal_jacobian,
    force_n,
    target_n,
    dt,
    limits,
    settings,
    *,
    tangent_jacobian=None,
    tangent_velocity=None,
):
    """Map a signed force error into a bounded normal pressing velocity.

    Positive outward-normal motion releases pressure. Joint stops remove blocked
    directions before solving; clipping updates the integral offset (anti-windup).
    """
    q, delta, jac = [np.asarray(v, float) for v in (nominal, offset, normal_jacobian)]
    limits = np.asarray(limits, float)
    if (
        q.ndim != 1
        or not len(q)
        or delta.shape != q.shape
        or jac.shape != q.shape
        or limits.shape != (len(q), 2)
        or not np.isfinite(
            np.r_[q, delta, jac, limits.ravel(), force_n, target_n, dt]
        ).all()
        or force_n < 0
        or target_n <= 0
        or dt <= 0
        or (limits[:, 0] >= limits[:, 1]).any()
        or (q < limits[:, 0] - 1e-10).any()
        or (q > limits[:, 1] + 1e-10).any()
    ):
        raise ValueError(
            "Finite legal commands, normal Jacobian, forces and positive step required"
        )
    desired = np.clip(
        settings["gain_m_per_n_s"] * (force_n - target_n),
        -settings["max_normal_speed_m_s"],
        settings["max_normal_speed_m_s"],
    )
    if tangent_jacobian is None:
        matrix, task_velocity = jac[None, :].copy(), np.array([desired])
    else:
        tangent_jacobian = np.asarray(tangent_jacobian, float)
        tangent_velocity = np.asarray(tangent_velocity, float)
        if (
            tangent_jacobian.shape != (3, len(q))
            or tangent_velocity.shape != (3,)
            or not np.isfinite(tangent_jacobian).all()
            or not np.isfinite(tangent_velocity).all()
        ):
            raise ValueError("Finite projected tangent Jacobian and velocity required")
        matrix = np.vstack([jac, tangent_jacobian])
        task_velocity = np.r_[desired, tangent_velocity]

    def solve(j):
        if tangent_jacobian is None:
            # Preserve the archived scalar-normal controller exactly.
            return desired * j[0] / (j[0] @ j[0] + settings["damping_m2"])
        return np.linalg.solve(
            j.T @ j + settings["damping_m2"] * np.eye(len(q)), j.T @ task_velocity
        )

    velocity = solve(matrix)
    current = np.clip(q + delta, *limits.T)
    blocked = ((current <= limits[:, 0] + 1e-8) & (velocity < 0)) | (
        (current >= limits[:, 1] - 1e-8) & (velocity > 0)
    )
    matrix[:, blocked] = 0
    velocity = solve(matrix)
    velocity *= min(
        1, settings["max_joint_speed_rad_s"] / max(np.linalg.norm(velocity), 1e-12)
    )
    updated = delta + dt * velocity
    updated *= min(
        1, settings["max_joint_offset_rad"] / max(np.linalg.norm(updated), 1e-12)
    )
    command = np.clip(q + updated, *limits.T)
    return command, command - q


def validate_settings(settings):
    required = {
        "target_force_n",
        "gain_m_per_n_s",
        "max_normal_speed_m_s",
        "max_joint_speed_rad_s",
        "max_joint_offset_rad",
        "damping_m2",
        "maximum_normal_gap_m",
        "maximum_tangent_gap_m",
        "close_activation_fraction",
    }
    optional = {
        "normal_observation",
        "minimum_observed_force_n",
        "tangent_gain_s_inv",
        "max_tangent_speed_m_s",
        "payload_adaptation",
        "posture_transition",
        "contact_anchor",
        "loaded_tangent_gap_m",
        "unloaded_acquisition",
    }
    if (set(settings) - optional) != required or set(settings["target_force_n"]) != {
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
    }:
        raise ValueError(
            "Explicit five-pad targets and bounded force-control settings required"
        )
    if settings.get("normal_observation", "planned") not in (
        "planned",
        "solved_contact",
    ):
        raise ValueError("Unknown force-normal observation mode")
    if not 0 < settings.get("minimum_observed_force_n", 0.1) <= 1:
        raise ValueError("Finite positive observed-contact threshold required")
    if "loaded_tangent_gap_m" in settings:
        if (
            not np.isfinite(settings["loaded_tangent_gap_m"])
            or not settings["maximum_tangent_gap_m"]
            < settings["loaded_tangent_gap_m"]
            <= 0.03
            or settings.get("normal_observation", "planned") != "planned"
        ):
            raise ValueError(
                "Loaded continuation needs a bounded wider tangent guard and planned normal"
            )
    if "payload_adaptation" in settings:
        from .payload_pressure import validate_settings as validate_payload

        validate_payload(settings["payload_adaptation"])
    if "unloaded_acquisition" in settings:
        from .finger_acquisition import validate

        validate(settings["unloaded_acquisition"], settings["close_activation_fraction"])
    if "posture_transition" in settings:
        from .payload_posture import validate_settings as validate_posture

        if "payload_adaptation" not in settings:
            raise ValueError("Posture transitions require supported-load observations")
        validate_posture(settings["posture_transition"])
    if "contact_anchor" in settings:
        from .contact_anchor import validate_settings as validate_anchor

        validate_anchor(settings["contact_anchor"])
        if "tangent_gain_s_inv" in settings or "posture_transition" in settings:
            raise ValueError("Contact anchors require exclusive tangent ownership")
    if "tangent_gain_s_inv" in settings or "max_tangent_speed_m_s" in settings:
        if (
            not 0 < settings.get("tangent_gain_s_inv", 0) <= 5
            or not 0 < settings.get("max_tangent_speed_m_s", 0) <= 0.01
            or settings.get("normal_observation", "planned") != "planned"
        ):
            raise ValueError("Bounded planned-surface tangent control required")
    caps = dict(
        gain_m_per_n_s=0.01,
        max_normal_speed_m_s=0.01,
        max_joint_speed_rad_s=1.0,
        max_joint_offset_rad=0.5,
        damping_m2=0.001,
        maximum_normal_gap_m=0.02,
        maximum_tangent_gap_m=0.03,
    )
    if (
        any(
            not np.isfinite(settings[k]) or not 0 < settings[k] <= cap
            for k, cap in caps.items()
        )
        or any(
            not np.isfinite(v) or not 0 < v <= 10
            for v in settings["target_force_n"].values()
        )
        or not 0.5 <= settings["close_activation_fraction"] <= 1
    ):
        raise ValueError("Force feedback exceeds finite planning/control bounds")


def object_pad_observations(model, data, object_id, pad_ids):
    """Solved pad forces, positions and object-outward normals from this object."""
    import mujoco

    values = {
        int(p): dict(force_n=0.0, point_sum=np.zeros(3), normal_sum=np.zeros(3))
        for p in pad_ids
    }
    for i, contact in enumerate(data.contact):
        for pad, other, sign in (
            (int(contact.geom1), int(contact.geom2), -1),
            (int(contact.geom2), int(contact.geom1), 1),
        ):
            if pad not in values or model.geom_bodyid[other] != object_id:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            if not np.isfinite(force).all():
                raise ValueError("Nonfinite solved contact force")
            normal_force = max(0.0, float(force[0]))
            row = values[pad]
            row["force_n"] += normal_force
            row["point_sum"] += normal_force * contact.pos
            row["normal_sum"] += (
                normal_force * sign * np.asarray(contact.frame).reshape(3, 3)[0]
            )
    for row in values.values():
        norm = np.linalg.norm(row["normal_sum"])
        row["point_world_m"] = (
            row["point_sum"] / row["force_n"] if row["force_n"] > 0 else None
        )
        row["outward_normal_world"] = row["normal_sum"] / norm if norm > 1e-10 else None
    return values


def object_pad_forces(model, data, object_id, pad_ids):
    """Solved normal forces on named pads from this object only, never table/self."""
    return {
        p: r["force_n"]
        for p, r in object_pad_observations(model, data, object_id, pad_ids).items()
    }


class BilateralContactForce:
    """Own only independent finger joints in two explicit grasp-hand subtrees."""

    def __init__(
        self,
        model,
        data,
        settings,
        candidate,
        world_from_object,
        world_mesh,
        *,
        scene_sha256=None,
        allowed_penetration_m=None,
    ):
        import mujoco
        import trimesh
        from real2sim2real.retargeting.grasp_kinematics import robot_indices

        validate_settings(settings)
        self.settings = settings
        self.anchors = None
        if "contact_anchor" in settings:
            from .contact_anchor import ContactAnchors

            self.anchors = ContactAnchors(settings["contact_anchor"])
        self.object_id = model.body("task_object").id
        self.pads = []
        self.role_geoms = {}
        self.payload = None
        if "payload_adaptation" in settings:
            from .payload_pressure import PayloadPressure

            self.payload = PayloadPressure(settings["payload_adaptation"])
            if not np.allclose(model.opt.gravity[:2], 0) or model.opt.gravity[2] >= 0:
                raise ValueError("Payload adaptation requires an upright gravity frame")
        transform = np.asarray(world_from_object, float)
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(
                transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6
            )
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1)
        ):
            raise ValueError("Finite metric object pose required")
        primary_root = (
            model.body(candidate["root_body"]).id
            if "root_body" in candidate
            else int(model.jnt_bodyid[model.joint(candidate["joint_names"][0]).id])
        )
        for role, hand, root in [
            (
                "primary",
                candidate,
                primary_root,
            ),
            (
                "secondary",
                candidate["support"],
                model.body(candidate["support"]["root_body"]).id,
            ),
        ]:
            indices = robot_indices(model, model.body(root).name)
            self.role_geoms[role] = set(indices["geoms"])
            frames = hand["pad_plan"]["frames"]
            if len(frames) != 5 or len({f["geom"] for f in frames}) != 5:
                raise ValueError(
                    "Five distinct admitted collision pads per hand required"
                )
            near, gap, face = trimesh.proximity.closest_point(
                world_mesh, np.asarray(hand["targets_m"])
            )
            if gap.max() > 0.0005:
                raise ValueError("Force targets differ from admitted source surface")
            used = set()
            for i, frame in enumerate(frames):
                gid = model.geom(frame["geom"]).id
                body = model.geom_bodyid[gid]
                if (
                    gid not in indices["geoms"]
                    or model.body(body).name != frame["body"]
                ):
                    raise ValueError("Pad ownership changed")
                group = next(
                    (g for g in settings["target_force_n"] if g in frame["geom"]), None
                )
                if group is None or group in used:
                    raise ValueError("One explicit pad per anatomical group required")
                used.add(group)
                ancestors = set()
                parent = int(body)
                while parent and parent != root:
                    ancestors.add(parent)
                    parent = int(model.body_parentid[parent])
                joints = [
                    int(j)
                    for j in indices["joints"]
                    if int(model.jnt_bodyid[j]) in ancestors
                ]
                if not joints or any(
                    model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE
                    or not model.jnt_limited[j]
                    for j in joints
                ):
                    raise ValueError("Bounded finger ancestor hinges required")
                qids = model.jnt_qposadr[joints]
                dofs = model.jnt_dofadr[joints]
                limits = model.jnt_range[joints].copy()
                for k, j in enumerate(joints):
                    aids = np.flatnonzero(model.actuator_trnid[:, 0] == j)
                    if len(aids) != 1:
                        raise ValueError(
                            "Unique owned position actuator per finger joint required"
                        )
                    a = int(aids[0])
                    gain = model.actuator_gainprm[a, 0]
                    if (
                        model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT
                        or not np.array_equal(
                            model.actuator_gear[a], [1, 0, 0, 0, 0, 0]
                        )
                        or gain <= 0
                        or model.actuator_biasprm[a, 1] != -gain
                        or not model.actuator_forcelimited[a]
                        or not np.isfinite(model.actuator_forcerange[a]).all()
                        or model.actuator_forcerange[a, 0]
                        >= model.actuator_forcerange[a, 1]
                    ):
                        raise ValueError(
                            "Force feedback requires existing force-limited position actuators"
                        )
                    if model.actuator_ctrllimited[a]:
                        limits[k] = (
                            max(limits[k, 0], model.actuator_ctrlrange[a, 0]),
                            min(limits[k, 1], model.actuator_ctrlrange[a, 1]),
                        )
                if (
                    not np.isfinite(limits).all()
                    or (limits[:, 0] >= limits[:, 1]).any()
                ):
                    raise ValueError(
                        "Finite intersected finger/control limits required"
                    )
                self.pads.append(
                    dict(
                        role=role,
                        group=group,
                        gid=gid,
                        body=int(body),
                        point_body=np.asarray(frame["point_body_m"]),
                        point_object=transform[:3, :3].T @ (near[i] - transform[:3, 3]),
                        normal_object=transform[:3, :3].T
                        @ world_mesh.face_normals[face[i]],
                        qids=qids,
                        dofs=dofs,
                        limits=limits,
                        offset=np.zeros(len(joints)),
                        updated=0,
                        frozen=0,
                        loaded_continuation=0,
                        force=0.0,
                        normal_gap=0.0,
                        tangent_gap=0.0,
                    )
                )
        all_q = [int(q) for pad in self.pads for q in pad["qids"]]
        if len(set(all_q)) != len(all_q):
            raise ValueError("Independent per-pad finger joints required")
        self.posture = None
        if "posture_transition" in settings:
            from .payload_posture import PayloadPosture

            self.posture = PayloadPosture(
                model,
                data,
                settings["posture_transition"],
                candidate,
                self.pads,
                scene_sha256=scene_sha256,
                allowed_penetration_m=allowed_penetration_m,
            )

    def target(self, model, data, nominal, *, active, dt, retention_active=False,
               pressure_offsets=None, acquisition_active=False):
        import mujoco

        if pressure_offsets is not None:
            from .residual_pressure import validate_offsets

            validate_offsets(pressure_offsets)

        output = np.asarray(nominal, float).copy()
        if (
            output.shape != (model.nq,)
            or not np.isfinite(output).all()
            or not np.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError("Finite owned command and positive clock required")
        observations = object_pad_observations(
            model, data, self.object_id, [p["gid"] for p in self.pads]
        )
        if acquisition_active and "unloaded_acquisition" not in self.settings:
            raise ValueError("Earlier pad activation requires explicit acquisition settings")
        if self.payload:
            from .support_control import contact_wrench_on_body

            forces = np.zeros(2)
            isolated = True
            for i, contact in enumerate(data.contact):
                pair = {int(contact.geom1), int(contact.geom2)}
                if not any(model.geom_bodyid[g] == self.object_id for g in pair):
                    continue
                owned = False
                for role, number in (("primary", 0), ("secondary", 1)):
                    if pair & self.role_geoms[role]:
                        forces[number] += contact_wrench_on_body(
                            model, data, i, self.object_id
                        )[0][2]
                        owned = True
                if not owned:
                    isolated = False
            dof = int(model.joint("task_object_free").dofadr[0])
            previous_updates = self.payload.valid_updates
            self.payload.update(
                forces,
                data.qacc[dof + 2],
                -model.opt.gravity[2],
                dt,
                active=active,
                isolated=isolated
                and all(
                    any(
                        observations[p["gid"]]["force_n"] > 0.1
                        for p in self.pads
                        if p["role"] == role
                    )
                    for role in ("primary", "secondary")
                ),
            )
            if self.posture:
                output = self.posture.apply(
                    output,
                    self.pads,
                    self.payload.weight_n / self.payload.settings["reference_weight_n"],
                    dt,
                    active=active and self.payload.valid_updates > previous_updates,
                )
        # Refresh only positions/Jacobians. Do not assign a physical coordinate.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        rotation = data.xmat[self.object_id].reshape(3, 3)
        position = data.xpos[self.object_id]
        for pad in self.pads:
            p = (
                data.xpos[pad["body"]]
                + data.xmat[pad["body"]].reshape(3, 3) @ pad["point_body"]
            )
            normal = rotation @ pad["normal_object"]
            error = p - (position + rotation @ pad["point_object"])
            normal_gap = float(error @ normal)
            tangent_gap = float(np.linalg.norm(error - normal_gap * normal))
            observation = observations[pad["gid"]]
            seek = False
            if "unloaded_acquisition" in self.settings:
                from .finger_acquisition import seeking

                seek = seeking(
                    pad, observation["force_n"],
                    active=acquisition_active,
                    threshold_n=self.settings.get("minimum_observed_force_n", 0.1),
                )
                pad["acquisition_seeking"] = seek
            pad.update(
                force=observation["force_n"],
                normal_gap=normal_gap,
                tangent_gap=tangent_gap,
            )
            ids = pad["qids"]
            q = output[ids]
            goal_n = self.settings["target_force_n"][pad["group"]]
            if self.payload:
                goal_n = self.payload.target(pad["role"], goal_n, group=pad["group"])
            if pressure_offsets is not None:
                from .residual_pressure import pressure_goal

                channel = pad["role"] + ":" + (
                    "thumb" if pad["group"] == "thumb" else "fingers"
                )
                goal_n = pressure_goal(goal_n, pressure_offsets[channel])
            pad["target_n"] = goal_n
            observed = (
                self.settings.get("normal_observation", "planned") == "solved_contact"
                and pad["force"] >= self.settings.get("minimum_observed_force_n", 0.1)
                and observation["outward_normal_world"] is not None
            )
            pad["normal_source"] = "solved_contact" if observed else "planned"
            if observed:
                p = observation["point_world_m"]
                normal = observation["outward_normal_world"]
            pad_active = active or (acquisition_active and pad["group"] != "thumb")
            valid = pad_active and (
                observed
                or contact_geometry_allowed(
                    normal_gap, tangent_gap, pad["force"], self.settings
                )
            )
            if valid:
                pad["loaded_continuation"] += int(
                    not observed
                    and tangent_gap > self.settings["maximum_tangent_gap_m"]
                )
                jac = np.zeros((3, model.nv))
                tangent = {}
                if self.anchors:
                    body_pose, object_pose = np.eye(4), np.eye(4)
                    body_pose[:3, :3] = data.xmat[pad["body"]].reshape(3, 3)
                    body_pose[:3, 3] = data.xpos[pad["body"]]
                    object_pose[:3, :3] = rotation
                    object_pose[:3, 3] = position
                    task = self.anchors.task(
                        pad["role"] + ":" + pad["group"],
                        body_pose,
                        object_pose,
                        observation["point_world_m"],
                        normal,
                        pad["force"],
                        active=bool(retention_active),
                    )
                    if task is not None:
                        p, projection, velocity = task
                        mujoco.mj_jac(model, data, jac, None, p, pad["body"])
                        tangent = dict(
                            tangent_jacobian=projection @ jac[:, pad["dofs"]],
                            tangent_velocity=velocity,
                        )
                if not tangent:
                    mujoco.mj_jac(model, data, jac, None, p, pad["body"])
                if "tangent_gain_s_inv" in self.settings:
                    from .wrist_coordination import bounded_vector

                    projection = np.eye(3) - np.outer(normal, normal)
                    tangent = dict(
                        tangent_jacobian=projection @ jac[:, pad["dofs"]],
                        tangent_velocity=bounded_vector(
                            -self.settings["tangent_gain_s_inv"] * (projection @ error),
                            self.settings["max_tangent_speed_m_s"],
                        ),
                    )
                rates = self.settings
                if seek:
                    rates = dict(self.settings, **{
                        k: self.settings["unloaded_acquisition"][k]
                        for k in ("gain_m_per_n_s", "max_normal_speed_m_s")
                    })
                command, offset = normal_force_step(
                    q,
                    pad["offset"],
                    normal @ jac[:, pad["dofs"]],
                    pad["force"],
                    goal_n,
                    dt,
                    pad["limits"],
                    rates,
                    **tangent,
                )
                output[ids] = command
                pad["offset"] = offset
                pad["updated"] += 1
            elif pad_active:
                output[ids] = np.clip(q + pad["offset"], *pad["limits"].T)
                pad["offset"] = output[ids] - q
                pad["frozen"] += 1
            # Before activation the archived approach/close command is exact.
        return output

    def diagnostics(self):
        return dict(
            **(dict(contact_anchor=self.anchors.diagnostics()) if self.anchors else {}),
            **(
                dict(payload_posture=self.posture.diagnostics()) if self.posture else {}
            ),
            **(
                dict(payload_pressure=self.payload.diagnostics())
                if self.payload
                else {}
            ),
            contact_force_feedback={
                p["role"] + ":" + p["group"]: dict(
                    force_n=p["force"],
                    normal_source=p.get("normal_source", "planned"),
                    target_n=p.get(
                        "target_n", self.settings["target_force_n"][p["group"]]
                    ),
                    joint_offset_rad=p["offset"].tolist(),
                    normal_gap_m=p["normal_gap"],
                    tangent_gap_m=p["tangent_gap"],
                    updated_steps=p["updated"],
                    frozen_steps=p["frozen"],
                    loaded_continuation_steps=p["loaded_continuation"],
                    **(dict(
                        acquisition_contact_seen=p.get("acquisition_contact_seen", False),
                        acquisition_seeking=p.get("acquisition_seeking", False),
                    ) if "unloaded_acquisition" in self.settings else {}),
                )
                for p in self.pads
            },
        )
