"""Continuous, actuator-only policy tools for reconstructed manipulation scenes.

Optional MuJoCo loads on construction. Policies receive serialized observations
and tools, never model/data objects. The downstream host owns isolation, asset
hash checks, fixed task configuration and author permissions.
"""

import time
from copy import deepcopy

import numpy as np

from real2sim2real.mujoco_assets import load_model as load_mujoco_model

from .pad_pressure_correction import PadPressureCorrection
from .simulation_health import step_rejection


class ContinuousPolicy:
    def __init__(self, config):
        import mujoco
        from real2sim2real.retargeting.grasp_kinematics import robot_indices

        from .arm_path_audit import place_bases

        self.mj = mujoco
        self.config = deepcopy(config)
        self.model = m = load_mujoco_model(config["scene"])
        place_bases(m, {"base_body": config["base_body"]}, config["base_placement"])
        self.data = d = mujoco.MjData(m)
        with np.load(config["initial"]) as a:
            d.qpos[:] = a["qpos"]
            d.qvel[:] = a["qvel"]
        self.names = [m.joint(int(j)).name for j in m.actuator_trnid[:, 0]]
        self.jids = m.actuator_trnid[:, 0].astype(int)
        if len(set(self.names)) != len(self.names) or not np.all(
            m.jnt_limited[self.jids]
        ):
            raise ValueError("Unique limited robot actuator joints required")
        if not np.all(m.jnt_type[self.jids] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Only scalar robot hinges may be commanded")
        self.qids = m.jnt_qposadr[self.jids]
        self.dofs = m.jnt_dofadr[self.jids]
        self.limits = m.jnt_range[self.jids]
        self.speeds = np.array([config["velocity_limits_rad_s"][n] for n in self.names])
        self.arm = [self.names.index(n) for n in config["arm_joints"]]
        self.hand = robot_indices(m, config["hand_root"])
        self.finger = [
            self.names.index(m.joint(int(j)).name) for j in self.hand["joints"]
        ]
        self.object = m.body(config["object_body"]).id
        self.wrist = m.body(config["hand_root"]).id
        if self.object in set(m.jnt_bodyid[self.jids]):
            raise ValueError("Object must not be actuated")
        self.robot = set(self.hand["bodies"]) | {
            int(m.jnt_bodyid[j]) for j in self.jids
        }
        # Include fixed arm descendants in collision checks.
        base = m.body(config["base_body"]).id
        for b in range(1, m.nbody):
            p = b
            while p and p != base:
                p = int(m.body_parentid[p])
            if p == base:
                self.robot.add(b)
        self.support = {m.geom(n).id for n in config["support_geoms"]}
        self.pad_ids = [m.geom(f["geom"]).id for f in config["pad_frames"]]
        self.nominal = d.qpos[self.qids].copy()
        self.home = self.nominal.copy()
        self.dt = float(m.opt.timestep)
        self.feedback = config["feedback_s"]
        self.tick = 0
        self.feedback_steps = round(self.feedback / self.dt)
        self.sample_steps = round(config["sample_s"] / self.dt)
        if self.sample_steps < 1 or not np.isclose(
            self.sample_steps * self.dt, config["sample_s"]
        ):
            raise ValueError("Integer physics/sample ratio required")
        if self.feedback_steps < 1 or not np.isclose(
            self.feedback_steps * self.dt, self.feedback
        ):
            raise ValueError("Integer physics/feedback ratio required")
        self.pressure = None
        self.pressure_targets = None
        self.fault = None
        self.history = []
        self.states = []
        self.velocities = []
        self.commands = []
        self.events = []
        self.deadline = time.monotonic() + config["wall_deadline_s"]
        self.support_since = None
        self.mj.mj_forward(m, d)
        self._sample("initial")

    def _alive(self):
        if self.fault:
            raise RuntimeError(self.fault)
        if time.monotonic() > self.deadline:
            self.fault = "wall_deadline"
            raise RuntimeError(self.fault)

    def _vector(self, positions):
        if not isinstance(positions, dict) or set(positions) != set(self.names):
            raise ValueError("Explicit full robot joint map required")
        q = np.array([positions[n] for n in self.names], float)
        if (
            not np.isfinite(q).all()
            or (q < self.limits[:, 0]).any()
            or (q > self.limits[:, 1]).any()
        ):
            raise ValueError("Command outside finite native joint limits")
        return q

    def _contacts(self, data):
        from .arm_path_audit import joint_neighbors
        from .support_control import contact_wrench_on_body

        m = self.model
        forces = np.zeros(len(self.pad_ids))
        support = hand = peak = 0.0
        witness = None
        for ci, co in enumerate(data.contact):
            gs = [int(co.geom1), int(co.geom2)]
            bs = [int(m.geom_bodyid[g]) for g in gs]
            if (
                co.dist < 0
                and (
                    (any(b in self.robot for b in bs) and not joint_neighbors(m, *bs))
                    or (self.object in bs and 0 in bs)
                )
                and -co.dist > peak
            ):
                peak = float(-co.dist)
                witness = [m.geom(g).name for g in gs]
            if self.object not in bs:
                continue
            other = gs[0] if bs[1] == self.object else gs[1]
            f, _ = contact_wrench_on_body(m, data, ci, self.object)
            cf = np.zeros(6)
            self.mj.mj_contactForce(m, data, ci, cf)
            if other in self.pad_ids:
                forces[self.pad_ids.index(other)] += max(0, float(cf[0]))
            if other in self.hand["geoms"]:
                hand += max(0, float(cf[0]))
            if other in self.support:
                support += float(f[2])
        return forces, support, hand, peak, witness

    def observe(self):
        d = self.data
        m = self.model
        self.mj.mj_forward(m, d)
        f, s, h, p, w = self._contacts(d)
        velocity = np.zeros(6)
        self.mj.mj_objectVelocity(
            m, d, self.mj.mjtObj.mjOBJ_BODY, self.object, velocity, 0
        )
        points, normals = [], []
        for frame in self.config["pad_frames"]:
            bid = m.body(frame["body"]).id
            rotation = d.xmat[bid].reshape(3, 3)
            points.append(
                (d.xpos[bid] + rotation @ np.asarray(frame["point_body_m"])).tolist()
            )
            normal = np.asarray(frame["normal_body"], float)
            normals.append((rotation @ (normal / np.linalg.norm(normal))).tolist())
        return {
            "time_s": float(d.time),
            "joints_rad": dict(zip(self.names, d.qpos[self.qids].tolist())),
            "commanded_joints_rad": dict(zip(self.names, self.nominal.tolist())),
            "object_position_m": d.xpos[self.object].tolist(),
            "object_com_m": d.xipos[self.object].tolist(),
            "object_rotation": d.xmat[self.object].reshape(3, 3).tolist(),
            "wrist_position_m": d.xpos[self.wrist].tolist(),
            "wrist_rotation": d.xmat[self.wrist].reshape(3, 3).tolist(),
            "pad_forces_n": f.tolist(),
            "pad_points_world_m": points,
            "pad_normals_world": normals,
            "hand_force_n": h,
            "support_fraction": s / (float(m.body_mass[self.object]) * 9.81),
            "object_speed_m_s": float(np.linalg.norm(velocity[3:])),
            "penetration_m": p,
            "collision_pair": w,
            "fault": self.fault,
        }

    def describe(self) -> dict:
        return {
            "joints": self.names,
            "joint_limits_rad": dict(zip(self.names, self.limits.tolist())),
            "velocity_limits_rad_s": dict(zip(self.names, self.speeds.tolist())),
            "home_joints_rad": dict(zip(self.names, self.home.tolist())),
            "arm_joints": self.config["arm_joints"],
            "pad_frames": deepcopy(self.config["pad_frames"]),
            "horizon_s": self.config["horizon_s"],
            "task": deepcopy(self.config["task"]),
            "observation_class": "privileged simulator state",
            "geometry_uncertainty": "downstream frozen conditional scene",
        }

    def _sample(self, label):
        obs = self.observe()
        obs["label"] = label
        self.history.append(obs)
        self.states.append(self.data.qpos.copy())
        self.velocities.append(self.data.qvel.copy())
        self.commands.append(self.data.ctrl.copy())
        if obs["penetration_m"] > self.config["penetration_limit_m"]:
            self.fault = "physical_penetration"
        if obs["object_com_m"][2] < self.config.get("minimum_object_height_m", -0.1):
            self.fault = "object_below_scene"
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        ):
            self.fault = "nonfinite_state"

    def _validate_path(self, positions, durations_s):
        self._alive()
        if not 1 <= len(positions) <= 100 or len(positions) != len(durations_s):
            raise ValueError("One to 100 explicit path segments required")
        goals = [self._vector(q) for q in positions]
        durations = np.asarray(durations_s, float)
        if (
            not np.isfinite(durations).all()
            or (durations <= 0).any()
            or self.data.time + durations.sum() > self.config["horizon_s"] + 1e-9
        ):
            raise ValueError("Path exceeds positive finite task budget")
        if not np.allclose(
            durations / self.feedback,
            np.rint(durations / self.feedback),
            rtol=0,
            atol=1e-8,
        ):
            raise ValueError("Durations must align to feedback ticks")
        start = self.nominal.copy()
        for goal, duration in zip(goals, durations):
            if (1.5 * abs(goal - start) / duration > self.speeds + 1e-9).any():
                raise ValueError("Path exceeds declared velocity limits")
            start = goal
        return goals, durations

    def screen_path(self, positions: list, durations_s: list) -> dict:
        goals, durations = self._validate_path(positions, durations_s)
        scratch = self.mj.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        start = self.nominal.copy()
        peak = 0.0
        witness = None
        for goal, duration in zip(goals, durations):
            for u in np.linspace(0, 1, round(duration / self.feedback) + 1):
                scratch.qpos[self.qids] = start + (goal - start) * u * u * (3 - 2 * u)
                self.mj.mj_forward(self.model, scratch)
                _, _, _, p, w = self._contacts(scratch)
                if p > peak:
                    peak = p
                    witness = w
            start = goal
        return {
            "admitted": peak <= self.config["penetration_limit_m"],
            "peak_penetration_m": peak,
            "witness": witness,
            "sample_s": self.feedback,
            "continuous_collision_certificate": False,
            "time_s": float(self.data.time),
        }

    def execute_joint_path(self, positions: list, durations_s: list) -> dict:
        screen = self.screen_path(positions, durations_s)
        if not screen["admitted"]:
            return {"executed": False, "screen": screen, "observation": self.observe()}
        return self._execute_screened_path(positions, durations_s)

    def _execute_screened_path(self, positions, durations_s, *, feedback_guard=None):
        """Trusted-host entry after a screen; never exposed as a candidate tool.

        A transport adapter may supply a contact-aware screen and a read-only
        guard evaluated on the existing feedback observation. The actuator loop
        and its clock remain identical to ordinary execute_joint_path.
        """
        self._validate_path(positions, durations_s)
        from real2sim2real.retargeting.pad_contacts import observe_frames

        from .contact_force_control import object_pad_observations

        for mapping, duration in zip(positions, durations_s):
            start = self.nominal.copy()
            goal = self._vector(mapping)
            count = round(duration / self.dt)
            for i in range(count):
                self._alive()
                u = i / count
                q = start + (goal - start) * u * u * (3 - 2 * u)
                v = (goal - start) * 6 * u * (1 - u) / duration
                if self.tick % self.feedback_steps == 0:
                    self.mj.mj_forward(self.model, self.data)
                    if self.pressure is not None:
                        _, normals, jac, _ = observe_frames(
                            self.model,
                            self.data,
                            self.config["pad_frames"],
                            self.hand,
                            jacobian=True,
                        )
                        observed = object_pad_observations(
                            self.model, self.data, self.object, self.pad_ids
                        )
                        forces = []
                        outward = -normals
                        for k, g in enumerate(self.pad_ids):
                            o = observed[g]
                            forces.append(o["force_n"])
                            if (
                                o["force_n"] > 0.1
                                and o["outward_normal_world"] is not None
                            ):
                                outward[k] = o["outward_normal_world"]
                                jp = np.zeros((3, self.model.nv))
                                self.mj.mj_jac(
                                    self.model,
                                    self.data,
                                    jp,
                                    None,
                                    o["point_world_m"],
                                    self.model.body(
                                        self.config["pad_frames"][k]["body"]
                                    ).id,
                                )
                                jac[k] = jp[:, self.hand["dofs"]]
                        q[self.finger] = self.pressure.update(
                            q[self.finger],
                            np.einsum("ij,ijk->ik", outward, jac),
                            forces,
                            self.feedback,
                            targets_n=self.pressure_targets,
                        )
                    d = self.data
                    m = self.model
                    for ai, j in enumerate(self.jids):
                        ff = v[ai] if ai in self.arm else 0.0
                        gain = m.actuator_gainprm[ai, 0]
                        kv = -m.actuator_biasprm[ai, 2]
                        value = q[ai] + (d.qfrc_bias[self.dofs[ai]] + kv * ff) / gain
                        d.ctrl[ai] = (
                            np.clip(value, *m.actuator_ctrlrange[ai])
                            if m.actuator_ctrllimited[ai]
                            else value
                        )
                    obs = self.observe()
                    if feedback_guard is not None:
                        reason = feedback_guard(obs)
                        if reason:
                            self.fault = reason
                            raise RuntimeError(reason)
                    supported = obs["support_fraction"] >= self.config["task"].get(
                        "minimum_support_fraction", 0.5
                    ) and obs["object_speed_m_s"] < self.config["task"].get(
                        "maximum_object_speed_m_s", 0.02
                    )
                    self.support_since = (
                        (
                            float(d.time)
                            if self.support_since is None
                            else self.support_since
                        )
                        if supported
                        else None
                    )
                before = float(self.data.time)
                self.mj.mj_step(self.model, self.data)
                self.tick += 1
                failure = step_rejection(
                    before,
                    float(self.data.time),
                    self.dt,
                    {str(i): int(w.number) for i, w in enumerate(self.data.warning)},
                )
                if failure:
                    self.fault = failure
                if self.tick % self.sample_steps == 0:
                    self._sample("execute")
                if self.fault:
                    raise RuntimeError(self.fault)
            self.nominal = goal.copy()
        self.events.append(
            {
                "tool": "execute_joint_path",
                "end_s": float(self.data.time),
                "segments": len(positions),
            }
        )
        return {"executed": True, "observation": self.observe()}

    def _fork_for_preview(self):
        """Clone simulator/controller state for a bounded trusted-host forecast.

        Model parameters are shared read-only; integration state, cached dynamics,
        pressure memory, commands and records are independent. No reset or extra
        forward call occurs on the live session. The wall deadline is not extended.
        """
        from copy import copy, deepcopy

        clone = object.__new__(type(self))
        for key, value in vars(self).items():
            if key in {"mj", "model"}:
                setattr(clone, key, value)
            elif key == "data":
                clone.data = copy(value)
            elif key in {"history", "states", "velocities", "commands", "events"}:
                setattr(clone, key, [])
            else:
                setattr(clone, key, deepcopy(value))
        return clone

    def hold(self, duration_s: float) -> dict:
        return self.execute_joint_path(
            [dict(zip(self.names, self.nominal.tolist()))], [duration_s]
        )

    def plan_wrist(self, pose_world: list, seed_joints_rad: dict) -> dict:
        from .arm_path_audit import fit_pose

        self._alive()
        pose = np.asarray(pose_world, float)
        from .arm_object_feedback import rigid_pose

        rigid_pose(pose)
        q = self._vector(seed_joints_rad)
        scratch = self.mj.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self.qids] = q
        fitted, pe, re = fit_pose(
            self.model,
            scratch,
            self.wrist,
            self.config["arm_joints"],
            pose,
            q[self.arm],
            160,
        )
        q[self.arm] = fitted
        return {
            "joints_rad": dict(zip(self.names, q.tolist())),
            "position_error_m": float(pe),
            "rotation_error_rad": float(re),
            "ik_pass": bool(pe <= 0.002 and re <= np.deg2rad(2)),
            "path_admission_required": True,
        }

    def fit_hand(
        self,
        seed_joints_rad: dict,
        pad_indices: list,
        targets_world_m: list,
        inward_normals_world: list,
    ) -> dict:
        self._alive()
        q = self._vector(seed_joints_rad)
        if (
            not pad_indices
            or len(set(pad_indices)) != len(pad_indices)
            or any(
                type(i) is not int or not 0 <= i < len(self.pad_ids)
                for i in pad_indices
            )
        ):
            raise ValueError("Declared unique pad indices required")
        from real2sim2real.retargeting.pad_contacts import fit_pad_contacts

        scratch = self.mj.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self.qids] = q
        result = fit_pad_contacts(
            self.model,
            scratch,
            self.config["hand_root"],
            [self.config["pad_frames"][i] for i in pad_indices],
            targets_world_m,
            inward_normals_world,
            q[self.finger],
            max_nfev=160,
            position_scale_m=0.0015,
            normal_guard_deg=25,
        )
        q[self.finger] = result["q"]
        return {
            "joints_rad": dict(zip(self.names, q.tolist())),
            "pad_errors_m": result["site_errors_m"].tolist(),
            "normal_errors_deg": result["normal_errors_deg"].tolist(),
            "path_admission_required": True,
        }

    def prepare_pressure(self, targets_n):
        """Validate targets/controller configuration without changing live commands."""
        self._alive()
        targets = np.asarray(targets_n, float)
        if (
            targets.shape != (len(self.pad_ids),)
            or not np.isfinite(targets).all()
            or (targets < 0).any()
            or (targets > 5).any()
        ):
            raise ValueError("Declared pad targets in [0,5] N required")
        if (targets == 0).all():
            return None, None
        elif (targets <= 0).any():
            raise ValueError("Use all positive pad targets or all zero to disable")
        else:
            groups = []
            for f in self.config["pad_frames"]:
                finger = f["body"].split("_")[1]
                groups.append(
                    [
                        i
                        for i, j in enumerate(self.hand["joints"])
                        if "_" + finger + "_" in self.model.joint(int(j)).name
                    ]
                )
            pressure = self.pressure
            if pressure is None:
                pressure = PadPressureCorrection(
                    groups,
                    self.model.jnt_range[self.hand["joints"]],
                    self.config["pressure_settings"],
                )
            return pressure, targets.copy()

    def set_pressure(self, targets_n: list) -> dict:
        self.pressure, self.pressure_targets = self.prepare_pressure(targets_n)
        return {
            "targets_n": np.asarray(targets_n, float).tolist(),
            "time_s": float(self.data.time),
        }

    def release(self, open_joints_rad: dict, duration_s: float) -> dict:
        self._alive()
        self._vector(open_joints_rad)
        if (
            self.support_since is None
            or self.data.time - self.support_since
            < self.config["task"]["support_dwell_s"]
        ):
            return {
                "released": False,
                "reason": "supported placement dwell required",
                "observation": self.observe(),
            }
        screen = self.screen_path([open_joints_rad], [duration_s])
        if not screen["admitted"]:
            return {"released": False, "reason": "path collision", "screen": screen}
        self.set_pressure([0.0] * len(self.pad_ids))
        result = self.execute_joint_path([open_joints_rad], [duration_s])
        result["released"] = bool(result["executed"])
        return result

    def save(self, output):
        from pathlib import Path

        from real2sim2real.artifacts import write_json

        output = Path(output)
        output.mkdir(exist_ok=False)
        if not np.isclose(
            self.history[-1]["time_s"], self.data.time, rtol=0, atol=1e-10
        ):
            self._sample("final")
        np.savez_compressed(
            output / "states.npz",
            qpos=self.states,
            qvel=self.velocities,
            commands=self.commands,
            timestamps_s=[r["time_s"] for r in self.history],
        )
        write_json(output / "observations.json", self.history)
        write_json(output / "events.json", self.events)
        write_json(
            output / "status.json",
            {
                "time_s": float(self.data.time),
                "fault": self.fault,
                "task_evaluation": "downstream evaluator required",
                "control_accepted": False,
            },
        )


def register_policy_tools(registry, session):
    """Register serialized tools only; no reset, model edit or object command."""
    for name in [
        "describe",
        "observe",
        "plan_wrist",
        "fit_hand",
        "screen_path",
        "execute_joint_path",
        "hold",
        "set_pressure",
        "release",
    ]:
        registry.register_callable(
            "robot." + name,
            getattr(session, name),
            summary=name.replace("_", " "),
            requires_ctx=False,
        )
