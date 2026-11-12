"""CPU binding of measured slip/tilt feedback to one existing arm-target owner."""

from pathlib import Path
import numpy as np

from real2sim2real.artifacts import digest
from .paired_acquisition_audit import pinned_json
from .slip_tilt_feedback import SlipTiltFeedback
from .arm_object_feedback import joint_offset


def validate_source(spec, config):
    if config.get("descent_pad_control") is not None:
        from .descent_pad_control import validate_source as validate_descent

        validate_descent(config["descent_pad_control"], config)
    if set(spec) != {"baseline", "controller"}:
        raise ValueError("Pinned baseline and explicit feedback controller required")
    source = Path(spec["baseline"]["path"])
    baseline = pinned_json(spec["baseline"])
    if (
        not baseline.get("physics_run")
        or baseline.get("motion_condition") != "demonstrated"
        or baseline.get("residual_adapter")
        or baseline.get("bounded_grasp_correction")
        or baseline.get("closed_loop_grasp_feedback")
        or "incomplete_pilot" in baseline["rejected_for"]
    ):
        raise ValueError("Complete uncorrected continuous physics baseline required")
    from .continuous_residual import validate_baseline

    validate_baseline(baseline)
    for name, key in [("states.npz", "states_sha256"), ("scene.xml", "scene_sha256")]:
        if digest(source.with_name(name)) != baseline[key]:
            raise ValueError("Frozen physical baseline changed")
    ignored = {
        "output",
        "purpose",
        "motion_condition",
        "closed_loop_grasp_feedback",
        "descent_pad_control",
        "table_load_transfer",
        "record_arm_commands",
    }
    strip = lambda c: {k: v for k, v in c.items() if k not in ignored}
    if strip(config) != strip(baseline["config"]) or config.get(
        "motion_condition"
    ) not in ("demonstrated", "stationary_hold"):
        raise ValueError(
            "Closed-loop probes must retain the frozen controller/physics/reference"
        )
    SlipTiltFeedback(spec["controller"], baseline["continuous_protocol"])


def pose(data, body):
    result = np.eye(4)
    result[:3, 3] = data.xpos[body]
    result[:3, :3] = data.xmat[body].reshape(3, 3)
    return result


class ClosedLoopGrasp:
    def __init__(
        self,
        model,
        data,
        roots,
        arms,
        qids,
        vids,
        ranges,
        speeds,
        parent,
        settings,
        control_period,
        robot_geoms,
        arm_geoms,
        object_geoms,
        hand_geoms,
        criteria,
        table_settings=None,
        table_geom=None,
    ):
        import mujoco

        self.model, self.live = model, data
        self.measured = mujoco.MjData(model)
        self.command = mujoco.MjData(model)
        self.guard = mujoco.MjData(model)
        self.roots, self.arms, self.qids, self.vids = roots, arms, qids, vids
        self.ranges, self.speeds, self.parent = ranges, speeds, parent
        self.robot_geoms, self.arm_geoms, self.object_geoms = (
            robot_geoms,
            arm_geoms,
            object_geoms,
        )
        self.hand_geoms = hand_geoms
        self.criteria = criteria
        self.obj = model.body("task_object").id
        ratio = settings["observation_period_s"] / control_period
        if not np.isclose(ratio, round(ratio), atol=1e-9, rtol=0) or ratio < 1:
            raise ValueError(
                "Observation period must divide the native controller clock"
            )
        self.stride = int(round(ratio))
        self.tick = 0
        self.feedback = SlipTiltFeedback(settings, parent["continuous_protocol"])
        self.forces = np.zeros(2)
        self.ever_corrected = False
        self.guard_checks = self.guard_rejections = self.rate_limited = 0
        self.rows = []
        self.previous_checked = None
        self.table_policy = None
        if table_settings is not None:
            from .table_load_transfer import TableLoadTransfer

            if table_geom is None:
                raise ValueError("Explicit table geometry required")
            self.table_geom = table_geom
            self.table_policy = TableLoadTransfer(
                table_settings, parent["continuous_protocol"]
            )
            self.table_force_n, self.table_contacts = 0.0, 0

    def record(self, record):
        self.feedback.record(record)
        if self.table_policy is not None:
            record["table_load_transfer"] = self.table_policy.diagnostics()
        self.rows.append(
            dict(
                time_s=record["time_s"],
                **self.feedback.diagnostics(),
                forces_n=self.forces.tolist(),
            )
        )

    def target(self, time, dt, nominal, previous, i, alpha):
        import mujoco

        m, d = self.model, self.measured
        if self.tick % self.stride == 0:
            for field in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
                getattr(d, field)[:] = getattr(self.live, field)
            d.time = self.live.time
            mujoco.mj_forward(m, d)
            self.actual = pose(d, self.obj)
            if self.table_policy is not None:
                from .table_load_transfer import observe_table_force

                self.table_force_n, self.table_contacts = observe_table_force(
                    m, d, self.object_geoms, self.table_geom, self.obj
                )
            self.wrists = [pose(d, self.roots[r]) for r in ("right", "left")]
            self.forces[:] = 0
            for cid, c in enumerate(d.contact):
                pair = {int(c.geom1), int(c.geom2)}
                if not pair & self.object_geoms:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(m, d, cid, force)
                for k, r in enumerate(("right", "left")):
                    if pair & self.hand_geoms[r]:
                        self.forces[k] += max(0, float(force[0]))
        self.tick += 1
        source, following = self.parent["records"][i : i + 2]
        from scipy.spatial.transform import Rotation

        ref = np.eye(4)
        ref[:3, 3] = (1 - alpha) * np.asarray(
            source["reference_position_m"]
        ) + alpha * np.asarray(following["reference_position_m"])
        first = np.asarray(source["reference_rotation"])
        rv = Rotation.from_matrix(
            np.asarray(following["reference_rotation"]) @ first.T
        ).as_rotvec()
        ref[:3, :3] = Rotation.from_rotvec(alpha * rv).as_matrix() @ first
        cmd = self.command
        cmd.qpos[:] = self.live.qpos
        cmd.qpos[self.qids] = nominal
        mujoco.mj_kinematics(m, cmd)
        mujoco.mj_comPos(m, cmd)
        nominal_wrists = [pose(cmd, self.roots[r]) for r in ("right", "left")]
        offsets = self.feedback.update(
            time, dt, self.actual, ref, self.wrists, nominal_wrists, self.forces
        )
        if self.table_policy is not None:
            self.table_policy.advance(
                time, dt, self.table_force_n, float(m.body_mass[self.obj] * 9.81),
                table_contacts=self.table_contacts, phase=source["phase"],
            )
            offsets = self.table_policy.compose(
                offsets, self.feedback.settings["max_offset_m"]
            )
        if not np.any(offsets) and not self.ever_corrected:
            return nominal
        if not self.ever_corrected:
            self.previous_checked = previous.copy()
        self.ever_corrected = True
        target = nominal.copy()
        start = 0
        for arm in self.arms:
            role = arm["name"]
            k = ("right", "left").index(role)
            n = len(arm["joints"])
            sl = slice(start, start + n)
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, cmd, jp, jr, self.roots[role])
            target[sl] += joint_offset(
                np.vstack([jp, jr])[:, self.vids[sl]], offsets[k], max_offset_rad=0.08
            )
            start += n
        margin = self.criteria["joint_margin_rad"]
        target = np.clip(target, self.ranges[:, 0] + margin, self.ranges[:, 1] - margin)
        bounded = np.clip(
            target, previous - self.speeds * dt, previous + self.speeds * dt
        )
        self.rate_limited += int(not np.allclose(target, bounded, atol=1e-12, rtol=0))
        return bounded

    def check_command(self, target):
        """Screen the requested command and midpoint in scratch; abort on failure."""
        import mujoco
        from .arm_path_audit import joint_neighbors

        if not self.ever_corrected:
            return True
        m, d = self.model, self.guard
        d.qpos[:] = self.live.qpos
        checks = [target]
        if self.previous_checked is not None:
            checks.insert(0, (self.previous_checked + target) / 2)
        for q in checks:
            d.qpos[self.qids] = q
            mujoco.mj_fwdPosition(m, d)
            self.guard_checks += 1
            for c in d.contact:
                pair = {int(c.geom1), int(c.geom2)}
                if pair & self.object_geoms or not pair & self.robot_geoms:
                    continue
                bodies = [int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2)]
                if pair & self.arm_geoms and joint_neighbors(m, *bodies):
                    continue
                allowed = (
                    self.criteria["arm_penetration_m"]
                    if pair & self.arm_geoms
                    else self.criteria["robot_penetration_m"]
                )
                if -float(c.dist) > allowed:
                    self.guard_rejections += 1
                    self.worst_contact = dict(
                        geoms=[m.geom(g).name for g in (c.geom1, c.geom2)],
                        penetration_m=-float(c.dist),
                    )
                    return False
        self.previous_checked = target.copy()
        return True

    def finish(self, out):
        from real2sim2real.artifacts import write_json

        write_json(out / "closed-loop-trace.json", self.rows)
        return dict(
            settings=self.feedback.settings,
            **self.feedback.diagnostics(),
            guard_checks=self.guard_checks,
            guard_rejections=self.guard_rejections,
            worst_contact=getattr(self, "worst_contact", None),
            rate_limited_steps=self.rate_limited,
            trace_sha256=digest(out / "closed-loop-trace.json"),
            promotion_eligible=False,
            observations="privileged separate-data simulator poses and coherent solved hand/object contact forces",
            observation_period_s=self.feedback.settings["observation_period_s"],
            collision_check="sampled requested command and preceding-command midpoint; not continuous collision certification",
            training_run=False,
        )
