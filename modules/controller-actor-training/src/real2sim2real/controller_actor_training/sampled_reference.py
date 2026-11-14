"""Portable replay of a sampled nominal reference with live contact feedback.

Inputs are planned finger references, arm knots and a single initial state, not
recorded actuator commands or future rollout states. Pressure, release and home
gates are recomputed from fresh physics. This preserves the retained controller's
sampling order; it is intentionally separate from the broad joint-tool executor.
"""

import time
from pathlib import Path

import numpy as np
from real2sim2real.artifacts import digest, write_json
from real2sim2real.retargeting.grasp_kinematics import robot_indices
from real2sim2real.retargeting.pad_contacts import observe_frames
from scipy.spatial.transform import Rotation

from .arm_path_audit import joint_neighbors, place_bases
from .contact_force_control import object_pad_observations
from .diagnostic_verdict import DiagnosticVerdict
from .finger_release import FingerRelease
from .joint_boundary_path import JointBoundaryPath, return_admitted
from .object_grasp_pilot import verify_unassisted_model
from .pad_pressure_correction import PadPressureCorrection
from .simulation_health import step_rejection
from .support_control import contact_wrench_on_body


def simulation_clock(duration_s, timestep_s, feedback_s):
    """Mirror successive physics-clock additions, avoiding linspace roundoff."""
    if (
        not np.isfinite([duration_s, timestep_s, feedback_s]).all()
        or min(duration_s, timestep_s, feedback_s) <= 0
    ):
        raise ValueError("Positive finite duration and clock periods required")
    ratios = np.array([duration_s / timestep_s, feedback_s / timestep_s])
    if (
        not np.isfinite(ratios).all()
        or min(duration_s, timestep_s, feedback_s) <= 0
        or not np.allclose(ratios, np.rint(ratios), rtol=0, atol=1e-8)
        or not np.isclose(
            duration_s / feedback_s, round(duration_s / feedback_s), rtol=0, atol=1e-8
        )
    ):
        raise ValueError(
            "Positive duration and feedback aligned to physics steps required"
        )
    steps, interval = (round(x) for x in ratios)
    t, values = 0.0, []
    for step in range(steps + 1):
        if step % interval == 0:
            values.append(t)
        t += timestep_s
    return np.asarray(values)


def validate_reference(reference, config, finger_count, arm_count):
    fields = {
        "feedback_times_s",
        "reference_times_s",
        "previous_reference_times_s",
        "finger_targets_rad",
        "object_targets_world",
        "phases",
        "arm_times_s",
        "arm_q_rad",
        "pre_times_s",
        "pre_q_rad",
        "post_times_s",
        "post_q_rad",
    }
    if set(reference) != fields:
        raise ValueError(
            "Only declared planned-reference arrays allowed; no rollout commands/states"
        )
    expected = simulation_clock(
        config["duration_s"], config["timestep_s"], config["feedback_s"]
    )
    if not np.array_equal(reference["feedback_times_s"], expected):
        raise ValueError("Reference must cover the exact frozen feedback clock")
    n = len(expected)
    for key, shape in {
        "reference_times_s": (n,),
        "previous_reference_times_s": (n,),
        "finger_targets_rad": (n, finger_count),
        "object_targets_world": (n, 4, 4),
    }.items():
        if reference[key].shape != shape or not np.isfinite(reference[key]).all():
            raise ValueError("Invalid sampled reference array: " + key)
    if reference["phases"].shape != (n,) or reference["phases"].dtype.kind != "U":
        raise ValueError("One explicit phase label per feedback tick required")
    times, path = reference["arm_times_s"], reference["arm_q_rad"]
    if (
        times.ndim != 1
        or len(times) < 2
        or path.shape != (len(times), arm_count)
        or not np.isfinite(times).all()
        or not np.isfinite(path).all()
        or (np.diff(times) <= 0).any()
        or times[0] != 0
        or times[-1] < max(reference["reference_times_s"])
    ):
        raise ValueError("Finite increasing arm reference covering the task required")


class SampledReferenceReplay:
    def __init__(self, config):
        import mujoco

        self.config = c = config
        self.mj = mujoco
        self.model = m = mujoco.MjModel.from_xml_path(c["scene"])
        place_bases(m, {"base_body": c["base_body"]}, c["base_placement"])
        verify_unassisted_model(m, c["object_body"])
        if m.opt.timestep != c["timestep_s"]:
            raise ValueError("Scene and protocol timestep differ")
        self.data = d = mujoco.MjData(m)
        with np.load(c["initial"], allow_pickle=False) as a:
            if set(a.files) != {"qpos", "qvel"}:
                raise ValueError("Exactly one initial qpos/qvel state required")
            if a["qpos"].shape != (m.nq,) or a["qvel"].shape != (m.nv,):
                raise ValueError("Initial state dimensions differ from model")
            d.qpos[:] = a["qpos"]
            d.qvel[:] = a["qvel"]
        if not np.isfinite(np.r_[d.qpos, d.qvel]).all():
            raise ValueError("Finite initial state required")
        with np.load(c["reference"], allow_pickle=False) as a:
            self.ref = {key: a[key].copy() for key in a.files}
        self.arm_ids = [m.joint(name).id for name in c["arm_joints"]]
        self.aq = m.jnt_qposadr[self.arm_ids]
        self.hand = robot_indices(m, c["hand_root"])
        names = [m.joint(int(j)).name for j in self.hand["joints"]]
        if names != c["finger_joints"]:
            raise ValueError("Finger reference joint order differs from model")
        validate_reference(self.ref, c, len(names), len(self.arm_ids))
        self.pre = JointBoundaryPath(self.ref["pre_times_s"], self.ref["pre_q_rad"])
        self.post = JointBoundaryPath(self.ref["post_times_s"], self.ref["post_q_rad"])
        self.boundary_ids = [m.joint(n).qposadr[0] for n in c["boundary_joints"]]
        if c["boundary_joints"][: len(self.arm_ids)] != c["arm_joints"]:
            raise ValueError(
                "Boundary paths must begin with arm joints in reference order"
            )
        if self.pre.positions_rad.shape[1] != len(
            self.boundary_ids
        ) or self.post.positions_rad.shape[1] != len(self.boundary_ids):
            raise ValueError("Boundary joint map mismatch")
        self.obj = m.body(c["object_body"]).id
        self.wrist = m.body(c["hand_root"]).id
        self.support = {m.geom(n).id for n in c["support_geoms"]}
        self.table = {m.geom(n).id for n in c["table_geoms"]}
        self.robot_bodies = {m.body(n).id for n in c["robot_bodies"]}
        self.padlookup = {
            m.geom(f["geom"]).id: i for i, f in enumerate(c["pad_frames"])
        }
        self.pressure = PadPressureCorrection(
            c["pressure_groups"], m.jnt_range[self.hand["joints"]], c["pressure"]
        )
        self.release = FingerRelease(c["release"])
        self.feedback_steps = round(c["feedback_s"] / c["timestep_s"])
        self.sample_steps = round(c["sample_s"] / c["timestep_s"])
        if (
            self.sample_steps < 1
            or not np.isclose(self.sample_steps * c["timestep_s"], c["sample_s"])
            or self.sample_steps % self.feedback_steps
        ):
            raise ValueError("Logging must be aligned to feedback and physics")
        self.home = d.qpos.copy()
        mujoco.mj_forward(m, d)
        self.home_wrist = d.xpos[self.wrist].copy()
        self.home_rotation = d.xmat[self.wrist].reshape(3, 3).copy()
        self.records, self.states, self.velocities, self.commands = [], [], [], []
        self.started = self.finished = False
        self.result = None
        self.refinement = None
        self.contact_refinement = None

    def _loads(self):
        m, d, mj = self.model, self.data, self.mj
        forces = np.zeros(len(self.padlookup))
        support = table = hand = 0.0
        contacts, distance = 0, 0.1
        for ci, co in enumerate(d.contact):
            gs = [int(co.geom1), int(co.geom2)]
            bs = [int(m.geom_bodyid[g]) for g in gs]
            if self.obj not in bs:
                continue
            other = gs[0] if bs[1] == self.obj else gs[1]
            force, _ = contact_wrench_on_body(m, d, ci, self.obj)
            cf = np.zeros(6)
            mj.mj_contactForce(m, d, ci, cf)
            if other in self.padlookup:
                forces[self.padlookup[other]] += max(0, cf[0])
            if other in self.hand["geoms"]:
                hand += max(0, cf[0])
            if other in self.support:
                support += force[2]
                contacts += 1
                distance = min(distance, co.dist)
            if other in self.table:
                table += force[2]
        return forces, support, table, hand, contacts, distance

    def _penetrations(self):
        m, d = self.model, self.data
        robot, support = [], []
        for co in d.contact:
            if co.dist >= 0:
                continue
            gs = [int(co.geom1), int(co.geom2)]
            bs = [int(m.geom_bodyid[g]) for g in gs]
            row = {
                "depth_m": float(-co.dist),
                "geoms": [m.geom(g).name for g in gs],
                "bodies": [m.body(b).name for b in bs],
            }
            if any(b in self.robot_bodies for b in bs) and not joint_neighbors(m, *bs):
                robot.append(row)
            if self.obj in bs and 0 in bs:
                support.append(row)
        # Preserve legacy tie ordering for witness/diagnostic comparison.
        return sorted(
            sorted(robot, key=lambda r: r["depth_m"], reverse=True) + support,
            key=lambda r: r["depth_m"],
            reverse=True,
        )

    def _placed(self, rows):
        c = self.config["task"]
        return bool(
            len(rows) >= c["minimum_window_samples"]
            and all(
                np.linalg.norm(np.array(r["object_com_m"])[:2] - c["bay_xy_m"])
                < c["bay_radius_m"]
                and r["object_axis"][2] < -np.cos(np.deg2rad(c["inverted_angle_deg"]))
                and r["support_upward_force_n"] >= c["minimum_support_force_n"]
                and r["hand_force_n"] < c["maximum_hand_force_n"]
                and r["object_speed_m_s"] < c["maximum_object_speed_m_s"]
                for r in rows
            )
        )

    def run(self) -> dict:
        if self.started:
            raise RuntimeError("Frozen reference can execute only once; no reset")
        self.started = True
        m, d, mj, c, a = self.model, self.data, self.mj, self.config, self.ref
        obj, wrist, idx, aq = self.obj, self.wrist, self.hand, self.aq
        pressure, release = self.pressure, self.release
        normals = np.asarray(c["normals_object"], float)
        goals = np.asarray(c["pressure_targets_n"], float)
        thresholds, schedule = c["quality"], c["schedule"]
        last_com, anchor, returned = d.xipos[obj].copy(), None, None
        monitor = DiagnosticVerdict(
            continue_quality_warnings=c["continue_quality_warnings"]
        )
        monitor.observe(0, warnings=c["initial_warnings"])
        started, stop = time.monotonic(), None
        steps = round(c["duration_s"] / c["timestep_s"])
        for step in range(steps + 1):
            t = float(d.time)
            if time.monotonic() - started > c["wall_deadline_s"]:
                stop = "wall_clock_deadline"
                break
            if step % self.feedback_steps == 0:
                k = step // self.feedback_steps
                if t != a["feedback_times_s"][k]:
                    raise RuntimeError("Physics and frozen reference clocks diverged")
                u = float(a["reference_times_s"][k])
                mj.mj_forward(m, d)
                forces, supported, tableup, handforce, support_contacts, distance = (
                    self._loads()
                )
                speed = float(np.linalg.norm(d.xipos[obj] - last_com) / c["feedback_s"])
                last_com = d.xipos[obj].copy()
                target_object, phase = a["object_targets_world"][k], str(a["phases"][k])
                qfinger = a["finger_targets_rad"][k].copy()
                if self.refinement is not None:
                    qfinger, goals, gain = self.refinement.command(u, qfinger)
                    pressure.settings["gain_m_per_n_s"] = gain
                if self.contact_refinement is not None:
                    try:
                        qfinger = self.contact_refinement.command(
                            a["finger_targets_rad"][k],
                            qfinger,
                            reference_time=u,
                            phase=phase,
                            forces_n=forces,
                            support_n=supported,
                            releasing=release.started_s is not None,
                        )
                    except ValueError:
                        stop = "contact_correction_rejected"
                        monitor.observe(t, hard_failures=[stop])
                        if not self.records or self.records[-1]["t"] != t:
                            self.records.append({"t": t, "correction_rejected": True})
                            self.states.append(d.qpos.copy())
                            self.velocities.append(d.qvel.copy())
                            self.commands.append(d.ctrl.copy())
                        break
                if (
                    u >= schedule["pressure_start_reference_s"]
                    and release.started_s is None
                ):
                    _, _, jac, _ = observe_frames(
                        m, d, c["pad_frames"], idx, jacobian=True
                    )
                    outward = normals @ d.xmat[obj].reshape(3, 3).T
                    if c["observed_normals"]:
                        observations = object_pad_observations(
                            m, d, obj, list(self.padlookup)
                        )
                        for i, frame in enumerate(c["pad_frames"]):
                            ob = observations[m.geom(frame["geom"]).id]
                            if (
                                ob["force_n"] > 0.1
                                and ob["outward_normal_world"] is not None
                            ):
                                outward[i] = ob["outward_normal_world"]
                                jp = np.zeros((3, m.nv))
                                mj.mj_jac(
                                    m,
                                    d,
                                    jp,
                                    None,
                                    ob["point_world_m"],
                                    m.body(frame["body"]).id,
                                )
                                jac[i] = jp[:, idx["dofs"]]
                    nj = np.einsum("ij,ijk->ik", outward, jac)
                    qfinger = pressure.update(
                        qfinger,
                        nj,
                        forces,
                        c["feedback_s"],
                        targets_n=goals,
                        tangent_tasks=None,
                    )
                if u >= schedule["release_reference_s"]:
                    released = release.command(
                        t,
                        c["feedback_s"],
                        qfinger,
                        np.asarray(c["open_fingers_rad"]),
                        d.qpos[idx["q"]],
                        table_contacts=support_contacts,
                        clearance_m=float(distance),
                        speed_m_s=speed,
                        table_load_fraction=supported / c["object_weight_n"],
                    )
                    if released is not None:
                        qfinger = released
                clock = u
                if u >= schedule["withdraw_reference_s"]:
                    clock = (
                        schedule["release_reference_s"]
                        if release.started_s is None
                        else min(
                            u,
                            schedule["withdraw_reference_s"]
                            + max(
                                0,
                                t - (release.started_s + schedule["withdraw_delay_s"]),
                            ),
                        )
                    )
                arm_target = np.array(
                    [
                        np.interp(clock, a["arm_times_s"], a["arm_q_rad"][:, i])
                        for i in range(len(aq))
                    ]
                )
                prev_clock = max(0, clock - (u - a["previous_reference_times_s"][k]))
                arm_velocity = (
                    arm_target
                    - np.array(
                        [
                            np.interp(
                                prev_clock, a["arm_times_s"], a["arm_q_rad"][:, i]
                            )
                            for i in range(len(aq))
                        ]
                    )
                ) / c["feedback_s"]
                target = d.qpos.copy()
                target[aq], target[idx["q"]] = arm_target, qfinger
                if t < self.pre.duration_s:
                    cmd, vel = self.pre.command(t)
                    target[self.boundary_ids], arm_velocity, phase = (
                        cmd,
                        vel[: len(aq)],
                        "home_approach",
                    )
                if t >= schedule["return_eligible_s"] - 1e-8:
                    stable = [
                        r
                        for r in self.records
                        if r["t"] >= t - c["task"]["stable_window_s"]
                    ]
                    if returned is None and return_admitted(
                        task_completed=self._placed(stable),
                        release_fraction=release.fraction,
                        hand_force_n=handforce,
                        support_fraction=supported / c["object_weight_n"],
                        object_speed_m_s=speed,
                    ):
                        returned = t
                    if returned is not None:
                        cmd, vel = self.post.command(t - returned)
                        target[self.boundary_ids], arm_velocity = cmd, vel[: len(aq)]
                        phase = (
                            "return_home"
                            if t - returned < self.post.duration_s - c["home_settle_s"]
                            else "home_settle"
                        )
                    else:
                        phase = "return_gate_wait"
                arm_target = target[aq].copy()
                for ai, j in enumerate(m.actuator_trnid[:, 0]):
                    qid, dof = m.jnt_qposadr[j], m.jnt_dofadr[j]
                    gain, kv = m.actuator_gainprm[ai, 0], -m.actuator_biasprm[ai, 2]
                    ff = (
                        arm_velocity[self.arm_ids.index(j)]
                        if j in self.arm_ids
                        else 0.0
                    )
                    value = target[qid] + (d.qfrc_bias[dof] + kv * ff) / gain
                    d.ctrl[ai] = (
                        np.clip(value, *m.actuator_ctrlrange[ai])
                        if m.actuator_ctrllimited[ai]
                        else value
                    )
                if u >= schedule["carry_reference_s"] and anchor is None:
                    anchor = d.xmat[wrist].reshape(3, 3).T @ (
                        d.xipos[obj] - d.xpos[wrist]
                    )
            if step % self.sample_steps == 0:
                mj.mj_fwdPosition(m, d)
                contacts = self._penetrations()
                physical = max([r["depth_m"] for r in contacts], default=0.0)
                exclusion, witness = 0.0, None
                for name in c["exclusion_geoms"]:
                    for g in idx["geoms"]:
                        distance = mj.mj_geomDistance(
                            m, d, m.geom(name).id, int(g), 0.10, np.zeros(6)
                        )
                        if -distance > exclusion:
                            exclusion, witness = (
                                -float(distance),
                                [name, m.geom(int(g)).name],
                            )
                pos = d.xipos[obj].copy()
                axis = d.xmat[obj].reshape(3, 3) @ np.array([0, 0, 1])
                _, actualn = observe_frames(m, d, c["pad_frames"], idx)
                desired = -normals @ d.xmat[obj].reshape(3, 3).T
                normal_error = float(
                    np.rad2deg(
                        np.arccos(np.clip((actualn * desired).sum(1), -1, 1))
                    ).max()
                )
                drift = (
                    0.0
                    if anchor is None
                    else float(
                        np.linalg.norm(
                            d.xmat[wrist].reshape(3, 3).T @ (pos - d.xpos[wrist])
                            - anchor
                        )
                    )
                )
                warnings, hard = [], []
                if exclusion > thresholds["exclusion_m"]:
                    warnings.append("uncertainty_envelope_overlap")
                if (
                    schedule["carry_reference_s"] <= u < schedule["release_reference_s"]
                    and drift > thresholds["drift_m"]
                ):
                    warnings.append("carry_relative_drift")
                if (
                    schedule["orientation_check_reference_s"]
                    < u
                    < schedule["release_reference_s"]
                    and normal_error > thresholds["normal_deg"]
                ):
                    warnings.append("pad_orientation")
                if (
                    schedule["carry_reference_s"] <= u < schedule["release_reference_s"]
                    and min(forces) < thresholds["minimum_pad_force_n"]
                ):
                    warnings.append("pad_contact_loss")
                if physical > thresholds["physical_penetration_m"]:
                    hard.append("physical_penetration")
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                    hard.append("nonfinite_state")
                if pos[2] < thresholds["minimum_object_height_m"]:
                    hard.append("object_below_scene")
                should_stop = monitor.observe(t, warnings=warnings, hard_failures=hard)
                row = {
                    "object_rotation_error_deg": float(
                        np.rad2deg(
                            Rotation.from_matrix(
                                target_object[:3, :3].T @ d.xmat[obj].reshape(3, 3)
                            ).magnitude()
                        )
                    ),
                    "object_position_error_m": float(
                        np.linalg.norm(
                            pos
                            - (
                                target_object[:3, :3]
                                @ np.asarray(c["object_reference_com_m"])
                                + target_object[:3, 3]
                            )
                        )
                    ),
                    "t": t,
                    "reference_time_s": u,
                    "pressure_offset_rad": pressure.offset.tolist(),
                    "pressure_targets_n": goals.tolist(),
                    "anchor": None,
                    "phase": "await_support"
                    if u >= schedule["release_reference_s"]
                    and release.started_s is None
                    else phase,
                    "object_com_m": pos.tolist(),
                    "object_axis": axis.tolist(),
                    "object_speed_m_s": speed,
                    "pad_forces_n": forces.tolist(),
                    "support_upward_force_n": float(supported),
                    "table_upward_force_n": float(tableup),
                    "hand_force_n": handforce,
                    "physical_penetration_m": physical,
                    "physical_witness": contacts[0] if contacts else None,
                    "exclusion_overlap_m": exclusion,
                    "exclusion_witness": witness,
                    "carry_drift_m": drift,
                    "pad_normal_error_deg": normal_error,
                    "release": release.diagnostics(),
                    "arm_error_rad": float(max(abs(d.qpos[aq] - arm_target))),
                    "max_actuator_torque_nm": float(max(abs(d.actuator_force))),
                }
                self.records.append(row)
                self.states.append(d.qpos.copy())
                self.velocities.append(d.qvel.copy())
                self.commands.append(d.ctrl.copy())
                if should_stop:
                    stop = hard[0] if hard else "strict_warning"
                    break
            if step == steps:
                break
            before = float(d.time)
            mj.mj_step(m, d)
            health = step_rejection(
                before,
                float(d.time),
                c["timestep_s"],
                {str(i): int(w.number) for i, w in enumerate(d.warning)},
            )
            if health:
                monitor.observe(float(d.time), hard_failures=[health])
                stop = health
                break
        complete = bool(float(d.time) >= c["duration_s"] - 1e-6 and stop is None)
        window = [
            r
            for r in self.records
            if r["t"] >= c["duration_s"] - c["task"]["stable_window_s"]
        ]
        task = bool(complete and release.fraction >= 0.99 and self._placed(window))
        boundary = {
            "return_started_s": returned,
            "home_arm_error_rad": float(np.max(np.abs(d.qpos[aq] - self.home[aq]))),
            "home_finger_error_rad": float(
                np.max(np.abs(d.qpos[idx["q"]] - self.home[idx["q"]]))
            ),
            "home_wrist_error_m": float(
                np.linalg.norm(d.xpos[wrist] - self.home_wrist)
            ),
            "home_wrist_angle_deg": float(
                np.rad2deg(
                    Rotation.from_matrix(
                        self.home_rotation.T @ d.xmat[wrist].reshape(3, 3)
                    ).magnitude()
                )
            ),
        }
        limits = c["home_limits"]
        boundary["passed"] = bool(
            returned is not None
            and boundary["home_arm_error_rad"] < limits["arm_rad"]
            and boundary["home_finger_error_rad"] < limits["finger_rad"]
            and boundary["home_wrist_error_m"] < limits["wrist_m"]
            and boundary["home_wrist_angle_deg"] < limits["wrist_deg"]
        )
        self.result = dict(
            **monitor.finish(complete_horizon=complete, task_completed=task),
            stop_reason=stop,
            final_time_s=float(d.time),
            release=release.diagnostics(),
            boundary=boundary,
        )
        self.finished = True
        return self.result.copy()

    def save(self, output):
        output = Path(output)
        output.mkdir(exist_ok=False)
        if not self.finished:
            raise RuntimeError("Reference sequence did not complete evaluation")
        np.savez_compressed(
            output / "states.npz",
            qpos=self.states,
            qvel=self.velocities,
            commands=self.commands,
            timestamps_s=[r["t"] for r in self.records],
        )
        result = self.result | {
            "records": self.records,
            "states_sha256": digest(output / "states.npz"),
            "scene_sha256": digest(self.config["scene"]),
        }
        write_json(output / "result.json", result)
        return result
