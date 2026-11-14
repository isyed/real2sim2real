"""Articulated-hand adaptations of grasp, carry and placement skill concepts.

All motion delegates to the existing ContinuousPolicy actuator loop. Planning
uses scratch state. No reset, weld, object teleport or implicit dwell is added.
Upstream source references remain separate from these project implementations.
"""

from copy import deepcopy

import numpy as np

DEFAULT_CHECKS = {
    "pad_error_m": 0.005,
    "normal_error_deg": 25.0,
    "minimum_pad_force_n": 0.05,
    "minimum_contact_pads": 2,
    "maximum_support_fraction": 0.1,
    "maximum_grasp_translation_m": 0.01,
    "maximum_grasp_rotation_deg": 10.0,
    "minimum_hold_s": 0.1,
}


def pose(observation, prefix):
    result = np.eye(4)
    result[:3, :3] = observation[prefix + "_rotation"]
    result[:3, 3] = observation[prefix + "_position_m"]
    return result


def relative_pose(observation):
    return np.linalg.inv(pose(observation, "wrist")) @ pose(observation, "object")


def relative_error(reference, current):
    translation = float(np.linalg.norm(reference[:3, 3] - current[:3, 3]))
    rotation = reference[:3, :3].T @ current[:3, :3]
    angle = float(np.rad2deg(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
    return translation, angle


class ManipulationSkills:
    """Trusted adapter; diagnostic checks are host-configured, not author inputs."""

    def __init__(self, session):
        self.session = session
        supplied = session.config.get("skill_checks", {})
        if set(supplied) - set(DEFAULT_CHECKS):
            raise ValueError("Unknown skill diagnostic check")
        self.checks = DEFAULT_CHECKS | supplied
        if any(
            type(v) not in (int, float) or not np.isfinite(v) or v <= 0
            for v in self.checks.values()
        ):
            raise ValueError("Skill checks must be finite positive values")
        if type(self.checks["minimum_contact_pads"]) is not int:
            raise ValueError("minimum_contact_pads must be an integer")
        self.anchor = None
        self.plans = {}
        self.events = []
        self.transport_mode = session.config.get(
            "transport_screen_mode", "stationary_and_carried"
        )
        if self.transport_mode not in {
            "stationary_and_carried",
            "contact_aware",
            "contact_aware_preview",
        }:
            raise ValueError("Unknown transport screen mode")
        self.transport_previews = 0

    def describe_skills(self) -> dict:
        """Return frozen diagnostic checks, coordinate convention and limitations."""
        return {
            "checks": deepcopy(self.checks),
            "poses": "4x4 right-handed world transforms, translation in metres",
            "contact_inputs": "world pad points and inward normals; explicit full native joint maps in radians",
            "observations": "privileged simulator state",
            "scope": "Single articulated hand/arm per configured continuous backend. Diagnostic checks do not replace campaign acceptance.",
            "planning": "Sampled full-robot screening plus predicted carried-object screening; not a continuous collision certificate or a force-closure proof.",
            "timing": "Only explicit motion/release/hold durations advance physics. Each path segment uses the existing cubic interpolation.",
            "transport_screen_mode": self.transport_mode,
        }

    def observe_environment(self, surface_geom: str) -> dict:
        """Read whole-object clearance and fixed-environment forces without live stepping."""
        from real2sim2real.controller_actor_training.environment_observation import observe_environment

        return self._record("observe_environment", observe_environment(self.session, surface_geom))

    def _record(self, name, result):
        self.events.append({"tool": name, "result": deepcopy(result)})
        return result

    def fit_grasp(
        self,
        wrist_pose_world: list,
        seed_joints_rad: dict,
        pad_indices: list,
        targets_world_m: list,
        inward_normals_world: list,
    ) -> dict:
        """Fit wrist IK and opposing pads in scratch state; does not acquire a grasp."""
        targets = np.asarray(targets_world_m, float)
        normals = np.asarray(inward_normals_world, float)
        if (
            targets.shape != (len(pad_indices), 3)
            or normals.shape != targets.shape
            or not np.isfinite(targets).all()
            or not np.isfinite(normals).all()
            or (np.linalg.norm(normals, axis=1) <= 1e-9).any()
        ):
            raise ValueError("Finite pad targets and nonzero normals required")
        ik = self.session.plan_wrist(wrist_pose_world, seed_joints_rad)
        if not ik["ik_pass"]:
            return {"fit_pass": False, "ik": ik, "reason": "wrist IK rejected"}
        fit = self.session.fit_hand(
            ik["joints_rad"], pad_indices, targets_world_m, inward_normals_world
        )
        errors = np.asarray(fit["pad_errors_m"], float)
        normals = np.asarray(fit["normal_errors_deg"], float)
        passed = bool(
            len(errors) == len(pad_indices) > 0
            and len(normals) == len(errors)
            and np.isfinite(errors).all()
            and np.isfinite(normals).all()
            and (errors <= self.checks["pad_error_m"]).all()
            and (normals <= self.checks["normal_error_deg"]).all()
        )
        return {
            "fit_pass": passed,
            "ik": ik,
            "fit": fit,
            "joints_rad": fit["joints_rad"],
            "path_admission_required": True,
            "loaded_hold_verified": False,
        }

    def acquire_grasp(
        self, positions: list, durations_s: list, pressure_targets_n: list
    ) -> dict:
        """Execute an explicit opening/insertion/closure path, then enable pressure.

        No dwell or lift is inserted. Pressure takes effect on the next explicit
        motion/hold; a completed closure is not evidence of a held object.
        """
        self.session.prepare_pressure(pressure_targets_n)
        screen = self.session.screen_path(positions, durations_s)
        if not screen["admitted"]:
            return {"executed": False, "screen": screen}
        self.anchor = None
        result = self.session.execute_joint_path(positions, durations_s)
        if result["executed"]:
            self.session.set_pressure(pressure_targets_n)
            self.anchor = relative_pose(self.session.observe())
        return self._record("acquire_grasp", result | {"loaded_hold_verified": False})

    def _snapshot(self):
        s = self.session
        return {
            "qpos": s.data.qpos.copy(),
            "qvel": s.data.qvel.copy(),
            "nominal": s.nominal.copy(),
            "tick": s.tick,
            "pressure": None
            if s.pressure_targets is None
            else s.pressure_targets.copy(),
        }

    def _unchanged(self, before):
        now = self._snapshot()
        return all(np.array_equal(before[k], now[k]) for k in now)

    def _carried_screen(self, positions, durations_s, relative):
        """Predict a rigidly carried object only in scratch kinematic screening."""
        s = self.session
        m, mj = s.model, s.mj
        jid = int(m.body_jntadr[s.object])
        if (
            jid < 0
            or m.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE
            or m.body_parentid[s.object] != 0
        ):
            raise ValueError("Carried screen requires a world-parent free object")
        address = int(m.jnt_qposadr[jid])
        scratch = mj.MjData(m)
        scratch.qpos[:] = s.data.qpos
        start = s.nominal.copy()
        peak, witness = 0.0, None
        for mapping, duration in zip(positions, durations_s):
            goal = s._vector(mapping)
            for u in np.linspace(0, 1, round(duration / s.feedback) + 1):
                scratch.qpos[s.qids] = start + (goal - start) * u * u * (3 - 2 * u)
                mj.mj_forward(m, scratch)
                wrist = np.eye(4)
                wrist[:3, 3] = scratch.xpos[s.wrist]
                wrist[:3, :3] = scratch.xmat[s.wrist].reshape(3, 3)
                obj = wrist @ relative
                scratch.qpos[address : address + 3] = obj[:3, 3]
                mj.mju_mat2Quat(
                    scratch.qpos[address + 3 : address + 7], obj[:3, :3].flatten()
                )
                mj.mj_forward(m, scratch)
                _, _, _, penetration, pair = s._contacts(scratch)
                # Include object versus fixed obstacle bodies, not only world geoms.
                for contact in scratch.contact:
                    gs = [int(contact.geom1), int(contact.geom2)]
                    bs = [int(m.geom_bodyid[g]) for g in gs]
                    if s.object in bs and -contact.dist > penetration:
                        penetration = float(-contact.dist)
                        pair = [m.geom(g).name for g in gs]
                if penetration > peak:
                    peak, witness = penetration, pair
            start = goal
        return {
            "admitted": peak <= s.config["penetration_limit_m"],
            "peak_penetration_m": peak,
            "witness": witness,
            "sample_s": s.feedback,
            "continuous_collision_certificate": False,
            "assumption": "Measured object-to-wrist transform held rigidly for screening only; live object remains free.",
        }

    def plan_transport(self, object_poses_world: list, durations_s: list) -> dict:
        """Plan from desired object poses using the measured wrist/object offset.

        Caller chooses the lift, inversion and placement waypoints. This is seeded
        IK and sampled screening, not a global obstacle-avoidance planner.
        """
        from real2sim2real.controller_actor_training.arm_object_feedback import (
            rigid_pose,
        )

        if not 1 <= len(object_poses_world) <= 100 or len(object_poses_world) != len(
            durations_s
        ):
            raise ValueError("One to 100 poses with matching durations required")
        observation = self.session.observe()
        relative = relative_pose(observation)
        seed = observation["commanded_joints_rad"]
        positions, fits = [], []
        for value in object_poses_world:
            target = rigid_pose(np.asarray(value, float))
            wrist = target @ np.linalg.inv(relative)
            fit = self.session.plan_wrist(wrist.tolist(), seed)
            fits.append(fit)
            if not fit["ik_pass"]:
                return {"admitted": False, "reason": "wrist IK rejected", "fits": fits}
            seed = fit["joints_rad"]
            positions.append(seed)
        screen = self.session.screen_path(positions, durations_s)
        carried = self._carried_screen(positions, durations_s, relative)
        contact_screen = None
        if self.transport_mode in {"contact_aware", "contact_aware_preview"}:
            from .contact_transport import screen_transport

            contact_screen = screen_transport(self, positions, durations_s, observation)
        result = {
            "admitted": bool(
                contact_screen["admitted"]
                if contact_screen is not None
                else screen["admitted"] and carried["admitted"]
            ),
            "screen": screen,
            "carried_object_screen": carried,
            "fits": fits,
            "positions": positions,
            "durations_s": durations_s,
            "loaded_retention_verified": False,
        }
        if contact_screen is not None:
            result["contact_aware_screen"] = contact_screen
        if result["admitted"]:
            key = f"transport-{len(self.plans)}"
            self.plans[key] = {
                "snapshot": self._snapshot(),
                "positions": deepcopy(positions),
                "durations_s": deepcopy(durations_s),
                "relative_pose": relative.copy(),
            }
            result["plan_id"] = key
        return result

    def execute_transport(self, plan_id: str) -> dict:
        """Execute a host-retained plan only if its planning state is unchanged."""
        if plan_id not in self.plans:
            raise ValueError("Unknown transport plan")
        plan = self.plans[plan_id]
        if not self._unchanged(plan["snapshot"]):
            return {"executed": False, "reason": "stale plan; observe and replan"}
        if self.transport_mode in {"contact_aware", "contact_aware_preview"}:
            from .contact_transport import retention_guard, screen_transport

            # Recheck admission in the current state; no caller-provided token
            # can bypass a screen or the actuator loop's command validation.
            screen = screen_transport(
                self, plan["positions"], plan["durations_s"], self.session.observe()
            )
            if not screen["admitted"]:
                return {"executed": False, "screen": screen}
            guard = retention_guard(self, plan["relative_pose"])
            result = self.session._execute_screened_path(
                plan["positions"], plan["durations_s"], feedback_guard=guard
            )
            if screen.get("dynamic_preview", {}).get("passed"):
                forecast = screen["dynamic_preview"]["final_state"]
                result["preview_reproduced_exactly"] = (
                    all(
                        np.array_equal(
                            getattr(self.session.data, field), forecast[field]
                        )
                        for field in ["qpos", "qvel", "ctrl"]
                    )
                    and self.session.data.time == forecast["time_s"]
                )
            failure = guard(result["observation"])
            if failure:
                self.session.fault = failure
                raise RuntimeError(failure)
        else:
            result = self.session.execute_joint_path(
                plan["positions"], plan["durations_s"]
            )
        return self._record("execute_transport", result)

    def _records(self, start, end):
        # Include exact endpoints even when they lie between logging samples.
        return (
            [start]
            + [
                r
                for r in self.session.history
                if start["time_s"] < r["time_s"] < end["time_s"]
            ]
            + [end]
        )

    def verify_grasp(self, duration_s: float) -> dict:
        """Run an explicit loaded hold and check contact, support and relative drift."""
        if self.anchor is None:
            return {"held": False, "reason": "no acquisition reference"}
        if not np.isfinite(duration_s) or duration_s < self.checks["minimum_hold_s"]:
            raise ValueError("Hold duration below frozen diagnostic minimum")
        start = self.session.observe()
        if self.transport_mode in {"contact_aware", "contact_aware_preview"}:
            from .contact_transport import retention_guard, screen_transport

            s = self.session
            positions = [dict(zip(s.names, s.nominal.tolist()))]
            screen = screen_transport(self, positions, [duration_s], start)
            if not screen["admitted"]:
                result = {"executed": False, "screen": screen}
            else:
                guard = retention_guard(self, relative_pose(start))
                result = s._execute_screened_path(
                    positions, [duration_s], feedback_guard=guard
                )
                failure = guard(result["observation"])
                if failure:
                    s.fault = failure
                    raise RuntimeError(failure)
                if screen.get("dynamic_preview", {}).get("passed"):
                    expected = screen["dynamic_preview"]["final_state"]
                    result["preview_reproduced_exactly"] = (
                        all(
                            np.array_equal(getattr(s.data, key), expected[key])
                            for key in ["qpos", "qvel", "ctrl"]
                        )
                        and s.data.time == expected["time_s"]
                    )
                self._record("execute_loaded_hold", result)
        else:
            result = self.session.hold(duration_s)
        if not result["executed"]:
            return {"held": False, "hold": result}
        rows = self._records(start, self.session.observe())
        drift = [relative_error(self.anchor, relative_pose(r)) for r in rows]
        contacts = [
            sum(f >= self.checks["minimum_pad_force_n"] for f in r["pad_forces_n"])
            for r in rows
        ]
        metrics = {
            "minimum_contact_pads": min(contacts),
            "maximum_support_fraction": max(r["support_fraction"] for r in rows),
            "translation_drift_m": max(t for t, _ in drift),
            "rotation_drift_deg": max(a for _, a in drift),
            "maximum_sample_gap_s": float(np.max(np.diff([r["time_s"] for r in rows]))),
        }
        checks = {
            "contact_count": metrics["minimum_contact_pads"]
            >= self.checks["minimum_contact_pads"],
            "off_support": metrics["maximum_support_fraction"]
            <= self.checks["maximum_support_fraction"],
            "translation": metrics["translation_drift_m"]
            <= self.checks["maximum_grasp_translation_m"],
            "rotation": metrics["rotation_drift_deg"]
            <= self.checks["maximum_grasp_rotation_deg"],
            "sample_coverage": metrics["maximum_sample_gap_s"]
            <= self.session.config["sample_s"] + 1e-8,
            "healthy": not self.session.fault,
        }
        return self._record(
            "verify_grasp",
            {
                "held": all(checks.values()),
                "checks": checks,
                "metrics": metrics,
                "start_s": start["time_s"],
                "end_s": rows[-1]["time_s"],
                "scope": "Sampled contact retention; contact count alone does not prove force closure or opposed normals.",
            },
        )

    def place_and_release(
        self,
        positions: list,
        durations_s: list,
        open_joints_rad: dict,
        release_duration_s: float,
    ) -> dict:
        """Approach via explicit joints; release only at the supported task placement.

        No dwell is invented. If support dwell is missing, return that reason so
        the policy can explicitly wait or revise the approach within its budget.
        """
        from real2sim2real.controller_actor_training.placement_outcome import (
            placement_state_checks,
        )

        s = self.session
        s._vector(open_joints_rad)
        if (
            not np.isfinite(release_duration_s)
            or release_duration_s <= 0
            or not np.isclose(
                release_duration_s / s.feedback,
                round(release_duration_s / s.feedback),
                rtol=0,
                atol=1e-8,
            )
            or s.data.time + sum(durations_s) + release_duration_s
            > s.config["horizon_s"] + 1e-9
        ):
            raise ValueError(
                "Positive tick-aligned release within the remaining horizon required"
            )
        # Require the task fields before executing any approach.
        placement_state_checks(s.observe(), s.config["task"])
        result = s.execute_joint_path(positions, durations_s)
        if not result["executed"]:
            return {"released": False, "approach": result}
        checks = placement_state_checks(s.observe(), s.config["task"])
        if not all(value for name, value in checks.items() if name != "released"):
            return self._record(
                "place_and_release",
                {
                    "released": False,
                    "reason": "task placement not supported/stationary/aligned",
                    "checks": checks,
                },
            )
        result = s.release(open_joints_rad, release_duration_s)
        if result.get("released"):
            self.anchor = None
        return self._record("place_and_release", result)

    def verify_placement(self) -> dict:
        """Check the recorded stable placement window; does not advance time."""
        from real2sim2real.controller_actor_training.placement_outcome import (
            placement_state_checks,
        )

        s = self.session
        end = s.observe()
        task = s.config["task"]
        start_time = end["time_s"] - task["stable_window_s"]
        rows = [
            r for r in s.history if start_time - 1e-8 <= r["time_s"] < end["time_s"]
        ] + [end]
        maximum_gap = task.get("maximum_sample_gap_s", 0.020001)
        covered = (
            start_time >= -1e-8
            and rows[0]["time_s"] <= start_time + maximum_gap
            and len(rows) >= task.get("minimum_window_samples", 20)
            and np.all(np.diff([r["time_s"] for r in rows]) <= maximum_gap)
        )
        states = [placement_state_checks(r, task) for r in rows]
        checks = {name: all(r[name] for r in states) for name in states[0]}
        return self._record(
            "verify_placement",
            {
                "placed": bool(covered and all(checks.values()) and not s.fault),
                "window_covered": bool(covered),
                "checks": checks,
                "scope": "Placement only; final home/horizon, strict quality and control admission remain independent.",
            },
        )
