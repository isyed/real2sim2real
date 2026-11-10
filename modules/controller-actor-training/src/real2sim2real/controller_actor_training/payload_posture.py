"""Bounded finger-posture transitions from observed supported-load estimates.

The policy never reads simulator mass or a case label. Its fixture adapter pins
an admitted endpoint and preserves wrists, object targets and physical limits.
"""

import json
from pathlib import Path

import numpy as np
from real2sim2real.artifacts import digest


def validate_settings(settings):
    if set(settings) != {
        "target_result",
        "target_sha256",
        "activation_weight_ratio",
        "full_weight_ratio",
        "maximum_joint_change_rad",
        "maximum_joint_speed_rad_s",
        "maximum_fraction_speed_s_inv",
    }:
        raise ValueError(
            "Explicit pinned posture endpoint and bounded load policy required"
        )
    values = [
        settings[k]
        for k in (
            "activation_weight_ratio",
            "full_weight_ratio",
            "maximum_joint_change_rad",
            "maximum_joint_speed_rad_s",
            "maximum_fraction_speed_s_inv",
        )
    ]
    if (
        not np.isfinite(values).all()
        or not 0.5
        <= settings["full_weight_ratio"]
        < settings["activation_weight_ratio"]
        <= 1
        or not 0 < settings["maximum_joint_change_rad"] <= 0.35
        or not 0 < settings["maximum_joint_speed_rad_s"] <= 0.5
        or not 0 < settings["maximum_fraction_speed_s_inv"] <= 2
    ):
        raise ValueError("Posture transition exceeds finite ratio/travel/speed bounds")


class BoundedPostureFraction:
    def __init__(self, settings, maximum_joint_delta_rad):
        validate_settings(settings)
        if (
            not np.isfinite(maximum_joint_delta_rad)
            or not 0
            <= maximum_joint_delta_rad
            <= settings["maximum_joint_change_rad"] + 1e-8
        ):
            raise ValueError("Endpoint exceeds the declared posture budget")
        self.settings = settings
        self.rate = min(
            settings["maximum_fraction_speed_s_inv"],
            settings["maximum_joint_speed_rad_s"] / max(maximum_joint_delta_rad, 1e-12),
        )
        self.fraction = 0.0
        self.desired = 0.0
        self.updated = self.frozen = 0

    def update(self, weight_ratio, dt, *, active):
        if not np.isfinite([weight_ratio, dt]).all() or weight_ratio <= 0 or dt <= 0:
            raise ValueError("Positive supported-weight ratio and clock required")
        s = self.settings
        self.desired = float(
            np.clip(
                (s["activation_weight_ratio"] - weight_ratio)
                / (s["activation_weight_ratio"] - s["full_weight_ratio"]),
                0,
                1,
            )
        )
        if active:
            self.fraction = float(
                np.clip(
                    self.fraction
                    + np.clip(
                        self.desired - self.fraction, -dt * self.rate, dt * self.rate
                    ),
                    0,
                    1,
                )
            )
            self.updated += 1
        else:
            self.frozen += 1
        return self.fraction


class PayloadPosture:
    def __init__(
        self,
        model,
        data,
        settings,
        candidate,
        pads,
        *,
        scene_sha256,
        allowed_penetration_m,
    ):
        import trimesh
        from real2sim2real.retargeting.grasp_kinematics import robot_indices
        from real2sim2real.retargeting.pad_contacts import collision_pad_hull
        from .pad_grasp_planning import path_audit
        from .retention_experiment import assess

        validate_settings(settings)
        path = Path(settings["target_result"])
        if digest(path) != settings["target_sha256"]:
            raise ValueError("Pinned target posture changed")
        target = json.loads(path.read_text())
        protocol = json.loads((path.parent.parent / "protocol.json").read_text())
        if (
            target["success"] is not True
            or target["candidate"]["accepted"] is not True
            or not np.isfinite(
                [
                    target["simulation_final_time_s"],
                    target["simulation_requested_duration_s"],
                ]
            ).all()
            or target["simulation_requested_duration_s"] <= 0
            or target["simulation_final_time_s"]
            < target["simulation_requested_duration_s"] - 1e-6
            or target["mass_scale"] != 1
            or target["friction"] != target["configuration"]["physics"]["friction"]
            or digest(path.with_name("states.npz")) != target["states_sha256"]
            or scene_sha256 is None
            or target["source_scene_sha256"] != scene_sha256
            or allowed_penetration_m
            != target["configuration"]["planning"]["max_penetration_m"]
            or not assess(target, protocol["fidelity_limits"])["combined_pass"]
        ):
            raise ValueError(
                "Complete admitted nominal posture, same scene and gates required"
            )
        self.settings = settings
        self.target = target["candidate"]
        self.owned = np.array(sorted({int(q) for p in pads for q in p["qids"]}), int)
        limits = {int(q): lim for p in pads for q, lim in zip(p["qids"], p["limits"])}
        self.limits = np.array([limits[int(q)] for q in self.owned])
        changes = {}
        ids = []
        before = []
        after = []
        frame_map = {}
        for source, destination in [
            (candidate, self.target),
            (candidate["support"], self.target["support"]),
        ]:
            if (
                source["joint_names"] != destination["joint_names"]
                or not np.allclose(
                    source["base_position_m"],
                    destination["base_position_m"],
                    atol=1e-6,
                    rtol=0,
                )
                or not np.allclose(
                    source["base_rotation"],
                    destination["base_rotation"],
                    atol=1e-6,
                    rtol=0,
                )
                or not np.allclose(
                    source["targets_m"], destination["targets_m"], atol=1e-8, rtol=0
                )
            ):
                raise ValueError(
                    "Posture endpoints must share fixed wrists and immutable object targets"
                )
            root = int(model.jnt_bodyid[model.joint(source["joint_names"][0]).id])
            if model.body_parentid[root] != 0 or model.body_jntnum[root] != 6:
                raise ValueError(
                    "Current posture adapter requires Cartesian wrist fixtures"
                )
            index = robot_indices(model, model.body(root).name)
            if [model.joint(int(j)).name for j in index["joints"]] != source[
                "joint_names"
            ]:
                raise ValueError("Posture joint ordering differs from the source hand")
            old = np.asarray(source["q_closed"], float)
            new = old.copy()
            for i, n in enumerate(source["joint_names"]):
                q = int(model.joint(n).qposadr[0])
                delta = destination["q_closed"][i] - old[i]
                if q in self.owned:
                    changes[q] = delta
                    new[i] += delta
                elif abs(delta) > 1e-6:
                    raise ValueError("Posture may only change owned fingers")
            ids.append(index)
            before.append(old)
            after.append(new)
            frames = {f["geom"]: f for f in destination["pad_plan"]["frames"]}
            for f in source["pad_plan"]["frames"]:
                if f["geom"] not in frames or frames[f["geom"]]["body"] != f["body"]:
                    raise ValueError("Target posture changes collision-pad ownership")
                frame_map[f["geom"]] = (f, frames[f["geom"]])
        self.delta = np.array([changes[int(q)] for q in self.owned])
        self.policy = BoundedPostureFraction(
            settings, float(np.max(abs(self.delta), initial=0))
        )
        endpoint = np.concatenate(after)
        combined = {
            k: np.concatenate([i[k] for i in ids])
            for k in ("joints", "q", "dofs", "geoms")
        }
        self.path_audit = path_audit(
            model, data, combined, [np.concatenate(before), endpoint]
        )
        if (
            not self.path_audit["within_joint_limits"]
            or self.path_audit["max_penetration_m"] > allowed_penetration_m
        ):
            raise ValueError(
                "Intermediate posture path failed native-limit/collision admission"
            )
        final_by_q = {int(q): v for q, v in zip(combined["q"], endpoint)}
        if any(
            not limits[int(q)][0] - 1e-8
            <= final_by_q[int(q)]
            <= limits[int(q)][1] + 1e-8
            for q in self.owned
        ):
            raise ValueError(
                "Posture endpoint exceeds existing actuator control bounds"
            )
        self.points = {}
        self.grid = np.linspace(0, 1, 33)
        self.maximum_projection_m = 0.0
        for p in pads:
            old, new = frame_map[model.geom(p["gid"]).name]
            a, b = np.asarray(old["point_body_m"]), np.asarray(new["point_body_m"])
            if np.array_equal(a, b):
                continue
            hull = collision_pad_hull(model, p["gid"])
            near, gap, _ = trimesh.proximity.closest_point(
                hull, a + (b - a) * self.grid[:, None]
            )
            self.maximum_projection_m = max(self.maximum_projection_m, float(gap.max()))
            if gap.max() > 0.001:
                raise ValueError("Intermediate pad-frame projection exceeds 1 mm")
            near[0] = a
            near[-1] = b
            self.points[p["gid"]] = near
        self.clipped_steps = 0

    def apply(self, nominal, pads, weight_ratio, dt, *, active):
        previous = self.policy.fraction
        fraction = self.policy.update(weight_ratio, dt, active=active)
        if fraction == 0:
            if previous > 0:
                for p in pads:
                    if p["gid"] in self.points:
                        p["point_body"] = self.points[p["gid"]][0].copy()
            return nominal
        output = np.asarray(nominal, float).copy()
        requested = output[self.owned] + fraction * self.delta
        output[self.owned] = np.clip(requested, *self.limits.T)
        self.clipped_steps += int(not np.array_equal(requested, output[self.owned]))
        for p in pads:
            if p["gid"] in self.points:
                points = self.points[p["gid"]]
                p["point_body"] = np.array(
                    [np.interp(fraction, self.grid, points[:, i]) for i in range(3)]
                )
        return output

    def diagnostics(self):
        return dict(
            fraction=self.policy.fraction,
            desired_fraction=self.policy.desired,
            maximum_fraction_speed_s_inv=self.policy.rate,
            maximum_joint_delta_rad=float(np.max(abs(self.delta), initial=0)),
            updated_steps=self.policy.updated,
            frozen_steps=self.policy.frozen,
            clipped_command_steps=self.clipped_steps,
            target_sha256=self.settings["target_sha256"],
            transition_path=self.path_audit,
            maximum_pad_projection_m=self.maximum_projection_m,
        )
