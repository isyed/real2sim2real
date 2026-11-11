"""Fit finger opening along bounded wrist-rotation/withdrawal hypotheses on CPU.

Uses exact hand collision geometry on Cartesian scratch fixtures matched to a
saved arm state. No dynamics run is claimed. Every accepted proposal still needs
fresh full-arm IK/collision admission and free-object release evaluation.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, write_json
from .wrist_withdrawal import validate_withdrawal, transform


def matching_geom(model, original, geom):
    """Match unnamed collision geoms by their body and compiled local geometry."""
    name = model.geom(geom).name
    if name:
        return original.geom(name).id
    body = original.body(model.body(model.geom_bodyid[geom]).name).id
    candidates = [
        g
        for g in range(original.ngeom)
        if original.geom_bodyid[g] == body
        and original.geom_type[g] == model.geom_type[geom]
        and original.geom_contype[g] == model.geom_contype[geom]
        and original.geom_conaffinity[g] == model.geom_conaffinity[geom]
        and np.allclose(original.geom_size[g], model.geom_size[geom], atol=1e-7, rtol=0)
        and np.allclose(original.geom_pos[g], model.geom_pos[geom], atol=1e-6, rtol=0)
    ]
    if len(candidates) != 1:
        raise ValueError("Collision geometry needs an unambiguous saved-body match")
    return candidates[0]


def contact_profile(model, data, primary, robot, objects, required_gap):
    """Fixed-size per-geometry residuals from position-only collision witnesses."""
    primary, robot = sorted(primary), sorted(robot)
    primary_set = set(primary)
    object_error = np.zeros(len(primary))
    robot_error = np.zeros(len(robot))
    a, b = {g: i for i, g in enumerate(primary)}, {g: i for i, g in enumerate(robot)}
    worst = None
    minimum = None
    for contact in data.contact:
        pair = {int(contact.geom1), int(contact.geom2)}
        distance = float(contact.dist)
        if pair & objects:
            for g in pair & a.keys():
                object_error[a[g]] = max(object_error[a[g]], required_gap - distance)
                minimum = distance if minimum is None else min(minimum, distance)
        else:
            for g in pair & b.keys():
                robot_error[b[g]] = max(robot_error[b[g]], -distance - 0.003)
        if pair & primary_set and (worst is None or distance < worst["distance_m"]):
            worst = dict(
                geoms=[model.geom(g).name for g in (contact.geom1, contact.geom2)],
                distance_m=distance,
                object_contact=bool(pair & objects),
            )
    return object_error, robot_error, minimum, worst


def bounded_rotation(parameters):
    """Project finite rotation parameters to a strict 0.2 rad outer bound."""
    values = np.asarray(parameters, float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Finite three-parameter wrist rotation required")
    return (0.2 - 1e-9) * values / max(1.0, float(np.linalg.norm(values)))


class FixturePath:
    def __init__(self, config):
        import mujoco
        from .object_grasp_pilot import load_scene, set_root_frame
        from .arm_finger_feedback import ArmFingerFeedback
        from .finger_release_plan import bind, read_posture
        from real2sim2real.retargeting.grasp_kinematics import robot_indices

        path = Path(config["source_result"])
        if digest(path) != config["source_sha256"]:
            raise ValueError("Pinned release source changed")
        self.source = json.loads(path.read_text())
        source = self.source
        if source.get("schema") != "real2sim2real.arm-replay-pilot/1" or not source.get(
            "continuous_protocol", {}
        ).get("release_planned"):
            raise ValueError("Continuous arm release source required")
        for name, key in [
            ("states.npz", "states_sha256"),
            ("scene.xml", "scene_sha256"),
        ]:
            if digest(path.with_name(name)) != source[key]:
                raise ValueError("Saved release artifacts changed")
        audit_path = Path(source["config"]["audit"])
        if digest(audit_path) != source["config"]["audit_sha256"]:
            raise ValueError("Arm audit changed")
        audit = json.loads(audit_path.read_text())
        for name, sha in audit["sources"].items():
            if digest(Path(name)) != sha:
                raise ValueError("Release planning source changed")
        parent = json.loads(Path(audit["config"]["trial"]).read_text())
        original = mujoco.MjModel.from_xml_path(str(path.with_name("scene.xml")))
        observed = mujoco.MjData(original)
        with np.load(path.with_name("states.npz"), allow_pickle=False) as states:
            index = int(np.argmin(abs(states["timestamps_s"] - config["time_s"])))
            observed.qpos[:] = states["qpos"][index]
        row = source["records"][index]
        protocol = source["continuous_protocol"]
        if (
            row["table_contacts"] <= 0
            or not protocol["capture_duration_s"]
            <= row["time_s"]
            < protocol["finger_release"]["start_s"]
        ):
            raise ValueError("Use a returned table-contact state before release")
        mujoco.mj_forward(original, observed)
        self.model, self.data, _ = load_scene(parent["configuration"])
        model, data = self.model, self.data
        roots = {
            a["name"]: original.body(a["root"]).id for a in audit["config"]["arms"]
        }
        motors = [
            a
            for a in range(original.nu)
            if not original.actuator(a).name.startswith("pilot__")
        ]
        # The live binder supplies exactly the existing owned finger identities.
        _, _, mesh = load_scene(parent["configuration"])
        live = ArmFingerFeedback(original, observed, parent, roots, mesh, motors)
        seed_spec = config["seed_posture"]
        seed_header = json.loads(Path(seed_spec["path"]).read_text())
        seed_withdrawal = seed_header.get("wrist_withdrawal")
        seed_release = seed_header.get("release_settings")
        read_posture(seed_spec, withdrawal=seed_withdrawal, release=seed_release)
        bind(
            live, original, seed_spec, withdrawal=seed_withdrawal, release=seed_release
        )
        self.release = deepcopy(protocol["finger_release"])
        if (
            self.release.get("thumb_duration_s", self.release["duration_s"])
            != self.release["duration_s"]
        ):
            raise ValueError("This path search requires a common finger opening clock")
        self.withdrawal = deepcopy(config["wrist_withdrawal"])
        validate_withdrawal(self.withdrawal, self.release)
        self.hands = []
        self.nominal = {}
        self.goal = {}
        for arm, role in [("right", "primary"), ("left", "secondary")]:
            root_name = original.body(roots[arm]).name
            idx = robot_indices(model, root_name)
            if model.body_parentid[idx["root"]] != 0:
                raise ValueError("World-parented hand fixture required")
            prefix = (
                "fixture_"
                if role == "primary"
                else parent["candidate"]["support"]["fixture_prefix"]
            )
            base = np.eye(4)
            base[:3, :3] = observed.xmat[roots[arm]].reshape(3, 3)
            base[:3, 3] = observed.xpos[roots[arm]]
            set_root_frame(model, data, root_name, prefix, base[:3, 3], base[:3, :3])
            fixture = [
                model.joint(prefix + s).qposadr[0]
                for s in ("x", "y", "z", "rx", "ry", "rz")
            ]
            live_hand = next(h for h in live.hands if h["role"] == role)
            names = []
            # Map owned scalar coordinates by joint name, independent of arm layout.
            start = []
            goal = []
            qids = []
            for q, opened, released in zip(
                live_hand["qids"],
                live_hand["waypoints"][1],
                live_hand.get("release_open", live_hand["waypoints"][1]),
            ):
                joint = next(
                    j for j in range(original.njnt) if original.jnt_qposadr[j] == q
                )
                name = original.joint(joint).name
                names.append(name)
                qids.append(int(model.joint(name).qposadr[0]))
                start.append(float(observed.qpos[q]))
                goal.append(float(released))
                self.nominal[name] = float(opened)
                self.goal[name] = float(released)
            self.hands.append(
                dict(
                    arm=arm,
                    role=role,
                    root=idx["root"],
                    geoms=set(map(int, idx["geoms"])),
                    fixture=np.array(fixture),
                    base=base,
                    names=names,
                    qids=np.array(qids),
                    start=np.array(start),
                    goal=np.array(goal),
                )
            )
        self.original_q = model.qpos0.copy()
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                other = original.joint(model.joint(j).name)
                self.original_q[model.jnt_qposadr[j] : model.jnt_qposadr[j] + 7] = (
                    observed.qpos[other.qposadr[0] : other.qposadr[0] + 7]
                )
        for hand in self.hands:
            self.original_q[hand["qids"]] = hand["start"]
        data.qpos[:] = self.original_q
        mujoco.mj_fwdPosition(model, data)
        # Establish that extracting the hand did not change its measured geometry.
        maximum_error = 0.0
        maximum_angle = 0.0
        for hand in self.hands:
            for g in hand["geoms"]:
                other = matching_geom(model, original, g)
                maximum_error = max(
                    maximum_error,
                    float(
                        np.linalg.norm(data.geom_xpos[g] - observed.geom_xpos[other])
                    ),
                )
                angle = Rotation.from_matrix(
                    data.geom_xmat[g].reshape(3, 3)
                    @ observed.geom_xmat[other].reshape(3, 3).T
                ).magnitude()
                maximum_angle = max(maximum_angle, float(angle))
        if maximum_error > 2e-6 or maximum_angle > 2e-6:
            raise ValueError(
                "Scratch hand geometry does not reproduce saved arm geometry"
            )
        self.source_time = float(row["time_s"])
        self.geometry_alignment_error_m = maximum_error
        self.geometry_alignment_error_rad = maximum_angle
        self.primary = next(h for h in self.hands if h["role"] == "primary")
        self.secondary = next(h for h in self.hands if h["role"] == "secondary")
        role = config.get("role", "primary")
        if role not in ("primary", "secondary"):
            raise ValueError("Explicit primary or secondary hand role required")
        self.moving = next(h for h in self.hands if h["role"] == role)
        self.bilateral = bool(config.get("require_bilateral_clearance", False))
        self.robot = set().union(*(h["geoms"] for h in self.hands))
        obj = model.body("task_object").id
        self.objects = {g for g in range(model.ngeom) if model.geom_bodyid[g] == obj}
        self.table = model.geom("table").id
        self.names = config["joints"]
        if (
            not self.names
            or len(set(self.names)) != len(self.names)
            or not set(self.names) <= set(self.moving["names"])
        ):
            raise ValueError(
                "Distinct owned finger joints of the selected role required"
            )
        self.ids = np.array([model.joint(n).qposadr[0] for n in self.names])
        self.seed = np.array([self.goal[n] for n in self.names])
        native = np.array([model.joint(n).range for n in self.names])
        nominal = np.array([self.nominal[n] for n in self.names])
        self.lower = np.maximum(native[:, 0], nominal - 0.5)
        self.upper = np.minimum(native[:, 1], nominal + 0.5)
        # Position-only near-contact queries; these margins are never exported to
        # a dynamics model or used to infer forces.
        query_geoms = self.robot if self.bilateral else self.primary["geoms"]
        model.geom_margin[list(query_geoms)] = np.maximum(
            model.geom_margin[list(query_geoms)], config["clearance_m"]
        )
        self.clearance = config["clearance_m"]

    def sample(self, fraction, q, rotation):
        import mujoco
        from .object_grasp_pilot import smoothstep
        from .arm_clearance_audit import table_gap

        model, data = self.model, self.data
        data.qpos[:] = self.original_q
        withdrawal = deepcopy(self.withdrawal)
        withdrawal.setdefault(
            "rotation_vectors_world_rad",
            dict(right=[0.0, 0.0, 0.0], left=[0.0, 0.0, 0.0]),
        )
        withdrawal["rotation_vectors_world_rad"][self.moving["arm"]] = np.asarray(
            rotation
        ).tolist()
        time = self.release["start_s"] + fraction * self.release["duration_s"]
        for hand in self.hands:
            goal = hand["goal"].copy()
            if hand["role"] == self.moving["role"]:
                for name, value in zip(self.names, q):
                    goal[hand["names"].index(name)] = value
            a = smoothstep(fraction)
            data.qpos[hand["qids"]] = (1 - a) * hand["start"] + a * goal
            pose = transform(time, hand["arm"], hand["base"], withdrawal, self.release)
            data.qpos[hand["fixture"][:3]] = pose[:3, 3] - hand["base"][:3, 3]
            data.qpos[hand["fixture"][3:]] = Rotation.from_matrix(
                hand["base"][:3, :3].T @ pose[:3, :3]
            ).as_euler("XYZ")
        mujoco.mj_fwdPosition(model, data)
        required = -0.003 + fraction * (self.clearance + 0.003)
        object_error, robot_error, gap, worst = contact_profile(
            model, data, self.primary["geoms"], self.robot, self.objects, required
        )
        secondary_error = np.zeros(0)
        secondary_gap = None
        secondary_worst = None
        if self.bilateral:
            required_secondary = -0.003 + smoothstep(
                np.clip((fraction - 0.8) / 0.2, 0, 1)
            ) * (self.clearance + 0.003)
            secondary_error, _, secondary_gap, secondary_worst = contact_profile(
                model,
                data,
                self.secondary["geoms"],
                self.robot,
                self.objects,
                required_secondary,
            )
        table = table_gap(model, data, self.robot, self.table)
        return np.r_[
            object_error / 0.002,
            secondary_error / 0.002,
            robot_error / 0.002,
            max(0.0, 0.005 - table) / 0.002,
        ], dict(
            fraction=float(fraction),
            required_object_gap_m=required,
            object_gap_violation_m=float(object_error.max(initial=0)),
            robot_penetration_excess_m=float(robot_error.max(initial=0)),
            table_gap_m=table,
            minimum_near_object_distance_m=gap,
            worst_contact=worst,
            secondary_object_gap_violation_m=float(secondary_error.max(initial=0)),
            secondary_minimum_near_object_distance_m=secondary_gap,
            secondary_worst_contact=secondary_worst,
        )

    def evaluate(self, q, rotation, samples):
        fractions = np.linspace(0, 1, samples) if np.isscalar(samples) else samples
        values = [self.sample(f, q, rotation) for f in fractions]
        return np.concatenate([v[0] for v in values]), [v[1] for v in values]


def run(config):
    from scipy.optimize import least_squares

    implementation = Path(__file__).read_bytes()
    if (
        not 1 <= len(config["rotation_seeds_world_rad"]) <= 9
        or not 0.001 <= config["clearance_m"] <= 0.005
        or not 5 <= config["fit_samples"] <= 17
        or not 51 <= config["verification_samples"] <= 151
        or not 1 <= config["maximum_evaluations"] <= 100
    ):
        raise ValueError("Finite bounded release path search required")
    fixture = FixturePath(config)
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "release_path_search.py").write_bytes(implementation)
    rows = []
    for index, vector in enumerate(config["rotation_seeds_world_rad"]):
        rotation = np.asarray(vector, float)
        if (
            rotation.shape != (3,)
            or not np.isfinite(rotation).all()
            or np.linalg.norm(rotation) > 0.2
        ):
            raise ValueError("World rotation seed exceeds 0.2 rad")

        optimize = bool(config.get("optimize_rotation", False))
        fractions = list(np.linspace(0, 1, config["fit_samples"]))
        start = fixture.seed.copy()
        lo = fixture.lower.copy()
        hi = fixture.upper.copy()
        if optimize:
            start = np.r_[start, rotation / (0.2 - 1e-9)]
            lo = np.r_[lo, [-1.0] * 3]
            hi = np.r_[hi, [1.0] * 3]

        def split(value):
            return (
                (value[:-3], bounded_rotation(value[-3:]))
                if optimize
                else (value, rotation)
            )

        evaluations = 0
        rounds = config.get("refinement_rounds", 0)
        if type(rounds) is not int or not 0 <= rounds <= 3:
            raise ValueError("At most three bounded path refinement rounds allowed")
        for iteration in range(rounds + 1):

            def residual(value):
                q, angle = split(value)
                values, _ = fixture.evaluate(q, angle, fractions)
                return (
                    np.r_[values, 0.03 * (q - fixture.seed), 0.02 * angle / 0.2]
                    if optimize
                    else np.r_[values, 0.03 * (q - fixture.seed)]
                )

            fit = least_squares(
                residual,
                np.clip(start, lo + 1e-9, hi - 1e-9),
                bounds=(lo, hi),
                max_nfev=config["maximum_evaluations"],
                diff_step=1e-4,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-7,
            )
            evaluations += int(fit.nfev)
            q, angle = split(fit.x)
            _, verification = fixture.evaluate(q, angle, config["verification_samples"])
            worst = sorted(
                verification,
                key=lambda s: max(
                    s["object_gap_violation_m"],
                    s["secondary_object_gap_violation_m"],
                    s["robot_penetration_excess_m"],
                ),
                reverse=True,
            )[:3]
            additions = [s["fraction"] for s in worst if s["fraction"] not in fractions]
            if not additions or iteration == rounds:
                break
            fractions = sorted(set(fractions + additions))
            start = fit.x.copy()
        fitted, actual_rotation = split(fit.x)
        _, samples = fixture.evaluate(
            fitted, actual_rotation, config["verification_samples"]
        )
        error = max(
            max(s["object_gap_violation_m"], s["secondary_object_gap_violation_m"])
            for s in samples
        )
        robot = max(s["robot_penetration_excess_m"] for s in samples)
        table = min(s["table_gap_m"] for s in samples)
        accepted = bool(error <= 0.0001 and robot <= 1e-8 and table >= 0.005)
        target = dict(fixture.goal)
        target.update(zip(fixture.names, fitted.tolist()))
        withdrawal = deepcopy(fixture.withdrawal)
        withdrawal.setdefault(
            "rotation_vectors_world_rad",
            dict(right=[0.0, 0.0, 0.0], left=[0.0, 0.0, 0.0]),
        )
        withdrawal["rotation_vectors_world_rad"][fixture.moving["arm"]] = (
            actual_rotation.tolist()
        )
        folder = out / f"candidate-{index:02d}"
        folder.mkdir()
        posture = dict(
            schema="real2sim2real.finger-release-posture/1",
            accepted=accepted,
            joint_targets_rad=target,
            nominal_open_rad=fixture.nominal,
            wrist_withdrawal=withdrawal,
            release_settings=fixture.release,
            source_sha256=config["source_sha256"],
            geometry_sampling="position-only contact query with clearance margin on exact hand fixture geometry; requires full-arm admission",
            complete_opening_path_validated=False,
            hand_fixture_path_sampled=True,
            full_arm_path_validated=False,
            dynamic_release_validated=False,
            config=config,
            implementation_sha256=digest(out / "release_path_search.py"),
        )
        write_json(folder / "posture.json", posture)
        row = dict(
            candidate=index,
            optimized_role=fixture.moving["role"],
            accepted=accepted,
            rotation_world_rad=actual_rotation.tolist(),
            rotation_seed_world_rad=rotation.tolist(),
            maximum_object_gap_violation_m=error,
            maximum_robot_penetration_excess_m=robot,
            minimum_table_gap_m=table,
            evaluations=evaluations,
            fit_fractions=fractions,
            posture=str((folder / "posture.json").resolve()),
            posture_sha256=digest(folder / "posture.json"),
            samples=samples,
        )
        rows.append(row)
        write_json(out / "progress.json", dict(candidates=rows))
        print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
    admitted = [r for r in rows if r["accepted"]]
    report = dict(
        schema="real2sim2real.release-path-search/1",
        config=config,
        source_time_s=fixture.source_time,
        hand_geometry_alignment_error_m=fixture.geometry_alignment_error_m,
        hand_geometry_alignment_error_rad=fixture.geometry_alignment_error_rad,
        source_success=fixture.source["success"],
        source_rejected_for=fixture.source["rejected_for"],
        candidates=rows,
        selected=None
        if not admitted
        else min(
            admitted,
            key=lambda r: (np.linalg.norm(r["rotation_world_rad"]), r["candidate"]),
        )["candidate"],
        full_arm_path_validated=False,
        physical_release_validated=False,
        implementation_sha256=digest(out / "release_path_search.py"),
    )
    write_json(out / "search.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if not 0 < config["deadline_s"] <= 900:
        raise ValueError("Finite search deadline required")
    if a.worker:
        return run(config)
    from real2sim2real.bounded_process import run as bounded

    return bounded(
        [
            sys.executable,
            "-m",
            __spec__.name,
            "--config",
            str(a.config.resolve()),
            "--worker",
        ],
        timeout=config["deadline_s"],
    )


if __name__ == "__main__":
    main()
