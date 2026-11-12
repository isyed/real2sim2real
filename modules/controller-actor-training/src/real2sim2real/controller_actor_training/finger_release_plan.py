"""Plan bounded finger opening against reconstructed collision geometry on CPU.

This is endpoint geometry evidence, not a successful dynamic release. The live
adapter must screen the complete opening path and evaluate actual separation.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from real2sim2real.artifacts import digest, write_json


def read_posture(spec, *, withdrawal=None, release=None):
    """Read one pinned geometric proposal with its coupled path/clock contract."""
    if set(spec) != {"path", "sha256"}:
        raise ValueError("Pinned release posture path and digest required")
    path = Path(spec["path"])
    if digest(path) != spec["sha256"]:
        raise ValueError("Pinned release posture changed")
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "real2sim2real.finger-release-posture/1"
        or report.get("accepted") is not True
    ):
        raise ValueError("Geometrically admitted release posture required")
    if report.get("wrist_withdrawal") is not None and (
        report["wrist_withdrawal"] != withdrawal
        or report.get("release_settings") != release
    ):
        raise ValueError(
            "Opening posture must use its jointly screened wrist path and release clock"
        )
    return path, report


def bind(live, model, spec, *, withdrawal=None, release=None):
    path, report = read_posture(spec, withdrawal=withdrawal, release=release)
    targets = report["joint_targets_rad"]
    if not targets:
        raise ValueError("Nonempty release posture required")
    mapped = {}
    for name, value in targets.items():
        joint = model.joint(name)
        qid = int(joint.qposadr[0])
        if (
            qid not in live.qids
            or not np.isfinite(value)
            or not joint.range[0] <= value <= joint.range[1]
        ):
            raise ValueError("Release may change owned limited finger coordinates only")
        mapped[qid] = value
    proposals = []
    for hand in live.hands:
        released = hand["waypoints"][1].copy()
        for i, qid in enumerate(hand["qids"]):
            if qid in mapped:
                if abs(mapped[qid] - released[i]) > 0.5 + 1e-9:
                    raise ValueError("Release posture exceeds bounded opening change")
                released[i] = mapped[qid]
        proposals.append(released)
    for hand, released in zip(live.hands, proposals):
        hand["release_open"] = released
    live.source_hashes[str(path)] = digest(path)


def planning_targets(model, hands, targets):
    """Opening waypoints for fixture planning, preserving acquisition and wrists."""
    allowed = {
        n
        for hand in hands
        for n in hand["joint_names"]
        if not n.startswith(hand["fixture_prefix"])
    }
    if not targets or not set(targets) <= allowed:
        raise ValueError("Opening targets may own admitted finger joints only")
    result = {}
    for hand in hands:
        values = np.asarray(hand["q_open"], float).copy()
        for i, name in enumerate(hand["joint_names"]):
            if name in targets:
                value = targets[name]
                joint = model.joint(name)
                if (
                    not np.isfinite(value)
                    or abs(value - values[i]) > 0.5 + 1e-9
                    or not joint.range[0] <= value <= joint.range[1]
                ):
                    raise ValueError(
                        "Planned opening exceeds native finger limits or the 0.5 rad change budget"
                    )
                values[i] = value
        result[hand["root_body"]] = values
    return result


def run(config):
    import mujoco
    from scipy.optimize import least_squares

    implementation = Path(__file__).read_bytes()
    path = Path(config["source_result"])
    if digest(path) != config["source_sha256"]:
        raise ValueError("Pinned release diagnostic changed")
    source = json.loads(path.read_text())
    if source.get("schema") != "real2sim2real.arm-replay-pilot/1" or not source.get(
        "continuous_protocol", {}
    ).get("release_planned"):
        raise ValueError("Continuous arm release diagnostic required")
    for name, key in [("scene.xml", "scene_sha256"), ("states.npz", "states_sha256")]:
        if digest(path.with_name(name)) != source[key]:
            raise ValueError("Saved release artifacts changed")
    if not (
        0 < config["maximum_joint_change_rad"] <= 0.5
        and 0.001 <= config["clearance_m"] <= 0.005
        and 1 <= config["maximum_evaluations"] <= 200
    ):
        raise ValueError("Bounded opening search required")
    model = mujoco.MjModel.from_xml_path(str(path.with_name("scene.xml")))
    data = mujoco.MjData(model)
    with np.load(path.with_name("states.npz")) as states:
        index = int(np.argmin(abs(states["timestamps_s"] - config["time_s"])))
        data.qpos[:] = states["qpos"][index]
    state_time = source["records"][index]["time_s"]
    if (
        source["records"][index]["table_contacts"] <= 0
        or state_time < source["continuous_protocol"]["capture_duration_s"]
    ):
        raise ValueError("Use a returned, table-contact state before release")
    audit_path = Path(source["config"]["audit"])
    if digest(audit_path) != source["config"]["audit_sha256"]:
        raise ValueError("Pinned arm audit changed")
    audit = json.loads(audit_path.read_text())
    for name, sha in audit["sources"].items():
        if digest(Path(name)) != sha:
            raise ValueError("Release planning source changed")
    parent = json.loads(Path(audit["config"]["trial"]).read_text())
    opened = {}
    for hand in [parent["candidate"], parent["candidate"]["support"]]:
        opened.update(zip(hand["joint_names"], hand["q_open"]))
    names = config["joints"]
    if (
        not names
        or len(set(names)) != len(names)
        or any(n not in opened for n in names)
    ):
        raise ValueError("Distinct source finger joints required")
    # Only live finger-owned coordinates can be exported by the search.
    from .arm_finger_feedback import ArmFingerFeedback
    from .object_grasp_pilot import load_scene

    _, _, mesh = load_scene(parent["configuration"])
    arms = audit["config"]["arms"]
    roots = {a["name"]: model.body(a["root"]).id for a in arms}
    motors = [
        a for a in range(model.nu) if not model.actuator(a).name.startswith("pilot__")
    ]
    live = ArmFingerFeedback(model, data, parent, roots, mesh, motors)
    ids = [model.joint(n).id for n in names]
    qids = model.jnt_qposadr[ids]
    if not set(qids) <= set(live.qids) or not np.all(model.jnt_limited[ids]):
        raise ValueError("Search must own bounded finger joints only")
    for hand in live.hands:
        data.qpos[hand["qids"]] = hand["waypoints"][1]
    seed = np.array([opened[n] for n in names])
    original = data.qpos.copy()
    object_geoms = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == model.body("task_object").id
        and model.geom_contype[i]
    ]
    geoms = [model.geom(n).id for n in config["geoms"]]
    owned_geoms = set().union(*live.servo.role_geoms.values())
    if not geoms or not set(geoms) <= owned_geoms:
        raise ValueError("Explicit hand collision geometries required")
    lower = np.maximum(
        model.jnt_range[ids, 0], seed - config["maximum_joint_change_rad"]
    )
    upper = np.minimum(
        model.jnt_range[ids, 1], seed + config["maximum_joint_change_rad"]
    )

    def distances(q):
        data.qpos[qids] = q
        mujoco.mj_kinematics(model, data)
        return np.array(
            [
                min(
                    mujoco.mj_geomDistance(model, data, g, o, 0.02, None)
                    for o in object_geoms
                )
                for g in geoms
            ]
        )

    def residual(q):
        return np.r_[
            np.maximum(0, config["clearance_m"] - distances(q)), 0.0001 * (q - seed)
        ]

    before = distances(seed)
    fit = least_squares(
        residual,
        np.clip(seed, lower + 1e-9, upper - 1e-9),
        bounds=(lower, upper),
        max_nfev=config["maximum_evaluations"],
        diff_step=1e-4,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    after = distances(fit.x)
    untouched = np.ones(model.nq, bool)
    untouched[qids] = False
    if not np.array_equal(data.qpos[untouched], original[untouched]):
        raise RuntimeError("Opening search changed wrist or object state")
    report = dict(
        schema="real2sim2real.finger-release-posture/1",
        config=config,
        source_time_s=state_time,
        source_sha256=digest(path),
        joint_targets_rad=dict(zip(names, fit.x.tolist())),
        nominal_open_rad=dict(zip(names, seed.tolist())),
        minimum_clearance_before_m=float(before.min()),
        minimum_clearance_after_m=float(after.min()),
        accepted=bool(after.min() >= config["clearance_m"] - 0.0001),
        evaluations=int(fit.nfev),
        dynamic_release_validated=False,
        complete_opening_path_validated=False,
        implementation_sha256=digest(Path(__file__)),
    )
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "finger_release_plan.py").write_bytes(implementation)
    report["implementation_sha256"] = digest(out / "finger_release_plan.py")
    write_json(out / "posture.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "accepted",
                    "minimum_clearance_before_m",
                    "minimum_clearance_after_m",
                    "evaluations",
                ]
            }
        )
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if not 0 < config["deadline_s"] <= 900:
        raise ValueError("Finite opening search deadline required")
    if args.worker:
        return run(config)
    from real2sim2real.bounded_process import run as bounded

    return bounded(
        [
            sys.executable,
            "-m",
            __spec__.name,
            "--config",
            str(args.config.resolve()),
            "--worker",
        ],
        timeout=config["deadline_s"],
    )


if __name__ == "__main__":
    main()
