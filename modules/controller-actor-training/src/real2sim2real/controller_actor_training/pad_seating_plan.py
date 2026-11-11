"""Bounded opposing-pad seating proposals from a pinned saved arm state.

Collision witnesses are simulator geometry evidence. They are not independent
observations or a certificate of force closure; stationary dynamics is required.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from real2sim2real.artifacts import digest, write_json


def nonpad_chain_geoms(model, pad):
    """Collidable geoms on this finger's controlled joint bodies, excluding its pad."""
    bodies = set(int(model.jnt_bodyid[int(model.dof_jntid[d])]) for d in pad["dofs"])
    return [
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in bodies
        and g != pad["gid"]
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]


def nonpad_clearance_passes(rows, clearance):
    return bool(rows) and all(
        np.isfinite(r["gap_m"])
        and r["gap_m"] >= clearance - 0.0001
        and (
            r.get("distance_is_lower_bound", False)
            or (
                np.isfinite(r["witness_distance_error_m"])
                and r["witness_distance_error_m"] <= 0.00001
            )
        )
        for r in rows
    )


def contact_witness(data, geom, objects, cutoff):
    """Signed near-contact witness, pad first, from a position-only margin query.

    No pair within the declared margin means a censored lower bound, not a
    measured zero distance. Query margins must cover the requested cutoff.
    """
    choices = []
    for c in data.contact:
        if c.geom1 == geom and c.geom2 in objects:
            sign = 1
        elif c.geom2 == geom and c.geom1 in objects:
            sign = -1
        else:
            continue
        normal = sign * np.asarray(c.frame).reshape(3, 3)[0]
        delta = 0.5 * float(c.dist) * normal
        choices.append((float(c.dist), np.r_[c.pos - delta, c.pos + delta]))
    return min(choices, key=lambda v: v[0]) if choices else (cutoff, np.zeros(6))


def bind(live, model, spec):
    if set(spec) - {"close_fraction"} != {"path", "sha256"}:
        raise ValueError("Pinned seating proposal required")
    fraction = spec.get("close_fraction", 1.0)
    if not np.isfinite(fraction) or not 0.5 <= fraction <= 1.0:
        raise ValueError("Bounded finger closure fraction required")
    path = Path(spec["path"])
    if digest(path) != spec["sha256"]:
        raise ValueError("Seating proposal changed")
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "real2sim2real.pad-seating-plan/1"
        or report.get("geometry_accepted") is not True
    ):
        raise ValueError("Geometrically accepted seating proposal required")
    pads = {p["role"] + ":" + p["group"]: p for p in live.servo.pads}
    changes, points = {}, []
    for item in report["pads"]:
        pad = pads.get(item["label"])
        if pad is None or model.geom(pad["gid"]).name != item["geom"]:
            raise ValueError("Seating pad identity changed")
        names = [model.joint(int(model.dof_jntid[d])).name for d in pad["dofs"]]
        if names != item["joint_names"] or item["label"].endswith(":thumb"):
            raise ValueError("Only the declared opposing finger chains may change")
        proposed = np.asarray(item["closed_rad"], float)
        hand = next(h for h in live.hands if h["role"] == pad["role"])
        old = np.array(
            [hand["waypoints"][2][list(hand["qids"]).index(q)] for q in pad["qids"]]
        )
        if (
            proposed.shape != old.shape
            or not np.isfinite(proposed).all()
            or np.max(abs(proposed - old)) > 0.35 + 1e-9
            or np.any(proposed < pad["limits"][:, 0])
            or np.any(proposed > pad["limits"][:, 1])
        ):
            raise ValueError("Seating exceeds original finger/control bounds")
        for q, value in zip(pad["qids"], proposed):
            if int(q) in changes:
                raise ValueError("Seating proposals require disjoint finger chains")
            changes[int(q)] = value
        body_point, object_point, normal = [
            np.asarray(item[k], float)
            for k in ("point_body_m", "point_object_m", "normal_object")
        ]
        if any(
            v.shape != (3,) or not np.isfinite(v).all()
            for v in (body_point, object_point, normal)
        ) or not np.isclose(np.linalg.norm(normal), 1):
            raise ValueError("Finite metric contact frames and unit normal required")
        points.append((pad, body_point, object_point, normal))
    if not changes:
        raise ValueError("Nonempty opposing-pad proposal required")
    for hand in live.hands:
        closed = hand["waypoints"][2].copy()
        for i, q in enumerate(hand["qids"]):
            if int(q) in changes:
                closed[i] = changes[int(q)]
        hand["waypoints"][2] = closed
    for pad, point, obj, normal in points:
        pad.update(point_body=point, point_object=obj, normal_object=normal)
    if fraction != 1.0:
        from copy import deepcopy

        live.rollout = deepcopy(live.rollout)
        live.rollout["finger_close_fraction"] = fraction
    live.source_hashes[str(path)] = digest(path)


def run(config):
    import mujoco
    import trimesh
    from scipy.optimize import least_squares
    from .arm_finger_feedback import ArmFingerFeedback
    from .object_grasp_pilot import load_scene

    implementation = Path(__file__).read_bytes()
    path = Path(config["source_result"])
    if digest(path) != config["source_sha256"]:
        raise ValueError("Pinned arm observation changed")
    source = json.loads(path.read_text())
    if source.get("schema") != "real2sim2real.arm-replay-pilot/1" or not source.get(
        "continuous_protocol"
    ):
        raise ValueError("A continuous saved arm trial is required")
    for name, key in [("scene.xml", "scene_sha256"), ("states.npz", "states_sha256")]:
        if digest(path.with_name(name)) != source[key]:
            raise ValueError("Saved arm artifacts changed")
    if not (
        0 < config["maximum_joint_change_rad"] <= 0.35
        and -0.0005 <= config["target_gap_m"] <= 0
        and 1 <= config["maximum_evaluations"] <= 150
    ):
        raise ValueError("Seating is bounded to 0.35 rad and <=0.5 mm compression")
    nonpad_clearance = config.get("nonpad_clearance_m")
    if nonpad_clearance is not None and (
        isinstance(nonpad_clearance, bool)
        or not np.isfinite(nonpad_clearance)
        or not 0.0005 <= nonpad_clearance <= 0.003
    ):
        raise ValueError("Non-pad finger clearance must be 0.5–3 mm")
    audit_path = Path(source["config"]["audit"])
    if digest(audit_path) != source["config"]["audit_sha256"]:
        raise ValueError("Pinned arm audit changed")
    audit = json.loads(audit_path.read_text())
    for name, sha in audit["sources"].items():
        if digest(Path(name)) != sha:
            raise ValueError("Seating source changed")
    parent = json.loads(Path(audit["config"]["trial"]).read_text())
    model = mujoco.MjModel.from_xml_path(str(path.with_name("scene.xml")))
    data = mujoco.MjData(model)
    with np.load(path.with_name("states.npz"), allow_pickle=False) as saved:
        index = int(np.argmin(abs(saved["timestamps_s"] - config["time_s"])))
        data.qpos[:] = saved["qpos"][index]
    row = source["records"][index]
    if (
        abs(row["time_s"] - source["continuous_protocol"]["acquisition_end_s"]) > 0.03
        or row["table_contacts"] <= 0
    ):
        raise ValueError("Use the table-supported acquisition-completion state")
    _, _, mesh = load_scene(parent["configuration"])
    roots = {a["name"]: model.body(a["root"]).id for a in audit["config"]["arms"]}
    motors = [
        i for i in range(model.nu) if not model.actuator(i).name.startswith("pilot__")
    ]
    live = ArmFingerFeedback(model, data, parent, roots, mesh, motors)
    for hand in live.hands:
        data.qpos[hand["qids"]] = hand["waypoints"][2]
    mujoco.mj_kinematics(model, data)
    original = data.qpos.copy()
    obj = model.body("task_object").id
    object_geoms = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == obj and model.geom_contype[g]
    ]
    rotation, position = data.xmat[obj].reshape(3, 3).copy(), data.xpos[obj].copy()
    world_mesh = mesh.copy()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    world_mesh.apply_transform(transform)
    selected = [p for p in live.servo.pads if p["group"] != "thumb"]
    if nonpad_clearance is not None:
        query_geoms = set().union(
            *(set(nonpad_chain_geoms(model, p)) | {p["gid"]} for p in selected)
        )
        # Position-only query model; never saved or exported into dynamics.
        model.geom_margin[list(query_geoms)] = np.maximum(
            model.geom_margin[list(query_geoms)], 0.03
        )
    reports = []
    for pad in selected:
        ids = pad["qids"]
        limits = pad["limits"]
        seed = data.qpos[ids].copy()
        lo = np.maximum(limits[:, 0], seed - config["maximum_joint_change_rad"])
        hi = np.minimum(limits[:, 1], seed + config["maximum_joint_change_rad"])
        nonpads = nonpad_chain_geoms(model, pad) if nonpad_clearance is not None else []

        def object_distance(gid):
            if nonpad_clearance is not None:
                return contact_witness(data, gid, set(object_geoms), 0.03)
            choices = []
            for g in object_geoms:
                segment = np.zeros(6)
                distance = mujoco.mj_geomDistance(model, data, gid, g, 0.03, segment)
                choices.append((float(distance), segment.copy()))
            return min(choices, key=lambda v: v[0])

        def witness(q):
            data.qpos[ids] = q
            if nonpad_clearance is not None:
                mujoco.mj_fwdPosition(model, data)
            else:
                mujoco.mj_kinematics(model, data)
            return object_distance(pad["gid"])

        def residual(q):
            distance, _ = witness(q)
            base = np.r_[distance - config["target_gap_m"], 0.0002 * (q - seed)]
            if nonpad_clearance is not None:
                return np.r_[
                    base,
                    [
                        10 * max(0, nonpad_clearance - object_distance(g)[0])
                        for g in nonpads
                    ],
                ]
            return base

        before, _ = witness(seed)
        fit = least_squares(
            residual,
            np.clip(seed, lo + 1e-9, hi - 1e-9),
            bounds=(lo, hi),
            max_nfev=config["maximum_evaluations"],
            diff_step=1e-4,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        after, segment = witness(fit.x)
        nonpad_rows = []
        for g in nonpads:
            gap, points = object_distance(g)
            censored = bool(gap == 0.03 and not np.any(points))
            nonpad_rows.append(
                dict(
                    geom_id=g,
                    geom=model.geom(g).name,
                    body=model.body(int(model.geom_bodyid[g])).name,
                    gap_m=gap,
                    distance_is_lower_bound=censored,
                    witness_distance_error_m=None
                    if censored
                    else abs(abs(gap) - float(np.linalg.norm(points[:3] - points[3:]))),
                )
            )
        near, gap, faces = trimesh.proximity.closest_point(world_mesh, [segment[3:]])
        body_rotation = data.xmat[pad["body"]].reshape(3, 3)
        reports.append(
            dict(
                label=pad["role"] + ":" + pad["group"],
                geom=model.geom(pad["gid"]).name,
                joint_names=[
                    model.joint(int(model.dof_jntid[d])).name for d in pad["dofs"]
                ],
                nominal_closed_rad=seed.tolist(),
                closed_rad=fit.x.tolist(),
                gap_before_m=before,
                gap_after_m=after,
                witness_distance_error_m=abs(
                    abs(after) - float(np.linalg.norm(segment[:3] - segment[3:]))
                ),
                source_mesh_projection_m=float(gap[0]),
                point_body_m=(
                    body_rotation.T @ (segment[:3] - data.xpos[pad["body"]])
                ).tolist(),
                point_object_m=(rotation.T @ (near[0] - position)).tolist(),
                normal_object=(rotation.T @ world_mesh.face_normals[faces[0]]).tolist(),
                evaluations=int(fit.nfev),
                **(
                    dict(
                        nonpad_geoms=nonpad_rows,
                        nonpad_clearance_accepted=nonpad_clearance_passes(
                            nonpad_rows, nonpad_clearance
                        ),
                    )
                    if nonpad_clearance is not None
                    else {}
                ),
            )
        )
    mujoco.mj_fwdPosition(model, data)
    robot_geoms = set().union(*live.servo.role_geoms.values())
    penetration = max(
        (
            max(0.0, -float(c.dist))
            for c in data.contact
            if set((c.geom1, c.geom2)) & robot_geoms
        ),
        default=0.0,
    )
    changed = np.zeros(model.nq, bool)
    for p in selected:
        changed[p["qids"]] = True
    if not np.array_equal(data.qpos[~changed], original[~changed]):
        raise RuntimeError("Seating changed a wrist, thumb or object coordinate")
    accepted = (
        all(
            abs(p["gap_after_m"] - config["target_gap_m"]) <= 0.001
            and p["source_mesh_projection_m"] <= 0.003
            and p["witness_distance_error_m"] <= 0.00001
            and p.get("nonpad_clearance_accepted", True)
            for p in reports
        )
        and penetration <= 0.003
    )
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "pad_seating_plan.py").write_bytes(implementation)
    report = dict(
        schema="real2sim2real.pad-seating-plan/1",
        config=config,
        source_sha256=digest(path),
        source_time_s=row["time_s"],
        source_success=source["success"],
        source_rejected_for=source["rejected_for"],
        geometry_accepted=bool(accepted),
        pads=reports,
        maximum_robot_penetration_m=penetration,
        wrists_and_thumbs_frozen=True,
        physical_retention_validated=False,
        independent_contact_observations=False,
        **(
            dict(
                geometry_queries="position-only contact-margin queries; margins are not exported to dynamics",
                query_margin_m=0.03,
            )
            if nonpad_clearance is not None
            else {}
        ),
        implementation_sha256=digest(out / "pad_seating_plan.py"),
    )
    write_json(out / "plan.json", report)
    print(
        json.dumps(
            dict(
                geometry_accepted=bool(accepted),
                maximum_robot_penetration_m=penetration,
                pad_gaps_m={p["label"]: p["gap_after_m"] for p in reports},
            )
        ),
        flush=True,
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if not 0 < config["deadline_s"] <= 900:
        raise ValueError("Finite seating search deadline required")
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
