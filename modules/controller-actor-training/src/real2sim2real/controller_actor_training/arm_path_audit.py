"""CPU arm IK and collision audit of saved wrist-fixture states, not dynamics."""

import argparse
from copy import deepcopy
import html
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from real2sim2real.artifacts import digest, write_json


def interval_speed(q, times, limits):
    q, times, limits = map(np.asarray, (q, times, limits))
    if (
        q.ndim != 2
        or len(q) < 2
        or times.shape != (len(q),)
        or limits.shape != (q.shape[1],)
        or not np.isfinite(q).all()
        or not np.isfinite(times).all()
        or not np.isfinite(limits).all()
        or (np.diff(times) <= 0).any()
        or (limits <= 0).any()
    ):
        raise ValueError(
            "Finite positions, positive limits and increasing timestamps required"
        )
    return np.max(np.abs(np.diff(q, axis=0)) / np.diff(times)[:, None] / limits, axis=0)


def joint_neighbors(model, a, b):
    """Recognize robot joint neighbors through fixed mounts; never ignore world.

    With fixed frames retained, joint housings can overlap across massless links.
    Keep those contacts in raw diagnostics. Two moving joints are not neighbors.
    """
    if a == 0 or b == 0:
        return False
    for child, ancestor in [(a, b), (b, a)]:
        joints = 0
        while child and child != ancestor:
            joints += int(model.body_jntnum[child])
            child = int(model.body_parentid[child])
        if child == ancestor and joints <= 1:
            return True
    return False


def build_model(urdf, scene, output):
    """Import collision URDF with absolute assets; preserve fixed mounting links."""
    import mujoco

    urdf, scene = Path(urdf), Path(scene)
    root = ET.parse(urdf).getroot()
    hashes = {str(urdf): digest(urdf), str(scene): digest(scene)}
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
    for mesh in root.findall(".//mesh"):
        p = (urdf.parent / mesh.get("filename")).resolve()
        mesh.set("filename", str(p))
        hashes[str(p)] = digest(p)
    ext = root.find("mujoco")
    if ext is None:
        ext = ET.SubElement(root, "mujoco")
    compiler = ext.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(ext, "compiler")
    compiler.attrib.update(strippath="false", fusestatic="false", balanceinertia="true")
    path = output / "collision-robot.urdf"
    ET.ElementTree(root).write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    path = output / "collision-robot.xml"
    mujoco.mj_saveLastXML(str(path), model)
    robot = ET.parse(path).getroot()
    task = ET.parse(scene).getroot()
    for mesh in task.findall("asset/mesh"):
        if mesh.get("name", "").startswith("task_"):
            mesh = deepcopy(mesh)
            p = (
                scene.parent
                / task.find("compiler").get("meshdir", "")
                / mesh.get("file")
            ).resolve()
            mesh.set("file", str(p))
            hashes[str(p)] = digest(p)
            robot.find("asset").append(mesh)
    for element in task.find("worldbody"):
        if element.get("name") in ("table", "task_object"):
            element = deepcopy(element)
            for geom in element.findall(".//geom"):
                if geom.get("name") == "object_visual":
                    element.remove(geom)
            robot.find("worldbody").append(element)
    path = output / "audit-scene.xml"
    ET.ElementTree(robot).write(path)
    return mujoco.MjModel.from_xml_path(str(path)), hashes


def fit_pose(model, data, body, joints, target, initial, max_nfev):
    import mujoco

    ids = [model.joint(n).id for n in joints]
    qids = model.jnt_qposadr[ids]
    if not np.all(model.jnt_limited[ids]) or not np.all(
        model.jnt_type[ids] == mujoco.mjtJoint.mjJNT_HINGE
    ):
        raise ValueError("Limited scalar hinge arm joints required")
    lo, hi = model.jnt_range[ids].T
    initial = np.clip(initial, lo + 1e-8, hi - 1e-8)

    def residual(q):
        data.qpos[qids] = q
        mujoco.mj_kinematics(model, data)
        rotation = Rotation.from_matrix(
            data.xmat[body].reshape(3, 3) @ target[:3, :3].T
        ).as_rotvec()
        return np.r_[
            (data.xpos[body] - target[:3, 3]) / 0.005,
            rotation / 0.05,
            (q - initial) * 0.01,
        ]

    fit = least_squares(
        residual,
        initial,
        bounds=(lo, hi),
        max_nfev=max_nfev,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-6,
    )
    residual(fit.x)
    return (
        fit.x,
        float(np.linalg.norm(data.xpos[body] - target[:3, 3])),
        float(
            np.linalg.norm(
                Rotation.from_matrix(
                    data.xmat[body].reshape(3, 3) @ target[:3, :3].T
                ).as_rotvec()
            )
        ),
    )


def run(config):
    import mujoco
    from .object_grasp_pilot import load_scene, set_wrist_frame, set_root_frame

    for p, s in config["sources"].items():
        if digest(Path(p)) != s:
            raise ValueError("Arm audit source changed")
    parent = json.loads(Path(config["trial"]).read_text())
    states_path = Path(config["trial"]).with_name("states.npz")
    if digest(states_path) != parent["states_sha256"]:
        raise ValueError("Saved states changed")
    if (
        not 1 <= len(config["placements"]) <= 8
        or not 2 <= config["maximum_samples"] <= 200
        or not 1 <= config["max_nfev"] <= 200
    ):
        raise ValueError("Bounded placement/sample/IK budget required")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", config)
    base = parent["configuration"]
    old, od, _ = load_scene(base)
    candidate = parent["candidate"]
    set_wrist_frame(
        old, od, base, candidate["base_position_m"], candidate["base_rotation"]
    )
    s = candidate["support"]
    set_root_frame(
        old,
        od,
        s["root_body"],
        s["fixture_prefix"],
        s["base_position_m"],
        s["base_rotation"],
    )
    with np.load(states_path) as archive:
        states = {k: archive[k] for k in archive.files}
    indices = np.unique(
        np.linspace(
            0,
            len(states["qpos"]) - 1,
            min(config["maximum_samples"], len(states["qpos"])),
            dtype=int,
        )
    )
    model, hashes = build_model(
        config["urdf"], Path(base["output"]) / "scene/scene.xml", out
    )
    data = mujoco.MjData(model)
    robot_base = model.body(config["base_body"]).id
    arms = config["arms"]
    if any(
        not 1 <= len(a.get("initial_seeds_rad", [a["seed_rad"]])) <= 4 for a in arms
    ):
        raise ValueError("At most four initial IK seeds per arm required")
    arm_bodies = {model.body(n).id for arm in arms for n in arm["collision_bodies"]}
    limits = {
        j.get("name"): float(j.find("limit").get("velocity"))
        for j in ET.parse(config["urdf"]).getroot().findall("joint")
        if j.find("limit") is not None and j.find("limit").get("velocity") is not None
    }
    old_obj = old.joint("task_object_free").qposadr[0]
    new_obj = model.joint("task_object_free").qposadr[0]
    pairs = []
    for arm in arms:
        for name in (
            candidate["joint_names"] if arm["source_prefix"] == "" else s["joint_names"]
        ):
            if "fixture_" in name:
                continue
            target = name.removeprefix(arm["source_prefix"])
            pairs.append((old.joint(name).qposadr[0], model.joint(target).qposadr[0]))
    rows = []
    for placement in config["placements"]:
        model.body_pos[robot_base] = placement["position_m"]
        model.body_quat[robot_base] = Rotation.from_euler(
            "z", placement["yaw_rad"]
        ).as_quat()[[3, 0, 1, 2]]
        data.qpos[:] = model.qpos0
        seeds = {a["name"]: np.array(a["seed_rad"]) for a in arms}
        samples = []
        for i in indices:
            od.qpos[:] = states["qpos"][i]
            mujoco.mj_forward(old, od)
            data.qpos[new_obj : new_obj + 7] = od.qpos[old_obj : old_obj + 7]
            for src, dst in pairs:
                data.qpos[dst] = od.qpos[src]
            sample = dict(
                source_index=int(i),
                time_s=float(states["timestamps_s"][i]),
                phase=parent["records"][i]["phase"],
                arms={},
            )
            for arm in arms:
                src = old.body(arm["source_root"]).id
                dst = model.body(arm["root"]).id
                target = np.eye(4)
                target[:3, 3] = od.xpos[src]
                target[:3, :3] = od.xmat[src].reshape(3, 3)
                starts = (
                    arm.get("initial_seeds_rad", [arm["seed_rad"]])
                    if not samples
                    else [seeds[arm["name"]]]
                )
                fits = [
                    fit_pose(
                        model,
                        data,
                        dst,
                        arm["joints"],
                        target,
                        np.asarray(start),
                        config["max_nfev"],
                    )
                    for start in starts
                ]
                q, pe, re = min(
                    fits, key=lambda f: (f[1] / 0.005) ** 2 + (f[2] / 0.05) ** 2
                )
                data.qpos[[model.joint(n).qposadr[0] for n in arm["joints"]]] = q
                seeds[arm["name"]] = q
                sample["arms"][arm["name"]] = dict(
                    q=q.tolist(),
                    position_error_m=pe,
                    rotation_error_deg=float(np.rad2deg(re)),
                )
            # Position/collision audit only: do not invoke the contact dynamics
            # solver on an imported, unactuated arm model.
            mujoco.mj_fwdPosition(model, data)
            contacts = []
            for contact in data.contact:
                bodies = [
                    int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)
                ]
                if contact.dist < 0 and any(b in arm_bodies for b in bodies):
                    contacts.append(
                        dict(
                            bodies=[model.body(b).name for b in bodies],
                            penetration_m=float(-contact.dist),
                            joint_neighbors=joint_neighbors(model, *bodies),
                        )
                    )
            sample["arm_contacts"] = contacts
            sample["hand_frame_consistency"] = {}
            for arm in arms:
                src = old.body(arm["source_root"]).id
                dst = model.body(arm["root"]).id
                rs = od.xmat[src].reshape(3, 3)
                rd = data.xmat[dst].reshape(3, 3)
                errors = []
                for b in range(old.nbody):
                    ancestor = b
                    while ancestor and ancestor != src:
                        ancestor = int(old.body_parentid[ancestor])
                    if ancestor != src:
                        continue
                    name = old.body(b).name.removeprefix(arm["source_prefix"])
                    ident = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    if ident < 0:
                        raise ValueError("Missing hand body in arm model: " + name)
                    delta = rs.T @ (od.xpos[b] - od.xpos[src]) - rd.T @ (
                        data.xpos[ident] - data.xpos[dst]
                    )
                    angle = Rotation.from_matrix(
                        (rs.T @ od.xmat[b].reshape(3, 3))
                        @ (rd.T @ data.xmat[ident].reshape(3, 3)).T
                    ).magnitude()
                    errors.append(
                        (float(np.linalg.norm(delta)), float(np.rad2deg(angle)))
                    )
                sample["hand_frame_consistency"][arm["name"]] = dict(
                    maximum_position_error_m=max(e[0] for e in errors),
                    maximum_rotation_error_deg=max(e[1] for e in errors),
                    compared_bodies=len(errors),
                )
            samples.append(sample)
        scores = {}
        for arm in arms:
            name = arm["name"]
            q = np.array([s["arms"][name]["q"] for s in samples])
            lohi = model.jnt_range[[model.joint(n).id for n in arm["joints"]]]
            scores[name] = dict(
                maximum_position_error_m=max(
                    s["arms"][name]["position_error_m"] for s in samples
                ),
                maximum_rotation_error_deg=max(
                    s["arms"][name]["rotation_error_deg"] for s in samples
                ),
                minimum_joint_margin_rad=float(
                    np.minimum(q - lohi[:, 0], lohi[:, 1] - q).min()
                ),
                maximum_sampled_velocity_ratio=float(
                    interval_speed(
                        q,
                        [s["time_s"] for s in samples],
                        [limits[n] for n in arm["joints"]],
                    ).max()
                ),
            )
        max_pen = max(
            [
                c["penetration_m"]
                for s in samples
                for c in s["arm_contacts"]
                if not c["joint_neighbors"]
            ],
            default=0,
        )
        gates = config["criteria"]
        passed = (
            all(
                v["maximum_position_error_m"] <= gates["position_error_m"]
                and v["maximum_rotation_error_deg"] <= gates["rotation_error_deg"]
                and v["minimum_joint_margin_rad"] >= gates["joint_margin_rad"]
                and v["maximum_sampled_velocity_ratio"] <= 1
                for v in scores.values()
            )
            and max_pen <= gates["arm_penetration_m"]
            and all(
                s["hand_frame_consistency"][a["name"]]["maximum_position_error_m"]
                <= gates["position_error_m"]
                and s["hand_frame_consistency"][a["name"]]["maximum_rotation_error_deg"]
                <= gates["rotation_error_deg"]
                for s in samples
                for a in arms
            )
        )
        row = dict(
            placement=placement,
            arms=scores,
            maximum_arm_penetration_m=max_pen,
            screen_pass=passed,
            samples=samples,
            hand_frame_consistency={
                a["name"]: dict(
                    maximum_position_error_m=max(
                        s["hand_frame_consistency"][a["name"]][
                            "maximum_position_error_m"
                        ]
                        for s in samples
                    ),
                    maximum_rotation_error_deg=max(
                        s["hand_frame_consistency"][a["name"]][
                            "maximum_rotation_error_deg"
                        ]
                        for s in samples
                    ),
                )
                for a in arms
            },
        )
        rows.append(row)
        write_json(out / "progress.json", dict(placements=rows))
        print(
            placement["name"],
            scores,
            "arm_penetration",
            max_pen,
            "pass",
            passed,
            flush=True,
        )
    report = dict(
        schema="real2sim2real.arm-path-audit/1",
        config=config,
        placements=rows,
        sources={**config["sources"], **hashes, str(states_path): digest(states_path)},
        physics_run=False,
        hand_mount_validation="URDF hand-root frames, named joints and relative hand-body FK compared; collision geometry and dynamics equivalence not established",
        base_placement_measured=False,
        torque_validated=False,
        continuous_collision_certified=False,
        arm_integration_ready=False,
        implementation_sha256=digest(Path(__file__)),
    )
    write_json(out / "audit.json", report)
    table = "".join(
        "<tr>"
        + "".join(
            "<td>" + html.escape(str(v)) + "</td>"
            for v in [
                r["placement"]["name"],
                r["screen_pass"],
                round(r["maximum_arm_penetration_m"] * 1000, 2),
                r["arms"],
            ]
        )
        + "</tr>"
        for r in rows
    )
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;max-width:1200px;margin:30px auto}td,th{padding:10px;border:1px solid #ccc}</style><h1>Arm feasibility: saved wrist paths</h1><p>Sampled collision-URDF inverse kinematics. Base/table placement is a declared assumption. No arm dynamics, torque or continuous collision validation. Exact saved wrist targets are retained; fitting failure does not prove global infeasibility.</p><a href="audit.json">Full metrics and source hashes</a><table><tr><th>Placement</th><th>Screen pass</th><th>Arm penetration mm</th><th>Arm metrics</th></tr>'
        + table
        + "</table>"
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    if not a.worker:
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
            timeout=c["deadline_s"],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    run(c)


if __name__ == "__main__":
    main()
