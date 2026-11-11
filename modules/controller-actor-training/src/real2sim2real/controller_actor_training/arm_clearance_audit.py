"""Robot/table intersection witnesses from saved arm states, without dynamics."""

import argparse
import json
from pathlib import Path
import numpy as np
from real2sim2real.artifacts import digest, write_json


def geometry_height_bounds(model, data, geom):
    """World vertical extent of compiled collision geometry (not visual meshes)."""
    import mujoco

    kind = model.geom_type[geom]
    center = data.geom_xpos[geom, 2]
    vertical = data.geom_xmat[geom].reshape(3, 3)[2]
    size = model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        heights = model.mesh_vert[start : start + count] @ vertical + center
        return float(heights.min()), float(heights.max())
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        radius = abs(vertical) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = size[0] + abs(vertical[2]) * size[1]
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = size[0] * np.linalg.norm(vertical[:2]) + abs(vertical[2]) * size[1]
    else:
        raise ValueError("Unsupported collision geometry for table clearance")
    return float(center - radius), float(center + radius)


def table_gap(model, data, geoms, table):
    """Conservative gap above the table's horizontal upper envelope.

    Extends the tabletop outside its finite footprint. Positive values guarantee
    separation; negative values are NOT proof of intersection. Avoids unreliable
    zero returns from separated mesh/mesh GJK distance queries.
    """
    top = geometry_height_bounds(model, data, table)[1]
    return min(geometry_height_bounds(model, data, int(g))[0] - top for g in geoms)


def table_contacts(model, data, table_geom, robot_geoms):
    rows = []
    for contact in data.contact:
        pair = {int(contact.geom1), int(contact.geom2)}
        if table_geom not in pair or contact.dist >= 0:
            continue
        robot = next(iter(pair - {table_geom}))
        if robot in robot_geoms:
            rows.append(
                dict(
                    role=robot_geoms[robot],
                    geometry=model.geom(robot).name,
                    penetration_m=float(-contact.dist),
                )
            )
    return rows


def audit(directory):
    import mujoco
    from .arm_replay_pilot import subtree_geoms

    directory = Path(directory)
    report = json.loads((directory / "result.json").read_text())
    for name, key in [("states.npz", "states_sha256"), ("scene.xml", "scene_sha256")]:
        if digest(directory / name) != report[key]:
            raise ValueError("Saved clearance input changed")
    audit_path = Path(report["config"]["audit"])
    if digest(audit_path) != report["config"]["audit_sha256"]:
        raise ValueError("Arm definition changed")
    source = json.loads(audit_path.read_text())
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    roles = {}
    for arm in source["config"]["arms"]:
        roles.update(
            {
                g: arm["name"] + "_hand"
                for g in subtree_geoms(model, model.body(arm["root"]).id)
            }
        )
        bodies = {model.body(n).id for n in arm["collision_bodies"]}
        roles.update(
            {
                g: arm["name"] + "_arm"
                for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) in bodies
            }
        )
    table = model.geom("table").id
    with np.load(directory / "states.npz", allow_pickle=False) as states:
        qpos = states["qpos"]
        times = states["timestamps_s"]
        if (
            not 1 <= len(times) <= 1000
            or len(times) != len(report["records"])
            or qpos.shape != (len(times), model.nq)
            or not np.isfinite(qpos).all()
            or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any()
        ):
            raise ValueError("Bounded aligned finite saved states required")
        phases = {}
        hand_geoms = {
            role: [g for g, r in roles.items() if r == role]
            for role in set(roles.values())
            if role.endswith("_hand")
        }
        witnesses = []
        for q, t, row in zip(qpos, times, report["records"]):
            if abs(t - row["time_s"]) > 1e-8:
                raise ValueError("Saved clearance clock changed")
            data.qpos[:] = q
            mujoco.mj_fwdPosition(model, data)
            contacts = table_contacts(model, data, table, roles)
            summary = phases.setdefault(
                row["phase"],
                dict(
                    samples=0,
                    contact_samples=0,
                    maximum_robot_table_penetration_m=0,
                    minimum_table_plane_gap_m={
                        role: float("inf") for role in hand_geoms
                    },
                ),
            )
            for role, geoms in hand_geoms.items():
                summary["minimum_table_plane_gap_m"][role] = min(
                    summary["minimum_table_plane_gap_m"][role],
                    table_gap(model, data, geoms, table),
                )
            summary["samples"] += 1
            summary["contact_samples"] += int(bool(contacts))
            summary["maximum_robot_table_penetration_m"] = max(
                summary["maximum_robot_table_penetration_m"],
                max((c["penetration_m"] for c in contacts), default=0),
            )
            if contacts:
                witnesses.append(
                    dict(time_s=float(t), phase=row["phase"], contacts=contacts)
                )
    result = dict(
        schema="real2sim2real.arm-clearance-audit/1",
        by_phase=phases,
        witnesses=witnesses,
        result_sha256=digest(directory / "result.json"),
        states_sha256=report["states_sha256"],
        scene_sha256=report["scene_sha256"],
        physics_rerun=False,
        clearance_basis="conservative world vertical gap above table upper envelope; extends beyond finite footprint",
        contact_force_inferred=False,
        original_acceptance_changed=False,
        implementation_sha256=digest(Path(__file__)),
    )
    output = directory / "table-clearance.json"
    if output.exists():
        raise FileExistsError("Preserve existing clearance audit")
    write_json(output, result)
    return result


def main():
    import sys
    from real2sim2real.bounded_process import run

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    if not a.worker:
        return run(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--directory",
                str(a.directory.resolve()),
                "--worker",
            ],
            timeout=120,
        )
    audit(a.directory)


if __name__ == "__main__":
    main()
