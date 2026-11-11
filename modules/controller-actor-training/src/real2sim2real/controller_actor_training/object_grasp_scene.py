"""Build a CPU MuJoCo grasp scene with a dynamic object and actuated wrist.

Robot assets and mass/friction assumptions are supplied by the caller. There
are no object actuators, welds, mocap bodies, reference forces or gravity aids.
The Cartesian wrist is a declared fixture, not evidence of arm reachability.
"""

import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from real2sim2real.artifacts import digest, write_json


def numbers(values):
    return " ".join(format(float(x), ".12g") for x in np.asarray(values).ravel())


def quaternion(rotation):
    q = Rotation.from_matrix(rotation).as_quat()
    return q[[3, 0, 1, 2]]


def prepare(config):
    import coacd
    from real2sim2real.retargeting.object_inertia import mass_properties
    from real2sim2real.egocentric.mesh_surface import MeshSurfaceQuery

    task_path = Path(config["task"])
    task = json.loads(task_path.read_text())
    if (
        task["schema"] != "real2sim2real.object-task/1"
        or task["units"] != "metres"
        or task["up_axis"] != "+Z"
    ):
        raise ValueError("A metric upright object task is required")
    mesh_path = task_path.parent / task["object"]["mesh"]
    if digest(mesh_path) != task["object"]["sha256"]:
        raise ValueError("Object mesh differs from the task")
    output = Path(config["output"]) / "scene"
    output.mkdir(parents=True, exist_ok=False)
    original = trimesh.load(mesh_path, force="mesh", process=True)
    if config["collision"].get("microhole_repair") is not None:
        from real2sim2real.retargeting.mesh_microholes import close_microholes

        original, repair = close_microholes(
            original, **config["collision"]["microhole_repair"]
        )
        repair["source_sha256"] = digest(mesh_path)
        write_json(output / "microhole_repair.json", repair)
    inertia = mass_properties(
        original.vertices, original.faces, mass_kg=config["physics"]["mass_kg"]
    )
    original.export(output / "object.obj")
    simplified = original.simplify_quadric_decimation(
        face_count=config["collision"]["face_count"]
    )
    if not simplified.is_watertight:
        raise ValueError("Collision preparation must preserve a closed surface")
    coacd.set_log_level("warn")
    pieces = coacd.run_coacd(
        coacd.Mesh(simplified.vertices, simplified.faces),
        **config["collision"]["coacd"],
    )
    if not pieces or len(pieces) > config["collision"]["max_pieces"]:
        raise ValueError("Collision decomposition exceeds its piece budget")
    meshes = [trimesh.Trimesh(v, f, process=True).convex_hull for v, f in pieces]
    for i, mesh in enumerate(meshes):
        mesh.export(output / f"collision_{i:03d}.obj")
    # Surface-distance diagnostics are independent of subsequent grasp scoring.
    union_surface = trimesh.util.concatenate(meshes)
    a, _ = trimesh.sample.sample_surface(original, 1500, seed=11)
    b, _ = trimesh.sample.sample_surface(union_surface, 1500, seed=12)
    forward = MeshSurfaceQuery(union_surface).closest(a)[1]
    reverse = MeshSurfaceQuery(original).closest(b)[1]
    audit = dict(
        pieces=len(meshes),
        source_volume_m3=float(original.volume),
        sum_piece_volumes_m3=float(sum(x.volume for x in meshes)),
        source_to_proxy_p90_m=float(np.quantile(forward, 0.9)),
        proxy_to_source_p90_m=float(np.quantile(reverse, 0.9)),
        proxy_to_source_max_m=float(reverse.max()),
        approximation="Convex pieces, including internal piece faces; sampled surface distances do not certify topology.",
    )
    audit["accepted"] = (
        audit["source_to_proxy_p90_m"] <= config["collision"]["max_p90_m"]
        and audit["proxy_to_source_p90_m"] <= config["collision"]["max_p90_m"]
        and audit["proxy_to_source_max_m"] <= config["collision"]["max_error_m"]
    )
    write_json(output / "collision_audit.json", audit)
    if not audit["accepted"]:
        raise ValueError("Collision proxy fails the declared surface-distance limits")
    robot = Path(config["robot"]["xml"])
    root = ET.parse(robot).getroot()
    compiler = root.find("compiler")
    meshdir = (robot.parent / compiler.get("meshdir", "")).resolve()
    compiler.set("meshdir", "robot_assets")
    compiler.set("inertiafromgeom", "false")
    asset_dir = output / "robot_assets"
    asset_dir.mkdir()
    sources = {str(task_path): digest(task_path), str(robot): digest(robot)}
    for node in root.findall("asset/mesh"):
        source = meshdir / node.get("file")
        target = asset_dir / source.name
        if target.exists() and digest(target) != digest(source):
            raise ValueError("Conflicting robot mesh basenames")
        shutil.copyfile(source, target)
        sources[str(source)] = digest(source)
        node.set("file", target.name)
    for name in ["LICENSE", "NOTICE"]:
        source = robot.parent / name
        if source.is_file():
            shutil.copyfile(source, asset_dir / name)
    option = ET.SubElement(
        root,
        "option",
        timestep=str(config["physics"]["timestep_s"]),
        gravity="0 0 -9.81",
        integrator="implicitfast",
        cone="elliptic",
        iterations="60",
    )
    ET.SubElement(option, "flag", multiccd="enable")
    asset = root.find("asset")
    ET.SubElement(asset, "mesh", name="task_visual", file="../object.obj")
    for i in range(len(meshes)):
        ET.SubElement(
            asset, "mesh", name=f"task_collision_{i}", file=f"../collision_{i:03d}.obj"
        )
    table = task["table"]
    xy = np.asarray(table["boundary_xy"])
    slab = trimesh.convex.convex_hull(
        np.r_[
            np.c_[xy, np.full(len(xy), table["top_z_m"])],
            np.c_[xy, np.full(len(xy), table["top_z_m"] - table["thickness_m"])],
        ]
    )
    slab.export(output / "table.obj")
    ET.SubElement(asset, "mesh", name="task_table", file="../table.obj")
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="mesh",
        mesh="task_table",
        rgba=".55 .35 .18 1",
        friction=numbers([config["physics"]["friction"], 0.005, 0.0001]),
    )
    hand = world.find("body")
    if (
        hand is None
        or hand.get("name") != config["robot"]["root_body"]
        or root.findall(".//freejoint")
    ):
        raise ValueError("Require the declared single fixed-root hand")
    hand.set("pos", "0 0 0.6")
    for axis, vector in zip("xyz", np.eye(3)):
        ET.SubElement(
            hand,
            "joint",
            name="fixture_" + axis,
            type="slide",
            axis=numbers(vector),
            range="-1 1",
            damping="5",
        )
    for axis, vector in zip("xyz", np.eye(3)):
        ET.SubElement(
            hand,
            "joint",
            name="fixture_r" + axis,
            type="hinge",
            axis=numbers(vector),
            range="-.5 .5",
            damping=".1",
        )
    actuators = root.find("actuator")
    for node in actuators:
        limit = config["robot"]["finger_torque_limit_nm"]
        node.set("forcerange", numbers([-limit, limit]))
        node.set("forcelimited", "true")
    for axis in "xyz":
        limit = config["robot"]["wrist_force_limit_n"]
        ET.SubElement(
            actuators,
            "position",
            name="fixture_" + axis,
            joint="fixture_" + axis,
            kp="2000",
            kv="90",
            ctrlrange="-1 1",
            forcerange=numbers([-limit, limit]),
        )
        limit = config["robot"]["wrist_torque_limit_nm"]
        ET.SubElement(
            actuators,
            "position",
            name="fixture_r" + axis,
            joint="fixture_r" + axis,
            kp="100",
            kv="5",
            ctrlrange="-.5 .5",
            forcerange=numbers([-limit, limit]),
        )
    transform = np.asarray(task["object"]["world_from_object"])
    obj = ET.SubElement(
        world,
        "body",
        name="task_object",
        pos=numbers(transform[:3, 3]),
        quat=numbers(quaternion(transform[:3, :3])),
    )
    ET.SubElement(obj, "freejoint", name="task_object_free")
    tensor = np.asarray(inertia["inertia_kg_m2"])
    ET.SubElement(
        obj,
        "inertial",
        pos=numbers(inertia["center_of_mass_m"]),
        mass=str(inertia["mass_kg"]),
        fullinertia=numbers(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[0, 1],
                tensor[0, 2],
                tensor[1, 2],
            ]
        ),
    )
    ET.SubElement(
        obj,
        "geom",
        name="object_visual",
        type="mesh",
        mesh="task_visual",
        contype="0",
        conaffinity="0",
        group="1",
        rgba=".85 .85 .87 1",
    )
    for i in range(len(meshes)):
        ET.SubElement(
            obj,
            "geom",
            name=f"object_collision_{i}",
            type="mesh",
            mesh=f"task_collision_{i}",
            group="3",
            rgba=".5 .7 .9 .3",
            friction=numbers([config["physics"]["friction"], 0.005, 0.0001]),
            condim="4",
            solref=".005 1",
        )
    for geom in hand.iter("geom"):
        if geom.get("contype", "1") != "0":
            geom.set(
                "friction", numbers([config["physics"]["friction"], 0.005, 0.0001])
            )
    ET.indent(root)
    ET.ElementTree(root).write(output / "scene.xml", encoding="unicode")
    receipt = dict(
        schema="real2sim2real.object-grasp-scene/1",
        sources=sources,
        config=config,
        collision=audit,
        inertia=inertia,
        scene_sha256=digest(output / "scene.xml"),
        assets={p.name: digest(p) for p in output.glob("*.obj")},
        robot_assets={p.name: digest(p) for p in asset_dir.iterdir()},
        object_actuated=False,
        object_assistance=False,
        arm_reachability_validated=False,
        wrist_fixture="Six force-limited actuated joints; robot arm is not modelled",
        human_finger_reference_used=False,
    )
    write_json(output / "receipt.json", receipt)
    return receipt
