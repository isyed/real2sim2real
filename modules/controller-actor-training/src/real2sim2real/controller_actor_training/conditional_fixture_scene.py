"""Explicit free-object dynamics fork of a static full-hand collision scene."""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from real2sim2real.artifacts import digest


def prepare(
    source,
    output,
    *,
    object_geoms,
    center_of_mass_m,
    mass_kg,
    diagonal_inertia_kg_m2,
    friction,
    finger_torque_nm=2.0,
    wrist_force_n=80.0,
    wrist_torque_nm=8.0,
    timestep_s=0.001,
):
    """Move declared world geoms into one free body without changing their poses.

    Mass, inertia, friction and actuator ceilings are explicit assumptions. All
    obstacle/hand collision shapes remain. Object-attached uncertainty exclusions
    must be declared by the caller; they are not measured physical surfaces.
    Source and output stay distinct. No loaded-retention admission is implied.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    com, inertia = (
        np.asarray(center_of_mass_m, float),
        np.asarray(diagonal_inertia_kg_m2, float),
    )
    vals = np.asarray(
        [
            mass_kg,
            friction,
            finger_torque_nm,
            wrist_force_n,
            wrist_torque_nm,
            timestep_s,
        ],
        float,
    )
    if (
        com.shape != (3,)
        or inertia.shape != (3,)
        or not np.isfinite(com).all()
        or not np.isfinite(inertia).all()
        or not np.isfinite(vals).all()
        or (inertia <= 0).any()
        or any(inertia[i] > inertia.sum() - inertia[i] for i in range(3))
        or not 0.01 <= mass_kg <= 2
        or not 0.05 <= friction <= 2
        or not 0 < finger_torque_nm <= 5
        or not 0 < wrist_force_n <= 200
        or not 0 < wrist_torque_nm <= 20
        or not 0.0001 <= timestep_s <= 0.001
    ):
        raise ValueError("Bounded finite physical assumptions required")
    names = list(object_geoms)
    if not names or len(set(names)) != len(names):
        raise ValueError("Distinct object geom names required")
    tree = ET.parse(source)
    root = tree.getroot()
    world = root.find("worldbody")
    act = root.find("actuator")
    if (
        world is None
        or act is None
        or root.findall(".//freejoint")
        or root.find("keyframe") is not None
    ):
        raise ValueError(
            "Fixed scene with explicit actuators and no keyframes required"
        )
    if root.find("equality") is not None and len(root.find("equality")):
        raise ValueError("Constraint assistance is not allowed")
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Explicit compiler required")
    for attr in ["meshdir", "texturedir"]:
        compiler.set(attr, str((source.parent / compiler.get(attr, "")).resolve()))
    geoms = []
    for name in names:
        geom = next((g for g in world.findall("geom") if g.get("name") == name), None)
        if geom is None:
            raise ValueError("Object geoms must be named world geoms")
        geoms.append(geom)
    nums = lambda x: " ".join(str(float(v)) for v in x)
    body = ET.SubElement(world, "body", name="task_object")
    ET.SubElement(body, "freejoint", name="task_object_free")
    ET.SubElement(
        body, "inertial", pos=nums(com), mass=str(mass_kg), diaginertia=nums(inertia)
    )
    for geom in geoms:
        world.remove(geom)
        body.append(geom)
    for node in act:
        if node.tag != "position" or not node.get("joint"):
            raise ValueError("Explicit position actuators required")
        node.set("forcerange", nums([-finger_torque_nm, finger_torque_nm]))
        node.set("forcelimited", "true")
    for axis in "xyz":
        for prefix, kp, kv, limit, bound in [
            ("fixture_", 2000, 90, wrist_force_n, 1),
            ("fixture_r", 100, 5, wrist_torque_nm, 0.5),
        ]:
            name = prefix + axis
            if not any(j.get("name") == name for j in root.findall(".//joint")):
                raise ValueError("Six existing fixture joints required")
            if any(a.get("joint") == name for a in act):
                raise ValueError("Fixture already actuated")
            ET.SubElement(
                act,
                "position",
                name=name + "_probe",
                joint=name,
                kp=str(kp),
                kv=str(kv),
                ctrlrange=nums([-bound, bound]),
                ctrllimited="true",
                forcerange=nums([-limit, limit]),
                forcelimited="true",
            )
    for geom in root.findall(".//geom"):
        geom.set("friction", nums([friction, 0.005, 0.0001]))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(timestep_s))
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting a source or prior dynamics scene.
    with output.open("xb") as stream:
        tree.write(stream)
    return dict(
        source_sha256=digest(source),
        scene_sha256=digest(output),
        object_geoms=names,
        mass_kg=mass_kg,
        center_of_mass_m=com.tolist(),
        diagonal_inertia_kg_m2=inertia.tolist(),
        friction=friction,
        finger_torque_nm=finger_torque_nm,
        wrist_force_n=wrist_force_n,
        wrist_torque_nm=wrist_torque_nm,
        timestep_s=timestep_s,
        classification="conditional dynamics assumptions, not verified reconstruction",
    )


def object_obstacle_preflight(model, data, obstacle_geoms, *, max_penetration_m=0.003):
    """Audit free-object/environment overlaps separately from hand-path checks.

    MuJoCo omits fixed/fixed contact pairs. Turning a planning exclusion into a
    physical obstacle can therefore expose collisions invisible to hand-only
    checks. Must run after positioning the scene and before the first step.
    """
    import mujoco

    if not 0 < max_penetration_m <= 0.01:
        raise ValueError("Bounded penetration limit required")
    obj = model.body("task_object").id
    names = list(obstacle_geoms)
    if not names or len(set(names)) != len(names):
        raise ValueError("Distinct obstacle names required")
    obstacles = {model.geom(n).id for n in names}
    if any(model.geom_bodyid[g] == obj for g in obstacles):
        raise ValueError("Obstacle cannot belong to the object")
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    mujoco.mj_fwdPosition(model, scratch)
    contacts = []
    for con in scratch.contact:
        a, b = int(con.geom1), int(con.geom2)
        if (
            (model.geom_bodyid[a] == obj and b in obstacles)
            or (model.geom_bodyid[b] == obj and a in obstacles)
        ) and con.dist < 0:
            contacts.append(
                dict(
                    geoms=[model.geom(a).name, model.geom(b).name],
                    penetration_m=float(-con.dist),
                    position_m=np.asarray(con.pos).tolist(),
                )
            )
    peak = max((r["penetration_m"] for r in contacts), default=0.0)
    return dict(
        passed=bool(peak <= max_penetration_m),
        maximum_penetration_m=peak,
        limit_m=max_penetration_m,
        contacts=contacts,
        scope="initial free-object versus declared obstacles; not grasp retention",
    )
