"""Add a namespaced articulated hand to a pinned, prepared object scene."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .object_grasp_scene import numbers


def namespace_robot(root, prefix):
    """Namespace MJCF names and references before composing flattened robots.

    Unsupported cross-file mechanisms fail closed. Unnamed global defaults are
    scoped to a new childclass so they cannot modify the original robot.
    """
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("Use an alphanumeric MJCF namespace")
    if any(
        root.findall(tag) for tag in ("include", ".//freejoint", "equality", "keyframe")
    ):
        raise ValueError(
            "Require a flattened fixed-root robot without constraints or keyframes"
        )
    if any(j.get("type") == "free" for j in root.findall(".//joint")):
        raise ValueError("Support robot must have a fixed root")
    refs = {
        "name",
        "class",
        "childclass",
        "mesh",
        "material",
        "texture",
        "joint",
        "joint1",
        "joint2",
        "body",
        "body1",
        "body2",
        "geom",
        "geom1",
        "geom2",
        "site",
        "site1",
        "site2",
        "tendon",
        "tendon1",
        "tendon2",
        "actuator",
        "objname",
        "refname",
        "target",
        "cranksite",
        "slidersite",
    }
    for node in root.iter():
        for key in refs & node.attrib.keys():
            node.set(key, prefix + node.get(key))
    defaults = root.find("default")
    if defaults is not None:
        defaults.set("class", prefix + "defaults")
        for body in root.findall("worldbody/body"):
            if "childclass" not in body.attrib:
                body.set("childclass", prefix + "defaults")
    return root


def append_hand(config):
    """Copy checked parent assets; append an independently actuated support hand."""
    from .object_grasp_pilot import load_scene

    parent = config["parent"]
    path = Path(parent["trial"])
    if digest(path) != parent["trial_sha256"]:
        raise ValueError("Pinned baseline trial changed")
    trial = json.loads(path.read_text())
    baseline = trial["configuration"]
    if not trial["success"]:
        raise ValueError("A successful baseline is required")
    load_scene(baseline)
    for key in (
        "task",
        "physics",
        "robot",
        "clearance_feedback",
        "orientation_feedback",
    ):
        if config.get(key) != baseline.get(key):
            raise ValueError("Preserve baseline scene and primary controller")
    return _append_prepared_hand(
        config, baseline, {"parent_trial_sha256": parent["trial_sha256"]}
    )


def append_hypothesis_hand(config):
    """Compose a fresh pair from pinned geometry without inventing a passing grasp.

    This explicit development entry point preserves the old successful-parent
    requirement in append_hand. It never admits observations, control or hardware.
    """
    from .paired_acquisition_audit import pinned_json
    from .object_grasp_pilot import verify_scene_assets

    if config.get("hypothesis_only") is not True or "parent" in config:
        raise ValueError("Explicit fresh hypothesis condition required")
    source = config["prepared_source"]
    baseline = pinned_json(source["config"])
    root, _ = verify_scene_assets(baseline)
    if (
        digest(root / "receipt.json") != source["scene_receipt_sha256"]
        or digest(root.parent / "settling.json") != source["settling_sha256"]
    ):
        raise ValueError("Pinned source scene or initialization changed")
    settling = json.loads((root.parent / "settling.json").read_text())
    if not settling["accepted"] or settling["initialization"] != "gravity_settled_task":
        raise ValueError("Explicit stable gravity-settled development scene required")
    for key in ("task", "physics", "robot"):
        if config[key] != baseline[key]:
            raise ValueError("Preserve source task, physics and primary robot")
    return _append_prepared_hand(
        config,
        baseline,
        {
            "hypothesis_only": True,
            "source_configuration": source["config"],
            "source_settling_sha256": source["settling_sha256"],
            "contact_observation_admitted": False,
            "control_ready": False,
        },
    )


def verify_preserved_dynamics(before, after):
    """Namespaced composition must not retune the original object, table or hand."""
    for key in (
        "timestep",
        "gravity",
        "integrator",
        "solver",
        "iterations",
        "tolerance",
    ):
        if not np.array_equal(getattr(before.opt, key), getattr(after.opt, key)):
            raise ValueError("Composition changed physics option: " + key)
    groups = (
        ("body", before.nbody, ("mass", "inertia", "ipos", "iquat", "gravcomp")),
        (
            "geom",
            before.ngeom,
            (
                "friction",
                "contype",
                "conaffinity",
                "solref",
                "solimp",
                "pos",
                "quat",
                "size",
            ),
        ),
        ("joint", before.njnt, ("range", "axis", "stiffness", "limited")),
        (
            "actuator",
            before.nu,
            (
                "gainprm",
                "biasprm",
                "forcerange",
                "ctrlrange",
                "forcelimited",
                "ctrllimited",
            ),
        ),
    )
    checked = 0
    for kind, count, fields in groups:
        for i in range(count):
            old = getattr(before, kind)(i)
            if not old.name:
                continue
            new = getattr(after, kind)(old.name)
            prefix = "jnt" if kind == "joint" else kind
            for field in fields:
                a = getattr(before, prefix + "_" + field)[old.id]
                b = getattr(after, prefix + "_" + field)[new.id]
                if not np.allclose(a, b, rtol=1e-10, atol=1e-12):
                    raise ValueError(f"Composition changed {kind} {old.name}: {field}")
            checked += 1
    return {"named_entities_checked": checked, "original_dynamics_preserved": True}


def _append_prepared_hand(config, baseline, provenance):
    from .object_grasp_pilot import load_scene

    source = Path(baseline["output"]) / "scene"
    output = Path(config["output"]) / "scene"
    if output.exists():
        raise ValueError("Preserve existing scenes")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)
    root = ET.parse(output / "scene.xml").getroot()
    support = config["support_robot"]
    limits = np.array(
        [
            support[k]
            for k in (
                "finger_torque_limit_nm",
                "wrist_force_limit_n",
                "wrist_torque_limit_nm",
            )
        ]
    )
    if not np.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError("Positive finite support actuator limits required")
    robot_path = Path(support["xml"])
    robot = ET.parse(robot_path).getroot()
    if any(node.tag != "mesh" for node in robot.findall("asset/*")):
        raise ValueError("This composer supports mesh robot assets only")
    compiler = robot.find("compiler")
    if compiler is None or compiler.get("angle", "degree") != "radian":
        raise ValueError("Support robot must use explicit radians")
    meshdir = (robot_path.parent / compiler.get("meshdir", "")).resolve()
    namespace_robot(robot, support["namespace"])
    bodies = robot.findall("worldbody/body")
    if len(bodies) != 1 or bodies[0].get("name") != support["root_body"]:
        raise ValueError("Exactly one declared support hand root required")
    sources = {str(robot_path): digest(robot_path)}
    for mesh in robot.findall("asset/mesh"):
        asset = meshdir / mesh.get("file")
        target = output / "robot_assets" / (support["namespace"] + asset.name)
        if target.exists():
            raise ValueError("Support asset filename collision")
        shutil.copyfile(asset, target)
        mesh.set("file", target.name)
        sources[str(asset)] = digest(asset)
    for name in ("LICENSE", "NOTICE"):
        src = robot_path.parent / name
        if src.is_file():
            shutil.copyfile(
                src, output / "robot_assets" / (support["namespace"] + name)
            )
    defaults = robot.find("default")
    if defaults is not None:
        parent_default = root.find("default")
        if parent_default is None:
            parent_default = ET.SubElement(root, "default")
        parent_default.append(deepcopy(defaults))
    for section in ("asset", "contact", "tendon", "sensor", "actuator"):
        nodes = robot.find(section)
        if nodes is not None:
            destination = root.find(section)
            if destination is None:
                destination = ET.SubElement(root, section)
            destination.extend(deepcopy(list(nodes)))
    hand = bodies[0]
    hand.set("pos", "0 0 0.7")
    root.find("worldbody").append(hand)
    actuators = root.find("actuator")
    for actuator in actuators:
        if actuator.get("name", "").startswith(support["namespace"]):
            limit = support["finger_torque_limit_nm"]
            actuator.set("forcerange", numbers([-limit, limit]))
            actuator.set("forcelimited", "true")
    prefix = support["fixture_prefix"]
    for rotational in (False, True):
        for axis, vector in zip("xyz", np.eye(3)):
            name = prefix + ("r" if rotational else "") + axis
            bounds = "-.5 .5" if rotational else "-1 1"
            ET.SubElement(
                hand,
                "joint",
                name=name,
                type="hinge" if rotational else "slide",
                axis=numbers(vector),
                range=bounds,
                damping=".1" if rotational else "5",
            )
            limit = support[
                "wrist_torque_limit_nm" if rotational else "wrist_force_limit_n"
            ]
            ET.SubElement(
                actuators,
                "position",
                name=name,
                joint=name,
                kp="100" if rotational else "2000",
                kv="5" if rotational else "90",
                ctrlrange=bounds,
                forcerange=numbers([-limit, limit]),
                forcelimited="true",
            )
    for geom in hand.iter("geom"):
        if geom.get("contype", "1") != "0":
            geom.set(
                "friction", numbers([config["physics"]["friction"], 0.005, 0.0001])
            )
    ET.indent(root)
    ET.ElementTree(root).write(output / "scene.xml", encoding="unicode")
    receipt = json.loads((source / "receipt.json").read_text())
    receipt.update(
        schema="real2sim2real.multi-hand-scene/1",
        config=config,
        parent_scene_receipt_sha256=digest(source / "receipt.json"),
        **provenance,
        support_sources=sources,
        scene_sha256=digest(output / "scene.xml"),
        robot_assets={p.name: digest(p) for p in (output / "robot_assets").iterdir()},
    )
    write_json(output / "receipt.json", receipt)
    # Original reconstruction/settled task start remain byte-identical.
    shutil.copyfile(
        Path(baseline["output"]) / "settling.json", output.parent / "settling.json"
    )
    receipt["original_dynamics_verification"] = verify_preserved_dynamics(
        load_scene(baseline)[0], load_scene(config)[0]
    )
    write_json(output / "receipt.json", receipt)
    return receipt
