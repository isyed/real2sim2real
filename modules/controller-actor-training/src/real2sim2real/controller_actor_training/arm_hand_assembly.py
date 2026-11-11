"""Compose fixed-base MJCF arms with existing articulated hands for IK audits.

Mount transforms and collision proxies are declared assumptions, not manufactured
adapter specifications. Source hands, task geometry and finger actuators are kept.
"""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from real2sim2real.artifacts import digest, write_json
from .multi_hand_scene import namespace_robot


def absolute_assets(root, source):
    source = Path(source).resolve()
    compiler = root.find("compiler")
    options = {} if compiler is None else compiler.attrib
    hashes = {str(source): digest(source)}
    if root.findall(".//include"):
        raise ValueError("Flatten MJCF includes before assembly")
    if options.get("angle", "degree") != "radian":
        raise ValueError("Assembly inputs must explicitly use radians")
    for element in root.findall("asset/*"):
        if "file" not in element.attrib:
            continue
        directory = options.get(
            "meshdir" if element.tag == "mesh" else "texturedir",
            options.get("assetdir", ""),
        )
        path = (source.parent / directory / element.get("file")).resolve()
        hashes[str(path)] = digest(path)
        if element.tag == "mesh" and "name" not in element.attrib:
            element.set("name", Path(element.get("file")).stem)
        element.set("file", str(path))
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(key, None)
    return hashes


def extract_mounted_arm(source, included_hand, hand_root, output):
    """Keep an upstream arm and adapter, recording its original hand transform.

    For a single explicit body include: removes the included hand subtree and
    its actuators/contacts/assets, retaining the mounting bodies unchanged.
    Upstream keyframes are removed because their joint layouts no longer apply.
    This adapter does not execute upstream code or alter arm kinematics.
    """
    import mujoco
    from scipy.spatial.transform import Rotation

    root = ET.parse(source).getroot()
    included = ET.parse(included_hand).getroot()
    includes = root.findall(".//include")
    if (
        len(includes) != 1
        or (Path(source).parent / includes[0].get("file")).resolve()
        != Path(included_hand).resolve()
    ):
        raise ValueError("Exactly the declared hand include is required")
    parents = {child: node for node in root.iter() for child in node}
    container = parents[includes[0]]
    container.remove(includes[0])
    container.extend(list(included))
    hand = root.find(f".//body[@name='{hand_root}']")
    if hand is None or root.findall(".//include"):
        raise ValueError("Missing hand or nested includes")
    parents = {child: node for node in root.iter() for child in node}
    mount = parents[hand]
    if mount.tag != "body":
        raise ValueError("Hand must attach directly to a named mount body")
    transform = dict(hand.attrib)
    matrix = np.eye(4)
    matrix[:3, 3] = np.fromstring(hand.get("pos", "0 0 0"), sep=" ")
    if any(k in hand.attrib for k in ("axisangle", "xyaxes", "zaxis")):
        raise ValueError("Use a quaternion or Euler hand mounting frame")
    if "euler" in hand.attrib:
        if root.find("compiler").get("angle") != "radian":
            raise ValueError("Explicit radians required")
        quat = np.empty(4)
        mujoco.mju_euler2Quat(
            quat,
            np.fromstring(hand.get("euler"), sep=" "),
            root.find("compiler").get("eulerseq", "xyz"),
        )
    else:
        quat = np.fromstring(hand.get("quat", "1 0 0 0"), sep=" ")
    matrix[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    removed_joints = {j.get("name") for j in hand.findall(".//joint")}
    removed_bodies = {b.get("name") for b in hand.iter("body")}
    mount.remove(hand)
    for section in ("actuator", "contact"):
        node = root.find(section)
        if node is not None:
            for element in list(node):
                if element.get("joint") in removed_joints or any(
                    element.get(k) in removed_bodies for k in ("body1", "body2")
                ):
                    node.remove(element)
    keyframes = root.find("keyframe")
    if keyframes is not None:
        root.remove(keyframes)
    meshes = {g.get("mesh") for g in root.findall("worldbody//geom")}
    assets = root.find("asset")
    for mesh in list(assets.findall("mesh")):
        if mesh.get("name", Path(mesh.get("file")).stem) not in meshes:
            assets.remove(mesh)
    hashes = absolute_assets(root, source)
    hashes[str(Path(included_hand).resolve())] = digest(Path(included_hand))
    ET.ElementTree(root).write(output)
    return dict(
        sources=hashes,
        mount_body=mount.get("name"),
        original_hand_frame=transform,
        mount_from_hand=matrix.tolist(),
        derived_arm_sha256=digest(Path(output)),
    )


def build_assembly(scene, specification, output):
    """Attach each source hand to a separately namespaced arm, removing fixtures.

    Supports flattened fixed-root MJCF arms without equality constraints. Rigid
    mount transforms map hand-root coordinates into the selected arm mount body.
    A cylinder proxy may explicitly represent the otherwise unmodelled adapter.
    """
    import mujoco

    output = Path(output)
    task = ET.parse(scene).getroot()
    hashes = absolute_assets(task, scene)
    receipt = []
    for entry in specification["arms"]:
        arm_path = Path(entry["mjcf"])
        arm = ET.parse(arm_path).getroot()
        hashes.update(absolute_assets(arm, arm_path))
        for tag in ("equality", "sensor", "tendon", "keyframe"):
            node = arm.find(tag)
            if node is not None:
                if len(node):
                    raise ValueError(f"Unsupported arm section: {tag}")
                arm.remove(node)
        if arm.find("default") is None:
            ET.SubElement(arm, "default")
        # Explicitly fill missing collision coverage with user-selected visual
        # meshes. MuJoCo uses their convex hulls; this is an approximation.
        proxies = []
        for name in entry.get("visual_collision_bodies", []):
            body = arm.find(f".//body[@name='{name}']")
            if body is None:
                raise ValueError("Unknown collision proxy body: " + name)
            for geom in list(body.findall("geom")):
                if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                    proxy = deepcopy(geom)
                    proxy.attrib.update(
                        contype="1", conaffinity="1", group="3", density="0"
                    )
                    proxy.set("name", geom.get("name", name) + "_audit_collision")
                    body.append(proxy)
                    proxies.append(proxy.get("name"))
        namespace_robot(arm, entry["namespace"])
        hand = task.find(f"worldbody/body[@name='{entry['hand_root']}']")
        if hand is None:
            raise ValueError("Hand must be an existing top-level body")
        joints = hand.findall("joint")
        if {j.get("name") for j in joints} != set(entry["remove_fixture_joints"]):
            raise ValueError("Explicit fixture joints must match hand-root joints")
        for joint in joints:
            hand.remove(joint)
        actuator = task.find("actuator")
        if actuator is not None:
            for node in list(actuator):
                if node.get("joint") in entry["remove_fixture_joints"]:
                    actuator.remove(node)
        transform = np.asarray(entry["mount_from_hand"], dtype=float)
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(
                transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-7
            )
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1)
        ):
            raise ValueError("Rigid, right-handed mount_from_hand required")
        from scipy.spatial.transform import Rotation

        for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
            hand.attrib.pop(key, None)
        hand.set("pos", " ".join(map(str, transform[:3, 3])))
        hand.set(
            "quat",
            " ".join(
                map(
                    str, Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
                )
            ),
        )
        # Preserve the task defaults for hand descendants after reparenting.
        if "childclass" not in hand.attrib:
            hand.set("childclass", "assembly_source_hand_defaults")
            defaults = task.find("default")
            if defaults is None:
                defaults = ET.SubElement(task, "default")
            if defaults.find("default[@class='assembly_source_hand_defaults']") is None:
                ET.SubElement(
                    defaults, "default", {"class": "assembly_source_hand_defaults"}
                )
        task.find("worldbody").remove(hand)
        mount = arm.find(f".//body[@name='{entry['namespace'] + entry['mount_body']}']")
        if mount is None:
            raise ValueError("Unknown mount body")
        mount.append(hand)
        if "adapter_cylinder" in entry:
            proxy = entry["adapter_cylinder"]
            values = np.asarray([proxy["radius_m"], *proxy["fromto_m"]], float)
            if (
                values.shape != (7,)
                or not np.isfinite(values).all()
                or values[0] <= 0
                or np.linalg.norm(values[1:4] - values[4:7]) <= 0
            ):
                raise ValueError("Finite nonzero adapter cylinder required")
            ET.SubElement(
                mount,
                "geom",
                name=entry["namespace"] + "adapter_proxy",
                type="cylinder",
                size=str(values[0]),
                fromto=" ".join(map(str, values[1:])),
                contype="1",
                conaffinity="1",
                group="3",
                density="0",
                rgba=".8 .5 .1 1",
            )
        for tag in ("asset", "default", "worldbody", "actuator", "contact"):
            node = arm.find(tag)
            if node is None:
                continue
            destination = task.find(tag)
            if destination is None:
                destination = ET.SubElement(task, tag)
            if tag == "default":
                destination.append(node)
            else:
                destination.extend(list(node))
        receipt.append(
            dict(
                hand_root=entry["hand_root"],
                removed_fixture_joints=entry["remove_fixture_joints"],
                assumed_mount_from_hand=entry["mount_from_hand"],
                visual_collision_proxies=proxies,
                adapter_geometry=entry.get("adapter_cylinder"),
                hardware_compatibility_verified=False,
            )
        )
    path = output / "audit-scene.xml"
    ET.ElementTree(task).write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    write_json(
        output / "assembly.json",
        dict(
            schema="real2sim2real.arm-hand-assembly/1",
            arms=receipt,
            sources=hashes,
            scene_sha256=digest(path),
            implementation_sha256=digest(Path(__file__)),
            dynamics_validated=False,
        ),
    )
    return model, hashes
