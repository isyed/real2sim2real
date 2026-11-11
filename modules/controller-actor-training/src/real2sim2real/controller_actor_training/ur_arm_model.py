"""Expand an explicit official UR description checkout into collision MJCF.

Xacro and MuJoCo are optional dependencies. This is an offline model import,
not a ROS driver, a calibrated installation or a validated torque controller.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

from real2sim2real.artifacts import digest, write_json


def prepare(description, variant, output, *, joint_limit_overrides=None):
    import mujoco
    import xacro

    description, output = Path(description).resolve(), Path(output).resolve()
    if not variant.isalnum():
        raise ValueError("A plain UR variant name is required")
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    # Resolve this package's two-file geometry macro without installing ROS.
    for relative in ("urdf/ur_macro.xacro", "urdf/inc/ur_common.xacro"):
        source = description / relative
        sources[str(source)] = digest(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            source.read_text().replace("$(find ur_description)", str(output))
        )
    root = ET.Element(
        "robot", name=variant, attrib={"xmlns:xacro": "http://www.ros.org/wiki/xacro"}
    )
    ET.SubElement(root, "xacro:include", filename=str(output / "urdf/ur_macro.xacro"))
    ET.SubElement(root, "link", name="world")
    attributes = dict(name=variant, tf_prefix="", parent="world", smart_panel="false")
    for option, filename in [
        ("joint_limits", "joint_limits"),
        ("kinematics", "default_kinematics"),
        ("physical", "physical_parameters"),
        ("visual", "visual_parameters"),
    ]:
        path = description / "config" / variant / (filename + ".yaml")
        sources[str(path)] = digest(path)
        attributes[option + "_parameters_file"] = str(path)
    invocation = ET.SubElement(root, "xacro:ur_robot", attributes)
    ET.SubElement(invocation, "origin", xyz="0 0 0", rpy="0 0 0")
    wrapper = output / "wrapper.xacro"
    ET.ElementTree(root).write(wrapper)
    expanded = ET.fromstring(xacro.process_file(str(wrapper)).toxml())
    for link in expanded.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
    for mesh in expanded.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename.startswith("package://ur_description/"):
            raise ValueError("Unexpected mesh package")
        path = description / filename.removeprefix("package://ur_description/")
        sources[str(path)] = digest(path)
        mesh.set("filename", str(path))
    for name, limits in (joint_limit_overrides or {}).items():
        joint = expanded.find(f"joint[@name='{name}']")
        if joint is None:
            raise ValueError("Unknown joint limit override")
        joint.find("limit").attrib.update({k: str(v) for k, v in limits.items()})
    compiler = ET.SubElement(ET.SubElement(expanded, "mujoco"), "compiler")
    compiler.attrib.update(fusestatic="false", strippath="false", balanceinertia="true")
    urdf = output / "collision.urdf"
    ET.ElementTree(expanded).write(urdf)
    model = mujoco.MjModel.from_xml_path(str(urdf))
    mjcf = output / "arm.xml"
    mujoco.mj_saveLastXML(str(mjcf), model)
    receipt = dict(
        schema="real2sim2real.ur-arm-import/1",
        variant=variant,
        sources=sources,
        urdf_sha256=digest(urdf),
        mjcf_sha256=digest(mjcf),
        joint_limit_overrides=joint_limit_overrides or {},
        implementation_sha256=digest(Path(__file__)),
        dynamics_validated=False,
    )
    write_json(output / "import.json", receipt)
    return receipt
