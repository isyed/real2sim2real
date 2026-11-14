"""Explicit, topology-checked collision exclusions at robot joint interfaces."""

import xml.etree.ElementTree as ET

from .arm_path_audit import joint_neighbors


def exclude_joint_interface(root, model, body1, body2):
    """Exclude one declared pair, never world or a multi-joint collision.

    The caller must establish that overlap belongs to an intended mechanical
    joint interface. Topology alone does not establish that physical fact.
    The XML and compiled model must describe the same bodies. Recompile after
    adding exclusions; this function does not mutate the running model.
    """
    ids = [int(model.body(name).id) for name in (body1, body2)]
    if ids[0] == ids[1] or not joint_neighbors(model, *ids):
        raise ValueError(
            "Only a named non-world adjacent joint interface may be excluded"
        )
    xml_names = {body.get("name") for body in root.iter("body")}
    if body1 not in xml_names or body2 not in xml_names:
        raise ValueError("Interface bodies must exist in the supplied XML")
    section = root.find("contact")
    if section is None:
        section = ET.SubElement(root, "contact")
    for existing in section.findall("exclude"):
        if {existing.get("body1"), existing.get("body2")} == {body1, body2}:
            return False
    ET.SubElement(section, "exclude", body1=body1, body2=body2)
    return True
