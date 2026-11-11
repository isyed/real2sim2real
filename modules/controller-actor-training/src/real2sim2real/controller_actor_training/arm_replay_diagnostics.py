"""Explicit contact isolation and Cartesian tracking for articulated replays."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .arm_path_audit import joint_neighbors


def configure_contacts(
    root,
    model,
    *,
    condition,
    arm_bodies,
    hand_geoms,
    object_geoms,
    exclude_adjacent=False,
):
    """Record every excluded body pair; retain nonadjacent/environment contacts."""
    if condition not in ("loaded", "no_hand_object_contact"):
        raise ValueError("Unknown contact diagnostic condition")
    pairs = {}
    if condition == "no_hand_object_contact":
        for first in {int(model.geom_bodyid[g]) for g in hand_geoms}:
            for second in {int(model.geom_bodyid[g]) for g in object_geoms}:
                pairs[tuple(sorted((first, second)))] = (
                    "diagnostic_hand_object_isolation"
                )
    if exclude_adjacent:
        for first in sorted(arm_bodies):
            for second in sorted(arm_bodies):
                if first < second and joint_neighbors(model, first, second):
                    pairs[first, second] = "adjacent_arm_links"
    result = []
    contact = root.find("contact")
    if pairs and contact is None:
        contact = ET.SubElement(root, "contact")
    existing = {
        frozenset((e.get("body1"), e.get("body2")))
        for e in root.findall("contact/exclude")
    }
    for (first, second), reason in sorted(pairs.items()):
        names = [model.body(first).name, model.body(second).name]
        if first == 0 or second == 0 or first == second or not all(names):
            raise ValueError("Cannot exclude unnamed, world or same-body pairs")
        if frozenset(names) not in existing:
            ET.SubElement(contact, "exclude", body1=names[0], body2=names[1])
            existing.add(frozenset(names))
        result.append(dict(bodies=names, reason=reason))
    return result


def wrist_errors(actual, desired, roots):
    return {
        name: {
            "position_error_m": float(
                np.linalg.norm(actual.xpos[body] - desired.xpos[body])
            ),
            "rotation_error_deg": float(
                np.rad2deg(
                    Rotation.from_matrix(
                        actual.xmat[body].reshape(3, 3)
                        @ desired.xmat[body].reshape(3, 3).T
                    ).magnitude()
                )
            ),
        }
        for name, body in roots.items()
    }


def phase_maxima(records, field):
    """Report the same named scalar tracking errors separately for each phase."""
    if not records:
        return {}
    example = records[0][field]
    return {
        phase: {
            name: {
                key: max(r[field][name][key] for r in records if r["phase"] == phase)
                for key in values
            }
            for name, values in example.items()
        }
        for phase in sorted({r["phase"] for r in records})
    }
