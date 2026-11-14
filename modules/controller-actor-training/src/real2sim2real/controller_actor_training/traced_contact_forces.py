"""Read-only force accounting from saved solved contacts, without re-solving physics."""

import numpy as np


def grouped_object_forces(contacts, object_geoms, groups):
    """Resolve recorded contact-frame forces onto an object in world coordinates.

    A contact frame's first row points from geom1 to geom2; its recorded wrench
    acts on geom2. Groups explicitly name counterpart geoms. Unmapped counterparts
    are retained under `unclassified`, never interpreted as no support. This
    reports net forces and contact counts, not force closure or task acceptance.
    Caller must align cached contacts with their pre-integration state/time.
    """
    objects = set(object_geoms)
    if not objects or "" in objects or "unclassified" in groups:
        raise ValueError("Named object geoms and nonreserved group labels required")
    owner = {}
    for label, names in groups.items():
        for name in names:
            if not name or name in objects or name in owner:
                raise ValueError("Distinct named counterpart geoms required")
            owner[name] = label
    rows = {
        name: {"force_world_n": np.zeros(3), "contacts": 0, "counterparts": set()}
        for name in [*groups, "unclassified"]
    }
    for contact in contacts:
        pair = contact["geoms"]
        if len(pair) != 2:
            raise ValueError("Two contact geom names required")
        membership = [name in objects for name in pair]
        if sum(membership) != 1:
            continue
        frame = np.asarray(contact["frame"], float)
        wrench = np.asarray(contact["wrench_contact_frame"], float)
        if (
            frame.size != 9
            or wrench.shape != (6,)
            or not np.isfinite(frame).all()
            or not np.isfinite(wrench).all()
        ):
            raise ValueError("Finite contact frame and six-component wrench required")
        frame = frame.reshape(3, 3)
        if not np.allclose(frame @ frame.T, np.eye(3), atol=1e-6):
            raise ValueError("Orthonormal world contact frame required")
        other = pair[0] if membership[1] else pair[1]
        sign = 1 if membership[1] else -1
        row = rows[owner.get(other, "unclassified")]
        row["force_world_n"] += sign * frame.T @ wrench[:3]
        row["contacts"] += 1
        row["counterparts"].add(other)
    return {
        name: {
            "force_world_n": row["force_world_n"].tolist(),
            "contacts": row["contacts"],
            "counterparts": sorted(row["counterparts"]),
        }
        for name, row in rows.items()
    }
