"""Nonperturbing object/surface geometry and whole-environment support observations."""

import numpy as np


def _fixed(model, body):
    while body:
        if model.body_jntnum[body]:
            return False
        body = int(model.body_parentid[body])
    return True


def _projection(model, data, geom, axis):
    """Exact support interval for supported convex collision geometry."""
    import mujoco

    local_axis = data.geom_xmat[geom].reshape(3, 3).T @ axis
    center = float(data.geom_xpos[geom] @ axis)
    kind = model.geom_type[geom]
    size = model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        values = model.mesh_vert[start : start + count] @ local_axis
        return center + float(values.min()), center + float(values.max())
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = float(np.abs(local_axis) @ size)
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = float(size[0])
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = float(size[0] + size[1] * abs(local_axis[2]))
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = float(
            size[0] * np.linalg.norm(local_axis[:2]) + size[1] * abs(local_axis[2])
        )
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = float(np.linalg.norm(size * local_axis))
    else:
        raise ValueError("Unsupported carried collision geometry")
    return center - extent, center + extent


def observe_environment(session, surface_geom):
    """Estimate current forces in a scratch copy; never step or forward live data.

    Surface clearance uses all enabled object collision geoms. The signed top-plane
    distance is exact for their primitive/convex geometry, but does not certify a
    swept path or hidden reconstruction. Finite top footprint overlap is explicit.
    Forces are scratch-forward estimates, distinct from cached last-step telemetry.
    Existing configured support_fraction and task acceptance remain unchanged.
    """
    import mujoco

    from .support_control import contact_wrench_on_body

    model = session.model
    surface = model.geom(surface_geom).id
    if (
        not _fixed(model, int(model.geom_bodyid[surface]))
        or int(model.geom_bodyid[surface]) in session.robot
        or not (model.geom_contype[surface] or model.geom_conaffinity[surface])
    ):
        raise ValueError("Enabled fixed-environment surface required")
    kind = model.geom_type[surface]
    if kind not in (
        mujoco.mjtGeom.mjGEOM_BOX,
        mujoco.mjtGeom.mjGEOM_PLANE,
        mujoco.mjtGeom.mjGEOM_MESH,
    ):
        raise ValueError("Box, plane or mesh support surface required")
    gravity = np.asarray(model.opt.gravity)
    gravity_norm = float(np.linalg.norm(gravity))
    if not gravity_norm > 0:
        raise ValueError("Nonzero gravity required to define support")
    up = -gravity / gravity_norm
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, session.data)
    mujoco.mj_forward(model, scratch)
    mesh_surface = kind == mujoco.mjtGeom.mjGEOM_MESH
    rotation = scratch.geom_xmat[surface].reshape(3, 3).copy()
    if mesh_surface:
        # Conservative gravity-aligned upper envelope, not a fitted local plane.
        x = np.eye(3)[int(np.argmin(np.abs(up)))]
        x = x - up * float(x @ up)
        x /= np.linalg.norm(x)
        rotation = np.column_stack([x, np.cross(up, x), up])
    normal = rotation[:, 2].copy()
    if normal @ up < 0:
        normal *= -1
    if normal @ up < 0.5:
        raise ValueError("Surface top is not sufficiently upward-facing")
    top = scratch.geom_xpos[surface].copy()
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        top += normal * model.geom_size[surface, 2]
    if mesh_surface:
        top += normal * (
            _projection(model, scratch, surface, normal)[1] - float(top @ normal)
        )
    objects = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == session.object
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    if not objects:
        raise ValueError("Object requires physical collision geometry")
    heights = [
        (float(_projection(model, scratch, g, normal)[0] - top @ normal), g)
        for g in objects
    ]
    height, witness = min(heights)
    overlap = True
    footprint = []
    for i in [0, 1]:
        values = [_projection(model, scratch, g, rotation[:, i]) for g in objects]
        offset = float(top @ rotation[:, i])
        low, high = (
            min(v[0] for v in values) - offset,
            max(v[1] for v in values) - offset,
        )
        footprint.append([float(low), float(high)])
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            overlap = (
                overlap
                and low <= model.geom_size[surface, i]
                and high >= -model.geom_size[surface, i]
            )
        elif mesh_surface:
            lower, upper = _projection(model, scratch, surface, rotation[:, i])
            overlap = overlap and low <= upper - offset and high >= lower - offset
    environment = {}
    for index, contact in enumerate(scratch.contact):
        geoms = [int(contact.geom1), int(contact.geom2)]
        bodies = [int(model.geom_bodyid[g]) for g in geoms]
        if session.object not in bodies or bodies[0] == bodies[1]:
            continue
        other = geoms[0] if bodies[1] == session.object else geoms[1]
        body = int(model.geom_bodyid[other])
        if body in session.robot or not _fixed(model, body):
            continue
        force, _ = contact_wrench_on_body(model, scratch, index, session.object)
        row = environment.setdefault(
            other,
            {
                "geom_id": other,
                "geom_name": model.geom(other).name or None,
                "force_world_n": np.zeros(3),
                "contacts": 0,
                "maximum_penetration_m": 0.0,
            },
        )
        row["force_world_n"] += force
        row["contacts"] += 1
        row["maximum_penetration_m"] = max(
            row["maximum_penetration_m"], max(0.0, -float(contact.dist))
        )
    rows = []
    for key in sorted(environment):
        row = environment[key]
        row["up_force_n"] = float(row["force_world_n"] @ up)
        row["force_world_n"] = row["force_world_n"].tolist()
        rows.append(row)
    weight = float(model.body_mass[session.object]) * gravity_norm
    total = sum(r["up_force_n"] for r in rows)
    surface_force = environment.get(surface, {}).get("up_force_n", 0.0)
    return {
        "time_s": float(session.data.time),
        "surface_geom": surface_geom,
        "object_top_plane_clearance_m": height,
        "clearance_witness_geom": model.geom(witness).name or f"geom:{witness}",
        "finite_surface": bool(kind != mujoco.mjtGeom.mjGEOM_PLANE),
        "surface_plane_kind": "gravity_aligned_mesh_upper_envelope"
        if mesh_surface
        else "primitive_top",
        "top_footprint_aabb_overlap": bool(overlap),
        "object_top_frame_xy_intervals_m": footprint,
        "surface_up_force_n": surface_force,
        "surface_support_fraction": surface_force / weight,
        "fixed_environment_up_force_n": total,
        "fixed_environment_support_fraction": total / weight,
        "fixed_environment_contacts": rows,
        "physics_steps": 0,
        "live_state_modified": False,
        "force_scope": "Scratch forward estimate at current state; not original last-step force telemetry",
        "geometry_scope": "Enabled conditional collision geometry; top-plane distance (mesh uses conservative highest-vertex envelope) plus footprint AABB overlap; negative mesh clearance does not prove contact; not full surface or swept-path admission",
    }
