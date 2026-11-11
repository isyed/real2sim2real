"""Contact frames on compiled collision pads and bounded position/normal IK."""

import numpy as np
from scipy.optimize import least_squares, linprog
import trimesh

from .grasp_kinematics import robot_indices, contact_penetrations


def unit(value):
    value = np.asarray(value, float)
    if (
        value.shape != (3,)
        or not np.isfinite(value).all()
        or np.linalg.norm(value) < 1e-9
    ):
        raise ValueError("Finite nonzero 3-vector required")
    return value / np.linalg.norm(value)


def pad_frame(model, root_name, geom_name, outward_body, *, point_body_m=None):
    """Intersect a declared body-frame ray with the actual compiled convex pad.

    Returns a body-local point and outward normal; no new collision geometry is
    created. The supplied direction is an embodiment assumption, recorded here.
    """
    import mujoco

    idx = robot_indices(model, root_name)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid not in idx["geoms"] or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError("An owned collidable mesh pad is required")
    hull = collision_pad_hull(model, gid)
    direction = unit(outward_body)
    if point_body_m is not None:
        point = np.asarray(point_body_m, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Finite body-local pad patch point required")
        near, distance, faces = trimesh.proximity.closest_point(hull, [point])
        if distance[0] > 0.001:
            raise ValueError(
                "Declared pad patch exceeds 1 mm collision-hull projection"
            )
        selected_point = near[0]
        normal = hull.face_normals[faces[0]]
        representation = "compiled convex collision pad; explicit surface patch"
    else:
        center = hull.center_mass
        distance = np.linalg.norm(hull.extents) + 0.01
        points, _, faces = hull.ray.intersects_location(
            [center + direction * distance], [-direction], multiple_hits=True
        )
        if not len(points):
            raise ValueError("Declared pad ray misses collision surface")
        selected = np.argmax((points - center) @ direction)
        selected_point = points[selected]
        normal = hull.face_normals[faces[selected]]
        representation = "compiled convex collision pad; ray-selected patch"
    if normal @ direction < 0.5:
        raise ValueError("Pad patch has a grazing or wrong-facing normal")
    return dict(
        geom=geom_name,
        body=model.body(model.geom_bodyid[gid]).name,
        point_body_m=selected_point.tolist(),
        normal_body=normal.tolist(),
        declared_outward_body=direction.tolist(),
        representation=representation,
    )


def collision_pad_hull(model, gid):
    import mujoco

    mid = model.geom_dataid[gid]
    start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
    vertices = (
        model.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T
        + model.geom_pos[gid]
    )
    return trimesh.Trimesh(vertices=vertices, process=False).convex_hull


def contact_patch_samples(
    model, data, frames, world_mesh, *, radius_m=0.004, max_surface_gap_m=0.0005
):
    """Sample actual placed pad hulls; retain only nearby opposing object faces.

    Force budgets must be shared across a pad's samples. No contact plane or
    torsional capacity is invented for parts of the pad missing the object.
    """
    if not 0 < radius_m <= 0.005 or not 0 < max_surface_gap_m <= 0.001:
        raise ValueError("Bounded pad patch radius and surface gap required")
    points, normals, groups, counts = [], [], [], []
    for i, frame in enumerate(frames):
        gid = model.geom(frame["geom"]).id
        body = model.geom_bodyid[gid]
        if model.body(body).name != frame["body"]:
            raise ValueError("Pad frame does not match collision geometry body")
        hull = collision_pad_hull(model, gid)
        n = unit(frame["normal_body"])
        a = unit(np.cross(n, np.eye(3)[np.argmin(abs(n))]))
        b = np.cross(n, a)
        center = np.asarray(frame["point_body_m"])
        count = 0
        for offset in [
            np.zeros(3),
            radius_m * a,
            -radius_m * a,
            radius_m * b,
            -radius_m * b,
        ]:
            hits, _, faces = hull.ray.intersects_location(
                [center + offset + 0.05 * n], [-n], multiple_hits=True
            )
            if not len(hits):
                continue
            pick = np.argmax(hits @ n)
            rotation = data.xmat[body].reshape(3, 3)
            p = data.xpos[body] + rotation @ hits[pick]
            pn = rotation @ hull.face_normals[faces[pick]]
            near, gap, face = trimesh.proximity.closest_point(world_mesh, [p])
            on = world_mesh.face_normals[face[0]]
            if gap[0] <= max_surface_gap_m and pn @ -on >= np.cos(np.deg2rad(25)):
                points.append(near[0])
                normals.append(on)
                groups.append(i)
                count += 1
        counts.append(count)
    return dict(
        points_m=np.asarray(points).reshape(-1, 3).tolist(),
        normals=np.asarray(normals).reshape(-1, 3).tolist(),
        groups=groups,
        samples_per_pad=counts,
        radius_m=radius_m,
        max_surface_gap_m=max_surface_gap_m,
        all_pads_observed=all(counts),
        physical_contact_validated=False,
    )


def observe_frames(model, data, frames, indices, *, jacobian=False):
    import mujoco

    points, normals, jp, jn = [], [], [], []
    for frame in frames:
        body = model.body(frame["body"]).id
        if body not in indices["bodies"]:
            raise ValueError("Contact frame must belong to the robot")
        r = data.xmat[body].reshape(3, 3)
        p = data.xpos[body] + r @ np.asarray(frame["point_body_m"])
        n = r @ unit(frame["normal_body"])
        points.append(p)
        normals.append(n)
        if jacobian:
            a, b = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, a, b, p, body)
            jp.append(a[:, indices["dofs"]])
            jn.append(np.cross(b[:, indices["dofs"]].T, n).T)
    result = (np.asarray(points), np.asarray(normals))
    return (*result, np.asarray(jp), np.asarray(jn)) if jacobian else result


def fit_pad_contacts(
    model,
    data,
    root_name,
    frames,
    targets,
    normals,
    initial,
    *,
    max_nfev=160,
    joint_bounds=None,
    fixture_prefix="fixture_",
    position_scale_m=0.006,
):
    """Fit pad points AND normals in scratch data; leave the caller state intact."""
    import mujoco

    roots = [root_name] if isinstance(root_name, str) else list(root_name)
    if not 1 <= len(roots) <= 2 or len(set(roots)) != len(roots):
        raise ValueError("One or two distinct hand roots required")
    groups = [robot_indices(model, root) for root in roots]
    if len(groups) == 2 and groups[0]["bodies"] & groups[1]["bodies"]:
        raise ValueError("Jointly fitted hands must have disjoint body subtrees")
    idx = (
        groups[0]
        if len(groups) == 1
        else dict(
            joints=np.concatenate([g["joints"] for g in groups]),
            q=np.concatenate([g["q"] for g in groups]),
            dofs=np.concatenate([g["dofs"] for g in groups]),
            geoms=np.concatenate([g["geoms"] for g in groups]),
            bodies=set.union(*(g["bodies"] for g in groups)),
        )
    )
    prefixes = (
        [fixture_prefix] if isinstance(fixture_prefix, str) else list(fixture_prefix)
    )
    if len(prefixes) != len(roots) or any(
        not isinstance(p, str) or not p for p in prefixes
    ):
        raise ValueError("One named fixture prefix per hand required")
    targets, normals = np.asarray(targets, float), np.asarray(normals, float)
    if (
        not 1 <= len(frames) <= 5 * len(roots)
        or targets.shape != (len(frames), 3)
        or normals.shape != targets.shape
        or not np.isfinite(targets).all()
        or not np.isfinite(normals).all()
        or not np.allclose(np.linalg.norm(normals, axis=1), 1)
        or not 1 <= max_nfev <= 400
        or not 0.0005 <= position_scale_m <= 0.006
    ):
        raise ValueError("Bounded finite pad targets and unit normals required")
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    lower, upper = model.jnt_range[idx["joints"]].T.copy()
    names = [model.joint(int(j)).name for j in idx["joints"]]
    for i, name in enumerate(names):
        prefix = next((p for p in prefixes if name.startswith(p)), None)
        if prefix is not None:
            bound = 0.35 if name.startswith(prefix + "r") else 0.06
            lower[i], upper[i] = max(lower[i], -bound), min(upper[i], bound)
    for name, pair in (joint_bounds or {}).items():
        if name not in names:
            raise ValueError("Bounds must name owned joints")
        i = names.index(name)
        pair = np.asarray(pair, float)
        if (
            pair.shape != (2,)
            or not np.isfinite(pair).all()
            or pair[0] >= pair[1]
            or pair[0] < lower[i]
            or pair[1] > upper[i]
        ):
            raise ValueError("Bounds must narrow physical and planning limits")
        lower[i], upper[i] = pair
    initial = np.asarray(initial, float)
    if initial.shape != lower.shape or not np.isfinite(initial).all():
        raise ValueError("Finite owned joint seed required")
    seed = np.clip(initial, lower + 1e-8, upper - 1e-8)
    cache = {}

    def compute(q):
        if "q" in cache and np.array_equal(q, cache["q"]):
            return cache["r"], cache["j"]
        scratch.qpos[idx["q"]] = q
        mujoco.mj_fwdPosition(model, scratch)
        p, n, a, b = observe_frames(model, scratch, frames, idx, jacobian=True)
        penetration, jac = contact_penetrations(model, scratch, idx, jacobian=True)
        residual = np.r_[
            (p - targets).ravel() / position_scale_m,
            (n - normals).ravel() / 0.35,
            (q - seed) * 0.03,
            np.maximum(penetration - 0.001, 0) / 0.002,
        ]
        derivative = np.vstack(
            [
                a.reshape(-1, len(q)) / position_scale_m,
                b.reshape(-1, len(q)) / 0.35,
                np.eye(len(q)) * 0.03,
                jac * (penetration > 0.001)[:, None] / 0.002,
            ]
        )
        cache.update(q=q.copy(), r=residual, j=derivative)
        return residual, derivative

    fitted = least_squares(
        lambda q: compute(q)[0],
        seed,
        jac=lambda q: compute(q)[1],
        bounds=(lower, upper),
        max_nfev=max_nfev,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-5,
    )
    compute(fitted.x)
    p, n = observe_frames(model, scratch, frames, idx)
    return dict(
        q=fitted.x,
        site_errors_m=np.linalg.norm(p - targets, axis=1),
        normal_errors_deg=np.rad2deg(np.arccos(np.clip((n * normals).sum(1), -1, 1))),
        penetration_m=contact_penetrations(model, scratch, idx),
        nfev=fitted.nfev,
        converged=bool(fitted.success),
        objective=float(fitted.cost),
    )


def gravity_wrench_screen(
    points,
    outward_normals,
    center_of_mass,
    mass_kg,
    friction,
    max_normal_force_n,
    *,
    contact_groups=None,
):
    """Conservative static gravity balance using an inscribed friction pyramid.

    Explicit force ceilings are assumptions, not actuator/Jacobian feasibility.
    No claim of full force closure or dynamic grasp success is made.
    """
    p, n, com = map(
        lambda x: np.asarray(x, float), (points, outward_normals, center_of_mass)
    )
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or n.shape != p.shape
        or len(p) < 2
        or com.shape != (3,)
        or not all(np.isfinite(x).all() for x in (p, n, com))
        or not np.allclose(np.linalg.norm(n, axis=1), 1)
        or not np.isfinite([mass_kg, friction, max_normal_force_n]).all()
        or mass_kg <= 0
        or friction < 0
        or max_normal_force_n <= 0
    ):
        raise ValueError("Finite metric contacts and positive mass/force required")
    labels = np.arange(len(p)) if contact_groups is None else np.asarray(contact_groups)
    if (
        labels.shape != (len(p),)
        or labels.dtype.kind not in "iu"
        or set(labels.tolist()) != set(range(int(labels.max()) + 1))
    ):
        raise ValueError("Contiguous nonnegative integer contact groups required")
    columns, groups = [], np.zeros((int(labels.max()) + 1, len(p) * 8))
    for i, (point, normal) in enumerate(zip(p, n)):
        tangent = unit(np.cross(normal, np.eye(3)[np.argmin(abs(normal))]))
        other = np.cross(normal, tangent)
        for j, angle in enumerate(np.arange(8) * np.pi / 4):
            force = -normal + friction * (
                np.cos(angle) * tangent + np.sin(angle) * other
            )
            columns.append(np.r_[force, np.cross(point - com, force)])
            groups[labels[i], i * 8 + j] = 1
    matrix = np.asarray(columns).T
    target = np.array([0, 0, mass_kg * 9.81, 0, 0, 0])
    result = linprog(
        np.ones(len(columns)),
        A_ub=groups,
        b_ub=np.full(len(groups), max_normal_force_n),
        A_eq=matrix,
        b_eq=target,
        bounds=(0, None),
        method="highs",
    )
    return dict(
        feasible=bool(result.success),
        friction=float(friction),
        max_normal_force_n=float(max_normal_force_n),
        mass_kg=float(mass_kg),
        required_normal_forces_n=(groups @ result.x).tolist()
        if result.success
        else None,
        actuator_feasibility_validated=False,
        dynamic_success_validated=False,
    )
