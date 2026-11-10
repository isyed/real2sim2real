"""Bounded contact-target IK using MuJoCo Jacobians and collision witnesses."""

import numpy as np
from scipy.optimize import least_squares


def robot_indices(model, root_name):
    import mujoco

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root < 1:
        raise ValueError("Unknown robot root")
    bodies = {root}
    for i in range(root + 1, model.nbody):
        if model.body_parentid[i] in bodies:
            bodies.add(i)
    joints = np.array([i for i in range(model.njnt) if model.jnt_bodyid[i] in bodies])
    if not len(joints) or (model.jnt_type[joints] < 2).any():
        raise ValueError("Grasp IK requires scalar robot joints")
    geoms = np.array(
        [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] in bodies
            and (model.geom_contype[i] or model.geom_conaffinity[i])
        ]
    )
    return dict(
        root=root,
        joints=joints,
        q=model.jnt_qposadr[joints],
        dofs=model.jnt_dofadr[joints],
        geoms=geoms,
        bodies=bodies,
    )


def contact_penetrations(model, data, indices, *, jacobian=False):
    """Maximum penetration per robot geom, including self and tabletop contacts."""
    import mujoco

    values = np.zeros(len(indices["geoms"]))
    jac = np.zeros((len(values), len(indices["dofs"])))
    lookup = {int(g): i for i, g in enumerate(indices["geoms"])}
    for c in data.contact:
        if c.dist >= 0:
            continue
        for g in (c.geom1, c.geom2):
            if g not in lookup or -c.dist <= values[lookup[g]]:
                continue
            i = lookup[g]
            values[i] = -c.dist
            if jacobian:
                ja, jb = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jac(
                    model, data, ja, None, c.pos, int(model.geom_bodyid[c.geom1])
                )
                mujoco.mj_jac(
                    model, data, jb, None, c.pos, int(model.geom_bodyid[c.geom2])
                )
                jac[i] = -(np.asarray(c.frame).reshape(3, 3)[0] @ (jb - ja))[
                    indices["dofs"]
                ]
    return (values, jac) if jacobian else values


def fit_contacts(
    model,
    data,
    root_name,
    site_names,
    targets,
    initial,
    *,
    max_nfev=160,
    penetration_tolerance_m=0.001,
    fixture_prefix="fixture_",
    joint_bounds=None,
):
    import mujoco

    idx = robot_indices(model, root_name)
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in site_names]
    if min(sites) < 0 or np.asarray(targets).shape != (len(sites), 3):
        raise ValueError("Declared sites and metric targets required")
    lower, upper = model.jnt_range[idx["joints"]].T.copy()
    weights = np.full(len(initial), 0.03)
    for i, j in enumerate(idx["joints"]):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
        if name.startswith(fixture_prefix):
            limit = 0.45 if name.startswith(fixture_prefix + "r") else 0.06
            lower[i], upper[i], weights[i] = -limit, limit, 0.05 / limit
    # Optional posture search may reserve travel, never enlarge physical or IK
    # bounds. Keeping the default unchanged preserves existing saved plans.
    names = [model.joint(int(j)).name for j in idx["joints"]]
    for name, bounds in (joint_bounds or {}).items():
        if name not in names:
            raise ValueError("Posture bound must name a robot-owned joint")
        i = names.index(name)
        values = np.asarray(bounds, float)
        physical = model.jnt_range[idx["joints"][i]]
        if (
            values.shape != (2,)
            or not np.isfinite(values).all()
            or values[0] >= values[1]
            or values[0] < max(lower[i], physical[0])
            or values[1] > min(upper[i], physical[1])
        ):
            raise ValueError("Posture bounds must narrow existing joint limits")
        lower[i], upper[i] = values
    initial = np.clip(initial, lower + 1e-7, upper - 1e-7)
    cache = {}

    def compute(q):
        if "q" in cache and np.array_equal(q, cache["q"]):
            return cache["r"], cache["j"]
        data.qpos[idx["q"]] = q
        # IK needs positions, Jacobians and collision witnesses, not a force
        # solve on intermediate (possibly deeply intersecting) hypotheses.
        mujoco.mj_fwdPosition(model, data)
        r = [(data.site_xpos[sites] - targets).ravel() / 0.006, (q - initial) * weights]
        rows = []
        for site in sites:
            jp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, None, site)
            rows.append(jp[:, idx["dofs"]] / 0.006)
        rows.append(np.diag(weights))
        penetration, jac = contact_penetrations(model, data, idx, jacobian=True)
        active = penetration > penetration_tolerance_m
        r.append(np.maximum(penetration - penetration_tolerance_m, 0) / 0.002)
        rows.append(jac * active[:, None] / 0.002)
        cache.update(q=q.copy(), r=np.concatenate(r), j=np.vstack(rows))
        return cache["r"], cache["j"]

    fit = least_squares(
        lambda q: compute(q)[0],
        initial,
        jac=lambda q: compute(q)[1],
        bounds=(lower, upper),
        max_nfev=max_nfev,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-5,
    )
    compute(fit.x)
    return dict(
        q=fit.x,
        site_errors_m=np.linalg.norm(data.site_xpos[sites] - targets, axis=1),
        penetration_m=contact_penetrations(model, data, idx),
        nfev=fit.nfev,
        objective=float(fit.cost),
        converged=bool(fit.success),
    )
