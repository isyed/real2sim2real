"""Shared bounded finger offsets fitted to frozen loaded configurations.

Only position kinematics and collision detection run. A fit is not a prediction
of compliant contact forces, subsequent motion or grasp retention.
"""

import numpy as np
from scipy.optimize import least_squares

from .grasp_kinematics import contact_penetrations, robot_indices
from .pad_contacts import observe_frames


def fit_loaded_pad_offset(
    model,
    root_name,
    frames,
    qpos_samples,
    joint_names,
    target_normals,
    *,
    target_points_m=None,
    command_samples_rad=None,
    maximum_offset_rad=0.02,
    position_scale_m=0.0005,
    normal_scale=0.1,
    penetration_tolerance_m=0.001,
    normal_regression_deg=None,
    max_nfev=80,
):
    """Fit one offset on named hinge joints, preserving other configurations.

    Targets have shape (samples, pads, 3). Omitted point targets preserve each
    snapshot's pad positions. Native bounds intersect over measured snapshots
    and, when supplied, planned command samples in the named-joint ordering.
    Existing penetration is not optimized away by moving unrelated joints; the
    collision penalty discourages worsening it. Absolute admission is separate.
    """
    import mujoco

    indices = robot_indices(model, root_name)
    names = list(joint_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Unique owned hinge-joint names required")
    joints = np.array([model.joint(name).id for name in names])
    if (
        any(j not in indices["joints"] for j in joints)
        or np.any(model.jnt_type[joints] != mujoco.mjtJoint.mjJNT_HINGE)
        or not np.all(model.jnt_limited[joints])
    ):
        raise ValueError("Only owned limited hinge joints can be corrected")
    qadr = model.jnt_qposadr[joints]
    columns = [list(indices["joints"]).index(j) for j in joints]
    qpos = np.asarray(qpos_samples, float)
    normals = np.asarray(target_normals, float)
    if (
        qpos.ndim != 2
        or qpos.shape[1] != model.nq
        or not len(qpos)
        or not np.isfinite(qpos).all()
        or normals.shape != (len(qpos), len(frames), 3)
        or not np.isfinite(normals).all()
        or not np.allclose(np.linalg.norm(normals, axis=-1), 1)
    ):
        raise ValueError("Finite configurations and per-sample unit normals required")
    settings = [
        maximum_offset_rad,
        position_scale_m,
        normal_scale,
        penetration_tolerance_m,
    ]
    if not np.isfinite(settings).all() or not (
        0 < maximum_offset_rad <= 0.05
        and position_scale_m > 0
        and normal_scale > 0
        and penetration_tolerance_m >= 0
        and type(max_nfev) is int
        and 1 <= max_nfev <= 400
    ):
        raise ValueError("Positive finite bounded fitting settings required")
    commands = qpos[:, qadr]
    if command_samples_rad is not None:
        extra = np.asarray(command_samples_rad, float)
        if (
            extra.ndim != 2
            or extra.shape[1] != len(joints)
            or not np.isfinite(extra).all()
        ):
            raise ValueError("Finite command samples in named-joint ordering required")
        commands = np.vstack([commands, extra])
    limits = model.jnt_range[joints]
    lower = np.maximum(-maximum_offset_rad, np.max(limits[:, 0] - commands, axis=0))
    upper = np.minimum(maximum_offset_rad, np.min(limits[:, 1] - commands, axis=0))
    if np.any(lower >= upper):
        raise ValueError("No shared native-limit interval across loaded samples")
    scratch = mujoco.MjData(model)
    observed, observed_normals, baseline_penetration = [], [], []
    for q in qpos:
        scratch.qpos[:] = q
        mujoco.mj_fwdPosition(model, scratch)
        p, n = observe_frames(model, scratch, frames, indices)
        observed.append(p)
        observed_normals.append(n)
        baseline_penetration.append(contact_penetrations(model, scratch, indices))
    points = np.asarray(observed if target_points_m is None else target_points_m, float)
    if points.shape != normals.shape or not np.isfinite(points).all():
        raise ValueError("Finite per-sample pad positions required")
    tolerated = np.maximum(baseline_penetration, penetration_tolerance_m)
    guard_dot = None
    if normal_regression_deg is not None:
        if (
            not np.isfinite(normal_regression_deg)
            or not 0 <= normal_regression_deg <= 45
        ):
            raise ValueError(
                "Finite normal-regression allowance in [0,45] degrees required"
            )
        initial_angle = np.arccos(
            np.clip(np.sum(np.asarray(observed_normals) * normals, axis=-1), -1, 1)
        )
        guard_dot = np.cos(
            np.minimum(np.pi, initial_angle + np.deg2rad(normal_regression_deg))
        )
    cache = {}

    def compute(offset):
        if "offset" in cache and np.array_equal(offset, cache["offset"]):
            return cache["residual"], cache["jacobian"]
        residuals, derivatives = [], []
        for i, q in enumerate(qpos):
            scratch.qpos[:] = q
            scratch.qpos[qadr] += offset
            mujoco.mj_fwdPosition(model, scratch)
            p, n, jp, jn = observe_frames(
                model, scratch, frames, indices, jacobian=True
            )
            penetration, jc = contact_penetrations(
                model, scratch, indices, jacobian=True
            )
            excess = penetration - tolerated[i]
            residuals.extend(
                [
                    (p - points[i]).ravel() / position_scale_m,
                    (n - normals[i]).ravel() / normal_scale,
                    np.maximum(excess, 0) / position_scale_m,
                ]
            )
            derivatives.extend(
                [
                    jp[:, :, columns].reshape(-1, len(joints)) / position_scale_m,
                    jn[:, :, columns].reshape(-1, len(joints)) / normal_scale,
                    jc[:, columns] * (excess > 0)[:, None] / position_scale_m,
                ]
            )
            if guard_dot is not None:
                violation = guard_dot[i] - np.sum(n * normals[i], axis=1)
                residuals.append(np.maximum(violation, 0) / 0.001)
                normal_jac = -np.einsum("ij,ijk->ik", normals[i], jn[:, :, columns])
                derivatives.append(normal_jac * (violation > 0)[:, None] / 0.001)
        residuals.append(offset / 0.1)
        derivatives.append(np.eye(len(joints)) / 0.1)
        cache.update(
            offset=offset.copy(),
            residual=np.concatenate(residuals),
            jacobian=np.vstack(derivatives),
        )
        return cache["residual"], cache["jacobian"]

    fit = least_squares(
        lambda x: compute(x)[0],
        np.clip(np.zeros(len(joints)), lower, upper),
        jac=lambda x: compute(x)[1],
        bounds=(lower, upper),
        max_nfev=max_nfev,
    )
    return {
        "joint_names": names,
        "offsets_rad": fit.x.tolist(),
        "lower_rad": lower.tolist(),
        "upper_rad": upper.tolist(),
        "converged": bool(fit.success),
        "objective": float(fit.cost),
        "nfev": int(fit.nfev),
        "dynamic_retention_validated": False,
    }


def minimum_geom_separation(model, data, first, second, *, cutoff_m=0.02):
    """Signed closest distance, capped above at cutoff, with geometry witness.

    The caller explicitly selects physical obstacles or planning exclusions.
    Collision flags are not changed or interpreted; avoid using adjacent robot
    pairs excluded by the embodiment as physical clearance requirements.
    """
    import mujoco

    if not np.isfinite(cutoff_m) or cutoff_m <= 0:
        raise ValueError("Positive finite distance cutoff required")
    a, b = np.asarray(first, dtype=int), np.asarray(second, dtype=int)
    if (
        a.ndim != 1
        or b.ndim != 1
        or any(np.any((x < 0) | (x >= model.ngeom)) for x in (a, b))
    ):
        raise ValueError("Valid geometry index vectors required")
    best, witness = float(cutoff_m), None
    for g in a:
        # Bounding spheres conservatively skip distant pairs; infinite planes
        # are never filtered by their finite stored radius.
        distance = np.linalg.norm(data.geom_xpos[b] - data.geom_xpos[g], axis=1)
        eligible = distance - model.geom_rbound[b] - model.geom_rbound[g] <= cutoff_m
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            eligible[:] = True
        eligible |= model.geom_type[b] == mujoco.mjtGeom.mjGEOM_PLANE
        for h in b[eligible]:
            if g == h:
                continue
            value = float(
                mujoco.mj_geomDistance(model, data, int(g), int(h), cutoff_m, None)
            )
            if value < best:
                best, witness = (
                    value,
                    [model.geom(int(g)).name, model.geom(int(h)).name],
                )
    return {"distance_m": best, "pair": witness, "capped_at_cutoff": witness is None}
