"""Multi-seed endpoint IK with explicit arm collision and margin checks.

An endpoint pass says nothing about connecting paths, velocity or loaded motion.
"""

import numpy as np

from .arm_path_audit import fit_pose, joint_neighbors


def screen(
    model,
    data,
    root,
    joints,
    target,
    seeds,
    collision_bodies,
    criteria,
    *,
    max_nfev=100,
):
    import mujoco

    target = np.asarray(target, float)
    seeds = np.asarray(seeds, float)
    required = {
        "position_error_m",
        "rotation_error_deg",
        "joint_margin_rad",
        "arm_penetration_m",
    }
    if (
        target.shape != (4, 4)
        or not np.isfinite(target).all()
        or not np.allclose(target[3], [0, 0, 0, 1])
        or not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-7)
        or not np.isclose(np.linalg.det(target[:3, :3]), 1)
        or seeds.ndim != 2
        or seeds.shape[1] != len(joints)
        or not 1 <= len(seeds) <= 8
        or not np.isfinite(seeds).all()
        or not 1 <= max_nfev <= 200
        or set(criteria) != required
        or not all(np.isfinite(v) and v >= 0 for v in criteria.values())
        or criteria["position_error_m"] == 0
        or criteria["rotation_error_deg"] == 0
    ):
        raise ValueError(
            "Rigid pose, bounded seeds/budget and explicit criteria required"
        )
    bids = {model.body(n).id for n in collision_bodies}
    if not bids or 0 in bids:
        raise ValueError("Explicit robot collision bodies required")
    ids = [model.joint(n).id for n in joints]
    lo, hi = model.jnt_range[ids].T
    rows = []
    for seed in seeds:
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        q, pe, re = fit_pose(
            model, scratch, model.body(root).id, joints, target, seed, max_nfev
        )
        mujoco.mj_fwdPosition(model, scratch)
        contacts = []
        for co in scratch.contact:
            gs = [int(co.geom1), int(co.geom2)]
            bs = [int(model.geom_bodyid[g]) for g in gs]
            if co.dist < 0 and any(b in bids for b in bs):
                contacts.append(
                    {
                        "geoms": [model.geom(g).name for g in gs],
                        "geom_ids": gs,
                        "bodies": [model.body(b).name for b in bs],
                        "penetration_m": float(-co.dist),
                        "joint_neighbors": joint_neighbors(model, *bs),
                    }
                )
        depth = max(
            [c["penetration_m"] for c in contacts if not c["joint_neighbors"]],
            default=0,
        )
        margin = float(np.minimum(q - lo, hi - q).min())
        angle = float(np.rad2deg(re))
        passed = (
            pe <= criteria["position_error_m"]
            and angle <= criteria["rotation_error_deg"]
            and margin >= criteria["joint_margin_rad"]
            and depth <= criteria["arm_penetration_m"]
        )
        rows.append(
            {
                "q": q.tolist(),
                "position_error_m": pe,
                "rotation_error_deg": angle,
                "joint_margin_rad": margin,
                "arm_penetration_m": depth,
                "contacts": contacts,
                "endpoint_pass": bool(passed),
            }
        )
    best = min(
        range(len(rows)),
        key=lambda i: (
            not rows[i]["endpoint_pass"],
            rows[i]["position_error_m"] / criteria["position_error_m"]
            + rows[i]["rotation_error_deg"] / criteria["rotation_error_deg"],
        ),
    )
    return {
        "seeds": rows,
        "selected_seed": best,
        "selected": rows[best],
        "path_validated": False,
        "dynamics_validated": False,
    }
