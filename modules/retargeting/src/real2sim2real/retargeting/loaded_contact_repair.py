"""Bounded posture proposals that preserve pad locations at a loaded snapshot.

This is a geometric proposal, not force control or acceptance. Replay from the
original acquisition state with unchanged physics and all clearance/force gates.
"""

from .collision_reserve import CollisionReserve
from .grasp_kinematics import robot_indices
from .pad_contacts import fit_pad_contacts, observe_frames


def propose(
    model,
    data,
    root,
    frames,
    limits,
    reserve_geoms,
    *,
    position_scale_m=0.001,
    reserve_scale_m=0.001,
    max_nfev=100,
):
    """Fit in scratch state; limits are explicit named command intervals.

    The caller supplies the current object/obstacle poses, including exclusions,
    and command bounds. Pads target their observed loaded positions and normals.
    Actual forces, friction and pressure commands are deliberately not inferred.
    """
    import mujoco

    idx = robot_indices(model, root)
    names = [model.joint(int(j)).name for j in idx["joints"]]
    if set(limits) != set(names):
        raise ValueError("Explicit bounds required for every owned joint")
    if not 0.0005 <= position_scale_m <= 0.003 or not 1 <= max_nfev <= 200:
        raise ValueError("Bounded metric fit and evaluation budget required")
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    mujoco.mj_fwdPosition(model, scratch)
    points, normals = observe_frames(model, scratch, frames, idx)
    initial = scratch.qpos[idx["q"]].copy()
    reserve = CollisionReserve(
        model, scratch, root, reserve_geoms, threshold_m=0, scale_m=reserve_scale_m
    )
    fit = fit_pad_contacts(
        model,
        scratch,
        root,
        frames,
        points,
        normals,
        initial,
        joint_bounds=limits,
        position_scale_m=position_scale_m,
        normal_scale=0.1,
        normal_guard_deg=5,
        collision_reserves=[reserve],
        max_nfev=max_nfev,
    )
    return dict(
        **fit,
        offset=fit["q"] - initial,
        retained_pad_points_m=points,
        retained_pad_normals=normals,
        loaded_retention_validated=False,
    )
