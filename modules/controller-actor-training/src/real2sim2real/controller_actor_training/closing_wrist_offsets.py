"""Bounded object-frame wrist preload on the original closure/capture clock."""

import numpy as np
from scipy.spatial.transform import Rotation


def validate(settings):
    if set(settings) != {"vectors_object_m", "rotation_vectors_object_rad"}:
        raise ValueError(
            "Explicit object-frame closure translations and rotations required"
        )
    for key, cap in (
        ("vectors_object_m", 0.006),
        ("rotation_vectors_object_rad", 0.06),
    ):
        if set(settings[key]) != {"right", "left"}:
            raise ValueError("Both wrist roles required")
        for value in settings[key].values():
            v = np.asarray(value, float)
            if (
                v.shape != (3,)
                or not np.isfinite(v).all()
                or np.linalg.norm(v) > cap + 1e-12
            ):
                raise ValueError("Closure offset exceeds 6 mm / 0.06 rad wrist bounds")


def fraction(time_s, rollout, capture_duration_s, release_start_s):
    from .object_grasp_pilot import smoothstep

    approach, close = rollout["approach_s"], rollout["close_s"]
    if (
        not np.isfinite(
            [time_s, approach, close, capture_duration_s, release_start_s]
        ).all()
        or time_s < 0
        or approach <= 0
        or close <= 0
        or not approach + close < capture_duration_s < release_start_s
    ):
        raise ValueError(
            "Finite ordered acquisition, capture and release clocks required"
        )
    acquired = smoothstep(np.clip((time_s - approach) / close, 0, 1))
    recovery = smoothstep(
        np.clip(
            (time_s - capture_duration_s) / (release_start_s - capture_duration_s), 0, 1
        )
    )
    return acquired * (1 - recovery)


def nonzero(settings):
    if settings is None:
        return False
    validate(settings)
    return any(
        np.any(np.asarray(value, float) != 0)
        for roles in settings.values()
        for value in roles.values()
    )


def transform(
    time_s,
    role,
    wrist,
    object_pose,
    settings,
    rollout,
    capture_duration_s,
    release_start_s,
):
    validate(settings)
    a = fraction(time_s, rollout, capture_duration_s, release_start_s)
    result = np.array(wrist, float, copy=True)
    obj = np.asarray(object_pose, float)
    if (
        role not in ("right", "left")
        or result.shape != (4, 4)
        or obj.shape != (4, 4)
        or not np.isfinite(result).all()
        or not np.isfinite(obj).all()
    ):
        raise ValueError("Finite metric object and wrist transforms required")
    for pose in (result, obj):
        r = pose[:3, :3]
        if (
            not np.allclose(r.T @ r, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-7, rtol=0)
            or not np.array_equal(pose[3], [0, 0, 0, 1])
        ):
            raise ValueError("Rigid homogeneous metric transforms required")
    rotation = obj[:3, :3]
    result[:3, 3] += a * rotation @ np.asarray(settings["vectors_object_m"][role])
    result[:3, :3] = (
        Rotation.from_rotvec(
            a * rotation @ np.asarray(settings["rotation_vectors_object_rad"][role])
        ).as_matrix()
        @ result[:3, :3]
    )
    return result
