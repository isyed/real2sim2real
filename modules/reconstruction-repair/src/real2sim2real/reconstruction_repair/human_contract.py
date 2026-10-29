"""Typed native-MHR boundary; learned-model loading is optional and explicit."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class HumanParameters:
    pose: np.ndarray
    shape: np.ndarray
    scales: np.ndarray
    expression: np.ndarray


@dataclass
class DecodedHuman:
    vertices_world: np.ndarray
    joints_world: np.ndarray  # COCO17 order, meters


class HumanDecoder(Protocol):
    identity: dict
    parameter_names: list[str]
    faces: np.ndarray
    hand_vertices: tuple[np.ndarray, np.ndarray]

    def decode(self, parameters: HumanParameters, native_to_world: np.ndarray) -> DecodedHuman: ...


# Anatomical scalar controls, not arbitrary entries of a pose vector. Root
# orientation, fingers and all flexible-length/identity controls remain fixed.
REPAIR_CONTROLS = (
    "spine_twist0", "spine_lean0", "spine_bend0", "spine_twist1", "spine_lean1", "spine_bend1",
    "neck_twist", "neck_lean", "neck_bend", "head_twist", "head_lean", "head_bend",
    "r_uparm_twist", "r_uparm_ry", "r_uparm_rz", "r_elbow_bend", "r_lowarm_twist",
    "l_uparm_twist", "l_uparm_ry", "l_uparm_rz", "l_elbow_bend", "l_lowarm_twist",
    "r_upleg_twist", "r_upleg_ry", "r_upleg_rz", "r_knee_bend",
    "l_upleg_twist", "l_upleg_ry", "l_upleg_rz", "l_knee_bend",
)


def human_parameters(arrays: dict, n: int) -> HumanParameters:
    if not {"pose", "shape", "scales"} <= set(arrays):
        raise ValueError("Human repair requires native MHR pose/shape/scales")
    if arrays["pose"].shape != (n, 136):
        raise ValueError("Native MHR pose must be [T,136]")
    for key, width in (("shape", 45), ("scales", 68)):
        if arrays[key].shape not in ((width,), (n, width)):
            raise ValueError(f"Native MHR {key} has an invalid shape")
    expression = arrays.get("expression", np.zeros((n, 72), np.float32))
    if expression.shape not in ((72,), (n, 72)):
        raise ValueError("Native expression must be [72] or [T,72]")
    if any(not np.isfinite(a).all() for a in [arrays["pose"], arrays["shape"], arrays["scales"], expression]):
        raise ValueError("Nonfinite native human parameters")
    return HumanParameters(arrays["pose"], arrays["shape"], arrays["scales"], expression)


def human_select(p: HumanParameters, indices: np.ndarray) -> HumanParameters:
    return HumanParameters(p.pose[indices], *[a if a.ndim == 1 else a[indices] for a in (p.shape, p.scales, p.expression)])


def human_with_pose(p: HumanParameters, pose: np.ndarray) -> HumanParameters:
    return HumanParameters(pose, p.shape, p.scales, p.expression)
