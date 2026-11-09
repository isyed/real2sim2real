# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed runtime actions; load simulation dependencies only when requested.

Numerical utilities such as reference conditioning also live in this package.
Importing those utilities must not require Warp, Newton or a policy runtime.
"""

from importlib import import_module

__all__ = [
    "Action",
    "ActionBuffers",
    "ActionConfig",
    "ActionDiagnostics",
    "ActionLayout",
    "ActionSpec",
    "FinalTargetTransformConfig",
    "PolicyAction",
    "ReferenceOffsetAction",
    "RegularizedAction",
    "ResidualHandPoseAction",
    "ResidualJointPositionAction",
    "ResidualJointPositionActionConfig",
    "SonicJointResidualAction",
    "SonicJointResidualActionConfig",
]

_MODULES = {
    name: ".hand_pose"
    for name in (
        "Action",
        "ActionBuffers",
        "ActionConfig",
        "ActionDiagnostics",
        "ActionLayout",
        "ActionSpec",
        "PolicyAction",
        "ReferenceOffsetAction",
        "RegularizedAction",
        "ResidualHandPoseAction",
    )
}
_MODULES.update(
    {
        name: ".joint_position"
        for name in (
            "FinalTargetTransformConfig",
            "ResidualJointPositionAction",
            "ResidualJointPositionActionConfig",
        )
    }
)
_MODULES.update(
    {
        name: ".sonic"
        for name in (
            "SonicJointResidualAction",
            "SonicJointResidualActionConfig",
        )
    }
)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
