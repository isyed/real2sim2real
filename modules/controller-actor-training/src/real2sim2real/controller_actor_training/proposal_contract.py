"""Pure-data contract for a single bounded, baseline-seeded graph proposal.

This module deliberately uses only the standard library so the identical
validator can be supplied to an isolated author without simulation access.
"""

import math
from copy import deepcopy

CONFIGURE_TOOL = "controller.configure_bounded_grasp"
PARAMETERS = {
    "vectors_object_m",
    "rotation_vectors_object_rad",
    "finger_joint_offsets_rad",
}


def zero_parameters():
    return {
        "vectors_object_m": {h: [0.0, 0.0, 0.0] for h in ("right", "left")},
        "rotation_vectors_object_rad": {h: [0.0, 0.0, 0.0] for h in ("right", "left")},
        "finger_joint_offsets_rad": {},
    }


def seeded_workflow(phase_graph):
    graph = deepcopy(phase_graph)
    if "configure" in graph["nodes"]:
        raise ValueError("Configuration node already present")
    graph["nodes"]["configure"] = {
        "type": "tool",
        "tool": CONFIGURE_TOOL,
        "inputs": zero_parameters(),
    }
    starts = [edge for edge in graph["edges"] if edge[0] == "START"]
    if len(starts) != 1:
        raise ValueError("One linear entry required")
    starts[0][0] = "configure"
    graph["edges"].insert(0, ["START", "configure"])
    return graph


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_parameters(parameters, allowed_joints):
    if not isinstance(parameters, dict) or set(parameters) != PARAMETERS:
        raise ValueError("Only the declared wrist and finger parameters are allowed")
    for key, limit in (
        ("vectors_object_m", 0.006),
        ("rotation_vectors_object_rad", 0.06),
    ):
        vectors = parameters[key]
        if not isinstance(vectors, dict) or set(vectors) != {"left", "right"}:
            raise ValueError("Explicit bilateral corrections required")
        for vector in vectors.values():
            if (
                not isinstance(vector, list)
                or len(vector) != 3
                or not all(finite_number(x) for x in vector)
                or math.sqrt(sum(x * x for x in vector)) > limit
            ):
                raise ValueError("Wrist correction exceeds its finite norm bound")
    offsets = parameters["finger_joint_offsets_rad"]
    if not isinstance(offsets, dict) or set(offsets) - set(allowed_joints):
        raise ValueError("Only explicitly admitted finger joints are allowed")
    for name, value in offsets.items():
        if (
            not name.endswith(("MCP_FE", "PIP", "DIP", "thumb_IP"))
            or not finite_number(value)
            or abs(value) > 0.06
        ):
            raise ValueError("Finger offset exceeds its finite joint bound")
    for support in (False, True):
        if (
            math.sqrt(
                sum(
                    v * v
                    for n, v in offsets.items()
                    if n.startswith("support__") == support
                )
            )
            > 0.12
        ):
            raise ValueError("Finger offset exceeds its per-hand norm bound")


def validate_proposal(document, seed, allowed_joints):
    """No executable code, new tools, branches, clocks or phase changes admitted."""
    try:
        parameters = document["nodes"]["configure"]["inputs"]
        validate_parameters(parameters, allowed_joints)
        restored = deepcopy(document)
        restored["nodes"]["configure"]["inputs"] = zero_parameters()
    except (KeyError, TypeError) as error:
        raise ValueError("A seeded bounded graph is required") from error
    if restored != seed:
        raise ValueError("All graph content outside correction parameters is frozen")
    return deepcopy(parameters)
