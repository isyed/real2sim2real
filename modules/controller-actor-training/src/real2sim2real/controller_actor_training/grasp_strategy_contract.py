"""Pure-data authority for fresh paired-grasp proposals over fixed tool primitives."""

from copy import deepcopy
import math

PLAN_TOOL = "grasp.plan_pair"
HOLD_TOOL = "grasp.test_hold"
FIELDS = {
    "layout_id",
    "thumb_side",
    "wrist_roll_rad",
    "opening_gap_m",
    "preload_m",
    "seed_flexion_rad",
    "approach",
}


def validate_parameters(hands, layout_ids):
    if not isinstance(hands, dict) or set(hands) != {"right", "left"}:
        raise ValueError("Exactly one explicit proposal for each hand required")
    for hand in hands.values():
        if not isinstance(hand, dict) or set(hand) != FIELDS:
            raise ValueError("Only declared geometric grasp parameters allowed")
        if type(hand["layout_id"]) is not int or hand["layout_id"] not in layout_ids:
            raise ValueError("Select an existing catalogued surface layout")
        if type(hand["thumb_side"]) is not int or hand["thumb_side"] not in (-1, 1):
            raise ValueError("Explicit thumb side required")
        if hand["approach"] not in ("above", "outward"):
            raise ValueError("Only bounded above/outward acquisition supported")
        for key, lower, upper in (
            ("wrist_roll_rad", -math.pi / 2, math.pi / 2),
            ("opening_gap_m", 0.002, 0.012),
            ("preload_m", 0, 0.0015),
            ("seed_flexion_rad", 0.3, 0.9),
        ):
            value = hand[key]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError("Parameter outside finite authority: " + key)
    if hands["right"]["layout_id"] == hands["left"]["layout_id"]:
        raise ValueError("Two distinct contact layouts required")
    return deepcopy(hands)


def workflow(hands):
    return {
        "version": 3,
        "nodes": {
            "plan": {
                "type": "tool",
                "tool": PLAN_TOOL,
                "inputs": {"hands": deepcopy(hands)},
            },
            "hold": {"type": "tool", "tool": HOLD_TOOL, "inputs": {}},
            "complete": {"type": "end", "status": "success"},
        },
        "edges": [["START", "plan"], ["plan", "hold"], ["hold", "complete"]],
        "conditional_edges": {},
        "subgraphs": {},
    }


def validate_graph(document, layout_ids):
    try:
        hands = validate_parameters(
            document["nodes"]["plan"]["inputs"]["hands"], layout_ids
        )
    except (KeyError, TypeError) as error:
        raise ValueError("A declared paired strategy graph is required") from error
    if document != workflow(hands):
        raise ValueError(
            "Only grasp inputs may change; tools, ordering and gates are fixed"
        )
    return hands
