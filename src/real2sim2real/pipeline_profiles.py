"""Named recipes over composable stages; recipes do not imply acceptance."""

PROFILES = {
    "human-reference": {
        "description": "Reconstruct, repair and retarget human motion, then train its correction controller.",
        "input_kind": "video",
        "stages": [
            "reconstruction",
            "reconstruction-repair",
            "retargeting",
            "controller-actor-training",
        ],
    },
    "egocentric-interaction": {
        "description": "Reconstruct calibrated object motion and metric hand observations for review.",
        "input_kind": "calibrated-capture",
        "stages": ["egocentric-reconstruction", "metric-interaction"],
    },
    "object-reference-control": {
        "description": "Screen a planned robot grasp path and test object-motion control with torque-driven arms.",
        "input_kind": "grasp-trial",
        "stages": ["arm-path-audit", "arm-replay"],
    },
}


def profile(name):
    if name not in PROFILES:
        raise ValueError("Unknown pipeline profile: " + str(name))
    return PROFILES[name]
