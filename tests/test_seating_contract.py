from copy import deepcopy

import pytest
from real2sim2real.controller_actor_training.seating_contract import (
    validate_seating_proposal,
)


def test_seed_and_bounded_edit_without_mutating_baseline():
    seed = {
        "nodes": {
            "configure": {
                "inputs": {
                    "pad_offsets_object_m": {"thumb": [0, 0, 0]},
                    "closure_offsets_rad": {},
                }
            }
        },
        "edges": [],
    }
    proposal = deepcopy(seed)
    proposal["nodes"]["configure"]["inputs"]["pad_offsets_object_m"]["thumb"][2] = 0.003
    validate_seating_proposal(proposal, seed, pads=["thumb"], joints=["joint"])
    assert seed["nodes"]["configure"]["inputs"]["pad_offsets_object_m"]["thumb"][2] == 0
    for vector in [[0.003, 0.003, 0], [float("nan"), 0, 0]]:
        proposal["nodes"]["configure"]["inputs"]["pad_offsets_object_m"]["thumb"] = (
            vector
        )
        with pytest.raises(ValueError):
            validate_seating_proposal(proposal, seed, pads=["thumb"], joints=["joint"])
    proposal = deepcopy(seed)
    proposal["edges"] = [["START", "skip"]]
    with pytest.raises(ValueError):
        validate_seating_proposal(proposal, seed, pads=["thumb"], joints=["joint"])
