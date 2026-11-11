import pytest
from real2sim2real.controller_actor_training.arm_path_planning import (
    initial_starts,
    rank_paths,
)


def test_explicit_initial_branch_and_unchanged_legacy_default():
    arm = {"name": "left", "seed_rad": [0, 1], "initial_seeds_rad": [[0, 1], [2, 3]]}
    assert initial_starts(arm, {}) == [[0, 1], [2, 3]]
    assert initial_starts(arm, {"ik_seed_indices": {"left": 1}}) == [[2, 3]]
    for index in [-1, 2, True, 1.0]:
        with pytest.raises(ValueError, match="IK seed index"):
            initial_starts(arm, {"ik_seed_indices": {"left": index}})


def row(name, passed, speed, margin):
    return {
        "placement": {"name": name},
        "screen_pass": passed,
        "arms": {
            "left": {
                "maximum_sampled_velocity_ratio": speed,
                "minimum_joint_margin_rad": margin,
            }
        },
    }


def test_collision_failure_cannot_be_outweighed_by_fast_or_accurate_fit():
    rejected = row("collision", False, 0, 3)
    slow = row("slow", True, 0.8, 0.5)
    fast = row("fast", True, 0.5, 0.2)
    margin = row("margin", True, 0.5, 0.4)
    original = [rejected, slow, fast, margin]
    assert rank_paths(original) == [margin, fast, slow]
    assert original[0] is rejected
    assert rank_paths([rejected]) == []
