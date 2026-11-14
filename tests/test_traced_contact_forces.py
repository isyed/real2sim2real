import numpy as np
import pytest
from real2sim2real.controller_actor_training.traced_contact_forces import (
    grouped_object_forces,
)


def contact(pair, frame=None):
    return {
        "geoms": pair,
        "frame": frame if frame is not None else [0, 0, 1, 0, 1, 0, -1, 0, 0],
        "wrench_contact_frame": [2, 0.3, 0.1, 0, 0, 0],
    }


def test_world_force_sign_groups_and_unknown_counterparts():
    contacts = [
        contact(["table", "cup"]),
        contact(["cup", "finger"]),
        contact(["", "cup"]),
    ]
    result = grouped_object_forces(
        contacts, ["cup"], {"table": ["table"], "hand": ["finger"]}
    )
    np.testing.assert_allclose(result["table"]["force_world_n"], [-0.1, 0.3, 2])
    np.testing.assert_allclose(result["hand"]["force_world_n"], [0.1, -0.3, -2])
    assert result["unclassified"]["contacts"] == 1
    assert result["unclassified"]["counterparts"] == [""]
    assert contacts[0]["wrench_contact_frame"] == [2, 0.3, 0.1, 0, 0, 0]


def test_empty_contact_group_is_not_dropped_and_invalid_frames_rejected():
    result = grouped_object_forces([], ["cup"], {"table": ["table"]})
    assert result["table"]["force_world_n"] == [0, 0, 0]
    with pytest.raises(ValueError, match="Orthonormal"):
        grouped_object_forces(
            [contact(["table", "cup"], [2, 0, 0, 0, 1, 0, 0, 0, 1])],
            ["cup"],
            {"table": ["table"]},
        )
    with pytest.raises(ValueError, match="Distinct"):
        grouped_object_forces([], ["cup"], {"a": ["table"], "b": ["table"]})
