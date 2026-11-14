import numpy as np
import pytest
from real2sim2real.retargeting.observed_colliders import (
    apply_contact_roles,
    coverage_upper_bounds,
    occupied_boxes,
)


def test_occupied_cells_preserve_gap_and_cover_input_without_global_hull():
    p = np.array([[0, 0, 0], [0.001, 0.001, 0.001], [0.1, 0, 0]])
    boxes = occupied_boxes(p)
    assert len(boxes["centers_m"]) == 2
    np.testing.assert_allclose(coverage_upper_bounds(p, boxes), 0, atol=1e-15)
    assert coverage_upper_bounds([[0.05, 0, 0]], boxes)[0] > 0.04
    np.testing.assert_array_equal(
        occupied_boxes(p[::-1])["centers_m"], boxes["centers_m"]
    )
    with pytest.raises(ValueError):
        occupied_boxes([[np.nan, 0, 0]])
    with pytest.raises(ValueError):
        occupied_boxes(p, max_cells=1)


def test_roles_retain_exclusions_and_materialize_observed_cells():
    import xml.etree.ElementTree as ET

    tree = ET.ElementTree(
        ET.fromstring(
            '<mujoco><worldbody><geom name="rack"/><geom name="unknown"/><geom name="table" contype="1"/></worldbody></mujoco>'
        )
    )
    r = apply_contact_roles(
        tree, "rack", occupied_boxes([[0, 0, 0]]), planning_only_geoms=["unknown"]
    )
    root = tree.getroot()
    assert root.find("worldbody/geom[@name='rack']").get("contype") == "0"
    assert root.find("worldbody/geom[@name='table']").get("contype") == "1"
    assert root.find("worldbody/geom[@name='rack_observed_0']").get("contype") == "1"
    assert not r["unobserved_space_certified_free"]


def test_local_box_evidence_does_not_certify_unobserved_space():
    from real2sim2real.retargeting.observed_colliders import box_observation_evidence

    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = box_observation_evidence(
        [[0, 0.015, 0], [0.02, 0, 0]],
        [0, 0, 0],
        [0.02, 0.01, 0.01],
        world_from_box_rotation=rotation,
    )
    assert result["observations_inside"] == 1
    assert result["nearest_observation_distance_to_box_m"] == 0
    empty = box_observation_evidence(np.empty((0, 3)), [0, 0, 0], [0.01] * 3)
    assert empty["nearest_observation_distance_to_box_m"] is None
    assert empty["missing_evidence_is_not_free_space"]
    with pytest.raises(ValueError):
        box_observation_evidence([[np.nan, 0, 0]], [0, 0, 0], [0.01] * 3)
    with pytest.raises(ValueError):
        box_observation_evidence(
            [[0, 0, 0]],
            [0, 0, 0],
            [0.01] * 3,
            world_from_box_rotation=np.diag([1, 1, -1]),
        )
