from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from real2sim2real.artifacts import digest
from real2sim2real.controller_actor_training.pad_seating_plan import bind
from real2sim2real.controller_actor_training.pad_seating_plan import (
    nonpad_chain_geoms,
    nonpad_clearance_passes,
    contact_witness,
)
from real2sim2real.controller_actor_training.arm_finger_feedback import finger_waypoint


def fixture():
    pad = dict(
        role="primary",
        group="index",
        gid=3,
        dofs=np.array([0]),
        qids=np.array([1]),
        limits=np.array([[-1.0, 1.0]]),
        point_body=np.zeros(3),
        point_object=np.zeros(3),
        normal_object=np.array([1.0, 0, 0]),
    )
    hand = dict(
        role="primary",
        qids=np.array([1]),
        waypoints=[np.array([0.4]), np.array([0.3]), np.array([0.1])],
    )
    live = SimpleNamespace(
        hands=[hand],
        servo=SimpleNamespace(pads=[pad]),
        source_hashes={},
        rollout=dict(approach_s=1.0, close_s=0.8),
    )
    model = SimpleNamespace(
        geom=lambda g: SimpleNamespace(name="pad"),
        joint=lambda j: SimpleNamespace(name="index_joint"),
        dof_jntid=np.array([0]),
    )
    proposal = dict(
        schema="real2sim2real.pad-seating-plan/1",
        geometry_accepted=True,
        pads=[
            dict(
                label="primary:index",
                geom="pad",
                joint_names=["index_joint"],
                closed_rad=[0.2],
                point_body_m=[0.001, 0, 0],
                point_object_m=[0.01, 0, 0],
                normal_object=[1.0, 0, 0],
            )
        ],
    )
    return live, model, proposal


def write(tmp_path, proposal, **extra):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(proposal))
    return dict(path=str(path), sha256=digest(path), **extra)


def test_seating_changes_closed_fingers_and_frames_without_retiming_acquisition_anchor(
    tmp_path,
):
    live, model, proposal = fixture()
    original = deepcopy(live.rollout)
    bind(live, model, write(tmp_path, proposal, close_fraction=0.5))
    np.testing.assert_array_equal(live.hands[0]["waypoints"][0], [0.4])
    np.testing.assert_array_equal(live.hands[0]["waypoints"][1], [0.3])
    np.testing.assert_array_equal(live.hands[0]["waypoints"][2], [0.2])
    assert live.rollout["close_s"] == original["close_s"]
    assert live.rollout["approach_s"] + live.rollout["close_s"] == 1.8
    np.testing.assert_allclose(
        finger_waypoint(1.4, *live.hands[0]["waypoints"], live.rollout), [0.2]
    )
    np.testing.assert_array_equal(live.servo.pads[0]["point_body"], [0.001, 0, 0])


@pytest.mark.parametrize(
    "case",
    [
        "wrong_pad",
        "wrong_joint",
        "excess_offset",
        "nonfinite",
        "bad_normal",
        "rejected",
    ],
)
def test_unusable_seating_is_rejected_before_mutating_live_commands(tmp_path, case):
    live, model, p = fixture()
    item = p["pads"][0]
    if case == "wrong_pad":
        item["geom"] = "arm"
    if case == "wrong_joint":
        item["joint_names"] = ["arm_joint"]
    if case == "excess_offset":
        item["closed_rad"] = [0.5]
    if case == "nonfinite":
        item["closed_rad"] = [float("nan")]
    if case == "bad_normal":
        item["normal_object"] = [2.0, 0, 0]
    if case == "rejected":
        p["geometry_accepted"] = False
    with pytest.raises(ValueError):
        bind(live, model, write(tmp_path, p))
    np.testing.assert_array_equal(live.hands[0]["waypoints"][2], [0.1])
    np.testing.assert_array_equal(live.servo.pads[0]["point_body"], [0, 0, 0])


def test_nonpad_clearance_excludes_visuals_pad_and_other_fingers():
    model = SimpleNamespace(
        dof_jntid=np.array([0, 1]),
        jnt_bodyid=np.array([2, 3]),
        ngeom=5,
        geom_bodyid=np.array([2, 2, 3, 4, 3]),
        geom_contype=np.array([0, 1, 1, 1, 1]),
        geom_conaffinity=np.array([0, 1, 1, 1, 1]),
    )
    assert nonpad_chain_geoms(model, dict(dofs=[0, 1], gid=4)) == [1, 2]
    good = [dict(gap_m=0.001, witness_distance_error_m=0)]
    assert nonpad_clearance_passes(good, 0.001)
    assert not nonpad_clearance_passes([], 0.001)
    assert not nonpad_clearance_passes(
        [dict(gap_m=0, witness_distance_error_m=0)], 0.001
    )
    assert not nonpad_clearance_passes(
        [dict(gap_m=0.001, witness_distance_error_m=0.003)], 0.001
    )
    assert nonpad_clearance_passes(
        [dict(gap_m=0.03, witness_distance_error_m=None, distance_is_lower_bound=True)],
        0.001,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_contact_witness_orders_pad_then_object_and_preserves_signed_gap(reverse):
    normal = np.array([-1 if reverse else 1, 0, 0])
    c = SimpleNamespace(
        geom1=2 if reverse else 1,
        geom2=1 if reverse else 2,
        pos=np.array([0.1, 0.2, 0.3]),
        frame=np.r_[normal, np.zeros(6)],
        dist=0.002,
    )
    gap, segment = contact_witness(SimpleNamespace(contact=[c]), 1, {2}, 0.03)
    assert gap == 0.002
    np.testing.assert_allclose(segment[:3], [0.099, 0.2, 0.3])
    np.testing.assert_allclose(segment[3:], [0.101, 0.2, 0.3])
    gap, segment = contact_witness(SimpleNamespace(contact=[c]), 7, {2}, 0.03)
    assert gap == 0.03 and not np.any(segment)
