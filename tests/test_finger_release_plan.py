import json
from types import SimpleNamespace

import numpy as np
import pytest

from real2sim2real.artifacts import digest
from real2sim2real.controller_actor_training.finger_release_plan import bind


def test_release_binding_is_bounded_and_never_changes_nominal_acquisition(tmp_path):
    hand = dict(
        qids=np.array([1]),
        waypoints=[np.array([0.2]), np.array([0.1]), np.array([0.3])],
    )
    live = SimpleNamespace(qids=np.array([1]), hands=[hand], source_hashes={})
    model = SimpleNamespace(
        joint=lambda name: SimpleNamespace(
            qposadr=[1 if name == "finger" else 2], range=[-1.0, 1.0]
        )
    )
    path = tmp_path / "posture.json"

    def spec(targets, accepted=True):
        path.write_text(
            json.dumps(
                dict(
                    schema="real2sim2real.finger-release-posture/1",
                    accepted=accepted,
                    joint_targets_rad=targets,
                )
            )
        )
        return dict(path=str(path), sha256=digest(path))

    bind(live, model, spec({"finger": -0.2}))
    assert hand["release_open"][0] == -0.2
    np.testing.assert_array_equal(hand["waypoints"][1], [0.1])
    for targets in [{"finger": -0.5}, {"arm": 0.1}]:
        with pytest.raises(ValueError):
            bind(live, model, spec(targets))
        np.testing.assert_array_equal(hand["release_open"], [-0.2])
    with pytest.raises(ValueError):
        bind(live, model, spec({"finger": 0.1}, False))
    pinned = spec({"finger": 0.1})
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        bind(live, model, pinned)


def test_jointly_planned_opening_rejects_a_different_wrist_path_or_clock(tmp_path):
    path = tmp_path / "posture.json"
    withdrawal = {
        "rotation_vectors_world_rad": {"right": [0, 0.1, 0], "left": [0, 0, 0]}
    }
    release = {"start_s": 13.0, "duration_s": 1.5}
    path.write_text(
        json.dumps(
            dict(
                schema="real2sim2real.finger-release-posture/1",
                accepted=True,
                joint_targets_rad={"finger": 0.1},
                wrist_withdrawal=withdrawal,
                release_settings=release,
            )
        )
    )
    spec = dict(path=str(path), sha256=digest(path))
    with pytest.raises(ValueError, match="jointly screened"):
        bind(None, None, spec)
    with pytest.raises(ValueError, match="jointly screened"):
        bind(
            None,
            None,
            spec,
            withdrawal=withdrawal,
            release=dict(release, duration_s=2.0),
        )


def test_fixture_planning_only_changes_bounded_open_fingers():
    from real2sim2real.controller_actor_training.finger_release_plan import (
        planning_targets,
    )

    model = SimpleNamespace(joint=lambda n: SimpleNamespace(range=[-0.5, 0.5]))
    hands = [
        dict(
            joint_names=["fixture_x", "finger"],
            fixture_prefix="fixture_",
            root_body="hand",
            q_open=[0.01, 0.2],
        )
    ]
    result = planning_targets(model, hands, {"finger": 0.4})
    np.testing.assert_array_equal(result["hand"], [0.01, 0.4])
    assert hands[0]["q_open"] == [0.01, 0.2]
    for targets in [
        {"fixture_x": 0.02},
        {"object_free": 0.01},
        {"finger": 0.6},
        {"finger": float("nan")},
    ]:
        with pytest.raises(ValueError):
            planning_targets(model, hands, targets)
