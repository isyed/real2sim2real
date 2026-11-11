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
