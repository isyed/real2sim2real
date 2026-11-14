import json

import numpy as np
import pytest
from real2sim2real.controller_actor_training.replay_comparison import compare_replays


def inputs(tmp_path):
    a, b = tmp_path / "expected", tmp_path / "actual"
    for directory in [a, b]:
        directory.mkdir()
        np.savez(
            directory / "states.npz",
            qpos=np.zeros((2, 1)),
            qvel=np.zeros((2, 1)),
            commands=np.zeros((2, 1)),
            timestamps_s=[0.0, 0.1],
        )
        result = {
            "task_completed": True,
            "complete_horizon": True,
            "stop_reason": None,
            "release": {},
            "boundary": {"passed": True},
            "warnings": {},
            "hard_failures": {},
            "strict_quality_pass": False,
            "control_accepted": False,
            "records": [{"t": 0.0, "phase": "acquisition", "old_load": 1.0}],
        }
        if directory == b:
            result["records"][0]["support_load"] = result["records"][0].pop("old_load")
        (directory / "result.json").write_text(json.dumps(result))
    return a, b


def test_exact_arrays_and_explicit_semantic_alias(tmp_path):
    a, b = inputs(tmp_path)
    assert not compare_replays(a, b)["passed"]
    result = compare_replays(a, b, record_field_aliases={"old_load": "support_load"})
    assert result["passed"] and all(result["exact"].values())
    assert not result["control_accepted"]


@pytest.mark.parametrize(
    "change", ["command", "nonfinite", "clock", "release", "failure"]
)
def test_bad_replay_cannot_pass_by_matching_only_saved_positions(tmp_path, change):
    a, b = inputs(tmp_path)
    with np.load(b / "states.npz") as data:
        arrays = {k: data[k].copy() for k in data.files}
    row = json.loads((b / "result.json").read_text())
    if change == "command":
        arrays["commands"][1, 0] = 1e-12
    elif change == "nonfinite":
        arrays["qvel"][1, 0] = np.nan
    elif change == "clock":
        arrays["timestamps_s"][1] = 0.0
    elif change == "release":
        row["release"] = {"started_s": 0.1}
    else:
        row["task_completed"] = False
    np.savez(b / "states.npz", **arrays)
    (b / "result.json").write_text(json.dumps(row))
    assert not compare_replays(a, b, record_field_aliases={"old_load": "support_load"})[
        "passed"
    ]
