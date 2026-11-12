"""Authority boundaries and one-clock execution for seeded policy proposals."""

from copy import deepcopy

import pytest
from real2sim2real.controller_actor_training.bounded_policy_graph import (
    NativePhaseSession,
    validate_config,
)
from real2sim2real.controller_actor_training.passthrough_graph import workflow
from real2sim2real.controller_actor_training.proposal_contract import (
    seeded_workflow,
    validate_proposal,
    zero_parameters,
)

PHASES = ("approach", "close", "return", "release")
JOINTS = ("right_index_PIP", "support__left_thumb_IP")


def test_seed_and_small_corrections_preserve_every_other_graph_field():
    seed = seeded_workflow(workflow(PHASES))
    candidate = deepcopy(seed)
    candidate["nodes"]["configure"]["inputs"]["vectors_object_m"]["left"] = [
        0.002,
        0,
        0,
    ]
    candidate["nodes"]["configure"]["inputs"]["finger_joint_offsets_rad"] = {
        JOINTS[0]: 0.01
    }
    parameters = validate_proposal(candidate, seed, JOINTS)
    parameters["vectors_object_m"]["left"][0] = 100
    assert (
        candidate["nodes"]["configure"]["inputs"]["vectors_object_m"]["left"][0]
        == 0.002
    )
    assert seed["nodes"]["configure"]["inputs"] == zero_parameters()


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "0", 0.00601])
def test_invalid_wrist_values_fail_closed(value):
    seed = seeded_workflow(workflow(PHASES))
    candidate = deepcopy(seed)
    candidate["nodes"]["configure"]["inputs"]["vectors_object_m"]["left"][0] = value
    with pytest.raises(ValueError):
        validate_proposal(candidate, seed, JOINTS)


@pytest.mark.parametrize(
    "change", ["phase", "tool", "script", "wait", "gain", "joint", "rotation"]
)
def test_author_cannot_expand_authority(change):
    seed = seeded_workflow(workflow(PHASES))
    candidate = deepcopy(seed)
    inputs = candidate["nodes"]["configure"]["inputs"]
    if change == "phase":
        candidate["nodes"]["phase_01"]["inputs"]["phase"] = "hold"
    elif change == "tool":
        candidate["nodes"]["configure"]["tool"] = "object.teleport"
    elif change == "script":
        candidate["nodes"]["configure"]["type"] = "script"
    elif change == "wait":
        candidate["edges"].append(["phase_01", "phase_01"])
    elif change == "gain":
        inputs["kp"] = 2000
    elif change == "joint":
        inputs["finger_joint_offsets_rad"]["arm_PIP"] = 0.01
    else:
        inputs["rotation_vectors_object_rad"]["right"] = [0.05, 0.05, 0]
    with pytest.raises(ValueError):
        validate_proposal(candidate, seed, JOINTS)


def native_replay():
    for i, phase in enumerate(PHASES):
        sent = yield {"phase": phase, "time_s": i * 0.1, "physics_steps": i * 800}
        assert sent is None
    return {"records": [{"time_s": 0.4}], "success": False}


def test_native_handoff_has_one_reset_no_actions_no_extra_simulation_time():
    session = NativePhaseSession(native_replay, PHASES, zero_parameters())
    assert session.configure(**zero_parameters())["simulation_steps"] == 0
    for phase in PHASES:
        span = session.advance(phase)
        span["end_time_s"] = -1  # Returned metadata is not mutable simulator state.
    receipt = session.receipt()
    assert receipt["complete"] and receipt["reset_count"] == 1
    assert receipt["spans"][-1]["end_time_s"] == 0.4
    assert all(b["simulation_gap_s"] == 0 for b in receipt["boundaries"])
    assert (
        session.result["success"] is False
    )  # Graph completion is not task acceptance.
    with pytest.raises(RuntimeError):
        session.configure(**zero_parameters())
    with pytest.raises(RuntimeError):
        session.advance("approach")


def test_wrong_phase_cannot_start_physics():
    session = NativePhaseSession(
        lambda: pytest.fail("Do not start"), PHASES, zero_parameters()
    )
    session.configure(**zero_parameters())
    with pytest.raises(ValueError):
        session.advance("close")
    assert session.reset_count == 0


def test_premature_completion_cannot_claim_graph_success():
    def early():
        yield {"phase": "approach", "time_s": 0}
        return {"records": [{"time_s": 0.1}]}

    session = NativePhaseSession(early, PHASES, zero_parameters())
    session.configure(**zero_parameters())
    with pytest.raises(RuntimeError, match="outside the final"):
        session.advance("approach")
    assert not session.receipt()["complete"]


def test_physical_adapter_cannot_accept_two_action_owners():
    from real2sim2real.controller_actor_training.arm_replay_pilot import rollout

    with pytest.raises(ValueError, match="cannot share"):
        next(rollout({}, interaction=object(), phase_handoff=True))


def test_abort_during_final_phase_does_not_claim_completion():
    def truncated():
        for i, phase in enumerate(PHASES):
            yield {"phase": phase, "time_s": i * 0.1}
        return {"records": [{"time_s": 0.35}], "rejected_for": ["incomplete_pilot"]}

    session = NativePhaseSession(truncated, PHASES, zero_parameters())
    session.configure(**zero_parameters())
    for phase in PHASES[:-1]:
        session.advance(phase)
    with pytest.raises(RuntimeError, match="incomplete pilot"):
        session.advance(PHASES[-1])
    assert not session.receipt()["complete"]


@pytest.mark.parametrize("time", [float("nan"), float("inf"), 0])
def test_invalid_handoff_clock_cannot_claim_progress(time):
    def invalid():
        yield {"phase": "approach", "time_s": 0}
        yield {"phase": "close", "time_s": time}

    session = NativePhaseSession(invalid, PHASES, zero_parameters())
    session.configure(**zero_parameters())
    with pytest.raises(RuntimeError, match="clock"):
        session.advance("approach")
    assert not session.receipt()["complete"]
    session.close()


@pytest.mark.parametrize(
    "deadline", [True, 0, -1, 901, float("nan"), float("inf"), "900"]
)
def test_invalid_deadline_rejected_before_optional_runtime_imports(deadline):
    config = dict(
        baseline={},
        workflow={},
        allowed_joints=[],
        output="unused",
        deadline_s=deadline,
    )
    with pytest.raises(ValueError, match="deadline"):
        validate_config(config)
