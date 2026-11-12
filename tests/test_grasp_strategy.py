"""Fresh proposals retain fixed authority, source separation and admission gates."""

from copy import deepcopy
import json

import pytest
from real2sim2real.controller_actor_training.grasp_strategy_contract import (
    validate_graph,
    workflow,
)
from real2sim2real.controller_actor_training.grasp_strategy import StrategySession
from real2sim2real.controller_actor_training.multi_hand_scene import (
    append_hypothesis_hand,
)
from real2sim2real.controller_actor_training.object_grasp_pilot import (
    required_hold_duration,
)


def hands():
    return {
        side: dict(
            layout_id=i,
            thumb_side=1 if i == 0 else -1,
            wrist_roll_rad=0,
            opening_gap_m=0.008,
            preload_m=0.001,
            seed_flexion_rad=0.7,
            approach="above",
        )
        for i, side in enumerate(("right", "left"))
    }


def test_parameter_copy_and_canonical_graph():
    graph = workflow(hands())
    original = deepcopy(graph)
    parameters = validate_graph(graph, {0, 1})
    parameters["right"]["layout_id"] = 999
    assert graph == original


def test_expanding_deterministic_budget_preserves_existing_proposals():
    from real2sim2real.controller_actor_training.grasp_strategy import (
        deterministic_proposals,
    )

    cat = {
        "center_of_mass_m": [0, 0, 0],
        "layouts": [
            {"layout_id": i, "center_m": point}
            for i, point in enumerate(
                ([0.06, 0, 0], [-0.06, 0, 0], [0, 0.05, 0], [0, -0.05, 0])
            )
        ],
    }
    original = deepcopy(cat)
    four = deterministic_proposals(cat, 4)
    five = deterministic_proposals(cat, 5)
    assert len(five) == 5 and five[:4] == four
    assert cat == original
    for graph in five:
        validate_graph(graph, {0, 1, 2, 3})
    for count in (True, 0, 9):
        with pytest.raises(ValueError):
            deterministic_proposals(cat, count)


@pytest.mark.parametrize(
    "change",
    [
        "tool",
        "edges",
        "force",
        "nan",
        "bool",
        "surface",
        "same_surface",
        "excess_rotation",
    ],
)
def test_strategy_cannot_gain_undeclared_authority(change):
    graph = workflow(hands())
    p = graph["nodes"]["plan"]["inputs"]["hands"]["right"]
    if change == "tool":
        graph["nodes"]["hold"]["tool"] = "object.teleport"
    elif change == "edges":
        graph["edges"].append(["hold", "plan"])
    elif change == "force":
        p["force_n"] = 100
    elif change == "nan":
        p["preload_m"] = float("nan")
    elif change == "bool":
        p["thumb_side"] = True
    elif change == "surface":
        p["layout_id"] = 999
    elif change == "same_surface":
        p["layout_id"] = 1
    else:
        p["wrist_roll_rad"] = 2
    with pytest.raises(ValueError):
        validate_graph(graph, {0, 1})


def test_rejected_geometry_never_starts_physics(tmp_path, monkeypatch):
    from real2sim2real.controller_actor_training import grasp_strategy as module

    monkeypatch.setattr(
        module,
        "plan_pair",
        lambda *args: {"accepted": False, "rejected_for": ["paired_path_collision"]},
    )
    monkeypatch.setattr(
        module,
        "run_hold",
        lambda *args: pytest.fail("No dynamics on rejected geometry"),
    )
    session = StrategySession({}, {}, hands(), tmp_path)
    with pytest.raises(ValueError):
        session.test_hold()
    session.propose(hands())
    assert session.test_hold()["executed"] is False
    with pytest.raises(ValueError):
        session.test_hold()
    with pytest.raises(ValueError):
        session.propose(hands())
    assert json.loads((tmp_path / "plan.json").read_text())["rejected_for"] == [
        "paired_path_collision"
    ]


def test_fresh_scene_requires_explicit_hypothesis_condition():
    with pytest.raises(ValueError, match="hypothesis"):
        append_hypothesis_hand({})
    with pytest.raises(ValueError, match="hypothesis"):
        append_hypothesis_hand({"hypothesis_only": True, "parent": {}})


def test_legacy_task_hold_is_explicit_but_task_minimum_cannot_be_shortened():
    assert required_hold_duration({}, {"hold_s": 3}) == 3
    assert required_hold_duration({"hold_s": 2}, {"hold_s": 3}) == 2
    for goal in ({"hold_s": 4}, {"hold_s": float("nan")}, {"hold_s": 0}):
        with pytest.raises(ValueError):
            required_hold_duration(goal, {"hold_s": 3})
    for hold in (float("nan"), True, 0):
        with pytest.raises(ValueError):
            required_hold_duration({"hold_s": 2}, {"hold_s": hold})


def test_scene_extension_detects_unintended_physical_retuning():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.multi_hand_scene import (
        verify_preserved_dynamics,
    )

    original = '<mujoco><worldbody><body name="object"><freejoint name="free"/><geom name="ball" type="sphere" size=".02" mass=".2"/></body></worldbody></mujoco>'
    extension = original.replace(
        "</worldbody>",
        '<body name="extra" pos="1 1 1"><geom name="extra_pad" type="sphere" size=".01"/></body></worldbody>',
    )
    before, after = (
        mj.MjModel.from_xml_string(original),
        mj.MjModel.from_xml_string(extension),
    )
    assert verify_preserved_dynamics(before, after)["original_dynamics_preserved"]
    after.geom_friction[after.geom("ball").id, 0] += 0.1
    with pytest.raises(ValueError, match="friction"):
        verify_preserved_dynamics(before, after)


def test_feedback_preserves_missing_physics_and_rejects_contradictory_verdicts():
    from real2sim2real.controller_actor_training.grasp_strategy_feedback import (
        summarize,
    )

    plan = dict(
        parameters=hands(),
        accepted=False,
        rejected_for=["paired_path_collision"],
        hands=[{}, {}],
        paired_path=None,
        gravity_wrench={"feasible": False},
    )
    evaluation = dict(
        geometric_pass=False,
        rejected_for=["paired_path_collision"],
        hold={"executed": False, "simulated_hold_pass": False},
    )
    report = summarize(evaluation, plan)
    assert report["hands"][0]["maximum_target_error_m"] is None
    assert report["hold"]["executed"] is False
    evaluation["hold"]["executed"] = True
    with pytest.raises(ValueError, match="bypass"):
        summarize(evaluation, plan)


def test_review_and_feedback_require_pinned_matching_run_artifacts(tmp_path):
    from real2sim2real.artifacts import digest
    from real2sim2real.controller_actor_training.grasp_strategy_feedback import load_run

    evaluation = {
        "geometric_pass": False,
        "rejected_for": ["missing_model_pad_patch"],
        "hold": {"executed": False},
    }
    plan = {"accepted": False, "rejected_for": evaluation["rejected_for"]}
    ep, pp = tmp_path / "evaluation.json", tmp_path / "plan.json"
    ep.write_text(json.dumps(evaluation))
    pp.write_text(json.dumps(plan))
    run = {
        "evaluation": {"path": str(ep), "sha256": digest(ep)},
        "plan": {"path": str(pp), "sha256": digest(pp)},
    }
    assert load_run(run) == (evaluation, plan)
    other = tmp_path / "other"
    other.mkdir()
    (other / "plan.json").write_bytes(pp.read_bytes())
    mismatched = deepcopy(run)
    mismatched["plan"]["path"] = str(other / "plan.json")
    with pytest.raises(ValueError, match="one archived run"):
        load_run(mismatched)
    pp.write_text(json.dumps({**plan, "accepted": True}))
    with pytest.raises(ValueError, match="input changed"):
        load_run(run)
    run["plan"]["sha256"] = digest(pp)
    with pytest.raises(ValueError, match="verdicts disagree"):
        load_run(run)


def test_actual_gap_scheduler_preserves_declared_tool_order(tmp_path):
    pytest.importorskip("gap")
    from gap.runtime.executor import WorkflowExecutor
    from gap_core.tools import ToolRegistry
    from real2sim2real.controller_actor_training.grasp_strategy_contract import (
        PLAN_TOOL,
        HOLD_TOOL,
    )

    visits = []

    def plan(hands: dict) -> dict:
        visits.append("plan")
        return {"geometric_pass": False}

    def hold() -> dict:
        assert visits == ["plan"]
        visits.append("hold")
        return {"executed": False}

    registry = ToolRegistry()
    registry.register_callable(PLAN_TOOL, plan, summary="test", requires_ctx=False)
    registry.register_callable(HOLD_TOOL, hold, summary="test", requires_ctx=False)
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(workflow(hands())))
    executor = WorkflowExecutor(
        path,
        tool_registry=registry,
        trace_dir=tmp_path / "trace",
        max_node_workers=1,
        checkpoints="off",
    )
    executor.execute()
    assert executor.exit_status == "success" and visits == ["plan", "hold"]
