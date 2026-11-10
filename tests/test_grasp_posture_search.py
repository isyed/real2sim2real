import numpy as np
import pytest

from real2sim2real.controller_actor_training.grasp_posture_search import (
    local_name,
    rotation_margin,
    select_candidate,
)


def test_incomplete_success_cannot_promote_posture():
    groups = [
        dict(
            name="baseline",
            candidate_variant=False,
            trials=[dict(success=True, hold_slip_m=0.0)],
        ),
        dict(
            name="partial",
            candidate_variant=True,
            trials=[
                dict(success=True, hold_slip_m=0.001),
                dict(success=False, hold_slip_m=0.002),
            ],
        ),
    ]
    assert select_candidate(groups) is None
    groups.append(
        dict(
            name="passing",
            candidate_variant=True,
            trials=[
                dict(success=True, hold_slip_m=0.003),
                dict(success=True, hold_slip_m=0.004),
            ],
        )
    )
    assert select_candidate(groups) == "passing"
    with pytest.raises(ValueError):
        local_name("../trial")


def wrist_model():
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><body name="hand"><joint name="fixture_rx" axis="1 0 0" range="-.5 .5"/>
    <joint name="fixture_ry" axis="0 1 0" range="-.5 .5"/>
    <joint name="fixture_rz" axis="0 0 1" range="-.5 .5"/>
    <geom type="sphere" size=".01" mass="1"/><site name="tip" pos=".1 0 0"/>
    </body></worldbody></mujoco>""")
    return mj, model


def test_posture_constraints_reserve_travel_without_changing_physical_limits():
    from real2sim2real.retargeting.grasp_kinematics import fit_contacts

    mj, m = wrist_model()
    data = mj.MjData(m)
    limits = m.jnt_range.copy()
    target = [[0.1 * np.cos(0.2), 0.1 * np.sin(0.2), 0]]
    result = fit_contacts(
        m,
        data,
        "hand",
        ["tip"],
        target,
        np.zeros(3),
        joint_bounds={"fixture_rz": [0.15, 0.3]},
    )
    assert 0.15 <= result["q"][2] <= 0.3
    assert max(result["site_errors_m"]) < 0.001
    np.testing.assert_array_equal(limits, m.jnt_range)
    for bounds in (
        {"fixture_rz": [-0.6, 0.3]},
        {"alien": [0, 0.1]},
        {"fixture_rz": [0.3, 0.2]},
        {"fixture_rz": [np.nan, 0.1]},
    ):
        with pytest.raises(ValueError):
            fit_contacts(
                m, data, "hand", ["tip"], target, np.zeros(3), joint_bounds=bounds
            )


def test_margin_audit_retains_negative_limit_violations():
    _, model = wrist_model()
    positions = np.array([[0, 0, -0.4], [0, 0, -0.51]])
    margins = rotation_margin(model, positions, "fixture_")
    assert margins["rz"] == pytest.approx(-0.01)
    assert margins["rx"] == 0.5
    with pytest.raises(ValueError):
        rotation_margin(model, np.empty((0, 3)), "fixture_")


def test_comparison_refuses_changed_baseline_feedback_or_geometric_gates(
    tmp_path, monkeypatch
):
    import json
    from copy import deepcopy
    from real2sim2real.artifacts import digest
    from real2sim2real.controller_actor_training import object_grasp_pilot
    from real2sim2real.controller_actor_training.grasp_posture_search import source

    config = dict(
        rollout=dict(hold_s=1, close_offset_rad=0.45),
        planning=dict(contact_error_m=0.008),
        contact_feedback=None,
    )
    (tmp_path / "states.npz").write_bytes(b"hash fixture; no dynamics")
    candidate = dict(candidate_id=1, accepted=True)
    parent = dict(
        success=True,
        configuration=deepcopy(config),
        candidate=candidate,
        states_sha256=digest(tmp_path / "states.npz"),
    )
    path = tmp_path / "result.json"
    path.write_text(json.dumps(parent))
    config["posture_search"] = dict(
        parent_trial=str(path), parent_trial_sha256=digest(path)
    )
    monkeypatch.setattr(
        object_grasp_pilot, "load_plan", lambda _: dict(candidates=[candidate])
    )
    source(config)
    for key, value in (
        ("contact_feedback", {"gain": 2}),
        ("finger_closure", {"offsets_rad": {"finger_PIP": 0.1}}),
        ("finger_surface_feedback", {"gain_s_inv": 4}),
        ("planning", {"contact_error_m": 0.02}),
    ):
        changed = deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match="preserve"):
            source(changed)
