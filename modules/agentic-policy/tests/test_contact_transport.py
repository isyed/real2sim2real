from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from real2sim2real.agentic_policy.contact_transport import (
    retention_guard,
    screen_transport,
)
from real2sim2real.agentic_policy.manipulation import DEFAULT_CHECKS


def fixture(*, obstacle=False, pad_overlap=0.001, proximal=False):
    wall = (
        '<body pos=".21 0 0"><geom name="wall" type="box" size=".05 .05 .05"/></body>'
        if obstacle
        else ""
    )
    link = (
        '<geom name="proximal" type="box" pos=".02 .055 0" size=".02 .02 .02" mass=".1"/>'
        if proximal
        else ""
    )
    m = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>{wall}
      <body name="hand"><joint name="x" type="slide" axis="1 0 0"/>
      <geom name="pad" type="box" size=".05 .05 .05" mass="1"/>{link}</body>
      <body name="object" pos="{0.1 - pad_overlap} 0 0"><freejoint/>
      <geom name="cup" type="box" size=".05 .05 .05" mass="1"/></body>
      </worldbody></mujoco>''')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    s = SimpleNamespace(
        model=m,
        data=d,
        mj=mujoco,
        object=m.body("object").id,
        wrist=m.body("hand").id,
        robot={m.body("hand").id},
        qids=np.array([0]),
        nominal=np.array([0.0]),
        feedback=0.01,
        _vector=lambda q: np.array([q["x"]]),
        _validate_path=lambda *args: None,
        pad_ids=[m.geom("pad").id],
        config={"penetration_limit_m": 0.003},
        fault=None,
    )
    skills = SimpleNamespace(
        session=s,
        anchor=np.eye(4),
        checks=DEFAULT_CHECKS | {"minimum_contact_pads": 1},
        events=[],
    )
    obs = {
        "time_s": 0.0,
        "pad_forces_n": [1.0],
        "wrist_position_m": [0, 0, 0],
        "wrist_rotation": np.eye(3),
        "object_position_m": [0.1 - pad_overlap, 0, 0],
        "object_rotation": np.eye(3),
    }
    return skills, obs


def test_only_loaded_pad_stationary_conflict_is_relaxed_and_world_obstacle_still_rejects():
    skills, obs = fixture()
    s = skills.session
    before = s.data.qpos.copy()
    result = screen_transport(skills, [{"x": 0.02}], [1.0], obs)
    assert result["admitted"]
    assert (
        result["audit"]["groups"]["stationary_object"]["stationary_pad_contacts"][
            "peak_penetration_m"
        ]
        > 0.02
    )
    np.testing.assert_array_equal(s.data.qpos, before)
    skills, obs = fixture(obstacle=True)
    blocked = screen_transport(skills, [{"x": 0.02}], [1.0], obs)
    assert not blocked["admitted"] and "wall" in blocked["witness"]
    skills, obs = fixture(pad_overlap=0.004)
    assert not screen_transport(skills, [{"x": 0.02}], [1.0], obs)["admitted"]


def test_no_contact_or_acquisition_cannot_claim_rigid_carrying():
    skills, obs = fixture()
    assert not screen_transport(
        skills, [{"x": 0.02}], [1.0], obs | {"pad_forces_n": [0.0]}
    )["admitted"]
    skills.anchor = None
    assert not screen_transport(skills, [{"x": 0.02}], [1.0], obs)["admitted"]


def test_live_guard_separates_quality_warnings_from_contact_and_collision_failures():
    from real2sim2real.agentic_policy.manipulation import relative_pose

    skills, obs = fixture()
    guard = retention_guard(skills, relative_pose(obs))
    assert guard(obs) is None
    assert guard(obs | {"pad_forces_n": [0.0]}) == "transport_contact_loss"
    assert guard(obs | {"object_position_m": [0.2, 0, 0]}) is None
    assert skills.events[-1]["tool"] == "transport_retention_warning"
    assert skills.events[-1]["result"]["translation_m"] > 0.1
    skills.session.data.qpos[0] = 0.02
    mujoco.mj_forward(skills.session.model, skills.session.data)
    assert guard(obs) == "transport_physical_penetration"


def test_nonpad_links_are_not_exempt_from_stationary_screen():
    skills, obs = fixture(proximal=True)
    result = screen_transport(skills, [{"x": 0.02}], [1.0], obs)
    assert not result["admitted"] and "proximal" in result["witness"]
    skills, obs = fixture()
    skills.session.pad_ids = []
    with pytest.raises(ValueError, match="configured pads"):
        from real2sim2real.agentic_policy.transport_screen_audit import (
            audit_transport_screen,
        )

        audit_transport_screen(
            skills.session, [{"x": 0.02}], [1.0], stationary_pad_ids=[0]
        )


def test_loaded_preview_cannot_excuse_external_or_proximal_collisions(monkeypatch):
    import real2sim2real.agentic_policy.contact_transport as module

    calls = []
    monkeypatch.setattr(
        module, "preview_transport", lambda *args: calls.append(1) or {"passed": True}
    )
    skills, obs = fixture(pad_overlap=0.004)
    skills.transport_mode = "contact_aware_preview"
    assert screen_transport(skills, [{"x": 0.02}], [0.4], obs)["admitted"]
    assert len(calls) == 1
    for kwargs in [{"obstacle": True}, {"proximal": True}]:
        skills, obs = fixture(**kwargs)
        skills.transport_mode = "contact_aware_preview"
        assert not screen_transport(skills, [{"x": 0.02}], [0.4], obs)["admitted"]
    assert len(calls) == 1
