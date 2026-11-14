import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from real2sim2real.controller_actor_training.environment_observation import (
    observe_environment,
)
from test_continuous_policy import make_session


def state(session):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(session.model, spec))
    mujoco.mj_getState(session.model, session.data, values, spec)
    return values


def test_whole_environment_support_includes_unconfigured_floor_without_mutation(
    tmp_path,
):
    session = make_session(tmp_path, "support")
    session.hold(0.1)
    session.support.clear()
    before = state(session)
    nominal = session.nominal.copy()
    count = len(session.history)
    report = observe_environment(session, "floor")
    assert report["fixed_environment_support_fraction"] > 0.9
    assert report["surface_support_fraction"] > 0.9
    assert report["object_top_plane_clearance_m"] < 0.001
    assert report["fixed_environment_contacts"][0]["geom_name"] == "floor"
    np.testing.assert_array_equal(before, state(session))
    np.testing.assert_array_equal(nominal, session.nominal)
    assert len(session.history) == count
    assert session._contacts(session.data)[1] == 0


def test_height_follows_actual_object_not_commanded_arm_and_invalid_surface_rejected(
    tmp_path,
):
    session = make_session(tmp_path, "clearance")
    # Test fixture change, not an exposed policy command.
    address = session.model.jnt_qposadr[session.model.body_jntadr[session.object]]
    session.data.qpos[address + 2] += 0.03
    before = state(session)
    report = observe_environment(session, "floor")
    assert report["object_top_plane_clearance_m"] == pytest.approx(0.03, abs=1e-9)
    assert report["surface_support_fraction"] == 0
    np.testing.assert_array_equal(before, state(session))
    with pytest.raises(ValueError):
        observe_environment(session, "pad")


def test_observation_does_not_change_following_dynamics(tmp_path):
    a = make_session(tmp_path, "plain")
    b = make_session(tmp_path, "observed")
    for _ in range(5):
        observe_environment(b, "floor")
        a.hold(0.02)
        b.hold(0.02)
    np.testing.assert_array_equal(state(a), state(b))


def test_mesh_table_envelope_and_support(tmp_path):
    from types import SimpleNamespace

    model = mujoco.MjModel.from_xml_string("""<mujoco><asset>
    <mesh name="slab" vertex="-1 -1 -.1 1 -1 -.1 1 1 -.1 -1 1 -.1 -1 -1 0 1 -1 0 1 1 0 -1 1 0"/>
    </asset><worldbody><geom name="table" type="mesh" mesh="slab"/>
    <body name="object" pos="0 0 .049"><freejoint/><geom type="sphere" size=".05" mass=".2"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    session = SimpleNamespace(
        model=model, data=data, object=model.body("object").id, robot=set()
    )
    before = data.qpos.copy()
    result = observe_environment(session, "table")
    assert result["surface_plane_kind"] == "gravity_aligned_mesh_upper_envelope"
    assert result["object_top_plane_clearance_m"] == pytest.approx(-0.001, abs=1e-7)
    assert result["surface_up_force_n"] > 0
    assert result["top_footprint_aabb_overlap"]
    np.testing.assert_array_equal(data.qpos, before)
    data.qpos[0] = 3
    assert not observe_environment(session, "table")["top_footprint_aabb_overlap"]
