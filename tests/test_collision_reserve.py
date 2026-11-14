import numpy as np
import pytest


def test_reserve_jacobian_and_scratch_state():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.retargeting.collision_reserve import CollisionReserve

    model = mj.MjModel.from_xml_string(
        """<mujoco><worldbody><geom name="table" type="plane" size="1 1 .1"/><body name="hand"><joint name="z" type="slide" axis="0 0 1" range="0 .1"/><geom name="proximal" type="sphere" size=".01"/></body></worldbody></mujoco>"""
    )
    data = mj.MjData(model)
    data.qpos[0] = 0.05
    reserve = CollisionReserve(model, data, "hand", ["proximal"])
    r, j = reserve.evaluate([0.005])
    plus, _ = reserve.evaluate([0.005001])
    minus, _ = reserve.evaluate([0.004999])
    assert np.allclose(r, [2.0]) and np.allclose(
        j[:, 0], (plus - minus) / 0.000002, atol=1e-6
    )
    assert data.qpos[0] == 0.05
    with pytest.raises(ValueError):
        CollisionReserve(model, data, "hand", ["table"])


def test_empty_reserve_reproduces_fitter_and_additive_cost_changes_fit():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.retargeting.collision_reserve import CollisionReserve
    from real2sim2real.retargeting.pad_contacts import fit_pad_contacts

    body = '<body name="hand"><joint name="z" type="slide" axis="0 0 1" range="0 .1"/><geom name="proximal" type="sphere" size=".01"/></body>'
    plain = mj.MjModel.from_xml_string(
        "<mujoco><worldbody>" + body + "</worldbody></mujoco>"
    )
    d = mj.MjData(plain)
    context = mj.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="table" type="plane" size="1 1 .1"/>'
        + body
        + "</worldbody></mujoco>"
    )
    cd = mj.MjData(context)
    frames = [{"body": "hand", "point_body_m": [0, 0, 0], "normal_body": [0, 0, 1]}]
    args = (plain, d, "hand", frames, [[0, 0, 0.005]], [[0, 0, 1]], [0.005])
    old = fit_pad_contacts(*args)
    empty = fit_pad_contacts(*args, collision_reserves=())
    assert (
        np.array_equal(old["q"], empty["q"]) and old["objective"] == empty["objective"]
    )
    extra = fit_pad_contacts(
        *args, collision_reserves=[CollisionReserve(context, cd, "hand", ["proximal"])]
    )
    assert extra["q"][0] > old["q"][0] + 0.003
    assert np.array_equal(d.qpos, [0]) and np.array_equal(cd.qpos, [0])


def test_normal_weight_preserves_default_and_can_prioritize_orientation():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.retargeting.pad_contacts import fit_pad_contacts

    model = mj.MjModel.from_xml_string(
        """<mujoco><worldbody><body name="hand"><joint name="angle" type="hinge" axis="0 0 1" range="-90 90"/><geom type="sphere" size=".01"/></body></worldbody></mujoco>"""
    )
    data = mj.MjData(model)
    frames = [{"body": "hand", "point_body_m": [0.01, 0, 0], "normal_body": [1, 0, 0]}]
    args = (model, data, "hand", frames, [[0.01, 0, 0]], [[0, 1, 0]], [0.0])
    default = fit_pad_contacts(*args)
    explicit = fit_pad_contacts(*args, normal_scale=0.35)
    stronger = fit_pad_contacts(*args, normal_scale=0.1)
    assert np.array_equal(default["q"], explicit["q"])
    assert stronger["normal_errors_deg"][0] < default["normal_errors_deg"][0]
    assert np.array_equal(data.qpos, [0])
    for bad in [0, 0.09, 0.36, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            fit_pad_contacts(*args, normal_scale=bad)


def test_normal_guard_reduces_violation_without_claiming_feasibility():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.retargeting.pad_contacts import fit_pad_contacts

    model = mj.MjModel.from_xml_string(
        """<mujoco><worldbody><body name="hand"><joint name="angle" type="hinge" axis="0 0 1" range="-90 90"/><geom type="sphere" size=".01"/></body></worldbody></mujoco>"""
    )
    data = mj.MjData(model)
    frames = [dict(body="hand", point_body_m=[0.03, 0, 0], normal_body=[1, 0, 0])]
    args = (model, data, "hand", frames, [[0.03, 0, 0]], [[0, 1, 0]], [0.0])
    default = fit_pad_contacts(*args)
    omitted = fit_pad_contacts(*args, normal_guard_deg=None)
    guard = fit_pad_contacts(*args, normal_guard_deg=24)
    assert np.array_equal(default["q"], omitted["q"])
    assert guard["normal_errors_deg"][0] < default["normal_errors_deg"][0] - 10
    assert np.array_equal(data.qpos, [0])
    for bad in [0, 46, float("nan")]:
        with pytest.raises(ValueError):
            fit_pad_contacts(*args, normal_guard_deg=bad)
