import numpy as np
import pytest


def test_free_object_fork_preserves_geometry_and_has_no_assistance(tmp_path):
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.conditional_fixture_scene import (
        prepare,
    )
    from real2sim2real.controller_actor_training.object_grasp_pilot import (
        verify_unassisted_model,
    )

    joints = "".join(
        f'<joint name="fixture_{prefix}{a}" type="{kind}" axis="{v}" range="-1 1"/>'
        for prefix, kind in [("", "slide"), ("r", "hinge")]
        for a, v in [("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")]
    )
    source = tmp_path / "source.xml"
    output = tmp_path / "free.xml"
    source.write_text(
        f'<mujoco><compiler angle="radian"/><worldbody><geom name="cup" type="box" size=".02 .02 .03" pos=".1 0 .03"/><geom name="table" type="plane" size="1 1 .1"/><body name="hand" pos="0 0 .2">{joints}<geom type="sphere" size=".01"/><body><joint name="finger" range="-1 1"/><geom type="sphere" size=".01" pos=".03 0 0"/></body></body></worldbody><actuator><position joint="finger" kp="2" ctrlrange="-1 1"/></actuator></mujoco>'
    )
    before = source.read_bytes()
    args = dict(
        object_geoms=["cup"],
        center_of_mass_m=[0.1, 0, 0.03],
        mass_kg=0.2,
        diagonal_inertia_kg_m2=[0.001] * 3,
        friction=0.6,
    )
    prepare(source, output, **args)
    original = mj.MjModel.from_xml_path(str(source))
    new = mj.MjModel.from_xml_path(str(output))
    d = mj.MjData(new)
    old = mj.MjData(original)
    mj.mj_forward(new, d)
    mj.mj_forward(original, old)
    verify_unassisted_model(new)
    assert source.read_bytes() == before and new.ngeom == original.ngeom
    for name in ["cup", "table"]:
        np.testing.assert_allclose(
            d.geom_xpos[new.geom(name).id], old.geom_xpos[original.geom(name).id]
        )
    assert new.nu == 7 and new.actuator_forcelimited.all()
    assert new.body_mass[new.body("task_object").id] == 0.2
    with pytest.raises(FileExistsError):
        prepare(source, output, **args)
    with pytest.raises(ValueError):
        prepare(source, tmp_path / "bad.xml", **dict(args, mass_kg=float("nan")))


def test_initial_overlap_check_is_separate_from_hand_path_and_preserves_state():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.controller_actor_training.conditional_fixture_scene import (
        object_obstacle_preflight,
    )

    m = mj.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="rack" type="box" size=".02 .02 .02" pos="0 0 .02"/><body name="task_object"><freejoint/><geom name="cup" type="sphere" size=".02" pos="0 0 .03" mass=".2"/></body></worldbody></mujoco>'
    )
    d = mj.MjData(m)
    before = d.qpos.copy()
    audit = object_obstacle_preflight(m, d, ["rack"])
    assert not audit["passed"] and audit["maximum_penetration_m"] > 0.003
    np.testing.assert_array_equal(d.qpos, before)
    d.qpos[0] = 0.2
    assert object_obstacle_preflight(m, d, ["rack"])["passed"]
