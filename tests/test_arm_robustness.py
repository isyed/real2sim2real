import json
import numpy as np
import pytest
from real2sim2real.artifacts import digest
from real2sim2real.controller_actor_training.arm_robustness import (
    apply_case,
    validate_case,
)


def test_frozen_controller_and_bounds(tmp_path):
    states = tmp_path / "states.npz"
    states.write_bytes(b"fixture")
    result = tmp_path / "result.json"
    config = dict(
        output="original", timestep_s=0.00025, gains=[1200, 24], audit="fixed"
    )
    result.write_text(
        json.dumps(dict(success=True, states_sha256=digest(states), config=config))
    )
    case = dict(
        mass_scale=1.25, sliding_friction=0.6, initial_translation_m=[0.002, 0, 0]
    )
    probe = dict(
        config,
        output="new",
        simulation_case=case,
        frozen_controller=dict(result=str(result), sha256=digest(result)),
    )
    assert validate_case(probe) == case
    content = json.loads(result.read_text())
    result.write_text(
        json.dumps(dict(content, residual_adapter={"all_actions_zero": False}))
    )
    with pytest.raises(ValueError, match="policy-aware"):
        validate_case(
            dict(
                probe, frozen_controller=dict(result=str(result), sha256=digest(result))
            )
        )
    result.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="may not change"):
        validate_case(dict(probe, gains=[1500, 24]))
    with pytest.raises(ValueError, match="bounds"):
        validate_case(
            dict(probe, simulation_case=dict(case, initial_translation_m=[0, 0, 0.001]))
        )
    states.write_bytes(b"changed")
    with pytest.raises(ValueError, match="pinned passing"):
        validate_case(probe)


def test_physical_change_preserves_initialized_robot_and_velocity():
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="robot"><joint name="arm"/><geom size=".02" mass="1"/></body>
    <body name="task_object" pos="0 0 1"><freejoint name="task_object_free"/><geom size=".02" mass=".2"/></body>
    </worldbody><actuator><motor joint="arm"/></actuator></mujoco>""")
    data = mj.MjData(model)
    data.qpos[0] = 0.8
    data.qvel[:] = 0.1
    data.ctrl[:] = 0.3
    data.time = 0.01
    original = data.qpos.copy()
    apply_case(
        model,
        data,
        dict(mass_scale=1, sliding_friction=None, initial_translation_m=[0, 0, 0]),
    )
    np.testing.assert_array_equal(data.qpos, original)
    np.testing.assert_array_equal(data.qvel, 0.1)
    np.testing.assert_array_equal(data.ctrl, 0.3)
    assert data.time == 0.01
    apply_case(
        model,
        data,
        dict(
            mass_scale=1.25, sliding_friction=0.4, initial_translation_m=[0.002, 0, 0]
        ),
    )
    assert data.qpos[0] == 0.8
    assert data.qpos[1] == 0.002
    assert model.body_mass[model.body("task_object").id] == pytest.approx(0.25)
    np.testing.assert_array_equal(model.geom_friction[:, 0], 0.4)
