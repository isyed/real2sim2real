import numpy as np
import pytest


def test_loaded_proposal_respects_bounds_and_preserves_live_state():
    mj = pytest.importorskip("mujoco")
    from real2sim2real.retargeting.loaded_contact_repair import propose

    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body name="hand"><joint name="z" type="slide" axis="0 0 1" range="0 .1"/>
        <geom name="proximal" type="sphere" size=".01"/>
      </body></worldbody></mujoco>""")
    data = mj.MjData(model)
    data.qpos[0], data.qvel[0], data.time = 0.008, 0.02, 3.0
    frames = [{"body": "hand", "point_body_m": [0, 0, 0], "normal_body": [0, 0, 1]}]
    result = propose(
        model,
        data,
        "hand",
        frames,
        {"z": [0.007, 0.009]},
        ["proximal"],
        position_scale_m=0.003,
        reserve_scale_m=0.0005,
    )
    assert 0.008 < result["q"][0] <= 0.009
    assert np.allclose(result["retained_pad_points_m"], [[0, 0, 0.008]])
    assert not result["loaded_retention_validated"]
    assert data.qpos[0] == 0.008 and data.qvel[0] == 0.02 and data.time == 3.0
    with pytest.raises(ValueError, match="every owned joint"):
        propose(model, data, "hand", frames, {}, ["proximal"])
