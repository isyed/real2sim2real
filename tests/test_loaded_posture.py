import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from real2sim2real.retargeting.grasp_kinematics import robot_indices
from real2sim2real.retargeting.loaded_posture import (
    fit_loaded_pad_offset,
    minimum_geom_separation,
)
from real2sim2real.retargeting.pad_contacts import observe_frames


def scene():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom name="floor" type="plane" size="1 1 .01"/>
    <body name="hand" pos="0 0 .3"><joint name="base" axis="0 0 1" range="-90 90"/>
    <geom name="palm" type="sphere" size=".01"/>
    <body name="finger" pos=".05 0 0"><joint name="finger" axis="0 0 1" range="-30 30"/>
    <geom name="pad" type="sphere" size=".01"/></body></body>
    <body name="other" pos=".3 0 .3"><joint name="other" axis="0 0 1" range="-30 30"/>
    <geom name="other_pad" type="sphere" size=".01"/></body>
    </worldbody></mujoco>""")
    frames = [{"body": "finger", "point_body_m": [0, 0, 0], "normal_body": [1, 0, 0]}]
    return model, frames


def test_shared_offset_improves_loaded_normals_without_steps_or_input_mutation(
    monkeypatch,
):
    model, frames = scene()
    snapshots = np.array([[0, -0.1, 0], [0.1, 0.1, 0]])
    original = snapshots.copy()
    data = mujoco.MjData(model)
    desired = []
    for q in snapshots:
        data.qpos[:] = q
        data.qpos[1] += 0.015
        mujoco.mj_fwdPosition(model, data)
        desired.append(
            observe_frames(model, data, frames, robot_indices(model, "hand"))[1]
        )
    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("dynamic step"))
    result = fit_loaded_pad_offset(
        model, "hand", frames, snapshots, ["finger"], desired
    )
    assert 0.005 < result["offsets_rad"][0] < 0.016
    assert result["converged"] and not result["dynamic_retention_validated"]
    np.testing.assert_array_equal(snapshots, original)
    # Native command margin narrows the solution even if snapshot motion allows it.
    limited = fit_loaded_pad_offset(
        model,
        "hand",
        frames,
        snapshots,
        ["finger"],
        desired,
        command_samples_rad=[[model.jnt_range[1, 1] - 0.002]],
    )
    assert limited["offsets_rad"][0] <= 0.002 + 1e-9


def test_rejects_unowned_joint_and_incompatible_loaded_joint_ranges():
    model, frames = scene()
    targets = np.array([[[1, 0, 0]]])
    with pytest.raises(ValueError, match="owned"):
        fit_loaded_pad_offset(model, "hand", frames, [[0, 0, 0]], ["other"], targets)
    with pytest.raises(ValueError, match="shared native"):
        fit_loaded_pad_offset(model, "hand", frames, [[0, 2, 0]], ["finger"], targets)


def test_clearance_includes_plane_and_reports_cutoff_as_bound():
    model, _ = scene()
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    pad, floor = model.geom("pad").id, model.geom("floor").id
    clear = minimum_geom_separation(model, data, [pad], [floor], cutoff_m=0.5)
    assert clear["distance_m"] == pytest.approx(0.29)
    assert clear["pair"] == ["pad", "floor"]
    far = minimum_geom_separation(model, data, [pad], [floor], cutoff_m=0.02)
    assert far["distance_m"] == 0.02 and far["capped_at_cutoff"]
    model.body_pos[model.body("hand").id, 2] = 0.005
    mujoco.mj_fwdPosition(model, data)
    overlap = minimum_geom_separation(model, data, [pad], [floor])
    assert overlap["distance_m"] == pytest.approx(-0.005)


def test_normal_regression_guard_reduces_average_fit_tradeoff():
    model, frames = scene()
    desired = [[[np.cos(a), np.sin(a), 0]] for a in [0.04, -0.01]]
    inputs = (model, "hand", frames, np.zeros((2, 3)), ["finger"], desired)
    plain = fit_loaded_pad_offset(*inputs)
    guarded = fit_loaded_pad_offset(*inputs, normal_regression_deg=0)
    # Improving the first target by positive rotation harms the second target.
    # The guard penalizes that trade-off; post-fit validation is still required.
    assert 0 < guarded["offsets_rad"][0] < plain["offsets_rad"][0]
    with pytest.raises(ValueError, match="regression"):
        fit_loaded_pad_offset(*inputs, normal_regression_deg=float("nan"))
