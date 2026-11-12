from types import SimpleNamespace
import json

import numpy as np
import pytest

from real2sim2real.artifacts import digest
from real2sim2real.controller_actor_training.bounded_grasp_correction import (
    validate,
    finger_command,
    wrist_targets,
    command_amplitude,
    bind_fingers,
)


def spec():
    return dict(
        baseline={},
        vectors_object_m={r: [0, 0, 0] for r in ("right", "left")},
        rotation_vectors_object_rad={r: [0, 0, 0] for r in ("right", "left")},
        finger_joint_offsets_rad={},
    )


def test_probe_requires_pinned_source_and_frozen_physical_settings(tmp_path):
    path = tmp_path / "result.json"
    (tmp_path / "states.npz").write_bytes(b"fixed states")
    (tmp_path / "scene.xml").write_text("<mujoco/>")
    config = dict(output="original", control_period_s=0.001, kp_nm_per_rad=1200)
    baseline = dict(
        physics_run=True,
        motion_condition="demonstrated",
        rejected_for=["hold_slip"],
        config=config,
        continuous_protocol=dict(release_planned=True),
        states_sha256=digest(tmp_path / "states.npz"),
        scene_sha256=digest(tmp_path / "scene.xml"),
    )
    path.write_text(json.dumps(baseline))
    s = spec()
    s["baseline"] = dict(path=str(path), sha256=digest(path))
    c = dict(config, output="probe", bounded_grasp_correction=s)
    validate(s, c)
    with pytest.raises(ValueError, match="retune"):
        validate(s, dict(c, control_period_s=0.002))
    bad = dict(s, vectors_object_m={"right": [0.007, 0, 0], "left": [0, 0, 0]})
    with pytest.raises(ValueError, match="bounds"):
        validate(bad, c)
    with pytest.raises(ValueError, match="finger"):
        validate(dict(s, finger_joint_offsets_rad={"object_free": 0.01}), c)
    (tmp_path / "states.npz").write_bytes(b"changed states")
    with pytest.raises(ValueError, match="physical artifacts"):
        validate(s, c)


def test_owned_posture_envelope_preserves_other_commands_and_release():
    live = SimpleNamespace(
        rollout=dict(approach_s=1, close_s=0.8),
        correction_protocol=dict(
            capture_duration_s=12, finger_release=dict(start_s=13)
        ),
        hands=[
            dict(
                qids=np.array([2, 4]),
                bounded_posture_offset_rad=np.array([0.02, -0.02]),
            )
        ],
    )
    original = np.arange(7, dtype=float)
    for t in [0, 1, 13, 14]:
        np.testing.assert_array_equal(
            finger_command(live, t, original.copy()), original
        )
    command = finger_command(live, 2, original.copy())
    np.testing.assert_allclose(command - original, [0, 0, 0.02, 0, -0.02, 0, 0])
    halfway = finger_command(live, 12.5, original.copy())
    np.testing.assert_allclose(halfway - original, (command - original) * 0.5)


def test_stationary_correction_does_not_fade_during_complete_hold():
    rollout = dict(approach_s=1, close_s=0.8)
    protocol = dict(capture_duration_s=12, finger_release=dict(start_s=13))
    assert command_amplitude(14, rollout, protocol) == 0
    protocol["release_disabled_for_stationary_diagnostic"] = True
    assert command_amplitude(14, rollout, protocol) == 1
    assert command_amplitude(16, rollout, protocol) == 1


def test_finger_binding_is_atomic_and_rejects_excessive_correction_rate():
    model = SimpleNamespace(
        dof_jntid=np.array([0]), joint=lambda j: SimpleNamespace(name="right_index_PIP")
    )
    hand = dict(qids=np.array([0]), waypoints=[np.array([0.2])] * 3)
    live = SimpleNamespace(
        rollout=dict(approach_s=1, close_s=0.1),
        hands=[hand],
        servo=SimpleNamespace(pads=[dict(qids=[0], dofs=[0], limits=[[0, 1]])]),
    )
    s = spec()
    s["finger_joint_offsets_rad"] = {"right_index_PIP": 0.06}
    protocol = dict(capture_duration_s=12, finger_release=dict(start_s=13))
    with pytest.raises(ValueError, match="speed"):
        bind_fingers(live, model, s, protocol)
    assert "bounded_posture_offset_rad" not in hand
    assert not hasattr(live, "correction_protocol")
    live.rollout["close_s"] = 0.8
    before = [a.copy() for a in hand["waypoints"]]
    bind_fingers(live, model, s, protocol)
    for a, b in zip(before, hand["waypoints"]):
        np.testing.assert_array_equal(a, b)


def test_wrist_scratch_ik_respects_metric_frames_and_restores_release():
    mj = pytest.importorskip("mujoco")
    bodies = []
    for role, x in [("right", -0.2), ("left", 0.2)]:
        joints = "".join(
            f'<joint name="{role}_{k}" type="{kind}" axis="{axis}" range="-1 1"/>'
            for k, (kind, axis) in enumerate(
                [
                    ("slide", "1 0 0"),
                    ("slide", "0 1 0"),
                    ("slide", "0 0 1"),
                    ("hinge", "1 0 0"),
                    ("hinge", "0 1 0"),
                    ("hinge", "0 0 1"),
                ]
            )
        )
        bodies.append(
            f'<body name="{role}" pos="{x} 0 .5">{joints}<geom size=".02" mass="1"/></body>'
        )
    model = mj.MjModel.from_xml_string(
        '<mujoco><compiler autolimits="true"/><worldbody>'
        + "".join(bodies)
        + "</worldbody></mujoco>"
    )
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    arms = [
        dict(name=r, joints=[f"{r}_{i}" for i in range(6)]) for r in ["right", "left"]
    ]
    qids = np.array([model.joint(n).qposadr[0] for a in arms for n in a["joints"]])
    vids = qids.copy()
    roots = {r: model.body(r).id for r in ["right", "left"]}
    times = np.array([0, 1, 1.8, 8, 12, 12.999, 13, 14])
    targets = np.zeros((len(times), 12))
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    parent = dict(
        records=[dict(reference_rotation=rotation)] * len(times),
        configuration=dict(rollout=dict(approach_s=1, close_s=0.8)),
        continuous_protocol=dict(
            capture_duration_s=12, finger_release=dict(start_s=13)
        ),
    )
    s = spec()
    s["vectors_object_m"]["right"] = [0.002, 0, 0]
    before = [data.qpos.copy(), data.qvel.copy(), model.jnt_range.copy()]
    result, receipt = wrist_targets(
        model,
        data,
        targets,
        times,
        qids,
        vids,
        arms,
        roots,
        parent,
        s,
        np.ones(12),
        0.01,
    )
    assert receipt["accepted"]
    assert result[2, 1] == pytest.approx(0.002, abs=1e-5)  # Object X maps to world Y.
    np.testing.assert_array_equal(result[[0, 1, 5, 6, 7]], targets[[0, 1, 5, 6, 7]])
    for a, b in zip([data.qpos, data.qvel, model.jnt_range], before):
        np.testing.assert_array_equal(a, b)
    # A coarse ramp that would interpolate nonzero correction into release
    # must be rejected rather than silently modify the original release path.
    coarse_times = np.delete(times, 5)
    parent["records"] = [dict(reference_rotation=rotation)] * len(coarse_times)
    _, coarse = wrist_targets(
        model,
        data,
        np.zeros((len(coarse_times), 12)),
        coarse_times,
        qids,
        vids,
        arms,
        roots,
        parent,
        s,
        np.ones(12),
        0.01,
    )
    assert not coarse["accepted"] and not coarse["corrections_zero_before_release"]
