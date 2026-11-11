from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest
from real2sim2real.controller_actor_training.acquisition_compare import (
    PHASES,
    compare,
    describe,
    load_audit,
    relative_object,
    review,
    validate_records,
)


def settings():
    return dict(
        minimum_force_n=0.1,
        sliding_speed_m_s=0.001,
        rolling_speed_rad_s=0.02,
        maximum_sample_gap_s=0.05,
    )


def records():
    result = []
    for i in range(8):
        pads = {
            name: dict(
                force_n=2.0,
                loaded=True,
                object_local_centroid_m=[0, 0, 0],
                pad_local_centroid_m=[0, 0, 0],
                tangent_speed_m_s=0.002,
            )
            for name in ("primary:thumb", "secondary:thumb")
        }
        servo = {
            name: dict(
                force_n=0.5,
                target_n=3.0,
                joint_offset_rad=[0.01, 0],
                updated_steps=i + 1,
                frozen_steps=0,
            )
            for name in pads
        }
        result.append(
            dict(
                time_s=i * 0.02,
                phase=PHASES[i // 2],
                contact_patch_motion=dict(pads=pads, other_object_contacts_n={}),
                contact_force_feedback=servo,
                object_position_m=[0, 0, 1],
                object_rotation=np.eye(3).tolist(),
                wrist_position_m=[0.1, 0, 0],
                wrist_rotation=np.eye(3).tolist(),
                support_wrist_position_m=[-0.1, 0, 0],
                support_wrist_rotation=np.eye(3).tolist(),
                table_contacts=0,
                payload_pressure=dict(valid_updates=i),
            )
        )
    return result


def trial():
    hand = dict(
        joint_names=["root", "finger"],
        base_position_m=[0, 0, 0],
        base_rotation=np.eye(3).tolist(),
        targets_m=[[0, 0, 0]],
        q_open=[0, 0],
    )
    return dict(
        records=records(),
        source_scene_sha256="scene",
        mass_scale=0.75,
        friction=0.8,
        simulation_timestep_s=0.001,
        simulation_requested_duration_s=1.0,
        success=True,
        configuration=dict(physics=dict(friction=0.8)),
        candidate=dict(**hand, support=deepcopy(hand)),
    )


def test_phase_integration_and_coherent_force_deficit_do_not_use_old_servo_force():
    r = records()
    report = describe(r, settings())
    for phase in PHASES:
        p = report["phases"][phase]["pads"]["primary:thumb"]
        assert p["sampled_sliding_path_m"] == pytest.approx(0.00004)
        assert p["active_mean_force_deficit_n"] == 1.0
        assert p["mean_coherent_force_n"] == 2.0
        assert p["object_centroid_migration_m"] == 0.0
    # Do not integrate a contact across an unloaded sample or a long clock gap.
    r[6]["contact_patch_motion"]["pads"]["primary:thumb"] = dict(
        force_n=0.0, loaded=False
    )
    assert (
        describe(r, settings())["phases"]["hold"]["pads"]["primary:thumb"][
            "sampled_sliding_path_m"
        ]
        == 0
    )
    r = records()
    r[7]["time_s"] += 0.1
    assert (
        describe(r, settings())["phases"]["hold"]["pads"]["primary:thumb"][
            "sampled_sliding_path_m"
        ]
        == 0
    )


def test_arm_retention_timeline_keeps_native_relative_drift_and_freeze_counts():
    from real2sim2real.controller_actor_training.arm_retention_audit import (
        retention_timeline,
    )

    source = records()
    for row in source:
        for pad in row["contact_patch_motion"]["pads"].values():
            pad.update(
                sliding=True, rolling_compatible=False, object_collision_geoms=[1]
            )
    source[-1]["object_position_m"][0] = 0.006
    source[-1]["contact_force_feedback"]["primary:thumb"]["frozen_steps"] = 7
    before = deepcopy(source)
    report = retention_timeline(source, settings())
    assert report["timeline"][-1]["relative_change_from_first_hold"]["primary"][
        "translation_m"
    ] == pytest.approx(0.006)
    assert report["timeline"][-1]["frozen_pad_counts"]["primary:thumb"] == 7
    assert source == before


def test_world_transform_invariance_and_no_candidate_promotion():
    a, b = trial(), trial()
    b["success"] = False
    q = Rotation.from_euler("xyz", [0.4, -0.2, 0.8]).as_matrix()
    shift = np.array([0.2, -0.1, 0.7])
    for row in b["records"]:
        for key in ("object", "wrist", "support_wrist"):
            row[key + "_position_m"] = (q @ row[key + "_position_m"] + shift).tolist()
            row[key + "_rotation"] = (q @ row[key + "_rotation"]).tolist()
    for role in ("primary", "secondary"):
        np.testing.assert_allclose(
            relative_object(a["records"][0], role)[0],
            relative_object(b["records"][0], role)[0],
            atol=1e-14,
        )
    out = compare(a, b, settings(), [])
    assert out["original_physical_verdicts"] == dict(baseline=True, candidate=False)
    assert out["source_verdicts_promoted"] is False
    for point in out["sampled_pair_differences"]:
        assert (
            point["wrist_relative_object_difference"]["primary"]["translation_m"]
            < 1e-14
        )


def test_unloaded_centroid_is_missing_not_zero_separation_and_changes_are_explicit():
    a, b = trial(), trial()
    b["records"][2]["contact_patch_motion"]["pads"]["primary:thumb"] = dict(
        force_n=0, loaded=False
    )
    b["configuration"]["posture"] = dict(ratio=0.8)
    with pytest.raises(ValueError, match="Undeclared"):
        compare(a, b, settings(), [])
    out = compare(a, b, settings(), ["posture"])
    assert (
        out["sampled_pair_differences"][2]["pads"]["primary:thumb"][
            "object_centroid_separation_m"
        ]
        is None
    )
    assert out["actual_configuration_changes"] == ["posture"]


@pytest.mark.parametrize("mutation", ["clock", "phase", "force", "centroid", "loading"])
def test_invalid_diagnostic_observations_are_rejected(mutation):
    r = records()
    if mutation == "clock":
        r[2]["time_s"] = r[1]["time_s"]
    if mutation == "phase":
        r[2]["phase"] = "hold"
    if mutation == "force":
        r[2]["contact_patch_motion"]["pads"]["primary:thumb"]["force_n"] = float("nan")
    if mutation == "centroid":
        r[2]["contact_patch_motion"]["pads"]["primary:thumb"][
            "object_local_centroid_m"
        ] = [float("inf"), 0, 0]
    if mutation == "loading":
        r[2]["contact_patch_motion"]["pads"]["primary:thumb"]["loaded"] = False
    with pytest.raises(ValueError):
        validate_records(r, 0.1)


@pytest.mark.parametrize(
    "field", ["mass_scale", "friction", "source_scene_sha256", "simulation_timestep_s"]
)
def test_unmatched_physics_are_rejected(field):
    a, b = trial(), trial()
    b[field] = "changed"
    with pytest.raises(ValueError, match="scene, load, friction and clock"):
        compare(a, b, settings(), [])


def test_wrist_or_open_posture_change_cannot_hide_in_declared_controller_change():
    a, b = trial(), trial()
    b["candidate"]["support"]["q_open"][1] += 0.1
    with pytest.raises(ValueError, match="Fixed wrists"):
        compare(a, b, settings(), [])


def test_closed_fixture_motion_cannot_hide_in_a_finger_posture_change():
    a, b = trial(), trial()
    for t in (a, b):
        hand = t["candidate"]
        hand["joint_names"] = [
            "fixture_" + name for name in ("x", "y", "z", "rx", "ry", "rz")
        ] + ["finger"]
        hand["q_open"] = [0] * 7
        hand["q_grasp"] = [0] * 7
        hand["q_closed"] = [0] * 7
    b["candidate"]["q_closed"][0] = 0.01
    with pytest.raises(ValueError, match="fixture coordinates changed"):
        compare(a, b, settings(), [])
    a, b = trial(), trial()
    b["records"][2]["time_s"] += 0.001
    with pytest.raises(ValueError, match="Matched native"):
        compare(a, b, settings(), [])


def test_audit_requires_immutable_state_identical_replay_and_complete_source(tmp_path):
    source_dir, replay_dir = tmp_path / "source", tmp_path / "audit/replay"
    source_dir.mkdir()
    replay_dir.mkdir(parents=True)
    for p in (source_dir, replay_dir):
        (p / "states.npz").write_bytes(b"saved states")
    source = dict(
        states_sha256=digest(source_dir / "states.npz"),
        simulation_requested_duration_s=1.0,
        simulation_final_time_s=1.0,
        success=False,
    )
    for p in (source_dir, replay_dir):
        (p / "result.json").write_text(json.dumps(source))
    audit = dict(
        config=dict(
            source_result=str(source_dir / "result.json"),
            source_sha256=digest(source_dir / "result.json"),
            diagnostic_settings=settings(),
        ),
        replay_result_sha256=digest(replay_dir / "result.json"),
        states_identical=True,
        source_states_sha256=source["states_sha256"],
    )
    path = tmp_path / "audit/audit.json"
    path.write_text(json.dumps(audit))
    assert load_audit(path, digest(path))[0]["success"] is False
    audit["states_identical"] = False
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="Byte-identical"):
        load_audit(path, digest(path))
    audit["states_identical"] = True
    path.write_text(json.dumps(audit))
    (replay_dir / "states.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Byte-identical"):
        load_audit(path, digest(path))


def test_viewer_embeds_data_without_script_termination(tmp_path):
    path = tmp_path / "review.html"
    review(dict(label="</script><script>unsafe</script>"), {}, path)
    text = path.read_text()
    assert "</script><script>unsafe" not in text
    assert r"\u003c/script>" in text
    assert "__DATA__" not in text
