import json
import shutil

import numpy as np
import pytest

from real2sim2real.artifacts import digest, publish, write_json
from real2sim2real.egocentric.hand_comparison import compare


@pytest.fixture
def refinement_pair(tmp_path):
    parent, candidate = tmp_path / "parent", tmp_path / "candidate"
    parent.mkdir()
    joints = np.zeros((3, 2, 21, 3))
    joints[:, :, :, 0] = np.arange(3)[:, None, None] * 0.01
    accepted = np.ones((3, 2), bool)
    accepted[1, 0] = False
    data = dict(
        accepted=accepted,
        frame_ids=np.array([40, 41, 42]),
        timestamps_s=np.array([0, 0.1, 0.2]),
        scale_per_hand=np.array([0.8, 0.9]),
        proposed_camera_joints=joints.copy(),
    )
    geometry = ("camera_vertices", "camera_joints", "world_vertices", "world_joints")
    for key in geometry:
        data[key] = joints.copy()
        data[key][~accepted] = np.nan
    (parent / "hands_metric").mkdir()
    np.savez(parent / "hands_metric/hands.npz", **data)
    metric = dict(
        reprojection_median_px=12,
        reprojection_p90_px=30,
        validation_surface_median_m=0.005,
        validation_surface_p90_m=0.01,
    )
    quality = dict(limits={"max_reprojection_p90_px": 25}, records=[[metric] * 2] * 3)
    write_json(parent / "hands_metric/quality.json", quality)
    write_json(parent / "support/support.json", {"units": "m"})
    np.savez(
        parent / "support/scene.npz", world_from_camera=np.tile(np.eye(4), (3, 1, 1))
    )
    for name in (
        "support/tabletop.glb",
        "hands_raw/predictions.npz",
        "hands2d/landmarks.npz",
    ):
        path = parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unchanged fixture")
    publish(
        parent,
        "metric-interaction-reconstruction",
        {str(p.relative_to(parent)): p for p in parent.rglob("*") if p.is_file()},
    )
    shutil.copytree(parent, candidate)
    (candidate / "artifact.json").unlink()
    data["accepted"] = np.ones_like(accepted)
    for key in geometry:
        data[key][1, 0] = joints[1, 0]
    np.savez(candidate / "hands_metric/hands.npz", **data)
    quality = json.loads((candidate / "hands_metric/quality.json").read_text())
    quality["records"][1][0]["reprojection_median_px"] = 3
    quality["refinement_parent_sha256"] = digest(parent / "artifact.json")
    write_json(candidate / "hands_metric/quality.json", quality)
    write_json(
        candidate / "contacts/diagnostics.json",
        dict(
            summary={
                "sources": {"hands": digest(candidate / "hands_metric/hands.npz")}
            },
            records=[{"hands": [{"object": {"penetration_depth_m": 0.001}}] * 2}] * 3,
        ),
    )
    return parent, candidate


def test_comparison_pairs_recovered_frames_and_preserves_existing_report(
    refinement_pair,
):
    parent, candidate = refinement_pair
    result = compare(parent, candidate)
    left, right = result["hands"]["left"], result["hands"]["right"]
    assert left["recovered_frame_ids"] == [41]
    assert left["remaining_intervals"] == []
    assert left["new_temporal_edges"] == 2
    assert left["new_edge_joint_speed_m_s"]["refined"]["maximum"] == pytest.approx(0.1)
    residual = left["recovered_fit_metrics"]["reprojection_median_px"]
    assert residual["parent_proposals"]["median"] == 12
    assert residual["refined"]["median"] == 3
    assert right["recovered_frames"] == 0
    assert (
        right["recovered_fit_metrics"]["reprojection_median_px"]["refined"]["count"]
        == 0
    )
    assert not result["control_ready"]
    assert not result["ground_truth_hand_accuracy_established"]
    with pytest.raises(FileExistsError):
        compare(parent, candidate)


@pytest.mark.parametrize(
    "change", ["geometry", "acceptance", "limits", "source", "contacts"]
)
def test_comparison_refuses_changed_anchors_sources_or_contact_binding(
    refinement_pair, change
):
    parent, candidate = refinement_pair
    if change in ("geometry", "acceptance"):
        path = candidate / "hands_metric/hands.npz"
        with np.load(path) as archive:
            data = {key: archive[key] for key in archive.files}
        if change == "geometry":
            data["camera_joints"][0, 0, 0, 0] += 0.01
        else:
            data["accepted"][0, 0] = False
        np.savez(path, **data)
    elif change == "limits":
        path = candidate / "hands_metric/quality.json"
        quality = json.loads(path.read_text())
        quality["limits"]["max_reprojection_p90_px"] = 50
        write_json(path, quality)
    elif change == "source":
        (candidate / "hands_raw/predictions.npz").write_bytes(
            b"different shape or pose"
        )
    else:
        path = candidate / "contacts/diagnostics.json"
        contacts = json.loads(path.read_text())
        contacts["summary"]["sources"]["hands"] = "outdated"
        write_json(path, contacts)
    with pytest.raises(ValueError, match="changed|lost|different hands"):
        compare(parent, candidate)
    assert not (candidate / "hand_comparison.json").exists()
