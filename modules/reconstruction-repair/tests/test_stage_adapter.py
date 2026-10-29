import json

import numpy as np

from conftest import box_geometry, observations
from real2sim2real.artifacts import publish, load, role_path
from real2sim2real.reconstruction_repair.stage import run


def test_artifact_to_actual_repair_engine_and_back(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    geometry = box_geometry()
    obs = observations(geometry)
    np.savez_compressed(source / "geometry.npz", **geometry)
    n = len(geometry["frame_ids"])
    np.savez_compressed(
        source / "human.npz",
        mhr_model_params=np.zeros((n, 204)),
        shape_params=np.zeros((n, 45)),
        face_params=np.ones((n, 72)),
    )
    publish(
        source,
        "reconstruction",
        {"geometry": source / "geometry.npz", "human": source / "human.npz"},
        metadata={"sequence_id": "synthetic", "world_frame_id": "camera"},
    )
    np.savez_compressed(tmp_path / "observations.npz", **obs)
    np.savez_compressed(
        tmp_path / "metric.npz",
        frame_ids=geometry["frame_ids"],
        depth_m=np.full(obs["object_mask"].shape, 2.0),
        fit=np.arange(n) % 2 == 0,
        floor_planes=np.tile([0.0, -1.0, 0.0, 1.5], (n, 1)),
        floor_valid=np.ones(n, bool),
    )
    out = tmp_path / "result"
    out.mkdir()
    run(
        dict(
            input=str(source),
            output=str(out),
            config_base=str(tmp_path),
            options=dict(
                observations="observations.npz",
                metric_evidence="metric.npz",
                floor_roi_source="synthetic test floor",
                evidence={"hand_points_source": "procedural fixture"},
            ),
        )
    )
    artifact = load(out)
    report = json.loads(role_path(out, "diagnostics").read_text())
    assert artifact["kind"] == "reconstruction"
    assert report["metric_size_verified"]
    with np.load(role_path(out, "human")) as human:
        assert np.array_equal(human["face_params"], np.ones((n, 72)))
