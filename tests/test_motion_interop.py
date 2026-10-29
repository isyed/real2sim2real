import shutil

import numpy as np

from flash_chord.data.parquet import read_parquet_row
from flash_chord.assets.registry import resolve_asset_path


def test_motion_roundtrip_and_moved_object_paths(tmp_path):
    try:
        import torch
    except ImportError:
        import pytest
        pytest.skip('Motion schema decoding requires the retargeting Torch environment')
    from real2sim2real.retargeting.motion_schema import (
        MotionData, load_motion_data_parquet, save_motion_parquet,
    )
    out = tmp_path / "source"
    out.mkdir()
    (out / "mesh.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    n = 4
    motion = MotionData(
        sequence_id="synthetic",
        robot_name="test_robot",
        motion_kind="single_robot",
        source_dataset="user",
        raw_motion_file="source",
        fps=30.0,
        coord_frame="robot_base_z_up",
        robot_joint_names=["joint"],
        robot_root_position=[[0.0, 0.0, 0.8]] * n,
        robot_root_wxyz=[[1.0, 0.0, 0.0, 0.0]] * n,
        robot_joint_positions=[[0.0]] * n,
        ee_link_names=["palm"],
        ee_pose_w=[[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]] * n,
        object_body_names=["object"],
        object_body_position=[[[0.3, 0.0, 0.4]]] * n,
        object_body_wxyz=[[[1.0, 0.0, 0.0, 0.0]]] * n,
        object_mesh_paths=["mesh.obj"],
        object_mesh_radius=[0.5],
    )
    partition = save_motion_parquet(
        motion, out / "partitioned", partition_cols=[], file_name="reference.parquet"
    )
    shutil.copy2(partition / "reference.parquet", out / "reference.parquet")
    loaded = load_motion_data_parquet(out / "reference.parquet")
    assert np.asarray(loaded.robot_joint_positions).shape == (n, 1)
    moved = tmp_path / "moved"
    out.rename(moved)
    row = read_parquet_row(moved / "reference.parquet")
    assert row.cells["sequence_id"] == "synthetic"
    assert resolve_asset_path(
        row.cells["object_mesh_paths"][0], reference_path=row.path
    ) == str(moved / "mesh.obj")
