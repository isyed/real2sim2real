"""Configurable nested-object Parquet reference reader; evaluation only."""

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def read_reference(
    path,
    object_name,
    *,
    objects_column="observation.objects",
    frame_column="frame_index",
    time_column="timestamp",
    quaternion_order="wxyz",
):
    table = pq.read_table(
        path, columns=[objects_column, frame_column, time_column]
    ).to_pydict()
    poses = []
    visible = []
    for row in table[objects_column]:
        items = [x for x in row if x["name"] == object_name]
        if len(items) != 1:
            raise ValueError("Reference object must occur exactly once per frame")
        item = items[0]
        valid = bool(item["visible"])
        visible.append(valid)
        transform = np.full((4, 4), np.nan)
        if valid:
            p = np.asarray(item["pose"], float)
            q = p[3:]
            if quaternion_order == "wxyz":
                q = q[[1, 2, 3, 0]]
            elif quaternion_order != "xyzw":
                raise ValueError("Explicit wxyz or xyzw quaternion order required")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(q).as_matrix()
            transform[:3, 3] = p[:3]
        poses.append(transform)
    return dict(
        poses=np.asarray(poses),
        visible=np.asarray(visible),
        frame_ids=np.asarray(table[frame_column]),
        timestamps_s=np.asarray(table[time_column]),
    )
