"""Rigid frame changes for paired object meshes, trajectories and supports."""
import numpy as np


def rigid(value):
    value=np.asarray(value,dtype=float)
    if (value.shape!=(4,4) or not np.isfinite(value).all() or
            not np.allclose(value[3],[0,0,0,1],atol=1e-7) or
            not np.allclose(value[:3,:3].T@value[:3,:3],np.eye(3),atol=1e-5) or
            not np.isclose(np.linalg.det(value[:3,:3]),1.,atol=1e-5)):
        raise ValueError('Expected a rigid metre-frame transform, without scale')
    return value


def reframe_object(poses,vertices,world_from_source,source_body_from_target):
    """Return A @ P @ B and inv(B)-rebaked vertices, preserving world surfaces."""
    poses=np.asarray(poses,dtype=float);vertices=np.asarray(vertices,dtype=float)
    if poses.ndim!=3 or poses.shape[1:]!=(4,4) or not len(poses):
        raise ValueError('Expected a nonempty [T,4,4] pose sequence')
    for pose in poses:
        rigid(pose)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
        raise ValueError('Expected finite metre mesh vertices')
    a,b=rigid(world_from_source),rigid(source_body_from_target)
    inverse=np.linalg.inv(b)
    return a@poses@b, vertices@inverse[:3,:3].T+inverse[:3,3]
