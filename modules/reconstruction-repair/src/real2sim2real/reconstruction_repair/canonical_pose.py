"""Invert fixed mesh normalization for row-vector pose samplers.

The sampler transforms ``(vertices @ axes - center) * normalization`` by a
uniform scale, row rotation and translation, then changes camera axes. Matrices
enter/leave this module in conventional column-vector camera coordinates.
"""
import numpy as np


def _rotation(value):
    R=np.asarray(value,float)
    if (R.shape!=(3,3) or not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-4)
        or not np.isclose(np.linalg.det(R),1.,atol=1e-4)):raise ValueError('Proper rigid rotation required')
    return R


def encode(pose,axes,normalization,center,camera_axes):
    p=np.asarray(pose,float);Q=_rotation(axes);D=_rotation(camera_axes);c=np.asarray(center,float)
    if (p.shape!=(4,4) or not np.isfinite(p).all() or not np.allclose(p[3],[0,0,0,1])
        or not np.isfinite(normalization) or normalization<=0 or c.shape!=(3,) or not np.isfinite(c).all()):
        raise ValueError('Rigid metric pose and fixed canonicalization required')
    R=_rotation(p[:3,:3]);raw=Q.T@R.T@D.T
    translation=p[:3,3]@D.T+c@raw
    return dict(rotation=raw,translation=translation,scale=np.repeat(1./normalization,3))


def decode(rotation,translation,scale,axes,normalization,center,camera_axes):
    R=_rotation(rotation);Q=_rotation(axes);D=_rotation(camera_axes);t=np.asarray(translation,float).reshape(3)
    s=np.asarray(scale,float).reshape(3);c=np.asarray(center,float)
    if (not np.isfinite(t).all() or not np.isfinite(s).all() or np.any(s<=0)
        or not np.allclose(s,s[0],rtol=1e-5) or not np.isclose(normalization*s[0],1.,rtol=1e-5)
        or c.shape!=(3,) or not np.isfinite(c).all()):raise ValueError('The fixed metric mesh scale must be preserved')
    out=np.eye(4);out[:3,:3]=(Q@R@D).T;out[:3,3]=t@D-c@R@D
    return out
