"""Restore a calibrated object size without changing its pinhole projection.

A monocular fitter may change object scale and camera translation together.
Overwriting only its returned scale changes the rendered object footprint.
This utility couples the two quantities. It does not resolve camera calibration,
absolute-depth ambiguity, contact, or temporal validity; those need separate
checks before a proposed pose replaces its source.
"""
import numpy as np


def restore_fixed_scale_projection(rotation, translation, fitted_scale, target_scale):
    """Return ``s * (vertices @ R.T) + t`` in the same camera frame.

    Translation units follow the caller's camera frame (normally metres).
    Scales are positive uniform mesh-to-camera multipliers. The target must
    come from an independently specified metric-size contract. Multiplying both
    fitted scale and translation by the same positive ratio preserves every
    pinhole projection. Anisotropic scale is rejected rather than averaged.
    """
    R=np.asarray(rotation,dtype=float);t=np.asarray(translation,dtype=float)
    if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError('Finite 3x3 rotation and camera translation of shape (3,) required')
    if not np.allclose(R.T@R,np.eye(3),atol=1e-5,rtol=0) or not np.isclose(np.linalg.det(R),1.,atol=1e-5,rtol=0):
        raise ValueError('Proper rigid rotation required')
    def uniform(value):
        a=np.asarray(value,dtype=float)
        if a.shape not in ((),(1,),(3,)) or not np.isfinite(a).all() or np.any(a<=0):
            raise ValueError('Positive finite scalar or uniform three-component scale required')
        values=a.reshape(-1)
        if not np.allclose(values,values[0],rtol=1e-5,atol=0):
            raise ValueError('Anisotropic scale requires a different geometric model')
        return float(values[0])
    fitted,target=uniform(fitted_scale),uniform(target_scale)
    ratio=target/fitted
    corrected=t*ratio
    if not np.isfinite(corrected).all() or not np.isfinite(ratio):
        raise ValueError('Scale ratio produces nonfinite translation')
    return dict(rotation=R.copy(),translation=corrected,scale=target,scale_ratio=ratio,
                translation_delta=corrected-t,projection_preserved=True,
                metric_truth_validated=False,candidate_approved=False)
