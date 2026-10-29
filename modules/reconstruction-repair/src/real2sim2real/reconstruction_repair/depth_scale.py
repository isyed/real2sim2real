"""Bounded depth-scale stabilization from reviewed static floor samples.

The metric anchor is an existing predicted plane, never independently measured
ground truth. This helper does not select pixels, load videos or alter a bundle.
Consumers must establish a static-camera/floor assumption and retain provenance.
"""
import numpy as np


def metric_pointmap(depth_m, intrinsics):
    """Native-grid OpenCV XYZ metres; consumers explicitly decode depth first.

    Challenge H5 millimetres and common-library inverse-depth PNGs use different
    encodings. Keeping this boundary numeric prevents accidental cross-decoding.
    """
    z=np.asarray(depth_m,dtype=np.float32);k=np.asarray(intrinsics,float)
    if (z.ndim!=2 or k.shape!=(3,3) or not np.isfinite(k).all()
            or min(k[0,0],k[1,1])<=0 or not np.allclose(k[2],[0,0,1])):
        raise ValueError('Native metric depth grid and pixel camera are required')
    valid=np.isfinite(z)&(z>0)&(z<65.535)
    if valid.sum()<100:raise ValueError('Insufficient metric depth or incompatible units')
    v,u=np.indices(z.shape,dtype=np.float32)
    p=np.stack(((u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z),axis=-1)
    p[~valid]=np.nan
    return p


def static_floor_gain(points, regions, plane, *, first_gain=1.):
    """Fit one multiplicative depth gain; reject motion/inconsistent floor ROIs."""
    p=np.asarray(points,float);r=np.asarray(regions);plane=np.asarray(plane,float)
    if (p.ndim!=2 or p.shape[1]!=3 or len(p)<300 or r.shape!=(len(p),)
            or not np.isfinite(p).all() or plane.shape!=(4,) or not np.isfinite(plane).all()
            or not np.isclose(np.linalg.norm(plane[:3]),1.,atol=1e-5)
            or not np.isfinite(first_gain) or first_gain<=0):
        raise ValueError('Finite metric floor samples, unit plane and positive anchor gain required')
    ids=np.unique(r)
    if len(ids)<2 or any(np.count_nonzero(r==k)<100 for k in ids):
        raise ValueError('Two independently selected floor regions with sufficient samples are required')
    denominator=p@plane[:3]
    if np.any(abs(denominator)<1e-6):raise ValueError('Invalid floor ray/plane intersection')
    gains=-plane[3]/denominator
    gain=float(np.median(gains))
    regional=[float(np.median(gains[r==k])) for k in ids]
    residual=np.abs(denominator*gain+plane[3])
    if (not .9<=gain/first_gain<=1.1 or np.any(gains<=0)
            or max(abs(x/gain-1) for x in regional)>.01 or np.percentile(residual,95)>.02):
        raise ValueError('Static-floor gain, region agreement or residual failed')
    return gain,dict(floor_p95_m=float(np.percentile(residual,95)),region_gains=regional,
        points=len(p),anchor_is_predicted=True,absolute_metric_accuracy_established=False,
        thresholds=dict(gain_ratio_min=.9,gain_ratio_max=1.1,max_region_disagreement=.01,max_p95_m=.02))
