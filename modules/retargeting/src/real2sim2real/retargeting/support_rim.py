"""Estimate slab thickness from independently selected visible rim depth.

Top-surface masks commonly exclude a table's vertical rim. Their depth noise
must not be mistaken for thickness. This helper requires explicit side samples
and reports a hypothesis; the caller still validates held-out observations.
"""
import numpy as np


def measured_thickness(side_points, plane, *, quantile=.9, min_points=20,
                       range_m=(.005,.08), minimum_below_fraction=.8):
    points,p=np.asarray(side_points,float),np.asarray(plane,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or len(points)<min_points
            or p.shape!=(4,) or not np.isfinite(points).all() or not np.isfinite(p).all()
            or not np.isclose(np.linalg.norm(p[:3]),1.,atol=1e-6)
            or not np.isfinite([quantile,*range_m,minimum_below_fraction]).all()
            or not 0<quantile<1 or not 0<range_m[0]<range_m[1]
            or not 0<minimum_below_fraction<=1 or type(min_points) is not int or min_points<3):
        raise ValueError('Finite side samples, unit top plane and bounded thickness settings required')
    depth=-(points@p[:3]+p[3]);fraction=float(np.mean(depth>0))
    if fraction<minimum_below_fraction:raise ValueError('Selected side observations are not below the top plane')
    below=depth[depth>0]
    if len(below)<min_points:raise ValueError('Insufficient observed lower-side samples')
    thickness=float(np.quantile(below,quantile))
    if not range_m[0]<=thickness<=range_m[1]:raise ValueError('Measured rim thickness exceeds the declared limits')
    return dict(thickness_m=thickness,quantile=quantile,observed_side_points=len(points),
        below_top_fraction=fraction,p05_m=float(np.percentile(below,5)),p95_m=float(np.percentile(below,95)),
        source='Explicit visible side-depth selection; not top-surface noise',
        physical_accuracy_established=False)
