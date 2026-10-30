"""Opt-in qualification of contradictory depth inside a rigid object's mask.

Explicitly reviewed temporal anchors are required. Outliers become missing (0),
never invented surface values. This is predicted-data qualification, not a metric
measurement or a reference approval. No mesh, pose or fit/held-out labels enter.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class DepthEvidenceConfig:
    relative_tolerance: float = .20
    absolute_tolerance_m: float = .25
    max_anchor_relative_change: float = .20
    min_foreground_pixels: int = 128
    min_foreground_fraction: float = .10
    min_outlier_fraction: float = .25

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError('Depth qualification limits must be positive finite numbers')
        if type(self.min_foreground_pixels) is not int:
            raise ValueError('Pixel count must be an integer')
        for key in ['relative_tolerance','max_anchor_relative_change','min_foreground_fraction','min_outlier_fraction']:
            if getattr(self,key) >= 1:
                raise ValueError('Fraction limits must be less than one')


def qualify_depth(depth_m, object_mask, occluder_mask, *, frame_id, anchor_before,
                  anchor_after, anchor_review, config=DepthEvidenceConfig()):
    """Return a copy and a receipt. Anchors are (frame_id, positive depth_m).

    Anchor depths must be obtained independently from reviewed visible source
    depth/masks. A caller owns their hashes and selection protocol. No extrapolation
    or adjustment is allowed without a substantial consistent foreground component.
    """
    config.validate()
    z=np.asarray(depth_m);obj=np.asarray(object_mask);occ=np.asarray(occluder_mask)
    if (z.ndim!=2 or z.dtype.kind!='f' or not np.isfinite(z).all() or np.any(z<0)
            or obj.shape!=z.shape or occ.shape!=z.shape or obj.dtype!=bool or occ.dtype!=bool):
        raise ValueError('Expected finite nonnegative metre depth and matching boolean masks')
    if not isinstance(anchor_review,str) or not anchor_review.strip():
        raise ValueError('Explicit reviewed anchor evidence is required')
    if type(frame_id) is not int or frame_id<0:
        raise ValueError('Frame ID must be nonnegative integer')
    a,b=anchor_before,anchor_after
    if (len(a)!=2 or len(b)!=2 or type(a[0]) is not int or type(b[0]) is not int
            or not a[0]<frame_id<b[0] or not np.isfinite([a[1],b[1]]).all() or min(a[1],b[1])<=0):
        raise ValueError('Ordered bracketing positive depth anchors are required')
    if abs(b[1]/a[1]-1)>config.max_anchor_relative_change:
        raise ValueError('Temporal depth anchors are inconsistent')
    t=(frame_id-a[0])/(b[0]-a[0]);reference=float((1-t)*a[1]+t*b[1])
    tolerance=max(config.absolute_tolerance_m,reference*config.relative_tolerance)
    visible=obj&~occ&(z>0)
    near=visible&(np.abs(z-reference)<=tolerance)
    outlier=visible&~near
    count=int(visible.sum());retained=int(near.sum());bad=int(outlier.sum())
    receipt=dict(frame_id=frame_id,reference_depth_m=reference,tolerance_m=tolerance,
        visible_pixels=count,consistent_pixels=retained,outlier_pixels=bad,
        anchor_before=list(a),anchor_after=list(b),anchor_review=anchor_review,
        config=asdict(config),changed=False,status='unchanged',
        missing_depth_value=0.,surface_depth_invented=False,approved_for_use=False)
    result=z.copy()
    if retained<config.min_foreground_pixels or retained/max(count,1)<config.min_foreground_fraction:
        receipt['status']='insufficient_consistent_foreground'
    elif bad/max(count,1)<config.min_outlier_fraction:
        receipt['status']='no_substantial_depth_conflict'
    else:
        result[outlier]=0.
        receipt.update(changed=True,status='outliers_marked_missing',invalidated_pixels=bad,
            retained_median_depth_m=float(np.median(z[near])))
    return result,receipt
