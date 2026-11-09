"""Short diagnostic recovery between independently verified endpoint poses.

The caller supplies registration evidence; interpolation initializes a hypothesis,
never an observed trajectory. Existing image/depth/support gates remain active.
"""
from dataclasses import asdict
import numpy as np
from .anchor_hypothesis import initialize
from .temporal_pose import TemporalPoseConfig, measurements, acceptance, refine_window, validate


def visible_occluder(object_masks, human_masks, frame_ids, polygon_xy, active_ids):
    """Combine an explicitly RGB-reviewed static occluder with human masks.

    Only declared frames are affected. Observed object pixels can never be hidden
    by an annotation. Polygon coordinates are native image pixels, not 3D poses.
    """
    from PIL import Image, ImageDraw
    obj, human = np.asarray(object_masks, bool), np.asarray(human_masks, bool)
    ids, active, polygon = np.asarray(frame_ids), np.asarray(active_ids), np.asarray(polygon_xy, float)
    if (obj.ndim != 3 or human.shape != obj.shape or ids.shape != (len(obj),)
            or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids)
            or active.ndim != 1 or (len(active) and (active.dtype.kind not in 'iu' or not np.isin(active, ids).all()))
            or polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not np.isfinite(polygon).all()):
        raise ValueError('Native masks, unique source IDs and explicit polygon/active IDs required')
    h,w=obj.shape[1:]
    if np.any(polygon < 0) or np.any(polygon >= [w,h]):
        raise ValueError('Occluder polygon outside the native image')
    image=Image.new('1',(w,h));ImageDraw.Draw(image).polygon([tuple(p) for p in polygon],fill=1)
    extra=np.asarray(image,bool)[None]&np.isin(ids,active)[:,None,None]&~obj
    return human | extra


def refine(data, anchor_ids, anchor_poses, *, config=TemporalPoseConfig(benefit_mode='observation'),
           unobserved_image_ids=(), unobserved_depth_ids=()):
    """Initialize and assess an interval while preserving its registered anchors."""
    times=np.asarray(data['timestamps_s']);delta=np.diff(times)
    if not len(delta) or delta[0]<=0 or not np.allclose(delta,delta[0]):
        raise ValueError('Explicit uniform source timestamps required')
    original, receipt=initialize(data['frame_ids'],anchor_ids,anchor_poses,fps=1./delta[0])
    window=dict(data,poses_camera=original,locked=np.isin(data['frame_ids'],anchor_ids))
    config.validate(); validate(window)
    # Do not perturb an interval which already passes independent gates.
    before=measurements(window,original,config)
    missing=dict(unobserved_image_ids=unobserved_image_ids,unobserved_depth_ids=unobserved_depth_ids)
    decision=acceptance(window,before,before,original,original,config,**missing)
    needed=[r for r in decision['reasons'] if r not in ('no_temporal_benefit','no_observation_benefit')]
    if not needed:
        candidate=original.copy();decision=dict(decision,accepted=True,reasons=[])
        report=dict(before=before,after=before,decision=decision,action='retain_valid_anchor_hypothesis',config=asdict(config))
    else:
        candidate,report=refine_window(window,config,**missing);report['action']='refine_registered_endpoint_interval'
    report.update(initialization=receipt,diagnostic_only=True,full_reference_accepted=False,
                  physics_validated=False,metric_size_certified=False,clean_trajectory_used=False,
                  unobserved_image_frame_ids=list(unobserved_image_ids),
                  unobserved_depth_frame_ids=list(unobserved_depth_ids),
                  inferred_frames_verified_by_observation=False)
    return candidate,report
