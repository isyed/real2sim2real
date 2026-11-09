"""Continuous bounded CPU recovery proposals after a discrete pose search.

This is a single-frame diagnostic/registration hypothesis, not fine reference
repair or calibration approval. Optimization may use a smaller full image grid;
acceptance recomputes the native triangle silhouette and all tracking gates.
"""
from dataclasses import asdict, dataclass
import time
import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from real2sim2real.reconstruction.recovery_quality import (
    Limits, consistency, rejection_reasons, rigid, sample_vertices,
)
from .geometry import mesh_silhouette


@dataclass(frozen=True)
class LocalRegistrationConfig:
    max_seconds: float = 45.
    max_evaluations: int = 180
    optimization_image_max_size: int = 384
    depth_violation_weight: float = 0.
    preserve_rotation: bool = False

    def validate(self):
        if type(self.preserve_rotation) is not bool:
            raise ValueError('preserve_rotation must be an explicit boolean')
        if not np.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError('Positive finite runtime bound required')
        if (isinstance(self.depth_violation_weight,bool) or not np.isfinite(self.depth_violation_weight)
                or not 0 <= self.depth_violation_weight <= 5):
            raise ValueError('Depth objective weight must be finite and within [0,5]')
        for value in (self.max_evaluations, self.optimization_image_max_size):
            if isinstance(value,bool) or not np.isfinite(value) or value < 1 or int(value) != value:
                raise ValueError('Positive integer evaluation/image bounds required')


def propose(vertices, faces, previous, initial, K, object_mask, human_mask,
            depth_m, support, *, config=LocalRegistrationConfig()):
    """Fit one frame within existing 8 cm / 30 degree recovery constraints.

    ``previous`` is the last measured accepted pose, never an arbitrary clean
    target. The caller must verify source/seed provenance. No missing frame is
    invented and no temporal continuation is assembled here.
    """
    return _fit(vertices, faces, previous, initial, K, object_mask, human_mask,
                depth_m, support, config=config, initial_registration=False)


def propose_initial(vertices, faces, initial, K, object_mask, human_mask,
                    depth_m, support, *, config=LocalRegistrationConfig()):
    """Refine an unaccepted fresh-registration hypothesis for ONE visible frame.

    There is no previous accepted state or temporal claim. Search stays within
    8 cm / 30 degrees of this hypothesis. Native image/depth/support gates are
    retained; every returned proposal still needs independent GPU validation.
    This cannot authorize carrying a pose through an occlusion or splicing it.
    """
    return _fit(vertices, faces, initial, initial, K, object_mask, human_mask,
                depth_m, support, config=config, initial_registration=True)


def propose_translation_first(vertices, faces, previous, initial, K, object_mask,
                              human_mask, depth_m, support, *, config=LocalRegistrationConfig()):
    """Opt-in staged recovery: hold the proposed rotation, then try the full fit.

    The first fit searches around an unaccepted hypothesis, not around a trusted
    state. Its final pose MUST pass motion checks against the real previous
    accepted pose. Neither an initial-registration flag nor a better silhouette
    authorizes a temporal jump. Each stage has the explicitly supplied finite
    time/evaluation budget; the wrapper uses at most two stages.
    """
    from dataclasses import replace
    previous,initial=np.asarray(previous),np.asarray(initial)
    if not rigid(previous) or not rigid(initial):raise ValueError('Rigid temporal reference and hypothesis required')
    if config.preserve_rotation:raise ValueError('Translation-first wrapper requires an explicit full-fit fallback')
    first=propose_initial(vertices,faces,initial,K,object_mask,human_mask,depth_m,support,
                          config=replace(config,preserve_rotation=True))
    candidate=np.asarray(first['pose_camera'])
    reasons=rejection_reasons(candidate,previous,first['metrics'],first['geometry'],recovery=True)
    if not reasons:
        return dict(first,schema='real2sim2real.translation-first-recovery/1',
            method='tracked_rotation_translation_refit',proposal_valid_for_tracking=True,
            rejected_for=[],previous_accepted_pose_used=True,temporal_continuity_validated=True,
            seed_rotation_preserved_exactly=bool(np.array_equal(candidate[:3,:3],initial[:3,:3])),
            rotation_preserved_exactly=bool(np.array_equal(candidate[:3,:3],previous[:3,:3])),
            requires_gpu_revalidation=True,full_reference_accepted=False)
    second=propose(vertices,faces,previous,initial,K,object_mask,human_mask,depth_m,support,config=config)
    second.update(translation_first_attempt=dict(metrics=first['metrics'],geometry=first['geometry'],
        rejected_for=reasons,pose_camera=first['pose_camera']),translation_first_fallback=True)
    return second


def _fit(vertices, faces, previous, initial, K, object_mask, human_mask,
         depth_m, support, *, config, initial_registration):
    config.validate();limits=Limits();start=time.monotonic()
    v,f,p,seed,K,z=map(np.asarray,(vertices,faces,previous,initial,K,depth_m))
    obj,human=np.asarray(object_mask,bool),np.asarray(human_mask,bool)
    if (v.ndim!=2 or v.shape[1]!=3 or not len(v) or f.ndim!=2 or f.shape[1]!=3 or not len(f)
        or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=len(v) or not np.isfinite(v).all()
        or not rigid(p) or not rigid(seed) or K.shape!=(3,3) or not np.isfinite(K).all()
        or min(K[0,0],K[1,1])<=0 or not np.allclose(K[2],[0,0,1])
        or obj.ndim!=2 or human.shape!=obj.shape or z.shape!=obj.shape):
        raise ValueError('Finite metric geometry, rigid poses and native camera/mask/depth grids required')
    if support.get('ground_truth_used') is not False or not support.get('evidence'):
        raise ValueError('Explicit source-derived support provenance required')
    observed=obj&~human
    if config.preserve_rotation and not np.array_equal(seed[:3,:3],p[:3,:3]):
        raise ValueError('A rotation-preserving fit requires the same seed/reference rotation')
    if observed.sum()<limits.depth_points:raise ValueError('Insufficient visible object pixels')
    points=sample_vertices(v);height,width=obj.shape
    ratio=min(1.,config.optimization_image_max_size/max(height,width))
    small=(max(2,round(width*ratio)),max(2,round(height*ratio)))
    Ks=K.astype(float).copy();Ks[0]*=small[0]/width;Ks[1]*=small[1]/height
    target=np.asarray(Image.fromarray(obj).resize(small,Image.Resampling.NEAREST)).copy()
    occluder=np.asarray(Image.fromarray(human).resize(small,Image.Resampling.NEAREST))
    target &= ~occluder
    coordinate=max(1.,float(np.abs(p[:3,3]).max()))
    translation_bound=max(0.,limits.step_translation_m-8*np.finfo(np.float32).eps*coordinate)
    rotation_bound=np.deg2rad(limits.step_rotation_deg-.001)

    def pose(x):
        value=p.astype(float).copy();value[:3,3]+=x[:3]
        if not config.preserve_rotation:
            value[:3,:3]=Rotation.from_rotvec(x[3:]).as_matrix()@p[:3,:3]
        return value

    def mask(value,k,h,w):
        g=dict(vertices=v,faces=f,object_scale=np.array(1.),object_rotation=value[None,:3,:3],
               object_translation=value[None,:3,3],intrinsics=k[None],world_to_camera=np.eye(4)[None])
        return mesh_silhouette(g,0,h,w)

    best=[None,float('inf')];evaluations=[0];timed_out=[False]
    class Deadline(Exception):pass
    def objective(x):
        if time.monotonic()-start>config.max_seconds:
            timed_out[0]=True;raise Deadline()
        evaluations[0]+=1
        move,angle=np.linalg.norm(x[:3]),np.linalg.norm(x[3:])
        if move>translation_bound or angle>rotation_bound:
            return 10.+max(0.,move/translation_bound-1)+max(0.,angle/rotation_bound-1)
        value=pose(x);render=mask(value,Ks,small[1],small[0])&~occluder
        inter=int((render&target).sum());iou=inter/max(1,int((render|target).sum()))
        geo=consistency(points,value,K,z,observed,support)
        depth=geo['depth_error_m']
        cost=1-iou+.05*min(5.,depth/limits.depth_error_m if depth is not None else 5.)
        depth_weight = .5 if initial_registration else config.depth_violation_weight
        if depth_weight:
            # A new registration has no reliable temporal prior. Avoid improving
            # its silhouette while retaining a depth-invalid seed.
            cost+=depth_weight*max(0.,depth/limits.depth_error_m-1.) if depth is not None else 5.
        if geo['behind_camera']:cost+=10
        cost+=10*max(0.,geo['support_penetration_m']-limits.support_penetration_m)
        if geo['depth_points']<limits.depth_points:cost+=5
        if cost<best[1]:best[:]=[value,cost]
        return cost

    x=np.r_[seed[:3,3]-p[:3,3],Rotation.from_matrix(seed[:3,:3]@p[:3,:3].T).as_rotvec()]
    if config.preserve_rotation:x=x[:3]
    if np.linalg.norm(x[:3])>translation_bound or np.linalg.norm(x[3:])>rotation_bound:
        raise ValueError('Initial hypothesis must already lie inside the declared motion bounds')
    try:
        objective(x)
        if config.max_evaluations>1:
            minimize(objective,x,method='Powell',bounds=[(-translation_bound,translation_bound)]*3+
                     ([] if config.preserve_rotation else [(-rotation_bound,rotation_bound)]*3),
                     options=dict(maxfev=config.max_evaluations-1,xtol=1e-4,ftol=1e-4))
    except Deadline:pass
    if best[0] is None:raise ValueError('No bounded local hypothesis measured before deadline')
    candidate=best[0];render=mask(candidate,K,height,width)&~human
    inter=int((render&observed).sum())
    metrics=dict(iou=inter/max(1,int((render|observed).sum())),
                 precision=inter/max(1,int(render.sum())),recall=inter/max(1,int(observed.sum())))
    geometry=consistency(points,candidate,K,z,observed,support)
    rejected=rejection_reasons(candidate,candidate if initial_registration else p,metrics,geometry,recovery=True,limits=limits)
    result=dict(schema='real2sim2real.local-registration/1',pose_camera=candidate.tolist(),
        metrics=metrics,geometry=geometry,rejected_for=rejected,proposal_valid_for_tracking=not rejected,
        limits=asdict(limits),config=asdict(config),evaluations=evaluations[0],runtime_s=time.monotonic()-start,
        optimizer_deadline_reached=timed_out[0],native_acceptance=True,thresholds_relaxed=False,
        rotation_preserved_exactly=bool(np.array_equal(candidate[:3,:3],p[:3,:3])),
        full_reference_accepted=False,metric_calibration_approved=False,physics_validated=False)
    if initial_registration:
        result.update(schema='real2sim2real.initial-registration-refinement/1',
            proposal_valid_for_tracking=False, initial_pose_gates_passed=not rejected,
            previous_accepted_pose_used=False, temporal_continuity_validated=False,
            requires_gpu_revalidation=True,
            initial_pose_camera=seed.tolist(), search_translation_m=float(translation_bound),
            search_rotation_deg=float(np.rad2deg(rotation_bound)))
    return result
