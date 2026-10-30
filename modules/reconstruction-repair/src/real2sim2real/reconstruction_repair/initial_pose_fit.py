"""Bounded native-grid CPU refinement of initial pose hypotheses.

Frame zero has no previous accepted pose. Search radii describe initial-pose
uncertainty, never a relaxed inter-frame motion gate. Results require independent
renderer validation and do not approve a reference or certify metric scale.
"""
from dataclasses import dataclass,asdict
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from real2sim2real.reconstruction.recovery_quality import Limits,rigid,consistency,sample_vertices,rejection_reasons
from .geometry import mesh_silhouette


@dataclass(frozen=True)
class InitialFitConfig:
    max_seconds:float=180.
    max_evaluations:int=900
    max_seeds:int=3
    translation_radius_m:float=.25
    rotation_radius_deg:float=60.

    def validate(self):
        for value in [self.max_seconds,self.translation_radius_m,self.rotation_radius_deg]:
            if isinstance(value,bool) or not np.isfinite(value) or value<=0:raise ValueError('Positive finite initial-search bounds required')
        for value in [self.max_evaluations,self.max_seeds]:
            if type(value) is not int or value<1:raise ValueError('Positive integer work bounds required')
        if self.rotation_radius_deg>180:raise ValueError('Initial rotation radius cannot exceed 180 degrees')


def fit(vertices,faces,seeds,K,object_mask,human_mask,depth_m,support,*,config=InitialFitConfig()):
    config.validate();start=time.monotonic();limits=Limits()
    v,f,seeds,K,z=map(np.asarray,(vertices,faces,seeds,K,depth_m));obj,human=np.asarray(object_mask),np.asarray(human_mask)
    if (v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all()
        or f.ndim!=2 or f.shape[1]!=3 or not len(f) or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=len(v)
        or seeds.ndim!=3 or seeds.shape[1:]!=(4,4) or not len(seeds) or not all(rigid(p) for p in seeds)
        or K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0 or not np.allclose(K[2],[0,0,1])
        or obj.ndim!=2 or obj.dtype!=bool or human.shape!=obj.shape or human.dtype!=bool
        or z.shape!=obj.shape or z.dtype.kind!='f' or not np.isfinite(z).all() or np.any(z<0)):
        raise ValueError('Native metric mesh, rigid hypotheses, camera and boolean mask/depth grids required')
    if support.get('ground_truth_used') is not False or support.get('coordinate_frame')!='camera' or not support.get('evidence'):
        raise ValueError('Predicted camera-frame support provenance required')
    observed=obj&~human;H,W=obj.shape
    if observed.sum()<limits.depth_points:raise ValueError('Insufficient observed support')
    points=sample_vertices(v);center=(v.min(0)+v.max(0))/2;reports=[];best_valid=None;best_cost=None;attempts=0;timed_out=False
    radius=config.translation_radius_m;angle=np.deg2rad(config.rotation_radius_deg)
    class Stop(Exception):pass
    def assess(pose,seed_index):
        nonlocal attempts,best_valid,best_cost,timed_out
        if attempts>=config.max_evaluations:raise Stop()
        if time.monotonic()-start>config.max_seconds:timed_out=True;raise Stop()
        attempts+=1
        g=dict(vertices=v,faces=f,object_scale=np.array(1.),object_rotation=pose[None,:3,:3],object_translation=pose[None,:3,3],intrinsics=K[None],world_to_camera=np.eye(4)[None])
        mask=mesh_silhouette(g,0,H,W)&~human;inter=int((mask&observed).sum())
        m=dict(iou=inter/max(1,int((mask|observed).sum())),precision=inter/max(1,int(mask.sum())),recall=inter/max(1,int(observed.sum())))
        geo=consistency(points,pose,K,z,observed,support)
        # Initial validity has no invented previous-pose or temporal constraint.
        rejected=rejection_reasons(pose,pose,m,geo,recovery=True,limits=limits)
        d=geo['depth_error_m'];cost=1-m['iou']+.1*min(5.,d/limits.depth_error_m if d is not None else 5.)
        cost+=3*max(0.,(d or 0)/limits.depth_error_m-1)+20*max(0.,geo['support_penetration_m']-limits.support_penetration_m)
        if geo['depth_points']<limits.depth_points or geo['behind_camera']:cost+=10
        row=dict(seed_index=seed_index,pose_camera=pose.tolist(),metrics=m,geometry=geo,rejected_for=rejected,cost=cost)
        reports.append(row)
        if best_cost is None or cost<reports[best_cost]['cost']:best_cost=len(reports)-1
        if not rejected and (best_valid is None or m['iou']>reports[best_valid]['metrics']['iou']):best_valid=len(reports)-1
        return cost
    try:
        per_seed=max(1,config.max_evaluations//min(config.max_seeds,len(seeds)))
        for i,seed in enumerate(seeds[:config.max_seeds]):
            origin=seed[:3,:3]@center+seed[:3,3];begin=attempts
            def pose(x):
                out=seed.astype(float).copy();out[:3,:3]=Rotation.from_rotvec(x[3:]).as_matrix()@seed[:3,:3]
                out[:3,3]=origin+x[:3]-out[:3,:3]@center;return out
            def objective(x):
                if attempts-begin>=per_seed:raise Stop()
                shift,turn=np.linalg.norm(x[:3]),np.linalg.norm(x[3:])
                if shift>radius or turn>angle:return 20+shift/radius+turn/angle
                return assess(pose(x),i)
            x=np.zeros(6);objective(x)
            try:
                if per_seed<=1:continue
                minimize(objective,x,method='Powell',bounds=[(-radius,radius)]*3+[(-angle,angle)]*3,
                         options=dict(maxfev=per_seed-1,xtol=1e-4,ftol=1e-4))
            except Stop:
                if time.monotonic()-start>config.max_seconds or attempts>=config.max_evaluations:raise
    except Stop:pass
    if not reports:raise ValueError('No initial candidate assessed within bounds')
    selected=best_valid if best_valid is not None else best_cost
    return dict(schema='real2sim2real.initial-pose-fit/1',config=asdict(config),evaluations=attempts,runtime_s=time.monotonic()-start,
        selected=reports[selected],proposal_valid_on_cpu=best_valid is not None,all_candidates=reports,
        optimizer_deadline_reached=timed_out,frame=0,temporal_reference=None,search_bounds_are_initial_uncertainty=True,
        native_image_grid=True,mesh_changed=False,depth_changed=False,thresholds_relaxed=False,
        requires_independent_gpu_validation=True,metric_scale_certified=False,full_reference_accepted=False)
