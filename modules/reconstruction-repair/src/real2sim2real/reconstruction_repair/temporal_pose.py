"""Bounded multi-frame object-pose hypotheses from masks, depth and support.

The optimizer uses a contour proxy; final acceptance uses full triangle masks
and dense visible depth. Passing one window is not full-reference, interaction,
physics or submission validation. No human parameters are fitted.
"""
from dataclasses import asdict,dataclass
import time

import numpy as np
from scipy.ndimage import binary_erosion,distance_transform_edt,map_coordinates,median_filter
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation,Slerp

from .geometry import mesh_silhouette


@dataclass(frozen=True)
class TemporalPoseConfig:
    knot_stride: int = 15
    max_function_calls: int = 1200
    max_seconds: float = 180.
    translation_bound_m: float = .15
    rotation_bound_deg: float = 20.
    maximum_frame_change_m: float = .20
    contour_samples: int = 32
    mesh_samples: int = 2048
    depth_samples: int = 64
    pixel_sigma: float = 2.
    depth_sigma_m: float = .03
    acceleration_sigma_m_s2: float = 3.
    angular_acceleration_sigma_deg_s2: float = 120.
    min_iou: float = .60
    min_precision: float = .60
    min_recall: float = .60
    max_frame_iou_drop: float = .02
    max_depth_error_m: float = .10
    min_depth_points: int = 100
    max_support_penetration_m: float = .02
    max_step_m: float = .08
    max_rotation_step_deg: float = 30.
    minimum_motion_improvement: float = .20

    def validate(self):
        for key,value in asdict(self).items():
            if isinstance(value,bool) or not isinstance(value,(int,float,np.number)) or not np.isfinite(value) or value<=0:
                raise ValueError('Invalid positive bound: '+key)
        for key in ('knot_stride','max_function_calls','contour_samples','mesh_samples','depth_samples','min_depth_points'):
            if int(getattr(self,key))!=getattr(self,key):raise ValueError('Integer required: '+key)
        if any(not 0<getattr(self,key)<1 for key in ('min_iou','min_precision','min_recall','minimum_motion_improvement')):
            raise ValueError('Invalid acceptance fraction')


FIELDS={'vertices','faces','poses_camera','intrinsics','timestamps_s','frame_ids','object_mask','occluder_mask',
    'depth_m','fit','locked','support_plane_camera','support_center_camera','support_radius_m'}


def validate(data):
    if set(data)!=FIELDS:raise ValueError('Temporal window input fields differ')
    n=len(data['frame_ids']);v=data['vertices'];f=data['faces'];p=data['poses_camera'];m=data['object_mask']
    if (n<5 or v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or f.ndim!=2 or f.shape[1]!=3
        or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=len(v) or p.shape!=(n,4,4)
        or m.ndim!=3 or m.shape[0]!=n or data['occluder_mask'].shape!=m.shape or data['depth_m'].shape!=m.shape
        or data['intrinsics'].shape!=(n,3,3) or data['timestamps_s'].shape!=(n,)
        or data['support_plane_camera'].shape!=(n,4) or data['support_center_camera'].shape!=(n,3)
        or data['support_radius_m'].shape!=(n,) or np.any(data['support_radius_m']<=0)):
        raise ValueError('Invalid window geometry, camera, depth or support shapes')
    if any(not np.isfinite(x).all() for x in data.values()):raise ValueError('Nonfinite window input')
    if (data['frame_ids'].dtype.kind not in 'iu' or np.any(np.diff(data['frame_ids'])!=1)
        or np.any(np.diff(data['timestamps_s'])<=0) or not np.allclose(p[:,3],[0,0,0,1])
        or not np.allclose(p[:,:3,:3]@p[:,:3,:3].swapaxes(1,2),np.eye(3),atol=1e-4)
        or not np.allclose(np.linalg.det(p[:,:3,:3]),1,atol=1e-4)
        or not np.allclose(np.linalg.norm(data['support_plane_camera'][:,:3],axis=1),1,atol=1e-5)):
        raise ValueError('Invalid timeline, poses or metric support normals')
    for name in ('fit','locked'):
        if data[name].shape!=(n,) or not np.isin(data[name],[0,1]).all():raise ValueError('Invalid frame flag: '+name)
    if data['fit'].sum()<3 or (~data['fit'].astype(bool)).sum()<2 or not data['locked'].any():
        raise ValueError('Need predeclared fit/held-out frames and a trusted locked anchor')


def sample_mesh(vertices,count):
    ids=np.unique(np.r_[np.linspace(0,len(vertices)-1,min(count,len(vertices))).astype(int),vertices.argmin(0),vertices.argmax(0)])
    return vertices[ids]


def projected(points,pose,K):
    xyz=points@pose[:3,:3].T+pose[:3,3];uv=xyz@K.T
    return xyz,uv[:,:2]/np.maximum(uv[:,2:],1e-6)


def front_vertices(vertices,faces,pose):
    """Front-facing surface samples; do not fit rear vertices to foreground depth."""
    tri=vertices[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])@pose[:3,:3].T
    center=tri.mean(1)@pose[:3,:3].T+pose[:3,3]
    front=np.einsum('ij,ij->i',normal,center)<-1e-12
    if not front.any():raise ValueError('No front-facing surface; check mesh winding and camera pose')
    return vertices[np.unique(faces[front])]


def polygon(points,count):
    try:points=points[ConvexHull(points).vertices]
    except QhullError:return np.repeat(points.mean(0)[None],count,axis=0),None
    closed=np.r_[points,points[:1]];length=np.linalg.norm(np.diff(closed,axis=0),axis=1)
    distance=np.r_[0,np.cumsum(length)]
    q=np.linspace(0,max(distance[-1],1e-6),count,endpoint=False)
    contour=np.stack([np.interp(q,distance,closed[:,i]) for i in range(2)],axis=1)
    return contour,ConvexHull(points).equations


def motion(poses,times):
    dt=np.diff(times);velocity=np.diff(poses[:,:3,3],axis=0)/dt[:,None]
    acceleration=np.diff(velocity,axis=0)/((dt[1:]+dt[:-1])*.5)[:,None]
    steps=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)
    angle=np.rad2deg((Rotation.from_matrix(poses[1:,:3,:3])*Rotation.from_matrix(poses[:-1,:3,:3]).inv()).magnitude())
    return dict(step_p95_m=float(np.quantile(steps,.95)),step_max_m=float(steps.max()),
        acceleration_p95_m_s2=float(np.quantile(np.linalg.norm(acceleration,axis=1),.95)),
        rotation_step_max_deg=float(angle.max()))


def measurements(data,poses,config):
    n=len(poses);v=data['vertices'][np.unique(data['faces'])];K=data['intrinsics'];rows=[]
    g=dict(vertices=data['vertices'],faces=data['faces'],object_scale=np.array(1.),object_rotation=poses[:,:3,:3],
        object_translation=poses[:,:3,3],intrinsics=K,world_to_camera=np.tile(np.eye(4),(n,1,1)))
    for i,p in enumerate(poses):
        obs=data['object_mask'][i].astype(bool)&~data['occluder_mask'][i].astype(bool)
        mask=mesh_silhouette(g,i,*obs.shape)&~data['occluder_mask'][i].astype(bool)
        inter=int((mask&obs).sum());union=int((mask|obs).sum())
        xyz,uv=projected(front_vertices(data['vertices'],data['faces'],p),p,K[i]);xy=np.rint(uv).astype(int);h,w=obs.shape
        keep=(xyz[:,2]>.01)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        flat=xy[keep,1]*w+xy[keep,0];unique,inverse=np.unique(flat,return_inverse=True)
        z=np.full(len(unique),np.inf);np.minimum.at(z,inverse,xyz[keep,2])
        measured=data['depth_m'][i].ravel()[unique];use=obs.ravel()[unique]&(measured>0)&(measured<65.535)
        all_camera=v@p[:3,:3].T+p[:3,3]
        plane=data['support_plane_camera'][i];delta=all_camera-data['support_center_camera'][i]
        radius=np.linalg.norm(delta-np.outer(delta@plane[:3],plane[:3]),axis=1)
        over=radius<=data['support_radius_m'][i]
        penetration=max(0.,-float(np.min(all_camera[over]@plane[:3]+plane[3]))) if over.any() else 0.
        rows.append(dict(frame=int(data['frame_ids'][i]),fit=bool(data['fit'][i]),iou=inter/max(1,union),
            precision=inter/max(1,int(mask.sum())),recall=inter/max(1,int(obs.sum())),depth_points=int(use.sum()),
            depth_error_m=float(np.median(np.abs(measured[use]-z[use]))) if use.any() else None,
            support_penetration_m=penetration,behind_camera=bool(np.any(all_camera[:,2]<=.01))))
    return rows


def acceptance(data,before,after,original,candidate,config):
    reasons=[]
    for a,b in zip(before,after):
        if b['iou']<config.min_iou or b['precision']<config.min_precision or b['recall']<config.min_recall:reasons.append('image_gate')
        if a['iou']>=config.min_iou and b['iou']<a['iou']-config.max_frame_iou_drop:reasons.append('previous_good_image_regressed')
        if b['depth_points']<config.min_depth_points or b['depth_error_m'] is None or b['depth_error_m']>config.max_depth_error_m:reasons.append('depth_gate')
        if b['support_penetration_m']>config.max_support_penetration_m:reasons.append('support_gate')
        if b['behind_camera']:reasons.append('camera_gate')
    a=motion(original,data['timestamps_s']);b=motion(candidate,data['timestamps_s'])
    if b['step_max_m']>config.max_step_m or b['rotation_step_max_deg']>config.max_rotation_step_deg:reasons.append('motion_step_gate')
    benefit=(a['step_p95_m']>1e-4 and b['step_p95_m']<=a['step_p95_m']*(1-config.minimum_motion_improvement)
        or a['acceleration_p95_m_s2']>1e-3 and b['acceleration_p95_m_s2']<=a['acceleration_p95_m_s2']*(1-config.minimum_motion_improvement))
    if not benefit:reasons.append('no_temporal_benefit')
    if np.linalg.norm(candidate[:,:3,3]-original[:,:3,3],axis=1).max()>config.maximum_frame_change_m:reasons.append('frame_displacement_bound')
    rotation=np.rad2deg((Rotation.from_matrix(candidate[:,:3,:3])*Rotation.from_matrix(original[:,:3,:3]).inv()).magnitude())
    if rotation.max()>config.rotation_bound_deg+1e-5:reasons.append('frame_rotation_bound')
    if not np.array_equal(candidate[data['locked'].astype(bool)],original[data['locked'].astype(bool)]):reasons.append('locked_anchor_changed')
    return dict(accepted=not reasons,reasons=sorted(set(reasons)),motion_before=a,motion_after=b,
        heldout_mean_iou_before=float(np.mean([r['iou'] for r in before if not r['fit']])),
        heldout_mean_iou_after=float(np.mean([r['iou'] for r in after if not r['fit']])))


def refine_window(data,config=TemporalPoseConfig()):
    config.validate();validate(data);started=time.monotonic();original=data['poses_camera'];n=len(original);times=data['timestamps_s']
    locked=data['locked'].astype(bool);knots=np.unique(np.r_[np.arange(0,n,config.knot_stride),n-1,np.flatnonzero(locked)])
    editable=np.flatnonzero(~locked[knots]);base_t=median_filter(original[:,:3,3],size=(5,1),mode='nearest')[knots]
    base_t[locked[knots]]=original[knots[locked[knots]],:3,3];base_r=Rotation.from_matrix(original[knots,:3,:3])
    points=sample_mesh(data['vertices'][np.unique(data['faces'])],config.mesh_samples);prepared=[]
    for i in np.flatnonzero(data['fit']):
        mask=data['object_mask'][i].astype(bool);occ=data['occluder_mask'][i].astype(bool);visible=mask&~occ
        edge=mask&~binary_erosion(mask)&~occ;y,x=np.where(edge)
        if len(x)<3 or visible.sum()<config.min_depth_points:raise ValueError('Insufficient fit-frame silhouette evidence')
        # Match the integer-cell convention of the final Pillow triangle union.
        # Sampling contours at cell centers avoids a systematic half-pixel shrink.
        contour,_=polygon(np.c_[x,y]+.5,config.contour_samples)
        depth_surface=front_vertices(data['vertices'],data['faces'],original[i])
        xyz,uv=projected(depth_surface,original[i],data['intrinsics'][i]);pixels=np.rint(uv).astype(int);h,w=mask.shape
        keep=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
        ids=np.flatnonzero(keep);ids=ids[visible[pixels[ids,1],pixels[ids,0]]]
        # Front-most sample for each initial projected pixel, never the back surface.
        if len(ids):
            flat=pixels[ids,1]*w+pixels[ids,0];order=np.lexsort((xyz[ids,2],flat));ids=ids[order];flat=flat[order]
            ids=ids[np.r_[True,np.diff(flat)!=0]]
        if not len(ids):raise ValueError('No visible depth correspondences in fit frame')
        selected=depth_surface[ids[np.linspace(0,len(ids)-1,config.depth_samples).astype(int)]]
        prepared.append((i,distance_transform_edt(~mask),occ,contour,selected))
    def decode(x):
        change=np.zeros((len(knots),6));change[editable]=x.reshape(-1,6)
        t=base_t+change[:,:3];r=Rotation.from_rotvec(change[:,3:])*base_r
        poses=np.tile(np.eye(4),(n,1,1));poses[:,:3,3]=np.stack([np.interp(times,times[knots],t[:,j]) for j in range(3)],axis=1)
        poses[:,:3,:3]=Slerp(times[knots],r)(times).as_matrix();poses[locked]=original[locked]
        return poses
    calls=0;best_x=np.zeros(len(editable)*6);best_cost=float('inf');budget_hit=False
    class Budget(Exception):pass
    def residual(x):
        nonlocal calls,best_x,best_cost
        if calls>=config.max_function_calls or time.monotonic()-started>config.max_seconds:raise Budget()
        calls+=1;poses=decode(x);result=[]
        for i,dt,occ,target,depth_points in prepared:
            _,uv=projected(points,poses[i],data['intrinsics'][i]);outline,equations=polygon(uv,config.contour_samples)
            coords=np.array([outline[:,1]-.5,outline[:,0]-.5])
            outside=map_coordinates(dt,coords,order=1,mode='constant',cval=100.)
            hidden=map_coordinates(occ.astype(float),coords,order=1,mode='constant',cval=0.)
            result.extend([outside*(1-hidden)/config.pixel_sigma,
                np.maximum(0,np.max(target@equations[:,:2].T+equations[:,2],axis=1))/config.pixel_sigma if equations is not None else np.full(config.contour_samples,100.)])
            xyz,pixels=projected(depth_points,poses[i],data['intrinsics'][i]);coords=np.array([pixels[:,1],pixels[:,0]])
            depth=map_coordinates(data['depth_m'][i],coords,order=1,mode='constant',cval=0.)
            visible=map_coordinates((data['object_mask'][i]&~occ).astype(float),coords,order=1,mode='constant',cval=0.)
            valid=(depth>0)&(depth<65.535)
            result.append((xyz[:,2]-depth)*visible*valid/config.depth_sigma_m)
            camera=points@poses[i,:3,:3].T+poses[i,:3,3];plane=data['support_plane_camera'][i]
            delta=camera-data['support_center_camera'][i];radial=np.linalg.norm(delta-np.outer(delta@plane[:3],plane[:3]),axis=1)
            over=radial<=data['support_radius_m'][i]
            penetration=max(0.,-float((camera[over]@plane[:3]+plane[3]).min())-config.max_support_penetration_m) if over.any() else 0.
            result.append(np.array([penetration/.01]))
        dt=np.diff(times);velocity=np.diff(poses[:,:3,3],axis=0)/dt[:,None]
        result.append((np.diff(velocity,axis=0)/((dt[1:]+dt[:-1])*.5)[:,None]/config.acceleration_sigma_m_s2).ravel())
        omega=(Rotation.from_matrix(poses[1:,:3,:3])*Rotation.from_matrix(poses[:-1,:3,:3]).inv()).as_rotvec()/dt[:,None]
        result.append((np.diff(omega,axis=0)/((dt[1:]+dt[:-1])*.5)[:,None]/np.deg2rad(config.angular_acceleration_sigma_deg_s2)).ravel())
        result.append(x.reshape(-1,6)[:,:3].ravel()/.10)
        out=np.concatenate(result);cost=float(np.sum(2*(np.sqrt(1+out*out)-1)))
        if cost<best_cost:best_cost=cost;best_x=x.copy()
        return out
    if len(editable):
        bounds=np.tile([config.translation_bound_m]*3+[np.deg2rad(config.rotation_bound_deg)]*3,len(editable))
        try:least_squares(residual,best_x,bounds=(-bounds,bounds),loss='soft_l1',diff_step=1e-3,max_nfev=config.max_function_calls)
        except Budget:budget_hit=True
    candidate=decode(best_x);before=measurements(data,original,config);after=measurements(data,candidate,config)
    decision=acceptance(data,before,after,original,candidate,config)
    report=dict(schema='r2s2r_engines.temporal-object-window-result/1',config=asdict(config),function_calls=calls,budget_hit=budget_hit,
        seconds=time.monotonic()-started,decision=decision,before=before,after=after,human_changed=False,
        full_trajectory_accepted=False,interaction_validated=False,physics_validated=False,approved_for_use=False,
        limitation='Contour proxy used only for fitting; full-triangle masks and dense depth for window checks. Depth/support are predictions, not truth. Whole-clip and interaction/physics checks remain required.')
    return candidate,report


def run_window(source,provenance_path,output,config=TemporalPoseConfig(), *,
               metric_bundle=None, metric_calibration=None, diagnostic_only=False):
    import json
    from pathlib import Path
    from .bundle import arrays_read,arrays_atomic,file_sha256,json_atomic
    source,output=Path(source),Path(output)
    if output.exists():raise ValueError('Preserve completed trials; use a new output path')
    provenance=json.loads(Path(provenance_path).read_text())
    if (provenance.get('schema')!='r2s2r_engines.temporal-object-window/1' or not provenance.get('source')
        or provenance.get('clean_trajectory_used') is not False or provenance.get('input_sha256')!=file_sha256(source)):
        raise ValueError('Require hashed prediction-only window provenance')
    data=arrays_read(source)
    if diagnostic_only:
        if metric_bundle is not None or metric_calibration is not None:
            raise ValueError('Diagnostic bypass cannot claim metric-size validation')
        metric=dict(status='explicit_diagnostic_without_metric_validation',passed=False)
    else:
        from .metric_preflight import window_metric_preflight
        metric=window_metric_preflight(data,metric_bundle,metric_calibration)
    candidate,report=refine_window(data,config)
    report.update(metric_size_preflight=metric,diagnostic_only=bool(diagnostic_only),metric_size_verified=bool(metric['passed']))
    report.update(provenance=provenance,source_sha256={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')})
    output.mkdir(parents=True)
    from .gate_report import write_explanation
    report['gate_explanation'] = write_explanation(report, output)
    arrays_atomic(output/'candidate.npz',dict(poses_camera=candidate))
    json_atomic(output/'report.json',report)
    return report
