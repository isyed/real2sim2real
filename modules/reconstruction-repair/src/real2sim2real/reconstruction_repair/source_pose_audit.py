"""Source-only CPU checks for a fixed mesh/pose sequence and its exported copy.

These helpers do not run a tracker, alter a candidate or consume task ground
truth. Depth agreement is consistency with a prediction, not metric truth.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .surface_depth import front_surface_depth,compare_surface_depth


def corresponding_mesh_transform(source,target):
    """Fit a fixed similarity using explicit corresponding vertex indices."""
    a,b=np.asarray(source,float),np.asarray(target,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or len(a)<3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Matching finite mesh vertex correspondence required')
    ca,cb=a.mean(0),b.mean(0);x,y=a-ca,b-cb
    u,_,vt=np.linalg.svd(x.T@y);r=vt.T@np.diag([1,1,np.linalg.det(vt.T@u.T)])@u.T
    denominator=float(np.sum(x*x))
    if denominator<=0:raise ValueError('Degenerate mesh correspondence')
    scale=float(np.sum(y*(x@r.T))/denominator);shift=cb-scale*(ca@r.T)
    residual=np.linalg.norm(a@(scale*r.T)+shift-b,axis=1)
    return dict(scale=scale,rotation=r,translation=shift,
        median_residual_m=float(np.median(residual)),max_residual_m=float(residual.max()),
        scope='Corresponding-vertex fit only; does not establish source-image correctness')


def bind_pose_frames(source_rotation,source_translation,export_rotation,export_translation,mesh_transform):
    """Check all poses under one fixed world transform and mesh-local transform."""
    rc,tc,rw,tw=map(lambda x:np.asarray(x,float),(source_rotation,source_translation,export_rotation,export_translation))
    n=len(tc)
    if rc.shape!=(n,3,3) or rw.shape!=rc.shape or tc.shape!=(n,3) or tw.shape!=tc.shape or not n:
        raise ValueError('Matching complete source and export pose sequences required')
    if any(not np.isfinite(x).all() for x in (rc,tc,rw,tw)):raise ValueError('Nonfinite poses')
    if abs(mesh_transform['scale']-1)>1e-5:raise ValueError('World-frame proof requires unit-scale mesh conversion')
    s=np.asarray(mesh_transform['rotation']);d=np.asarray(mesh_transform['translation'])
    world=rw[0]@s@rc[0].T;offset=rw[0]@d+tw[0]-world@tc[0]
    position=np.linalg.norm(np.einsum('tij,j->ti',rw,d)+tw-(tc@world.T+offset),axis=1)
    actual=np.einsum('tij,jk->tik',rw,s);expected=np.einsum('ij,tjk->tik',world,rc)
    angle=Rotation.from_matrix(actual@expected.transpose(0,2,1)).magnitude()*180/np.pi
    return dict(camera_to_export_rotation=world,camera_to_export_translation=offset,
        position_residual_m=position,rotation_residual_deg=angle,
        max_position_residual_m=float(position.max()),max_rotation_residual_deg=float(angle.max()))


def projection_audit(vertices_camera,faces,intrinsics,object_mask,predicted_depth,*,human_mask=None):
    """Rasterize exact triangles in a native-grid ROI containing mask and mesh.

    Cropping changes only the principal point; there is no resize, mesh
    decimation or depth substitution. Both raw and human-excluded overlap are
    reported because a human segmentation mask is not a true visibility oracle.
    """
    v=np.asarray(vertices_camera,float);k=np.asarray(intrinsics,float);mask=np.asarray(object_mask,bool)
    depth=np.asarray(predicted_depth,float)
    if mask.ndim!=2 or depth.shape!=mask.shape:raise ValueError('Matching native mask/depth grids required')
    human=np.zeros_like(mask) if human_mask is None else np.asarray(human_mask,bool)
    if human.shape!=mask.shape:raise ValueError('Matching human-mask grid required')
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or np.any(v[:,2]<=1e-5):
        raise ValueError('Finite camera-space mesh wholly in front of camera required')
    xy=v@k.T;uv=xy[:,:2]/xy[:,2:];h,w=mask.shape
    yy,xx=np.nonzero(mask)
    lo=np.floor(uv.min(0))-2;hi=np.ceil(uv.max(0))+3
    if len(xx):lo=np.minimum(lo,[xx.min()-2,yy.min()-2]);hi=np.maximum(hi,[xx.max()+3,yy.max()+3])
    lo=np.clip(lo,[0,0],[w-1,h-1]).astype(int);hi=np.clip(hi,[1,1],[w,h]).astype(int)
    x0,y0=map(int,lo);x1,y1=map(int,hi)
    if x1<=x0 or y1<=y0:raise ValueError('Projected mesh/mask outside native image')
    kk=k.copy();kk[0,2]-=x0;kk[1,2]-=y0
    surface=front_surface_depth(v,faces,kk,(y1-y0,x1-x0))
    rendered=np.isfinite(surface);observed=mask[y0:y1,x0:x1];occ=human[y0:y1,x0:x1]
    def overlap(a,b):
        intersection=int(np.count_nonzero(a&b));union=int(np.count_nonzero(a|b))
        return dict(iou=intersection/union if union else None,
            precision=intersection/int(a.sum()) if a.any() else None,
            recall=intersection/int(b.sum()) if b.any() else None,
            intersection_pixels=intersection,rendered_pixels=int(a.sum()),observed_pixels=int(b.sum()))
    result=dict(roi_xyxy=[x0,y0,x1,y1],native_shape=list(mask.shape),full_silhouette=overlap(rendered,observed),
        human_excluded_silhouette=overlap(rendered&~occ,observed&~occ),
        predicted_surface_depth=compare_surface_depth(surface,depth[y0:y1,x0:x1],observed,
            occluder_mask=occ,erosion_pixels=1,min_points=100,max_error_m=.10),
        object_mask_pixels=int(mask.sum()),object_human_overlap_pixels=int(np.count_nonzero(mask&human)),
        depth_is_prediction=True,candidate_approved=False,source_semantics_verified=False)
    return result,surface
