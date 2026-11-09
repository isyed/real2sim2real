"""Recover a compact observed support surface, never a whole rectangular hull.

Plane/foreground connectivity and visible side depth are required. Unknown legs
are not invented. Callers can supply explicit segmented parts for full furniture.
"""
import numpy as np


def planar_region_geometry(depth,K,human,obj,rectangles,floor,*,polygons=(),
                           max_span_m=1.5,minimum_height_m=.2):
    from PIL import Image,ImageDraw
    from scipy.ndimage import binary_dilation,binary_erosion,distance_transform_edt
    from scipy.spatial import ConvexHull
    from .segmented_support import component_hull,planar_component_hull
    depth=np.asarray(depth,float);K=np.asarray(K,float);floor=np.asarray(floor,float)
    if (depth.ndim!=2 or K.shape!=(3,3) or floor.shape!=(4,)
            or not np.isfinite(K).all() or not np.isfinite(floor).all()
            or not np.isclose(np.linalg.norm(floor[:3]),1,atol=1e-5)):
        raise ValueError('Finite camera intrinsics and unit floor plane required')
    mask=Image.new('1',(depth.shape[1],depth.shape[0]));draw=ImageDraw.Draw(mask)
    for x0,y0,x1,y1 in rectangles:
        if not 0<=x0<x1<=depth.shape[1] or not 0<=y0<y1<=depth.shape[0]:raise ValueError('Invalid support rectangle')
        draw.rectangle((x0,y0,x1-1,y1-1),fill=1)
    for poly in polygons:
        a=np.asarray(poly)
        if a.ndim!=2 or a.shape[1]!=2 or len(a)<3 or not np.isfinite(a).all() or (a<0).any() or (a[:,0]>=depth.shape[1]).any() or (a[:,1]>=depth.shape[0]).any():raise ValueError('Invalid support polygon')
        draw.polygon([tuple(p) for p in a.tolist()],fill=1)
    keep=np.asarray(mask,bool)&~binary_dilation(np.asarray(human,bool)|np.asarray(obj,bool),iterations=4)
    keep &= np.isfinite(depth)&(depth>0)&(depth<65.535)
    v,u=np.nonzero(keep);points=np.c_[u,v,np.ones(len(u))]@np.linalg.inv(K).T*depth[v,u,None]
    h=points@floor[:3]+floor[3];valid=(h>minimum_height_m)&(h<1.8)
    u,v,points=u[valid],v[valid],points[valid]
    if len(points)<300:raise ValueError('No sufficient above-floor support evidence')
    ids=np.linspace(0,len(points)-1,min(6000,len(points)),dtype=int);sample=points[ids]
    distance=distance_transform_edt(~np.asarray(obj,bool)) if np.asarray(obj,bool).any() else np.zeros(depth.shape)
    weights=np.maximum(.01,np.exp(-distance[v[ids],u[ids]]/40.))
    rng=np.random.default_rng(42);best=None;up=floor[:3]
    for _ in range(300):
        a,b,c=sample[rng.choice(len(sample),3,replace=False)];normal=np.cross(b-a,c-a);size=np.linalg.norm(normal)
        if size<1e-10:continue
        normal/=size
        if normal@up<0:normal=-normal
        if normal@up<np.cos(np.deg2rad(15)):continue
        offset=-normal@a;inside=np.abs(sample@normal+offset)<.01
        score=float(weights[inside].sum())
        if best is None or score>best[0]:best=(score,int(inside.sum()),normal,offset)
    if best is None or best[1]<max(100,len(sample)*.15):raise ValueError('Support plane requires explicit segmentation')
    _,_,normal,offset=best
    plane_mask=np.zeros(depth.shape,bool);near=np.abs(points@normal+offset)<.01;plane_mask[v[near],u[near]]=True
    core=binary_erosion(plane_mask,iterations=1)
    if not core.any():raise ValueError('No coherent support-plane interior')
    cy,cx=np.nonzero(core);order=np.argsort(distance[cy,cx],kind='stable');attempts=[];selected=None
    for index in order:
        seed=(int(cx[index]),int(cy[index]))
        if any(np.linalg.norm(np.asarray(seed)-s)<20 for s in attempts):continue
        attempts.append(np.asarray(seed))
        try:
            selected=component_hull(depth,K,plane_mask,seeds_xy=[seed],floor_plane=floor,
                height_range_m=(minimum_height_m,1.8),max_neighbor_distance_m=.04,
                min_points=200,min_component_fraction=.03,max_axis_span_m=max_span_m,
                allow_planar_component=True)
            break
        except ValueError:
            if len(attempts)>=12:break
    if selected is None:raise ValueError('No compact connected support surface; explicit parts required')
    top,_,component=selected;center=top.mean(0);_,_,axes=np.linalg.svd(top-center,full_matrices=False);normal=axes[-1]
    if normal@up<0:normal=-normal
    offset=-normal@center;uv=(top-center)@axes[:2].T;footprint=ConvexHull(uv)
    pixels=np.asarray(component['retained_pixels_xy']);component_mask=np.zeros(depth.shape,bool);component_mask[pixels[:,1],pixels[:,0]]=True
    adjacent=binary_dilation(component_mask,iterations=8)
    signed=points@normal+offset;uv_all=(points-center)@axes[:2].T
    in_outline=np.max(uv_all@footprint.equations[:,:2].T+footprint.equations[:,2],axis=1)<=.025
    side=(signed<-.015)&(signed>-.10)&in_outline&adjacent[v,u]
    if int(side.sum())<20:raise ValueError('Support side thickness is unobserved; explicit part review required')
    vertices,faces,refinement=planar_component_hull(np.r_[top,points[side]],expected_up=up,
        residual_tolerance_m=.01,maximum_tilt_deg=15,minimum_inlier_fraction=.5,
        thickness_range_m=(.005,.1))
    heights=vertices@up+floor[3]
    if np.max(np.ptp(vertices,axis=0))>max_span_m or np.ptp(heights)>.25:
        raise ValueError('Recovered support still exceeds surface bounds')
    return vertices,faces,dict(method='connected planar foreground plus measured side thickness',
        collision_approximation='observed horizontal surface only; explicit parts required for legs/base',
        vertical_structure_recovered=False,visible_pixels=int(keep.sum()),
        hull_volume_m3=refinement['hull_volume_m3'],component=component,plane=refinement,
        auto_recovery_requires_physics_review=True)


def validate_components(meshes,*,maximum_component_span_m=1.5,maximum_component_volume_m3=.25):
    """Reject grossly oversized filled supports before any physical rollout."""
    from scipy.spatial import ConvexHull
    if not np.isfinite([maximum_component_span_m,maximum_component_volume_m3]).all() or min(maximum_component_span_m,maximum_component_volume_m3)<=0:raise ValueError('Invalid support quality bounds')
    rows=[]
    for vertices,faces in meshes:
        p=np.asarray(vertices,float);f=np.asarray(faces)
        if p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all() or f.ndim!=2 or f.shape[1]!=3 or not len(f) or not np.issubdtype(f.dtype,np.integer) or f.min()<0 or f.max()>=len(p):raise ValueError('Invalid support mesh')
        span=np.ptp(p,axis=0);volume=float(ConvexHull(p).volume)
        if np.max(span)>maximum_component_span_m or volume>maximum_component_volume_m3:
            raise ValueError(f'Support part is oversized: spans={span.tolist()} m, volume={volume:.5g} m3; segmentation repair required')
        rows.append(dict(extents_m=span.tolist(),volume_m3=volume))
    if not rows:raise ValueError('Declared support contains no valid collision components')
    return dict(passed=True,components=rows,max_component_span_m=maximum_component_span_m,max_component_volume_m3=maximum_component_volume_m3)
