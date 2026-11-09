"""Build separate collision hulls from explicitly segmented predicted-depth parts.

Masks/seeds belong to the caller. Neighbour connectivity rejects disconnected
background; caller-provided metric bounds reject contaminated components. The
method never adds thickness to a degenerate observed surface.
"""
import numpy as np


def component_hull(depth, intrinsics, mask, *, seeds_xy, exclusion_mask=None,
                   floor_plane=None, height_range_m=None,
                   depth_range_m=None,
                   max_neighbor_distance_m=.04, min_points=100,
                   min_component_fraction=.25, max_axis_span_m=1.,
                   max_height_extent_m=None, maximum_points=6000,
                   allow_planar_component=False):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import ConvexHull
    depth=np.asarray(depth,float);K=np.asarray(intrinsics,float);mask=np.asarray(mask)
    if (depth.ndim!=2 or mask.shape!=depth.shape or mask.dtype!=np.bool_
            or K.shape!=(3,3) or not np.isfinite(K).all() or abs(np.linalg.det(K))<1e-12):
        raise ValueError('Require matching depth/boolean mask and invertible intrinsics')
    if (not np.isfinite([max_neighbor_distance_m,max_axis_span_m]).all() or max_neighbor_distance_m<=0
            or min_points<4 or maximum_points<min_points or max_axis_span_m<=0
            or not 0<min_component_fraction<=1):
        raise ValueError('Invalid component bounds')
    keep=mask & np.isfinite(depth) & (depth>0)
    if depth_range_m is not None:
        bounds=np.asarray(depth_range_m,float)
        if bounds.shape!=(2,) or not np.isfinite(bounds).all() or not 0<bounds[0]<bounds[1]:
            raise ValueError('Invalid camera-depth bounds')
        keep &= (depth>=bounds[0]) & (depth<=bounds[1])
    if exclusion_mask is not None:
        exclusion=np.asarray(exclusion_mask)
        if exclusion.shape!=depth.shape or exclusion.dtype!=np.bool_:raise ValueError('Invalid exclusion mask')
        keep &= ~exclusion
    normal=None
    if floor_plane is not None:
        floor=np.asarray(floor_plane,float)
        if floor.shape!=(4,) or not np.isfinite(floor).all() or not np.isclose(np.linalg.norm(floor[:3]),1.,atol=1e-5):
            raise ValueError('Require a unit, finite floor plane')
        normal=floor[:3]
    if height_range_m is not None:
        bounds=np.asarray(height_range_m,float)
        if normal is None or bounds.shape!=(2,) or not np.isfinite(bounds).all() or not 0<=bounds[0]<bounds[1]:
            raise ValueError('Finite increasing height bounds require a floor plane')
    if max_height_extent_m is not None and (not np.isfinite(max_height_extent_m) or max_height_extent_m<=0):
        raise ValueError('Invalid height-extent limit')
    v,u=np.nonzero(keep)
    points=np.c_[u,v,np.ones(len(u))]@np.linalg.inv(K).T*depth[v,u,None]
    if height_range_m is not None:
        heights=points@normal+floor[3]
        valid=(heights>=height_range_m[0])&(heights<=height_range_m[1])
        u,v,points=u[valid],v[valid],points[valid]
    if len(points)<min_points:raise ValueError('Too few valid part points')
    index=np.full(depth.shape,-1,np.int32);index[v,u]=np.arange(len(u))
    sources=[];targets=[]
    for dy,dx in [(0,1),(1,0)]:
        y,x=v+dy,u+dx;valid=(y<depth.shape[0])&(x<depth.shape[1])
        a=np.flatnonzero(valid);b=index[y[valid],x[valid]];valid=b>=0;a,b=a[valid],b[valid]
        near=np.linalg.norm(points[a]-points[b],axis=1)<=max_neighbor_distance_m
        sources.append(a[near]);targets.append(b[near])
    a,b=np.concatenate(sources),np.concatenate(targets)
    graph=coo_matrix((np.ones(len(a)),(a,b)),shape=(len(points),len(points)))
    _,labels=connected_components(graph,directed=False)
    selected=[]
    for seed in seeds_xy:
        if len(seed)!=2 or any(int(x)!=x for x in seed):raise ValueError('Seeds must be integer pixel coordinates')
        x,y=map(int,seed)
        if not (0<=x<depth.shape[1] and 0<=y<depth.shape[0]) or index[y,x]<0:
            raise ValueError('Seed is outside the valid selected part')
        selected.append(int(labels[index[y,x]]))
    if not selected or len(set(selected))!=1:raise ValueError('Seeds must identify one connected depth part')
    chosen=labels==selected[0];fraction=float(np.mean(chosen));count=int(chosen.sum())
    if count<min_points or fraction<min_component_fraction:raise ValueError('Selected depth component is too small')
    points=points[chosen];pixels=np.c_[u[chosen],v[chosen]]
    span=np.ptp(points,axis=0)
    if np.max(span)>max_axis_span_m:raise ValueError('Part exceeds its metric span bound')
    height_extent=float(np.ptp(points@normal)) if normal is not None else None
    if max_height_extent_m is not None and (height_extent is None or height_extent>max_height_extent_m):
        raise ValueError('Part exceeds its vertical extent bound')
    if len(points)>maximum_points:
        ids=np.linspace(0,len(points)-1,maximum_points,dtype=int);points=points[ids];pixels=pixels[ids]
    singular=np.linalg.svd(points-points.mean(0),compute_uv=False)
    if singular[-1]<1e-5:
        if not allow_planar_component:raise ValueError('Observed part has no supported volume; do not invent thickness')
        faces=np.empty((0,3),int);volume=0.
    else:
        hull=ConvexHull(points);faces=hull.simplices.copy();volume=float(hull.volume)
        if not np.isfinite(volume) or volume<=1e-9:raise ValueError('Invalid part volume')
        for i,face in enumerate(faces):
            x,y,z=points[face]
            if np.dot(np.cross(y-x,z-x),hull.equations[i,:3])<0:faces[i]=face[::-1]
    return points,faces,dict(selected_points=count,valid_mask_points=len(labels),
        component_fraction=fraction,axis_extents_m=span.tolist(),height_extent_m=height_extent,
        hull_volume_m3=volume,seeds_xy=[list(s) for s in seeds_xy],
        closed_collision_hull=bool(len(faces)),
        max_neighbor_distance_m=max_neighbor_distance_m,retained_pixels_xy=pixels.tolist(),
        depth_range_m=list(depth_range_m) if depth_range_m is not None else None,
        method='one depth-connected segmented part; observed-volume convex enclosure')


def planar_component_hull(points, *, expected_up, residual_tolerance_m=.008,
                          maximum_tilt_deg=10., minimum_inlier_fraction=.7,
                          side_quantile=.9, minimum_side_points=20,
                          thickness_range_m=(.005,.08), seed=42):
    """Denoise a planar top, deriving slab thickness from visible lower-side points.

    No object/task target enters the fit. A plane without sufficient observed
    side depth is rejected rather than padded with an invented thickness.
    """
    from scipy.spatial import ConvexHull
    p=np.asarray(points,float);up=np.asarray(expected_up,float)
    if (p.ndim!=2 or p.shape[1]!=3 or len(p)<64 or not np.isfinite(p).all()
            or up.shape!=(3,) or not np.isfinite(up).all() or np.linalg.norm(up)<1e-8
            or not 0<minimum_inlier_fraction<=1 or not 0<side_quantile<1
            or not 0<maximum_tilt_deg<90 or residual_tolerance_m<=0
            or minimum_side_points<3 or not 0<thickness_range_m[0]<thickness_range_m[1]):
        raise ValueError('Invalid planar-part inputs or bounds')
    up=up/np.linalg.norm(up);minimum_cos=np.cos(np.deg2rad(maximum_tilt_deg));rng=np.random.default_rng(seed)
    best=None
    for _ in range(300):
        a,b,c=p[rng.choice(len(p),3,replace=False)];normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
        if length<1e-10:continue
        normal/=length
        if normal@up<0:normal=-normal
        if normal@up<minimum_cos:continue
        offset=-normal@a;inside=np.abs(p@normal+offset)<=residual_tolerance_m
        if best is None or inside.sum()>best.sum():best=inside
    if best is None or best.mean()<minimum_inlier_fraction:raise ValueError('No dominant admissible support plane')
    inside=best
    for _ in range(3):
        q=p[inside];center=q.mean(0);_,_,axes=np.linalg.svd(q-center,full_matrices=False);normal=axes[-1]
        if normal@up<0:normal=-normal
        if normal@up<minimum_cos:raise ValueError('Refined support plane exceeds tilt bound')
        offset=-normal@center;inside=np.abs(p@normal+offset)<=residual_tolerance_m
        if inside.mean()<minimum_inlier_fraction:raise ValueError('Refined support plane lacks sufficient evidence')
    signed=p@normal+offset;side=-signed[signed < -1.5*residual_tolerance_m]
    if len(side)<minimum_side_points:raise ValueError('No measured lower-side evidence; do not invent thickness')
    thickness=float(np.quantile(side,side_quantile))
    if not thickness_range_m[0]<=thickness<=thickness_range_m[1]:raise ValueError('Observed thickness is outside declared bounds')
    # Preserve the observed footprint; replace depth spikes with the fitted top.
    basis=axes[:2];xy=(p-center)@basis.T;outline=ConvexHull(xy);uv=xy[outline.vertices]
    top=uv@basis+center;vertices=np.r_[top,top-thickness*normal]
    hull=ConvexHull(vertices);faces=hull.simplices.copy()
    for i,face in enumerate(faces):
        a,b,c=vertices[face]
        if np.dot(np.cross(b-a,c-a),hull.equations[i,:3])<0:faces[i]=face[::-1]
    return vertices,faces,dict(method='robust planar top; thickness from observed lower-side depth',
        plane_normal=normal.tolist(),plane_offset_m=float(offset),inlier_fraction=float(inside.mean()),
        plane_residual_tolerance_m=residual_tolerance_m,observed_side_points=len(side),
        side_thickness_quantile=side_quantile,measured_thickness_m=thickness,
        tilt_from_up_deg=float(np.rad2deg(np.arccos(np.clip(normal@up,-1,1)))),hull_volume_m3=float(hull.volume))
