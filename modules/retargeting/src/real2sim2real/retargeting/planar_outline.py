"""Complete a reviewed planar support outline using independently fitted depth.

Object poses and task trajectories are not inputs. A visible image contour can
recover a tabletop footprint missed by depth masking, while its plane/thickness
must come from observed depth. This is explicitly a planar-surface hypothesis.
"""
import numpy as np


def inspect_plane(points, plane, *, tolerance_m=.008, min_inlier_fraction=.7):
    """Check a fixed plane on held-out depth without refitting or rescaling it.

    RGB agreement and depth agreement are separate gates: an unchanged visible
    contour alone cannot validate the depth of a static support surface.
    """
    points,plane=np.asarray(points,float),np.asarray(plane,float)
    if (points.ndim!=2 or points.shape[1]!=3 or len(points)<20
            or not np.isfinite(points).all() or plane.shape!=(4,)
            or not np.isfinite(plane).all()
            or not np.isclose(np.linalg.norm(plane[:3]),1.,atol=1e-6)
            or not np.isfinite([tolerance_m,min_inlier_fraction]).all()
            or tolerance_m<=0 or not 0<min_inlier_fraction<=1):
        raise ValueError('Finite held-out points, unit plane and explicit depth limits required')
    residual=np.abs(points@plane[:3]+plane[3])
    fraction=float(np.mean(residual<=tolerance_m))
    return dict(passed=fraction>=min_inlier_fraction,point_count=len(points),
        inlier_fraction=fraction,median_residual_m=float(np.median(residual)),
        p95_residual_m=float(np.percentile(residual,95)),tolerance_m=tolerance_m,
        min_inlier_fraction=min_inlier_fraction,plane_refitted=False)


def fit_plane(points, expected_up, *, tolerance_m=.008, min_inlier_fraction=.7,
              maximum_tilt_deg=10., trials=300, seed=42):
    """Fit a visible surface, independently of its object or task trajectory."""
    p,up=np.asarray(points,float),np.asarray(expected_up,float)
    if (p.ndim!=2 or p.shape[1]!=3 or len(p)<100 or not np.isfinite(p).all()
            or up.shape!=(3,) or not np.isfinite(up).all() or not np.isclose(np.linalg.norm(up),1.)
            or not np.isfinite([tolerance_m,min_inlier_fraction,maximum_tilt_deg]).all()
            or tolerance_m<=0 or not 0<min_inlier_fraction<=1 or not 0<maximum_tilt_deg<90
            or type(trials) is not int or trials<1):
        raise ValueError('Finite observed surface points and bounded plane-fit settings required')
    rng=np.random.default_rng(seed);best=None
    sample=p[np.linspace(0,len(p)-1,min(6000,len(p))).astype(int)]
    for _ in range(trials):
        a,b,c=sample[rng.choice(len(sample),3,replace=False)]
        n=np.cross(b-a,c-a);size=np.linalg.norm(n)
        if size<1e-10:continue
        n/=size
        if n@up<0:n=-n
        if n@up<np.cos(np.deg2rad(maximum_tilt_deg)):continue
        d=-n@a;keep=np.abs(sample@n+d)<=tolerance_m
        if best is None or keep.sum()>best[0]:best=(int(keep.sum()),n,d)
    if best is None:raise ValueError('No plane agrees with the declared up direction')
    n,d=best[1:]
    for _ in range(3):
        keep=np.abs(p@n+d)<=tolerance_m
        if keep.sum()<100:raise ValueError('Too few surface inliers')
        q=p[keep];center=q.mean(0);_,singular,vectors=np.linalg.svd(q-center,full_matrices=False)
        n=vectors[-1]
        if n@up<0:n=-n
        d=-n@center
    residual=np.abs(p@n+d);fraction=float(np.mean(residual<=tolerance_m))
    if fraction<min_inlier_fraction or n@up<np.cos(np.deg2rad(maximum_tilt_deg)):
        raise ValueError('Visible-plane coverage or orientation failed')
    if singular[1]/np.sqrt(len(q))<.02:raise ValueError('Surface observations lack two-dimensional extent')
    return np.r_[n,d],dict(inlier_fraction=fraction,point_count=len(p),
        residual_median_m=float(np.median(residual)),residual_p95_m=float(np.percentile(residual,95)),
        tolerance_m=tolerance_m,min_inlier_fraction=min_inlier_fraction,
        expected_up=up.tolist(),maximum_tilt_deg=maximum_tilt_deg,depth_is_predicted=True)


def mask_outline(mask):
    """Convex perimeter of an explicitly reviewed, visible planar-surface mask."""
    from scipy.spatial import ConvexHull
    a=np.asarray(mask)
    if a.ndim!=2 or a.dtype!=bool or np.count_nonzero(a)<100:
        raise ValueError('A native boolean visible-support mask is required')
    y,x=np.nonzero(a);points=np.c_[x,y];hull=ConvexHull(points)
    return points[hull.vertices].astype(float)


def slab(intrinsics, polygon_xy, plane, *, thickness_m, observed_points,
         evidence, max_span_m=1.5, max_plane_residual_m=.008,
         min_inlier_fraction=.7):
    from scipy.spatial import ConvexHull
    K,uv,p,observed=map(lambda x:np.asarray(x,float),
                       (intrinsics,polygon_xy,plane,observed_points))
    if (K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0
            or not np.allclose(K[2],[0,0,1]) or abs(np.linalg.det(K))<1e-12
            or uv.ndim!=2 or uv.shape[1]!=2 or len(uv)<3 or not np.isfinite(uv).all()
            or p.shape!=(4,) or not np.isfinite(p).all() or not np.isclose(np.linalg.norm(p[:3]),1.,atol=1e-6)
            or observed.ndim!=2 or observed.shape[1]!=3 or len(observed)<20 or not np.isfinite(observed).all()
            or not isinstance(evidence,str) or not evidence.strip()
            or not np.isfinite([thickness_m,max_span_m,max_plane_residual_m,min_inlier_fraction]).all()
            or not .005<=thickness_m<=.08 or max_span_m<=0 or max_plane_residual_m<=0
            or not 0<min_inlier_fraction<=1):
        raise ValueError('Reviewed contour, metric plane, observed thickness/depth and explicit bounds required')
    hull=ConvexHull(uv)
    if len(hull.vertices)!=len(uv):raise ValueError('Require a convex, nonrepeated contour in boundary order')
    order=hull.vertices
    if not (np.all(np.diff(np.r_[order,order[0]+len(order)])%len(order)==1)
            or np.all(np.diff(np.r_[order,order[0]-len(order)])%len(order)==len(order)-1)):
        raise ValueError('Contour vertices must follow the boundary')
    residual=np.abs(observed@p[:3]+p[3]);fraction=float(np.mean(residual<=max_plane_residual_m))
    if fraction<min_inlier_fraction:raise ValueError('Insufficient independent depth support for the plane')
    rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T
    denominators=rays@p[:3]
    if np.any(np.abs(denominators)<1e-8):raise ValueError('Contour ray is parallel to the plane')
    depth=-p[3]/denominators
    if np.any(depth<=0) or not np.isfinite(depth).all():raise ValueError('Support outline is behind the camera')
    top=rays*depth[:,None]
    v=np.r_[top,top-p[:3]*float(thickness_m)]
    if np.max(np.ptp(v,axis=0))>max_span_m:raise ValueError('Planar footprint exceeds the metric extent bound')
    mesh=ConvexHull(v);faces=mesh.simplices.copy()
    for i,tri in enumerate(faces):
        a,b,c=v[tri]
        if np.dot(np.cross(b-a,c-a),mesh.equations[i,:3])<0:faces[i]=tri[::-1]
    return v,faces,dict(schema='real2sim2real.planar-outline-support/1',
        hypothesis='Reviewed planar tabletop perimeter; thickness and plane measured from depth',
        evidence=evidence,polygon_xy=uv.tolist(),plane_camera=p.tolist(),
        thickness_m=float(thickness_m),depth_inlier_fraction=fraction,
        observed_depth_points=len(observed),volume_m3=float(mesh.volume),
        axis_extents_m=np.ptp(v,axis=0).tolist(),object_or_task_targets_used=False,
        physical_stability_established=False)
