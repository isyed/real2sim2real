"""Visible boundary and projected diameter constraints for container hypotheses.

Only complete, explicitly observed end contours supply ellipse diameters.
Partial arcs never imply a completed contour. Measurements are image pixels;
metric diameter depends on the candidate pose and calibration.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .camera import intrinsic, rigid, transform
from .container_profile import validate, surface
from .tapered_container import projected_outline, outline_samples


def _points(annotation):
    p=np.asarray(annotation['pixels'],float)
    sigma=annotation['sigma_px']
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all() or not np.isfinite(sigma) or sigma<=0:
        raise ValueError("Finite visible boundary pixels and positive uncertainty required")
    if annotation['kind'] not in ['bottom','rim','silhouette']:
        raise ValueError("Known contour kind required")
    if annotation.get('complete',False) and (annotation['kind']=='silhouette' or len(p)<6):
        raise ValueError("Complete end contours need at least six observations")
    return p,float(sigma)


def ellipse_axes(points):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<6 or not np.isfinite(p).all():
        raise ValueError("At least six finite ellipse samples required")
    if np.linalg.svd(p-p.mean(0),compute_uv=False)[1]<1:
        raise ValueError("Degenerate end contour")
    center,axes,angle=cv2.fitEllipse(p.astype(np.float32))
    if not np.isfinite([*center,*axes,angle]).all() or min(axes)<=0:
        raise ValueError("Invalid image ellipse")
    return dict(center_px=list(center),diameters_px=sorted(axes,reverse=True),angle_deg=angle)


def projected_boundary(profile,pose,k,kind):
    a=validate(profile);rigid(pose);intrinsic(k)
    if kind=='silhouette':
        return outline_samples(projected_outline(surface(a),pose,k),360)
    if kind not in ['bottom','rim']:raise ValueError("Unknown end kind")
    angle=np.linspace(0,2*np.pi,360,endpoint=False)
    radius=a[0] if kind=='bottom' else a[3]
    z=0 if kind=='bottom' else a[4]
    p=np.c_[radius*np.cos(angle),radius*np.sin(angle),np.full(len(angle),z)]
    q=transform(p,pose)@np.asarray(k).T
    if (q[:,2]<=0).any():raise ValueError("Boundary behind camera")
    return q[:,:2]/q[:,2:]


def constraint_report(profile,pose,k,annotations):
    """Report symmetric error only for complete end boundaries.

    Partial observed-to-model boundary distances penalize undersize AND oversize,
    unlike containment-only masks. Occluded/unknown entries provide no residual.
    They are never counted as passing. Assistant annotations are not ground truth.
    """
    rows=[];residual=[]
    for annotation in annotations:
        visibility=annotation['visibility']
        if visibility not in ['visible','occluded','unknown']:raise ValueError("Explicit visibility required")
        if visibility!='visible':
            rows.append(dict(name=annotation['name'],available=False,visibility=visibility));continue
        p,sigma=_points(annotation)
        predicted=projected_boundary(profile,pose,k,annotation['kind'])
        errors=cKDTree(predicted).query(p)[0]
        # Fixed observed samples keep residual dimension independent of pose.
        residual.extend(errors/sigma/np.sqrt(len(p)))
        row=dict(name=annotation['name'],available=True,visibility=visibility,kind=annotation['kind'],
                 samples=len(p),sigma_px=sigma,complete=annotation.get('complete',False),
                 observed_to_model_median_px=float(np.median(errors)),observed_to_model_p90_px=float(np.quantile(errors,.9)),
                 diameter_available=False)
        if annotation.get('complete',False):
            observed=ellipse_axes(p);model=ellipse_axes(predicted)
            delta=np.array(model['diameters_px'])-observed['diameters_px']
            # Diameter uncertainty includes both opposing boundary locations.
            residual.extend(delta/(2*sigma))
            dense=outline_samples(p,360)
            reverse=cKDTree(dense).query(predicted)[0]
            residual.extend(reverse/sigma/np.sqrt(len(reverse)))
            row.update(diameter_available=True,observed_ellipse=observed,predicted_ellipse=model,
                       diameter_error_px=delta.tolist(),diameter_uncertainty_px=2*sigma,
                       model_to_observed_p90_px=float(np.quantile(reverse,.9)))
        rows.append(row)
    return dict(rows=rows,available_constraints=sum(r['available'] for r in rows),
                complete_diameters=sum(r.get('diameter_available',False) for r in rows),
                observations_are_independent_ground_truth=False,metric_diameter_measured=False),np.asarray(residual)


def contour_residuals(profile,pose,k,annotations):
    return constraint_report(profile,pose,k,annotations)[1]
