"""Read-only offset diagnostics with fixed identities and static-scene controls."""
import cv2
import numpy as np
from .camera import intrinsic


def track_points(reference, target, pixels, maximum_roundtrip_px=1.):
    """Direct forward/backward tracks from one anchor; no offset-dependent reseeding."""
    def gray(im):
        return cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) if im.ndim==3 else im
    a,b=gray(reference),gray(target);p=np.asarray(pixels,np.float32).reshape(-1,1,2)
    q,status,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(25,25),maxLevel=4)
    back,reverse,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(25,25),maxLevel=4)
    error=np.linalg.norm(back[:,0]-p[:,0],axis=1)
    valid=(status[:,0]>0)&(reverse[:,0]>0)&np.isfinite(q[:,0]).all(1)&(error<=maximum_roundtrip_px)
    valid&=(q[:,0,0]>=0)&(q[:,0,0]<b.shape[1])&(q[:,0,1]>=0)&(q[:,0,1]<b.shape[0])
    return q[:,0],valid,error


def project(points,pose,k):
    p=np.asarray(points)@pose[:3,:3].T+pose[:3,3];q=p@intrinsic(k).T
    if (p[:,2]<=0).any():raise ValueError('Points behind camera')
    return q[:,:2]/q[:,2:]


def static_registration(points,pixels,k,valid,fit_mask):
    """Absolute diagnostic pose from source 3D to target image, with held points.

    This may absorb calibration errors as well as camera motion. It is never an
    updated rig calibration. Holdout reprojection error is required for interpretation.
    """
    p,q=np.asarray(points,float),np.asarray(pixels,float)
    valid=np.asarray(valid,bool);fit=np.asarray(fit_mask,bool)
    if p.shape!=(len(q),3) or q.shape[1:]!=(2,) or valid.shape!=(len(q),) or fit.shape!=valid.shape:
        raise ValueError('Matching static 3D/2D observations and split required')
    train=valid&fit;held=valid&~fit
    if train.sum()<12 or held.sum()<8:return None,dict(available=False,reason='insufficient_static_tracks',fit_points=int(train.sum()),held_points=int(held.sum()))
    ok,rv,tv,inliers=cv2.solvePnPRansac(p[train],q[train],intrinsic(k),None,iterationsCount=200,reprojectionError=3,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<10:return None,dict(available=False,reason='static_pnp_rejected')
    ids=inliers.ravel();rv,tv=cv2.solvePnPRefineLM(p[train][ids],q[train][ids],k,None,rv,tv)
    t=np.eye(4);t[:3,:3]=cv2.Rodrigues(rv)[0];t[:3,3]=tv.ravel()
    errors=np.linalg.norm(project(p[held],t,k)-q[held],axis=1)
    return t,dict(available=True,fit_points=int(train.sum()),fit_inliers=len(ids),held_points=int(held.sum()),held_median_px=float(np.median(errors)),held_p90_px=float(np.quantile(errors,.9)),calibration_update=False)


def common_cohort(validity):
    """Require identical identities across the whole offset sweep."""
    v=np.asarray(validity,bool)
    if v.ndim!=2 or len(v)==0:raise ValueError('Offset by identity validity grid required')
    return v.all(axis=0)


def track_patch_series(reference,targets,center,radius=16):
    """Track one anchor-selected corner cohort; infer patch-center displacement.

    This is a local appearance-motion estimate, not physical center ground truth.
    No offset-dependent feature reseeding or projection-error selection is used.
    """
    gray=cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY) if reference.ndim==3 else reference
    mask=np.zeros_like(gray);cv2.circle(mask,tuple(np.rint(center).astype(int)),radius,255,-1)
    features=cv2.goodFeaturesToTrack(gray,maxCorners=12,qualityLevel=.02,minDistance=3,mask=mask)
    if features is None:return np.tile(center,(len(targets),1)),np.zeros(len(targets),bool),dict(common_corners=0)
    p=features[:,0];tracked=[track_points(reference,target,p) for target in targets]
    common=common_cohort([row[1] for row in tracked]);count=int(common.sum())
    positions=[];valid=[];spread=[]
    for q,_,_ in tracked:
        delta=q[common]-p[common]
        movement=np.median(delta,axis=0) if count else np.zeros(2)
        scatter=float(np.median(np.linalg.norm(delta-movement,axis=1))) if count else None
        positions.append(np.asarray(center)+movement);spread.append(scatter)
        valid.append(count>=3 and scatter<=2.)
    return np.asarray(positions),np.asarray(valid),dict(common_corners=count,median_motion_scatter_px=spread,physical_center_verified=False)
