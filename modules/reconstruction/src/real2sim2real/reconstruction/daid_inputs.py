"""Explicit shared-camera inputs for a controlled Do as I Do tracker ablation."""
import numpy as np


def camera_pointmap(depth_m, K):
    depth=np.asarray(depth_m,dtype=np.float32);K=np.asarray(K,dtype=np.float64)
    if depth.ndim!=2 or K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:
        raise ValueError('Native depth grid and finite positive focal lengths required')
    if not np.allclose(K[2],[0,0,1]) or abs(K[0,1])+abs(K[1,0])>1e-9:
        raise ValueError('Standard zero-skew pinhole calibration required')
    y,x=np.indices(depth.shape,dtype=np.float32)
    points=np.stack([-(x-K[0,2])*depth/K[0,0],-(y-K[1,2])*depth/K[1,1],depth],axis=-1)
    points[~np.isfinite(depth)|(depth<=0)]=np.nan
    return points.astype(np.float32,copy=False)


def shared_pointmap_dict(rgb,depth_m,K,device):
    import torch
    h,w=depth_m.shape
    if rgb.shape!=(h,w,3):raise ValueError('RGB and native depth grids differ')
    normalized=np.asarray(K,dtype=np.float32).copy();normalized[0]/=w;normalized[1]/=h
    return dict(pointmap=torch.as_tensor(camera_pointmap(depth_m,K),device=device).permute(2,0,1),
                intrinsics=torch.as_tensor(normalized,device=device),
                pts_color=torch.as_tensor(rgb.copy(),device=device).permute(2,0,1).float()/255.)


def recover_original_camera_pose(raw_rotation,raw_translation,raw_scale,Q,normalization,center_z):
    """Undo only fixed mesh canonicalization; never fit to later reference poses."""
    R=np.asarray(raw_rotation,float);t=np.asarray(raw_translation,float).reshape(3)
    s=np.asarray(raw_scale,float).reshape(3)
    if not np.allclose(s,s[0],rtol=1e-5,atol=0):raise ValueError('Uniform scale required')
    D=np.diag([-1.,-1.,1.]);result=np.eye(4)
    result[:3,:3]=(Q@R@D).T
    result[:3,3]=t@D-(np.asarray(center_z)*normalization*s[0])@R@D
    if not np.isfinite(result).all():raise ValueError('Nonfinite camera pose')
    return result
