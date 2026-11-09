"""Bake source RGB onto visible mesh vertices without changing geometry/poses.

This is appearance refinement, never pose, scale, tracking or physics approval.
Views must carry source provenance in the caller's artifact receipt. Occluder
masks and a triangle z-buffer prevent painting people or hidden back surfaces.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from .source_pose_audit import projection_audit


def bake(vertices, faces, normals, colors, views, *, visibility_tolerance_m=.005):
    v, f, n, c = np.asarray(vertices, float), np.asarray(faces), np.asarray(normals, float), np.asarray(colors)
    if (v.ndim != 2 or v.shape[1] != 3 or n.shape != v.shape or c.shape != v.shape
            or c.dtype != np.uint8 or not views or not np.isfinite(v).all()
            or not np.isfinite(n).all() or not np.allclose(np.linalg.norm(n, axis=1), 1., atol=1e-3)
            or not np.isfinite(visibility_tolerance_m) or visibility_tolerance_m <= 0):
        raise ValueError("Finite metric mesh, unit normals, uint8 RGB and observed views required")
    result=c.copy()
    strongest=np.zeros(len(v))
    counts=[]
    for view in views:
        r,t,k=map(lambda x:np.asarray(x,float),(view["rotation"],view["translation"],view["intrinsics"]))
        rgb=np.asarray(view["rgb"]);mask=np.asarray(view["object_mask"]);human=np.asarray(view["human_mask"])
        if (rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or
                mask.shape!=rgb.shape[:2] or mask.dtype!=bool or human.shape!=mask.shape or human.dtype!=bool):
            raise ValueError("Native RGB and explicit matching boolean masks required")
        points=v@r.T+t
        diagnostic,surface=projection_audit(points,f,k,mask,np.asarray(view["predicted_depth_m"],float),human_mask=human)
        project=points@k.T
        uv=project[:,:2]/project[:,2:]
        xy=np.rint(uv).astype(int)
        h,w=mask.shape;x0,y0,x1,y1=diagnostic["roi_xyxy"]
        good=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
        ids=np.flatnonzero(good)
        local=xy[ids]-[x0,y0]
        close=np.abs(points[ids,2]-surface[local[:,1],local[:,0]])<=visibility_tolerance_m
        observed=mask[xy[ids,1],xy[ids,0]]&~human[xy[ids,1],xy[ids,0]]
        cosine=-np.einsum("ij,ij->i",n[ids]@r.T,points[ids])/np.linalg.norm(points[ids],axis=1)
        weight=np.maximum(0.,cosine)**2
        keep=close&observed&(cosine>.15)&(weight>strongest[ids])
        ids=ids[keep]
        values=np.stack([map_coordinates(rgb[:,:,channel].astype(float),
            [uv[ids,1],uv[ids,0]],order=1,mode="nearest") for channel in range(3)],axis=1)
        result[ids]=np.clip(np.rint(values),0,255).astype(np.uint8)
        strongest[ids]=weight[keep]
        counts.append(dict(source_frame=int(view["source_frame"]),updated_vertices=len(ids),
                           native_silhouette_iou=diagnostic["full_silhouette"]["iou"]))
    return result,dict(views=counts,updated_vertices=int(np.count_nonzero(strongest)),
                       total_vertices=len(v),geometry_changed=False,poses_changed=False,
                       visibility_tolerance_m=visibility_tolerance_m,source_only=True,
                       candidate_approved=False)
