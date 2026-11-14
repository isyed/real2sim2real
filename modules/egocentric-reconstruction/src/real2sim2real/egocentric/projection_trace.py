"""Trace a z-buffered target depth pixel to its actual source observation."""
import numpy as np
from .camera import intrinsic, transform


def project_with_source(points, valid, target_from_source, k, size):
    """Match project_depth's float32 z-buffer; retain source IDs and subpixels.

    This verifies software provenance, not calibration or physical visibility in
    the target camera. Unobserved target occluders cannot be detected here.
    """
    p=np.asarray(points,float);v=np.asarray(valid,bool)
    if p.shape!=(*v.shape,3) or v.ndim!=2:
        raise ValueError("Organized points and matching validity required")
    ids=np.flatnonzero(v.ravel()&np.isfinite(p).all(-1).ravel())
    xyz=transform(p.reshape(-1,3)[ids],target_from_source)
    keep=np.isfinite(xyz).all(1)&(xyz[:,2]>0);xyz,ids=xyz[keep],ids[keep]
    q=xyz@intrinsic(k).T;uv=q[:,:2]/q[:,2:];pixels=np.rint(uv).astype(int)
    width,height=map(int,size)
    inside=(pixels[:,0]>=0)&(pixels[:,0]<width)&(pixels[:,1]>=0)&(pixels[:,1]<height)
    xyz,ids,uv,pixels=xyz[inside],ids[inside],uv[inside],pixels[inside]
    target_ids=pixels[:,1]*width+pixels[:,0]
    # Float32 depths preserve the same tie/rounding convention as project_depth.
    z=xyz[:,2].astype(np.float32)
    order=np.lexsort((ids,z,target_ids));ordered=target_ids[order]
    first=np.r_[True,np.diff(ordered)!=0] if len(order) else np.zeros(0,bool)
    chosen=order[first];dst=target_ids[chosen]
    depth=np.zeros(width*height,np.float32);source=np.full(width*height,-1,np.int64)
    subpixel=np.full((width*height,2),np.nan)
    depth[dst]=z[chosen];source[dst]=ids[chosen];subpixel[dst]=uv[chosen]
    return dict(depth=depth.reshape(height,width),source_index=source.reshape(height,width),
                target_subpixel=subpixel.reshape(height,width,2))
