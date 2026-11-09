"""Full-viewport native raster binning with conservative face capacity.

No crop, resampling, geometry or shader changes. This adapter still requires
native GPU parity before use. Every bin can hold every original mesh face;
near-plane geometry falls back to the unmodified scorer to avoid clipped-face
capacity assumptions.
"""
import math


def bin_parameters(width, height, faces, *, max_index_bytes=2500000000,
                   max_bins_per_axis=21):
    if any(type(v) is not int or v<=0 for v in
           (width,height,faces,max_index_bytes,max_bins_per_axis)):
        raise ValueError('Positive integer image/mesh/buffer sizes required')
    # Native coarse-bin indices are int32. Capacity >= all original faces avoids
    # accepting silent overflow when a candidate projects into a small region.
    for size in (32,64,128,256):
        # Pinned PyTorch3D requires ceil(max(image_size)/bin_size) < 22.
        # Its error calls this "faces per bin", but the calculation is bin count.
        if math.ceil(max(width,height)/size)>max_bins_per_axis:
            continue
        slots=math.ceil(width/size)*math.ceil(height/size)*(faces+1)
        if slots*4<=max_index_bytes:
            return dict(bin_size=size,max_faces_per_bin=faces+1,index_buffer_bytes=slots*4)
    return None


def binned_renderer(tracker,intrinsics,width,height,device,faces):
    params=bin_parameters(width,height,faces)
    if params is None:return None
    renderer=tracker._build_silhouette_renderer(intrinsics,width,height,device)
    renderer.rasterizer.raster_settings.bin_size=params['bin_size']
    renderer.rasterizer.raster_settings.max_faces_per_bin=params['max_faces_per_bin']
    return renderer


def native_binned_iou(tracker,mesh,pose,intrinsics,mask,width,height,device,*,renderer=None,full_scorer):
    import torch
    with torch.no_grad():
        q=pose['rotation'].detach().float().reshape(1,4).to(device)
        t=pose['translation'].detach().float().reshape(1,3).to(device)
        scale=pose['scale'].detach().float().reshape(1,-1).to(device)
        tfm=tracker.compose_transform(scale,tracker.quaternion_to_matrix(q),t)
        verts=tfm.transform_points(mesh.verts_packed()[None])[0]
        valid=bool(torch.isfinite(verts).all() and (verts[:,2]>1e-4).all())
    chosen=binned_renderer(tracker,intrinsics,width,height,device,len(mesh.faces_packed())) if valid else None
    return full_scorer(mesh,pose,intrinsics,mask,width,height,device,
                       renderer=renderer if chosen is None else chosen)
