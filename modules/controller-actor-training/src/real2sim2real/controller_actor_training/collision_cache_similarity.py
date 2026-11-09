"""Explicit reuse of a verified collision proxy under one mesh similarity.

This transforms existing convex pieces; it does not claim to reproduce a fresh
CoACD partition bit for bit. Use only for identical triangle correspondence
under a positive uniform scale and rigid local-frame change. It never repairs
or simplifies an unrelated/deformed reconstruction.
"""
from pathlib import Path
import hashlib,json
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def similarity(source,target,*,max_error_m=1e-6):
    a,b=map(lambda x:np.asarray(x,float),(source,target))
    if (a.ndim!=2 or a.shape[1]!=3 or a.shape!=b.shape or len(a)<4
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(max_error_m) or not 0<max_error_m<=1e-4):
        raise ValueError('Corresponding finite metric vertices and a strict tolerance required')
    ca,cb=a.mean(0),b.mean(0);x,y=a-ca,b-cb
    u,values,vt=np.linalg.svd(x.T@y)
    if values[-1]<=1e-10*values[0]:raise ValueError('Full-rank mesh correspondence required')
    correction=np.eye(3);correction[2,2]=np.linalg.det(vt.T@u.T)
    r=vt.T@correction@u.T
    scale=float(np.sum(y*(x@r.T))/np.sum(x*x));shift=cb-scale*(r@ca)
    error=float(np.linalg.norm(a@(scale*r.T)+shift-b,axis=1).max())
    if scale<=0 or error>max_error_m:raise ValueError('Meshes are not one positive uniform similarity')
    return dict(scale=scale,rotation=r,translation=shift,max_error_m=error)


def reuse(source_vertices,source_faces,target_vertices,target_faces,*,source_cache_dir,
          target_cache_dir,source_cache_sha256,params=None,max_error_m=1e-6):
    from flash_chord.assets.mesh import _cache_key,_load_cache,_save_cache,ConvexHull
    source_vertices,target_vertices=map(np.asarray,(source_vertices,target_vertices))
    source_faces,target_faces=map(np.asarray,(source_faces,target_faces))
    if (source_faces.ndim!=2 or source_faces.shape[1]!=3 or source_faces.dtype.kind not in 'iu'
            or target_faces.dtype.kind not in 'iu' or not len(source_faces)
            or not np.array_equal(source_faces,target_faces) or source_faces.min()<0
            or source_faces.max()>=len(source_vertices)):
        raise ValueError('Exact valid source/target triangle correspondence required')
    transform=similarity(source_vertices,target_vertices,max_error_m=max_error_m)
    params=dict(threshold=.05,max_convex_hull=-1,preprocess_mode='auto') if params is None else dict(params)
    if not all(k in params for k in ['threshold','max_convex_hull','preprocess_mode']):
        raise ValueError('Explicit complete decomposition parameters required')
    original=Path(source_cache_dir)/(_cache_key(source_vertices,source_faces,params)+'.npz')
    if digest(original)!=source_cache_sha256:raise ValueError('Verified source collision-cache hash differs')
    hulls=_load_cache(original)
    if not hulls or len(hulls)>10000:raise ValueError('Nonempty bounded source collision proxy required')
    result=[]
    for hull in hulls:
        v,f=np.asarray(hull.vertices),np.asarray(hull.faces)
        if (v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all()
                or f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in 'iu'
                or not len(f) or f.min()<0 or f.max()>=len(v)):
            raise ValueError('Invalid cached collision hull')
        result.append(ConvexHull((transform['scale']*(v@transform['rotation'].T)+transform['translation']).astype(np.float32),f.astype(np.int32)))
    target=Path(target_cache_dir)/(_cache_key(target_vertices,target_faces,params)+'.npz')
    if target.exists():
        saved=_load_cache(target)
        if len(saved)!=len(result) or any(not np.array_equal(a.vertices,b.vertices) or not np.array_equal(a.faces,b.faces) for a,b in zip(saved,result)):
            raise FileExistsError('Preserve a different existing collision cache')
    else:_save_cache(target,result)
    receipt=dict(schema='real2sim2real.similarity-collision-cache/1',source_cache_sha256=source_cache_sha256,
        source_cache_key=original.stem,target_cache_key=target.stem,target_cache_sha256=digest(target),
        scale=transform['scale'],rotation=transform['rotation'].tolist(),translation=transform['translation'].tolist(),
        max_vertex_error_m=transform['max_error_m'],triangle_correspondence_exact=True,
        hulls=len(result),decomposition_parameters=params,
        generated_as='Verified prior convex pieces transformed by the exact mesh similarity',
        fresh_coacd_rerun_equivalence_claimed=False,physics_validated=False)
    sidecar=target.with_suffix('.similarity.json')
    if sidecar.exists() and json.loads(sidecar.read_text())!=receipt:raise FileExistsError('Preserve a different collision-cache receipt')
    sidecar.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
