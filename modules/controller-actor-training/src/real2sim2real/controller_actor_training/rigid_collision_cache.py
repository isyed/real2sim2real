"""Reframe validated collision hulls when only the object body frame changes.

Convex decomposition need not be rotation invariant. Re-decomposing an
identical surface in another frame can change its contact geometry. This
explicit alternative preserves the original hulls after strict rigid-mesh
equivalence checks. Scaling, changed topology and fitted nonrigid geometry are
rejected. Cache files retain their normal geometry/parameter content keys.
"""
from pathlib import Path

def rigid_equivalence(source_vertices,source_faces,target_vertices,target_faces,*,max_error_m=2e-6):
    import numpy as np
    a,b=np.asarray(source_vertices,float),np.asarray(target_vertices,float)
    f,g=np.asarray(source_faces),np.asarray(target_faces)
    if (a.ndim!=2 or a.shape[1]!=3 or len(a)<4 or b.shape!=a.shape or
            f.ndim!=2 or f.shape[1]!=3 or g.shape!=f.shape or f.dtype.kind not in 'iu' or g.dtype.kind not in 'iu'
            or not len(f) or f.min()<0 or f.max()>=len(a) or not np.array_equal(f,g)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or isinstance(max_error_m,bool) or not np.isfinite(max_error_m) or max_error_m<=0):
        raise ValueError('Finite corresponding metric vertices and identical valid triangle topology required')
    ac,bc=a.mean(0),b.mean(0);u,s,v=np.linalg.svd((a-ac).T@(b-bc))
    if s[-1]<=s[0]*1e-12:raise ValueError('Rigid equivalence requires nondegenerate three-dimensional geometry')
    correction=np.eye(3);correction[-1,-1]=np.linalg.det(v.T@u.T)
    rotation=v.T@correction@u.T;translation=bc-rotation@ac
    error=float(np.linalg.norm(a@rotation.T+translation-b,axis=1).max())
    if error>max_error_m:raise ValueError('Meshes differ beyond a rigid body-frame change')
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=translation
    return transform,dict(maximum_vertex_error_m=error,max_error_m=max_error_m,
        vertices=len(a),faces=len(f),topology_identical=True,scale_changed=False)

def reframe(source_vertices,source_faces,target_vertices,target_faces,source_cache_dir,target_cache_dir,*,
            params=None,max_error_m=2e-6):
    import numpy as np
    from flash_chord.assets.mesh import _cache_key,_load_cache,_save_cache,ConvexHull
    from real2sim2real.artifacts import digest,write_json
    transform,check=rigid_equivalence(source_vertices,source_faces,target_vertices,target_faces,max_error_m=max_error_m)
    parameters=dict(threshold=.05,max_convex_hull=-1,preprocess_mode='auto')
    if params is not None:parameters.update(params)
    old_key=_cache_key(np.asarray(source_vertices),np.asarray(source_faces),parameters)
    new_key=_cache_key(np.asarray(target_vertices),np.asarray(target_faces),parameters)
    old=Path(source_cache_dir)/(old_key+'.npz');new=Path(target_cache_dir)/(new_key+'.npz')
    if not old.is_file() or old.is_symlink():raise ValueError('Validated source collision cache is unavailable')
    if new.exists() or new.resolve()==old.resolve():raise FileExistsError('Preserve existing cache results; use a new target directory')
    hulls=_load_cache(old);converted=[]
    if not hulls:raise ValueError('Empty source decomposition')
    for hull in hulls:
        vertices,faces=np.asarray(hull.vertices),np.asarray(hull.faces)
        if (vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<4 or not np.isfinite(vertices).all()
                or faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.dtype.kind not in 'iu'
                or faces.min()<0 or faces.max()>=len(vertices)):
            raise ValueError('Invalid source collision hull')
        points=vertices.astype(float)@transform[:3,:3].T+transform[:3,3]
        converted.append(ConvexHull(points.astype(np.float32),faces.astype(np.int32)))
    _save_cache(new,converted)
    receipt=dict(schema='real2sim2real.rigid-collision-cache/1',method='rigid_reframe_of_validated_hulls',
        **check,source_to_target=transform.tolist(),hulls=len(converted),parameters=parameters,
        source_cache_key=old_key,target_cache_key=new_key,source_cache_sha256=digest(old),target_cache_sha256=digest(new),
        collision_geometry_preserved_under_rigid_transform=True,convex_decomposition_rerun=False,
        target_settling_validation_required=True,task_success_established=False)
    write_json(new.with_suffix('.receipt.json'),receipt);return receipt
