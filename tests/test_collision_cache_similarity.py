import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.controller_actor_training.collision_cache_similarity import similarity,reuse,digest


def mesh():
    return np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32),np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])


def test_reuse_requires_exact_geometry_params_and_verified_cache(tmp_path):
    from flash_chord.assets.mesh import ConvexHull,_cache_key,_save_cache,_load_cache
    v,f=mesh();params=dict(threshold=.05,max_convex_hull=-1,preprocess_mode='auto')
    source=tmp_path/'source';target=tmp_path/'target';p=source/(_cache_key(v,f,params)+'.npz')
    _save_cache(p,[ConvexHull(v,f.astype(np.int32))])
    r=Rotation.from_euler('xyz',[10,30,70],degrees=True).as_matrix();t=np.array([.1,.2,-.3])
    changed=(.84*(v@r.T)+t).astype(np.float32)
    result=reuse(v,f,changed,f,source_cache_dir=source,target_cache_dir=target,source_cache_sha256=digest(p))
    assert result['hulls']==1 and result['max_vertex_error_m']<1e-6
    saved=_load_cache(target/(result['target_cache_key']+'.npz'))[0]
    np.testing.assert_allclose(saved.vertices,changed,atol=1e-6)
    assert not result['fresh_coacd_rerun_equivalence_claimed']
    with pytest.raises(ValueError,match='hash'):reuse(v,f,changed,f,source_cache_dir=source,target_cache_dir=target,source_cache_sha256='wrong')


def test_deformation_and_reflection_are_rejected():
    v,f=mesh();bad=v.copy();bad[3,2]*=1.1
    with pytest.raises(ValueError,match='similarity'):similarity(v,bad)
    with pytest.raises(ValueError,match='similarity'):similarity(v,v*[-1,1,1])


def test_target_cache_cannot_be_silently_overwritten(tmp_path):
    from flash_chord.assets.mesh import ConvexHull,_cache_key,_save_cache
    v,f=mesh();params=dict(threshold=.05,max_convex_hull=-1,preprocess_mode='auto')
    p=tmp_path/'source'/(_cache_key(v,f,params)+'.npz');_save_cache(p,[ConvexHull(v,f.astype(np.int32))])
    changed=v*.8;dest=tmp_path/'target'/(_cache_key(changed,f,params)+'.npz')
    _save_cache(dest,[ConvexHull(v,f.astype(np.int32))])
    with pytest.raises(FileExistsError):reuse(v,f,changed,f,source_cache_dir=p.parent,target_cache_dir=dest.parent,source_cache_sha256=digest(p))
