import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.controller_actor_training.rigid_collision_cache import rigid_equivalence,reframe

def mesh():
    return np.array([[0.,0,0],[.1,0,0],[0,.1,0],[0,0,.2]]),np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)

def test_rotated_cache_preserves_hulls_and_content_keys(tmp_path):
    from flash_chord.assets.mesh import ConvexHull,_save_cache,_load_cache,_cache_key
    a,f=mesh();r=Rotation.from_euler('xyz',[10,20,30],degrees=True).as_matrix();t=np.array([.02,-.04,.03]);b=a@r.T+t
    params=dict(threshold=.05,max_convex_hull=-1,preprocess_mode='auto');old=tmp_path/'old';old.mkdir()
    _save_cache(old/(_cache_key(a,f,params)+'.npz'),[ConvexHull(a.astype(np.float32),f)])
    receipt=reframe(a,f,b,f,old,tmp_path/'new')
    saved=_load_cache(tmp_path/'new'/(receipt['target_cache_key']+'.npz'))
    assert np.allclose(saved[0].vertices,a@r.T+t,atol=2e-8)
    assert np.array_equal(saved[0].faces,f) and receipt['hulls']==1
    assert not receipt['convex_decomposition_rerun']
    with pytest.raises(FileExistsError):reframe(a,f,b,f,old,tmp_path/'new')

@pytest.mark.parametrize('change',['scale','shear','reflection','topology','vertex_order'])
def test_nonrigid_or_unmatched_reuse_is_rejected(change):
    a,f=mesh();b=a.copy();g=f.copy()
    if change=='scale':b*=1.01
    if change=='shear':b[:,0]+=b[:,2]*.1
    if change=='reflection':b[:,0]*=-1
    if change=='topology':g=g[::-1]
    if change=='vertex_order':b=b[[1,2,0,3]]
    with pytest.raises(ValueError):rigid_equivalence(a,f,b,g)
