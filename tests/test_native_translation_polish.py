import numpy as np
import pytest
from real2sim2real.reconstruction_repair.native_translation_polish import propose


def test_submillimetre_depth_failure_gets_fixed_rotation_proposal():
    v=np.array([[-.1,-.1,0],[.1,-.1,0],[.1,.1,0],[-.1,.1,0]],float)
    f=np.array([[0,1,2],[0,2,3]]);k=np.array([[100.,0,20],[0,100.,20],[0,0,1]])
    p=np.eye(4);p[2,3]=1.;mask=np.zeros((40,40),bool);mask[10:31,10:31]=True
    result=propose(v,f,p,k,mask,np.zeros_like(mask),np.full(mask.shape,1.1005),p,p)
    q=np.asarray(result['pose_camera'])
    assert result['selected']['passed'] and result['requires_gpu_validation']
    np.testing.assert_array_equal(q[:3,:3],p[:3,:3])
    assert 0<q[2,3]-p[2,3]<=.004


def test_wrong_native_history_cannot_be_ignored():
    p=np.eye(4);p[2,3]=1.;bad=p.copy();bad[0,0]=2
    with pytest.raises(ValueError,match='neighbours'):
        propose(np.zeros((3,3)),np.array([[0,1,2]]),p,np.eye(3),np.ones((2,2),bool),np.zeros((2,2),bool),np.ones((2,2)),bad,p)


def test_visible_scope_is_explicit_and_support_is_not_ignored():
    v=np.array([[-.1,-.1,0],[.1,-.1,0],[.1,.1,0],[-.1,.1,0]],float)
    f=np.array([[0,1,2],[0,2,3]]);k=np.array([[100.,0,20],[0,100.,20],[0,0,1]])
    p=np.eye(4);p[2,3]=1.;mask=np.zeros((40,40),bool);mask[10:31,10:20]=True
    human=np.zeros_like(mask);human[:,20:]=True;depth=np.ones(mask.shape)
    result=propose(v,f,p,k,mask,human,depth,p,p,minimum_iou=.9,silhouette_scope='human_excluded')
    assert result['selected']['passed'] and result['silhouette_scope']=='human_excluded'
    failed=propose(v,f,p,k,mask,human,depth,p,p,minimum_iou=.9,silhouette_scope='human_excluded',
        support_plane_camera=np.array([0,-1,0,.05]))
    assert failed['passed'] is False


def test_dense_surface_pass_does_not_hide_sparse_native_depth_boundary():
    from scipy.spatial.transform import Rotation
    xx,yy=np.meshgrid(np.linspace(-.2,.2,16),np.linspace(-.2,.2,8));v=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    f=[]
    for y in range(7):
        for x in range(15):
            a=y*16+x;f.extend([[a,a+1,a+16],[a+1,a+17,a+16]])
    f=np.asarray(f);k=np.array([[100.,0,40],[0,100.,40],[0,0,1.]])
    p=np.eye(4);p[2,3]=1.;p[:3,:3]=Rotation.from_euler('z',.001,degrees=True).as_matrix()
    mask=np.zeros((80,80),bool);mask[20:61,20:61]=True
    depth=np.full(mask.shape,1.02);pixels=np.rint((v[:,:2]*100)+40).astype(int);depth[pixels[:,1],pixels[:,0]]=1.1005
    result=propose(v,f,p,k,mask,np.zeros_like(mask),depth,p,p,support_plane_camera=[0,-1,0,1],require_sparse_native_depth=True)
    assert result['before']['predicted_surface_depth']['consistency_passed']
    assert result['sparse_before']['depth_error_m']>.1
    assert result['selected']['sparse_native']['depth_error_m']<=.1
    assert result['selected']['passed'] and result['sparse_native_depth_required']
    np.testing.assert_array_equal(np.asarray(result['pose_camera'])[:3,:3],p[:3,:3])
