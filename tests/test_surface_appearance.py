import numpy as np
from real2sim2real.reconstruction_repair.surface_appearance import bake


def scene():
    front=np.array([[-.6,-.6,0.],[.6,-.6,0.],[.6,.6,0.],[-.6,.6,0.]])
    vertices=np.r_[front,front+[0,0,.2]]
    faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
    normals=np.tile([0.,0.,-1.],(8,1));colors=np.full((8,3),80,np.uint8)
    mask=np.zeros((40,40),bool);mask[14:27,14:27]=True
    rgb=np.zeros((40,40,3),np.uint8);rgb[:]=[240,20,10]
    view=dict(source_frame=7,rotation=np.eye(3),translation=np.array([0.,0.,2.]),
        intrinsics=np.array([[20.,0,20],[0,20,20],[0,0,1.]]),rgb=rgb,
        object_mask=mask,human_mask=np.zeros_like(mask),predicted_depth_m=np.full((40,40),2.))
    return vertices,faces,normals,colors,view


def test_bake_does_not_paint_hidden_surface_or_change_geometry():
    v,f,n,c,view=scene();original=v.copy()
    colors,receipt=bake(v,f,n,c,[view])
    assert np.array_equal(colors[:4],np.tile([240,20,10],(4,1)))
    assert np.array_equal(colors[4:],c[4:])
    assert np.array_equal(v,original)
    assert receipt["updated_vertices"]==4 and not receipt["candidate_approved"]


def test_human_occlusion_is_not_baked_into_object():
    v,f,n,c,view=scene();view["human_mask"][:]=True
    colors,receipt=bake(v,f,n,c,[view])
    assert np.array_equal(colors,c) and receipt["updated_vertices"]==0
