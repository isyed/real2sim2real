"""Dense initialization performance must preserve existing geometric decisions."""
import numpy as np
from PIL import Image, ImageDraw

from real2sim2real.reconstruction_repair.geometry import points_mesh_distance, mesh_silhouette


def test_dense_distance_matches_exhaustive_including_thin_and_degenerate_faces():
    rng=np.random.default_rng(821)
    tri=rng.normal(size=(2200,3,3))
    tri[:400]*=1e-5
    tri[400:800,2]=tri[400:800,1]
    tri[800:1000,2]=tri[800:1000,0]+.999999*(tri[800:1000,1]-tri[800:1000,0])
    vertices=tri.reshape(-1,3);faces=np.arange(len(vertices)).reshape(-1,3)
    points=np.r_[rng.normal(size=(20,3))*3,[[0,0,0],[1e-6,2e-6,3e-6]]]
    expected=np.minimum.reduce([points_mesh_distance(points,vertices,f) for f in np.array_split(faces,4)])
    np.testing.assert_allclose(points_mesh_distance(points,vertices,faces),expected,atol=1e-12,rtol=1e-12)


def test_dense_integer_polygon_union_matches_pillow_floats():
    rng=np.random.default_rng(822)
    # Include negative/subpixel coordinates and thin triangles, not just boxes.
    centers=np.array([[-1.2,3.6],[4.8,8.1],[21.7,23.4],[31.8,29.9]])
    pixels=np.repeat(centers,5251,axis=0)[:,None,:]+rng.uniform(-.8,.8,(21004,3,2))
    points=np.concatenate([pixels,np.ones((len(pixels),3,1))],axis=2).reshape(-1,3)
    faces=np.arange(len(points)).reshape(-1,3)
    g=dict(vertices=points,faces=faces,object_scale=np.array(1.),object_rotation=np.eye(3)[None],
           object_translation=np.zeros((1,3)),intrinsics=np.eye(3)[None],world_to_camera=np.eye(4)[None])
    # Small chunks exercise the unoptimized float renderer, including its
    # original screen/near-plane culling (a separate behavior from Pillow).
    expected=np.logical_or.reduce([mesh_silhouette({**g,'faces':part},0,32,32)
                                   for part in np.array_split(faces,4)])
    np.testing.assert_array_equal(mesh_silhouette(g,0,32,32),expected)
