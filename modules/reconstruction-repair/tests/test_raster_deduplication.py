import numpy as np
from PIL import Image,ImageDraw
from real2sim2real.reconstruction_repair.geometry import _unique_raster_triangles


def legacy(pixels,faces):
    return np.unique(np.trunc(pixels[faces]).astype(np.int64).reshape(-1,6),axis=0).reshape(-1,3,2)


def triangle_set(a):return {tuple(x) for x in a.reshape(-1,6)}


def test_packed_keys_preserve_every_ordered_polygon_with_aliases_and_negative_coordinates():
    rng=np.random.default_rng(13);pixels=rng.uniform(-15,60,(1500,2));faces=rng.integers(0,len(pixels),(26000,3));faces=np.r_[faces,faces[:2000]];before=pixels.copy()
    assert triangle_set(_unique_raster_triangles(pixels,faces))==triangle_set(legacy(pixels,faces))
    assert np.array_equal(pixels,before)


def test_pixel_union_is_exact_including_collinear_and_repeated_vertex_polygons():
    pixels=np.array([[-2.,0.],[0.,0.],[1.,1.],[2.,2.],[6.,0.],[0.,6.],[3.9,3.1]])
    faces=np.array([[a,b,c] for a in range(len(pixels)) for b in range(len(pixels)) for c in range(len(pixels))])
    def raster(triangles):
        im=Image.new('1',(8,8));draw=ImageDraw.Draw(im)
        for t in triangles:draw.polygon([tuple(x) for x in t],fill=1)
        return np.asarray(im)
    assert np.array_equal(raster(_unique_raster_triangles(pixels,faces)),raster(legacy(pixels,faces)))


def test_coordinate_range_overflow_uses_exact_legacy_fallback():
    pixels=np.array([[-1e10,-1e10],[1e10,0],[0,1e10],[1.,2.]])
    faces=np.array([[0,1,2],[1,2,3],[0,1,2]])
    assert np.array_equal(_unique_raster_triangles(pixels,faces),legacy(pixels,faces))
