from types import SimpleNamespace
import pytest
from real2sim2real.reconstruction.daid_render import bin_parameters,binned_renderer


def test_capacity_can_hold_all_faces_even_if_projected_into_one_bin():
    result=bin_parameters(1536,1152,326432)
    assert result['bin_size']==128
    assert result['max_faces_per_bin']>326432
    assert result['index_buffer_bytes']==12*9*326433*4
    assert result['index_buffer_bytes']<=2500000000


def test_low_budget_uses_larger_bins_or_falls_back():
    result=bin_parameters(1536,1152,326432,max_index_bytes=100000000)
    assert result['bin_size']==256
    assert result['max_faces_per_bin']==326433
    assert bin_parameters(1536,1152,326432,max_index_bytes=1) is None


def test_native_kernel_bin_count_limit_precedes_buffer_budget():
    result=bin_parameters(1536,1152,326432,max_bins_per_axis=21)
    assert (1536+result['bin_size']-1)//result['bin_size']<=21
    assert bin_parameters(1536,1152,326432,max_bins_per_axis=1) is None


@pytest.mark.parametrize('sizes',[(0,100,10),(100,-1,10),(100,100,0),(100.,100,10)])
def test_invalid_input_rejected(sizes):
    with pytest.raises(ValueError):bin_parameters(*sizes)


def test_only_binning_changes_camera_shader_and_other_settings_retained():
    camera,shader,geometry=object(),object(),object()
    settings=SimpleNamespace(image_size=(1152,1536),blur_radius=0.,faces_per_pixel=1,bin_size=0,max_faces_per_bin=50000)
    native=SimpleNamespace(rasterizer=SimpleNamespace(raster_settings=settings,cameras=camera),shader=shader,geometry=geometry)
    args=[]
    def build(*a):args.append(a);return native
    tracker=SimpleNamespace(_build_silhouette_renderer=build)
    K=object();result=binned_renderer(tracker,K,1536,1152,'cpu',326432)
    assert result is native and result.shader is shader and result.rasterizer.cameras is camera
    assert args==[(K,1536,1152,'cpu')]
    assert settings.image_size==(1152,1536) and settings.blur_radius==0 and settings.faces_per_pixel==1
