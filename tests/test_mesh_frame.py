from pathlib import Path

import pytest

from real2sim2real.reconstruction.mesh_frame import mask_path, select_mesh_frame


def test_mesh_frame_uses_source_index():
    assert select_mesh_frame(180, 988) == 180
    assert mask_path(Path('/output'), 180) == Path('/output/masks/1/000180.png')


@pytest.mark.parametrize('value,count', [(-1, 988), (988, 988),
    (True, 988), (1.0, 988), ('180', 988), (0, 0)])
def test_mesh_frame_rejects_invalid_index(value, count):
    with pytest.raises(ValueError, match='mesh_frame_index'):
        select_mesh_frame(value, count)
