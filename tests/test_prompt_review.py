import json
from PIL import Image
import pytest
from real2sim2real.reconstruction.prompt_review import review_first_frame


def run(tmp_path, prompt):
    image = tmp_path / 'rgb.png'
    Image.new('RGB', (100, 80), 'black').save(image)
    prompts = tmp_path / 'prompts.json'
    prompts.write_text(json.dumps({'prompts': [prompt]}))
    return review_first_frame(image, prompts, tmp_path / 'review', frame_count=3,
        source_video_sha256='a' * 64)


def test_pixel_bounds_are_not_semantic_approval(tmp_path):
    report = run(tmp_path, dict(frame_index=0, object_id=0,
        box=dict(x0=10, y0=5, x1=50, y1=70)))
    assert report['coordinates_in_bounds'] and not report['semantics_verified']
    assert report['native_size'] == [100, 80]
    assert (tmp_path / 'review/first_frame_prompts.png').exists()
    assert report['prompts_sha256'] and report['source_video_sha256'] == 'a' * 64


@pytest.mark.parametrize('box', [dict(x0=-1, y0=0, x1=20, y1=30),
    dict(x0=0, y0=0, x1=101, y1=30), dict(x0=20, y0=5, x1=20, y1=30),
    dict(x0=0, y0=0, x1=float('nan'), y1=30)])
def test_bad_native_box_is_rejected_before_model(tmp_path, box):
    with pytest.raises(ValueError):
        run(tmp_path, dict(frame_index=0, object_id=1, box=box))
    assert not (tmp_path / 'review/prompt_review.json').exists()


def test_later_frame_points_are_marked_unpreviewed(tmp_path):
    report = run(tmp_path, dict(frame_index=2, object_id=4,
        points=[dict(x=50, y=40)], point_labels=[1]))
    assert report['unpreviewed_prompt_frames'] == [2]
    assert not report['semantics_verified']


def test_frame_outside_timeline_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, dict(frame_index=3, object_id=0,
            box=dict(x0=0, y0=0, x1=50, y1=70)))
