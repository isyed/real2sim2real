"""Video comparisons must preserve frame count and reject clock mismatches."""

import numpy as np
import pytest
from real2sim2real.review_comparison import side_by_side


def clip(path, fps, values):
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("PIL")
    writer = ffmpeg.write_frames(
        str(path), (64, 48), fps=fps, output_params=["-threads", "1"]
    )
    writer.send(None)
    for value in values:
        writer.send(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.close()


def test_combined_video_keeps_equal_native_frame_count_and_sources(tmp_path):
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for p in videos:
        clip(p, 10, [0, 80, 160, 240])
    originals = [p.read_bytes() for p in videos]
    receipt = side_by_side(videos, ["Frozen", "Candidate"], tmp_path / "combined.mp4")
    assert receipt["decoded_frames"] == 4
    assert receipt["fps"] == 10
    assert receipt["time_resampling"] is False
    assert [p.read_bytes() for p in videos] == originals


def test_combined_video_refuses_to_hide_different_video_clocks(tmp_path):
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    clip(videos[0], 10, [0, 80])
    clip(videos[1], 20, [0, 80])
    with pytest.raises(ValueError, match="frame rate"):
        side_by_side(videos, ["Frozen", "Candidate"], tmp_path / "combined.mp4")
    assert not (tmp_path / "combined.mp4").exists()


def test_different_frame_counts_leave_no_partial_comparison(tmp_path):
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    clip(videos[0], 10, [0, 80])
    clip(videos[1], 10, [0, 80, 160])
    with pytest.raises(ValueError, match="zip"):
        side_by_side(videos, ["Frozen", "Candidate"], tmp_path / "combined.mp4")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.mp4", "b.mp4"]


def test_comparison_cannot_overwrite_a_source_or_existing_result(tmp_path):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        side_by_side([source, source], ["A", "B"], source)
    assert source.read_bytes() == b"preserve"


@pytest.mark.parametrize("counts", [(1, 2), (2, 1), (3, 3)])
def test_comparison_requires_exactly_two_sources_and_labels(tmp_path, counts):
    with pytest.raises(ValueError, match="Exactly two"):
        side_by_side(
            [tmp_path / "missing"] * counts[0], ["A"] * counts[1], tmp_path / "out.mp4"
        )
