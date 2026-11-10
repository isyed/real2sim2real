import numpy as np
import pytest

from real2sim2real.review_media import prepare, player


def test_portable_video_preserves_frames_and_rejects_stale_cache(tmp_path):
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    source = tmp_path / "original.mp4"
    writer = ffmpeg.write_frames(str(source), (64, 48), fps=10)
    writer.send(None)
    for value in [0, 80, 160, 240]:
        writer.send(np.full((48, 64, 3), value, np.uint8))
    writer.close()
    original = source.read_bytes()
    result = prepare(source, tmp_path / "media")
    assert result["files"]["webm"]["decoded_frames"] == 4
    assert result["files"]["gif"]["decoded_frames"] == 4
    assert source.read_bytes() == original
    assert prepare(source, tmp_path / "media") == result
    (tmp_path / "media/video.webm").write_bytes(b"changed")
    with pytest.raises(ValueError, match="media changed"):
        prepare(source, tmp_path / "media")


def test_player_embeds_fallback_and_escapes_resource_names(tmp_path):
    gif = tmp_path / "small.gif"
    gif.write_bytes(b"GIF89a")
    text = player('clip".mp4', "clip.webm", "clip.gif", embedded_gif=gif)
    assert "data:image/gif;base64,R0lGODlh" in text
    assert "clip&quot;.mp4" in text
    assert 'type="video/webm"' in text
    assert "<details>" in text
