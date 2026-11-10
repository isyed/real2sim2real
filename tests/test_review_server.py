import pytest

from real2sim2real.review_server import byte_range, resolve_media


@pytest.mark.parametrize(
    "header,expected",
    [
        (None, None),
        ("bytes=0-3", (0, 3)),
        ("bytes=3-", (3, 9)),
        ("bytes=-4", (6, 9)),
        ("bytes=5-99", (5, 9)),
        ("bytes=-99", (0, 9)),
    ],
)
def test_video_ranges(header, expected):
    assert byte_range(header, 10) == expected


@pytest.mark.parametrize(
    "header",
    ["bytes=10-", "bytes=6-2", "bytes=-0", "bytes=0-1,3-5", "items=0-2", "bytes=bad"],
)
def test_invalid_ranges_fail(header):
    with pytest.raises(ValueError):
        byte_range(header, 10)


def test_report_paths_allow_media_and_reject_escape_or_private_assets(tmp_path):
    root = tmp_path / "report"
    root.mkdir()
    (root / "review.html").write_text("review")
    (root / "clip.webm").write_bytes(b"video")
    (tmp_path / "private.json").write_text("{}")
    (root / "outside.json").symlink_to(tmp_path / "private.json")
    (root / "linked-index").mkdir()
    (root / "linked-index/review.html").symlink_to(tmp_path / "private.json")
    (root / "model.pkl").write_bytes(b"asset")
    assert resolve_media(root, "/") == root / "review.html"
    assert resolve_media(root, "/clip.webm?cache=1") == root / "clip.webm"
    for url in [
        "/%2e%2e/private.json",
        "/.git/config",
        "/outside.json",
        "/model.pkl",
        "/linked-index/",
    ]:
        with pytest.raises((ValueError, FileNotFoundError)):
            resolve_media(root, url)
