"""Portable local report playback with WebM and an animated-image fallback."""

import argparse
import base64
import html
import json
import subprocess
from pathlib import Path

from .artifacts import digest, write_json


def prepare(source, output, *, timeout_s=120):
    """Transcode retained frames, preserving the original and recording hashes."""
    import imageio_ffmpeg

    source, output = Path(source), Path(output)
    identity = {"source_sha256": digest(source), "webm_codec": "vp8", "preview_fps": 10}
    receipt = output / "media.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["identity"] != identity:
            raise ValueError("Existing media belongs to another source")
        for row in saved["files"].values():
            if digest(output / row["path"]) != row["sha256"]:
                raise ValueError("Saved review media changed")
        return saved
    output.mkdir(parents=True, exist_ok=False)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    commands = [
        [
            "-an",
            "-c:v",
            "libvpx",
            "-b:v",
            "1100k",
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-pix_fmt",
            "yuv420p",
            str(output / "video.webm"),
        ],
        [
            "-an",
            "-filter_complex",
            "fps=10,scale=720:-2:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
            "-loop",
            "0",
            str(output / "preview.gif"),
        ],
    ]
    for command in commands:
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-n",
                "-threads",
                "1",
                "-i",
                str(source),
                "-threads",
                "1",
                "-filter_complex_threads",
                "1",
                *command,
            ],
            check=True,
            timeout=timeout_s,
            capture_output=True,
        )
    files = {}
    for kind, name in [("webm", "video.webm"), ("gif", "preview.gif")]:
        reader = imageio_ffmpeg.read_frames(str(output / name))
        metadata = next(reader)
        count = sum(1 for _ in reader)
        reader.close()
        if not count:
            raise ValueError("Empty review media")
        files[kind] = {
            "path": name,
            "sha256": digest(output / name),
            "decoded_frames": count,
            "fps": metadata["fps"],
            "size": list(metadata["size"]),
        }
    reader = imageio_ffmpeg.read_frames(str(source))
    next(reader)
    original_count = sum(1 for _ in reader)
    reader.close()
    if files["webm"]["decoded_frames"] != original_count:
        raise ValueError("Transcoded video lost frames")
    result = {
        "schema": "real2sim2real.review-media/1",
        "identity": identity,
        "files": files,
        "original_frames": original_count,
        "physics_rerun": False,
    }
    write_json(receipt, result)
    return result


def player(
    mp4_url,
    webm_url,
    gif_url,
    *,
    poster_url=None,
    embedded_gif=None,
    embedded_webm=None,
):
    """No JavaScript dependency; an embedded GIF also survives local URL restrictions."""
    escape = lambda value: html.escape(str(value), quote=True)
    poster = "" if poster_url is None else f' poster="{escape(poster_url)}"'
    fallback = str(gif_url)
    if embedded_gif is not None:
        fallback = "data:image/gif;base64," + base64.b64encode(
            Path(embedded_gif).read_bytes()
        ).decode("ascii")
    direct_source = ""
    if embedded_webm is not None:
        video_data = base64.b64encode(Path(embedded_webm).read_bytes()).decode("ascii")
        direct_source = f' src="data:video/webm;base64,{video_data}"'
    return (
        f'<section class="review-player"><video controls muted loop playsinline preload="metadata"{poster}{direct_source}>'
        f'<source src="{escape(webm_url)}" type="video/webm">'
        f'<source src="{escape(mp4_url)}" type="video/mp4">'
        "</video><p>Open video: "
        f'<a href="{escape(webm_url)}">WebM</a> · <a href="{escape(mp4_url)}">MP4</a> · '
        f'<a href="{escape(gif_url)}">Animated preview</a></p>'
        "<details><summary>Video not playing? Show the animated preview</summary>"
        f'<img loading="lazy" src="{escape(fallback)}" alt="Animated preview of the same saved simulation frames">'
        "<p>This preview uses fewer display frames; it does not rerun physics.</p></details></section>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)


def comparison_video(left, right, output, *, timeout_s=180):
    """Join two saved videos on one clock for script-restricted HTML previews.

    Inputs must have identical dimensions, frame rate and decoded frame count.
    No implicit resampling, padding or physics replay is permitted. Labels and
    stopped-state banners must already be present in the source videos.
    """
    import imageio_ffmpeg

    left, right, output = map(Path, (left, right, output))
    if output.exists():
        raise FileExistsError(output)
    descriptions = []
    for path in (left, right):
        reader = imageio_ffmpeg.read_frames(str(path))
        try:
            metadata = next(reader)
            descriptions.append(
                (metadata["size"], metadata["fps"], sum(1 for _ in reader))
            )
        finally:
            reader.close()
    if descriptions[0] != descriptions[1] or not descriptions[0][2]:
        raise ValueError("Comparison requires matching size, fps and frame count")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-threads",
            "1",
            "-i",
            str(left),
            "-threads",
            "1",
            "-i",
            str(right),
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(descriptions[0][1]),
            "-fps_mode",
            "cfr",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
        timeout=timeout_s,
        capture_output=True,
    )
    reader = imageio_ffmpeg.read_frames(str(output))
    try:
        metadata = next(reader)
        count = sum(1 for _ in reader)
    finally:
        reader.close()
    size, fps, expected_count = descriptions[0]
    if (
        count != expected_count
        or metadata["fps"] != fps
        or metadata["size"] != (size[0] * 2, size[1])
    ):
        raise ValueError("Comparison output changed the source timeline or dimensions")
    receipt = {
        "schema": "real2sim2real.comparison-media/1",
        "physics_rerun": False,
        "sources": [{"path": str(p), "sha256": digest(p)} for p in (left, right)],
        "output_sha256": digest(output),
        "decoded_frames": count,
        "fps": fps,
        "size": list(metadata["size"]),
        "shared_video_clock": True,
    }
    write_json(output.with_suffix(".json"), receipt)
    return receipt


if __name__ == "__main__":
    main()
