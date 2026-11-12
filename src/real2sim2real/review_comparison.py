"""Equal-clock video comparison exports, independent of a simulator or dataset."""

import os
import tempfile
from pathlib import Path

from .artifacts import digest


def side_by_side(videos, labels, output):
    """Publish a verified two-video comparison without overwriting any input.

    A failed decode or unequal frame count must not leave a plausible partial
    comparison at the requested destination. Encode and verify in a temporary
    sibling directory, then publish with exclusive creation.
    """
    videos, labels, output = tuple(map(Path, videos)), tuple(labels), Path(output)
    if len(videos) != 2 or len(labels) != 2:
        raise ValueError("Exactly two videos and labels required")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    hashes = [digest(path) for path in videos]
    with tempfile.TemporaryDirectory(prefix=".comparison-", dir=output.parent) as tmp:
        staged = Path(tmp) / "comparison.mp4"
        receipt = _encode(videos, labels, staged)
        if hashes != [digest(path) for path in videos]:
            raise ValueError("Comparison sources changed during export")
        receipt.update(file=output.name, input_sha256=hashes)
        os.link(staged, output)
    return receipt


def _encode(videos, labels, output):
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    readers = [
        imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24") for path in videos
    ]
    writer = None
    count = 0
    try:
        metadata = [next(reader) for reader in readers]
        if len(metadata) != 2 or any(
            m["size"] != metadata[0]["size"]
            or abs(m["fps"] - metadata[0]["fps"]) > 1e-9
            for m in metadata
        ):
            raise ValueError("Comparison export needs matching size and frame rate")
        width, height = metadata[0]["size"]
        writer = imageio_ffmpeg.write_frames(
            str(output),
            (width * 2, height + 56),
            fps=metadata[0]["fps"],
            codec="libx264",
            quality=7,
            macro_block_size=1,
            output_params=[
                "-threads",
                "1",
                "-preset",
                "veryfast",
                "-movflags",
                "+faststart",
                "-pix_fmt",
                "yuv420p",
            ],
        )
        writer.send(None)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 23)
        except OSError:
            font = ImageFont.load_default()
        for frames in zip(*readers, strict=True):
            image = Image.new("RGB", (width * 2, height + 56), "white")
            draw = ImageDraw.Draw(image)
            for i, (frame, label) in enumerate(zip(frames, labels, strict=True)):
                image.paste(
                    Image.frombytes("RGB", (width, height), frame), (i * width, 56)
                )
                draw.text((i * width + 12, 14), label, font=font, fill="black")
            writer.send(np.asarray(image))
            count += 1
    finally:
        if writer is not None:
            writer.close()
        for reader in readers:
            reader.close()
    decoded, _ = imageio_ffmpeg.count_frames_and_secs(str(output))
    if not count or decoded != count:
        raise ValueError("Combined video failed decoded-frame verification")
    return {
        "file": Path(output).name,
        "sha256": digest(output),
        "decoded_frames": count,
        "fps": metadata[0]["fps"],
        "time_resampling": False,
    }
