"""Choose one native source frame for mutually aligned RGB, mask and depth."""

from pathlib import Path


def select_mesh_frame(value, frame_count):
    if (isinstance(value, bool) or not isinstance(value, int) or
            isinstance(frame_count, bool) or not isinstance(frame_count, int) or
            frame_count < 1 or not 0 <= value < frame_count):
        raise ValueError("mesh_frame_index must identify a source video frame")
    return value


def mask_path(output, frame_index):
    return Path(output) / "masks/1" / f"{frame_index:06d}.png"


def extract_rgb(video, frame_index, destination):
    """Decode the indexed source frame; never renumber a cropped subclip."""
    import av

    with av.open(str(video)) as capture:
        for index, frame in enumerate(capture.decode(video=0)):
            if index == frame_index:
                image = frame.to_image().convert("RGB")
                image.save(destination)
                return image.size
    raise ValueError("Selected mesh frame is absent from the source video")
