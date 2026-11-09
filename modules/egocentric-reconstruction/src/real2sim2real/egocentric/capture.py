"""Dataset-independent, hash-bound synchronized capture manifest."""

from pathlib import Path
import json
import numpy as np
from real2sim2real.artifacts import digest
from .camera import intrinsic, rigid


def load(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("schema") != "real2sim2real.calibrated-capture/1":
        raise ValueError("Unsupported capture schema")
    if data.get("units") != "metres" or data.get("camera_convention") != "opencv":
        raise ValueError("Explicit metres/OpenCV convention required")
    times = np.asarray(data["timestamps_s"], dtype=np.float64)
    ids = np.asarray(data["frame_ids"])
    if (
        len(times) < 2
        or ids.shape != times.shape
        or not np.isfinite(times).all()
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError("A synchronized increasing timeline is required")
    if ids.dtype.kind not in "iu" or (np.diff(ids) != 1).any():
        raise ValueError("Capture source frame IDs must be consecutive integers")
    for name, stream in data["streams"].items():
        video = (path.parent / stream["video"]).resolve()
        if digest(video) != stream["sha256"]:
            raise ValueError("Capture source changed: " + name)
        intrinsic(stream["intrinsics"])
        rigid(stream["rig_from_camera"])
        stream["video"] = str(video)
        if not np.isfinite(stream["distortion"]).all():
            raise ValueError("Invalid distortion")
    for name in (data["target_stream"], *data["stereo_pair"]):
        if name not in data["streams"]:
            raise ValueError("Missing declared camera")
    data["manifest_sha256"] = digest(path)
    return data


def frame(stream, index):
    import av

    with av.open(stream["video"]) as container:
        for i, decoded in enumerate(container.decode(video=0)):
            if i == index:
                image = decoded.to_ndarray(format="rgb24")
                if list(image.shape[1::-1]) != stream["size"]:
                    raise ValueError("Video resolution disagrees with calibration")
                return image
    raise IndexError("Video does not contain requested source frame")


def frames(stream, count):
    import av

    with av.open(stream["video"]) as container:
        for i, decoded in enumerate(container.decode(video=0)):
            if i >= count:
                break
            image = decoded.to_ndarray(format="rgb24")
            if list(image.shape[1::-1]) != stream["size"]:
                raise ValueError("Video resolution disagrees with calibration")
            yield image
