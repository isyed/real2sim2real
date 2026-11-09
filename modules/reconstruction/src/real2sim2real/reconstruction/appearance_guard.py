"""Optional observed-colour evidence for otherwise ambiguous rigid tracking.

The feature definition and evidence belong to the caller. A feature is checked
only on the frontmost rendered triangles, excluding observed human occlusion.
An unobserved/hidden model feature is uncertain, never evidence of a correct yaw.
This adds rejection criteria; it cannot replace geometric or temporal checks.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Feature:
    differences: tuple
    lower: tuple = ()
    upper: tuple = ()
    min_pixels: int = 8
    min_precision: float = .60
    tolerance_px: int = 3

    def validate(self):
        if not self.differences and not self.lower and not self.upper:
            raise ValueError("Explicit colour feature required")
        for a, b, margin in self.differences:
            if type(a) is not int or type(b) is not int or a not in range(3) or b not in range(3) or a == b or not np.isfinite(margin) or not 0 <= margin < 255:
                raise ValueError("Channel differences require indices0–2 and [0,255) margins")
        for channel, value in (*self.lower, *self.upper):
            if type(channel) is not int or channel not in range(3) or not np.isfinite(value) or not 0 <= value <= 255:
                raise ValueError("Channel bounds require indices0–2 and [0,255] intensities")
        if (type(self.min_pixels) is not int or self.min_pixels < 1 or
                not np.isfinite(self.min_precision) or not 0 < self.min_precision <= 1 or
                type(self.tolerance_px) is not int or not 0 <= self.tolerance_px <= 10):
            raise ValueError("Finite explicit feature acceptance limits required")
        return self


def feature_mask(rgb, feature):
    """Strict inequalities on explicit uint8 RGB, without uint8 subtraction."""
    feature.validate()
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim < 3 or rgb.shape[-1] != 3:
        raise ValueError("Explicit uint8 RGB required")
    a = rgb.astype(np.int16)
    out = np.ones(a.shape[:-1], bool)
    for x, y, value in feature.differences: out &= a[..., x]-a[..., y] > value
    for channel, value in feature.lower: out &= a[..., channel] > value
    for channel, value in feature.upper: out &= a[..., channel] < value
    return out


def assess_feature(predicted, rgb, object_mask, human_mask, feature):
    from scipy.ndimage import binary_dilation
    feature.validate()
    predicted, obj, human = map(np.asarray, (predicted, object_mask, human_mask))
    if (predicted.dtype != bool or obj.dtype != bool or human.dtype != bool or
            predicted.shape != obj.shape or human.shape != obj.shape or obj.ndim != 2):
        raise ValueError("Matching native boolean feature/object/human masks required")
    observed = feature_mask(rgb, feature)
    if observed.shape != obj.shape: raise ValueError("Native RGB/mask dimensions differ")
    observed &= obj & ~human
    predicted = predicted & ~human
    count = int(predicted.sum())
    # scipy iterations=0 means repeat until convergence, NOT no dilation.
    supported = binary_dilation(observed, iterations=feature.tolerance_px) if feature.tolerance_px else observed
    precision = float(np.count_nonzero(predicted & supported)/count) if count >= feature.min_pixels else None
    return dict(predicted_pixels=count, observed_pixels=int(observed.sum()),
        precision=precision, uncertain_hidden_feature=precision is None,
        rejected=precision is not None and precision < feature.min_precision,
        accepted_yaw=False)


def render_masks(upstream, estimator, poses_centered, K, shape, feature, *, chunk_size=8):
    """Same native triangle raster as geometry; interpolate unlit vertex RGB.

    Heavy dependencies are isolated to the explicitly requested GPU stage.
    Only boolean masks leave the GPU; no full RGB batch is retained on CPU.
    """
    import torch
    feature.validate()
    poses = upstream.validate_rigid_transforms(poses_centered, "appearance-checked candidates")
    height, width = map(int, shape)
    if height <= 0 or width <= 0 or type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("Positive native render dimensions and chunk size required")
    mesh = estimator.mesh_tensors
    if "vertex_color" not in mesh:
        raise ValueError("Observed-colour guard requires preserved vertex RGB")
    device = mesh["pos"].device
    projection = torch.as_tensor(upstream.Utils.projection_matrix_from_intrinsics(K,
        height=height, width=width, znear=.001, zfar=100.), dtype=torch.float32, device=device)
    conversion = torch.as_tensor(upstream.Utils.glcam_in_cvcam, dtype=torch.float32, device=device)
    points = upstream.Utils.to_homo_torch(mesh["pos"])
    geometry, features = [], []
    with torch.no_grad():
        for begin in range(0, len(poses), chunk_size):
            pose = torch.as_tensor(poses[begin:begin+chunk_size], dtype=torch.float32, device=device)
            clip = ((projection[None] @ conversion[None] @ pose)[:, None] @ points[None, ..., None])[..., 0]
            raster, _ = upstream.dr.rasterize(estimator.glctx, clip, mesh["faces"], resolution=(height, width))
            colors, _ = upstream.dr.interpolate(mesh["vertex_color"].contiguous(), raster, mesh["faces"])
            colors = colors*255.
            visible = raster[..., 3] > 0
            selected = visible.clone()
            for a, b, margin in feature.differences: selected &= colors[..., a]-colors[..., b] > margin
            for channel, value in feature.lower: selected &= colors[..., channel] > value
            for channel, value in feature.upper: selected &= colors[..., channel] < value
            geometry.append(torch.flip(visible, [1]).cpu().numpy())
            features.append(torch.flip(selected, [1]).cpu().numpy())
    return np.concatenate(geometry), np.concatenate(features)


def angular_proposals(poses, vertices, axis_camera):
    """Small appearance-recovery proposals, preserving the mesh's world centre.

    These are not accepted states. Every proposal needs the normal independent
    native image/depth/support and previous-state motion checks.
    """
    from scipy.spatial.transform import Rotation
    from .recovery_quality import rigid
    axis = np.asarray(axis_camera, float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1., atol=1e-5):
        raise ValueError("Explicit unit camera-frame rotation axis required")
    center = np.asarray(vertices, float).mean(0)
    out = []
    for pose in poses:
        if not rigid(pose): raise ValueError("Rigid appearance proposal base required")
        world_center = pose[:3, :3] @ center+pose[:3, 3]
        for angle in (-15., -10., -5., -2.5, 0., 2.5, 5., 10., 15.):
            candidate = np.array(pose, copy=True)
            candidate[:3, :3] = Rotation.from_rotvec(np.deg2rad(angle)*axis).as_matrix() @ pose[:3, :3]
            candidate[:3, 3] = world_center-candidate[:3, :3] @ center
            out.append(candidate)
    return np.asarray(out)
