"""Opt-in, bounded temporal conditioning of a prerecorded joint reference.

This is an offline motion-conditioning hypothesis, not contact inference or a
reconstruction repair. It never changes the source artifact, reference clock,
root, object, unselected joints, or motor commands directly. The caller must
recompute encoder velocities from the returned positions. Symmetric filtering
uses the already available future reference, not future physical observations.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ReferenceDenoisingConfig:
    joint_names: tuple[str, ...]
    sigma_seconds: float = 0.06
    strength: float = 0.5
    ramp_seconds: float = 0.5
    fade_start_seconds: float = 8.0
    end_seconds: float = 10.0
    max_change_rad: float = 0.03

    def __post_init__(self):
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        if (not self.joint_names or any(not isinstance(n, str) or not n for n in self.joint_names)
                or len(set(self.joint_names)) != len(self.joint_names)):
            raise ValueError("Explicit, unique joint names required")
        for name in ("sigma_seconds", "strength", "ramp_seconds", "fade_start_seconds",
                     "end_seconds", "max_change_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.ramp_seconds <= self.fade_start_seconds < self.end_seconds:
            raise ValueError("Require ramp <= fade start < end")
        if self.strength > 1 or self.sigma_seconds > 0.2 or self.max_change_rad > 0.1:
            raise ValueError("Conditioning exceeds bounded experimental limits")


def condition_reference(positions, joint_names, *, fps, config):
    """Return an independent array; refuse excessive changes instead of clipping.

    Frame zero and samples at/after the end remain exact. The symmetric filter
    is blended through smoothstep windows; it does not shift the playback clock.
    Joint angles must already be continuous (no wrapped angle discontinuities).
    """
    from scipy.ndimage import gaussian_filter1d

    x = np.asarray(positions)
    names = tuple(joint_names)
    if (x.ndim != 2 or len(x) < 2 or x.shape[1] != len(names)
            or len(set(names)) != len(names) or not np.issubdtype(x.dtype, np.floating)
            or not np.isfinite(x).all() or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Finite floating joint positions and explicit positive FPS required")
    if not isinstance(config, ReferenceDenoisingConfig):
        raise ValueError("Explicit ReferenceDenoisingConfig required")
    if any(n not in names for n in config.joint_names):
        raise ValueError("Selected joint missing from reference")
    ids = [names.index(n) for n in config.joint_names]
    t = np.arange(len(x), dtype=np.float64) / fps

    def smoothstep(a):
        a = np.clip(a, 0, 1)
        return a * a * (3 - 2 * a)

    weights = smoothstep(t / config.ramp_seconds) * smoothstep(
        (config.end_seconds - t) / (config.end_seconds - config.fade_start_seconds))
    filtered = gaussian_filter1d(x[:, ids].astype(np.float64),
                                config.sigma_seconds * fps, axis=0, mode="nearest")
    delta = config.strength * weights[:, None] * (filtered - x[:, ids])
    if np.max(np.abs(delta)) > config.max_change_rad:
        raise ValueError("Proposed conditioning exceeds max_change_rad; preserve input")
    out = x.copy()
    changed = np.flatnonzero(weights > 0)
    out[np.ix_(changed, ids)] = (x[np.ix_(changed, ids)] + delta[changed]).astype(x.dtype)
    return out
