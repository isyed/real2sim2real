"""Choose calibration evidence from visibility, before fitting scale or poses.

One eligible frame per fixed temporal bin prevents a visible fragment of a clip
from standing in for the entire sequence. Alternating bins define fit/held-out
splits. Eligibility never uses a mesh, pose, fitted scale or tracking score.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.ndimage import binary_erosion


@dataclass(frozen=True)
class VisibilitySamplingConfig:
    min_visible_fraction: float = .80
    min_depth_coverage: float = .90
    min_depth_pixels: int = 100
    erosion_pixels: int = 2
    max_relative_depth_iqr: float = .35

    def validate(self):
        if not 0 < self.min_visible_fraction <= 1 or not 0 < self.min_depth_coverage <= 1:
            raise ValueError('Coverage fractions must be in (0, 1]')
        for value in (self.min_depth_pixels, self.erosion_pixels):
            if not np.isfinite(value) or value < 1 or int(value) != value:
                raise ValueError('Positive integer pixel bounds required')
        if not np.isfinite(self.max_relative_depth_iqr) or self.max_relative_depth_iqr <= 0:
            raise ValueError('Positive finite depth spread required')


def visibility(object_mask, human_mask, depth_m, config=VisibilitySamplingConfig()):
    """Input-only eligibility; coherent extent/floor/scale remain later gates."""
    config.validate()
    obj, human, depth = np.asarray(object_mask, bool), np.asarray(human_mask, bool), np.asarray(depth_m)
    if obj.ndim != 2 or obj.shape != human.shape or obj.shape != depth.shape:
        raise ValueError('Native masks and depth must share one image grid')
    exposed = obj & ~human
    inner = binary_erosion(exposed, iterations=config.erosion_pixels)
    valid = inner & np.isfinite(depth) & (depth > 0) & (depth < 65.535)
    pixels = int(valid.sum())
    fraction = float(exposed.sum() / max(1, obj.sum()))
    coverage = float(pixels / max(1, inner.sum()))
    clipped = bool(obj[0].any() or obj[-1].any() or obj[:, 0].any() or obj[:, -1].any())
    reasons = []
    if fraction < config.min_visible_fraction: reasons.append('human_occlusion')
    if pixels < config.min_depth_pixels: reasons.append('insufficient_depth_pixels')
    if coverage < config.min_depth_coverage: reasons.append('depth_coverage')
    if clipped: reasons.append('image_boundary_clipping')
    relative_iqr = None
    if pixels >= config.min_depth_pixels:
        q = np.percentile(depth[valid], [25, 50, 75])
        relative_iqr = float((q[2] - q[0]) / q[1])
        if relative_iqr > config.max_relative_depth_iqr: reasons.append('depth_incoherent')
    return dict(eligible=not reasons, reasons=reasons, visible_fraction=fraction,
                visible_depth_pixels=pixels, depth_coverage=coverage, boundary_clipped=clipped,
                relative_depth_iqr=relative_iqr)


def select(frame_count, inspect, *, samples=12, config=VisibilitySamplingConfig()):
    """Search nearest each bin's midpoint; record every inspected input frame.

    ``inspect(index, config)`` returns ``visibility`` measurements. Missing bins
    fail explicitly; no retry with altered thresholds or fewer time bins.
    """
    config.validate()
    if frame_count < samples or samples < 6 or samples % 2:
        raise ValueError('Even sample count >= 6, with at least that many source frames required')
    bins = np.array_split(np.arange(frame_count), samples)
    selected, audit, missing = [], [], []
    for bin_id, indices in enumerate(bins):
        midpoint = (float(indices[0]) + float(indices[-1])) / 2
        order = sorted(indices.tolist(), key=lambda i: (abs(i - midpoint), i))
        found = None
        for index in order:
            row = inspect(index, config)
            audit.append(dict(index=index, bin=bin_id, **row))
            if row['eligible']:
                found = index
                selected.append(index)
                break
        if found is None: missing.append(bin_id)
    return dict(schema='real2sim2real.calibration-sampling/1', policy='visibility_bins',
                passed=not missing, selected_indices=selected, missing_bins=missing,
                bins=[dict(first=int(x[0]), last=int(x[-1]), split='fit' if i % 2 == 0 else 'held_out')
                      for i, x in enumerate(bins)], inspected=audit, config=asdict(config),
                uses_pose_scale_or_tracking_scores=False,
                limitation='Visibility does not establish depth accuracy or unforeshortened metric extent.')
