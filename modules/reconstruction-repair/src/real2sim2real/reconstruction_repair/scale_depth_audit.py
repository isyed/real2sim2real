"""Finite scale-identifiability diagnostic; never an automatic mesh repair.

Scaling all camera-space object points preserves their perspective projection.
It scales both mesh dimensions and pose translations, not the human or scene.
Thus image overlap alone cannot choose metric scale. Callers provide paired
rendered/observed depths at the SAME pixels for each frame and depth hypothesis.
An agreeing scale is a hypothesis, not independently measured geometry.
"""
import numpy as np


def audit(samples, scales, *, error_limit_m=.10, min_points=100):
    """Compare every declared hypothesis on one explicit finite scale grid.

    No fitting, input mutation, depth-map replacement or candidate output. The
    result is limited to this grid, these fixed orientations and surface pairs;
    it is not a proof that all possible shape/pose repairs fail. Correlated
    prediction sources must remain identified in ``provenance`` by the caller.
    """
    grid = np.asarray(scales, float)
    if (grid.ndim != 1 or not 2 <= len(grid) <= 1001
            or not np.isfinite(grid).all() or np.any(grid <= 0)
            or np.any(np.diff(grid) <= 0) or not np.any(grid == 1.)):
        raise ValueError('Ordered positive finite grid of 2..1001 scales including 1 required')
    if (isinstance(error_limit_m, bool) or not np.isfinite(error_limit_m)
            or error_limit_m <= 0 or type(min_points) is not int or min_points < 1):
        raise ValueError('Positive explicit metre tolerance and sample minimum required')
    if not samples:
        raise ValueError('At least one declared surface observation required')
    rows, errors, identities = [], [], set()
    for item in samples:
        identity = (item['frame_id'], item['hypothesis'])
        if identity in identities or not isinstance(identity[1], str) or not identity[1]:
            raise ValueError('Unique frame/hypothesis identities required')
        identities.add(identity)
        if not isinstance(item.get('provenance'), dict) or not item['provenance']:
            raise ValueError('Explicit source provenance required; predictions are not truth')
        rendered = np.asarray(item['rendered_depth_m'], float)
        observed = np.asarray(item['observed_depth_m'], float)
        if (rendered.ndim != 1 or observed.shape != rendered.shape
                or len(rendered) < min_points
                or not np.isfinite(rendered).all() or not np.isfinite(observed).all()
                or np.any(rendered <= 0) or np.any(observed <= 0)):
            raise ValueError('Enough finite positive corresponding depth samples required')
        # Bounded memory even for a large source mask; no scale-dependent filtering.
        error = np.array([np.median(np.abs(observed - scale * rendered)) for scale in grid])
        errors.append(error)
        rows.append(dict(frame_id=identity[0], hypothesis=identity[1],
                         points=len(rendered), provenance=item['provenance'],
                         error_m=error.tolist(),
                         passing_scales=grid[error <= error_limit_m].tolist()))
    errors = np.asarray(errors)
    worst = errors.max(axis=0)
    best = int(np.argmin(worst))
    return dict(schema='real2sim2real.scale-depth-audit/1', scales=grid.tolist(),
                error_limit_m=float(error_limit_m), min_points=min_points, rows=rows,
                common_passing_scales=grid[worst <= error_limit_m].tolist(),
                minimax_diagnostic=dict(scale=float(grid[best]), worst_error_m=float(worst[best])),
                scope='Finite common-scale grid; fixed orientation and same-pixel evidence only',
                image_invariant_under_camera_centered_scaling=True,
                independent_metric_evidence_established=False,
                human_contact_support_motion_and_seams_validated=False,
                source_modified=False, candidate_approved=False)
