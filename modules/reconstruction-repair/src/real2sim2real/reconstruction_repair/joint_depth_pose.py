"""Optional fixed-mesh pose/depth hypotheses from multiple source predictions.

Only knot observations enter the fit. Query poses are interpolated hypotheses,
never accepted tracks. The latent foreground depth is jointly constrained by
the rendered surface and every supplied prediction; it is NOT a repaired depth
map and must not be used to validate the fitted pose against itself.
"""
from dataclasses import dataclass, asdict, replace
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class Config:
    translation_bound_m: float = .30
    rotation_bound_deg: float = 25.
    depth_bound_m: float = .30
    pixel_sigma: float = 2.
    prediction_depth_sigma_m: float = .04
    surface_depth_sigma_m: float = .01
    acceleration_sigma_m_s2: float = 6.
    angular_acceleration_sigma_deg_s2: float = 600.
    max_calls: int = 1800
    max_seconds: float = 120.


def fit(vertices, knot_times_s, initial_poses_camera, observations, query_times_s,
        *, config=Config(), progress_callback=None):
    """Fit movable anchors; each observation requires explicit source provenance.

    Observation fields: K, directions (Dx2), outline_px (D), surface_points_local
    (Jx3), depth_predictions_m (one median per source), depth_source_names,
    support_plane_camera (4), source_provenance, ground_truth_used=False.
    Withheld image/depth arrays are deliberately absent from this interface.
    All translations and optical-axis depths use metres and camera coordinates.
    """
    v, t, seeds, query = map(lambda x: np.asarray(x, float),
                             (vertices, knot_times_s, initial_poses_camera, query_times_s))
    n = len(t)
    cfg = asdict(config)
    if (n < 3 or t.shape != (n,) or seeds.shape != (n, 4, 4)
            or v.ndim != 2 or v.shape[1] != 3 or len(v) < 4
            or query.ndim != 1 or not len(query) or len(observations) != n
            or not all(np.isfinite(x).all() for x in (v, t, seeds, query))
            or np.any(np.diff(t) <= 0) or np.any(np.diff(query) <= 0)
            or query[0] < t[0] or query[-1] > t[-1]
            or not np.allclose(seeds[:, 3], [0, 0, 0, 1])
            or not np.allclose(seeds[:, :3, :3] @ seeds[:, :3, :3].swapaxes(1, 2), np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(seeds[:, :3, :3]), 1, atol=1e-5)
            or any(isinstance(x, bool) or not np.isfinite(x) or x <= 0 for x in cfg.values())
            or type(config.max_calls) is not int or config.rotation_bound_deg >= 90
            or progress_callback is not None and not callable(progress_callback)):
        raise ValueError('Finite metric geometry, rigid anchors, bounded query times and positive limits required')
    prepared = []
    for item in observations:
        k, axes, target, surface, predicted, plane = [np.asarray(item[name], float) for name in
            ('K', 'directions', 'outline_px', 'surface_points_local', 'depth_predictions_m', 'support_plane_camera')]
        names = item['depth_source_names']
        if (k.shape != (3, 3) or not np.allclose(k[2], [0, 0, 1]) or min(k[0, 0], k[1, 1]) <= 0
                or axes.ndim != 2 or axes.shape[1] != 2 or len(axes) < 8 or target.shape != (len(axes),)
                or surface.ndim != 2 or surface.shape[1] != 3 or len(surface) < 4
                or predicted.ndim != 1 or len(predicted) < 2 or np.any(predicted <= 0)
                or len(names) != len(predicted) or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name for name in names)
                or plane.shape != (4,) or not np.isclose(np.linalg.norm(plane[:3]), 1)
                or not all(np.isfinite(x).all() for x in (k, axes, target, surface, predicted, plane))
                or item.get('ground_truth_used') is not False or not item.get('source_provenance')):
            raise ValueError('Complete source-only fit observations with distinct depth provenance required')
        prepared.append((k, axes, target, surface, predicted, plane))
    hull = v[ConvexHull(v).vertices]
    center = v.mean(0)
    base_center = np.einsum('nij,j->ni', seeds[:, :3, :3], center) + seeds[:, :3, 3]
    depth0 = np.array([np.median(o[4]) for o in prepared])
    rot_bound = np.deg2rad(config.rotation_bound_deg) / np.sqrt(3)
    limits = np.tile([config.translation_bound_m / np.sqrt(3)] * 3 + [rot_bound] * 3 + [config.depth_bound_m], n)
    started = time.monotonic(); calls = 0; best = None; best_cost = float('inf'); budget = False; last_log = started

    def decode(x):
        x = x.reshape(n, 7)
        r = (Rotation.from_rotvec(x[:, 3:6]) * Rotation.from_matrix(seeds[:, :3, :3])).as_matrix()
        centers = base_center + x[:, :3]
        trans = centers - np.einsum('nij,j->ni', r, center)
        return r, trans, centers, depth0 + x[:, 6]

    class Budget(Exception):
        pass

    def residual(x):
        nonlocal calls, best, best_cost, last_log
        if calls >= config.max_calls or time.monotonic() - started >= config.max_seconds:
            raise Budget()
        calls += 1
        r, trans, centers, depths = decode(x)
        parts = []
        for i, (k, axes, target, surface, predicted, plane) in enumerate(prepared):
            camera = hull @ r[i].T + trans[i]
            projected = camera @ k.T
            uv = projected[:, :2] / np.maximum(projected[:, 2:], .01)
            parts.append((np.max(uv @ axes.T, axis=0) - target) / config.pixel_sigma)
            z = np.median((surface @ r[i].T + trans[i])[:, 2])
            parts.append(np.array([(z - depths[i]) / config.surface_depth_sigma_m]))
            parts.append((depths[i] - predicted) / config.prediction_depth_sigma_m)
            parts.append(np.array([max(0., -np.min(camera @ plane[:3] + plane[3]) - .015) / .002,
                                   max(0., .01 - camera[:, 2].min()) / .001]))
        dt = np.diff(t)
        accel = np.diff(np.diff(centers, axis=0) / dt[:, None], axis=0) / ((dt[1:] + dt[:-1])[:, None] / 2)
        parts.append(accel.ravel() / config.acceleration_sigma_m_s2)
        angular = (Rotation.from_matrix(r[1:]) * Rotation.from_matrix(r[:-1]).inv()).as_rotvec() / dt[:, None]
        parts.append((np.diff(angular, axis=0) / ((dt[1:] + dt[:-1])[:, None] / 2)).ravel()
                     / np.deg2rad(config.angular_acceleration_sigma_deg_s2))
        depth_accel = np.diff(np.diff(depths) / dt) / ((dt[1:] + dt[:-1]) / 2)
        parts.append(depth_accel / config.acceleration_sigma_m_s2)
        parts.append(.1 * x / limits)
        values = np.concatenate(parts)
        cost = float(np.sum(2 * (np.sqrt(1 + values * values) - 1)))
        if cost < best_cost:
            best, best_cost = x.copy(), cost
        if progress_callback is not None and time.monotonic() - last_log >= 15:
            progress_callback(dict(calls=calls, seconds=time.monotonic()-started, proxy_cost=best_cost))
            last_log = time.monotonic()
        return values

    zero = np.zeros(7*n)
    residual(zero); initial_cost = best_cost
    try:
        least_squares(residual, zero, bounds=(-limits, limits), loss='soft_l1',
                      max_nfev=config.max_calls, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    except Budget:
        budget = True
    r, trans, centers, depths = decode(best)
    poses = np.tile(np.eye(4), (len(query), 1, 1))
    poses[:, :3, :3] = Slerp(t, Rotation.from_matrix(r))(query).as_matrix()
    positions = np.stack([np.interp(query, t, centers[:, j]) for j in range(3)], axis=1)
    poses[:, :3, 3] = positions - np.einsum('nij,j->ni', poses[:, :3, :3], center)
    report = dict(schema='real2sim2real.joint-depth-pose/1', config=cfg,
        fit_times_s=t.tolist(), query_times_s=query.tolist(), latent_surface_depth_m=depths.tolist(),
        predicted_depth_residual_m=[(depths[i]-o[4]).tolist() for i, o in enumerate(prepared)],
        depth_source_names=[o['depth_source_names'] for o in observations],
        source_provenance=[o['source_provenance'] for o in observations],
        calls=calls, seconds=time.monotonic()-started, budget_reached=budget,
        initial_proxy_cost=initial_cost, best_proxy_cost=best_cost,
        camera_changed=False, mesh_changed=False, original_depth_changed=False,
        former_anchor_reestimated=True, poses_are_proposals=True,
        between_knot_poses_inferred=True, heldout_observations_used=False,
        latent_depth_is_not_validation_evidence=True, reference_approved=False,
        independent_native_image_depth_support_motion_checks_required=True)
    return poses, report


def refine(vertices, knot_times_s, initial_poses_camera, observation_builder, query_times_s,
           *, config=Config(), max_rounds=3, progress_callback=None):
    """Refresh front-surface correspondences using fit frames only.

    A rotation can expose a different surface, invalidating a frozen fitting
    proxy. The builder receives knot poses only and must rebuild observations
    from those same fitting images. Withheld frames are not supplied. The total
    work budget and displacement bound remain relative to the original seed.
    """
    if not callable(observation_builder) or type(max_rounds) is not int or not 1 <= max_rounds <= 5:
        raise ValueError('Explicit fit-only observation builder and 1–5 rounds required')
    original = np.asarray(initial_poses_camera, float).copy()
    current = original.copy(); history = []; calls = 0; started = time.monotonic()
    t = np.asarray(knot_times_s, float); query = np.asarray(query_times_s, float)
    all_times = np.unique(np.r_[t, query]); query_index = np.searchsorted(all_times, query)
    knot_index = np.searchsorted(all_times, t)
    for iteration in range(max_rounds):
        remaining = config.max_seconds - (time.monotonic()-started)
        if remaining <= 0 or calls >= config.max_calls:
            break
        observations = observation_builder(current.copy())
        poses, receipt = fit(vertices, t, current, observations, all_times,
            config=replace(config, max_seconds=remaining, max_calls=config.max_calls-calls),
            progress_callback=progress_callback)
        calls += receipt['calls']; history.append(receipt); current = poses[knot_index].copy()
        if receipt['budget_reached']:
            break
    if not history:
        raise ValueError('No fitting round completed within the declared budget')
    v = np.asarray(vertices); center = v.mean(0)
    displacement = np.linalg.norm(np.einsum('nij,j->ni', current[:, :3, :3], center) + current[:, :3, 3]
        - np.einsum('nij,j->ni', original[:, :3, :3], center) - original[:, :3, 3], axis=1)
    angle = np.rad2deg((Rotation.from_matrix(current[:, :3, :3]) * Rotation.from_matrix(original[:, :3, :3]).inv()).magnitude())
    report = dict(schema='real2sim2real.joint-depth-pose-refresh/1', rounds=history,
        calls=calls, seconds=time.monotonic()-started,
        initial_seed_bounds_passed=bool(displacement.max() <= config.translation_bound_m and angle.max() <= config.rotation_bound_deg),
        max_seed_center_shift_m=float(displacement.max()), max_seed_rotation_deg=float(angle.max()),
        heldout_observations_used=False, surface_correspondences_refreshed=True,
        original_depth_changed=False, mesh_changed=False, camera_changed=False,
        latent_depth_is_not_validation_evidence=True, reference_approved=False)
    return poses[query_index], report
