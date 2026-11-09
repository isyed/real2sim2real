"""Translation-only temporal proposals constrained by observed surface proximity.

Fixed surface witnesses conservatively bound the distance from each observed
landmark to the translated mesh. They are not physical contacts, grasp labels,
or hidden-frame observations. Native image/depth/support checks remain external.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class InteractionTranslationConfig:
    max_step_m: float = .08
    maximum_change_m: float = .15
    maximum_gap_increase_m: float = .005
    fidelity_sigma_m: float = .02
    acceleration_sigma_m_s2: float = 3.
    max_seconds: float = 60.
    max_iterations: int = 300

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError('Positive finite bound required: '+key)
        if type(self.max_iterations) is not int:
            raise ValueError('Integer iteration limit required')


def surface_witnesses(vertices, faces, poses, landmarks):
    """Find mesh points nearest to fixed, own-reconstruction 3D landmarks.

    Returns camera-frame witnesses and unsigned distances; no human or object
    pose is changed. A single mesh-local acceleration structure serves all frames.
    """
    import trimesh
    from real2sim2real.reconstruction.recovery_quality import rigid
    v, f, p, h = map(np.asarray, (vertices, faces, poses, landmarks))
    if (v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1] != 3 or not len(f) or f.dtype.kind not in 'iu'
            or f.min() < 0 or f.max() >= len(v) or p.ndim != 3 or p.shape[1:] != (4, 4)
            or not all(rigid(x) for x in p) or h.ndim != 3 or h.shape[0] != len(p)
            or h.shape[2] != 3 or not h.shape[1] or not np.isfinite(h).all()):
        raise ValueError('Finite paired metric mesh, rigid poses and frame-matched landmarks required')
    local = np.einsum('nhj,njk->nhk', h-p[:, None, :3, 3], p[:, :3, :3])
    mesh = trimesh.Trimesh(v, f, process=False)
    q, distances, _ = trimesh.proximity.closest_point(mesh, local.reshape(-1, 3))
    q = q.reshape(h.shape)
    camera = np.einsum('nhj,nkj->nhk', q, p[:, :3, :3])+p[:, None, :3, 3]
    if not np.isfinite(camera).all() or not np.isfinite(distances).all():
        raise ValueError('Nearest-surface query failed')
    return camera, distances.reshape(h.shape[:2])


def propose(poses, timestamps_s, locked, landmarks, witnesses, baseline_distances,
            *, target_translations=None, config=InteractionTranslationConfig()):
    """Fit translations with hard step, displacement and landmark-gap bounds.

    Rotations and locked frames remain bitwise exact. Each original witness
    translates with its surface, so limiting its distance to a landmark also
    bounds that landmark's distance to the whole mesh. Witness provenance and
    independently measured final distances are the caller's responsibility.
    """
    from real2sim2real.reconstruction.recovery_quality import rigid
    config.validate()
    p, times, keep, h, q, distances = map(np.asarray,
        (poses, timestamps_s, locked, landmarks, witnesses, baseline_distances))
    n = len(p)
    if (p.shape != (n, 4, 4) or n < 3 or not all(rigid(x) for x in p)
            or times.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or keep.shape != (n,) or keep.dtype.kind != 'b' or not keep.any()
            or h.ndim != 3 or h.shape[0] != n or h.shape[2] != 3 or h.shape[1] < 1
            or q.shape != h.shape or distances.shape != h.shape[:2]
            or not all(np.isfinite(x).all() for x in (h, q, distances)) or np.any(distances < 0)
            or not np.allclose(np.linalg.norm(q-h, axis=2), distances, atol=1e-7, rtol=1e-6)):
        raise ValueError('Paired poses/times/locks and consistent metric surface witnesses required')
    original = p[:, :3, 3].copy()
    target = original if target_translations is None else np.asarray(target_translations, float)
    if target.shape != original.shape or not np.isfinite(target).all():
        raise ValueError('Finite frame-matched translation target required')
    free = np.flatnonzero(~keep)
    if not len(free):
        raise ValueError('At least one editable frame required')
    dt = np.diff(times)
    D = np.zeros((n-2, n))
    for i in range(n-2):
        mean_dt = (dt[i]+dt[i+1])*.5
        D[i, i:i+3] = [1/dt[i], -1/dt[i]-1/dt[i+1], 1/dt[i+1]]
        D[i] /= mean_dt
    A = D/config.acceleration_sigma_m_s2
    # Metric acceleration can otherwise make the first SLSQP line search dwarf
    # the normalized geometric constraints. A constant changes no minimizer.
    objective_scale=max(1.,float(np.sum(((original-target)/config.fidelity_sigma_m)**2)
                                +np.sum((A@original)**2)))
    step_limit = config.max_step_m-1e-7
    gap_limit = distances+config.maximum_gap_increase_m-1e-7
    change_limit = config.maximum_change_m-1e-7
    if min(step_limit, change_limit, gap_limit.min()) <= 0:
        raise ValueError('Bounds smaller than numerical margin')
    started = time.monotonic();calls = 0;best = None;best_cost = float('inf')
    class Deadline(Exception): pass

    def decode(x):
        t = original.copy();t[free] += x.reshape(-1, 3)
        return t

    def constraints(x):
        t = decode(x);delta = t-original
        gap = q+delta[:, None]-h
        return np.r_[1-np.sum(np.diff(t, axis=0)**2, axis=1)/step_limit**2,
            (1-np.sum(gap**2, axis=2)/gap_limit**2).ravel(),
            1-np.sum(delta[free]**2, axis=1)/change_limit**2]

    def constraint_jac(x):
        t = decode(x);step = np.diff(t, axis=0);delta = t-original;gap = q+delta[:, None]-h
        jac = np.zeros((n-1+n*h.shape[1]+len(free), n, 3))
        for i in range(n-1):
            jac[i, i] = 2*step[i]/step_limit**2
            jac[i, i+1] = -2*step[i]/step_limit**2
        for i in range(n):
            for j in range(h.shape[1]):jac[n-1+i*h.shape[1]+j, i] = -2*gap[i, j]/gap_limit[i, j]**2
        start = n-1+n*h.shape[1]
        for j, i in enumerate(free):jac[start+j, i] = -2*delta[i]/change_limit**2
        return jac[:, free].reshape(len(jac), -1)

    def objective(x):
        nonlocal calls, best, best_cost
        if time.monotonic()-started > config.max_seconds:raise Deadline()
        calls += 1;t = decode(x)
        residual = (t-target)/config.fidelity_sigma_m;acceleration = A@t
        value = float(np.sum(residual**2)+np.sum(acceleration**2))
        gradient = 2*(t-target)/config.fidelity_sigma_m**2+2*A.T@acceleration
        if constraints(x).min() >= -1e-8 and value < best_cost:
            best, best_cost = x.copy(), value
        return value/objective_scale, gradient[free].ravel()/objective_scale

    timeout = False;result = None
    try:
        result = minimize(objective, np.zeros(len(free)*3), method='SLSQP', jac=True,
            bounds=[(-change_limit, change_limit)]*(len(free)*3),
            constraints=[dict(type='ineq', fun=constraints, jac=constraint_jac)],
            options=dict(maxiter=config.max_iterations, ftol=1e-9))
        objective(result.x)
    except Deadline:timeout = True
    chosen = best
    candidate = p.copy()
    if chosen is not None:candidate[:, :3, 3] = decode(chosen)
    delta = candidate[:, :3, 3]-original
    increase = np.linalg.norm(q+delta[:, None]-h, axis=2)-distances
    report = dict(schema='real2sim2real.interaction-translation/1', config=asdict(config),
        feasible_proposal_found=chosen is not None, optimizer_success=bool(result is not None and result.success),
        optimizer_message=str(result.message) if result is not None else 'CPU time limit reached',
        calls=calls, seconds=time.monotonic()-started, deadline_reached=timeout,
        maximum_step_m=float(np.linalg.norm(np.diff(candidate[:, :3, 3], axis=0), axis=1).max()),
        maximum_change_m=float(np.linalg.norm(delta, axis=1).max()),
        maximum_witness_gap_increase_m=float(increase.max()),
        locked_frames_preserved=bool(np.array_equal(candidate[keep], p[keep])),
        rotations_preserved=bool(np.array_equal(candidate[:, :3, :3], p[:, :3, :3])),
        original_retained=chosen is None, native_observation_validation_required=True,
        physical_contact_established=False, full_reference_approved=False)
    return candidate, report
