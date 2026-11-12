"""Instantaneous force/moment LP bounds with elliptic contact friction and torques.

Inner contact rays are sufficient for the ideal friction cone. Outer supporting
facets are necessary. Neither model enforces contact-mode/velocity compatibility
or proves a trajectory/controller can realize the solution.
"""

from itertools import product

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linprog


def directions(dim):
    if dim == 0:
        return np.zeros((0, 0))
    if dim == 2:
        a = np.arange(24) * 2 * np.pi / 24
        return np.c_[np.cos(a), np.sin(a)]
    if not 1 <= dim <= 5:
        raise ValueError("At most five friction dimensions supported")
    x = np.array([v for v in product((-1.0, 0.0, 1.0), repeat=dim) if any(v)])
    return x / np.linalg.norm(x, axis=1)[:, None]


def contact_basis(frame_on_object, point_world, center_world, dim):
    """Map MuJoCo-style local contact components to [force N, torque Nm] at COM."""
    frame, p, c = (
        np.asarray(x, float) for x in (frame_on_object, point_world, center_world)
    )
    if (
        frame.shape != (3, 3)
        or p.shape != (3,)
        or c.shape != (3,)
        or dim not in (1, 3, 4, 6)
        or not all(np.isfinite(x).all() for x in (frame, p, c))
        or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-7)
    ):
        raise ValueError(
            "Finite orthogonal contact frame, metric point/COM and contact dimension required"
        )
    force = np.zeros((3, 6))
    moment = np.zeros((3, 6))
    force[:, :3] = frame
    moment[:, 3:] = frame
    moment += np.cross(np.broadcast_to(p - c, (6, 3)), force.T).T
    return np.vstack([force, moment])[:, :dim]


def build(
    contacts, torque_offset, torque_lower, torque_upper, group_caps, approximation
):
    offset, lo, hi = (
        np.asarray(x, float) for x in (torque_offset, torque_lower, torque_upper)
    )
    if (
        approximation not in ("inner", "outer")
        or offset.ndim != 1
        or lo.shape != offset.shape
        or hi.shape != offset.shape
        or not np.isfinite(np.r_[offset, lo, hi]).all()
        or (hi < lo).any()
    ):
        raise ValueError("Finite ordered actuator torque envelope required")
    if not contacts:
        raise ValueError("Nonempty contact set required")
    matrices = []
    facets = []
    bounds = []
    groups = []
    for item in contacts:
        dim = item["wrench"].shape[1]
        mu = np.asarray(item["friction"], float)
        if (
            dim not in (1, 3, 4, 6)
            or mu.shape != (dim - 1,)
            or (mu < 0).any()
            or not np.isfinite(mu).all()
            or item["torque"].shape != (len(offset), dim)
            or item["wrench"].shape != (6, dim)
            or not np.isfinite(item["wrench"]).all()
            or not np.isfinite(item["torque"]).all()
        ):
            raise ValueError(
                "Finite compatible contact wrench/Jacobian/friction matrices required"
            )
        active = np.flatnonzero(mu > 0)
        u = directions(len(active))
        if approximation == "inner":
            C = np.zeros((dim, max(1, len(u))))
            C[0] = 1
            if len(active):
                C[active + 1] = mu[active, None] * u.T
            matrices.append(C)
            bounds.extend([(0, None)] * C.shape[1])
            facets.append(np.zeros((0, C.shape[1])))
        else:
            matrices.append(np.eye(dim))
            bounds.append((0, None))
            bounds.extend([(None, None) if m > 0 else (0, 0) for m in mu])
            A = np.zeros((len(u), dim))
            if len(u):
                A[:, 0] = -1
                A[:, active + 1] = u / mu[active]
            facets.append(A)
        groups.append(item["group"])
    C = block_diag(*matrices)
    G = np.hstack([x["wrench"] for x in contacts]) @ C
    T = np.hstack([x["torque"] for x in contacts]) @ C
    cone = block_diag(*facets)
    group_names = sorted(set(groups))
    P = np.zeros((len(group_names), C.shape[1]))
    col = 0
    for item, M in zip(contacts, matrices):
        P[group_names.index(item["group"]), col : col + M.shape[1]] += M[0]
        col += M.shape[1]
    caps = np.array([group_caps[n] for n in group_names], float)
    if not np.isfinite(caps).all() or (caps < 0).any():
        raise ValueError("Finite group normal-force caps required")
    A = np.vstack([cone, T, -T, P])
    b = np.r_[np.zeros(len(cone)), hi - offset, offset - lo, caps]
    return {
        "C": C,
        "G": G,
        "T": T,
        "P": P,
        "groups": group_names,
        "A": A,
        "b": b,
        "bounds": bounds,
        "offset": offset,
        "lower": lo,
        "upper": hi,
        "contact_friction": [np.asarray(c["friction"]) for c in contacts],
    }


def solve(problem, required_wrench, *, capacity=False, time_limit_s=2.0):
    w = np.asarray(required_wrench, float)
    if w.shape != (6,) or not np.isfinite(w).all() or np.linalg.norm(w) == 0:
        raise ValueError("Finite nonzero world force/moment required")
    G, T, A, b = problem["G"], problem["T"], problem["A"], problem["b"]
    scale = np.array(
        [1.0, 1.0, 1.0, 10.0, 10.0, 10.0]
    )  # torque expressed with a declared 0.1 m characteristic length
    if capacity:
        eq = np.c_[G, -w]
        rhs = np.zeros(6)
        obj = np.r_[np.zeros(G.shape[1]), -1.0]
        ub = np.c_[A, np.zeros(len(A))]
        bounds = problem["bounds"] + [(0, None)]
    else:
        eq = G
        rhs = w
        obj = problem["P"].sum(axis=0)
        ub = A
        bounds = problem["bounds"]
    lp = linprog(
        obj,
        A_ub=ub,
        b_ub=b,
        A_eq=scale[:, None] * eq,
        b_eq=scale * rhs,
        bounds=bounds,
        method="highs",
        options={
            "time_limit": time_limit_s,
            "dual_feasibility_tolerance": 1e-8,
            "primal_feasibility_tolerance": 1e-8,
        },
    )
    result = {
        "status": int(lp.status),
        "solver_success": bool(lp.success),
        "message": lp.message,
    }
    if not lp.success:
        return result
    x = lp.x[:-1] if capacity else lp.x
    factor = float(lp.x[-1]) if capacity else 1.0
    torque = problem["offset"] + T @ x
    wrench = G @ x
    max_violation = float(np.max(ub @ lp.x - b, initial=0))
    residual = wrench - factor * w
    valid = (
        max_violation <= 2e-6
        and np.linalg.norm(residual[:3]) <= 2e-6
        and np.linalg.norm(residual[3:]) <= 2e-7
    )
    result.update(
        verified=bool(valid),
        capacity_factor=factor,
        force_residual_n=float(np.linalg.norm(residual[:3])),
        moment_residual_nm=float(np.linalg.norm(residual[3:])),
        max_inequality_violation=max_violation,
        group_normal_force_n=dict(zip(problem["groups"], (problem["P"] @ x).tolist())),
        actuator_torque_nm=torque.tolist(),
        local_contact_components=(problem["C"] @ x).tolist(),
        achieved_wrench=wrench.tolist(),
        total_normal_force_n=float((problem["P"] @ x).sum()),
    )
    components = problem["C"] @ x
    start = 0
    cone_excess = 0.0
    for mu in problem["contact_friction"]:
        local = components[start : start + len(mu) + 1]
        positive = mu > 0
        cone_excess = max(
            cone_excess,
            float(np.linalg.norm(local[1:][positive] / mu[positive]) - local[0]),
        )
        start += len(mu) + 1
    result.update(
        exact_cone_excess_n=cone_excess, exact_cone_satisfied=cone_excess <= 2e-6
    )
    return result
