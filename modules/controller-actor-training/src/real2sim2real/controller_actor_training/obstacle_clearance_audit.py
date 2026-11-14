"""Read-only geometric witnesses at saved configurations, not path admission."""

import numpy as np


def clearance_witness(model, data, moving_names, obstacle_names, *, search_m=0.02):
    """Minimum signed distance within search radius for explicit finite geom sets.

    The caller supplies current kinematics (mj_fwdPosition). No simulation step,
    force recomputation, exclusions, collision masks or tolerances are changed.
    This samples geometry only; it does not certify continuous swept clearance.
    Bounding spheres prune only pairs provably outside the search radius.
    """
    import mujoco

    if not np.isfinite(search_m) or search_m <= 0:
        raise ValueError("Positive finite search radius required")
    groups = []
    for names in (moving_names, obstacle_names):
        ids = [model.geom(name).id for name in names]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Nonempty distinct geometry sets required")
        if any(model.geom_rbound[g] <= 0 for g in ids):
            raise ValueError(
                "Finite bounding geometries required; planes are unsupported"
            )
        groups.append(np.array(ids))
    if set(groups[0]) & set(groups[1]):
        raise ValueError("Disjoint geometry sets required")
    best = None
    tested = 0
    invalid = []
    others = groups[1]
    for g in groups[0]:
        lower = (
            np.linalg.norm(data.geom_xpos[others] - data.geom_xpos[g], axis=1)
            - model.geom_rbound[others]
            - model.geom_rbound[g]
        )
        for h in others[lower < search_m]:
            segment = np.zeros(6)
            distance = mujoco.mj_geomDistance(
                model, data, int(g), int(h), search_m, segment
            )
            tested += 1
            if distance < search_m:
                # Some mesh distance failures return zero with unrelated witnesses.
                # Never promote those into a clearance/contact measurement.
                witness_length = float(np.linalg.norm(segment[:3] - segment[3:]))
                if (
                    not np.isfinite(distance)
                    or not np.isfinite(segment).all()
                    or abs(witness_length - abs(distance)) > 1e-6
                    or np.linalg.norm(segment[:3] - data.geom_xpos[g])
                    > model.geom_rbound[g] + 1e-6
                    or np.linalg.norm(segment[3:] - data.geom_xpos[h])
                    > model.geom_rbound[h] + 1e-6
                ):
                    invalid.append(
                        {
                            "geoms": [model.geom(g).name, model.geom(h).name],
                            "reported_distance_m": float(distance),
                            "witness_length_m": witness_length,
                        }
                    )
                    continue
            elif not np.isfinite(distance):
                invalid.append(
                    {
                        "geoms": [model.geom(g).name, model.geom(h).name],
                        "reason": "nonfinite distance",
                    }
                )
                continue
            if distance < search_m and (
                best is None or distance < best["signed_distance_m"]
            ):
                best = {
                    "geoms": [model.geom(g).name, model.geom(h).name],
                    "signed_distance_m": float(distance),
                    "witness_points_m": segment.reshape(2, 3).tolist(),
                }
    return {
        "nearest": best,
        "search_m": search_m,
        "tested_pairs": tested,
        "no_pair_within_search": best is None and not invalid,
        "invalid_query_count": len(invalid),
        "invalid_query_examples": invalid[:8],
        "complete_within_search": not invalid,
        "sampled_configuration_only": True,
        "physics_stepped": False,
    }
