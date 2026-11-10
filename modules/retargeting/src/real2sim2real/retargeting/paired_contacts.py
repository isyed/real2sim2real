"""Combine two contact sets without conflating patch samples with force budgets."""

import numpy as np

from .pad_contacts import gravity_wrench_screen


def paired_wrench_screen(
    first, second, center_of_mass, mass_kg, friction, max_normal_force_n
):
    """Screen geometry in one metric frame; this does not validate two robot hands.

    Each set supplies points_m, normals, groups and samples_per_pad. Every pad
    must have a nearby opposed surface sample. Pad labels are local to each set;
    the second set receives distinct force budgets, never one budget per sample.
    """
    points, normals, groups = [], [], []
    offset = 0
    missing = False
    for contact_set in (first, second):
        p = np.asarray(contact_set["points_m"], float).reshape(-1, 3)
        n = np.asarray(contact_set["normals"], float).reshape(-1, 3)
        labels = np.asarray(contact_set["groups"], dtype=int)
        counts = np.asarray(contact_set["samples_per_pad"])
        if (
            n.shape != p.shape
            or labels.shape != (len(p),)
            or counts.ndim != 1
            or not len(counts)
            or counts.dtype.kind not in "iu"
            or (counts < 0).any()
            or not np.isfinite(p).all()
            or not np.isfinite(n).all()
            or not np.allclose(np.linalg.norm(n, axis=1), 1)
            or list(labels) != list(contact_set["groups"])
            or (labels < 0).any()
            or (labels >= len(counts)).any()
            or not np.array_equal(np.bincount(labels, minlength=len(counts)), counts)
        ):
            raise ValueError(
                "Consistent finite contact samples and per-pad counts required"
            )
        missing |= bool((counts == 0).any())
        points.extend(p)
        normals.extend(n)
        groups.extend(labels + offset)
        offset += len(counts)
    if missing:
        return dict(
            feasible=False,
            reason="missing_nearby_opposed_pad_samples",
            robot_pair_validated=False,
            dynamic_success_validated=False,
        )
    result = gravity_wrench_screen(
        points,
        normals,
        center_of_mass,
        mass_kg,
        friction,
        max_normal_force_n,
        contact_groups=groups,
    )
    result.update(robot_pair_validated=False, pad_count=offset)
    return result
