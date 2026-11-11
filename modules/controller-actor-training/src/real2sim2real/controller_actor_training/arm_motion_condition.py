"""Explicit stationary diagnostic on an audited acquisition and hold clock."""

from copy import deepcopy
import numpy as np


def apply_condition(parent, targets, condition):
    if condition == "demonstrated":
        return parent, targets
    if condition != "stationary_hold":
        raise ValueError("Choose demonstrated motion or an explicit stationary hold")
    values = np.asarray(targets, float)
    records = parent["records"]
    indices = [i for i, r in enumerate(records) if r["phase"] == "hold"]
    if (
        values.ndim != 2
        or len(values) != len(records)
        or not np.isfinite(values).all()
        or not indices
        or indices != list(range(indices[0], len(records)))
    ):
        raise ValueError(
            "Complete contiguous hold and finite matching arm commands required"
        )
    first = indices[0]
    if not all(
        k in records[first] for k in ("reference_position_m", "reference_rotation")
    ):
        raise ValueError("An independently specified stationary reference is required")
    result = deepcopy(parent)
    held = values.copy()
    held[first:] = values[first]
    for i in indices:
        for key in (
            "reference_position_m",
            "reference_rotation",
            "object_position_m",
            "object_rotation",
        ):
            result["records"][i][key] = deepcopy(records[first][key])
    return result, held
