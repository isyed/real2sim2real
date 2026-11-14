"""Data-only, baseline-relative contact-seating proposal validation."""

import math
from copy import deepcopy


def validate_seating_proposal(document, seed, *, pads, joints):
    """Admit only configure.inputs edits; geometry/clock/tool graph stays frozen.

    Pad offsets are object-frame metres. Joint offsets are radians from the
    frozen closure, never increments on previous proposals. Fitting must enforce
    the same final joint bounds; validation is not collision or physics admission.
    """
    try:
        inputs = document["nodes"]["configure"]["inputs"]
        if set(inputs) != {"pad_offsets_object_m", "closure_offsets_rad"}:
            raise ValueError("Only pad and closure offsets are editable")
        offsets = inputs["pad_offsets_object_m"]
        closure = inputs["closure_offsets_rad"]
        if set(offsets) != set(pads) or set(closure) - set(joints):
            raise ValueError("Only declared pads/joints are admitted")
        for vector in offsets.values():
            if (
                not isinstance(vector, list)
                or len(vector) != 3
                or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in vector
                )
                or math.hypot(*vector) > 0.003
            ):
                raise ValueError("Finite pad offset norm must not exceed 3 mm")
        values = list(closure.values())
        if (
            any(
                type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 0.12
                for x in values
            )
            or math.hypot(*values) > 0.24
        ):
            raise ValueError("Closure offsets exceed joint or total bounds")
        if any(any(x != 0 for x in v) for v in offsets.values()) and any(values):
            raise ValueError(
                "Choose pad fitting or direct closure offsets per proposal"
            )
        restored = deepcopy(document)
        restored["nodes"]["configure"]["inputs"] = deepcopy(
            seed["nodes"]["configure"]["inputs"]
        )
        if restored != seed:
            raise ValueError("Graph and all non-parameter content are frozen")
        return deepcopy(inputs)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed seeded seating proposal") from error
