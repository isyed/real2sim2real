"""Explicit monotone retiming of a pinned reference without inserted pauses."""

import numpy as np


class PhaseTiming:
    def __init__(self, reference_s, execution_s):
        reference, execution = (
            np.asarray(x, float) for x in (reference_s, execution_s)
        )
        if (
            reference.ndim != 1
            or len(reference) < 2
            or reference.shape != execution.shape
            or not np.isfinite(np.r_[reference, execution]).all()
            or reference[0] != 0
            or execution[0] != 0
            or (np.diff(reference) <= 0).any()
            or (np.diff(execution) <= 0).any()
        ):
            raise ValueError(
                "Matching finite strictly increasing clocks starting at zero required"
            )
        self.reference_s = reference.copy()
        self.execution_s = execution.copy()

    def reference(self, execution_s):
        if not np.isfinite(execution_s):
            raise ValueError("Finite execution time required")
        return float(np.interp(execution_s, self.execution_s, self.reference_s))

    def execution(self, reference_s):
        if not np.isfinite(reference_s):
            raise ValueError("Finite reference time required")
        return float(np.interp(reference_s, self.reference_s, self.execution_s))

    @property
    def duration_s(self):
        return float(self.execution_s[-1])

    def diagnostics(self):
        return {
            "reference_s": self.reference_s.tolist(),
            "execution_s": self.execution_s.tolist(),
        }
