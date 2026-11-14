"""Separate diagnostic continuation from strict acceptance and task completion."""


class DiagnosticVerdict:
    """Record first warnings; explicit diagnostic mode never suppresses hard stops.

    The caller must always supply physical/numerical faults as hard_failures.
    Classification and thresholds belong in the pinned experiment protocol.
    """

    def __init__(self, *, continue_quality_warnings=False):
        if not isinstance(continue_quality_warnings, bool):
            raise TypeError("Explicit boolean continuation mode required")
        self.continue_quality_warnings = continue_quality_warnings
        self.warnings = {}
        self.hard_failures = {}

    def observe(self, time_s, *, warnings=(), hard_failures=()):
        import math

        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Finite nonnegative simulation time required")
        for events, destination in [
            (warnings, self.warnings),
            (hard_failures, self.hard_failures),
        ]:
            for event in events:
                if not isinstance(event, str) or not event:
                    raise ValueError("Named diagnostic events required")
                destination.setdefault(event, float(time_s))
        return bool(
            self.hard_failures or (self.warnings and not self.continue_quality_warnings)
        )

    def finish(self, *, complete_horizon, task_completed):
        if not isinstance(complete_horizon, bool) or not isinstance(
            task_completed, bool
        ):
            raise TypeError("Explicit completion verdicts required")
        completed = complete_horizon and task_completed and not self.hard_failures
        return {
            "task_completed": bool(completed),
            "complete_horizon": complete_horizon,
            "strict_quality_pass": bool(completed and not self.warnings),
            "warnings": dict(self.warnings),
            "hard_failures": dict(self.hard_failures),
            "control_accepted": False,
        }
