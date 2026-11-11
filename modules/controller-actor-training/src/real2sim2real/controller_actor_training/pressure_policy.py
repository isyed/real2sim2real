"""Small deterministic pressure policies for residual-authority experiments."""

from copy import deepcopy

import numpy as np

from .residual_pressure import validate_action


class PressurePolicy:
    """Use only the current public observation and its declared schema.

    Actions are normalized pressure requests. The environment owns physical
    bounds, phase activation and slew limits. No fit or optimizer runs here.
    """

    def __init__(self, settings):
        self.settings = deepcopy(settings)
        kind = settings.get("kind")
        fields = {"kind", "action"}
        if kind == "drift_ramp":
            fields |= {"start_m", "full_m"}
        if kind not in {"constant", "drift_ramp"} or set(settings) != fields:
            raise ValueError("Explicit constant or drift_ramp pressure policy required")
        self.action = validate_action(settings["action"])
        if kind == "drift_ramp":
            start, full = settings["start_m"], settings["full_m"]
            if (
                isinstance(start, bool)
                or isinstance(full, bool)
                or not np.isfinite([start, full]).all()
                or not 0 <= start < full <= 0.01
            ):
                raise ValueError(
                    "Finite ordered drift thresholds within 10 mm required"
                )

    def __call__(self, observation, info):
        if self.settings["kind"] == "constant":
            return self.action.copy()
        observation = np.asarray(observation, float)
        if observation.ndim != 1 or not np.isfinite(observation).all():
            raise ValueError("Finite vector observation required")
        schema = info["observation_schema"]
        a, b = schema["grasp_anchor_valid"]
        valid = observation[a:b]
        a, b = schema["grasp_drift_in_wrists_m"]
        drift = observation[a:b]
        if valid.shape != (1,) or drift.shape != (6,) or valid[0] not in (0, 1):
            raise ValueError("Declared anchor flag and bilateral metric drift required")
        if not valid[0]:
            return np.zeros(4)
        gain = np.clip(
            (np.linalg.norm(drift[:3]) - self.settings["start_m"])
            / (self.settings["full_m"] - self.settings["start_m"]),
            0,
            1,
        )
        return self.action * gain
