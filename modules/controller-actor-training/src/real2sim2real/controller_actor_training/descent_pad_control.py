"""Phase-scoped measured pad seating and pressure redistribution during descent."""

from pathlib import Path
import numpy as np

from real2sim2real.artifacts import digest, write_json
from .paired_acquisition_audit import pinned_json
from .contact_anchor import ContactAnchors

LABELS = tuple(
    r + ":" + g
    for r in ("primary", "secondary")
    for g in ("thumb", "index", "middle", "ring", "pinky")
)


def validate_source(spec, config):
    if set(spec) != {"baseline", "activation_start_s", "controller"}:
        raise ValueError(
            "Pinned comparison baseline and explicit descent policy required"
        )
    baseline = pinned_json(spec["baseline"])
    from .continuous_residual import validate_baseline

    validate_baseline(baseline)
    if (
        not baseline.get("physics_run")
        or baseline.get("descent_pad_control")
        or baseline.get("residual_adapter")
        or baseline.get("bounded_grasp_correction")
        or "incomplete_pilot" in baseline["rejected_for"]
    ):
        raise ValueError(
            "Complete original or fixed wrist-feedback comparison required"
        )
    path = Path(spec["baseline"]["path"])
    for name, key in [("scene.xml", "scene_sha256"), ("states.npz", "states_sha256")]:
        if digest(path.with_name(name)) != baseline[key]:
            raise ValueError("Comparison physical artifacts changed")
    ignored = {"output", "purpose", "motion_condition", "descent_pad_control"}
    strip = lambda c: {k: v for k, v in c.items() if k not in ignored}
    if strip(config) != strip(baseline["config"]) or config.get(
        "motion_condition"
    ) not in ("demonstrated", "stationary_hold"):
        raise ValueError(
            "Descent trials cannot retune comparison controller, physics or reference"
        )
    DescentPads(
        spec["controller"], baseline["continuous_protocol"], spec["activation_start_s"]
    )


class DescentPads:
    def __init__(self, settings, protocol, start_s):
        required = {
            "normal_labels",
            "anchor_labels",
            "anchor_settings",
            "rebalance_n",
            "share_gain_n",
            "maximum_share_offset_n",
            "maximum_offset_n",
            "maximum_offset_rate_n_s",
            "ramp_s",
        }
        if set(settings) - {"zero_tangent_labels"} != required:
            raise ValueError(
                "Explicit bounded descent normal, anchor and sharing settings required"
            )
        for key in ("normal_labels", "anchor_labels"):
            if (
                not isinstance(settings[key], list)
                or len(set(settings[key])) != len(settings[key])
                or set(settings[key]) - set(LABELS)
            ):
                raise ValueError("Distinct admitted pad identities required")
        if bool(settings["anchor_labels"]) != bool(settings["anchor_settings"]):
            raise ValueError("Anchor labels and settings must be supplied together")
        zero_tangent = settings.get("zero_tangent_labels", [])
        if (
            not isinstance(zero_tangent, list)
            or len(set(zero_tangent)) != len(zero_tangent)
            or not set(zero_tangent) <= set(settings["normal_labels"])
            or set(zero_tangent) & set(settings["anchor_labels"])
        ):
            raise ValueError(
                "Normal-only seating needs distinct observed pads and exclusive tangent ownership"
            )
        self.anchors = (
            ContactAnchors(settings["anchor_settings"])
            if settings["anchor_settings"]
            else None
        )
        caps = {
            "share_gain_n": 2.0,
            "maximum_share_offset_n": 0.5,
            "maximum_offset_n": 1.0,
            "maximum_offset_rate_n_s": 5.0,
            "ramp_s": 1.0,
        }
        if (
            any(
                not np.isfinite(settings[k])
                or settings[k] < 0
                or settings[k] > v
                or (settings[k] == 0 and k != "share_gain_n")
                for k, v in caps.items()
            )
            or settings["ramp_s"] < 0.1
        ):
            raise ValueError("Descent pressure/ramp exceeds finite bounds")
        if (
            not np.isfinite(start_s)
            or not protocol["acquisition_end_s"]
            <= start_s
            < protocol["capture_duration_s"] - settings["ramp_s"]
        ):
            raise ValueError(
                "Descent must begin after acquisition and before return completion"
            )
        gap = protocol["finger_release"]["start_s"] - protocol["capture_duration_s"]
        if (
            gap <= 0
            or 1.875 * settings["maximum_offset_n"] / gap
            > settings["maximum_offset_rate_n_s"]
        ):
            raise ValueError(
                "Return clock cannot recover pressure offsets within rate limits"
            )
        if set(settings["rebalance_n"]) - set(LABELS):
            raise ValueError("Rebalance targets must name admitted pads")
        self.rebalance = np.array(
            [settings["rebalance_n"].get(n, 0.0) for n in LABELS], float
        )
        if not np.isfinite(self.rebalance).all() or abs(self.rebalance.sum()) > 1e-12:
            raise ValueError(
                "Finite pressure redistribution must conserve pair target sum"
            )
        self.share_direction = np.array(
            [
                0 if n.endswith(":thumb") else (1 if n.startswith("secondary:") else -1)
                for n in LABELS
            ],
            float,
        )
        if any(
            np.max(
                abs(
                    self.rebalance
                    + sign * settings["maximum_share_offset_n"] * self.share_direction
                )
            )
            > settings["maximum_offset_n"] + 1e-12
            for sign in (-1, 1)
        ):
            raise ValueError("Combined rebalance/share exceeds pad offset bounds")
        self.settings, self.protocol, self.start = (
            dict(settings),
            protocol,
            float(start_s),
        )
        self.offset = np.zeros(10)
        self.amplitude = 0.0
        self.recovery = None
        self.updated = self.frozen = self.clipped = 0
        self.isolated = False
        self.secondary_fraction = None
        self.upward = np.zeros(2)
        self.rows = []

    def advance(self, time_s, dt, upward_n, *, active, isolated):
        from .object_grasp_pilot import smoothstep

        forces = np.asarray(upward_n, float)
        if (
            forces.shape != (2,)
            or not np.isfinite(forces).all()
            or not np.isfinite([time_s, dt]).all()
            or dt <= 0
        ):
            raise ValueError(
                "Finite signed bilateral load and positive controller clock required"
            )
        s, p = self.settings, self.protocol
        self.upward = forces.copy()
        self.isolated = bool(isolated)
        if (
            not p.get("release_disabled_for_stationary_diagnostic")
            and time_s >= p["capture_duration_s"]
        ):
            if self.recovery is None:
                self.recovery = self.offset.copy()
            self.amplitude = 1 - float(
                smoothstep(
                    np.clip(
                        (time_s - p["capture_duration_s"])
                        / (p["finger_release"]["start_s"] - p["capture_duration_s"]),
                        0,
                        1,
                    )
                )
            )
            self.offset = self.recovery * self.amplitude
            return
        self.amplitude = float(
            smoothstep(np.clip((time_s - self.start) / s["ramp_s"], 0, 1))
        )
        positive = np.maximum(forces, 0)
        valid = active and isolated and positive.sum() > 0.1
        share = 0.0
        if valid:
            self.secondary_fraction = float(positive[1] / positive.sum())
            share = float(
                np.clip(
                    s["share_gain_n"] * (0.5 - self.secondary_fraction),
                    -s["maximum_share_offset_n"],
                    s["maximum_share_offset_n"],
                )
            )
            self.updated += int(self.amplitude > 0)
        else:
            self.frozen += int(self.amplitude > 0)
        desired = (
            self.amplitude * (self.rebalance + share * self.share_direction)
            if valid
            else np.zeros(10)
        )
        delta = desired - self.offset
        # One common interpolation factor retains the conserved pair sum during slewing.
        step = min(
            1.0, s["maximum_offset_rate_n_s"] * dt / max(np.max(abs(delta)), 1e-15)
        )
        self.offset += step * delta

    def observed_normal(self, label):
        return self.amplitude > 0 and label in self.settings["normal_labels"]

    def target(self, label, nominal_n):
        from .residual_pressure import pressure_goal

        offset = float(self.offset[LABELS.index(label)])
        value = pressure_goal(nominal_n, offset)
        self.clipped += int(offset != 0 and not np.isclose(value, nominal_n + offset))
        return value

    def zero_tangent(self, label):
        return (
            self.amplitude > 0
            and self.isolated
            and label in self.settings.get("zero_tangent_labels", [])
        )

    def anchor_task(self, label, body, obj, observation, normal):
        if self.anchors is None or label not in self.settings["anchor_labels"]:
            return None
        task = self.anchors.task(
            label,
            body,
            obj,
            observation["point_world_m"],
            normal,
            observation["force_n"],
            active=bool(self.amplitude > 0 and self.isolated),
        )
        if task is None:
            return None
        point, projection, velocity = task
        return point, projection, self.amplitude * velocity

    def diagnostics(self):
        return dict(
            amplitude=self.amplitude,
            pressure_offsets_n=dict(zip(LABELS, self.offset.tolist())),
            pressure_offset_sum_n=float(self.offset.sum()),
            upward_force_n=self.upward.tolist(),
            secondary_positive_upward_fraction=self.secondary_fraction,
            isolated=self.isolated,
            updated_steps=self.updated,
            frozen_steps=self.frozen,
            clipped_targets=self.clipped,
            material_anchors=None
            if self.anchors is None
            else self.anchors.diagnostics(),
            recovery_started=self.recovery is not None,
        )

    def record(self, record):
        self.rows.append(dict(time_s=record["time_s"], **self.diagnostics()))

    def finish(self, output):
        write_json(output / "descent-control-trace.json", self.rows)
        return dict(
            settings=self.settings,
            activation_start_s=self.start,
            **self.diagnostics(),
            trace_sha256=digest(output / "descent-control-trace.json"),
            promotion_eligible=False,
            training_run=False,
            anchor_is_task_scoring_anchor=False,
        )


def bind(live, spec, parent):
    if (
        live.servo.anchors is not None
        or live.servo.posture is not None
        or "tangent_gain_s_inv" in live.settings
    ):
        raise ValueError("Descent pad tasks require exclusive finger tangent ownership")
    policy = DescentPads(
        spec["controller"], parent["continuous_protocol"], spec["activation_start_s"]
    )
    admitted = {p["role"] + ":" + p["group"] for p in live.servo.pads}
    if admitted != set(LABELS):
        raise ValueError("Five admitted pads on both hands required")
    if live.servo.payload is None:
        raise ValueError("Existing signed supported-load observation required")
    live.servo.descent = policy
    return policy
