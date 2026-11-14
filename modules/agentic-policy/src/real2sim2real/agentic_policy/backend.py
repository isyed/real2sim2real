"""Trusted simulator adapters; candidate code cannot import the live session."""

import inspect
import json


class ContinuousBackend:
    def __init__(self, bundle, config):
        from real2sim2real.controller_actor_training.continuous_policy import (
            ContinuousPolicy,
        )

        self.config = json.loads((bundle / config["runtime"]).read_text())
        for key in ["scene", "initial"]:
            path = (bundle / self.config[key]).resolve()
            if not path.is_relative_to(bundle.resolve()):
                raise ValueError("Runtime assets must be inside the portable bundle")
            self.config[key] = str(path)
        self.session = ContinuousPolicy(self.config)
        self.tools = {
            "robot." + name: getattr(self.session, name)
            for name in [
                "describe",
                "observe",
                "plan_wrist",
                "fit_hand",
                "screen_path",
                "execute_joint_path",
                "hold",
                "set_pressure",
                "release",
            ]
        }
        from .manipulation import ManipulationSkills
        from .skill_catalog import ADAPTATIONS

        self.skills = ManipulationSkills(self.session)
        self.tools.update(
            {"robot." + name: getattr(self.skills, name) for name in ADAPTATIONS}
        )
        self.tools["robot.describe_skills"] = self.skills.describe_skills

    def finish(self, output):
        from real2sim2real.controller_actor_training.placement_outcome import (
            placement_outcome,
        )

        self.session.save(output / "physics")
        from real2sim2real.artifacts import write_json

        write_json(output / "skill-events.json", self.skills.events)
        result = placement_outcome(
            self.session.history,
            horizon_s=self.config["horizon_s"],
            task=self.config["task"],
            arm_joints=self.config["arm_joints"],
            fault=self.session.fault,
        )
        result["peak_penetration_m"] = max(
            r["penetration_m"] for r in self.session.history
        )
        result["final_observation"] = self.session.history[-1]
        return result


class DiagnosticCounter:
    """Deterministic orchestration fixture, explicitly not a physical simulator."""

    def __init__(self, bundle, config):
        self.value = 0
        self.tools = {"diagnostic.advance": self.advance}

    def advance(self, amount: int) -> dict:
        if type(amount) is not int or not 0 <= amount <= 3:
            raise ValueError("Diagnostic amount must be an integer in [0,3]")
        self.value += amount
        return {"value": self.value}

    def finish(self, output):
        return {
            "task_completed": self.value == 2,
            "value": self.value,
            "control_accepted": False,
            "scope": "orchestration fixture, no physics",
        }


class FrozenReferenceBackend:
    """Comparison-only backend; never supplied to a from-scratch author."""

    def __init__(self, bundle, config):
        from real2sim2real.controller_actor_training.sampled_reference import (
            SampledReferenceReplay,
        )

        runtime = (bundle / config["runtime"]).resolve()
        if not runtime.is_relative_to(bundle.resolve()):
            raise ValueError("Runtime must be inside the portable bundle")
        self.config = json.loads(runtime.read_text())
        for key in ["scene", "initial", "reference"]:
            path = (bundle / self.config[key]).resolve()
            if not path.is_relative_to(bundle.resolve()):
                raise ValueError("Reference assets must be inside the portable bundle")
            self.config[key] = str(path)
        self.session = SampledReferenceReplay(self.config)
        self.tools = {"controller.run_reference": self.session.run}

    def finish(self, output):
        if not self.session.finished:
            return {
                "task_completed": False,
                "control_accepted": False,
                "reason": "Frozen reference did not complete evaluation",
            }
        result = self.session.save(output / "physics")
        return {key: value for key, value in result.items() if key != "records"}


def make_backend(bundle, config):
    if config["kind"] == "continuous":
        return ContinuousBackend(bundle, config)
    if config["kind"] == "diagnostic_counter":
        return DiagnosticCounter(bundle, config)
    if config["kind"] == "frozen_reference":
        return FrozenReferenceBackend(bundle, config)
    if config["kind"] == "contact_reference_refinement":
        from .contact_reference_refinement import ContactReferenceRefinementBackend

        return ContactReferenceRefinementBackend(bundle, config)
    if config["kind"] == "reference_refinement":
        from .reference_refinement import ReferenceRefinementBackend

        return ReferenceRefinementBackend(bundle, config)
    # Third-party adapters are opt-in installed entry points, never author input.
    from importlib.metadata import entry_points

    matches = [
        e
        for e in entry_points(group="real2sim2real.agentic_backends")
        if e.name == config["kind"]
    ]
    if len(matches) != 1:
        raise ValueError("Unknown or ambiguous installed backend: " + config["kind"])
    return matches[0].load()(bundle, config)


def schemas(backend):
    result = {}
    for name, fn in backend.tools.items():
        parameters = inspect.signature(fn).parameters
        inputs = {}
        for key, parameter in parameters.items():
            annotation = parameter.annotation
            inputs[key] = (
                annotation
                if isinstance(annotation, str)
                else getattr(annotation, "__name__", "")
            )
        result[name] = {
            "inputs": inputs,
            "description": inspect.getdoc(fn) or name.replace("_", " "),
        }
    return result
