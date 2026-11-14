"""Nominal-first objectives and separately declared downstream admission gates."""

import math


def _limits(verdict, limits):
    checks = {}
    for name, condition in limits.items():
        value = verdict
        try:
            for key in name.split("."):
                value = value[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError("Finite numeric evidence required")
            checks[name] = all(
                (value <= bound if op == "max" else value >= bound)
                for op, bound in condition.items()
            )
        except (KeyError, TypeError, ValueError):
            checks[name] = False
    return checks


def assessment(receipt, protocol):
    nominal = receipt["verdict"].get("task_completed") is True and not receipt.get(
        "infrastructure_error"
    )
    cases = receipt.get("robustness_cases", {})
    declared = protocol.get("robustness_cases", [])
    robustness = bool(
        declared
        and set(cases) == {case["name"] for case in declared}
        and all(
            r["verdict"].get("task_completed") is True
            and not r.get("infrastructure_error")
            for r in cases.values()
        )
    )
    admissions = {}
    for goal in ["sft", "rl", "control"]:
        criteria = protocol.get("admission", {}).get(goal)
        if not criteria:
            admissions[goal] = {"admitted": False, "reason": "criteria not configured"}
            continue
        nominal_checks = _limits(receipt["verdict"], criteria["limits"])
        case_checks = {
            name: _limits(row["verdict"], criteria["limits"])
            for name, row in cases.items()
        }
        require_cases = criteria.get("require_robustness", True)
        simulation_passed = (
            nominal
            and all(nominal_checks.values())
            and (
                not require_cases
                or robustness
                and all(all(c.values()) for c in case_checks.values())
            )
        )
        prerequisites = {
            name: protocol.get("evidence", {}).get(name) is True
            for name in criteria.get("prerequisites", [])
        }
        admitted = simulation_passed and all(prerequisites.values())
        admissions[goal] = {
            "admitted": bool(admitted),
            "simulation_criteria_passed": bool(simulation_passed),
            "prerequisites": prerequisites,
            "nominal_checks": nominal_checks,
            "case_checks": case_checks,
            "requires_robustness": require_cases,
            "scope": "declared simulation criteria only; hardware validation remains separate",
        }
    target = protocol.get("target", "nominal")
    objective = "nominal_end_to_end"
    if nominal:
        if target == "nominal":
            objective = "preserve_nominal_success"
        elif declared and not robustness:
            objective = "robustness_without_nominal_regression"
        elif target in admissions and not admissions[target]["admitted"]:
            objective = (
                "preserve_simulation_acceptance_pending_external_evidence"
                if admissions[target].get("simulation_criteria_passed")
                else target + "_acceptance_without_nominal_regression"
            )
        else:
            objective = "preserve_declared_acceptance"
    return {
        "nominal_completed": nominal,
        "robustness_passed": robustness,
        "robustness_completed_cases": sum(
            cases.get(c["name"], {}).get("verdict", {}).get("task_completed") is True
            and not cases.get(c["name"], {}).get("infrastructure_error")
            for c in declared
        ),
        "admission": admissions,
        "next_objective": objective,
    }


def validate(protocol):
    if protocol.get("target", "nominal") not in {"nominal", "sft", "rl", "control"}:
        raise ValueError("Unknown evaluation target")
    cases = protocol.get("robustness_cases", [])
    if len(cases) > 50 or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Unique bounded robustness cases required")
    for case in cases:
        if not case["name"].replace("-", "").replace("_", "").isalnum():
            raise ValueError("Case names must be simple identifiers")
    for criteria in protocol.get("admission", {}).values():
        if any(not isinstance(k, str) for k in criteria.get("prerequisites", [])):
            raise ValueError("Prerequisite names must be strings")
        if not criteria.get("limits"):
            raise ValueError("Admission needs explicit metric limits")
        for bounds in criteria["limits"].values():
            if (
                not bounds
                or set(bounds) - {"min", "max"}
                or any(
                    type(x) not in (int, float) or not math.isfinite(x)
                    for x in bounds.values()
                )
            ):
                raise ValueError("Finite min/max limits required")
            if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
                raise ValueError("Minimum cannot exceed maximum")
    target = protocol.get("target", "nominal")
    if target != "nominal" and target not in protocol.get("admission", {}):
        raise ValueError(
            "Declare acceptance criteria for the selected target before freezing"
        )
    if (
        target != "nominal"
        and protocol["admission"][target].get("require_robustness", True)
        and not cases
    ):
        raise ValueError(
            "Selected acceptance target requires explicit robustness cases"
        )
