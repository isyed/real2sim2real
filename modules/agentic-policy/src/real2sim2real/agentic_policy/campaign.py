"""One campaign engine for automated authors, review gates and manual proposals."""

import json
import os
import shutil
import time
from importlib.resources import files
from pathlib import Path

from real2sim2real.artifacts import write_json

from . import objectives, sandbox
from .backend import make_backend, schemas
from .catalog import author_catalog
from .execution import rollout
from .proposal_history import export_history, retained_decision, validate_options
from .storage import copy_inputs, hashes, load, locked, store


def initialize(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    config = json.loads((bundle / "experiment.json").read_text())
    if config["mode"] not in {"manual", "automatic", "review"}:
        raise ValueError("mode must be manual, automatic or review")
    objectives.validate(config.get("evaluation", {}))
    validate_options(config)
    revisions = config["feedback_revisions"]
    if type(revisions) is not int or not 0 <= revisions <= 100:
        raise ValueError("feedback_revisions must be an integer in [0,100]")
    if not 0 < config["rollout_timeout_s"] <= 3600:
        raise ValueError("Finite rollout timeout in (0,3600] required")
    if config["mode"] != "manual" and not config.get("author", {}).get("argv"):
        raise ValueError("Automatic/review mode requires an explicit author command")
    if not (bundle / "author").is_dir():
        raise ValueError("An explicit author-visible input directory is required")
    if config.get("author_scene"):
        from .scene_bundle import require_scene_input_parity

        runtime_path = (bundle / config["backend"]["runtime"]).resolve()
        author_scene = (bundle / config["author_scene"]).resolve()
        if not runtime_path.is_relative_to(bundle) or not author_scene.is_relative_to(
            bundle
        ):
            raise ValueError("Scene parity inputs must be inside the bundle")
        runtime = json.loads(runtime_path.read_text())
        host_scene = (bundle / runtime["scene"]).resolve()
        if not host_scene.is_relative_to(bundle):
            raise ValueError("Host scene must be inside the bundle")
        require_scene_input_parity(host_scene, author_scene)
    if "baseline_edits" in config.get("author_loop", {}):
        make_backend(bundle, config["backend"]).edit_fingerprint(
            config["author_loop"]["baseline_edits"]
        )
    if config.get("author_loop", {}).get("prior_edits"):
        backend = make_backend(bundle, config["backend"])
        for edits in config["author_loop"]["prior_edits"].values():
            backend.edit_fingerprint(edits)
    output.mkdir(exist_ok=False, parents=True)
    pins = copy_inputs(bundle, output / "bundle")
    (output / "rounds").mkdir()
    (output / "author-attempts").mkdir()
    state = {
        "version": 1,
        "config": config,
        "bundle_hashes": pins,
        "rounds": [],
        "author_attempts": [],
        "pending": None,
        "retained_nominal": None,
        "total_proposals": revisions + 1,
        "created_unix_s": time.time(),
    }
    store(output, state)
    return status(output)


def status(root):
    state = load(root)
    return {
        "mode": state["config"]["mode"],
        "total_proposals": state["total_proposals"],
        "completed_proposals": sum(r["status"] == "evaluated" for r in state["rounds"]),
        "submitted_proposals": len(state["rounds"]),
        "pending": state["pending"],
        "rounds": [
            {k: r.get(k) for k in ["id", "status", "result"]} for r in state["rounds"]
        ],
        "stopped": state.get("stopped"),
        "next": "stopped"
        if state.get("stopped")
        else "retry interrupted rollout"
        if state["rounds"]
        and state["rounds"][-1]["status"] in {"running", "infrastructure_error"}
        else "complete"
        if len(state["rounds"]) >= state["total_proposals"]
        else ("submit proposal" if state["config"]["mode"] == "manual" else "advance"),
        "retained_nominal": state.get("retained_nominal"),
        "automatic_promotion": False,
    }


def _available(state):
    if state.get("stopped"):
        raise ValueError("Campaign stopped; create a reviewed continuation to resume")
    if len(state["rounds"]) >= state["total_proposals"]:
        raise ValueError("Proposal budget exhausted")
    if state["rounds"] and state["rounds"][-1]["status"] in {
        "running",
        "infrastructure_error",
    }:
        raise ValueError(
            "Resolve the interrupted rollout with retry before proposing again"
        )


def author_workspace(root, state, work):
    """Prepare exactly the author-visible inputs without making an external call."""
    config = state["config"]
    index = len(state["rounds"])
    copy_inputs(root / "bundle/author", work)
    (work / "candidate").mkdir(exist_ok=True)
    if state["rounds"]:
        previous = state["rounds"][-1]
        shutil.copytree(
            root / "rounds" / previous["id"] / "candidate",
            work / "candidate",
            dirs_exist_ok=True,
        )
        write_json(work / "feedback.json", previous["result"])
        # Bounded measured traces from the author's own previous trial only.
        # Reference scores and the private manual comparator are never read.
        trial = root / previous["trials"][-1]
        traces = {}
        for path in [
            trial / "diagnostics/observations.json",
            *sorted(trial.glob("case-*/diagnostics/observations.json")),
        ]:
            if path.is_file():
                rows = json.loads(path.read_text())
                selected = set(range(0, len(rows), 10)) | {len(rows) - 1}
                selected.add(
                    max(range(len(rows)), key=lambda i: rows[i]["penetration_m"])
                )
                selected.add(
                    min(range(len(rows)), key=lambda i: sum(rows[i]["pad_forces_n"]))
                )
                traces[str(path.relative_to(trial))] = [
                    rows[i] for i in sorted(selected)
                ]
        write_json(work / "feedback-traces.json", traces)
        environment = {}
        for path in [
            trial / "skill-events.json",
            *sorted(trial.glob("case-*/skill-events.json")),
        ]:
            if path.is_file():
                rows = [
                    e["result"]
                    for e in json.loads(path.read_text())
                    if e.get("tool") == "observe_environment"
                ]
                if rows:
                    stride = max(1, (len(rows) + 198) // 199)
                    selected = set(range(0, len(rows), stride)) | {len(rows) - 1}
                    selected.add(
                        max(
                            range(len(rows)),
                            key=lambda i: rows[i]["surface_support_fraction"],
                        )
                    )
                    environment[str(path.relative_to(trial))] = [
                        rows[i] for i in sorted(selected)
                    ]
        write_json(work / "feedback-environment.json", environment)

    elif config.get("seed"):
        seed = (root / "bundle" / config["seed"]).resolve()
        if not seed.is_relative_to(root / "bundle"):
            raise ValueError("Seed must be in the bundle")
        shutil.copytree(seed, work / "candidate", dirs_exist_ok=True)
    if config.get("author_loop", {}).get("history"):
        export_history(root, state, work)
    backend = make_backend(root / "bundle", config["backend"])
    tools, access = author_catalog(schemas(backend), config.get("tool_extensions", []))
    write_json(work / "tools.json", tools)
    write_json(work / "CODE_ACCESS.json", access)
    write_json(work / "skills.json", access["skills"])
    write_json(work / "EVALUATION.json", config.get("evaluation", {}))
    brief = (
        f"Task: {config['instruction']}\nProposal P{index} of {state['total_proposals']}.\n"
        'Write candidate/workflow.json (GaP v3) and required scripts, OR remove workflow.json and write candidate/decision.json with {"action": "retain_baseline", "reason": "evidence-based reason"} to stop without another physics trial. Do not provide both.\n'
        "Bounded simulation candidates may fail and be rejected; the retained baseline stays immutable. Nominal preservation is a promotion requirement, not a requirement to guarantee an untested candidate. Do not weaken hard stops or acceptance. Read campaign-history/index.json when present; propose a testable hypothesis or explicitly stop, rather than repeat an empty edit.\n"
        "CODE_ACCESS.json inventories all installed project source. tools.json includes robot, bundled geometry and enabled extension tools. You may search/read all installed source, run Python/shell and reuse its algorithms.\n"
        "The project_source_archive in CODE_ACCESS.json contains remaining module code/docs without installing their optional dependencies; inspect it with Python zipfile if useful.\n"
        "Reuse installed real2sim2real code and bundled tools. Import NodeContext from real2sim2real.agentic_policy.api.\n"
        "skills.json lists all bundled upstream skill source and separately identifies executable robot adaptations. Read their supporting scripts and references. Source references retain upstream imports/connectors and must not be assumed executable. Use only adaptations registered in tools.json through ctx.tool. robot.describe_skills can report checks and limits only when listed there; unavailable adaptations cannot affect this backend.\n"
        "The scene and acceptance criteria are fixed. Do not change them or claim success without physical evaluation.\n"
        "Only registered robot tools affect the live simulator. No reset or object teleport exists.\n"
        "Use only the baseline inputs and development feedback explicitly supplied in this authorized workspace. A manual seed is available only when the campaign explicitly supplies it. Held-out evaluation remains separate and must not be sought or used for revisions.\n"
        "Do not run extra physical trials. Output files belong under candidate/.\n"
    )
    retained = state.get("retained_nominal")
    if retained:
        retained_row = next(r for r in state["rounds"] if r["id"] == retained)
        target = (
            retained_row["result"]
            .get("assessment", {})
            .get("next_objective", "preserve_nominal_success")
        )
        brief += f"Retained successful nominal policy: {retained}. Current objective: {target}. Improve the declared criteria while preserving complete nominal success.\n"
        copy_inputs(root / "rounds" / retained / "candidate", work / "retained-nominal")
    else:
        brief += "Priority: complete the nominal end-to-end task first. Robustness cannot replace nominal success.\n"
    from .authorization import approved_revision_feedback

    review = approved_revision_feedback(root, state)
    if review is not None:
        write_json(work / "HUMAN_REVIEW.json", review)
        brief += "Human outcome review: read HUMAN_REVIEW.json and follow its approved focus within the frozen contract.\n"
    (work / "INSTRUCTIONS.md").write_text(brief)
    shutil.copyfile(
        Path(str(files("real2sim2real.agentic_policy")))
        / "_vendor/asl_prompts/workflow_format.md",
        work / "WORKFLOW_FORMAT.md",
    )
    return work


def preview_author(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    with locked(root):
        state = load(root)
        _available(state)
        author_workspace(root, state, output)
        return {
            "workspace": str(output),
            "files": hashes(output),
            "external_call": False,
        }


def prepare_author(root, state):
    config = state["config"]
    index = len(state["rounds"])
    attempts = [a for a in state["author_attempts"] if a["proposal"] == index]
    if len(attempts) >= config.get("max_author_attempts", 3):
        raise ValueError("Author infrastructure attempt budget exhausted")
    work = root / "author-attempts" / f"P{index}-attempt{len(attempts)}"
    from .authorization import require_authorization, require_revision_review

    require_authorization(root, state)
    require_revision_review(root, state)
    author = config["author"]
    from real2sim2real.artifacts import digest

    for name, expected in author.get("readonly_hashes", {}).items():
        if digest(Path(name)) != expected:
            raise ValueError("Pinned author runtime changed: " + name)
    author_workspace(root, state, work)
    entry = {
        "proposal": index,
        "directory": str(work.relative_to(root)),
        "status": "running",
    }
    state["author_attempts"].append(entry)
    store(root, state)
    argv = [str(x).replace("{workspace}", str(work)) for x in author["argv"]]
    extra = {
        key: os.environ[key] for key in author.get("env_keys", []) if key in os.environ
    }
    try:
        stdout, stderr = sandbox.run(
            argv,
            timeout=author.get("timeout_s", 600),
            cwd=work,
            readonly=author.get("readonly", []),
            network=author.get("network", False),
            extra_env=extra,
        )
        (work / "author.stdout").write_text(stdout)
        (work / "author.stderr").write_text(stderr)
        if (
            not (work / "candidate/workflow.json").is_file()
            and retained_decision(work / "candidate") is None
        ):
            raise ValueError("Author produced no workflow.json")
        entry["status"] = "proposed"
        state["pending"] = str((work / "candidate").relative_to(root))
    except Exception as error:
        entry["status"] = "infrastructure_error"
        entry["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        store(root, state)
    return work / "candidate"


def _evaluate(root, state, row):
    trial = root / "rounds" / row["id"] / f"trial-{len(row['trials'])}"
    trial.mkdir()
    row["trials"].append(str(trial.relative_to(root)))
    row["status"] = "running"
    store(root, state)
    try:
        extra = {}
        if state["config"].get("author_loop", {}).get("stop_on_duplicate_edits"):
            extra["duplicate_edit_fingerprints"] = {
                r["result"]["edit_fingerprint"]: r["id"]
                for r in state["rounds"]
                if r.get("status") == "evaluated"
                and r.get("result", {}).get("edit_fingerprint")
            }
            extra["baseline_edits"] = state["config"]["author_loop"].get(
                "baseline_edits"
            )
            prior = state["config"]["author_loop"].get("prior_edits", {})
            if prior:
                backend = make_backend(root / "bundle", state["config"]["backend"])
                for label, edits in prior.items():
                    extra["duplicate_edit_fingerprints"].setdefault(
                        backend.edit_fingerprint(edits), "prior:" + label
                    )
        receipt = rollout(
            root / "bundle",
            state["config"],
            root / "rounds" / row["id"] / "candidate",
            trial,
            **extra,
        )
        if receipt.get("duplicate_edit"):
            row.update(status="duplicate_edit", result=receipt)
            state["stopped"] = {
                "reason": "duplicate_edit",
                "proposal": row["id"],
                **receipt["duplicate_edit"],
            }
            store(root, state)
            load(root)
            return row
        protocol = state["config"].get("evaluation", {})
        if (
            receipt["verdict"].get("task_completed")
            and not receipt["infrastructure_error"]
        ):
            receipt["robustness_cases"] = {}
            for case in protocol.get("robustness_cases", []):
                case_dir = trial / ("case-" + case["name"])
                case_dir.mkdir()
                case_config = dict(state["config"], backend=case["backend"])
                try:
                    case_result = rollout(
                        root / "bundle",
                        case_config,
                        root / "rounds" / row["id"] / "candidate",
                        case_dir,
                    )
                except Exception as error:  # noqa: BLE001 - preserve nominal evidence when a separate case cannot start
                    case_result = {
                        "infrastructure_error": f"{type(error).__name__}: {error}",
                        "verdict": {"task_completed": False},
                    }
                    write_json(case_dir / "failure.json", case_result)
                receipt["robustness_cases"][case["name"]] = case_result
        receipt["assessment"] = objectives.assessment(receipt, protocol)
        row["result"] = receipt
        if receipt["assessment"]["nominal_completed"]:
            previous = next(
                (
                    r
                    for r in state["rounds"]
                    if r["id"] == state.get("retained_nominal")
                ),
                None,
            )

            def rank(r):
                a = r["result"]["assessment"]
                return (
                    a["robustness_passed"],
                    a["robustness_completed_cases"],
                    sum(
                        v.get("simulation_criteria_passed", False)
                        for v in a["admission"].values()
                    ),
                )

            if previous is None or rank(row) > rank(previous):
                state["retained_nominal"] = row["id"]
        write_json(trial / "result.json", receipt)
        row["status"] = (
            "infrastructure_error" if receipt["infrastructure_error"] else "evaluated"
        )
    except Exception as error:  # noqa: BLE001 - archive tool/worker failures at the process boundary
        row["status"] = "infrastructure_error"
        row["result"] = {"infrastructure_error": f"{type(error).__name__}: {error}"}
        write_json(trial / "failure.json", row["result"])
    store(root, state)
    # Reverify frozen inputs and archived proposals after execution.
    load(root)
    return row


def _submit(root, state, candidate):
    _available(state)
    candidate = Path(candidate).resolve()
    decision = retained_decision(candidate)
    if decision is not None and not (
        state.get("retained_nominal") or state["config"].get("seed")
    ):
        raise ValueError("No retained baseline or declared seed exists")
    if not (candidate / "workflow.json").is_file() and decision is None:
        raise ValueError("Proposal requires workflow.json")
    hashes(candidate)
    proposal_id = f"P{len(state['rounds'])}"
    directory = root / "rounds" / proposal_id
    directory.mkdir()
    pins = copy_inputs(candidate, directory / "candidate")
    if "workflow.json" not in pins and decision is None:
        raise ValueError("Proposal requires workflow.json")
    row = {
        "id": proposal_id,
        "candidate_hashes": pins,
        "status": "running",
        "trials": [],
    }
    state["rounds"].append(row)
    state["pending"] = None
    if decision is not None:
        row.update(
            status="retained_baseline",
            result={"decision": decision, "physics_executed": False},
        )
        state["stopped"] = {
            "reason": "retain_baseline",
            "proposal": proposal_id,
            "detail": decision["reason"],
        }
        store(root, state)
        return row
    return _evaluate(root, state, row)


def submit(root, candidate):
    root = Path(root).resolve()
    with locked(root):
        return _submit(root, load(root), candidate)


def advance(root):
    root = Path(root).resolve()
    with locked(root):
        state = load(root)
        if state.get("stopped") or state["config"]["mode"] == "manual":
            return status(root)
        while len(state["rounds"]) < state["total_proposals"]:
            _available(state)
            candidate = (
                root / state["pending"]
                if state["pending"]
                else prepare_author(root, state)
            )
            if state["config"]["mode"] == "review":
                return {
                    "awaiting_human_review": str(candidate),
                    "action": "inspect/edit then submit this candidate",
                }
            row = _submit(root, state, candidate)
            if row["status"] != "evaluated":
                return row
        return status(root)


def retry(root):
    root = Path(root).resolve()
    with locked(root):
        state = load(root)
        if not state["rounds"] or state["rounds"][-1]["status"] not in {
            "running",
            "infrastructure_error",
        }:
            raise ValueError("No interrupted rollout to retry")
        row = state["rounds"][-1]
        if len(row["trials"]) >= state["config"].get("max_rollout_attempts", 3):
            raise ValueError("Rollout retry budget exhausted")
        return _evaluate(root, state, row)


def export_policy(root, proposal, output):
    """Both routes export the same handoff; export never grants training admission."""
    from real2sim2real.artifacts import publish

    root, output = Path(root).resolve(), Path(output).resolve()
    with locked(root):
        state = load(root)
        matches = [r for r in state["rounds"] if r["id"] == proposal]
        if len(matches) != 1 or matches[0]["status"] != "evaluated":
            raise ValueError("Choose an evaluated proposal explicitly")
        row = matches[0]
        output.mkdir(parents=True, exist_ok=False)
        copy_inputs(root / "rounds" / proposal / "candidate", output / "policy")
        copy_inputs(root / "bundle", output / "experiment")
        write_json(output / "evaluation.json", row["result"])
        write_json(
            output / "handoff.json",
            {
                "proposal": proposal,
                "workflow_mode": state["config"]["mode"],
                "runtime": "real2sim2real.agentic_policy",
                "upstream": "prepared scene/task from reconstruction; original evidence retained in experiment",
                "downstream": "optional residual RL through an embodiment-specific training adapter",
                "declared_rl_criteria_passed": row["result"]
                .get("assessment", {})
                .get("admission", {})
                .get("rl", {})
                .get("admitted", False),
                "training_admitted": False,
                "control_accepted": False,
                "assessment": row["result"].get("assessment"),
                "requirements": [
                    "frozen-policy replay",
                    "matching observation/action adapter",
                    "explicit training readiness decision",
                ],
            },
        )
        publish(
            output,
            "controller-policy-candidate",
            {str(p.relative_to(output)): p for p in output.rglob("*") if p.is_file()},
            metadata={"training_admitted": False, "control_accepted": False},
        )
    return {"artifact": str(output), "training_admitted": False}
