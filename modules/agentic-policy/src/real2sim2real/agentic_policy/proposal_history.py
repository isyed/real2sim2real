"""Author-visible campaign evidence and explicit non-execution decisions."""

import json

from real2sim2real.artifacts import digest, write_json

from .storage import copy_inputs


def validate_options(config):
    options = config.get("author_loop", {})
    if not isinstance(options, dict) or set(options) - {
        "history",
        "stop_on_duplicate_edits",
        "baseline_edits",
        "prior_edits",
    }:
        raise ValueError("Unknown author_loop options")
    if any(
        type(value) is not bool
        for key, value in options.items()
        if key not in {"baseline_edits", "prior_edits"}
    ):
        raise ValueError("author_loop options must be booleans")
    if "baseline_edits" in options and (
        not isinstance(options["baseline_edits"], dict)
        or not options.get("stop_on_duplicate_edits")
    ):
        raise ValueError(
            "baseline_edits requires a declared profile and duplicate detection"
        )
    if "prior_edits" in options and (
        not options.get("stop_on_duplicate_edits")
        or not isinstance(options["prior_edits"], dict)
        or any(
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(edits, dict)
            for label, edits in options["prior_edits"].items()
        )
    ):
        raise ValueError("prior_edits requires named profiles and duplicate detection")
    if options.get("stop_on_duplicate_edits") and config["backend"]["kind"] not in {
        "reference_refinement",
        "contact_reference_refinement",
    }:
        raise ValueError("Edit deduplication requires reference_refinement")


def retained_decision(candidate):
    path = candidate / "decision.json"
    if not path.exists():
        return None
    decision = json.loads(path.read_text())
    if (
        not isinstance(decision, dict)
        or set(decision) != {"action", "reason"}
        or decision["action"] != "retain_baseline"
        or not isinstance(decision["reason"], str)
        or not decision["reason"].strip()
        or len(decision["reason"]) > 4000
    ):
        raise ValueError(
            "decision.json requires retain_baseline and a nonempty bounded reason"
        )
    if (candidate / "workflow.json").exists():
        raise ValueError("Choose either decision.json or workflow.json, not both")
    return decision


def event_trace(rows, limit=300):
    """Bounded traces prioritizing contact transitions and penetration extrema."""
    if not rows:
        return {"rows": [], "total_rows": 0, "omitted_event_samples": 0}
    events = {0, len(rows) - 1}
    events.add(max(range(len(rows)), key=lambda i: rows[i].get("penetration_m", 0)))

    def contact(row):
        return tuple(v >= 0.1 for v in row.get("pad_forces_n", []))

    for i in range(1, len(rows)):
        if contact(rows[i]) != contact(rows[i - 1]):
            events.update(range(max(0, i - 2), min(len(rows), i + 3)))
    # Keep endpoints and worst penetration even when a chattering trace is large.
    essential = {
        0,
        len(rows) - 1,
        max(range(len(rows)), key=lambda i: rows[i].get("penetration_m", 0)),
    }
    selected = set(essential)
    selected.update(sorted(events - essential)[: max(0, limit - len(selected))])
    remaining = limit - len(selected)
    if remaining:
        stride = max(1, (len(rows) + remaining - 1) // remaining)
        selected.update(list(range(0, len(rows), stride))[:remaining])
    return {
        "rows": [rows[i] for i in sorted(selected)],
        "total_rows": len(rows),
        "omitted_event_samples": len(events - selected),
    }


def export_history(root, state, work):
    """Only own archived proposals/receipts; never scan adjacent private data."""
    history = work / "campaign-history"
    history.mkdir()
    index = []
    for row in state["rounds"]:
        dest = history / row["id"]
        copy_inputs(root / "rounds" / row["id"] / "candidate", dest / "candidate")
        write_json(dest / "result.json", row.get("result", {}))
        traces = {}
        if row.get("trials"):
            trial = root / row["trials"][-1]
            for path in [
                trial / "diagnostics/observations.json",
                *sorted(trial.glob("case-*/diagnostics/observations.json")),
            ]:
                if path.is_file():
                    traces[str(path.relative_to(trial))] = event_trace(
                        json.loads(path.read_text())
                    )
        write_json(dest / "event-traces.json", traces)
        index.append(
            {
                "id": row["id"],
                "status": row["status"],
                "candidate_hashes": row.get("candidate_hashes", {}),
                "result": f"{row['id']}/result.json",
                "result_sha256": digest(dest / "result.json"),
                "trace_sha256": digest(dest / "event-traces.json"),
                "event_traces": f"{row['id']}/event-traces.json",
            }
        )
    write_json(history / "index.json", index)
