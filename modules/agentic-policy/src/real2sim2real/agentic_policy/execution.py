"""Run a graph in isolation and score its physical effects in the trusted host."""

import json
import os
import select
import subprocess
import sys
import time

from real2sim2real.artifacts import write_json

from . import sandbox
from .backend import make_backend, schemas


class DuplicateEdit(Exception):
    """Trusted pre-step duplicate decision, not a simulation failure."""


def rollout(
    bundle,
    config,
    candidate,
    output,
    *,
    duplicate_edit_fingerprints=None,
    baseline_edits=None,
):
    backend = make_backend(bundle, config["backend"])
    if duplicate_edit_fingerprints is not None:
        duplicate_edit_fingerprints = dict(duplicate_edit_fingerprints)
        if baseline_edits is not None:
            duplicate_edit_fingerprints[backend.edit_fingerprint(baseline_edits)] = (
                "declared_baseline"
            )
    work = output / "work"
    work.mkdir()
    import shutil

    shutil.copytree(candidate, work, dirs_exist_ok=True)
    write_json(work / "tools.json", schemas(backend))
    write_json(work / "extensions.json", config.get("tool_extensions", []))
    started = time.monotonic()
    fault = None
    rejection = None
    calls = 0
    edit_fingerprint = None
    duplicate = None
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen(
            sandbox.command(
                [sys.executable, "-m", "real2sim2real.agentic_policy.worker", work],
                writable=[work],
                cwd=work,
            ),
            env=sandbox.environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            start_new_session=True,
        )
        buffer = b""
        try:
            while True:
                remaining = config["rollout_timeout_s"] - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Graph rollout deadline")
                ready, _, _ = select.select(
                    [process.stdout], [], [], min(remaining, 0.2)
                )
                if not ready:
                    if process.poll() is not None:
                        break
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                buffer += chunk
                if len(buffer) > 2_000_000:
                    raise ValueError("Oversized tool request")
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    request = json.loads(raw)
                    calls += 1
                    if calls > config.get("max_tool_calls", 10000):
                        raise ValueError("Tool call budget exceeded")
                    try:
                        name = request["tool"]
                        if name not in backend.tools:
                            raise ValueError("Unregistered robot tool")
                        if (
                            duplicate_edit_fingerprints is not None
                            and name == "controller.run_reference"
                        ):
                            edit_fingerprint = backend.edit_fingerprint(
                                **request.get("inputs", {})
                            )
                            if edit_fingerprint in duplicate_edit_fingerprints:
                                duplicate = {
                                    "matches_proposal": duplicate_edit_fingerprints[
                                        edit_fingerprint
                                    ],
                                    "fingerprint": edit_fingerprint,
                                }
                                raise DuplicateEdit("Repeated effective reference edit")
                        value = backend.tools[name](**request.get("inputs", {}))
                        reply = {"ok": True, "value": value}
                    except DuplicateEdit:
                        raise
                    except Exception as error:  # noqa: BLE001 - archive tool/worker failures at the process boundary
                        reply = {
                            "ok": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    process.stdin.write(
                        (json.dumps(reply, allow_nan=False) + "\n").encode()
                    )
                    process.stdin.flush()
            process.wait(
                timeout=max(
                    0.1, config["rollout_timeout_s"] - (time.monotonic() - started)
                )
            )
            if process.returncode:
                raise RuntimeError(
                    f"Graph worker exited {process.returncode}; see worker.log"
                )
        except DuplicateEdit:
            pass
        except (TimeoutError, subprocess.TimeoutExpired, ValueError) as error:
            rejection = f"{type(error).__name__}: {error}"
        except Exception as error:  # noqa: BLE001 - archive tool/worker failures at the process boundary
            fault = f"{type(error).__name__}: {error}"
        finally:
            sandbox.kill(process)
    if duplicate is not None:
        receipt = {
            "duplicate_edit": duplicate,
            "edit_fingerprint": edit_fingerprint,
            "physics_executed": False,
            "infrastructure_error": None,
            "verdict": {"task_completed": False},
            "tool_calls": calls,
        }
        write_json(output / "result.json", receipt)
        return receipt
    graph_file = work / "graph-result.json"
    graph = (
        json.loads(graph_file.read_text())
        if graph_file.is_file()
        else {"graph_status": "failure"}
    )
    verdict = backend.finish(output)
    if config.get("evaluation", {}).get("diagnostics"):
        from .diagnostics import audit

        report = audit(
            bundle,
            config["backend"].get("diagnostics_runtime", config["backend"]["runtime"]),
            output / "physics/states.npz",
            config["evaluation"]["diagnostics"],
            output / "diagnostics",
        )
        verdict["interface_task_completed"] = verdict.get("task_completed", False)
        verdict["task_completed"] = bool(
            verdict["interface_task_completed"] and report["task_completed"]
        )
        verdict["diagnostics"] = report["metrics"]
        verdict["matched_outcome"] = report["outcome"]
    if fault or rejection or graph.get("graph_status") != "success":
        verdict["task_completed"] = False
    receipt = {
        "edit_fingerprint": edit_fingerprint,
        "graph": graph,
        "infrastructure_error": fault,
        "execution_rejection": rejection,
        "tool_calls": calls,
        "elapsed_s": time.monotonic() - started,
        "verdict": verdict,
    }
    write_json(output / "result.json", receipt)
    return receipt
