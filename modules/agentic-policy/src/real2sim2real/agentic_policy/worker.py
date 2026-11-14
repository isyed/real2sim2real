"""Isolated graph worker; live physics is available only over JSON RPC."""

import inspect
import json
import sys
import threading
from pathlib import Path

from ._vendor.gap.runtime.executor import WorkflowExecutor
from ._vendor.gap_core.tools import ToolRegistry
from .catalog import register_library_tools


class Bridge:
    def __init__(self):
        self.lock = threading.Lock()

    def call(self, name, **inputs):
        with self.lock:
            print(
                json.dumps({"tool": name, "inputs": inputs}, allow_nan=False),
                file=sys.__stdout__,
                flush=True,
            )
            reply = json.loads(sys.stdin.readline())
            if not reply["ok"]:
                raise ValueError(reply["error"])
            return reply["value"]


def main():
    work = Path(sys.argv[1])
    # Candidate prints cannot impersonate protocol messages by accident.
    sys.stdout = sys.stderr
    schemas = json.loads((work / "tools.json").read_text())
    registry = ToolRegistry()
    bridge = Bridge()
    for name, description in schemas.items():
        fields = description["inputs"]
        if any(not key.isidentifier() or key.startswith("_") for key in fields):
            raise ValueError("Invalid trusted tool schema")
        types = {t.__name__: t for t in [dict, list, float, int, str, bool]}
        if any(v not in types for v in fields.values()):
            raise ValueError("Unsupported trusted type")

        def proxy(tool_name):
            def invoke(**kwargs):
                return bridge.call(tool_name, **kwargs)

            return invoke

        fn = proxy(name)
        fn.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    key, inspect.Parameter.KEYWORD_ONLY, annotation=types[value]
                )
                for key, value in fields.items()
            ],
            return_annotation=dict,
        )
        fn.__annotations__ = {key: types[value] for key, value in fields.items()} | {
            "return": dict
        }
        registry.register_callable(
            name, fn, summary=description["description"], requires_ctx=False
        )
    register_library_tools(registry, json.loads((work / "extensions.json").read_text()))
    try:
        executor = WorkflowExecutor(
            work / "workflow.json",
            tool_registry=registry,
            trace_dir=work / "trace",
            max_node_workers=1,
            node_visit_cap=256,
            checkpoints="off",
        )
        executor.execute()
        result = {"graph_status": executor.exit_status}
    except Exception as error:  # noqa: BLE001 - archive tool/worker failures at the process boundary
        result = {
            "graph_status": "failure",
            "error": f"{type(error).__name__}: {error}",
        }
    (work / "graph-result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
