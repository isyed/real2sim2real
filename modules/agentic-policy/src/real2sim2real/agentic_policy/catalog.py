"""Full author-facing source/tool inventory and explicit installed extensions."""

import inspect
from importlib.metadata import distributions, entry_points
from pathlib import Path

import real2sim2real

from ._vendor.gap_core.tools import ToolRegistry
from ._vendor.geometry import tools as geometry
from .skill_catalog import skill_catalog


def register_library_tools(registry, extensions=()):
    # Register this pinned bundle explicitly, not a process-global plugin queue.
    for name, function in inspect.getmembers(geometry, inspect.isfunction):
        if function.__module__ == geometry.__name__ and not name.startswith("_"):
            registry.register_callable(
                "geometry." + name,
                function,
                summary=inspect.getdoc(function) or name,
                requires_ctx=False,
            )
    installed = list(entry_points(group="real2sim2real.agentic_tools"))
    for name in extensions:
        matches = [e for e in installed if e.name == name]
        if len(matches) != 1:
            raise ValueError("Unknown or ambiguous tool extension: " + name)
        before = set(registry.runtime_tools())
        matches[0].load()(registry)
        added = set(registry.runtime_tools()) - before
        if any(n.startswith(("robot.", "sim.")) for n in added):
            raise ValueError("Library extensions cannot add simulator commands")


def author_catalog(robot_schemas, extensions=()):
    registry = ToolRegistry()
    register_library_tools(registry, extensions)
    tools = {
        name: {**description, "execution": "trusted simulator RPC"}
        for name, description in robot_schemas.items()
    }
    for name, descriptor in registry.runtime_tools().items():
        tools[name] = {
            "description": descriptor.summary,
            "inputs": {
                key: field.type_str for key, field in descriptor.schema.inputs.items()
            },
            "parameters": {
                key: {
                    "type": field.type_str,
                    "required": field.required,
                    "default": field.default,
                }
                for key, field in descriptor.schema.inputs.items()
            },
            "execution": "isolated worker; optional dependencies may be required",
        }
    resources = Path(__file__).parent / "resources"
    source_roots = [str(Path(p).resolve()) for p in real2sim2real.__path__]
    source_files = sorted(
        str(p) for root in source_roots for p in Path(root).rglob("*.py")
    )
    access = {
        "skills": skill_catalog(tools),
        "source_roots": source_roots,
        "project_source_archive": str(resources / "project-source.zip")
        if (resources / "project-source.zip").is_file()
        else None,
        "project_source_index": str(resources / "project-source.json")
        if (resources / "project-source.json").is_file()
        else None,
        "python_sources": source_files,
        "enabled_tool_extensions": list(extensions),
        "available_tool_extensions": sorted(
            e.name for e in entry_points(group="real2sim2real.agentic_tools")
        ),
        "access": "Read installed source, search files, run Python/shell and reuse algorithms. Write policy and scratch files in the author workspace.",
        "limits": "Only explicitly authorized campaign inputs are supplied; a private manual seed may be included when declared by the campaign. Held-out evaluation data and live MuJoCo objects are not exposed. Source access does not enable unregistered simulator commands. Missing models/dependencies are not silently provisioned.",
    }
    access["distributions"] = sorted(
        [
            {"name": d.metadata["Name"], "version": d.version}
            for d in distributions()
            if d.metadata["Name"].lower().startswith("real2sim2real")
        ],
        key=lambda x: x["name"],
    )
    return tools, access
