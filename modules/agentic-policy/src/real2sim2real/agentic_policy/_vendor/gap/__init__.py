"""real2sim2real.agentic_policy._vendor.gap — graph as policy.

Compile language instructions into typed, verified robot skill graphs and
execute them on simulators or real robots.

    import real2sim2real.agentic_policy._vendor.gap
    conn = real2sim2real.agentic_policy._vendor.gap.connector.sim("libero", task="libero_object/0")
    result = real2sim2real.agentic_policy._vendor.gap.execute(graph, conn)        # open-robot-skills auto-discovered
    g = real2sim2real.agentic_policy._vendor.gap.agent.generate("pick up the soup can")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0.dev0"

# Lazy submodule re-exports keep `import real2sim2real.agentic_policy._vendor.gap` light (no torch/mujoco/jax at
# import time). Populated as subsystems land: runtime (P1), connector (P3),
# agent (P5), benchmark/viz (P6).
_LAZY_ATTRS = {
    "execute": "real2sim2real.agentic_policy._vendor.gap.runtime.execute",
    "connector": "real2sim2real.agentic_policy._vendor.gap.connector",
    "agent": "real2sim2real.agentic_policy._vendor.gap.agent",
    "benchmark": "real2sim2real.agentic_policy._vendor.gap.benchmark",
    "viz": "real2sim2real.agentic_policy._vendor.gap.viz",
    "builder": "real2sim2real.agentic_policy._vendor.gap.builder",
    "types": "real2sim2real.agentic_policy._vendor.gap.types",
    "errors": "real2sim2real.agentic_policy._vendor.gap.errors",
    "NodeContext": "real2sim2real.agentic_policy._vendor.gap.runtime.context",
    "CancelToken": "real2sim2real.agentic_policy._vendor.gap.runtime.context",
}

if TYPE_CHECKING:  # pragma: no cover
    pass


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module 'real2sim2real.agentic_policy._vendor.gap' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target)
    if target == f"real2sim2real.agentic_policy._vendor.gap.{name}":
        # Plain submodule re-export (real2sim2real.agentic_policy._vendor.gap.types, real2sim2real.agentic_policy._vendor.gap.builder, ...).
        value = module
    else:
        # Named attribute living in the target module (real2sim2real.agentic_policy._vendor.gap.execute is the
        # `execute` function inside real2sim2real.agentic_policy._vendor.gap.runtime.execute, etc.).
        value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))
