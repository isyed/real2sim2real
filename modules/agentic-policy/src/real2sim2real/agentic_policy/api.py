"""Stable imports for candidate scripts and optional integration adapters."""

from ._vendor.gap.runtime.context import NodeContext
from ._vendor.gap_core.tools import ToolRegistry, tool

__all__ = ["NodeContext", "ToolRegistry", "tool"]
