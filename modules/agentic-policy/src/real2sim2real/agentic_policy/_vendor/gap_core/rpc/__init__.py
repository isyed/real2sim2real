"""real2sim2real.agentic_policy._vendor.gap_core.rpc — out-of-process tool RPC.

Bundle tools whose SKILL.md declares ``real2sim2real.agentic_policy._vendor.gap.serving.protocol: stdio-msgpack``
run in their own venv (the bundle's per-pyproject .venv). real2sim2real.agentic_policy._vendor.gap-runtime spawns
one persistent server per bundle and calls into it via msgpack frames over
stdio (no shell, no port). The server-side process imports the bundle's
``tools.py`` (against the bundle's own deps) and dispatches by tool name.

Public surface:
    from real2sim2real.agentic_policy._vendor.gap_core.rpc.client import ToolClient
    from real2sim2real.agentic_policy._vendor.gap_core.rpc.server import main as tool_server_main
    from real2sim2real.agentic_policy._vendor.gap_core.rpc.codec import encode_frame, decode_frame
"""
