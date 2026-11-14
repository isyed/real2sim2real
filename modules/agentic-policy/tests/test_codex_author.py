import pytest
from real2sim2real.agentic_policy.codex_author import validate_runtime


def test_version_executable_without_tool_host_is_insufficient(tmp_path):
    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    with pytest.raises(FileNotFoundError, match="codex-code-mode-host"):
        validate_runtime(binary)
    host = tmp_path / "codex-code-mode-host"
    host.write_text("#!/bin/sh\nexit 0\n")
    with pytest.raises(FileNotFoundError, match="codex-code-mode-host"):
        validate_runtime(binary)
    host.chmod(0o755)
    validate_runtime(binary)
