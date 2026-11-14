"""Optional Codex CLI author adapter; invoked only inside the host sandbox.

Credentials are mounted separately for provider authentication, never copied to
the packet or candidate. This adapter does not install or select a provider.
"""

import argparse
import os
import subprocess
from pathlib import Path


def validate_runtime(binary):
    """Check the complete supported CLI layout before spending an author call.

    A successful `codex --version` alone does not exercise the tool host. Mount
    the CLI distribution directory, not only its main executable.
    """
    binary = Path(binary)
    for executable in [binary, binary.with_name("codex-code-mode-host")]:
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(
                f"Required author executable unavailable: {executable}. "
                "Mount the complete pinned CLI distribution read-only."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--auth-file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    validate_runtime(args.binary)
    # HOME and /tmp belong to the bubblewrap namespace, not the host account.
    provider_home = Path("/tmp/r2s2r-author-provider")
    provider_home.mkdir(exist_ok=True, mode=0o700)
    link = provider_home / "auth.json"
    if not link.exists():
        link.symlink_to(Path(args.auth_file).resolve())
    env = {**os.environ, "CODEX_HOME": str(provider_home)}
    if args.check_only:
        subprocess.run([args.binary, "--version"], env=env, check=True)
        assert link.is_file() and (args.workspace / "tools.json").is_file()
        print(
            "Provider binary, authentication mount and packet are available; no model call made."
        )
        return
    command = [
        args.binary,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "--model",
        args.model,
        "--json",
        "--cd",
        str(args.workspace),
        "--output-last-message",
        str(args.workspace / "author-last-message.txt"),
        (
            "Read INSTRUCTIONS.md, EVALUATION.json, CODE_ACCESS.json, skills.json and tools.json. "
            "Produce this one proposal in candidate/. You are inside an externally isolated sandbox. "
            "Use the available source and tools for planning; do not make additional model calls, "
            "download dependencies, publish files or run extra dynamics trials. Provider authentication "
            "files are not task inputs: never inspect, print or copy them. Do not spawn other authors. "
            "No other remote service is authorized. Never copy packet data into candidate/ unnecessarily."
        ),
    ]
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
