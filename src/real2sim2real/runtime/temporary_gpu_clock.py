"""Temporarily cap GPU core clocks; reset default clocks on completion/timeout.

Run in a separate privileged terminal. GPU evaluation remains a normal-user job.
This helper never changes GPU power, memory clocks, or CPU settings.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def clock_command(*arguments):
    result = subprocess.run(
        ["/usr/bin/nvidia-smi", "-i", "0", *arguments],
        capture_output=True,
        text=True,
        timeout=15,
    )
    output = result.stdout + result.stderr
    print(output, end="", flush=True)
    # A successful laptop clock change may also print a persistence-mode warning.
    # That warning does not mean the clock request failed.
    if result.returncode or any(
        word in output.lower()
        for word in ["not supported", "not have permission", "error:"]
    ):
        raise RuntimeError(f"Clock command failed or was not applied: {output.strip()}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--mhz", type=int, default=1500)
    parser.add_argument("--seconds", type=int, default=2700)
    parser.add_argument(
        "--wait-for-session",
        type=Path,
        help="Wait for another bounded clock helper to restore before applying this cap",
    )
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("Administrator access is required for GPU clock control")
    if args.mhz <= 0 or not 0 < args.seconds <= 28800:
        parser.error(
            "Use a positive clock and a duration of at most 28800 seconds (eight hours)"
        )
    folder = args.session_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    # The completion marker is written by the normal-user evaluation process.
    # A directory newly created through sudo must remain writable by that user.
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if (
        sudo_uid
        and sudo_gid
        and folder.stat().st_uid == 0
        and folder.parent.stat().st_uid == int(sudo_uid)
    ):
        os.chown(folder, int(sudo_uid), int(sudo_gid))
    done, status = folder / "clock_session.done", folder / "clock_session.json"
    if done.exists():
        raise RuntimeError(f"Stale completion marker: {done}")
    record = dict(
        requested_max_clock_mhz=args.mhz,
        max_seconds=args.seconds,
        started_epoch=time.time(),
        status="starting",
        restore="reset_default_gpu_clocks",
    )

    def save():
        status.write_text(json.dumps(record, indent=2) + "\n")
        status.chmod(0o644)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    save()
    applied = False
    try:
        if args.wait_for_session is not None:
            previous = args.wait_for_session.resolve()
            if previous == folder or not (previous / "clock_session.json").is_file():
                raise RuntimeError(
                    "Require a different existing clock session for the handoff"
                )
            from_clock_start = time.monotonic()
            record.update(
                status="waiting_for_previous_session", previous_session=str(previous)
            )
            save()
            print(
                f"Waiting for {previous} to restore; no clocks changed. Leave this terminal open.",
                flush=True,
            )
            while time.monotonic() - from_clock_start < args.seconds:
                prior = json.loads((previous / "clock_session.json").read_text())
                if prior.get("status") == "restored":
                    # A restored record is written only after the reset command.
                    # Give the original helper time to exit before a new cap.
                    time.sleep(2)
                    break
                if prior.get("status") == "restore_failed":
                    raise RuntimeError(
                        "Previous clock restoration failed; manual review required"
                    )
                time.sleep(1)
            else:
                raise TimeoutError(
                    "Previous GPU session did not finish within the bounded handoff window"
                )
            args.seconds = max(
                1, int(args.seconds - (time.monotonic() - from_clock_start))
            )
            record["max_seconds"] = args.seconds
        record["driver_response"] = clock_command(f"--lock-gpu-clocks=0,{args.mhz}")
        applied = True
        record.update(status="active", started_epoch=time.time())
        save()
        print(
            f"Maximum GPU clock requested: {args.mhz} MHz. Leave this terminal open; "
            f"default GPU clocks will be restored when evaluation finishes or after {args.seconds / 60:g} minutes.",
            flush=True,
        )
        deadline = time.monotonic() + args.seconds
        while (
            time.monotonic() < deadline
            and time.time() < record["started_epoch"] + args.seconds
            and not done.exists()
        ):
            time.sleep(1)
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        save()
        raise
    finally:
        if applied:
            try:
                record["restore_response"] = clock_command("--reset-gpu-clocks")
                record["status"] = "restored"
            except Exception as exc:
                record.update(status="restore_failed", restore_error=str(exc))
                raise
            finally:
                record["finished_epoch"] = time.time()
                save()


if __name__ == "__main__":
    main()
