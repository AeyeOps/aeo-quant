"""Command-line interface for the aeo-quant harness.

    aeo-harness start [--format fp8|nvfp4] [--foreground]
                                              # detach by default; streams
                                              # the daemon's load log until
                                              # ready. --foreground runs the
                                              # server loop inline (for
                                              # supervisors / debugging).
    aeo-harness status                        # connect and print server state
    aeo-harness stop                          # shut the daemon down, including
                                              # strays that haven't yet bound
                                              # the socket (mid-load / crashed).

Kept deliberately small — argparse, a few stdlib prints, no Rich, no plugins.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import signal
import sys
import time

from .client import (
    HarnessClient,
    HarnessError,
    HarnessUnavailable,
    spawn_and_wait_for_ready,
    try_connect,
)
from .protocol import PIDFILE_PATH, SOCKET_PATH

# -- stray detection ------------------------------------------------------

# A harness daemon is spawned as
#   <python> -u -m aeo_quant.harness.cli start --foreground
# We match on argv shape — a Python interpreter whose module arg is
# aeo_quant.harness.cli and whose first positional arg is "start".
# Matching on argv tokens avoids false positives from shell wrappers
# whose cmdline happens to contain the same substrings (e.g., a zsh -c
# with our command in its history).


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, not ours — still "alive"
    return True


def _read_argv(pid: int) -> list[str] | None:
    """Return ``/proc/<pid>/cmdline`` split on NULs, or None if unreadable."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    if not raw:
        return None
    return [t.decode(errors="replace") for t in raw.rstrip(b"\x00").split(b"\x00")]


def _argv_is_daemon(argv: list[str]) -> bool:
    """True when ``argv`` invokes ``-m aeo_quant.harness.cli start``."""
    if not argv:
        return False
    # argv[0] must be a python interpreter (loose check — "python" substring)
    if "python" not in os.path.basename(argv[0]):
        return False
    # Look for the -m / module / subcommand sequence.
    try:
        m_idx = argv.index("-m")
    except ValueError:
        return False
    if m_idx + 1 >= len(argv):
        return False
    if argv[m_idx + 1] != "aeo_quant.harness.cli":
        return False
    # After the module name, the first positional must be "start".
    tail = argv[m_idx + 2:]
    for token in tail:
        if token.startswith("-"):
            continue
        return token == "start"
    return False


def _pidfile_pid() -> int | None:
    """Read + validate the pidfile; None if absent, malformed, or mismatched."""
    if not PIDFILE_PATH.exists():
        return None
    try:
        pid = int(PIDFILE_PATH.read_text().strip())
    except (ValueError, OSError):
        return None
    if not _pid_alive(pid):
        return None
    argv = _read_argv(pid)
    if argv is None or not _argv_is_daemon(argv):
        # pid reuse: the number is live but belongs to something else
        return None
    return pid


def _scan_for_stray_daemons() -> list[int]:
    """Return PIDs of harness daemon processes by argv shape, via /proc.

    Used as a fallback when the pidfile is missing or stale; covers the
    case where the daemon was killed without a chance to remove its
    pidfile and socket. Excludes the caller's own PID.
    """
    self_pid = os.getpid()
    pids: list[int] = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid:
            continue
        argv = _read_argv(pid)
        if argv is None:
            continue
        if _argv_is_daemon(argv):
            pids.append(pid)
    return sorted(pids)


def _find_stray_daemons() -> list[int]:
    """Union of pidfile-pointed PID and ps-scan results, de-duplicated."""
    pids: set[int] = set()
    p = _pidfile_pid()
    if p is not None:
        pids.add(p)
    pids.update(_scan_for_stray_daemons())
    return sorted(pids)


def _kill_pid(pid: int, *, grace_s: float = 5.0) -> bool:
    """SIGTERM → wait → SIGKILL. Returns True once the pid is gone."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    time.sleep(0.5)
    return not _pid_alive(pid)


def _cleanup_stale_files() -> bool:
    """Remove leftover pidfile / socket. Returns True if anything was removed."""
    removed = False
    for p in (PIDFILE_PATH, SOCKET_PATH):
        if p.exists():
            with contextlib.suppress(FileNotFoundError):
                p.unlink()
                removed = True
    return removed


# -- commands -------------------------------------------------------------


def _cmd_start(args: argparse.Namespace) -> int:
    if args.format:
        os.environ["QUANT_FORMAT"] = args.format

    if args.foreground:
        # Import lazily — starting the server pulls torch/transformers.
        from .server import run_server
        return run_server()

    # Default: detach a background daemon, tail its log until ready, return.
    if try_connect() is not None:
        print(
            f"[harness] already running on {SOCKET_PATH}; "
            f"use `aeo-harness stop` first to restart.",
            file=sys.stderr,
        )
        return 1

    # A daemon may be running but not yet listening on the socket (mid-load,
    # or crashed before binding). Refuse instead of stacking a second load
    # into the same GB10 unified memory pool.
    strays = _find_stray_daemons()
    if strays:
        pretty = ", ".join(str(p) for p in strays)
        print(
            f"[harness] daemon PID(s) {pretty} still running "
            f"(likely mid-load or crashed); run `aeo-harness stop` first.",
            file=sys.stderr,
        )
        return 1

    try:
        spawn_and_wait_for_ready(preflight_label="aeo_harness_start")
    except HarnessUnavailable as e:
        print(f"[harness] failed to start: {e}", file=sys.stderr)
        return 1
    print(
        "[harness] detached daemon running in the background. "
        "Stop with `aeo-harness stop`.",
    )
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    client = try_connect()
    if client is None:
        print(f"[harness] not running (no socket at {SOCKET_PATH})", file=sys.stderr)
        strays = _find_stray_daemons()
        if strays:
            pretty = ", ".join(str(p) for p in strays)
            print(
                f"[harness] note: daemon PID(s) {pretty} alive but not "
                f"serving yet (mid-load / crashed). "
                f"`aeo-harness stop` will clear them.",
                file=sys.stderr,
            )
        return 1
    try:
        info = client.status()
    except HarnessError as e:
        print(f"[harness] status error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(info, indent=2))
    return 0


def _cmd_stop(_args: argparse.Namespace) -> int:
    # Step 1: if the socket is live, ask for a graceful shutdown first.
    graceful = False
    if SOCKET_PATH.exists():
        c = HarnessClient(str(SOCKET_PATH))
        try:
            info = c.shutdown()
            print(json.dumps(info, indent=2))
            graceful = True
        except HarnessUnavailable:
            pass  # fall through to stray handling
        except HarnessError as e:
            print(f"[harness] shutdown error: {e}", file=sys.stderr)
            # still try stray cleanup

    # Step 2: catch any stray daemon process (mid-load, crashed, pid reuse).
    # Also re-scans after a graceful shutdown because the server may still
    # be finalizing when we return.
    if graceful:
        # Give the daemon a beat to exit on its own before we start killing.
        for _ in range(20):
            if not _find_stray_daemons():
                break
            time.sleep(0.25)

    strays = _find_stray_daemons()
    for pid in strays:
        print(f"[harness] stopping stray daemon PID {pid}", file=sys.stderr)
        if not _kill_pid(pid):
            print(
                f"[harness] PID {pid} did not exit after SIGTERM+SIGKILL",
                file=sys.stderr,
            )
            return 2

    # Step 3: clean up any leftover files the daemon didn't unlink itself.
    removed = _cleanup_stale_files()

    if not graceful and not strays:
        if removed:
            print("[harness] cleaned up stale pidfile/socket", file=sys.stderr)
            return 0
        print("[harness] not running", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="aeo-harness", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser(
        "start",
        help="launch the harness daemon (detached by default; "
             "use --foreground to run inline)",
    )
    p_start.add_argument(
        "--format", choices=["fp8", "nvfp4"],
        help="override QUANT_FORMAT for this run",
    )
    p_start.add_argument(
        "--foreground", action="store_true",
        help="run the server loop inline in this terminal "
             "(for supervisors / debugging); default is to detach",
    )
    p_start.set_defaults(func=_cmd_start)

    sub.add_parser("status", help="print daemon status").set_defaults(func=_cmd_status)
    sub.add_parser(
        "stop",
        help="stop the daemon, including strays that haven't bound the socket",
    ).set_defaults(func=_cmd_stop)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
