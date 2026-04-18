"""Command-line interface for the aeo-quant harness.

    aeo-harness start [--format fp8|nvfp4] [--foreground]
                                              # detach by default; streams
                                              # the daemon's load log until
                                              # ready. --foreground runs the
                                              # server loop inline (for
                                              # supervisors / debugging).
    aeo-harness status                        # connect and print server state
    aeo-harness stop                          # connect and ask server to exit

Kept deliberately small — argparse, a few stdlib prints, no Rich, no plugins.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .client import (
    HarnessClient,
    HarnessError,
    HarnessUnavailable,
    spawn_and_wait_for_ready,
    try_connect,
)
from .protocol import SOCKET_PATH


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
        return 1
    try:
        info = client.status()
    except HarnessError as e:
        print(f"[harness] status error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(info, indent=2))
    return 0


def _cmd_stop(_args: argparse.Namespace) -> int:
    if not SOCKET_PATH.exists():
        print(f"[harness] not running (no socket at {SOCKET_PATH})", file=sys.stderr)
        return 1
    c = HarnessClient(str(SOCKET_PATH))
    try:
        info = c.shutdown()
    except HarnessUnavailable as e:
        print(f"[harness] could not reach daemon: {e}", file=sys.stderr)
        return 2
    except HarnessError as e:
        print(f"[harness] shutdown error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(info, indent=2))
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
    sub.add_parser("stop", help="tell the daemon to exit").set_defaults(func=_cmd_stop)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
