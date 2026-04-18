"""Command-line interface for the aeo-quant harness.

    aeo-harness start [--format fp8|nvfp4]   # foreground; loads model, listens
    aeo-harness status                        # connect and print server state
    aeo-harness stop                          # connect and ask server to exit

Kept deliberately small — argparse, a few stdlib prints, no Rich, no plugins.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .client import HarnessClient, HarnessError, HarnessUnavailable, try_connect
from .protocol import SOCKET_PATH


def _cmd_start(args: argparse.Namespace) -> int:
    # Import lazily — starting the server pulls torch/transformers.
    if args.format:
        os.environ["QUANT_FORMAT"] = args.format
    from .server import run_server

    return run_server()


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
        help="load model and serve in the foreground (Ctrl+C or "
             "`aeo-harness stop` from another shell to exit)",
    )
    p_start.add_argument(
        "--format", choices=["fp8", "nvfp4"],
        help="override QUANT_FORMAT for this run",
    )
    p_start.set_defaults(func=_cmd_start)

    sub.add_parser("status", help="print daemon status").set_defaults(func=_cmd_status)
    sub.add_parser("stop", help="tell the daemon to exit").set_defaults(func=_cmd_stop)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
