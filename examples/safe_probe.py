#!/usr/bin/env python3
"""Run a kernel probe under the safety harness.

Wraps aeo_quant.gpu.kernel_probe.run_isolated with a CLI so probes can
be launched from the shell.  Each probe runs in a fresh subprocess with
a hard timeout and pre/post GPU snapshots.

Usage::

    uv run python examples/safe_probe.py PROBE [ARGS...] \\
        [--timeout SECONDS] [--min-free-gb GB]

Examples::

    uv run python examples/safe_probe.py examples/probe_nvfp4_torchao.py
    uv run python examples/safe_probe.py examples/fp4_probe.py --timeout 30
"""
from __future__ import annotations

import argparse
import sys

from aeo_quant.gpu.kernel_probe import run_isolated


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run a GPU kernel probe in a sandboxed subprocess.",
    )
    p.add_argument("probe", help="path to probe script")
    p.add_argument("args", nargs="*", help="args forwarded to the probe")
    p.add_argument("--timeout", type=int, default=60,
                   help="hard-kill subprocess after this many seconds (default: 60)")
    p.add_argument("--min-free-gb", type=float, default=5.0,
                   help="refuse to launch if host RAM free < this (default: 5.0)")
    ns = p.parse_args()

    try:
        result = run_isolated(
            ns.probe,
            *ns.args,
            timeout_s=ns.timeout,
            min_free_gb=ns.min_free_gb,
        )
    except RuntimeError as e:
        print(f"[FATAL] preflight refused: {e}", file=sys.stderr)
        return 2

    print(result.summary())
    print("--- STDOUT ---")
    print(result.stdout)
    if result.stderr:
        print("--- STDERR ---")
        print(result.stderr, file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
