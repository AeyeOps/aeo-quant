"""Model-held-loaded harness for aeo-quant.

A long-running daemon that loads the Gemma 4 model once and serves workload
requests from multiple clients over a UNIX domain socket. Each example script
auto-detects the harness and uses it if running; otherwise it loads the model
in-process as before.

Start: ``aeo-harness start --format {fp8|nvfp4}``
Stop:  ``aeo-harness stop``
"""

from .client import (
    HarnessClient,
    HarnessUnavailable,
    get_or_start_harness,
    spawn_and_wait_for_ready,
    try_connect,
)

__all__ = [
    "HarnessClient",
    "HarnessUnavailable",
    "get_or_start_harness",
    "spawn_and_wait_for_ready",
    "try_connect",
]
