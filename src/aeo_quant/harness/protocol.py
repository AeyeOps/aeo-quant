"""Wire protocol for the aeo-quant harness.

Newline-delimited JSON over a UNIX domain socket. Each side writes one JSON
object per line and reads one JSON object per line. No framing, no headers,
no schema versioning — the daemon and client are always deployed together
from the same tree.

Request shape::

    {"method": str, "id": str, "kwargs": dict}

Reply shape (success)::

    {"id": str, "status": "ok", "result": dict}

Reply shape (error)::

    {"id": str, "status": "error", "error": str}
"""

from __future__ import annotations

import os
from pathlib import Path

# ~/.aeo-quant/harness.sock — one user, one harness at a time.
_env_sock = os.environ.get("AEO_HARNESS_SOCKET")
SOCKET_PATH = Path(_env_sock) if _env_sock else Path.home() / ".aeo-quant" / "harness.sock"
PIDFILE_PATH = SOCKET_PATH.with_suffix(".pid")

# Methods the server understands.
METHOD_STATUS = "status"
METHOD_SHUTDOWN = "shutdown"
METHOD_RUN_WORKLOAD = "run_workload"

STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_EVENT = "event"  # streamed progress, multiple may precede an ok/error
