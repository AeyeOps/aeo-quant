"""Client for the aeo-quant harness.

Usage from an example script::

    from aeo_quant.harness import try_connect
    client = try_connect()            # None if harness not running
    if client:
        result = client.run_workload("parity", kv_bits=4, gen_tokens=50)
    else:
        # in-process fallback — load model locally as before
        ...
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .protocol import (
    METHOD_RUN_WORKLOAD,
    METHOD_SHUTDOWN,
    METHOD_STATUS,
    SOCKET_PATH,
    STATUS_ERROR,
    STATUS_EVENT,
    STATUS_OK,
)

HARNESS_LOG_PATH = Path.home() / ".aeo-quant" / "harness.log"


def _default_event_printer(event: dict) -> None:
    """Fallback handler for streaming events — print human-readable messages.

    Workloads should include a ``message`` field for human consumption; any
    other fields are included in a compact tail for context. Callers who
    want structured handling pass their own ``on_event`` to ``run_workload``.
    """
    msg = event.get("message")
    if msg:
        print(f"[event] {msg}", flush=True)
        return
    # No message field — dump the event compactly so nothing is lost.
    print(f"[event] {json.dumps(event, ensure_ascii=False)}", flush=True)


class HarnessUnavailable(Exception):
    """Raised when the harness socket exists but is not responding."""


class HarnessError(RuntimeError):
    """Raised when a harness method returns an error reply."""


class HarnessClient:
    """Blocking UNIX-socket client for the harness daemon.

    Each method call opens a fresh short-lived connection, sends one request,
    reads one reply, and closes. Keeps the wire protocol trivial and avoids
    any shared state between calls.

    ``connect_timeout`` bounds the initial socket connect. Method calls use
    ``request_timeout`` for both send and recv — set it generously for
    workloads that take minutes.
    """

    def __init__(self, socket_path: str, *, connect_timeout: float = 1.0,
                 request_timeout: float = 900.0) -> None:
        self.socket_path = socket_path
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout

    def _call(
        self,
        method: str,
        *,
        on_event: Any | None = None,
        **kwargs,
    ) -> dict:
        """Send one request, read a stream of reply lines.

        The server may emit zero or more ``status: event`` lines before the
        terminal ``status: ok`` or ``status: error`` line. Each event is
        dispatched to ``on_event`` (or printed by default).
        """
        req_id = uuid.uuid4().hex[:8]
        payload = {"id": req_id, "method": method, "kwargs": kwargs}
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode()

        handler = on_event if on_event is not None else _default_event_printer
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(self.connect_timeout)
            s.connect(self.socket_path)
            s.settimeout(self.request_timeout)
            s.sendall(line)
            # Use a file-like wrapper so we can read one JSON object per line
            # without re-implementing buffering.
            reader = s.makefile("rb")
            while True:
                raw = reader.readline()
                if not raw:
                    raise HarnessUnavailable("harness closed socket before reply")
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise HarnessUnavailable(f"bad json from harness: {e}") from e
                status = msg.get("status")
                if status == STATUS_EVENT:
                    event = msg.get("event") or {}
                    # A broken handler must not kill the stream — swallow.
                    with contextlib.suppress(Exception):
                        handler(event)
                    continue
                if status == STATUS_OK:
                    return msg.get("result", {})
                if status == STATUS_ERROR:
                    raise HarnessError(msg.get("error", "unknown error"))
                raise HarnessUnavailable(f"unknown status from harness: {status!r}")
        except (ConnectionRefusedError, FileNotFoundError) as e:
            raise HarnessUnavailable(f"harness socket not responding: {e}") from e
        except TimeoutError as e:
            raise HarnessUnavailable(f"harness timed out: {e}") from e
        finally:
            with contextlib.suppress(Exception):
                s.close()

    def status(self) -> dict:
        return self._call(METHOD_STATUS)

    def shutdown(self) -> dict:
        return self._call(METHOD_SHUTDOWN)

    def run_workload(
        self,
        name: str,
        *,
        on_event: Any | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run a workload on the harness. Events stream to ``on_event`` (or stdout)."""
        return self._call(
            METHOD_RUN_WORKLOAD,
            on_event=on_event,
            workload=name,
            **kwargs,
        )


def try_connect(*, connect_timeout: float = 0.5) -> HarnessClient | None:
    """Return a HarnessClient if the daemon is running and responding, else None.

    Two failure modes are silent (return None): socket file missing, or
    ConnectionRefused on connect. Any other transport error raises
    ``HarnessUnavailable`` so the caller knows the daemon is broken and
    shouldn't just silently run in-process.
    """
    if not SOCKET_PATH.exists():
        return None
    c = HarnessClient(str(SOCKET_PATH), connect_timeout=connect_timeout)
    try:
        c.status()
    except HarnessUnavailable:
        return None
    return c


def _spawn_detached_daemon() -> subprocess.Popen:
    """Start the harness daemon in the background, detached from this process.

    Inherits the current env (QUANT_FORMAT, CHECKPOINT, etc.) so the daemon
    loads the format the caller asked for. stdout/stderr go to harness.log.
    Uses ``start_new_session`` so the child survives the parent's exit.
    Passes ``-u`` so the child's stdout is unbuffered — otherwise tail-to-
    terminal would see nothing until the daemon's buffers flushed.

    The child invocation passes ``--foreground`` because the child *is* the
    detached daemon — it runs the server loop directly rather than
    re-detaching and looping forever.
    """
    HARNESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(HARNESS_LOG_PATH, "ab")  # noqa: SIM115 — passed to child
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "aeo_quant.harness.cli",
             "start", "--foreground"],
            stdout=log_fh,
            stderr=log_fh,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            env=os.environ.copy(),
        )
    finally:
        log_fh.close()
    return proc


def spawn_and_wait_for_ready(
    *,
    preflight_min_free_gb: float | None = None,
    preflight_label: str = "harness_spawn",
    wait_s: float = 300.0,
    verbose: bool = True,
) -> HarnessClient:
    """Detach the daemon, tail its startup log, and return a connected client.

    Used by both ``get_or_start_harness`` (auto-spawn when a client finds no
    running daemon) and the ``aeo-harness start`` CLI (explicit user-invoked
    spawn). Handles preflight, log tail, readiness polling, and timeout.

    Raises ``HarnessUnavailable`` if the daemon doesn't come up within
    ``wait_s`` seconds, or if it exits during startup.
    """
    if preflight_min_free_gb is None:
        preflight_min_free_gb = float(os.environ.get("HARNESS_MIN_FREE_GB", "60"))
    from aeo_quant.gpu.memory import preflight_memory
    preflight_memory(preflight_min_free_gb, label=preflight_label)

    start_pos = HARNESS_LOG_PATH.stat().st_size if HARNESS_LOG_PATH.exists() else 0

    if verbose:
        print(
            "[harness] spawning detached background process", flush=True,
        )
        print(f"[harness] log: {HARNESS_LOG_PATH}", flush=True)
    proc = _spawn_detached_daemon()
    if verbose:
        print(
            f"[harness] daemon PID {proc.pid} — streaming startup output below "
            f"(timeout {wait_s:.0f}s):",
            flush=True,
        )

    stop_tail = threading.Event()
    tail_thread: threading.Thread | None = None
    if verbose:
        tail_thread = threading.Thread(
            target=_tail_log_to_stdout,
            args=(start_pos, stop_tail),
            name="harness-log-tail",
            daemon=True,
        )
        tail_thread.start()

    try:
        deadline = time.monotonic() + wait_s
        while True:
            if time.monotonic() >= deadline:
                raise HarnessUnavailable(
                    f"daemon did not become ready within {wait_s:.0f}s; "
                    f"check {HARNESS_LOG_PATH}"
                )
            rc = proc.poll()
            if rc is not None:
                raise HarnessUnavailable(
                    f"daemon exited with code {rc} before becoming ready; "
                    f"check {HARNESS_LOG_PATH}"
                )
            client = try_connect(connect_timeout=0.5)
            if client is not None:
                time.sleep(0.3)
                if verbose:
                    elapsed = wait_s - (deadline - time.monotonic())
                    print(f"\n[harness] daemon ready after {elapsed:.1f}s", flush=True)
                return client
            time.sleep(1.0)
    finally:
        stop_tail.set()
        if tail_thread is not None:
            tail_thread.join(timeout=1.0)


def _tail_log_to_stdout(
    start_pos: int, stop_event: threading.Event, prefix: str = "    "
) -> None:
    """Stream new bytes from HARNESS_LOG_PATH to stdout until stop_event is set.

    Runs in a background thread. Indents each line so daemon output is
    visually distinct from the client's own prints. Handles tqdm's carriage
    returns correctly because we write raw bytes — the TTY rewrites the line.
    """
    deadline = time.monotonic() + 10.0
    while not HARNESS_LOG_PATH.exists():
        if stop_event.is_set() or time.monotonic() > deadline:
            return
        time.sleep(0.1)

    # Write a per-line prefix by inserting it after each \n/\r. tqdm uses \r
    # for in-place updates; we leave those alone so the terminal rewrites.
    prefix_bytes = prefix.encode()
    out = sys.stdout.buffer
    need_prefix = True

    try:
        with open(HARNESS_LOG_PATH, "rb") as f:
            f.seek(start_pos)
            while not stop_event.is_set():
                chunk = f.read(65536)
                if not chunk:
                    time.sleep(0.1)
                    continue
                # Insert prefix after newlines / carriage returns so the
                # daemon's output visibly belongs to the harness, not the
                # client's own prints.
                parts = []
                i = 0
                for j, byte in enumerate(chunk):
                    if need_prefix and byte not in (0x0A, 0x0D):  # \n, \r
                        parts.append(prefix_bytes)
                        need_prefix = False
                    if byte in (0x0A, 0x0D):
                        parts.append(chunk[i : j + 1])
                        i = j + 1
                        need_prefix = True
                if i < len(chunk):
                    parts.append(chunk[i:])
                out.write(b"".join(parts))
                out.flush()
    except Exception:
        # Never let the tail thread crash the client — just stop streaming.
        return


def get_or_start_harness(
    *,
    preflight_min_free_gb: float | None = None,
    preflight_label: str = "harness_spawn",
    wait_s: float = 300.0,
    verbose: bool = True,
) -> HarnessClient:
    """Return a connected client, auto-spawning the daemon if not running.

    The spawned daemon is detached: it survives after this process exits
    and must be stopped explicitly with ``aeo-harness stop``.

    A memory-headroom preflight runs **only when this call is about to
    spawn a new daemon** — a thin client reconnecting to an
    already-loaded harness doesn't need the headroom the fresh load
    would. The threshold is read from the same env var the daemon itself
    uses, ``HARNESS_MIN_FREE_GB`` (default 60 GB); explicit
    ``preflight_min_free_gb`` overrides the env for callers that need a
    different bar.

    During the spawn wait, the daemon's own log is streamed to this process's
    stdout — the user sees model-load progress in real time rather than a
    generic "waiting..." message.

    Raises ``HarnessUnavailable`` if the daemon doesn't come up within
    ``wait_s`` seconds, or if it exits during startup (check the log at
    ``HARNESS_LOG_PATH`` for the reason).
    """
    client = try_connect()
    if client is not None:
        return client
    if verbose:
        print("[harness] no daemon running", flush=True)
    return spawn_and_wait_for_ready(
        preflight_min_free_gb=preflight_min_free_gb,
        preflight_label=preflight_label,
        wait_s=wait_s,
        verbose=verbose,
    )
