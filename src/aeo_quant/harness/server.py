"""aeo-quant harness daemon.

Loads the Gemma 4 model once, listens on a UNIX domain socket, and serves
workload requests from a FIFO queue via a single worker task. The queue is
the lock: one ``model.generate()`` runs at a time, arriving clients wait
their turn, and there is no shared mutable state across requests.

Concurrency properties:
    - Many simultaneous client connections (each handled by an asyncio task).
    - Compute serialized through a single ``asyncio.Queue`` + worker.
    - No deadlock possible: one queue, no nested locks.
    - No multi-writer: each request owns its own KV cache for the duration
      of its dispatch call.
    - Unbounded queue — for a handful of cooperating test clients, a broken
      flooder is the only failure mode and restart clears it.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from aeo_quant.core.config import load_dotenv, quant_env, setup_cuda_allocator
from aeo_quant.gpu.memory import mem_report, preflight_memory
from aeo_quant.workloads import WORKLOADS

from .protocol import (
    METHOD_RUN_WORKLOAD,
    METHOD_SHUTDOWN,
    METHOD_STATUS,
    PIDFILE_PATH,
    SOCKET_PATH,
    STATUS_ERROR,
    STATUS_EVENT,
    STATUS_OK,
)

# Sized to the worst-case peak of any single workload (multi_turn).
# NVFP4 loads in ~20 GB and FP8 in ~27 GB, both well under this gate;
# the larger budget covers headroom for inference KV growth. Override
# via HARNESS_MIN_FREE_GB to tune.
MIN_FREE_GB = float(os.environ.get("HARNESS_MIN_FREE_GB", "60"))


@dataclass
class Job:
    req: dict
    done: asyncio.Event = field(default_factory=asyncio.Event)
    result: dict | None = None
    # Streaming: workloads may push event dicts here during execution. The
    # connection handler drains this queue and writes each event to the
    # requesting client's socket. The worker pushes a ``None`` sentinel
    # when the workload finishes so the drain task knows to stop.
    events: asyncio.Queue = field(default_factory=asyncio.Queue)


@dataclass
class ServerState:
    model: Any = None
    tokenizer: Any = None
    quant_format: str = ""
    checkpoint: str = ""
    kv_bits: int = 0
    started_at: float = 0.0
    jobs_served: int = 0
    queue: asyncio.Queue[Job] = field(default_factory=asyncio.Queue)
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)


def _write_pidfile() -> None:
    PIDFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE_PATH.write_text(f"{os.getpid()}\n")


def _remove_pidfile() -> None:
    with contextlib.suppress(FileNotFoundError):
        PIDFILE_PATH.unlink()


def _remove_stale_socket() -> None:
    # A prior unclean shutdown may have left the socket file in place.
    # bind() would fail with "Address already in use" — remove it first.
    with contextlib.suppress(FileNotFoundError):
        SOCKET_PATH.unlink()


def _reply(writer: asyncio.StreamWriter, payload: dict) -> None:
    line = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
    writer.write(line)


async def _handle_connection(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, state: ServerState
) -> None:
    try:
        raw = await reader.readline()
        if not raw:
            return
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as e:
            _reply(writer, {"id": "?", "status": STATUS_ERROR, "error": f"bad json: {e}"})
            await writer.drain()
            return

        req_id = req.get("id", "?")
        method = req.get("method")

        if method == METHOD_STATUS:
            _reply(writer, {
                "id": req_id,
                "status": STATUS_OK,
                "result": {
                    "quant_format": state.quant_format,
                    "checkpoint": state.checkpoint,
                    "kv_bits": state.kv_bits,
                    "uptime_s": round(time.time() - state.started_at, 1),
                    "jobs_served": state.jobs_served,
                    "queue_depth": state.queue.qsize(),
                    "workloads": sorted(WORKLOADS.keys()),
                },
            })
        elif method == METHOD_SHUTDOWN:
            _reply(writer, {"id": req_id, "status": STATUS_OK, "result": {"shutting_down": True}})
            await writer.drain()
            state.shutdown_event.set()
            return
        elif method == METHOD_RUN_WORKLOAD:
            job = Job(req=req)
            await state.queue.put(job)
            # Drain streaming events onto the socket until the worker signals
            # completion with a None sentinel. The drain runs concurrently
            # with the worker; because asyncio is single-threaded and the
            # worker runs in an executor, events flow responsively.
            while True:
                event = await job.events.get()
                if event is None:
                    break
                _reply(writer, {"id": req_id, "status": STATUS_EVENT, "event": event})
                await writer.drain()
            await job.done.wait()
            assert job.result is not None
            _reply(writer, {"id": req_id, **job.result})
        else:
            _reply(writer, {
                "id": req_id,
                "status": STATUS_ERROR,
                "error": f"unknown method: {method}",
            })

        await writer.drain()
    finally:
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()


async def _worker(state: ServerState) -> None:
    """Single consumer of the job queue. All GPU work happens here.

    The workload runs in a thread-pool executor so the asyncio event loop
    stays responsive — concurrent status requests are served, and the
    connection handler can drain streaming events from ``job.events``.
    """
    loop = asyncio.get_running_loop()
    while True:
        job = await state.queue.get()
        emit_sentinel_needed = True
        try:
            kwargs = job.req.get("kwargs", {}) or {}
            name = kwargs.pop("workload", None) or job.req.get("workload")
            if not name:
                job.result = {"status": STATUS_ERROR, "error": "missing 'workload' name"}
                continue
            if name not in WORKLOADS:
                job.result = {
                    "status": STATUS_ERROR,
                    "error": f"unknown workload: {name!r}. known: {sorted(WORKLOADS)}",
                }
                continue
            fn = WORKLOADS[name]

            # Thread-safe emitter: the workload runs in an executor thread,
            # but asyncio.Queue.put_nowait must be called on the event loop.
            # Bind ``job`` as a default arg so the closure captures this
            # iteration's value rather than the loop variable.
            def emit(event: dict, _job: Job = job) -> None:
                loop.call_soon_threadsafe(_job.events.put_nowait, event)

            kwargs["emit"] = emit
            t = time.time()
            try:
                result = await loop.run_in_executor(
                    None,
                    functools.partial(fn, state.model, state.tokenizer, **kwargs),
                )
            except TypeError as e:
                job.result = {"status": STATUS_ERROR, "error": f"bad kwargs: {e}"}
                continue
            except Exception as e:
                import traceback as _tb
                _tb.print_exc()
                job.result = {
                    "status": STATUS_ERROR,
                    "error": f"workload {name!r} raised: {type(e).__name__}: {e}",
                }
                continue
            state.jobs_served += 1
            job.result = {
                "status": STATUS_OK,
                "result": {
                    **result,
                    "_worker_elapsed_s": round(time.time() - t, 3),
                    "_quant_format": state.quant_format,
                },
            }
        finally:
            # Signal the drain task in the connection handler that streaming
            # is complete — it will then read job.result and send the final
            # ok/error line to the client.
            if emit_sentinel_needed:
                loop.call_soon_threadsafe(job.events.put_nowait, None)
            job.done.set()
            state.queue.task_done()


def _load_model(state: ServerState) -> None:
    from transformers import AutoTokenizer

    from aeo_quant.bridges.gemma4.loader import load_gemma4

    tokenizer_id = os.environ.get("TOKENIZER_ID", "google/gemma-4-26B-A4B-it")
    print(f"[harness] loading tokenizer: {tokenizer_id}", flush=True)
    state.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    print(
        f"[harness] loading model from {state.checkpoint} (format={state.quant_format})",
        flush=True,
    )
    t = time.time()
    state.model = load_gemma4(state.checkpoint, quant_format=state.quant_format)
    print(f"[harness] model loaded in {time.time() - t:.1f}s", flush=True)
    mem_report("harness:after model load")


async def _main_async(state: ServerState) -> int:
    _remove_stale_socket()
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)

    async def on_connect(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        await _handle_connection(r, w, state)

    server = await asyncio.start_unix_server(on_connect, path=str(SOCKET_PATH))
    # Owner-only socket — no cross-user access.
    os.chmod(SOCKET_PATH, 0o600)

    worker_task = asyncio.create_task(_worker(state), name="harness-worker")
    _write_pidfile()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, state.shutdown_event.set)

    print(f"[harness] ready on {SOCKET_PATH}", flush=True)
    print(f"[harness] workloads: {sorted(WORKLOADS)}", flush=True)
    print(
        "[harness] running in the foreground — Ctrl+C or "
        "`aeo-harness stop` from another shell to exit.",
        flush=True,
    )

    try:
        await state.shutdown_event.wait()
        print("[harness] shutting down...", flush=True)
    finally:
        server.close()
        await server.wait_closed()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
        _remove_stale_socket()
        _remove_pidfile()
        print(f"[harness] stopped. jobs_served={state.jobs_served}", flush=True)
    return 0


def run_server() -> int:
    """Entry point for ``aeo-harness start``."""
    load_dotenv()
    setup_cuda_allocator()

    quant_format, checkpoint, kv_bits = quant_env()
    preflight_memory(MIN_FREE_GB, label="harness")

    # Refuse to start a second daemon with the same socket.
    if SOCKET_PATH.exists():
        print(
            f"[FATAL] socket already exists at {SOCKET_PATH} — another harness may be running.\n"
            f"        run `aeo-harness stop` first, or remove the socket manually if stale.",
            file=sys.stderr,
        )
        return 2

    state = ServerState(
        quant_format=quant_format,
        checkpoint=str(checkpoint),
        kv_bits=kv_bits,
        started_at=time.time(),
    )
    _load_model(state)

    try:
        return asyncio.run(_main_async(state))
    except KeyboardInterrupt:
        return 0
