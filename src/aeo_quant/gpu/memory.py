"""Unified memory monitoring, cap enforcement, and GPU timing for inference.

Provides memory reporting, cap enforcement, CUDA-event timing, and a
HuggingFace-compatible stopping criteria that halts generation when a
memory cap is exceeded.
All cap values are passed as explicit parameters -- no module-level globals.
"""
from __future__ import annotations

import psutil
import torch

_GB = 1024**3


class MemoryCapExceeded(Exception):
    """Raised when unified memory usage exceeds the configured cap."""
    pass


def gb(n_bytes: int) -> str:
    """Format a byte count as a human-readable GB string."""
    return f"{n_bytes / _GB:6.2f} GB"


class CudaTimer:
    """Context manager for GPU-accurate timing via CUDA events.

    Uses ``torch.cuda.Event`` with ``enable_timing=True`` so the
    measurement reflects actual GPU execution, not CPU-side wall clock.

    Usage::

        with CudaTimer("prefill") as t:
            model.generate(...)
        print(f"prefill: {t.elapsed_ms:.1f} ms")
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_ms = self.start_event.elapsed_time(self.end_event)


def mem_report(label: str) -> dict:
    """Print memory status and return numeric values for JSONL recording."""
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    t_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    t_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    print(
        f"[mem] {label:<32} "
        f"sys_used={gb(vm.used)}  "
        f"sys_avail={gb(vm.available)}  "
        f"proc_rss={gb(rss)}  "
        f"torch_alloc={gb(t_alloc)}  "
        f"torch_peak={gb(t_peak)}",
        flush=True,
    )
    return {
        "sys_total_gb": round(vm.total / _GB, 2),
        "sys_used_gb": round(vm.used / _GB, 2),
        "torch_alloc_gb": round(t_alloc / _GB, 2),
        "torch_peak_gb": round(t_peak / _GB, 2),
    }


def enforce_cap(label: str, cap_gb: float) -> None:
    """Raise :class:`MemoryCapExceeded` if system memory exceeds *cap_gb*.

    Args:
        label: Human-readable checkpoint name for the error message.
        cap_gb: Memory cap in gigabytes.
    """
    vm = psutil.virtual_memory()
    if vm.used > cap_gb * _GB:
        raise MemoryCapExceeded(
            f"unified memory cap exceeded at '{label}': "
            f"sys_used={gb(vm.used)} > cap={cap_gb:.0f} GB"
        )


def preflight_memory(min_available_gb: float, *, label: str = "preflight") -> None:
    """Fail fast if insufficient memory headroom to safely run the workload.

    On shared unified-memory systems (e.g. GB10), other processes can consume
    enough of the pool that our workload's peak would push the system into
    swap-thrashing or OOM territory.  Call at script start with the min
    available memory your workload needs beyond baseline.

    Prints a PASS line on success so operators can see the headroom; calls
    ``sys.exit(2)`` on failure with a message explaining what to do.

    Args:
        min_available_gb: Minimum free memory required to proceed.
        label: Script name or stage label for the log message.
    """
    import sys

    vm = psutil.virtual_memory()
    avail_gb = vm.available / _GB
    used_gb = vm.used / _GB
    total_gb = vm.total / _GB
    if avail_gb < min_available_gb:
        print(
            f"[FATAL {label}] insufficient memory headroom: "
            f"{avail_gb:.1f} GB available, need >= {min_available_gb:.1f} GB",
            file=sys.stderr,
        )
        print(
            f"[FATAL {label}] sys_used={used_gb:.1f} GB / "
            f"{total_gb:.1f} GB total -- other processes are consuming too much. "
            f"Free up memory or wait for other workloads to finish.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        f"[{label}] memory headroom OK: {avail_gb:.1f} GB available "
        f"(min {min_available_gb:.1f} GB needed, sys_used={used_gb:.1f} GB / "
        f"{total_gb:.1f} GB)",
        flush=True,
    )


class MemoryCapStoppingCriteria:
    """HuggingFace StoppingCriteria that halts generation when memory cap is exceeded.

    Polls psutil every *check_every_n* tokens to avoid per-token
    overhead. When triggered, sets ``self.exceeded = True`` so the caller can
    distinguish a memory stop from a normal EOS stop.

    Args:
        cap_gb: Memory cap in gigabytes.
        check_every_n: Number of tokens between memory checks.
    """

    def __init__(self, cap_gb: float, check_every_n: int = 100) -> None:
        self.cap_bytes = cap_gb * _GB
        self.cap_gb = cap_gb
        self.check_every_n = check_every_n
        self.exceeded = False
        self._tokens_since_check = 0
        self.peak_seen_gb = 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        self._tokens_since_check += 1
        if self._tokens_since_check < self.check_every_n:
            return torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        self._tokens_since_check = 0
        vm = psutil.virtual_memory()
        used_gb = vm.used / _GB
        self.peak_seen_gb = max(self.peak_seen_gb, used_gb)
        if vm.used > self.cap_bytes:
            self.exceeded = True
            print(
                f"[watchdog] memory cap hit mid-generate: "
                f"sys_used={gb(vm.used)} > cap={self.cap_gb:.0f} GB -- "
                f"stopping generation",
                flush=True,
            )
            return torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        return torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
