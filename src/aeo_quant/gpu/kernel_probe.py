"""Safety harness for running new GPU kernel probes on shared hardware.

Each probe runs in a fresh python subprocess with a hard timeout and
pre/post nvidia-smi snapshots.  A kernel that hangs the CUDA context
dies with its subprocess, so the parent session and other GPU users on
the same card stay healthy.

Typical use::

    from aeo_quant.gpu.kernel_probe import run_isolated
    result = run_isolated("examples/probe_nvfp4_torchao.py", timeout_s=60)
    print(result.summary())
    if not result.ok:
        sys.exit(1)

Escape hatch: if a probe times out AND the GPU snapshot shows stuck
memory or stuck utilization, reset the context with::

    nvidia-smi --gpu-reset -i 0

(That command is only safe when no other CUDA workload is running.)
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


_GB = 1024**3


@dataclasses.dataclass
class GpuSnapshot:
    """Best-effort GPU state.  Fields are ``None`` when nvidia-smi
    reports ``[N/A]`` (e.g. unified-memory SoCs like GB10 don't expose
    per-GPU memory.used).  Callers should print or log, not assert.
    """
    mem_used_mib: int | None
    mem_free_mib: int | None
    temp_c: int | None
    util_pct: int | None
    torch_alloc_mib: int | None = None
    torch_peak_mib: int | None = None

    @classmethod
    def capture(cls) -> "GpuSnapshot":
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        parts = [p.strip() for p in out.stdout.strip().split(",")]

        def _maybe_int(s: str) -> int | None:
            try:
                return int(s)
            except (ValueError, TypeError):
                return None

        # torch_alloc/peak: only populated if torch is already imported
        # and CUDA initialised. Probing this here is a best-effort
        # convenience, not a correctness check.
        torch_alloc = torch_peak = None
        try:
            import torch as _t  # noqa: PLC0415
            if _t.cuda.is_available() and _t.cuda.is_initialized():
                torch_alloc = _t.cuda.memory_allocated() // (1024 * 1024)
                torch_peak = _t.cuda.max_memory_allocated() // (1024 * 1024)
        except ImportError:
            pass

        return cls(
            mem_used_mib=_maybe_int(parts[0]),
            mem_free_mib=_maybe_int(parts[1]),
            temp_c=_maybe_int(parts[2]),
            util_pct=_maybe_int(parts[3]),
            torch_alloc_mib=torch_alloc,
            torch_peak_mib=torch_peak,
        )

    @staticmethod
    def _fmt(n: int | None, unit: str) -> str:
        return f"{n}{unit}" if n is not None else f"N/A{unit}"

    def __str__(self) -> str:
        s = (
            f"mem_used={self._fmt(self.mem_used_mib, 'MiB')} "
            f"mem_free={self._fmt(self.mem_free_mib, 'MiB')} "
            f"temp={self._fmt(self.temp_c, 'C')} "
            f"util={self._fmt(self.util_pct, '%')}"
        )
        if self.torch_alloc_mib is not None:
            s += f" torch_alloc={self.torch_alloc_mib}MiB"
        return s


@dataclasses.dataclass
class ProbeResult:
    script: str
    args: tuple[str, ...]
    ok: bool
    returncode: int
    timed_out: bool
    elapsed_s: float
    stdout: str
    stderr: str
    pre_gpu: GpuSnapshot
    post_gpu: GpuSnapshot

    def summary(self) -> str:
        status = "OK" if self.ok else "FAIL"
        if self.timed_out:
            status = "TIMEOUT"
        return (
            f"[probe {status}] rc={self.returncode} elapsed={self.elapsed_s:.1f}s\n"
            f"  gpu pre:  {self.pre_gpu}\n"
            f"  gpu post: {self.post_gpu}"
        )


def preflight_mem(min_free_gb: float) -> None:
    """Fail fast if host RAM headroom is below *min_free_gb*."""
    vm = psutil.virtual_memory()
    free_gb = vm.available / _GB
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"insufficient host memory: {free_gb:.1f} GB free < "
            f"{min_free_gb:.1f} GB required"
        )


def run_isolated(
    script: str | Path,
    *args: str,
    timeout_s: int = 60,
    min_free_gb: float = 5.0,
    env_extra: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> ProbeResult:
    """Run a probe script in an isolated subprocess.

    Args:
        script: path to the probe script (relative or absolute).
        args: CLI arguments to pass through.
        timeout_s: hard-kill the subprocess after this many seconds.
            Default 60s is appropriate for synthetic-tensor probes;
            bump for real-weight workloads.
        min_free_gb: refuse to launch if host RAM available is less
            than this.  The GB10 is shared; default 5 GB matches the
            preflight guard used elsewhere in the repo.
        env_extra: extra env vars to export to the subprocess.
        cwd: working directory for the subprocess.

    Returns:
        :class:`ProbeResult` with stdout/stderr, exit code, timing,
        and GPU snapshots before/after.

    Raises:
        RuntimeError: if the host-memory preflight fails.
    """
    preflight_mem(min_free_gb)

    script_path = str(Path(script))
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    pre = GpuSnapshot.capture()
    t0 = time.monotonic()

    try:
        cp = subprocess.run(
            ["uv", "run", "python", script_path, *args],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        timed_out = False
        rc = cp.returncode
        out = cp.stdout
        err = cp.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        rc = -signal_sigkill()
        out = _decode(e.stdout)
        err = _decode(e.stderr)

    elapsed = time.monotonic() - t0
    post = GpuSnapshot.capture()

    return ProbeResult(
        script=script_path,
        args=tuple(args),
        ok=(rc == 0 and not timed_out),
        returncode=rc,
        timed_out=timed_out,
        elapsed_s=elapsed,
        stdout=out,
        stderr=err,
        pre_gpu=pre,
        post_gpu=post,
    )


def signal_sigkill() -> int:
    import signal
    return int(signal.SIGKILL)


def _decode(buf: bytes | str | None) -> str:
    if buf is None:
        return ""
    if isinstance(buf, bytes):
        return buf.decode("utf-8", errors="replace")
    return buf
