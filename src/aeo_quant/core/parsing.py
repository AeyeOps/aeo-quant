"""System metric parsing helpers (free, docker stats, vLLM logs).

Stdlib only — no third-party dependencies.
"""

from __future__ import annotations

import re
import subprocess
from typing import Optional

from .types import POOL_GIB_PATTERN, VLLM_LOG_PATTERN


def parse_free_m(text: str) -> tuple[int, int, int]:
    """Parse ``free -m`` output into (used_mib, available_mib, swap_used_mib)."""
    used = avail = swap_used = -1
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Mem:" and len(parts) >= 7:
            used = int(parts[2])
            avail = int(parts[6])
        elif parts[0] == "Swap:" and len(parts) >= 4:
            swap_used = int(parts[2])
    if used < 0 or avail < 0 or swap_used < 0:
        raise RuntimeError(f"could not parse free -m output: {text!r}")
    return used, avail, swap_used


def read_free_m() -> tuple[int, int, int]:
    """Run ``free -m`` and return parsed (used_mib, available_mib, swap_used_mib)."""
    out = subprocess.run(
        ["free", "-m"], capture_output=True, text=True, check=True
    ).stdout
    return parse_free_m(out)


def parse_size_to_mib(s: str) -> Optional[int]:
    """Parse a human-readable size string (e.g. '3.5GiB') to integer MiB."""
    s = s.strip()
    m = re.match(r"^([\d.]+)\s*([KMGT]i?B)$", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    factors = {
        "B": 1 / (1024 * 1024),
        "KB": 1 / 1024,
        "KiB": 1 / 1024,
        "MB": 1.0,
        "MiB": 1.0,
        "GB": 1024.0,
        "GiB": 1024.0,
        "TB": 1024.0 * 1024,
        "TiB": 1024.0 * 1024,
    }
    return int(val * factors.get(unit, 1.0))


def read_docker_stats_rss_mib(container_name: str) -> Optional[int]:
    """Read the current memory usage of a Docker container in MiB."""
    try:
        out = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.MemUsage}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    first = out.split("/")[0].strip()
    return parse_size_to_mib(first)


def read_latest_vllm_log_match(container_name: str) -> Optional[dict]:
    """Scan recent Docker logs for the latest vLLM throughput/status line."""
    try:
        out = subprocess.run(
            ["docker", "logs", "--tail", "300", container_name],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    blob = (out.stdout or "") + "\n" + (out.stderr or "")
    for line in reversed(blob.splitlines()):
        m = VLLM_LOG_PATTERN.search(line)
        if m:
            return {
                "prompt_tps": float(m.group(1)),
                "gen_tps": float(m.group(2)),
                "running": int(m.group(3)),
                "waiting": int(m.group(4)),
                "kv_pct": float(m.group(5)),
                "prefix_hit_pct": float(m.group(6)) if m.group(6) else None,
            }
    return None


def read_pool_gib(container_name: str) -> Optional[float]:
    """Scan Docker logs for the KV cache pool size in GiB."""
    try:
        out = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    blob = (out.stdout or "") + "\n" + (out.stderr or "")
    for line in blob.splitlines():
        m = POOL_GIB_PATTERN.search(line)
        if m:
            return float(m.group(1))
    return None
