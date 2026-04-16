"""Core data types, constants, and thread-safe state primitives.

Stdlib only — no third-party dependencies.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_KILLED = 2
EXIT_CRASH = 3

EXIT_CODES = {
    EXIT_OK: "ok",
    EXIT_FAIL: "fail",
    EXIT_KILLED: "killed",
    EXIT_CRASH: "crash",
}

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

VLLM_LOG_PATTERN = re.compile(
    r"Avg prompt throughput:\s*([\d.]+)\s*tokens/s.*?"
    r"Avg generation throughput:\s*([\d.]+)\s*tokens/s.*?"
    r"Running:\s*(\d+)\s*reqs.*?"
    r"Waiting:\s*(\d+)\s*reqs.*?"
    r"GPU KV cache usage:\s*([\d.]+)%"
    r"(?:.*?Prefix cache hit rate:\s*([\d.]+)%)?",
    re.DOTALL,
)

POOL_GIB_PATTERN = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")

# ---------------------------------------------------------------------------
# CSV headers
# ---------------------------------------------------------------------------

PER_TURN_HEADER = [
    "ts",
    "session_id",
    "turn_index",
    "status",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "wall_latency_s",
    "ttft_s",
    "running_max_during",
    "kv_pct_max_during",
    "mem_used_max_during",
    "swap_max_during",
    "ramp_event",
]

MEMTRAIL_HEADER = [
    "ts",
    "used_mib",
    "available_mib",
    "swap_mib",
    "container_rss_mib",
    "running",
    "waiting",
    "kv_pct",
    "prefix_hit_pct",
    "prompt_tps",
    "gen_tps",
]


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def iso(ts: float) -> str:
    """Format a Unix timestamp as an ISO 8601 string (UTC, millisecond precision)."""
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    ts: float
    used_mib: int
    available_mib: int
    swap_mib: int
    container_rss_mib: int | None
    running: int | None
    waiting: int | None
    kv_pct: float | None
    prefix_hit_pct: float | None
    prompt_tps: float | None
    gen_tps: float | None


@dataclass
class TurnRecord:
    session_id: int
    turn_index: int
    status: str
    wall: float
    ttft: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    start_ts: float
    end_ts: float
    mem_used_max_during: int | None
    kv_pct_max_during: float | None
    running_max_during: int | None
    ramp_event: str


@dataclass
class RampTransition:
    from_level: int
    to_level: int
    ts: float
    peak_used_mib: int
    max_kv_pct: float | None
    sample_count: int


@dataclass
class LevelStats:
    level: int
    sample_count: int
    max_used_mib: int
    mean_used_mib: float
    max_kv_pct: float | None
    mean_kv_pct: float | None


# ---------------------------------------------------------------------------
# Typed message segments (stream parser output)
# ---------------------------------------------------------------------------

SEGMENT_TYPES = frozenset({
    "user",
    "system",
    "assistant",
    "thinking",
    "tool_call",
    "tool_result",
    "unknown",
})


@dataclass
class Segment:
    """One typed slice of a model turn's output.

    type: well-known type from SEGMENT_TYPES. Consumers dispatch on this.
    content: raw text content (unescaped, as-decoded).
    metadata: optional type-specific info (channel name, tool name, reason, ...).
    """
    type: str
    content: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {"type": self.type, "content": self.content}
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        return cls(
            type=d["type"],
            content=d["content"],
            metadata=d.get("metadata", {}) or {},
        )


# ---------------------------------------------------------------------------
# Thread-safe state
# ---------------------------------------------------------------------------

@dataclass
class KillState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    tripped: bool = False
    reason: str = ""

    def trip(self, reason: str) -> None:
        with self.lock:
            if not self.tripped:
                self.tripped = True
                self.reason = reason
                print(f"!! KILL SWITCH TRIPPED: {reason}", flush=True)


# ---------------------------------------------------------------------------
# Monitor thread
# ---------------------------------------------------------------------------

class Monitor(threading.Thread):
    """Background thread that periodically samples system/container metrics,
    writes rows to a CSV file, and trips a KillState when resource thresholds
    are exceeded.

    Depends on writers.CSVWriter and parsing helpers for the actual data
    collection, but is defined here so that types.py remains the single
    source of truth for all core data types.  Callers must inject the
    csv_writer and configure the polling functions via the constructor.
    """

    def __init__(
        self,
        csv_writer: object,  # CSVWriter from .writers
        kill_state: KillState,
        swap_baseline_mib: int,
        *,
        container_name: str,
        sample_fn: object | None = None,
        monitor_interval_s: float = 0.5,
        heartbeat_interval_s: float = 15.0,
        docker_stats_interval_s: float = 5.0,
        sample_ring_size: int = 8000,
        kill_used_gib: float = 88.0,
        kill_swap_delta_gib: float = 1.5,
        kill_kv_pct: float = 95.0,
    ):
        super().__init__(name="monitor", daemon=True)
        self.csv_writer = csv_writer
        self.kill_state = kill_state
        self.swap_baseline_mib = swap_baseline_mib
        self.container_name = container_name
        self.sample_fn = sample_fn
        self.monitor_interval_s = monitor_interval_s
        self.heartbeat_interval_s = heartbeat_interval_s
        self.docker_stats_interval_s = docker_stats_interval_s
        self.sample_ring_size = sample_ring_size
        self.kill_used_gib = kill_used_gib
        self.kill_swap_delta_gib = kill_swap_delta_gib
        self.kill_kv_pct = kill_kv_pct
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.last_sample: Sample | None = None
        self.samples: list[Sample] = []
        self.peak_used_mib: int = 0
        self.peak_swap_mib: int = swap_baseline_mib
        self.peak_kv_pct: float = 0.0
        self.tick: int = 0
        self._last_heartbeat = 0.0
        self._last_docker_stats = 0.0
        self._cached_rss: int | None = None

    def run(self) -> None:
        # Import here to avoid circular dependency at module level
        from .parsing import (
            read_docker_stats_rss_mib,
            read_free_m,
            read_latest_vllm_log_match,
        )

        while not self.stop_event.is_set():
            try:
                self._sample_once(read_free_m, read_latest_vllm_log_match, read_docker_stats_rss_mib)
            except Exception as e:
                print(f"!! monitor sample error: {e}", flush=True)
            self.stop_event.wait(self.monitor_interval_s)

    def _sample_once(
        self,
        read_free_m_fn: object,
        read_latest_vllm_log_match_fn: object,
        read_docker_stats_rss_mib_fn: object,
    ) -> None:
        import time

        now = time.time()
        used, available, swap = read_free_m_fn()  # type: ignore[operator]
        log = read_latest_vllm_log_match_fn(self.container_name)  # type: ignore[operator]

        if now - self._last_docker_stats >= self.docker_stats_interval_s:
            self._cached_rss = read_docker_stats_rss_mib_fn(self.container_name)  # type: ignore[operator]
            self._last_docker_stats = now

        sample = Sample(
            ts=now,
            used_mib=used,
            available_mib=available,
            swap_mib=swap,
            container_rss_mib=self._cached_rss,
            running=log.get("running") if log else None,
            waiting=log.get("waiting") if log else None,
            kv_pct=log.get("kv_pct") if log else None,
            prefix_hit_pct=log.get("prefix_hit_pct") if log else None,
            prompt_tps=log.get("prompt_tps") if log else None,
            gen_tps=log.get("gen_tps") if log else None,
        )

        with self.lock:
            self.last_sample = sample
            self.samples.append(sample)
            if len(self.samples) > self.sample_ring_size:
                del self.samples[: len(self.samples) - self.sample_ring_size]
            self.tick += 1
            if sample.used_mib > self.peak_used_mib:
                self.peak_used_mib = sample.used_mib
            if sample.swap_mib > self.peak_swap_mib:
                self.peak_swap_mib = sample.swap_mib
            if sample.kv_pct is not None and sample.kv_pct > self.peak_kv_pct:
                self.peak_kv_pct = sample.kv_pct

        self.csv_writer.write(  # type: ignore[union-attr]
            {
                "ts": iso(sample.ts),
                "used_mib": sample.used_mib,
                "available_mib": sample.available_mib,
                "swap_mib": sample.swap_mib,
                "container_rss_mib": sample.container_rss_mib if sample.container_rss_mib is not None else "",
                "running": sample.running if sample.running is not None else "",
                "waiting": sample.waiting if sample.waiting is not None else "",
                "kv_pct": sample.kv_pct if sample.kv_pct is not None else "",
                "prefix_hit_pct": sample.prefix_hit_pct if sample.prefix_hit_pct is not None else "",
                "prompt_tps": sample.prompt_tps if sample.prompt_tps is not None else "",
                "gen_tps": sample.gen_tps if sample.gen_tps is not None else "",
            }
        )

        if now - self._last_heartbeat >= self.heartbeat_interval_s:
            self._last_heartbeat = now
            kv_str = f"{sample.kv_pct:5.1f}%" if sample.kv_pct is not None else "  -- "
            run_str = f"{sample.running}" if sample.running is not None else "-"
            print(
                f"[mon {iso(sample.ts)}] used={sample.used_mib / 1024:5.1f}GiB "
                f"avail={sample.available_mib / 1024:5.1f}GiB "
                f"swap={sample.swap_mib / 1024:5.2f}GiB "
                f"kv={kv_str} run={run_str}",
                flush=True,
            )

        # Kill switch
        reasons: list[str] = []
        if sample.used_mib / 1024.0 > self.kill_used_gib:
            reasons.append(f"used {sample.used_mib / 1024:.1f}GiB > {self.kill_used_gib}")
        if (sample.swap_mib - self.swap_baseline_mib) / 1024.0 > self.kill_swap_delta_gib:
            reasons.append(
                f"swap {sample.swap_mib / 1024:.2f}GiB > baseline+{self.kill_swap_delta_gib}"
            )
        if sample.kv_pct is not None and sample.kv_pct > self.kill_kv_pct:
            reasons.append(f"kv {sample.kv_pct:.1f}% > {self.kill_kv_pct}")
        if reasons:
            self.kill_state.trip("; ".join(reasons))

    def window_samples(self, start_ts: float, end_ts: float) -> list[Sample]:
        with self.lock:
            return [s for s in self.samples if start_ts <= s.ts <= end_ts]

    def window_max(
        self, start_ts: float, end_ts: float
    ) -> tuple[int | None, int | None, float | None, int | None, int]:
        """Return (max_used_mib, max_swap_mib, max_kv_pct, max_running, n) for [start, end]."""
        window = self.window_samples(start_ts, end_ts)
        if not window:
            return None, None, None, None, 0
        max_used = max(s.used_mib for s in window)
        max_swap = max(s.swap_mib for s in window)
        kv_values = [s.kv_pct for s in window if s.kv_pct is not None]
        max_kv = max(kv_values) if kv_values else None
        running_values = [s.running for s in window if s.running is not None]
        max_running = max(running_values) if running_values else None
        return max_used, max_swap, max_kv, max_running, len(window)

    def all_samples_snapshot(self) -> list[Sample]:
        with self.lock:
            return list(self.samples)

    def stop(self) -> None:
        self.stop_event.set()
