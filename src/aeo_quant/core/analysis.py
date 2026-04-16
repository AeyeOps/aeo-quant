"""Analysis helpers for load-test results.

Stdlib only — no third-party dependencies.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from .types import LevelStats, RampTransition, Sample, TurnRecord


def pct(values: list[float], p: float) -> float:
    """Compute the *p*-th percentile using linear interpolation.

    Returns 0.0 for empty input, the single value for length-1 input.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def find_ramp_transitions(samples: list[Sample], ramp_window_s: float) -> list[RampTransition]:
    """Walk samples in time order. For each 1->2, 2->3, 3->4 transition on the
    running counter, capture the peak used_mib in the ramp_window_s window
    starting at that transition."""
    transitions: list[RampTransition] = []
    seen_levels: set[tuple[int, int]] = set()
    ordered = sorted(samples, key=lambda s: s.ts)

    prev_running: int | None = None
    for i, s in enumerate(ordered):
        if s.running is None:
            continue
        if prev_running is not None and s.running == prev_running + 1 and 1 <= prev_running <= 3:
            key = (prev_running, s.running)
            if key not in seen_levels:
                seen_levels.add(key)
                start_ts = s.ts
                end_ts = start_ts + ramp_window_s
                window = [w for w in ordered[i:] if w.ts <= end_ts]
                if window:
                    peak = max(w.used_mib for w in window)
                    kv_values = [w.kv_pct for w in window if w.kv_pct is not None]
                    max_kv = max(kv_values) if kv_values else None
                    transitions.append(
                        RampTransition(
                            from_level=prev_running,
                            to_level=s.running,
                            ts=start_ts,
                            peak_used_mib=peak,
                            max_kv_pct=max_kv,
                            sample_count=len(window),
                        )
                    )
        prev_running = s.running

    return transitions


def per_level_stats(samples: list[Sample]) -> dict[int, LevelStats]:
    """Segment samples by running value and compute per-level stats."""
    by_level: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        if s.running is not None:
            by_level[s.running].append(s)
    stats: dict[int, LevelStats] = {}
    for level, group in by_level.items():
        used_vals = [g.used_mib for g in group]
        kv_vals = [g.kv_pct for g in group if g.kv_pct is not None]
        stats[level] = LevelStats(
            level=level,
            sample_count=len(group),
            max_used_mib=max(used_vals),
            mean_used_mib=statistics.fmean(used_vals),
            max_kv_pct=max(kv_vals) if kv_vals else None,
            mean_kv_pct=statistics.fmean(kv_vals) if kv_vals else None,
        )
    return stats


def per_session_summary(records: list[TurnRecord]) -> dict[int, dict]:
    """Compute per-session summary statistics from a list of TurnRecords."""
    by_session: dict[int, list[TurnRecord]] = defaultdict(list)
    for r in records:
        by_session[r.session_id].append(r)
    out: dict[int, dict] = {}
    for sid, rs in by_session.items():
        rs_sorted = sorted(rs, key=lambda r: r.turn_index)
        ok_turns = [r for r in rs_sorted if r.status == "ok"]
        first_four = list(ok_turns[:4])
        last_four = list(ok_turns[-4:])
        out[sid] = {
            "count": len(rs_sorted),
            "ok_count": len(ok_turns),
            "ttft_first": ok_turns[0].ttft if ok_turns else float("nan"),
            "ttft_last": ok_turns[-1].ttft if ok_turns else float("nan"),
            "wall_first": ok_turns[0].wall if ok_turns else float("nan"),
            "wall_last": ok_turns[-1].wall if ok_turns else float("nan"),
            "p95_first4": pct([r.wall for r in first_four], 95) if first_four else 0.0,
            "p95_last4": pct([r.wall for r in last_four], 95) if last_four else 0.0,
            "prompt_last": ok_turns[-1].prompt_tokens if ok_turns else 0,
            "kv_max": max(
                (r.kv_pct_max_during for r in ok_turns if r.kv_pct_max_during is not None),
                default=None,
            ),
        }
    return out
