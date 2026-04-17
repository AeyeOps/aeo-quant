#!/usr/bin/env python3
"""Long parity check: 2000 greedy tokens, diff against pinned baseline.

Distinct from ``parity_check.py`` (50 tokens) because SWA-bounded caches evict
tokens older than ``sliding_window`` from sliding-attention layers — a path
``parity_check.py`` never touches. 2000 tokens crosses the 1024 window on
sliding layers, so if SWA eviction breaks anything, this catches it. If parity
holds at 2000 tokens, it holds at any length (the logic doesn't change past
the first window crossing).

Baseline must be established against the pre-fix cache (commit 11db2e8).
Subsequent runs — with or without the new cache — must match bit-exact.

Usage:
    uv run examples/parity_long_check.py              # establish or diff baseline

Outputs:
    results/parity_long/<format>-<bits>bit-YYYYMMDD-HHMMSS/output.txt
    tests/fixtures/parity_long_baseline_{fp8,nvfp4}.txt

Exit codes:
    0 — match OR baseline established
    1 — divergence above 0.5% token mismatch vs own-format baseline
    2 — environment failure
"""
from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

import aeo_quant  # noqa: F401
from aeo_quant.core.config import load_dotenv, quant_env, results_dir, setup_cuda_allocator
from aeo_quant.gpu.memory import mem_report, preflight_memory
from aeo_quant.harness import HarnessUnavailable, get_or_start_harness

MIN_FREE_GB = 50.0

load_dotenv()
setup_cuda_allocator()

QUANT_FORMAT, CHECKPOINT, KV_BITS = quant_env()
GEN_TOKENS = 2000

RESULTS_DIR = results_dir("parity_long", format=QUANT_FORMAT, kv_bits=KV_BITS)
BASELINE_DIR = Path("tests/fixtures")
OWN_BASELINE = BASELINE_DIR / f"parity_long_baseline_{QUANT_FORMAT}.txt"


def load_token_ids(path: Path) -> list[int]:
    for line in path.read_text().splitlines():
        if line.startswith("# all_token_ids: "):
            return ast.literal_eval(line[len("# all_token_ids: "):])
    raise ValueError(f"no token-ids line in {path}")


def _run_via_harness(client) -> dict | None:
    info = client.status()
    srv_format = info.get("quant_format")
    if srv_format != QUANT_FORMAT:
        print(
            f"[FATAL] harness is loaded with {srv_format!r}, "
            f"you requested {QUANT_FORMAT!r}.\n"
            f"        Restart with `aeo-harness stop && "
            f"aeo-harness start --format {QUANT_FORMAT}`.",
            file=sys.stderr,
        )
        return None
    print(
        f"[parity_long] using harness (uptime={info.get('uptime_s')}s, "
        f"jobs_served={info.get('jobs_served')}, queue={info.get('queue_depth')})",
        flush=True,
    )
    t = time.time()
    result = client.run_workload(
        "parity",
        gen_tokens=GEN_TOKENS,
        kv_bits=KV_BITS,
    )
    print(f"[parity_long] harness call completed in {time.time() - t:.1f}s", flush=True)
    return result


def _compare(baseline_path: Path, new_ids: list[int]) -> int:
    baseline_ids = load_token_ids(baseline_path)
    n = min(len(baseline_ids), len(new_ids))
    mismatches = sum(1 for a, b in zip(baseline_ids, new_ids, strict=False) if a != b)
    pct = 100 * mismatches / n if n else 0.0

    max_prefix = 0
    for a, b in zip(baseline_ids, new_ids, strict=False):
        if a != b:
            break
        max_prefix += 1

    print(
        f"[parity_long] vs {QUANT_FORMAT} baseline: mismatches {mismatches}/{n} "
        f"({pct:.2f}%), max matching prefix {max_prefix} tokens"
    )
    if mismatches == 0:
        print(f"[parity_long] PASS — byte-for-byte match at {n} tokens")
        return 0
    if pct > 0.5:
        print(f"[parity_long] FAIL — divergence above 0.5% ({pct:.2f}%)")
        return 1
    print(f"[parity_long] WARN — small divergence {pct:.2f}%, below 0.5% gate")
    return 0


def main() -> int:
    print(
        f"[parity_long] QUANT_FORMAT={QUANT_FORMAT} CHECKPOINT={CHECKPOINT} "
        f"KV_BITS={KV_BITS} GEN_TOKENS={GEN_TOKENS}",
        flush=True,
    )
    preflight_memory(MIN_FREE_GB, label="parity_long")
    mem_report("parity_long:start")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        client = get_or_start_harness()
    except HarnessUnavailable as e:
        print(f"[FATAL] harness unavailable: {e}", file=sys.stderr)
        return 2
    result = _run_via_harness(client)
    if result is None:
        return 2

    new_ids = result["all_token_ids"]
    decoded = result["decoded"]

    out_path = RESULTS_DIR / "output.txt"
    out_path.write_text(
        f"# parity_long_check {RESULTS_DIR.name}\n"
        f"# quant_format: {QUANT_FORMAT}\n"
        f"# kv_bits: {KV_BITS}\n"
        f"# gen_tokens: {len(new_ids)}\n"
        f"# all_token_ids: {new_ids}\n"
        f"# ---\n"
        f"{decoded}\n"
    )
    print(f"[parity_long] wrote {out_path}")

    if not OWN_BASELINE.exists():
        OWN_BASELINE.parent.mkdir(parents=True, exist_ok=True)
        OWN_BASELINE.write_text(out_path.read_text())
        print(f"[parity_long] established {QUANT_FORMAT} baseline at {OWN_BASELINE}")
        return 0

    rc = _compare(OWN_BASELINE, new_ids)
    print(
        f"[parity_long] harness daemon still running ({QUANT_FORMAT}, ~27 GB). "
        f"Stop with `aeo-harness stop` when finished.",
        flush=True,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
