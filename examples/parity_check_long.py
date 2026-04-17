#!/usr/bin/env python3
"""Long-form parity check — 300 generated tokens.

50 tokens is enough to catch a single cliff-effect; it's not enough to
tell whether a path repeatedly near-ties and flips, drifts into
repetition, or quietly loses coherence. This generates 300 tokens via
the running harness, writes the output to disk, and compares to a
pinned long-baseline if one exists.

Usage::

    PARITY_MIN_FREE_GB=20 QUANT_FORMAT=nvfp4 uv run python examples/parity_check_long.py
"""
from __future__ import annotations

import ast
import os
import sys
import time
from pathlib import Path

import aeo_quant  # noqa: F401
from aeo_quant.core.config import load_dotenv, quant_env, results_dir, setup_cuda_allocator
from aeo_quant.gpu.memory import preflight_memory
from aeo_quant.harness import HarnessUnavailable, get_or_start_harness

MIN_FREE_GB = float(os.environ.get("PARITY_MIN_FREE_GB", "50"))
GEN_TOKENS = int(os.environ.get("PARITY_LONG_TOKENS", "300"))

load_dotenv()
setup_cuda_allocator()

QUANT_FORMAT, CHECKPOINT, KV_BITS = quant_env()
RESULTS_DIR = results_dir("parity_long", format=QUANT_FORMAT, kv_bits=KV_BITS)
BASELINE_DIR = Path("tests/fixtures")
OWN_BASELINE = BASELINE_DIR / f"parity_long_baseline_{QUANT_FORMAT}.txt"


def load_token_ids(path: Path) -> list[int]:
    for line in path.read_text().splitlines():
        if line.startswith("# all_token_ids: "):
            return ast.literal_eval(line[len("# all_token_ids: "):])
    raise ValueError(f"no token-ids line in {path}")


def main() -> int:
    print(f"[parity_long] QUANT_FORMAT={QUANT_FORMAT} gen_tokens={GEN_TOKENS}")
    preflight_memory(MIN_FREE_GB, label="parity_long")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        client = get_or_start_harness()
    except HarnessUnavailable as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 2

    info = client.status()
    if info.get("quant_format") != QUANT_FORMAT:
        print(f"[FATAL] format mismatch: harness={info.get('quant_format')} want={QUANT_FORMAT}", file=sys.stderr)
        return 2
    print(f"[parity_long] harness uptime={info.get('uptime_s')}s jobs={info.get('jobs_served')}")

    t = time.time()
    result = client.run_workload(
        "parity",
        gen_tokens=GEN_TOKENS,
        kv_bits=KV_BITS,
    )
    print(f"[parity_long] call completed in {time.time() - t:.1f}s")

    new_ids = result["all_token_ids"]
    decoded = result["decoded"]

    out_path = RESULTS_DIR / "output.txt"
    out_path.write_text(
        f"# parity_long {RESULTS_DIR.name}\n"
        f"# quant_format: {QUANT_FORMAT}\n"
        f"# gen_tokens: {len(new_ids)}\n"
        f"# tok_per_s: {result.get('tok_per_s')}\n"
        f"# all_token_ids: {new_ids}\n"
        f"# ---\n"
        f"{decoded}\n"
    )
    print(f"[parity_long] wrote {out_path}")
    print(f"[parity_long] tok/s: {result.get('tok_per_s')}")

    if not OWN_BASELINE.exists():
        OWN_BASELINE.parent.mkdir(parents=True, exist_ok=True)
        OWN_BASELINE.write_text(out_path.read_text())
        print(f"[parity_long] established long-baseline at {OWN_BASELINE}")
        return 0

    baseline_ids = load_token_ids(OWN_BASELINE)
    n = min(len(baseline_ids), len(new_ids))
    mismatches = sum(1 for a, b in zip(baseline_ids, new_ids, strict=False) if a != b)
    pct = 100 * mismatches / n if n else 0.0
    max_prefix = 0
    for a, b in zip(baseline_ids, new_ids, strict=False):
        if a != b:
            break
        max_prefix += 1
    print(
        f"[parity_long] mismatches {mismatches}/{n} ({pct:.1f}%), "
        f"max matching prefix {max_prefix} tokens"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
