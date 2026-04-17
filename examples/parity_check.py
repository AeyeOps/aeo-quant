#!/usr/bin/env python3
"""Parity check: generate 50 greedy tokens, save, diff against pinned baseline.

Regression canary for the FP8 MoE decode optimization plan. Does NOT measure
quality — it catches changes that silently alter output.

If the aeo-quant harness daemon is running (see `aeo-harness start`), this
script connects to it and reuses the already-loaded model. Otherwise it
loads the model in-process as before.

Usage:
    uv run examples/parity_check.py        # generates; establishes baseline if absent

Outputs:
    results/parity/<format>-<bits>bit-YYYYMMDD-HHMMSS/output.txt  # this run
    tests/fixtures/parity_baseline_{fp8,nvfp4}.txt                # per-format baseline

Baselines are per-format. NVFP4 runs also report divergence vs the FP8 baseline
as an informational quality delta (not a gate).

Exit codes:
    0 — match OR baseline established
    1 — divergence above 5% token mismatch vs own-format baseline
    2 — environment failure (no CUDA, no checkpoint, format mismatch with harness, etc.)
"""
from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

import aeo_quant  # noqa: F401 — triggers numpy compat shim
from aeo_quant.core.config import load_dotenv, quant_env, results_dir, setup_cuda_allocator
from aeo_quant.gpu.memory import mem_report
from aeo_quant.harness import HarnessUnavailable, get_or_start_harness

load_dotenv()
setup_cuda_allocator()

QUANT_FORMAT, CHECKPOINT, KV_BITS = quant_env()
GEN_TOKENS = 50

RESULTS_DIR = results_dir("parity", format=QUANT_FORMAT, kv_bits=KV_BITS)
BASELINE_DIR = Path("tests/fixtures")
OWN_BASELINE = BASELINE_DIR / f"parity_baseline_{QUANT_FORMAT}.txt"
FP8_BASELINE = BASELINE_DIR / "parity_baseline_fp8.txt"


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
            f"        Restart the harness with `aeo-harness stop && "
            f"aeo-harness start --format {QUANT_FORMAT}`\n"
            f"        or unset the harness to run in-process.",
            file=sys.stderr,
        )
        return None
    print(f"[parity] using harness (uptime={info.get('uptime_s')}s, "
          f"jobs_served={info.get('jobs_served')}, queue={info.get('queue_depth')})",
          flush=True)
    t = time.time()
    result = client.run_workload(
        "parity",
        gen_tokens=GEN_TOKENS,
        kv_bits=KV_BITS,
    )
    print(f"[parity] harness call completed in {time.time() - t:.1f}s", flush=True)
    return result


def main() -> int:
    print(
        f"[parity] QUANT_FORMAT={QUANT_FORMAT} CHECKPOINT={CHECKPOINT} KV_BITS={KV_BITS}",
        flush=True,
    )
    mem_report("parity:start")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        client = get_or_start_harness(preflight_label="parity")
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
        f"# parity_check {RESULTS_DIR.name}\n"
        f"# quant_format: {QUANT_FORMAT}\n"
        f"# gen_tokens: {len(new_ids)}\n"
        f"# all_token_ids: {new_ids}\n"
        f"# ---\n"
        f"{decoded}\n"
    )
    print(f"[parity] wrote {out_path}")

    if not OWN_BASELINE.exists():
        OWN_BASELINE.parent.mkdir(parents=True, exist_ok=True)
        OWN_BASELINE.write_text(out_path.read_text())
        print(f"[parity] established {QUANT_FORMAT} baseline at {OWN_BASELINE}")
        if QUANT_FORMAT != "fp8" and FP8_BASELINE.exists():
            _compare("vs fp8 (informational)", FP8_BASELINE, new_ids, fail=False)
        return 0

    own_rc = _compare(
        f"vs {QUANT_FORMAT} baseline", OWN_BASELINE, new_ids, fail=True
    )
    if QUANT_FORMAT != "fp8" and FP8_BASELINE.exists():
        _compare("vs fp8 (informational)", FP8_BASELINE, new_ids, fail=False)
    print(
        f"[parity] harness daemon still running ({QUANT_FORMAT}, ~27 GB). "
        f"Stop with `aeo-harness stop` when finished.",
        flush=True,
    )
    return own_rc


def _compare(label: str, baseline_path: Path, new_ids: list[int], *, fail: bool) -> int:
    """Compare new_ids to baseline. Returns exit code; fail=False never returns 1."""
    baseline_ids = load_token_ids(baseline_path)
    n = min(len(baseline_ids), len(new_ids))
    mismatches = sum(1 for a, b in zip(baseline_ids, new_ids, strict=False) if a != b)
    pct = 100 * mismatches / n if n else 0.0

    max_prefix = 0
    for a, b in zip(baseline_ids, new_ids, strict=False):
        if a != b:
            break
        max_prefix += 1

    print(f"[parity] {label}: mismatches {mismatches}/{n} ({pct:.1f}%), "
          f"max matching prefix {max_prefix} tokens")
    if mismatches == 0:
        print(f"[parity] {label}: PASS — byte-for-byte match")
        return 0
    if fail and pct > 5:
        print(f"[parity] {label}: FAIL — divergence above 5% ({pct:.1f}%)")
        return 1
    print(f"[parity] {label}: "
          + ("WARN" if fail else "delta")
          + f" — divergence {pct:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
