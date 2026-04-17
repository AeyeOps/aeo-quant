#!/usr/bin/env python3
"""Reasoning quality check: two hard prompts that stress attention precision.

Runs via the aeo-quant harness daemon (see `aeo-harness start`). The harness
holds the model loaded; this script dispatches the ``reasoning`` workload,
captures per-prompt events, writes per-prompt output files + timing.json, and
diffs against a pinned per-kv-bits baseline.

Usage:
    uv run examples/reasoning_check.py              # bits=4 (default)
    KV_BITS=3 uv run examples/reasoning_check.py    # test 3-bit KV cache

Outputs:
    results/reasoning/<format>-<bits>bit-<timestamp>/
        prompt_1_math.txt       — full output + token IDs
        prompt_2_lru.txt        — full output + token IDs
        timing.json             — aggregate timing
        stdout.log              — client-side prints (preflight, events, diffs).
                                   Daemon-side per-token detail lives in
                                   ~/.aeo-quant/harness.log.

    If a baseline exists at results/reasoning/baseline_<format>-<bits>bit/,
    diffs against it and reports token-level match statistics.

Exit codes:
    0 — completed (quality is assessed by reading the output, not automated)
    2 — environment failure (no harness, format mismatch, etc.)
"""
from __future__ import annotations

import ast
import atexit
import json
import sys
from pathlib import Path

import aeo_quant  # noqa: F401 — triggers numpy compat shim
from aeo_quant.core.config import load_dotenv, quant_env, results_dir, setup_cuda_allocator
from aeo_quant.core.writers import Tee
from aeo_quant.gpu.memory import mem_report
from aeo_quant.harness import HarnessUnavailable, get_or_start_harness

load_dotenv()
setup_cuda_allocator()

QUANT_FORMAT, CHECKPOINT, KV_BITS = quant_env()
GEN_TOKENS = 500

RESULTS_DIR = results_dir("reasoning", format=QUANT_FORMAT, kv_bits=KV_BITS)
BASELINE_DIR = Path(f"results/reasoning/baseline_{QUANT_FORMAT}-{KV_BITS}bit")


def save_output(path: Path, name: str, token_ids: list[int], decoded: str,
                gen_ms: float) -> None:
    path.write_text(
        f"# reasoning_check: {name}\n"
        f"# kv_bits: {KV_BITS}\n"
        f"# gen_tokens: {len(token_ids)}\n"
        f"# gen_time_ms: {gen_ms:.1f}\n"
        f"# all_token_ids: {token_ids}\n"
        f"# ---\n"
        f"{decoded}\n"
    )


def load_token_ids(path: Path) -> list[int]:
    for line in path.read_text().splitlines():
        if line.startswith("# all_token_ids: "):
            return ast.literal_eval(line[len("# all_token_ids: "):])
    raise ValueError(f"no token-ids line in {path}")


def diff_against_baseline(file_name: str, new_ids: list[int]) -> None:
    baseline_file = BASELINE_DIR / file_name
    if not baseline_file.exists():
        return
    baseline_ids = load_token_ids(baseline_file)
    n = min(len(baseline_ids), len(new_ids))
    mismatches = sum(1 for a, b in zip(baseline_ids, new_ids, strict=False) if a != b)
    pct = 100 * mismatches / n if n else 0.0
    max_prefix = 0
    for a, b in zip(baseline_ids, new_ids, strict=False):
        if a != b:
            break
        max_prefix += 1
    print(f"  [diff] vs baseline: {mismatches}/{n} mismatches ({pct:.1f}%)")
    print(f"  [diff] max matching prefix: {max_prefix} tokens")


def _on_event(event: dict) -> None:
    t = event.get("type")
    if t == "prompt_start":
        print(
            f"\n{'=' * 60}\n"
            f"[reasoning] prompt {event['idx']}: {event['name']}\n"
            f"{'=' * 60}",
            flush=True,
        )
    elif t == "prompt_complete":
        print(
            f"  [timing] {event['gen_ms']:.1f} ms "
            f"({event['tok_s']:.2f} tok/s)",
            flush=True,
        )
    else:
        msg = event.get("message")
        if msg:
            print(f"[event] {msg}", flush=True)


def _check_harness_format(client) -> int:
    info = client.status()
    srv_format = info.get("quant_format")
    if srv_format != QUANT_FORMAT:
        print(
            f"[FATAL] harness is loaded with {srv_format!r}, "
            f"you requested {QUANT_FORMAT!r}.\n"
            f"        Restart the harness with `aeo-harness stop && "
            f"aeo-harness start --format {QUANT_FORMAT}`.",
            file=sys.stderr,
        )
        return 2
    print(
        f"[reasoning] using harness (uptime={info.get('uptime_s')}s, "
        f"jobs_served={info.get('jobs_served')}, queue={info.get('queue_depth')})",
        flush=True,
    )
    return 0


def main() -> int:
    print(
        f"[reasoning] QUANT_FORMAT={QUANT_FORMAT} CHECKPOINT={CHECKPOINT} "
        f"KV_BITS={KV_BITS} GEN_TOKENS={GEN_TOKENS}",
        flush=True,
    )
    mem_report("reasoning:start")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _log = open(RESULTS_DIR / "stdout.log", "w")  # noqa: SIM115
    atexit.register(_log.close)
    sys.stdout = Tee(sys.__stdout__, _log)
    sys.stderr = Tee(sys.__stderr__, _log)

    try:
        client = get_or_start_harness(preflight_label="reasoning")
    except HarnessUnavailable as e:
        print(f"[FATAL] harness unavailable: {e}", file=sys.stderr)
        return 2

    rc = _check_harness_format(client)
    if rc != 0:
        return rc

    result = client.run_workload(
        "reasoning",
        on_event=_on_event,
        gen_tokens=GEN_TOKENS,
        kv_bits=KV_BITS,
    )

    baseline_established = not BASELINE_DIR.exists()
    if baseline_established:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    for record in result["prompts"]:
        out_path = RESULTS_DIR / record["file"]
        save_output(
            out_path,
            record["name"],
            record["all_token_ids"],
            record["decoded"],
            record["gen_ms"],
        )
        print(f"  [output] {out_path}", flush=True)
        diff_against_baseline(record["file"], record["all_token_ids"])

        if baseline_established:
            (BASELINE_DIR / record["file"]).write_text(out_path.read_text())

    aggregate = result["aggregate"]
    print(f"\n{'=' * 60}\n[reasoning] aggregate\n{'=' * 60}")
    print(f"  total tokens:  {aggregate['total_tokens']}")
    print(f"  total time:    {aggregate['total_ms']:.1f} ms")
    print(f"  avg tok/s:     {aggregate['avg_tok_s']:.2f}")
    print(f"  kv_bits:       {KV_BITS}")
    mem_report("reasoning:final")

    timing_path = RESULTS_DIR / "timing.json"
    with open(timing_path, "w") as f:
        json.dump({
            "kv_bits": KV_BITS,
            "gen_tokens_per_prompt": GEN_TOKENS,
            "prompts": [
                {
                    "name": r["name"],
                    "prompt_tokens": r["prompt_tokens"],
                    "generated_tokens": r["gen_tokens"],
                    "gen_ms": r["gen_ms"],
                    "tok_s": r["tok_s"],
                }
                for r in result["prompts"]
            ],
            "aggregate": aggregate,
        }, f, indent=2)

    print(f"  results:       {RESULTS_DIR}")
    if baseline_established:
        print(f"  [baseline] established at {BASELINE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
