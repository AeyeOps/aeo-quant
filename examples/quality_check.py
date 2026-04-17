#!/usr/bin/env python3
"""Quick quality check — three diverse prompts, coherence verified.

Runs via the aeo-quant harness daemon. The harness holds whatever quantization
format was requested at daemon start; this script enforces a format-match
against ``QUANT_FORMAT`` before running, so switching formats is explicit.

Loads nothing in-process; dispatches the ``quality`` workload, prints each
response, runs ``check_output_coherent`` + a tok/s >= 3.0 gate on the client
side, and fails fast on any prompt that regresses.

Usage:
    uv run python examples/quality_check.py
    QUANT_FORMAT=nvfp4 uv run python examples/quality_check.py

Exit codes:
    0 — all prompts passed
    2 — environment failure (no harness, format mismatch, no CUDA, etc.)
    5+idx — prompt idx (1-based) failed coherence or tok/s gate
"""
from __future__ import annotations

import sys

import aeo_quant  # noqa: F401 — triggers np.trapz compat shim before numpy is used
from aeo_quant.core.coherence import check_output_coherent
from aeo_quant.core.config import load_dotenv, quant_env, setup_cuda_allocator
from aeo_quant.gpu.memory import mem_report
from aeo_quant.harness import HarnessUnavailable, get_or_start_harness

load_dotenv()
setup_cuda_allocator()

QUANT_FORMAT, CHECKPOINT, KV_BITS = quant_env()
MAX_NEW_TOKENS = 512
TOK_S_FLOOR = 3.0


def _on_event(event: dict) -> None:
    t = event.get("type")
    if t == "prompt_start":
        print(f"\n[prompt {event['idx']}] {event['label']}", flush=True)
    elif t == "prompt_complete":
        # Fuller detail printed client-side after we have decoded text.
        pass


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
        f"[quality] using harness (uptime={info.get('uptime_s')}s, "
        f"jobs_served={info.get('jobs_served')}, queue={info.get('queue_depth')})",
        flush=True,
    )
    return 0


def main() -> int:
    print(
        f"[quality] QUANT_FORMAT={QUANT_FORMAT} CHECKPOINT={CHECKPOINT} "
        f"KV_BITS={KV_BITS}",
        flush=True,
    )
    mem_report("quality:start")

    try:
        client = get_or_start_harness(preflight_label="quality")
    except HarnessUnavailable as e:
        print(f"[FATAL] harness unavailable: {e}", file=sys.stderr)
        return 2

    rc = _check_harness_format(client)
    if rc != 0:
        return rc

    result = client.run_workload(
        "quality",
        on_event=_on_event,
        max_new_tokens=MAX_NEW_TOKENS,
        kv_bits=KV_BITS,
    )

    passed = 0
    records = result["prompts"]
    for idx, record in enumerate(records, start=1):
        decoded = record["decoded"]
        print(f"\n{'─' * 60}")
        print(decoded if decoded else "<EMPTY>")
        print(f"{'─' * 60}")
        print(
            f"  {record['gen_tokens']} tokens in {record['gen_ms'] / 1000:.1f}s "
            f"({record['tok_s']:.1f} tok/s)"
        )

        failures = check_output_coherent(decoded, record["new_ids"])
        if record["tok_s"] < TOK_S_FLOOR:
            failures.append(f"tok/s = {record['tok_s']:.1f} (< {TOK_S_FLOOR})")

        if failures:
            for f in failures:
                print(f"  FAIL: {f}", file=sys.stderr)
            print(
                f"\n[FATAL] prompt {idx} ({record['label']}) failed",
                file=sys.stderr,
            )
            return 5 + idx

        print("  PASS")
        passed += 1

    print(f"\n[done] {passed}/{len(records)} prompts passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
