#!/usr/bin/env python3
"""Parity check: generate 50 greedy tokens, save, diff against pinned baseline.

Regression canary for the FP8 MoE decode optimization plan. Does NOT measure
quality — it catches changes that silently alter output. For Step 4 (non-MoE
FP8 quantization) this check is supplemented with a teacher-forced top-1
agreement probe; see docs/plans/2026-04-15-fp8-moe-decode-optimization.md.

Usage:
    uv run examples/parity_check.py        # generates; establishes baseline if absent

Outputs:
    results/parity/YYYYMMDD-HHMMSS.txt     # this run
    results/parity/baseline.txt            # pinned baseline (first run)

Exit codes:
    0 — match OR baseline established
    1 — divergence above 5% token mismatch
    2 — environment failure (no CUDA, no checkpoint, etc.)
"""
from __future__ import annotations

import ast
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

import aeo_quant  # noqa: F401 — triggers numpy compat shim
from aeo_quant.bridges.gemma4.loader import load_gemma4_fp8
from aeo_quant.core.config import load_dotenv, setup_cuda_allocator

load_dotenv()
setup_cuda_allocator()

FP8_CHECKPOINT = os.environ.get("FP8_CHECKPOINT")
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "google/gemma-4-26B-A4B-it")
KV_BITS = int(os.environ.get("KV_BITS", "4"))
GEN_TOKENS = 50

PROMPT = (
    "You are a senior Python engineer.\n\n"
    "Design a thread-safe priority task queue with support for "
    "task cancellation, timeouts, and dead-letter handling. "
    "Show the full implementation with type hints."
)

RESULTS_DIR = Path("results/parity")
BASELINE_PATH = Path("tests/fixtures/parity_baseline.txt")


def load_token_ids(path: Path) -> list[int]:
    for line in path.read_text().splitlines():
        if line.startswith("# all_token_ids: "):
            return ast.literal_eval(line[len("# all_token_ids: "):])
    raise ValueError(f"no token-ids line in {path}")


def main() -> int:
    if not FP8_CHECKPOINT:
        print("[FATAL] FP8_CHECKPOINT not set", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available", file=sys.stderr)
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    model = load_gemma4_fp8(FP8_CHECKPOINT)

    from turboquant import TurboQuantCache
    cache = TurboQuantCache(bits=KV_BITS)

    messages = [
        {"role": "system", "content": "You are a senior Python engineer."},
        {"role": "user", "content": PROMPT},
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

    torch.manual_seed(0)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GEN_TOKENS,
            past_key_values=cache,
            use_cache=True,
            do_sample=False,
        )

    new_ids = outputs[0, inputs["input_ids"].shape[-1]:].tolist()
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"{ts}.txt"
    out_path.write_text(
        f"# parity_check {ts}\n"
        f"# gen_tokens: {len(new_ids)}\n"
        f"# all_token_ids: {new_ids}\n"
        f"# ---\n"
        f"{decoded}\n"
    )
    print(f"[parity] wrote {out_path}")

    if not BASELINE_PATH.exists():
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(out_path.read_text())
        print(f"[parity] established baseline at {BASELINE_PATH}")
        return 0

    baseline_ids = load_token_ids(BASELINE_PATH)
    n = min(len(baseline_ids), len(new_ids))
    mismatches = sum(1 for a, b in zip(baseline_ids, new_ids, strict=False) if a != b)
    pct = 100 * mismatches / n if n else 0.0

    max_prefix = 0
    for a, b in zip(baseline_ids, new_ids, strict=False):
        if a != b:
            break
        max_prefix += 1

    print(f"[parity] mismatches: {mismatches}/{n} ({pct:.1f}%)")
    print(f"[parity] max matching prefix: {max_prefix} tokens")
    if mismatches == 0:
        print("[parity] PASS — byte-for-byte match with baseline")
        return 0
    if pct > 5:
        print(f"[parity] FAIL — divergence above 5% ({pct:.1f}%)")
        return 1
    print(f"[parity] WARN — divergence below 5% threshold ({pct:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
