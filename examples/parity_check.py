#!/usr/bin/env python3
"""Parity check: generate 50 greedy tokens, save, diff against pinned baseline.

Regression canary for the FP8 MoE decode optimization plan. Does NOT measure
quality — it catches changes that silently alter output. For Step 4 (non-MoE
FP8 quantization) this check is supplemented with a teacher-forced top-1
agreement probe; see docs/plans/2026-04-15-fp8-moe-decode-optimization.md.

Usage:
    uv run examples/parity_check.py        # generates; establishes baseline if absent

Outputs:
    results/parity/YYYYMMDD-HHMMSS/output.txt     # this run (per-run subdir, SDK-standard)
    tests/fixtures/parity_baseline_{fp8,nvfp4}.txt  # per-format baseline

Baselines are per-format. NVFP4 runs also report divergence vs the FP8 baseline
as an informational quality delta (not a gate).

Exit codes:
    0 — match OR baseline established
    1 — divergence above 5% token mismatch vs own-format baseline
    2 — environment failure (no CUDA, no checkpoint, etc.)
"""
from __future__ import annotations

import ast
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import aeo_quant  # noqa: F401 — triggers numpy compat shim
from aeo_quant.core.config import load_dotenv, quant_env, results_dir, setup_cuda_allocator
from aeo_quant.gpu.memory import mem_report, preflight_memory

# Memory budget (unified LPDDR5X on GB10): load ~30 GB + torch.compile warmup
# ~10-15 GB + 5 GB safety. Fails fast if baseline is too high.
MIN_FREE_GB = 50.0

load_dotenv()
setup_cuda_allocator()

QUANT_FORMAT, CHECKPOINT, KV_BITS = quant_env()
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "google/gemma-4-26B-A4B-it")
GEN_TOKENS = 50

PROMPT = (
    "You are a senior Python engineer.\n\n"
    "Design a thread-safe priority task queue with support for "
    "task cancellation, timeouts, and dead-letter handling. "
    "Show the full implementation with type hints."
)

RESULTS_DIR = results_dir("parity")  # results/parity/YYYYMMDD-HHMMSS/
BASELINE_DIR = Path("tests/fixtures")
OWN_BASELINE = BASELINE_DIR / f"parity_baseline_{QUANT_FORMAT}.txt"
FP8_BASELINE = BASELINE_DIR / "parity_baseline_fp8.txt"


def load_token_ids(path: Path) -> list[int]:
    for line in path.read_text().splitlines():
        if line.startswith("# all_token_ids: "):
            return ast.literal_eval(line[len("# all_token_ids: "):])
    raise ValueError(f"no token-ids line in {path}")


def main() -> int:
    print(
        f"[parity] QUANT_FORMAT={QUANT_FORMAT} CHECKPOINT={CHECKPOINT} KV_BITS={KV_BITS}",
        flush=True,
    )
    preflight_memory(MIN_FREE_GB, label="parity")
    mem_report("parity:start")

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available", file=sys.stderr)
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[parity] loading tokenizer: {TOKENIZER_ID}", flush=True)
    t = time.time()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    print(f"[parity] tokenizer loaded in {time.time() - t:.1f}s", flush=True)

    print(f"[parity] loading model from {CHECKPOINT}", flush=True)
    t = time.time()
    from aeo_quant.bridges.gemma4.loader import load_gemma4
    model = load_gemma4(str(CHECKPOINT), quant_format=QUANT_FORMAT)
    print(f"[parity] model loaded in {time.time() - t:.1f}s (incl. compile wrap)", flush=True)
    mem_report("parity:after model load")

    print(f"[parity] creating TurboQuant KV cache (bits={KV_BITS})", flush=True)
    from turboquant import TurboQuantCache
    cache = TurboQuantCache(bits=KV_BITS)

    print("[parity] tokenizing prompt", flush=True)
    messages = [
        {"role": "system", "content": "You are a senior Python engineer."},
        {"role": "user", "content": PROMPT},
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    n_prompt = inputs["input_ids"].shape[-1]
    print(f"[parity] prompt tokens: {n_prompt}", flush=True)

    print(f"[parity] generating {GEN_TOKENS} tokens (first call triggers compile)", flush=True)
    mem_report("parity:before generate")
    t = time.time()
    torch.manual_seed(0)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GEN_TOKENS,
            past_key_values=cache,
            use_cache=True,
            do_sample=False,
        )
    gen_elapsed = time.time() - t
    n_new = outputs.shape[-1] - n_prompt
    tok_per_s = n_new / gen_elapsed if gen_elapsed > 0 else 0.0
    print(
        f"[parity] generated {n_new} tokens in {gen_elapsed:.1f}s "
        f"({tok_per_s:.2f} tok/s incl. compile warmup)",
        flush=True,
    )
    mem_report("parity:after generate")

    new_ids = outputs[0, n_prompt:].tolist()
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True)

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
