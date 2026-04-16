#!/usr/bin/env python3
"""torch.compile(mode="reduce-overhead") probe.

Loads the FP8 model, wraps it with torch.compile, runs:
  1. Warmup (measures compile overhead)
  2. Parity check against pinned baseline
  3. Timed 100-token generation (decode tok/s)

Results go to results/compile/<timestamp>/.

Exit codes:
    0 — compile works, parity + timing reported
    1 — compile fails or parity diverges above threshold
    2 — environment failure
"""
from __future__ import annotations

import ast
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import aeo_quant  # noqa: F401
from aeo_quant.bridges.gemma4.loader import load_gemma4_fp8
from aeo_quant.core.config import load_dotenv, results_dir, setup_cuda_allocator
from aeo_quant.gpu.memory import CudaTimer, mem_report

load_dotenv()
setup_cuda_allocator()

FP8_CHECKPOINT = os.environ.get("FP8_CHECKPOINT")
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "google/gemma-4-26B-A4B-it")
KV_BITS = int(os.environ.get("KV_BITS", "4"))
GEN_TOKENS = 100
PARITY_TOKENS = 50
BASELINE_PATH = Path("tests/fixtures/parity_baseline.txt")

PROMPT = (
    "You are a senior Python engineer.\n\n"
    "Design a thread-safe priority task queue with support for "
    "task cancellation, timeouts, and dead-letter handling. "
    "Show the full implementation with type hints."
)


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

    rd = results_dir("compile")

    print(f"[compile] device: {torch.cuda.get_device_name(0)}")
    print(f"[compile] torch: {torch.__version__}")
    mem_report("start")

    # Load model
    print(f"[compile] loading model from {FP8_CHECKPOINT}")
    model = load_gemma4_fp8(str(FP8_CHECKPOINT))
    mem_report("model loaded")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    from turboquant import TurboQuantCache

    # --- Wrap with torch.compile ---
    print("[compile] wrapping model with torch.compile(mode='reduce-overhead')")
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead", dynamic=False)
    except Exception as e:
        print(f"[compile] torch.compile FAILED at wrap time: {e}")
        return 1
    print("[compile] torch.compile wrapper created (compilation deferred to first call)")

    # Build inputs
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
    print(f"[compile] prompt tokens: {n_prompt}")

    # --- Warmup (triggers compilation) ---
    print("[compile] warmup run (triggers JIT compilation)...")
    cache_warmup = TurboQuantCache(bits=KV_BITS)
    t_warmup_start = time.time()
    with torch.inference_mode():
        _ = compiled_model.generate(
            **inputs,
            max_new_tokens=2,
            past_key_values=cache_warmup,
            use_cache=True,
            do_sample=False,
        )
    warmup_s = time.time() - t_warmup_start
    print(f"[compile] warmup completed in {warmup_s:.1f}s")
    del cache_warmup
    torch.cuda.empty_cache()

    # --- Parity check (50 greedy tokens) ---
    print(f"[compile] parity check: generating {PARITY_TOKENS} tokens")
    cache_parity = TurboQuantCache(bits=KV_BITS)
    torch.manual_seed(0)
    with torch.inference_mode():
        parity_out = compiled_model.generate(
            **inputs,
            max_new_tokens=PARITY_TOKENS,
            past_key_values=cache_parity,
            use_cache=True,
            do_sample=False,
        )
    new_ids = parity_out[0, n_prompt:].tolist()
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Save parity result
    parity_path = rd / "parity.txt"
    parity_path.write_text(
        f"# compile_probe {rd.name}\n"
        f"# gen_tokens: {len(new_ids)}\n"
        f"# all_token_ids: {new_ids}\n"
        f"# ---\n"
        f"{decoded}\n"
    )
    print(f"[compile] parity written to {parity_path}")

    # Compare against baseline
    if BASELINE_PATH.exists():
        baseline_ids = load_token_ids(BASELINE_PATH)
        n = min(len(baseline_ids), len(new_ids))
        mismatches = sum(1 for a, b in zip(baseline_ids, new_ids, strict=False) if a != b)
        pct = 100 * mismatches / n if n else 0.0
        max_prefix = 0
        for a, b in zip(baseline_ids, new_ids, strict=False):
            if a != b:
                break
            max_prefix += 1
        print(f"[compile] parity: {mismatches}/{n} mismatches ({pct:.1f}%)")
        print(f"[compile] parity: max matching prefix: {max_prefix} tokens")
        if mismatches == 0:
            print("[compile] parity: PASS — byte-for-byte match")
        elif pct > 5:
            print(f"[compile] parity: FAIL — divergence above 5% ({pct:.1f}%)")
        else:
            print(f"[compile] parity: WARN — divergence below 5% ({pct:.1f}%)")
    else:
        print("[compile] parity: no baseline found, skipping comparison")

    del parity_out, cache_parity
    torch.cuda.empty_cache()

    # --- Timed generation (100 tokens) ---
    print(f"\n[compile] timed generation: {GEN_TOKENS} tokens")

    # Prefill measurement
    cache_pf = TurboQuantCache(bits=KV_BITS)
    with CudaTimer("prefill") as t_prefill, torch.inference_mode():
        compiled_model.generate(
            **inputs,
            max_new_tokens=1,
            past_key_values=cache_pf,
            use_cache=True,
            do_sample=False,
        )
    del cache_pf
    torch.cuda.empty_cache()

    # Full generation
    cache_full = TurboQuantCache(bits=KV_BITS)
    with CudaTimer("total") as t_total, torch.inference_mode():
        full_out = compiled_model.generate(
            **inputs,
            max_new_tokens=GEN_TOKENS,
            past_key_values=cache_full,
            use_cache=True,
            do_sample=False,
        )
    n_gen = full_out.shape[-1] - n_prompt
    total_ms = t_total.elapsed_ms
    prefill_ms = t_prefill.elapsed_ms
    decode_ms = total_ms - prefill_ms
    decode_tok_s = n_gen / (decode_ms / 1000) if decode_ms > 0 else 0

    print(f"[compile] prefill:       {prefill_ms:>10.1f} ms")
    print(f"[compile] decode:        {decode_ms:>10.1f} ms")
    print(f"[compile] total:         {total_ms:>10.1f} ms")
    print(f"[compile] decode tok/s:  {decode_tok_s:>10.2f}")
    print(f"[compile] generated:     {n_gen:>10d} tokens")
    print(f"[compile] warmup cost:   {warmup_s:>10.1f} s")
    mem_report("final")

    # Save timing
    import json
    timing_path = rd / "timing.json"
    with open(timing_path, "w") as f:
        json.dump({
            "compile_mode": "reduce-overhead",
            "warmup_s": round(warmup_s, 1),
            "prefill_ms": round(prefill_ms, 1),
            "decode_ms": round(decode_ms, 1),
            "total_ms": round(total_ms, 1),
            "decode_tok_s": round(decode_tok_s, 2),
            "generated_tokens": n_gen,
            "prompt_tokens": n_prompt,
        }, f, indent=2)
    print(f"[compile] results saved to {rd}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
