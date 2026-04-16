#!/usr/bin/env python3
"""Quick quality check — three diverse prompts, coherence verified.

Loads the FP8 checkpoint once, fires three prompts (code, natural language,
mixed), prints each response, and fails fast if output quality degrades.
Takes about 5 minutes. Use this to verify a checkpoint before longer tests.

Usage:
    uv run python examples/quality_check.py
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import aeo_quant  # noqa: F401 — triggers np.trapz compat shim before numpy is used
from aeo_quant.bridges.gemma4.loader import load_gemma4_fp8
from aeo_quant.core.coherence import check_output_coherent
from aeo_quant.core.config import load_dotenv, setup_cuda_allocator
from aeo_quant.gpu.memory import enforce_cap, mem_report

load_dotenv()
setup_cuda_allocator()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VRAM_CAP_GB = float(os.environ.get("VRAM_CAP_GB", "90.0"))
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "google/gemma-4-26B-A4B-it")
MAX_NEW_TOKENS = 512
KV_BITS = int(os.environ.get("KV_BITS", "4"))

FP8_CHECKPOINT = os.environ.get("FP8_CHECKPOINT")
if not FP8_CHECKPOINT:
    print(
        "[FATAL] FP8_CHECKPOINT not set. Add it to .env or export it.",
        file=sys.stderr,
    )
    sys.exit(1)
FP8_CHECKPOINT = Path(FP8_CHECKPOINT)

PROMPTS = [
    ("code_quicksort",
     "Write a Python quicksort function and briefly explain how it works."),
    ("nl_merkle_tree",
     "Explain in two paragraphs, without any code, what a Merkle tree is "
     "and why Git uses them."),
    ("mixed_pandas_ts",
     "I have a pandas DataFrame with a 'timestamp' column of ISO-8601 "
     "strings. Show me how to convert it to datetime and filter rows from "
     "the last 7 days. Explain each step briefly."),
]


def main() -> None:
    mem_report("start")

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available.", file=sys.stderr)
        sys.exit(1)
    if not FP8_CHECKPOINT.exists():
        print(f"[FATAL] checkpoint missing: {FP8_CHECKPOINT}", file=sys.stderr)
        sys.exit(1)

    print(f"[preflight] device: {torch.cuda.get_device_name(0)}")
    print(f"[preflight] checkpoint: {FP8_CHECKPOINT}")
    enforce_cap("preflight", VRAM_CAP_GB)

    try:
        from turboquant import TurboQuantCache
    except ImportError:
        print("[FATAL] turboquant not installed.", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    enforce_cap("after tokenizer", VRAM_CAP_GB)

    print("[load] FP8 model...")
    t0 = time.time()
    model = load_gemma4_fp8(str(FP8_CHECKPOINT))
    print(f"[load] done in {time.time() - t0:.1f}s")
    mem_report("model loaded")
    enforce_cap("after model load", VRAM_CAP_GB)

    passed = 0
    for idx, (label, prompt_text) in enumerate(PROMPTS, 1):
        print(f"\n[prompt {idx}/{len(PROMPTS)}] {label}")
        messages = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        n_input = int(inputs["input_ids"].shape[-1])

        cache = TurboQuantCache(bits=KV_BITS)
        t0 = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                past_key_values=cache, use_cache=True, do_sample=False,
            )
        gen_time = time.time() - t0
        enforce_cap(f"prompt {idx} after generate", VRAM_CAP_GB)

        n_new = int(outputs.shape[-1] - n_input)
        tok_per_s = n_new / gen_time if gen_time > 0 else 0.0
        new_ids = outputs[0][n_input:].tolist()
        decoded = tokenizer.decode(outputs[0][n_input:], skip_special_tokens=True)

        # Show the user what the model said
        print(f"\n{'─' * 60}")
        print(decoded if decoded else "<EMPTY>")
        print(f"{'─' * 60}")
        print(f"  {n_new} tokens in {gen_time:.1f}s ({tok_per_s:.1f} tok/s)")

        failures = check_output_coherent(decoded, new_ids)
        if tok_per_s < 3.0:
            failures.append(f"tok/s = {tok_per_s:.1f} (< 3.0)")

        if failures:
            for f in failures:
                print(f"  FAIL: {f}", file=sys.stderr)
            print(f"\n[FATAL] prompt {idx} ({label}) failed", file=sys.stderr)
            sys.exit(5 + idx)

        print("  PASS")
        passed += 1

        del cache, outputs, inputs
        gc.collect()

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[done] {passed}/{len(PROMPTS)} prompts passed")


if __name__ == "__main__":
    main()
