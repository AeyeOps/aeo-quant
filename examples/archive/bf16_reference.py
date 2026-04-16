#!/usr/bin/env python3
"""Phase 7 quality reference: load google/gemma-4-26B-A4B-it at bf16 and run
the same prompt with the same settings as scripts/test_gemma4_fp8_load.py.

Produces a token-id sequence we can diff against the Phase 6 FP8 output to
measure FP8 quantization noise. Same prompt, same MAX_NEW_TOKENS, same
TurboQuantCache(bits=4), same greedy decoding. The only difference vs the
FP8 test is the source of the weights: native google bf16 vs our
self-built FP8 shards.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        uv run python scripts/test_gemma4_bf16_reference.py
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

VRAM_CAP_GB = 90.0
MIN_AVAIL_GB = 70.0  # bf16 weights ~52 GB + activations + baseline
MODEL_ID = "google/gemma-4-26B-A4B-it"
PROMPT = "Write a Python quicksort function and briefly explain how it works."
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
TURBOQUANT_BITS = 4

_GB = 1024**3


def gb(n_bytes: int) -> str:
    return f"{n_bytes / _GB:6.2f} GB"


def mem_report(label: str) -> None:
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    t_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    t_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    print(
        f"[mem] {label:<32} "
        f"sys_used={gb(vm.used)}  "
        f"sys_avail={gb(vm.available)}  "
        f"proc_rss={gb(rss)}  "
        f"torch_alloc={gb(t_alloc)}  "
        f"torch_peak={gb(t_peak)}",
        flush=True,
    )


def enforce_cap(label: str) -> None:
    vm = psutil.virtual_memory()
    if vm.used > VRAM_CAP_GB * _GB:
        print(
            f"\n[FATAL] unified memory cap exceeded at '{label}': "
            f"sys_used={gb(vm.used)} > cap={VRAM_CAP_GB:.0f} GB",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)


def preflight() -> None:
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available — GPU-only per project rules.", file=sys.stderr)
        sys.exit(1)

    dev_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    vm = psutil.virtual_memory()
    avail_gb = vm.available / _GB

    print(f"[preflight] device: {dev_name} (sm_{cc_major}{cc_minor})")
    print(f"[preflight] torch: {torch.__version__}")
    print(f"[preflight] unified mem total:     {gb(vm.total)}")
    print(f"[preflight] unified mem available: {gb(vm.available)}")
    print(f"[preflight] model: {MODEL_ID} (native bf16)")
    print(
        f"[preflight] safety cap: {VRAM_CAP_GB:.0f} GB  |  "
        f"min available to start: {MIN_AVAIL_GB:.0f} GB"
    )

    if avail_gb < MIN_AVAIL_GB:
        print(
            f"[FATAL] need {MIN_AVAIL_GB:.0f} GB available, only {avail_gb:.1f} GB. "
            f"Check contention (vLLM/trtllm/ollama/etc.) before retrying.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print(
            "[WARN] neither HF_TOKEN nor HUGGING_FACE_HUB_TOKEN set — gated model.",
            file=sys.stderr,
        )


def main() -> None:
    mem_report("start")
    preflight()
    enforce_cap("preflight")
    mem_report("preflight done")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from turboquant import TurboQuantCache
    except ImportError:
        print(
            "[FATAL] turboquant not installed. Run: uv pip install turboquant",
            file=sys.stderr,
        )
        sys.exit(1)

    mem_report("imports loaded")

    print(f"[load] tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    mem_report("tokenizer loaded")
    enforce_cap("after tokenizer")

    print(f"[load] bf16 model: {MODEL_ID}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    load_time = time.time() - t0
    print(f"[load] model loaded in {load_time:.1f}s")
    mem_report("model loaded")
    enforce_cap("after model load")

    messages = [{"role": "user", "content": PROMPT}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"[gen] prompt after chat template: {prompt_str!r}")
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    n_input_tokens = int(inputs["input_ids"].shape[-1])
    print(f"[gen] input tokens: {n_input_tokens}")
    print(f"[gen] first 20 input ids: {inputs['input_ids'][0][:20].tolist()}")
    mem_report("inputs prepared")
    enforce_cap("after inputs")

    print(f"[gen] TurboQuantCache(bits={TURBOQUANT_BITS})")
    cache = TurboQuantCache(bits=TURBOQUANT_BITS)

    print(f"[gen] generating up to {MAX_NEW_TOKENS} tokens (greedy)")
    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            past_key_values=cache,
            use_cache=True,
            do_sample=False,
        )
    gen_time = time.time() - t0
    enforce_cap("after generate")
    mem_report("after generate")

    n_new_tokens = int(outputs.shape[-1] - n_input_tokens)
    tok_per_s = n_new_tokens / gen_time if gen_time > 0 else 0.0
    print(f"[gen] generated {n_new_tokens} tokens in {gen_time:.1f}s ({tok_per_s:.1f} tok/s)")

    new_ids = outputs[0][n_input_tokens:].tolist()
    print(f"[gen] ALL new token ids: {new_ids}")
    print(f"[gen] first 20 new token ids: {new_ids[:20]}")
    print(f"[gen] unique token ids in output: {len(set(new_ids))}")

    decoded_clean = tokenizer.decode(outputs[0][n_input_tokens:], skip_special_tokens=True)
    print("\n=== BF16 REFERENCE OUTPUT (skip_special_tokens=True) ===")
    print(decoded_clean if decoded_clean else "<EMPTY>")
    print("=== END ===\n")

    peak_torch = torch.cuda.max_memory_allocated()
    peak_sys_at_end = psutil.virtual_memory().used

    del model, tokenizer, cache, outputs, inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mem_report("cleaned up")

    print(f"[summary] peak torch allocated: {gb(peak_torch)}")
    print(f"[summary] sys_used at generate-end: {gb(peak_sys_at_end)}")
    print(
        f"[summary] {n_new_tokens} new tokens at {tok_per_s:.1f} tok/s "
        f"(bf16 reference, TurboQuantCache {TURBOQUANT_BITS}-bit)"
    )
    print("[summary] status: OK")


if __name__ == "__main__":
    main()
