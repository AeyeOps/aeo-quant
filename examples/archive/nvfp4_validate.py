#!/usr/bin/env python3
"""Validate TurboQuant KV cache against the community NVFP4 Gemma 4 26B-A4B
checkpoint, loaded raw via transformers + compressed-tensors. No forward
patches, no dequant shims — if compressed-tensors 0.15.0.1 handles the
fused 3D MoE experts in this checkpoint correctly, this should "just work".

The community claim on bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4:
  - 49 GB bf16 → 16.5 GB NVFP4 (W4A4)
  - "First community NVFP4 quantization of Gemma 4 26B-A4B-it"
  - Includes implementation patches for vLLM integration
  - Solves the modelopt-skips-3D-expert-Parameters issue somehow

What we don't yet know:
  - Whether the patches are in the checkpoint (custom modeling_*.py) or
    only land in vLLM
  - Whether transformers + compressed-tensors 0.15.0.1 can load this
    out of the box
  - Whether the prefill memory explosion at 16K we saw on the FP8 path
    is weight-related (NVFP4 fixes it, ~16 GB instead of 27 GB) or
    forward-path-related (NVFP4 doesn't help)

This script answers all three by trying the simplest possible load.

Usage:
  bash -c 'set -a; source .env; set +a; \
    LONG_CTX_TOKENS=4000 MAX_NEW_TOKENS=128 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python scripts/test_nvfp4_gemma4.py'
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import psutil
import torch

# turboquant 0.2.0 calls np.trapz which numpy 2.x removed.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

VRAM_CAP_GB = 90.0
MIN_AVAIL_GB = 70.0
MODEL_ID = "bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4"
# Use the original google/ tokenizer rather than whatever ships in the
# community NVFP4 repo — same tradeoff as the FP8 LargitData path.
TOKENIZER_ID = "google/gemma-4-26B-A4B-it"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
TURBOQUANT_BITS = 4
PROMPT = "Write a Python quicksort function and briefly explain how it works."
LONG_CTX_TOKENS = int(os.environ.get("LONG_CTX_TOKENS", "0"))

_GB = 1024**3


def gb(n_bytes: int) -> str:
    return f"{n_bytes / _GB:6.2f} GB"


def mem_report(label: str) -> None:
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    t_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    t_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    print(
        f"[mem] {label:<22} "
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
    print(f"[preflight] safety cap: {VRAM_CAP_GB:.0f} GB  |  min available: {MIN_AVAIL_GB:.0f} GB")

    if avail_gb < MIN_AVAIL_GB:
        print(
            f"[FATAL] need {MIN_AVAIL_GB:.0f} GB available, only {avail_gb:.1f} GB.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print(
            "[WARN] no HF_TOKEN in env. Source your .env first.",
            file=sys.stderr,
        )


def main() -> None:
    mem_report("start")
    preflight()
    enforce_cap("preflight")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from turboquant import TurboQuantCache
    except ImportError:
        print("[FATAL] turboquant not installed.", file=sys.stderr)
        sys.exit(1)

    mem_report("imports loaded")

    print(f"[load] tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    mem_report("tokenizer loaded")
    enforce_cap("after tokenizer")

    attn_impl = os.environ.get("ATTN_IMPL", "sdpa")
    print(f"[load] model: {MODEL_ID} (NVFP4)")
    print(f"[load] attn_implementation={attn_impl}")
    print("[load] this is a fresh download on first run (~16.5 GB)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=attn_impl,
    )
    print(f"[load] model loaded in {time.time() - t0:.1f}s")
    mem_report("model loaded")
    enforce_cap("after model load")

    if LONG_CTX_TOKENS > 0:
        print(f"[gen] padding prompt toward {LONG_CTX_TOKENS} tokens")
        filler_sentence = (
            "The quick brown fox jumps over the lazy dog while reviewing the "
            "quarterly earnings report and debating the merits of various "
            "sorting algorithms used in distributed database systems. "
        )
        base_ids = tokenizer(filler_sentence, add_special_tokens=False)["input_ids"]
        tokens_per_sentence = max(len(base_ids), 1)
        reps = max(1, (LONG_CTX_TOKENS - 100) // tokens_per_sentence + 1)
        filler = filler_sentence * reps
        filler_ids = tokenizer(filler, add_special_tokens=False)["input_ids"]
        keep = max(0, LONG_CTX_TOKENS - 100)
        filler_truncated = tokenizer.decode(filler_ids[:keep], skip_special_tokens=True)
        padded_user = (
            f"Context document:\n\n{filler_truncated}\n\n"
            f"End of document.\n\nTask: {PROMPT}"
        )
        messages = [{"role": "user", "content": padded_user}]
    else:
        messages = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_input_tokens = inputs["input_ids"].shape[-1]
    print(f"[gen] input tokens: {n_input_tokens}")
    if n_input_tokens <= 64:
        print(f"[gen] input ids: {inputs['input_ids'][0].tolist()}")
    else:
        print(f"[gen] first 20 ids: {inputs['input_ids'][0][:20].tolist()}")
        print(f"[gen] last 20 ids:  {inputs['input_ids'][0][-20:].tolist()}")
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
    print(f"[gen] first 20 new token ids: {new_ids[:20]}")
    print(f"[gen] unique token ids in output: {len(set(new_ids))}")

    decoded = tokenizer.decode(outputs[0][n_input_tokens:], skip_special_tokens=True)
    print("\n=== MODEL OUTPUT (skip_special_tokens=True) ===")
    print(decoded if decoded else "<EMPTY>")
    print("=== END ===\n")

    peak_torch = torch.cuda.max_memory_allocated()
    del model, tokenizer, cache, outputs, inputs
    gc.collect()
    torch.cuda.empty_cache()
    mem_report("cleaned up")

    print(f"[summary] peak torch allocated: {gb(peak_torch)}")
    print(f"[summary] status: OK  |  {n_new_tokens} tokens at {tok_per_s:.1f} tok/s")


if __name__ == "__main__":
    main()
