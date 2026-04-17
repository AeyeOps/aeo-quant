#!/usr/bin/env python3
"""End-to-end NVFP4 smoke test.

Loads the full Gemma 4 NVFP4 checkpoint, does a 1-token warmup to
trigger ``torch.compile``, then generates ``GEN_TOKENS`` (default 50)
greedy tokens and reports tok/s.

Exercises the full forward path end-to-end (all 30 layers, all experts
through the 3D decode kernel, attention, embedding, lm_head). Per-expert
and per-kernel correctness live in ``test_nvfp4_bridge.py`` and
``test_nvfp4_3d_kernel.py``; this is the smallest "does the whole stack
compose" check.

Usage::

    TRITON_OVERRIDE_ARCH=sm120 QUANT_FORMAT=nvfp4 \\
        uv run python examples/smoke_nvfp4.py

Env overrides: ``GEN_TOKENS`` (default 50), ``SMOKE_PROMPT``.
Exits 0 if tokens generate; 1 on any error.
"""
from __future__ import annotations

import os
import sys
import time

import torch
from transformers import AutoTokenizer

from aeo_quant.core.config import load_dotenv, quant_env, setup_cuda_allocator
from aeo_quant.gpu.memory import mem_report, preflight_memory


MIN_FREE_GB = 20.0


def main() -> int:
    load_dotenv()
    setup_cuda_allocator()

    if os.environ.get("TRITON_OVERRIDE_ARCH") != "sm120":
        print("[WARN] TRITON_OVERRIDE_ARCH is not 'sm120' — "
              "kernel will fall back to slow decomposition")

    fmt, ckpt, kv = quant_env()
    if fmt != "nvfp4":
        print(f"[FATAL] QUANT_FORMAT must be 'nvfp4', got {fmt!r}", file=sys.stderr)
        return 2

    print(f"=== nvfp4 smoke test ===")
    print(f"checkpoint: {ckpt}")
    print(f"TRITON_OVERRIDE_ARCH = {os.environ.get('TRITON_OVERRIDE_ARCH')}")

    preflight_memory(MIN_FREE_GB, label="smoke_nvfp4")

    tokenizer_id = os.environ.get("TOKENIZER_ID", "google/gemma-4-26B-A4B-it")
    print(f"\n--- loading tokenizer {tokenizer_id} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception as e:
        print(f"[FATAL] tokenizer load: {e}", file=sys.stderr)
        return 1

    from aeo_quant.bridges.gemma4.loader import load_gemma4
    print("\n--- loading model ---")
    t_load = time.monotonic()
    try:
        model = load_gemma4(ckpt, quant_format="nvfp4")
    except Exception as e:
        print(f"[FATAL] model load: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 1
    print(f"[load] {time.monotonic() - t_load:.1f}s")
    mem_report("after_load")

    model.eval()
    prompt = os.environ.get(
        "SMOKE_PROMPT",
        "The capital of France is",
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"\n--- warmup from prompt {prompt!r} ({inputs['input_ids'].shape[1]} tokens input) ---")

    t0 = time.monotonic()
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    except Exception as e:
        print(f"[FATAL] warmup generate: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 1
    torch.cuda.synchronize()
    warmup_ms = (time.monotonic() - t0) * 1000
    print(f"[warmup] {warmup_ms:.1f}ms for 1 token (compile + prefill)")

    n_tokens = int(os.environ.get("GEN_TOKENS", "50"))
    print(f"\n--- generating {n_tokens} tokens ({prompt!r}) ---")
    t0 = time.monotonic()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    tok_per_s = new_tokens / elapsed
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)

    print(f"\n[generate] {elapsed:.2f}s for {new_tokens} tokens")
    print(f"[tok/s] {tok_per_s:.2f}")
    print(f"\n[OUTPUT]: {decoded!r}")
    mem_report("after_generate")

    if out.shape[1] > inputs["input_ids"].shape[1]:
        print(f"\n[PASS] {new_tokens} tokens at {tok_per_s:.2f} tok/s")
        return 0
    print("\n[FAIL] no new tokens produced")
    return 1


if __name__ == "__main__":
    sys.exit(main())
