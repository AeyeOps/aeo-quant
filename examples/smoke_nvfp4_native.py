#!/usr/bin/env python3
"""Minimal end-to-end NVFP4-native smoke test.

Loads the full Gemma 4 NVFP4 checkpoint with ``AEO_NVFP4_NATIVE=1``
and generates one token from a two-word prompt.  Intended as a
lightest-possible plan Gate 4 / 5 precursor.

Per-expert correctness is already validated by
``examples/test_nvfp4_bridge.py``.  This script exercises the full
forward path (all 30 layers, all experts, attention, embedding,
lm_head) once end-to-end.

MIN_FREE_GB here is 20 (vs profile_generate.py's 50) because the
native path avoids the NVFP4 -> FP8 dequant step that temporarily
doubles expert memory during load.

Usage::

    TRITON_OVERRIDE_ARCH=sm120 AEO_NVFP4_NATIVE=1 \\
        QUANT_FORMAT=nvfp4 \\
        uv run python examples/smoke_nvfp4_native.py

Exits 0 if one token generates; 1 on any error.
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

    if os.environ.get("AEO_NVFP4_NATIVE") != "1":
        print("[FATAL] AEO_NVFP4_NATIVE must be set to 1", file=sys.stderr)
        return 2
    if os.environ.get("TRITON_OVERRIDE_ARCH") != "sm120":
        print("[WARN] TRITON_OVERRIDE_ARCH is not 'sm120' — "
              "kernel will fall back to slow decomposition")

    fmt, ckpt, kv = quant_env()
    if fmt != "nvfp4":
        print(f"[FATAL] QUANT_FORMAT must be 'nvfp4', got {fmt!r}", file=sys.stderr)
        return 2

    print(f"=== nvfp4 native smoke test ===")
    print(f"checkpoint: {ckpt}")
    print(f"TRITON_OVERRIDE_ARCH = {os.environ.get('TRITON_OVERRIDE_ARCH')}")
    print(f"AEO_NVFP4_NATIVE = {os.environ.get('AEO_NVFP4_NATIVE')}")

    preflight_memory(MIN_FREE_GB, label="smoke_nvfp4_native")

    tokenizer_id = os.environ.get("TOKENIZER_ID", "google/gemma-4-26B-A4B-it")
    print(f"\n--- loading tokenizer {tokenizer_id} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception as e:
        print(f"[FATAL] tokenizer load: {e}", file=sys.stderr)
        return 1

    from aeo_quant.bridges.gemma4.loader import load_gemma4
    print("\n--- loading model with AEO_NVFP4_NATIVE=1 ---")
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
    prompt = "Hello world"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"\n--- generating 1 token from prompt {prompt!r} ({inputs['input_ids'].shape[1]} tokens input) ---")

    t0 = time.monotonic()
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    except Exception as e:
        print(f"[FATAL] generate: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 1
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)

    print(f"[generate] {elapsed * 1000:.1f}ms")
    print(f"\n[OUTPUT]: {decoded!r}")
    mem_report("after_generate")

    if out.shape[1] > inputs["input_ids"].shape[1]:
        print("\n[PASS] one token generated via native NVFP4 path")
        return 0
    print("\n[FAIL] no new token produced")
    return 1


if __name__ == "__main__":
    sys.exit(main())
