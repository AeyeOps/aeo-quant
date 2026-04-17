#!/usr/bin/env python3
"""Profile a NVFP4-native decode step — where does per-token time go now?

Loads the full Gemma 4 26B-A4B NVFP4 checkpoint with the native kernel
path enabled, warms up torch.compile, then uses ``torch.profiler`` to
capture 10 decode steps and reports the breakdown.

Launch-bound was the diagnosis on the per-expert 2D path (240 Triton
launches per token at ~0.5 ms each → ~234 ms CPU per token vs ~78 ms
CUDA). The 3D fused-experts kernel cut per-MoE-layer launches from 8
to 2, and we now run ~75 ms per token — effectively at the CUDA
floor for this kernel. Use this profiler to spot the next bottleneck
before picking a lever (on-device alpha, flash-attention-2, etc.).

Usage::

    TRITON_OVERRIDE_ARCH=sm120 AEO_NVFP4_NATIVE=1 QUANT_FORMAT=nvfp4 \\
        uv run python examples/profile_nvfp4_decode.py

Output: top-20 ops sorted by CPU time and top-20 by CUDA time. The
gap (CPU time − CUDA time) is the launch-overhead component.
"""
from __future__ import annotations

import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoTokenizer

from aeo_quant.bridges.gemma4.loader import load_gemma4
from aeo_quant.core.config import load_dotenv, quant_env, setup_cuda_allocator
from aeo_quant.gpu.memory import mem_report, preflight_memory


MIN_FREE_GB = 18.0


def main() -> int:
    load_dotenv()
    setup_cuda_allocator()

    if os.environ.get("AEO_NVFP4_NATIVE") != "1":
        print("[FATAL] AEO_NVFP4_NATIVE must be set to 1", file=sys.stderr)
        return 2
    fmt, ckpt, kv = quant_env()
    if fmt != "nvfp4":
        print(f"[FATAL] QUANT_FORMAT must be 'nvfp4'", file=sys.stderr)
        return 2

    preflight_memory(MIN_FREE_GB, label="profile_nvfp4_decode")

    tokenizer_id = os.environ.get("TOKENIZER_ID", "google/gemma-4-26B-A4B-it")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    print(f"--- loading native NVFP4 model ---")
    t0 = time.monotonic()
    model = load_gemma4(ckpt, quant_format="nvfp4")
    model.eval()
    print(f"[load] {time.monotonic() - t0:.1f}s")
    mem_report("after_load")

    prompt = "The quick brown fox jumps over the"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Warmup to trigger compile
    print("\n--- warmup (2 × 5 tokens) ---")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        torch.cuda.synchronize()
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        torch.cuda.synchronize()

    # Profiled decode
    n_tokens = 10
    print(f"\n--- profiling {n_tokens}-token generate ---")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        t0 = time.monotonic()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - t0

    new_toks = out.shape[1] - inputs["input_ids"].shape[1]
    print(f"\n[profiled] {elapsed * 1000:.1f}ms for {new_toks} tokens "
          f"→ {new_toks/elapsed:.2f} tok/s\n")

    # Get the tabular views
    print("=== top 20 by CPU self time ===")
    print(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=20,
    ))

    print("\n=== top 20 by CUDA time ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=20,
    ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
