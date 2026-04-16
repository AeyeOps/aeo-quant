#!/usr/bin/env python3
"""CUDA graph capture probe for _scaled_mm on Blackwell.

FP8-only — does not support QUANT_FORMAT env var. Both FP8 and NVFP4
use the same _scaled_mm path after load, so this probe validates both.

Answers one question: can we capture a single decode step (which runs
through Gemma4TextExpertsFP8._fp8_linear → torch._scaled_mm) inside a
CUDA graph and replay it with matching output?

Exit codes:
    0 — capture + replay succeeded, output matches eager within tolerance
    1 — capture or replay failed, or output diverges beyond tolerance
    2 — environment failure (no CUDA, no checkpoint, etc.)
"""
from __future__ import annotations

import os
import sys

import torch
from transformers import AutoTokenizer

import aeo_quant  # noqa: F401
from aeo_quant.bridges.gemma4.loader import load_gemma4_fp8
from aeo_quant.core.config import load_dotenv, setup_cuda_allocator

load_dotenv()
setup_cuda_allocator()

FP8_CHECKPOINT = os.environ.get("FP8_CHECKPOINT")
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "google/gemma-4-26B-A4B-it")
KV_BITS = int(os.environ.get("KV_BITS", "4"))


def main() -> int:
    if not FP8_CHECKPOINT:
        print("[FATAL] FP8_CHECKPOINT not set", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available", file=sys.stderr)
        return 2

    print(f"[probe] device: {torch.cuda.get_device_name(0)}")
    print(f"[probe] loading model from {FP8_CHECKPOINT}")
    model = load_gemma4_fp8(str(FP8_CHECKPOINT))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    from turboquant import TurboQuantCache

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello."},
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

    # --- Step 1: eager prefill + one decode token to warm up the cache ---
    print("[probe] running eager prefill + 2 decode tokens")
    cache = TurboQuantCache(bits=KV_BITS)
    with torch.inference_mode():
        eager_out = model.generate(
            **inputs,
            max_new_tokens=2,
            past_key_values=cache,
            use_cache=True,
            do_sample=False,
        )
    print(f"[probe] eager last token id: {eager_out[0, -1].item()}")

    # --- Step 2: attempt CUDA graph capture of one more decode step ---
    print("[probe] attempting CUDA graph capture of a single decode step")

    # Build the static input buffer for graph capture.
    # After generate(max_new_tokens=2), the last token is the input for the
    # next decode step.
    static_input_ids = eager_out[:, -1:].clone()

    try:
        # Warmup run (required before capture — populates internal state)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            warmup_out = model(
                input_ids=static_input_ids,
                past_key_values=cache,
                use_cache=True,
            )
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.inference_mode():
            graph_out = model(
                input_ids=static_input_ids,
                past_key_values=cache,
                use_cache=True,
            )

        print("[probe] CUDA graph capture succeeded")

    except Exception as e:
        print(f"[probe] CUDA graph capture FAILED: {e}")
        return 1

    # --- Step 3: replay and compare ---
    print("[probe] replaying captured graph")
    try:
        g.replay()
        replay_logits = graph_out.logits.clone()
    except Exception as e:
        print(f"[probe] CUDA graph replay FAILED: {e}")
        return 1

    # Compare warmup output vs replay output. Both used the same static
    # input, so logits should be identical (not just close — identical,
    # since the same kernels ran on the same data).
    warmup_logits = warmup_out.logits
    max_diff = (replay_logits - warmup_logits).abs().max().item()
    print(f"[probe] max abs diff (replay vs warmup): {max_diff:.6e}")

    if max_diff > 1e-3:
        print(f"[probe] FAIL — divergence too large ({max_diff:.6e} > 1e-3)")
        return 1

    print("[probe] PASS — CUDA graph capture + replay works with _scaled_mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
