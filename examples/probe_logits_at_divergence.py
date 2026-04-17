#!/usr/bin/env python3
"""Inspect logits at a chosen generation step to diagnose near-ties.

Runs ``generate()`` with ``output_scores=True`` and dumps the top-K
logits at one target step, so a parity-divergence between two forward
paths can be attributed to a true arithmetic shift versus a
cliff-effect near-tie (two close tokens swap ranks under tiny
perturbation).

Usage::

    TRITON_OVERRIDE_ARCH=sm120 AEO_NVFP4_NATIVE=1 QUANT_FORMAT=nvfp4 \\
        uv run python examples/probe_logits_at_divergence.py

Prints the top-10 tokens with logits and softmax probabilities at a
fixed step (default: index 40, matches the Phase 6 divergence).
"""
from __future__ import annotations

import os
import sys

import torch

import aeo_quant  # numpy shim
from aeo_quant.core.config import load_dotenv, quant_env, setup_cuda_allocator
from aeo_quant.gpu.memory import preflight_memory

load_dotenv()
setup_cuda_allocator()

QUANT_FORMAT, CHECKPOINT, KV_BITS = quant_env()
preflight_memory(20, label="probe_logits")

print(f"[probe] QUANT_FORMAT={QUANT_FORMAT}")
print(f"[probe] AEO_NVFP4_NATIVE={os.environ.get('AEO_NVFP4_NATIVE', '(unset)')}")

from aeo_quant.bridges.gemma4.cache import Gemma4HybridTurboQuantCache
from aeo_quant.bridges.gemma4.loader import load_gemma4
from transformers import AutoTokenizer

from aeo_quant.workloads.parity import DEFAULT_PROMPT, DEFAULT_SYSTEM

# Load the same tokenizer / model the parity workload uses.
TOKENIZER_ID = "google/gemma-4-26B-A4B-it"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
model = load_gemma4(CHECKPOINT, quant_format=QUANT_FORMAT)

messages = [
    {"role": "system", "content": DEFAULT_SYSTEM},
    {"role": "user", "content": DEFAULT_PROMPT},
]
prompt_str = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
)
inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
n_prompt = inputs["input_ids"].shape[-1]
print(f"[probe] prompt tokens: {n_prompt}")

# Same KV cache + seed setup as parity.py
cache = Gemma4HybridTurboQuantCache(bits=KV_BITS, config=model.config)
torch.manual_seed(0)

GEN_TOKENS = 42  # one past the divergence at index 40
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=GEN_TOKENS,
        past_key_values=cache,
        use_cache=True,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

new_ids = out.sequences[0, n_prompt:].tolist()
print(f"[probe] generated tokens ({len(new_ids)}): {new_ids}")

# scores is a tuple of length gen_tokens, each element is the logits
# tensor JUST BEFORE that step's argmax was picked. scores[40] therefore
# gives the logits at the divergence position.
scores = out.scores
print(f"[probe] scores tuple length: {len(scores)}")

divergence_idx = 40  # zero-based; this is the 41st generated token
logits = scores[divergence_idx][0]  # shape (vocab,)
top_k = 10
top_vals, top_idx = torch.topk(logits, k=top_k)
probs = torch.softmax(logits, dim=-1)
top_probs = probs[top_idx]

BASELINE_TOKEN = 236761  # what the per-expert path picks
NEW_TOKEN_3D = 568       # what the 3D path picks

print(f"\n=== LOGITS AT TOKEN POSITION {divergence_idx} ===")
print(f"This run's argmax: token={int(top_idx[0])} logit={float(top_vals[0]):.4f} prob={float(top_probs[0]):.6f}")
print(f"\nTop-{top_k} tokens at step {divergence_idx}:")
for i in range(top_k):
    t = int(top_idx[i])
    v = float(top_vals[i])
    p = float(top_probs[i])
    decoded = tokenizer.decode([t])
    mark = ""
    if t == BASELINE_TOKEN:
        mark = "  <-- BASELINE argmax (per-expert path)"
    elif t == NEW_TOKEN_3D:
        mark = "  <-- 3D path argmax"
    print(f"  #{i}: token={t:>6d} logit={v:>9.4f} prob={p:.6f}  {decoded!r}{mark}")

# Direct comparison: what are the logits for BOTH contenders?
baseline_logit = float(logits[BASELINE_TOKEN])
new_logit = float(logits[NEW_TOKEN_3D])
baseline_prob = float(probs[BASELINE_TOKEN])
new_prob = float(probs[NEW_TOKEN_3D])
print(f"\nLogit for baseline token {BASELINE_TOKEN}: {baseline_logit:.6f} (prob={baseline_prob:.8f})")
print(f"Logit for 3D token       {NEW_TOKEN_3D}: {new_logit:.6f} (prob={new_prob:.8f})")
print(f"Margin (top over runner): {float(top_vals[0]) - float(top_vals[1]):.6f}")
print(f"|logit_baseline - logit_3d|: {abs(baseline_logit - new_logit):.6f}")
print(f"Logit scale (top-1 logit): {float(top_vals[0]):.4f}")
