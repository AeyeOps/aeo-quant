"""Reasoning workload: hard prompts that stress attention precision.

Runs a list of PromptSpec entries through greedy decode with a TurboQuant KV
cache, returning per-prompt token ids and decoded text so the client can write
per-prompt output files, diff against baselines, and establish new baselines.

Pure compute — no filesystem I/O. Output files, timing captures, and
baseline establishment are the caller's responsibility.

PromptSpec shape: ``{"name": str, "file": str, "system": str, "user": str}``.
The ``file`` field is the per-prompt output filename that travels through the
workload unchanged and appears in every returned record, so existing baselines
at ``results/reasoning/baseline_{kv_bits}bit/`` remain valid without renaming.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from aeo_quant.bridges.gemma4.cache import Gemma4HybridTurboQuantCache

DEFAULT_REASONING_SUITE: list[dict] = [
    {
        "name": "math_proof",
        "file": "prompt_1_math.txt",
        "system": "You are a pure mathematician. Write rigorous proofs.",
        "user": (
            "Prove that for any group G of order p²q, where p and q are "
            "distinct primes with p² < q, the Sylow q-subgroup of G is normal.\n\n"
            "Show every step: state Sylow's theorems, enumerate the "
            "constraints on n_q, eliminate all possibilities except n_q = 1, "
            "and conclude normality from uniqueness."
        ),
    },
    {
        "name": "lru_bugs",
        "file": "prompt_2_lru.txt",
        "system": "You are a senior systems engineer specializing in concurrent data structures.",
        "user": (
            "The following concurrent LRU cache has exactly 4 bugs that interact "
            "with each other. Find all four, explain how they interact, and show "
            "the corrected code.\n\n"
            "```python\n"
            "import threading\n"
            "from collections import OrderedDict\n"
            "\n"
            "class ConcurrentLRUCache:\n"
            "    def __init__(self, capacity=10, initial_items={}):\n"
            "        self.capacity = capacity\n"
            "        self.cache = OrderedDict(initial_items)\n"
            "        self.lock = threading.Lock()\n"
            "        self.eviction_lock = threading.Lock()\n"
            "        self.hits = 0\n"
            "        self.misses = 0\n"
            "\n"
            "    def get(self, key):\n"
            "        with self.lock:\n"
            "            if key in self.cache:\n"
            "                self.cache.move_to_end(key)\n"
            "                self.hits += 1\n"
            "                return self.cache[key]\n"
            "            self.misses += 1\n"
            "            return None\n"
            "\n"
            "    def put(self, key, value):\n"
            "        if key in self.cache:\n"
            "            with self.lock:\n"
            "                self.cache[key] = value\n"
            "                self.cache.move_to_end(key)\n"
            "            return\n"
            "        with self.lock:\n"
            "            self.cache[key] = value\n"
            "        if len(self.cache) >= self.capacity:\n"
            "            self._evict()\n"
            "\n"
            "    def _evict(self):\n"
            "        with self.eviction_lock:\n"
            "            with self.lock:\n"
            "                if len(self.cache) >= self.capacity:\n"
            "                    self.cache.popitem(last=False)\n"
            "\n"
            "    def stats(self):\n"
            "        return {'hits': self.hits, 'misses': self.misses,\n"
            "                'size': len(self.cache)}\n"
            "```\n\n"
            "The 4 bugs are:\n"
            "1. A mutable default argument\n"
            "2. A race condition in put() (check-then-act without lock)\n"
            "3. A deadlock-prone lock ordering in _evict()\n"
            "4. An off-by-one in the capacity check\n\n"
            "For each bug: quote the exact line(s), explain the failure scenario, "
            "and show the fix."
        ),
    },
]


def run(
    model,
    tokenizer,
    *,
    prompts: list[dict] | None = None,
    gen_tokens: int = 500,
    kv_bits: int = 4,
    enable_thinking: bool = True,
    emit: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run each prompt in sequence and return per-prompt results + aggregate.

    Returned dict::

        {
          "prompts": [
            {"name": str, "file": str, "prompt_tokens": int, "gen_tokens": int,
             "all_token_ids": list[int], "decoded": str,
             "gen_ms": float, "tok_s": float},
            ...
          ],
          "aggregate": {"total_tokens": int, "total_ms": float, "avg_tok_s": float},
        }

    ``emit`` is the harness streaming callback; if None, no events are emitted.
    """
    if prompts is None:
        prompts = DEFAULT_REASONING_SUITE

    results: list[dict] = []
    for idx, spec in enumerate(prompts, start=1):
        if emit is not None:
            emit({"type": "prompt_start", "idx": idx, "name": spec["name"]})

        messages = [
            {"role": "system", "content": spec["system"]},
            {"role": "user", "content": spec["user"]},
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        n_prompt = inputs["input_ids"].shape[-1]

        cache = Gemma4HybridTurboQuantCache(bits=kv_bits, config=model.config)
        torch.manual_seed(0)

        t = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_tokens,
                past_key_values=cache,
                use_cache=True,
                do_sample=False,
            )
        gen_ms = (time.time() - t) * 1000.0

        new_ids = outputs[0, n_prompt:].tolist()
        decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
        n_new = len(new_ids)
        tok_s = n_new / (gen_ms / 1000.0) if gen_ms > 0 else 0.0

        record = {
            "name": spec["name"],
            "file": spec["file"],
            "prompt_tokens": int(n_prompt),
            "gen_tokens": int(n_new),
            "all_token_ids": new_ids,
            "decoded": decoded,
            "gen_ms": round(gen_ms, 1),
            "tok_s": round(tok_s, 2),
        }
        results.append(record)

        if emit is not None:
            emit({
                "type": "prompt_complete",
                "idx": idx,
                "name": spec["name"],
                "gen_ms": record["gen_ms"],
                "tok_s": record["tok_s"],
            })

        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_tokens = sum(r["gen_tokens"] for r in results)
    total_ms = sum(r["gen_ms"] for r in results)
    avg_tok_s = total_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0

    return {
        "prompts": results,
        "aggregate": {
            "total_tokens": total_tokens,
            "total_ms": round(total_ms, 1),
            "avg_tok_s": round(avg_tok_s, 2),
        },
    }
