#!/usr/bin/env python3
"""Reasoning quality check: two hard prompts that stress attention precision.

Prompts:
  1. Math proof — Sylow subgroup normality (tracks abstract constraints)
  2. Concurrent LRU cache — find 4 interacting bugs (tracks shared mutable state)

Both produce short output (~200-300 tokens) but require deep reasoning that
degrades visibly when KV cache precision drops.

Usage:
    uv run examples/reasoning_check.py              # bits=4 (default)
    KV_BITS=3 uv run examples/reasoning_check.py    # test 3-bit KV cache

Outputs:
    results/reasoning/<timestamp>/
        prompt_1_math.txt       — full output + token IDs
        prompt_2_lru.txt        — full output + token IDs
        timing.json             — aggregate timing + memory
        stdout.log              — full console output

    If a baseline exists at results/reasoning/baseline_<bits>bit/,
    diffs against it and reports token-level match statistics.

Exit codes:
    0 — completed (quality is assessed by reading the output, not automated)
    2 — environment failure
"""
from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

import aeo_quant  # noqa: F401
from aeo_quant.core.config import load_dotenv, quant_env, results_dir, setup_cuda_allocator
from aeo_quant.core.writers import Tee
from aeo_quant.gpu.memory import CudaTimer, mem_report, preflight_memory

# Memory budget (unified LPDDR5X on GB10): load ~30 GB + torch.compile warmup
# ~15 GB + 10 GB safety (longer decode than parity). Fails fast if baseline too high.
MIN_FREE_GB = 55.0

load_dotenv()
setup_cuda_allocator()

QUANT_FORMAT, CHECKPOINT, KV_BITS = quant_env()
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "google/gemma-4-26B-A4B-it")
GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "500"))

RESULTS_DIR = results_dir("reasoning")
BASELINE_DIR = Path(f"results/reasoning/baseline_{KV_BITS}bit")


PROMPTS = [
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


def save_output(path: Path, name: str, token_ids: list[int], decoded: str,
                gen_time_ms: float):
    path.write_text(
        f"# reasoning_check: {name}\n"
        f"# kv_bits: {KV_BITS}\n"
        f"# gen_tokens: {len(token_ids)}\n"
        f"# gen_time_ms: {gen_time_ms:.1f}\n"
        f"# all_token_ids: {token_ids}\n"
        f"# ---\n"
        f"{decoded}\n"
    )


def load_token_ids(path: Path) -> list[int]:
    import ast
    for line in path.read_text().splitlines():
        if line.startswith("# all_token_ids: "):
            return ast.literal_eval(line[len("# all_token_ids: "):])
    raise ValueError(f"no token-ids line in {path}")


def diff_against_baseline(prompt_info: dict, new_ids: list[int]):
    baseline_file = BASELINE_DIR / prompt_info["file"]
    if not baseline_file.exists():
        return
    baseline_ids = load_token_ids(baseline_file)
    n = min(len(baseline_ids), len(new_ids))
    mismatches = sum(1 for a, b in zip(baseline_ids, new_ids, strict=False) if a != b)
    pct = 100 * mismatches / n if n else 0.0
    max_prefix = 0
    for a, b in zip(baseline_ids, new_ids, strict=False):
        if a != b:
            break
        max_prefix += 1
    print(f"  [diff] vs baseline: {mismatches}/{n} mismatches ({pct:.1f}%)")
    print(f"  [diff] max matching prefix: {max_prefix} tokens")


def main() -> int:
    preflight_memory(MIN_FREE_GB, label="reasoning")
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available", file=sys.stderr)
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _log = open(RESULTS_DIR / "stdout.log", "w")  # noqa: SIM115
    atexit.register(_log.close)
    sys.stdout = Tee(sys.__stdout__, _log)
    sys.stderr = Tee(sys.__stderr__, _log)

    print(f"[reasoning] device: {torch.cuda.get_device_name(0)}")
    print(f"[reasoning] kv_bits: {KV_BITS}")
    print(f"[reasoning] gen_tokens: {GEN_TOKENS}")
    mem_report("start")

    from aeo_quant.bridges.gemma4.loader import load_gemma4
    model = load_gemma4(str(CHECKPOINT), quant_format=QUANT_FORMAT)
    mem_report("model loaded")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    from turboquant import TurboQuantCache

    # Warmup for torch.compile
    _wc = TurboQuantCache(bits=KV_BITS)
    _wi = tokenizer("warmup", return_tensors="pt").to(model.device)
    with torch.inference_mode():
        model.generate(**_wi, max_new_tokens=1, past_key_values=_wc,
                        use_cache=True, do_sample=False)
    del _wc, _wi
    torch.cuda.empty_cache()

    all_results = []

    for i, prompt_info in enumerate(PROMPTS, 1):
        print(f"\n{'=' * 60}")
        print(f"[reasoning] prompt {i}/{len(PROMPTS)}: {prompt_info['name']}")
        print(f"{'=' * 60}")

        messages = [
            {"role": "system", "content": prompt_info["system"]},
            {"role": "user", "content": prompt_info["user"]},
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        n_prompt = inputs["input_ids"].shape[-1]

        cache = TurboQuantCache(bits=KV_BITS)
        torch.manual_seed(0)

        with CudaTimer(f"generate_{prompt_info['name']}") as timer, \
                torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=GEN_TOKENS,
                past_key_values=cache,
                use_cache=True,
                do_sample=False,
            )

        new_ids = outputs[0, n_prompt:].tolist()
        decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
        gen_ms = timer.elapsed_ms

        out_path = RESULTS_DIR / prompt_info["file"]
        save_output(out_path, prompt_info["name"], new_ids, decoded, gen_ms)

        n_gen = len(new_ids)
        tok_s = n_gen / (gen_ms / 1000) if gen_ms > 0 else 0

        print(f"  [timing] {n_gen} tokens in {gen_ms:.1f} ms = {tok_s:.2f} tok/s")
        print(f"  [output] {out_path}")
        diff_against_baseline(prompt_info, new_ids)

        all_results.append({
            "name": prompt_info["name"],
            "prompt_tokens": n_prompt,
            "generated_tokens": n_gen,
            "gen_ms": round(gen_ms, 1),
            "tok_s": round(tok_s, 2),
        })

        del cache
        torch.cuda.empty_cache()

    # Aggregate summary
    total_tokens = sum(r["generated_tokens"] for r in all_results)
    total_ms = sum(r["gen_ms"] for r in all_results)
    avg_tok_s = total_tokens / (total_ms / 1000) if total_ms > 0 else 0

    print(f"\n{'=' * 60}")
    print("[reasoning] aggregate")
    print(f"{'=' * 60}")
    print(f"  total tokens:  {total_tokens}")
    print(f"  total time:    {total_ms:.1f} ms")
    print(f"  avg tok/s:     {avg_tok_s:.2f}")
    print(f"  kv_bits:       {KV_BITS}")
    mem_report("final")

    # Save timing
    timing_path = RESULTS_DIR / "timing.json"
    with open(timing_path, "w") as f:
        json.dump({
            "kv_bits": KV_BITS,
            "gen_tokens_per_prompt": GEN_TOKENS,
            "prompts": all_results,
            "aggregate": {
                "total_tokens": total_tokens,
                "total_ms": round(total_ms, 1),
                "avg_tok_s": round(avg_tok_s, 2),
            },
        }, f, indent=2)

    print(f"  results:       {RESULTS_DIR}")

    # Establish baseline if absent
    if not BASELINE_DIR.exists():
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        for prompt_info in PROMPTS:
            src = RESULTS_DIR / prompt_info["file"]
            dst = BASELINE_DIR / prompt_info["file"]
            dst.write_text(src.read_text())
        print(f"  [baseline] established at {BASELINE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
