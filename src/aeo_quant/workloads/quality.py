"""Quality workload: fire a list of [label, prompt] pairs through greedy decode.

Returns per-prompt token ids, decoded text, and timing. Coherence checks and
tok/s thresholds are **not** applied here — those are pure post-processing
handled by the caller. Keeping the workload free of pass/fail semantics lets
callers reuse it with different thresholds.

Prompts arrive over the JSON wire protocol as lists (tuples coerce to lists in
transit); the annotation uses ``list[list[str]]`` to reflect what the workload
actually sees at runtime.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from aeo_quant.bridges.gemma4.cache import Gemma4HybridTurboQuantCache

DEFAULT_QUALITY_SUITE: list[list[str]] = [
    [
        "code_quicksort",
        "Write a Python quicksort function and briefly explain how it works.",
    ],
    [
        "nl_merkle_tree",
        "Explain in two paragraphs, without any code, what a Merkle tree is "
        "and why Git uses them.",
    ],
    [
        "mixed_pandas_ts",
        "I have a pandas DataFrame with a 'timestamp' column of ISO-8601 "
        "strings. Show me how to convert it to datetime and filter rows from "
        "the last 7 days. Explain each step briefly.",
    ],
]


def run(
    model,
    tokenizer,
    *,
    prompts: list[list[str]] | None = None,
    max_new_tokens: int = 512,
    kv_bits: int = 4,
    emit: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run each prompt in sequence and return per-prompt results.

    Returned dict::

        {"prompts": [{"label": str, "decoded": str, "new_ids": list[int],
                      "gen_ms": float, "tok_s": float,
                      "prompt_tokens": int, "gen_tokens": int}, ...]}

    Events per prompt:
      - ``{"type": "prompt_start", "idx": n, "label": str}``
      - ``{"type": "prompt_complete", "idx": n, "label": str,
           "gen_tokens": int, "tok_s": float, "gen_ms": float}``
    """
    if prompts is None:
        prompts = DEFAULT_QUALITY_SUITE

    results: list[dict] = []
    for idx, (label, prompt_text) in enumerate(prompts, start=1):
        if emit is not None:
            emit({"type": "prompt_start", "idx": idx, "label": label})

        messages = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        n_prompt = int(inputs["input_ids"].shape[-1])

        cache = Gemma4HybridTurboQuantCache(bits=kv_bits, config=model.config)
        t = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
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
            "label": label,
            "decoded": decoded,
            "new_ids": new_ids,
            "gen_ms": round(gen_ms, 1),
            "tok_s": round(tok_s, 2),
            "prompt_tokens": n_prompt,
            "gen_tokens": n_new,
        }
        results.append(record)

        if emit is not None:
            emit({
                "type": "prompt_complete",
                "idx": idx,
                "label": label,
                "gen_tokens": n_new,
                "tok_s": record["tok_s"],
                "gen_ms": record["gen_ms"],
            })

        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"prompts": results}
