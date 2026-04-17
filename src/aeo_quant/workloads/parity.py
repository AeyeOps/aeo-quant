"""Parity workload: generate N greedy tokens from a fixed prompt.

Pure compute — no filesystem I/O, no baseline comparison. The CLI wrapper
(``examples/parity_check.py``) handles writing ``output.txt``, diffing against
the pinned baseline, and exit codes. The harness server calls ``run()``
directly and returns the dict to the client over the socket.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
from turboquant import TurboQuantCache

DEFAULT_PROMPT = (
    "You are a senior Python engineer.\n\n"
    "Design a thread-safe priority task queue with support for "
    "task cancellation, timeouts, and dead-letter handling. "
    "Show the full implementation with type hints."
)

DEFAULT_SYSTEM = "You are a senior Python engineer."


def run(
    model,
    tokenizer,
    *,
    prompt: str = DEFAULT_PROMPT,
    system: str = DEFAULT_SYSTEM,
    gen_tokens: int = 50,
    kv_bits: int = 4,
    enable_thinking: bool = True,
    emit: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run the parity generation and return JSON-serializable results.

    Returned dict:
        gen_tokens: int
        prompt_tokens: int
        all_token_ids: list[int]
        decoded: str
        gen_elapsed_s: float
        tok_per_s: float

    ``emit`` is a streaming-event callback used by the harness; if None,
    no events are emitted (parity is a single short call — streaming has
    little to offer here beyond a start/end heartbeat).
    """
    if emit is not None:
        emit({"type": "started", "message": f"parity: generating {gen_tokens} tokens"})
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    n_prompt = inputs["input_ids"].shape[-1]

    cache = TurboQuantCache(bits=kv_bits)

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
    gen_elapsed = time.time() - t

    new_ids = outputs[0, n_prompt:].tolist()
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
    n_new = len(new_ids)
    tok_per_s = n_new / gen_elapsed if gen_elapsed > 0 else 0.0

    if emit is not None:
        emit({
            "type": "completed",
            "message": f"parity: {n_new} tokens in {gen_elapsed:.1f}s ({tok_per_s:.2f} tok/s)",
            "gen_tokens": n_new,
            "tok_per_s": round(tok_per_s, 3),
        })

    return {
        "gen_tokens": n_new,
        "prompt_tokens": n_prompt,
        "all_token_ids": new_ids,
        "decoded": decoded,
        "gen_elapsed_s": round(gen_elapsed, 3),
        "tok_per_s": round(tok_per_s, 3),
    }
