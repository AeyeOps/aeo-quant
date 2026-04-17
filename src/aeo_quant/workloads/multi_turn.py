"""Multi-turn conversation workload with KV cache reuse across turns.

Runs progressive coding prompts from ``aeo_quant.prompts.project_arc`` until the
context fills to 80% of ``target``. Each turn reuses the prior turn's KV cache
(no redundant prefill). Per-turn metrics land in three client-provided files
under ``out_dir``:

    run_<target>.jsonl        — per-turn metrics
    transcript_<target>.jsonl — conversation record (system + per-turn + extras)
    memtrail_<target>.csv     — before/after memory snapshots per turn

**Filesystem I/O exception:** workloads are otherwise pure-compute, but
multi-turn artifacts are too large (10s–100s of MB) to stream back over the
harness socket. Writes are sanctioned because ``out_dir`` is a caller-provided
path — the workload never chooses its own destination.

**``enable_thinking`` scope:** applies **only** to the turn-0
``apply_chat_template`` call that builds the initial system + first-user prompt.
Turns 1+ use ``incremental_turn_tokens`` which emits the model-turn marker
unconditionally. ``enable_thinking`` is not consulted for incremental turns.

**Streamer:** always uses :class:`HarnessStreamer`. ``emit`` must be non-None;
we raise ``RuntimeError`` on misuse rather than blowing up inside the streamer.
``emit`` is always set by ``server._worker``.

Events:
  - ``turn_start``       — at the start of each turn
  - ``thinking_text``    — thinking streamed in chunks (from streamer)
  - ``thinking_end``     — phase transition (from streamer)
  - ``answer_chunk``     — per coalesced text chunk (from streamer)
  - ``turn_complete``    — aggregate per-turn metrics
  - ``memory_warning``   — emitted from except MemoryCapExceeded handler
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import torch
from aeo_quant.bridges.gemma4.cache import Gemma4HybridTurboQuantCache

from aeo_quant.bridges.gemma4.parser import GEMMA4_PARSER
from aeo_quant.bridges.gemma4.streamer import HarnessStreamer
from aeo_quant.bridges.gemma4.template import incremental_turn_tokens
from aeo_quant.core.coherence import check_output_coherent
from aeo_quant.core.writers import CSVWriter, JSONLWriter, TranscriptWriter
from aeo_quant.gpu.memory import (
    _GB,
    MemoryCapExceeded,
    MemoryCapStoppingCriteria,
    enforce_cap,
    mem_report,
)
from aeo_quant.prompts.project_arc import SYSTEM_MESSAGE, select_prompt

TEMPLATE_OVERHEAD_PER_TURN = 20
MIN_USEFUL_GENERATION = 512
MEMORY_CHECK_EVERY_N_TOKENS = 100

MEMTRAIL_HEADER = [
    "turn", "label", "sys_used_gb", "sys_avail_gb", "torch_alloc_gb", "torch_peak_gb",
]


def run(
    model,
    tokenizer,
    *,
    target: int,
    out_dir: str,
    vram_cap_gb: float = 90.0,
    max_new_tokens: int = 10000,
    kv_bits: int = 4,
    max_turns: int | None = None,
    enable_thinking: bool = True,
    emit: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run a multi-turn conversation at the given context ``target``.

    Writes per-turn artifacts under ``out_dir`` and returns a run summary.
    """
    if emit is None:
        raise RuntimeError(
            "multi_turn requires an emit callback — invoke via the harness"
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics_path = out_path / f"run_{target}.jsonl"
    transcript_path = out_path / f"transcript_{target}.jsonl"
    memtrail_path = out_path / f"memtrail_{target}.csv"
    events_path = out_path / f"events_{target}.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    # Tee every emitted event to disk at the moment of emission. No event
    # type is filtered — thinking_text, answer_chunk, turn_start, turn_complete,
    # memory_warning, and anything added later all land here as NDJSON.
    # Line-buffered so a crash mid-turn still has everything the user saw
    # scroll by on the terminal. This is additive to run_<target>.jsonl and
    # transcript_<target>.jsonl, which stay as turn-close summaries.
    # Persistence failures propagate — if we can't save model output, that's
    # a loud problem, not a quiet one.
    _events_fh = events_path.open("a", buffering=1)
    _upstream_emit = emit

    def _tee_emit(event: dict) -> None:
        _events_fh.write(json.dumps(event, default=str) + "\n")
        _upstream_emit(event)

    emit = _tee_emit

    fill_threshold = int(target * 0.80)

    # Capture model weight size up-front (after load, before any KV growth).
    mem0 = mem_report("multi_turn:start")
    model_weight_gb = mem0["torch_alloc_gb"]

    metrics = JSONLWriter(metrics_path)
    transcript = TranscriptWriter(
        transcript_path, SYSTEM_MESSAGE,
        config={"target": target, "cap_gb": vram_cap_gb, "kv_cache_reuse": True},
    )
    memtrail = CSVWriter(memtrail_path, MEMTRAIL_HEADER)

    conversation_history: list[dict] = [{"role": "system", "content": SYSTEM_MESSAGE}]

    system_text = tokenizer.apply_chat_template(
        conversation_history, tokenize=False,
        add_generation_prompt=False, enable_thinking=enable_thinking,
    )
    context_tokens = len(tokenizer.encode(system_text, add_special_tokens=False))

    band_counters: dict[str, int] = {}
    turn = 0
    prev_eos_hit = True
    cumulative_wall_s = 0.0
    run_summary: dict[str, Any] = {
        "target": target, "fill_threshold": fill_threshold,
        "turns_completed": 0, "final_context_tokens": context_tokens,
        "final_fill_ratio": 0.0, "peak_sys_used_gb": 0.0, "error": None,
    }

    # KV cache persists across turns
    cache = Gemma4HybridTurboQuantCache(bits=kv_bits, config=model.config)
    cache_seq_len = 0

    try:
        while True:
            if context_tokens >= fill_threshold:
                break
            if fill_threshold - context_tokens < MIN_USEFUL_GENERATION:
                break
            if max_turns is not None and turn >= max_turns:
                break

            fill_ratio = context_tokens / target if target > 0 else 0
            prompt_label, prompt_text, prompt_difficulty = select_prompt(
                turn, fill_ratio, band_counters,
            )

            emit({
                "type": "turn_start",
                "turn": turn,
                "prompt_label": prompt_label,
                "prompt_difficulty": prompt_difficulty,
                "context_tokens": context_tokens,
                "fill_ratio": round(fill_ratio, 4),
            })

            conversation_history.append({"role": "user", "content": prompt_text})

            # Turn 0: full prompt. Turn N>0: incremental tokens only.
            if turn == 0:
                prompt_str = tokenizer.apply_chat_template(
                    conversation_history, tokenize=False,
                    add_generation_prompt=True, enable_thinking=enable_thinking,
                )
                inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
                n_incremental = int(inputs["input_ids"].shape[-1])
            else:
                inc_str = incremental_turn_tokens(prompt_text, prev_eos_hit)
                inc_ids = tokenizer.encode(
                    inc_str, add_special_tokens=False, return_tensors="pt"
                ).to(model.device)
                attn_mask = torch.ones(
                    1, cache_seq_len + inc_ids.shape[-1],
                    dtype=torch.long, device=model.device,
                )
                inputs = {"input_ids": inc_ids, "attention_mask": attn_mask}
                n_incremental = int(inc_ids.shape[-1])

            n_total_context = cache_seq_len + n_incremental
            n_input = n_incremental

            mem_before = mem_report(f"turn {turn} before generate")
            memtrail.write({
                "turn": turn, "label": "before_generate",
                "sys_used_gb": mem_before["sys_used_gb"],
                "sys_avail_gb": round((psutil.virtual_memory().available) / _GB, 2),
                "torch_alloc_gb": mem_before["torch_alloc_gb"],
                "torch_peak_gb": mem_before["torch_peak_gb"],
            })

            watchdog = MemoryCapStoppingCriteria(vram_cap_gb, MEMORY_CHECK_EVERY_N_TOKENS)
            streamer = HarnessStreamer(tokenizer, emit=emit, turn=turn)

            try:
                enforce_cap(f"turn {turn} after tokenization", vram_cap_gb)

                t0 = time.time()
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        past_key_values=cache,
                        use_cache=True,
                        do_sample=False,
                        stopping_criteria=[watchdog],
                        streamer=streamer,
                    )
                gen_time = time.time() - t0

                cache_seq_len = cache.get_seq_length()

                if watchdog.exceeded:
                    emit({
                        "type": "memory_warning",
                        "turn": turn,
                        "peak_gb": round(watchdog.peak_seen_gb, 2),
                        "cap_gb": vram_cap_gb,
                    })
                    raise MemoryCapExceeded(
                        f"watchdog stopped generation at turn {turn}: "
                        f"peak={watchdog.peak_seen_gb:.1f} GB > cap={vram_cap_gb:.0f} GB"
                    )

                enforce_cap(f"turn {turn} after generate", vram_cap_gb)
                mem_after = mem_report(f"turn {turn} after generate")
                memtrail.write({
                    "turn": turn, "label": "after_generate",
                    "sys_used_gb": mem_after["sys_used_gb"],
                    "sys_avail_gb": round((psutil.virtual_memory().available) / _GB, 2),
                    "torch_alloc_gb": mem_after["torch_alloc_gb"],
                    "torch_peak_gb": mem_after["torch_peak_gb"],
                })

                n_new = int(outputs.shape[-1] - n_input)
                tok_per_s = n_new / gen_time if gen_time > 0 else 0.0

                raw_text = tokenizer.decode(outputs[0][n_input:], skip_special_tokens=False)
                segments = GEMMA4_PARSER.parse(raw_text)

                segment_token_counts: dict[str, int] = {}
                segment_texts: dict[str, str] = {}
                for seg in segments:
                    text = seg.content
                    toks = (
                        len(tokenizer.encode(text, add_special_tokens=False))
                        if text else 0
                    )
                    segment_token_counts[seg.type] = (
                        segment_token_counts.get(seg.type, 0) + toks
                    )
                    segment_texts[seg.type] = segment_texts.get(seg.type, "") + text

                new_ids = outputs[0][n_input:].tolist()
                thinking_tokens = segment_token_counts.get("thinking", 0)
                answer_tokens = segment_token_counts.get("assistant", 0)
                unknown_tokens = segment_token_counts.get("unknown", 0)
                total_generated = sum(segment_token_counts.values())
                thinking_ratio = (
                    thinking_tokens / total_generated if total_generated > 0 else 0.0
                )
                answer_text = segment_texts.get("assistant", "")

                eos_hit = n_new < max_new_tokens
                prev_eos_hit = eos_hit

                coherence_failures = check_output_coherent(raw_text, new_ids)

                transcript.write_turn(
                    session_id=0,
                    session_topic=f"context_scaling_{target}",
                    turn_index=turn,
                    user_msg=prompt_text,
                    segments=segments,
                    raw_output=raw_text,
                    status="ok",
                    wall=gen_time,
                    ttft=float(streamer.ttft) if streamer.ttft is not None else 0.0,
                    prompt_tokens=n_total_context,
                    completion_tokens=n_new,
                    extra={
                        "segment_token_counts": segment_token_counts,
                        "thinking_tokens": thinking_tokens,
                        "thinking_ratio": round(thinking_ratio, 4),
                        "prompt_label": prompt_label,
                        "prompt_difficulty": prompt_difficulty,
                        "cache_seq_len": cache_seq_len,
                        "coherence_failures": coherence_failures or None,
                    },
                )

                conversation_history.append(
                    {"role": "assistant", "content": answer_text}
                )

                user_tokens = len(
                    tokenizer.encode(prompt_text, add_special_tokens=False)
                )
                context_tokens += (
                    user_tokens + answer_tokens + TEMPLATE_OVERHEAD_PER_TURN
                )

                cumulative_wall_s += gen_time

                record = {
                    "run_target": target,
                    "turn": turn,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "prompt_label": prompt_label,
                    "prompt_difficulty": prompt_difficulty,
                    "user_tokens": user_tokens,
                    "max_new_tokens": max_new_tokens,
                    "n_input_tokens": n_total_context,
                    "n_incremental_tokens": n_incremental,
                    "cache_seq_len": cache_seq_len,
                    "total_generated": total_generated,
                    "segment_token_counts": segment_token_counts,
                    "thinking_tokens": thinking_tokens,
                    "answer_tokens": answer_tokens,
                    "unknown_tokens": unknown_tokens,
                    "thinking_ratio": round(thinking_ratio, 4),
                    "context_tokens_before": (
                        context_tokens - user_tokens - answer_tokens
                        - TEMPLATE_OVERHEAD_PER_TURN
                    ),
                    "context_tokens_after": context_tokens,
                    "context_fill_ratio": round(context_tokens / target, 4),
                    "total_time_s": round(gen_time, 2),
                    "tok_per_s": round(tok_per_s, 2),
                    "ttft_s": (
                        round(streamer.ttft, 4) if streamer.ttft is not None else None
                    ),
                    "model_weight_gb": model_weight_gb,
                    "cumulative_wall_s": round(cumulative_wall_s, 2),
                    "sys_total_gb": mem_after["sys_total_gb"],
                    "sys_used_before_gb": mem_before["sys_used_gb"],
                    "sys_used_after_gb": mem_after["sys_used_gb"],
                    "torch_alloc_gb": mem_after["torch_alloc_gb"],
                    "torch_peak_gb": mem_after["torch_peak_gb"],
                    "eos_hit": eos_hit,
                    "coherence_failures": coherence_failures or None,
                    "error": None,
                }
                metrics.write(record)

                emit({
                    "type": "turn_complete",
                    "turn": turn,
                    "gen_ms": round(gen_time * 1000.0, 1),
                    "tok_s": round(tok_per_s, 2),
                    "ttft_s": (
                        round(streamer.ttft, 4) if streamer.ttft is not None else None
                    ),
                    "thinking_tokens": thinking_tokens,
                    "answer_tokens": answer_tokens,
                    "eos_hit": eos_hit,
                    "coherence_failures": coherence_failures or None,
                    "context_tokens_after": context_tokens,
                    "context_fill_ratio": round(context_tokens / target, 4),
                })

                run_summary["turns_completed"] = turn + 1
                run_summary["final_context_tokens"] = context_tokens
                run_summary["final_fill_ratio"] = round(context_tokens / target, 4)
                run_summary["peak_sys_used_gb"] = max(
                    run_summary["peak_sys_used_gb"], mem_after["sys_used_gb"],
                )

                del outputs, inputs
                gc.collect()

            except (torch.cuda.OutOfMemoryError, MemoryCapExceeded) as e:
                error_type = (
                    "OOM" if isinstance(e, torch.cuda.OutOfMemoryError)
                    else "cap_exceeded"
                )
                metrics.write({
                    "run_target": target, "turn": turn,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "prompt_label": prompt_label,
                    "prompt_difficulty": prompt_difficulty,
                    "error": error_type,
                })
                transcript.write_turn(
                    session_id=0, session_topic=f"context_scaling_{target}",
                    turn_index=turn, user_msg=prompt_text, assistant_msg="",
                    status="error", wall=0.0, ttft=0.0,
                    prompt_tokens=n_total_context, completion_tokens=0,
                    extra={"error": error_type},
                )
                run_summary["error"] = error_type
                run_summary["turns_completed"] = turn
                break

            except Exception as e:
                metrics.write({
                    "run_target": target, "turn": turn,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "prompt_label": prompt_label,
                    "prompt_difficulty": prompt_difficulty,
                    "error": str(e),
                })
                run_summary["error"] = str(e)
                run_summary["turns_completed"] = turn
                break

            turn += 1

    finally:
        del cache
        gc.collect()
        transcript.close()
        memtrail.close()
        _events_fh.close()

    run_summary["metrics_path"] = str(metrics_path)
    run_summary["transcript_path"] = str(transcript_path)
    run_summary["memtrail_path"] = str(memtrail_path)
    run_summary["events_path"] = str(events_path)
    return run_summary
