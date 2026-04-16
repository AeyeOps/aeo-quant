#!/usr/bin/env python3
"""Multi-turn conversation benchmark.

Has a progressively harder coding conversation with the model across five
context window sizes (16K, 32K, 48K, 64K, 128K). Starts by asking the
model to build a task queue system, then layer on features, tests,
concurrency fixes, and architecture challenges as context grows.

Measures per turn: tok/s, memory, thinking vs answer ratio, generation time.
KV cache persists across turns (no redundant prefill).

Produces per target:
  - metrics JSONL    — per-turn numbers for analysis
  - transcript HTML  — the actual conversation, open in a browser to read
  - memory CSV       — per-turn memory time-series
  - dashboard PNG    — performance charts (tok/s, memory, thinking, time)
  - summary JSON     — per-target rollup

Usage:
    uv run python examples/multi_turn_32k.py

Set FP8_CHECKPOINT in .env or as an env var to point at your checkpoint.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aeo_quant  # noqa: F401 — triggers np.trapz compat shim before numpy is used
import numpy as np
import psutil
import torch
from transformers import AutoTokenizer

from aeo_quant.bridges.gemma4.loader import load_gemma4_fp8
from aeo_quant.bridges.gemma4.parser import GEMMA4_PARSER
from aeo_quant.bridges.gemma4.streamer import LiveStreamer
from aeo_quant.bridges.gemma4.template import incremental_turn_tokens
from aeo_quant.core.coherence import check_output_coherent
from aeo_quant.core.viewer import generate_html
from aeo_quant.core.writers import CSVWriter, JSONLWriter, TranscriptWriter
from aeo_quant.gpu.memory import (
    _GB,
    MemoryCapExceeded,
    MemoryCapStoppingCriteria,
    enforce_cap,
    gb,
    mem_report,
)
from aeo_quant.plots.context_scaling import generate_dashboard
from aeo_quant.prompts.project_arc import SYSTEM_MESSAGE, select_prompt

# .env is the final authority — overrides anything already in the shell
from aeo_quant.core.config import load_dotenv, setup_cuda_allocator

load_dotenv()  # .env overrides shell env vars
setup_cuda_allocator()

# ---------------------------------------------------------------------------
# Configuration — sensible defaults, overridable via env vars
# ---------------------------------------------------------------------------
VRAM_CAP_GB = float(os.environ.get("VRAM_CAP_GB", "90.0"))
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "google/gemma-4-26B-A4B-it")
MAX_NEW_TOKENS = 10000
TURBOQUANT_BITS = 4
TEMPLATE_OVERHEAD_PER_TURN = 20
MIN_USEFUL_GENERATION = 512
MEMORY_CHECK_EVERY_N_TOKENS = 100

CONTEXT_TARGETS = [32768]
MAX_TURNS = int(os.environ.get("MAX_TURNS", "0")) or None
LIVE = os.environ.get("LIVE", "1") != "0"
VERBOSE_THINK = os.environ.get("VERBOSE_THINK", "0") != "0"

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results/context_scaling_32k"))

FP8_CHECKPOINT = os.environ.get("FP8_CHECKPOINT")
if not FP8_CHECKPOINT:
    print(
        "[FATAL] FP8_CHECKPOINT not set. Either:\n"
        "  1. Add FP8_CHECKPOINT=/path/to/checkpoint to .env in this directory\n"
        "  2. Export FP8_CHECKPOINT=/path/to/checkpoint before running",
        file=sys.stderr,
    )
    sys.exit(1)
FP8_CHECKPOINT = Path(FP8_CHECKPOINT)

# Per-turn memory CSV header
MEMTRAIL_HEADER = [
    "turn", "label", "sys_used_gb", "sys_avail_gb", "torch_alloc_gb", "torch_peak_gb",
]


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def preflight() -> None:
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available — GPU-only.", file=sys.stderr)
        sys.exit(1)
    if not FP8_CHECKPOINT.exists():
        print(f"[FATAL] FP8 checkpoint missing at {FP8_CHECKPOINT}.", file=sys.stderr)
        sys.exit(1)

    dev_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    vm = psutil.virtual_memory()
    avail_gb = vm.available / _GB

    print(f"[preflight] device: {dev_name} (sm_{cc_major}{cc_minor})")
    print(f"[preflight] torch: {torch.__version__}")
    print(f"[preflight] unified mem available: {gb(vm.available)}")
    print(f"[preflight] fp8 checkpoint: {FP8_CHECKPOINT}")
    print(f"[preflight] targets: {CONTEXT_TARGETS}")
    print(f"[preflight] safety cap: {VRAM_CAP_GB:.0f} GB")

    if avail_gb < 50.0:
        print(f"[FATAL] need 50 GB available, only {avail_gb:.1f} GB.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Single run at a given context target
# ---------------------------------------------------------------------------
def run_context_target(target, model, tokenizer, TurboQuantCache, model_weight_gb: float) -> dict:
    fill_threshold = int(target * 0.80)

    # Output paths for this target
    metrics_path = RESULTS_DIR / f"run_{target}.jsonl"
    transcript_path = RESULTS_DIR / f"transcript_{target}.jsonl"
    memtrail_path = RESULTS_DIR / f"memtrail_{target}.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    print(f"\n{'=' * 60}")
    print(f"[run] target={target:,}  fill_threshold={fill_threshold:,}")
    print(f"{'=' * 60}")

    # Initialize writers
    metrics = JSONLWriter(metrics_path)
    transcript = TranscriptWriter(
        transcript_path, SYSTEM_MESSAGE,
        config={"target": target, "cap_gb": VRAM_CAP_GB, "kv_cache_reuse": True},
    )
    memtrail = CSVWriter(memtrail_path, MEMTRAIL_HEADER)

    conversation_history = [{"role": "system", "content": SYSTEM_MESSAGE}]

    system_text = tokenizer.apply_chat_template(
        conversation_history, tokenize=False,
        add_generation_prompt=False, enable_thinking=True,
    )
    context_tokens = len(tokenizer.encode(system_text, add_special_tokens=False))
    print(f"[run] system message tokens: {context_tokens}")

    band_counters: dict[str, int] = {}
    turn = 0
    prev_eos_hit = True
    cumulative_wall_s = 0.0
    run_summary = {
        "target": target, "fill_threshold": fill_threshold,
        "turns_completed": 0, "final_context_tokens": context_tokens,
        "final_fill_ratio": 0.0, "peak_sys_used_gb": 0.0, "error": None,
    }

    # KV cache persists across turns
    cache = TurboQuantCache(bits=TURBOQUANT_BITS)
    cache_seq_len = 0

    try:
      while True:
        if context_tokens >= fill_threshold:
            print(f"[run] fill threshold reached: {context_tokens:,} >= {fill_threshold:,}")
            break
        if fill_threshold - context_tokens < MIN_USEFUL_GENERATION:
            print(f"[run] remaining < {MIN_USEFUL_GENERATION}, stopping")
            break
        if MAX_TURNS is not None and turn >= MAX_TURNS:
            print(f"[run] MAX_TURNS={MAX_TURNS} reached")
            break

        fill_ratio = context_tokens / target if target > 0 else 0
        prompt_label, prompt_text, prompt_difficulty = select_prompt(turn, fill_ratio, band_counters)
        print(f"\n[turn {turn}] {prompt_label} ({prompt_difficulty}) fill={fill_ratio:.1%}")

        conversation_history.append({"role": "user", "content": prompt_text})

        # Turn 0: full prompt. Turn N>0: incremental tokens only.
        if turn == 0:
            prompt_str = tokenizer.apply_chat_template(
                conversation_history, tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
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
        print(f"[turn {turn}] incremental={n_incremental:,}  total_context={n_total_context:,}")

        mem_before = mem_report(f"turn {turn} before generate")
        memtrail.write({
            "turn": turn, "label": "before_generate",
            "sys_used_gb": mem_before["sys_used_gb"],
            "sys_avail_gb": round((psutil.virtual_memory().available) / _GB, 2),
            "torch_alloc_gb": mem_before["torch_alloc_gb"],
            "torch_peak_gb": mem_before["torch_peak_gb"],
        })

        try:
            enforce_cap(f"turn {turn} after tokenization", VRAM_CAP_GB)
            watchdog = MemoryCapStoppingCriteria(VRAM_CAP_GB, MEMORY_CHECK_EVERY_N_TOKENS)
            enforce_cap(f"turn {turn} before generate", VRAM_CAP_GB)

            streamer = LiveStreamer(tokenizer, verbose_think=VERBOSE_THINK) if LIVE else None

            t0 = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    past_key_values=cache,
                    use_cache=True,
                    do_sample=False,
                    stopping_criteria=[watchdog],
                    streamer=streamer,
                )
            gen_time = time.time() - t0

            cache_seq_len = cache.get_seq_length()

            if watchdog.exceeded:
                raise MemoryCapExceeded(
                    f"watchdog stopped generation at turn {turn}: "
                    f"peak={watchdog.peak_seen_gb:.1f} GB > cap={VRAM_CAP_GB:.0f} GB"
                )

            enforce_cap(f"turn {turn} after generate", VRAM_CAP_GB)
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

            # Generic per-type token counts — not hardcoded to thinking/answer
            segment_token_counts: dict[str, int] = {}
            segment_texts: dict[str, str] = {}
            for seg in segments:
                text = seg.content
                toks = len(tokenizer.encode(text, add_special_tokens=False)) if text else 0
                segment_token_counts[seg.type] = segment_token_counts.get(seg.type, 0) + toks
                segment_texts[seg.type] = segment_texts.get(seg.type, "") + text

            new_ids = outputs[0][n_input:].tolist()
            thinking_tokens = segment_token_counts.get("thinking", 0)
            answer_tokens = segment_token_counts.get("assistant", 0)
            unknown_tokens = segment_token_counts.get("unknown", 0)
            total_generated = sum(segment_token_counts.values())
            thinking_ratio = thinking_tokens / total_generated if total_generated > 0 else 0.0
            answer_text = segment_texts.get("assistant", "")

            if unknown_tokens:
                print(f"[turn {turn}] WARNING: {unknown_tokens} tokens in unknown segments")

            eos_hit = n_new < MAX_NEW_TOKENS
            prev_eos_hit = eos_hit

            coherence_failures = check_output_coherent(raw_text, new_ids)
            if coherence_failures:
                print(f"[turn {turn}] coherence issues: {coherence_failures}")

            # Record the conversation for human review
            transcript.write_turn(
                session_id=0,
                session_topic=f"context_scaling_{target}",
                turn_index=turn,
                user_msg=prompt_text,
                segments=segments,
                raw_output=raw_text,
                status="ok",
                wall=gen_time,
                ttft=0.0,  # not measured separately in local inference
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

            conversation_history.append({"role": "assistant", "content": answer_text})

            user_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            context_tokens += user_tokens + answer_tokens + TEMPLATE_OVERHEAD_PER_TURN

            print(
                f"[turn {turn}] generated={n_new} in {gen_time:.1f}s "
                f"({tok_per_s:.1f} tok/s), thinking={thinking_tokens}, "
                f"answer={answer_tokens}, eos={eos_hit}"
            )
            print(f"[turn {turn}] context={context_tokens:,} ({context_tokens / target:.1%})")

            cumulative_wall_s += gen_time

            # Metrics JSONL (for plots and analysis)
            record = {
                "run_target": target,
                "turn": turn,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_label": prompt_label,
                "prompt_difficulty": prompt_difficulty,
                "user_tokens": user_tokens,
                "max_new_tokens": MAX_NEW_TOKENS,
                "n_input_tokens": n_total_context,
                "n_incremental_tokens": n_incremental,
                "cache_seq_len": cache_seq_len,
                "total_generated": total_generated,
                "segment_token_counts": segment_token_counts,
                "thinking_tokens": thinking_tokens,
                "answer_tokens": answer_tokens,
                "unknown_tokens": unknown_tokens,
                "thinking_ratio": round(thinking_ratio, 4),
                "context_tokens_before": context_tokens - user_tokens - answer_tokens - TEMPLATE_OVERHEAD_PER_TURN,
                "context_tokens_after": context_tokens,
                "context_fill_ratio": round(context_tokens / target, 4),
                "total_time_s": round(gen_time, 2),
                "tok_per_s": round(tok_per_s, 2),
                "ttft_s": round(streamer.ttft, 4) if (streamer and streamer.ttft is not None) else None,
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

            run_summary["turns_completed"] = turn + 1
            run_summary["final_context_tokens"] = context_tokens
            run_summary["final_fill_ratio"] = round(context_tokens / target, 4)
            run_summary["peak_sys_used_gb"] = max(
                run_summary["peak_sys_used_gb"], mem_after["sys_used_gb"],
            )

            del outputs, inputs
            gc.collect()

        except (torch.cuda.OutOfMemoryError, MemoryCapExceeded) as e:
            error_type = "OOM" if isinstance(e, torch.cuda.OutOfMemoryError) else "cap_exceeded"
            print(f"[turn {turn}] {error_type}: {e}", file=sys.stderr)
            metrics.write({
                "run_target": target, "turn": turn,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_label": prompt_label, "prompt_difficulty": prompt_difficulty,
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
            print(f"[turn {turn}] unexpected error: {e}", file=sys.stderr)
            metrics.write({
                "run_target": target, "turn": turn,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_label": prompt_label, "prompt_difficulty": prompt_difficulty,
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

    # Generate HTML transcript for human review
    try:
        html_path = generate_html(transcript_path)
        print(f"[report] transcript: {html_path}")
    except Exception as e:
        print(f"[report] transcript generation failed: {e}", file=sys.stderr)

    return run_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mem_report("start")
    preflight()
    enforce_cap("preflight", VRAM_CAP_GB)

    try:
        from turboquant import TurboQuantCache
    except ImportError:
        print("[FATAL] turboquant not installed. Run: pip install turboquant", file=sys.stderr)
        sys.exit(1)

    all_summaries = []

    for target in CONTEXT_TARGETS:
        model = None
        tokenizer = None
        try:
            print(f"\n[load] tokenizer: {TOKENIZER_ID}")
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
            enforce_cap("after tokenizer", VRAM_CAP_GB)

            print(f"[load] FP8 model: {FP8_CHECKPOINT}")
            t0 = time.time()
            model = load_gemma4_fp8(str(FP8_CHECKPOINT))
            print(f"[load] model loaded in {time.time() - t0:.1f}s")
            post_load = mem_report("model loaded")
            model_weight_gb = post_load["torch_alloc_gb"]
            enforce_cap("after model load", VRAM_CAP_GB)

            summary = run_context_target(target, model, tokenizer, TurboQuantCache, model_weight_gb)

        except (torch.cuda.OutOfMemoryError, MemoryCapExceeded) as e:
            error_type = "OOM" if isinstance(e, torch.cuda.OutOfMemoryError) else "cap_exceeded"
            print(f"[load] {error_type} during setup for target={target}: {e}", file=sys.stderr)
            summary = {
                "target": target, "fill_threshold": int(target * 0.80),
                "turns_completed": 0, "final_context_tokens": 0,
                "final_fill_ratio": 0.0, "peak_sys_used_gb": 0.0,
                "error": f"{error_type}_at_load",
            }

        all_summaries.append(summary)

        print(f"\n[cleanup] tearing down model for target={target}")
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        post = mem_report("between runs")
        if post["sys_used_gb"] > VRAM_CAP_GB * 0.75:
            print(f"[warn] memory still high after cleanup: {post['sys_used_gb']:.1f} GB")

    # Write summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Generate performance dashboard
    try:
        generate_dashboard(RESULTS_DIR, title="Gemma 4 FP8 — 32K Context Scaling (KV Cache Reuse)")
        print(f"[report] dashboard: {RESULTS_DIR / 'plots'}")
    except Exception as e:
        print(f"[report] dashboard generation failed: {e}", file=sys.stderr)

    # Final summary
    print(f"\n[summary] written to {summary_path}")
    for s in all_summaries:
        status = f"error={s['error']}" if s["error"] else "OK"
        print(
            f"[summary] target={s['target']:>7,}  "
            f"turns={s['turns_completed']:>3}  "
            f"fill={s['final_fill_ratio']:.1%}  "
            f"peak_mem={s['peak_sys_used_gb']:.1f} GB  "
            f"status={status}"
        )

    # List all generated artifacts
    artifacts = sorted(RESULTS_DIR.glob("*"))
    print(f"\n[done] all results in {RESULTS_DIR}")
    for a in artifacts:
        if a.is_file():
            size = a.stat().st_size
            unit = "KB" if size < 1024 * 1024 else "MB"
            val = size / 1024 if unit == "KB" else size / (1024 * 1024)
            print(f"  {a.name:<35} {val:>8.1f} {unit}")

    print("\n[done] open transcript_*.html in a browser to review the conversation")

    # Prompt the user can paste into an LLM to generate an analysis spreadsheet
    file_list = ", ".join(a.name for a in artifacts if a.is_file())
    print(f"""
--- Copy the prompt below into Claude or ChatGPT to generate an analysis XLSX ---

I have benchmark results from a context scaling test. The files are:
{file_list}

The run_*.jsonl files contain per-turn metrics (one JSON object per line) with
fields: turn, tok_per_s, total_time_s, segment_token_counts, thinking_tokens,
answer_tokens, unknown_tokens, thinking_ratio, n_input_tokens,
n_incremental_tokens, cache_seq_len, ttft_s, model_weight_gb,
cumulative_wall_s, sys_total_gb, sys_used_before_gb, sys_used_after_gb,
torch_alloc_gb, torch_peak_gb, context_fill_ratio, eos_hit, and error.

segment_token_counts is a dict mapping segment type (e.g. "thinking",
"assistant", "unknown") to token count for that type. thinking_tokens and
answer_tokens are convenience aliases for segment_token_counts["thinking"]
and segment_token_counts["assistant"]. unknown_tokens counts tokens the
parser couldn't classify — any non-zero value is an anomaly worth flagging.

Key constants (same value every turn):
- sys_total_gb: total system memory
- model_weight_gb: torch allocation after model load, before any generation
- run_target: the context window target for this run

The memtrail_*.csv files have per-turn memory snapshots (before/after generate).

Derived metrics to compute per turn (do not assume these exist in the data):
- decode_time_s = total_time_s - ttft_s
- decode_tok_per_s = total_generated / decode_time_s
- prefill_tok_per_s = n_input_tokens / ttft_s
- kv_cache_gb = torch_alloc_gb - model_weight_gb
- tokens_per_gb_kv = cache_seq_len / kv_cache_gb
- headroom_gb = sys_total_gb - sys_used_after_gb

Please create an analysis XLSX with these sheets:
1. "Per-Turn Metrics" — all JSONL records as a flat table, plus columns for
   all derived metrics above. Flatten segment_token_counts into one column
   per observed type (e.g. "seg_thinking", "seg_assistant", "seg_unknown").
2. "Segment Breakdown" — one row per turn. Columns: turn, total_generated,
   then one column per segment type found across all turns (token count),
   plus a "segment_types_seen" column listing types present in that turn.
   Include a stacked bar chart: token count by segment type vs context fill.
   Flag any turn where unknown_tokens > 0 with conditional formatting (red).
3. "Memory Timeline" — memtrail CSV data with a chart
4. "Summary" — one row per target with: turns completed, final fill ratio,
   peak memory, min/max/mean tok/s, total wall time, thinking ratio trend,
   count of turns with unknown_tokens > 0
5. "Charts" — embedded charts for:
   - Decode speed (tok/s) vs context
   - Memory (GB and % of sys_total_gb) vs context
   - Thinking ratio vs context
   - Segment type distribution (stacked area or stacked bar, tokens by type
     vs context fill — shows how output composition shifts as context grows)
   - Time per turn vs context
   - Prefill vs decode time (stacked or grouped bar)
   - KV cache efficiency (tokens_per_gb_kv vs context)
   Every chart axis must have a label with units (e.g. "Context Fill (%)",
   "Decode Speed (tok/s)", "System Memory Used (GB)", "Thinking Ratio",
   "Generation Time (minutes)"). Memory charts should show both absolute
   GB and percentage of sys_total_gb (e.g. dual Y-axis or annotation).
   Context charts should show both absolute token count
   (context_tokens_after) and percentage of run_target (context_fill_ratio)
   on the X-axis. X-axis category labels must be human-readable
   (e.g. "3,515 (21%)" not 0.2145).

I'll paste the file contents below.
---""")


if __name__ == "__main__":
    main()
