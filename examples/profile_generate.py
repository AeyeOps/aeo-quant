#!/usr/bin/env python3
"""Profile a single generate() call to find where time is spent.

Produces:
  1. Stage timing breakdown (load, tokenize, prefill, decode)
  2. torch.profiler kernel-level breakdown (top 40 ops by CUDA time)
  3. Chrome trace file for visual inspection in chrome://tracing

Usage:
    uv run python examples/profile_generate.py

Set FP8_CHECKPOINT (or NVFP4_CHECKPOINT + QUANT_FORMAT=nvfp4) in .env or env var.
"""
from __future__ import annotations

import atexit
import gc
import json
import os
import sys
import time
from pathlib import Path

# AEO_MOE_TRACE is an all-or-nothing switch: NVTX markers (emitted from
# aeo_quant.bridges.gemma4.modeling) only produce data when nsys is wrapping
# the process. If the switch is on but nsys isn't attached, re-exec under it
# so the markers actually land in a trace. Run under nsys already -> skip.
if os.environ.get("AEO_MOE_TRACE") == "1" and "NSYS_PROFILING_SESSION_ID" not in os.environ:
    from aeo_quant.core.config import results_dir as _results_dir
    _outdir = _results_dir("nsys")
    os.execvp("nsys", [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        f"--output={_outdir}/trace",
        sys.executable, *sys.argv,
    ])

import torch
from transformers import AutoTokenizer

import aeo_quant  # noqa: F401 — triggers np.trapz compat shim before numpy is used
from aeo_quant.core.config import load_dotenv, quant_env, results_dir, setup_cuda_allocator
from aeo_quant.core.writers import Tee
from aeo_quant.gpu.memory import CudaTimer, mem_report

load_dotenv()
setup_cuda_allocator()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QUANT_FORMAT, CHECKPOINT, KV_BITS = quant_env()
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "google/gemma-4-26B-A4B-it")
GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "100"))
PROFILE_TRACE = os.environ.get("PROFILE_TRACE", "0") != "0"
COMPARE_KV = os.environ.get("COMPARE_KV", "0") != "0"
RESULTS_DIR = results_dir("profiling")

PROMPT = (
    "You are a senior Python engineer.\n\n"
    "Design a thread-safe priority task queue with support for "
    "task cancellation, timeouts, and dead-letter handling. "
    "Show the full implementation with type hints."
)


# ---------------------------------------------------------------------------
# Stage 1: Timing breakdown
# ---------------------------------------------------------------------------
def run_timing_breakdown(model, tokenizer, cache_cls, label: str = "default") -> dict:
    """Run a single generate() and break down time into stages."""
    print(f"\n{'=' * 60}")
    print(f"[timing] {label} — generating {GEN_TOKENS} tokens")
    print(f"{'=' * 60}")

    messages = [
        {"role": "system", "content": "You are a senior Python engineer."},
        {"role": "user", "content": PROMPT},
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )

    # Tokenization is CPU-bound; time it with a wall clock instead of CUDA events.
    t_tok_start = time.perf_counter()
    cpu_inputs = tokenizer(prompt_str, return_tensors="pt")
    tokenize_ms = (time.perf_counter() - t_tok_start) * 1000

    with CudaTimer("inputs_to_device") as t_h2d:
        inputs = cpu_inputs.to(model.device)
    n_prompt = inputs["input_ids"].shape[-1]
    print(f"[timing] prompt tokens: {n_prompt}")
    print(f"[timing] tokenize: {tokenize_ms:.1f} ms")
    print(f"[timing] h2d copy: {t_h2d.elapsed_ms:.1f} ms")

    # Build cache
    if cache_cls is not None:
        cache = cache_cls(bits=4)
        cache_label = "TurboQuant-4bit"
    else:
        cache = None
        cache_label = "native (DynamicCache)"

    print(f"[timing] KV cache: {cache_label}")

    # Generate with timing
    with CudaTimer("generate_total") as t_total, torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GEN_TOKENS,
            past_key_values=cache,
            use_cache=True,
            do_sample=False,
        )
    n_generated = outputs.shape[-1] - n_prompt
    total_ms = t_total.elapsed_ms

    # We can't separate prefill from decode with a single generate() call.
    # Run a second call with max_new_tokens=1 to measure prefill alone.
    del outputs, cache
    gc.collect()
    torch.cuda.empty_cache()

    cache2 = cache_cls(bits=4) if cache_cls is not None else None

    with CudaTimer("prefill_only") as t_prefill, torch.inference_mode():
        outputs_pf = model.generate(
            **inputs,
            max_new_tokens=1,
            past_key_values=cache2,
            use_cache=True,
            do_sample=False,
        )
    prefill_ms = t_prefill.elapsed_ms
    decode_ms = total_ms - prefill_ms
    decode_tok_per_s = (n_generated / (decode_ms / 1000)) if decode_ms > 0 else 0
    overall_tok_per_s = (n_generated / (total_ms / 1000)) if total_ms > 0 else 0

    del outputs_pf, cache2
    gc.collect()
    torch.cuda.empty_cache()

    results = {
        "label": label,
        "cache": cache_label,
        "prompt_tokens": n_prompt,
        "generated_tokens": n_generated,
        "tokenize_ms": round(tokenize_ms, 1),
        "h2d_ms": round(t_h2d.elapsed_ms, 1),
        "prefill_ms": round(prefill_ms, 1),
        "decode_ms": round(decode_ms, 1),
        "total_ms": round(total_ms, 1),
        "prefill_pct": round(100 * prefill_ms / total_ms, 1) if total_ms > 0 else 0,
        "decode_pct": round(100 * decode_ms / total_ms, 1) if total_ms > 0 else 0,
        "overall_tok_per_s": round(overall_tok_per_s, 2),
        "decode_tok_per_s": round(decode_tok_per_s, 2),
    }

    print("\n[timing] results:")
    print(f"  tokenize:       {results['tokenize_ms']:>10.1f} ms")
    print(f"  prefill:        {results['prefill_ms']:>10.1f} ms  ({results['prefill_pct']:.1f}%)")
    print(f"  decode:         {results['decode_ms']:>10.1f} ms  ({results['decode_pct']:.1f}%)")
    print(f"  total:          {results['total_ms']:>10.1f} ms")
    print(f"  overall tok/s:  {results['overall_tok_per_s']:>10.2f}")
    print(f"  decode tok/s:   {results['decode_tok_per_s']:>10.2f}")
    print(f"  generated:      {results['generated_tokens']:>10d} tokens")

    return results


# ---------------------------------------------------------------------------
# Stage 2: torch.profiler kernel breakdown
# ---------------------------------------------------------------------------
def run_profiler_trace(model, tokenizer, cache_cls) -> None:
    """Run generate() under torch.profiler and export results."""
    print(f"\n{'=' * 60}")
    print(f"[profiler] tracing {GEN_TOKENS} tokens of generation")
    print(f"{'=' * 60}")

    messages = [
        {"role": "system", "content": "You are a senior Python engineer."},
        {"role": "user", "content": PROMPT},
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

    cache = cache_cls(bits=4) if cache_cls is not None else None

    # Warm up — one forward pass to avoid first-call overhead in the trace
    with torch.inference_mode():
        _ = model.generate(
            **inputs,
            max_new_tokens=1,
            past_key_values=cache,
            use_cache=True,
            do_sample=False,
        )

    # Reset cache for the real run
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    cache = cache_cls(bits=4) if cache_cls is not None else None

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof, torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GEN_TOKENS,
            past_key_values=cache,
            use_cache=True,
            do_sample=False,
        )

    n_generated = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    print(f"[profiler] generated {n_generated} tokens\n")

    # Print top operations by CUDA time
    table = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=40,
    )
    print(table)

    # Also print grouped by input shape to see per-expert breakdown
    print(f"\n{'=' * 60}")
    print("[profiler] top ops grouped by input shape")
    print(f"{'=' * 60}")
    table_shapes = prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=30,
    )
    print(table_shapes)

    del outputs, cache
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _log = open(RESULTS_DIR / "stdout.log", "w")  # noqa: SIM115 — lifetime = process
    atexit.register(_log.close)
    sys.stdout = Tee(sys.__stdout__, _log)
    sys.stderr = Tee(sys.__stderr__, _log)

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available — GPU-only.", file=sys.stderr)
        sys.exit(1)

    mem_report("start")

    dev_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    print(f"[preflight] device: {dev_name} (sm_{cc_major}{cc_minor})")
    print(f"[preflight] torch: {torch.__version__}")
    print(f"[preflight] generating {GEN_TOKENS} tokens per run")

    # Load model
    print(f"\n[load] tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    print(f"[load] {QUANT_FORMAT.upper()} model: {CHECKPOINT}")
    t0 = time.time()
    from aeo_quant.bridges.gemma4.loader import load_gemma4
    model = load_gemma4(str(CHECKPOINT), quant_format=QUANT_FORMAT)
    print(f"[load] model loaded in {time.time() - t0:.1f}s")
    post_load = mem_report("model loaded")
    model_weight_gb = post_load["torch_alloc_gb"]
    print(f"[load] model weight footprint: {model_weight_gb:.2f} GB")

    # TurboQuant cache
    try:
        from turboquant import TurboQuantCache
    except ImportError:
        print(
            "[FATAL] turboquant is required for this benchmark path. "
            "Install the bridges stack and rerun.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Warmup: a short generate() to trigger torch.compile JIT before timing.
    _warmup_cache = TurboQuantCache(bits=KV_BITS)
    _warmup_inputs = tokenizer("warmup", return_tensors="pt").to(model.device)
    with torch.inference_mode():
        model.generate(**_warmup_inputs, max_new_tokens=1, past_key_values=_warmup_cache,
                        use_cache=True, do_sample=False)
    del _warmup_cache, _warmup_inputs
    gc.collect()
    torch.cuda.empty_cache()

    # --- Stage 1: Timing breakdown ---
    all_results = []

    timing = run_timing_breakdown(model, tokenizer, TurboQuantCache, "TurboQuant-4bit")
    all_results.append(timing)

    if COMPARE_KV:
        timing_native = run_timing_breakdown(model, tokenizer, None, "native-cache")
        all_results.append(timing_native)

    # Print comparison
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("[compare] KV cache impact")
        print(f"{'=' * 60}")
        for r in all_results:
            print(
                f"  {r['label']:<20} "
                f"prefill={r['prefill_ms']:>8.1f}ms  "
                f"decode={r['decode_ms']:>8.1f}ms  "
                f"tok/s={r['decode_tok_per_s']:>6.2f}"
            )

    # --- Stage 2: Profiler trace ---
    if PROFILE_TRACE:
        gc.collect()
        torch.cuda.empty_cache()
        run_profiler_trace(model, tokenizer, TurboQuantCache)

    # Summary
    print(f"\n{'=' * 60}")
    print("[done] profiling complete")
    print(f"{'=' * 60}")
    print(f"  results: {RESULTS_DIR}")
    mem_report("final")

    # Save timing results
    timing_path = RESULTS_DIR / "timing.json"
    with open(timing_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  timing: {timing_path}")


if __name__ == "__main__":
    main()
