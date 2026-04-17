#!/usr/bin/env python3
"""Multi-turn conversation benchmark at 16K context.

Runs via the aeo-quant harness daemon. Dispatches the ``multi_turn`` workload
at ``target=16384`` with ``out_dir`` pointing at a timestamped per-run folder;
the workload writes per-turn metrics/transcript/memtrail files directly there.
Client-side, this script reconstructs the live terminal UX from streamed
events, then generates the transcript HTML and performance dashboard.

Produces per target:
  - run_16384.jsonl      — per-turn numbers for analysis
  - transcript_16384.jsonl + transcript_16384.html — conversation for review
  - memtrail_16384.csv   — per-turn memory snapshots
  - plots/               — dashboard PNGs (via generate_dashboard)
  - summary.json         — per-target rollup

Usage:
    uv run python examples/multi_turn_16k.py

Set FP8_CHECKPOINT (or NVFP4_CHECKPOINT + QUANT_FORMAT=nvfp4) in .env or env var.
Exit codes:
    0 — completed (possibly with per-turn errors inside the summary)
    2 — environment failure (no harness, format mismatch, etc.)
"""
from __future__ import annotations

import json
import os
import sys
import time

import aeo_quant  # noqa: F401 — triggers np.trapz compat shim before numpy is used
from aeo_quant.core.config import load_dotenv, quant_env, results_dir, setup_cuda_allocator
from aeo_quant.core.viewer import generate_html
from aeo_quant.gpu.memory import mem_report, preflight_memory
from aeo_quant.harness import HarnessUnavailable, get_or_start_harness
from aeo_quant.plots.context_scaling import generate_dashboard

MIN_FREE_GB = 60.0
CONTEXT_TARGET = 16384

load_dotenv()
setup_cuda_allocator()

QUANT_FORMAT, CHECKPOINT, KV_BITS = quant_env()
VRAM_CAP_GB = float(os.environ.get("VRAM_CAP_GB", "90.0"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "10000"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "0")) or None

RESULTS_DIR = results_dir("context_scaling", format=QUANT_FORMAT, kv_bits=KV_BITS)


class _TerminalPrinter:
    """Reconstruct the live terminal UX from HarnessStreamer events.

    Thinking text streams in dim-cyan under a ``[thinking]`` header; answer
    text streams under a bold ``[answer]`` header. Turn boundaries and memory
    warnings print full lines.
    """

    # ANSI
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    def __init__(self) -> None:
        self._in_thinking = False
        self._in_answer = False

    def _reset_phase(self) -> None:
        self._in_thinking = False
        self._in_answer = False

    def __call__(self, event: dict) -> None:
        t = event.get("type")
        if t == "turn_start":
            self._reset_phase()
            print(
                f"\n[turn {event['turn']}] {event['prompt_label']} "
                f"({event['prompt_difficulty']}) "
                f"fill={event['fill_ratio']:.1%}  "
                f"context={event['context_tokens']:,}",
                flush=True,
            )
        elif t == "thinking_text":
            # Print the [thinking] header once, then stream dim-cyan text so
            # thinking is visually distinct from answer/user/tool content.
            if not self._in_thinking:
                self._in_thinking = True
                sys.stderr.write(
                    f"\n{self.DIM}{self.CYAN}[thinking]{self.RESET} "
                    f"{self.DIM}{self.CYAN}"
                )
            sys.stderr.write(event["text"])
            sys.stderr.flush()
        elif t == "thinking_end":
            if self._in_thinking:
                sys.stderr.write(
                    f"{self.RESET}\n"
                    f"{self.DIM}[thinking done] {event['tokens']} tokens in "
                    f"{event['elapsed_s']:.0f}s{self.RESET}\n"
                )
            else:
                sys.stderr.write("\n")
            sys.stderr.write(f"{self.BOLD}[answer]{self.RESET} ")
            sys.stderr.flush()
            self._in_thinking = False
            self._in_answer = True
        elif t == "answer_chunk":
            if not self._in_answer:
                sys.stderr.write(f"\n{self.BOLD}[answer]{self.RESET} ")
                self._in_answer = True
            sys.stderr.write(event["text"])
            sys.stderr.flush()
        elif t == "turn_complete":
            ttft = event.get("ttft_s")
            ttft_str = f"ttft={ttft:.2f}s" if ttft is not None else "ttft=n/a"
            print(
                f"\n[turn {event['turn']} done] "
                f"gen_ms={event['gen_ms']:.0f} "
                f"tok_s={event['tok_s']:.1f} {ttft_str} "
                f"think={event['thinking_tokens']} "
                f"answer={event['answer_tokens']} "
                f"eos={event['eos_hit']} "
                f"context={event['context_tokens_after']:,} "
                f"({event['context_fill_ratio']:.1%})",
                flush=True,
            )
            if event.get("coherence_failures"):
                print(
                    f"  [warn] coherence: {event['coherence_failures']}",
                    flush=True,
                )
            self._reset_phase()
        elif t == "memory_warning":
            print(
                f"\n{self.YELLOW}[memory_warning] turn {event['turn']}: "
                f"peak={event['peak_gb']:.1f} GB > cap={event['cap_gb']:.0f} GB"
                f"{self.RESET}",
                flush=True,
            )
        else:
            msg = event.get("message")
            if msg:
                print(f"[event] {msg}", flush=True)


def _check_harness_format(client) -> int:
    info = client.status()
    srv_format = info.get("quant_format")
    if srv_format != QUANT_FORMAT:
        print(
            f"[FATAL] harness is loaded with {srv_format!r}, "
            f"you requested {QUANT_FORMAT!r}.\n"
            f"        Restart the harness with `aeo-harness stop && "
            f"aeo-harness start --format {QUANT_FORMAT}`.",
            file=sys.stderr,
        )
        return 2
    print(
        f"[multi_turn] using harness (uptime={info.get('uptime_s')}s, "
        f"jobs_served={info.get('jobs_served')}, queue={info.get('queue_depth')})",
        flush=True,
    )
    return 0


def main() -> int:
    print(
        f"[multi_turn] QUANT_FORMAT={QUANT_FORMAT} CHECKPOINT={CHECKPOINT} "
        f"KV_BITS={KV_BITS} target={CONTEXT_TARGET:,} cap={VRAM_CAP_GB:.0f} GB",
        flush=True,
    )
    preflight_memory(MIN_FREE_GB, label="multi_turn_16k")
    mem_report("multi_turn:start")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        client = get_or_start_harness()
    except HarnessUnavailable as e:
        print(f"[FATAL] harness unavailable: {e}", file=sys.stderr)
        return 2

    rc = _check_harness_format(client)
    if rc != 0:
        return rc

    printer = _TerminalPrinter()
    t0 = time.time()
    summary = client.run_workload(
        "multi_turn",
        on_event=printer,
        target=CONTEXT_TARGET,
        out_dir=str(RESULTS_DIR),
        vram_cap_gb=VRAM_CAP_GB,
        max_new_tokens=MAX_NEW_TOKENS,
        kv_bits=KV_BITS,
        max_turns=MAX_TURNS,
    )
    print(f"\n[multi_turn] workload finished in {time.time() - t0:.1f}s", flush=True)

    all_summaries = [summary]

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"[summary] written to {summary_path}")

    transcript_path = RESULTS_DIR / f"transcript_{CONTEXT_TARGET}.jsonl"
    if transcript_path.exists():
        try:
            html_path = generate_html(transcript_path)
            print(f"[report] transcript: {html_path}")
        except Exception as e:
            print(f"[report] transcript generation failed: {e}", file=sys.stderr)

    try:
        generate_dashboard(
            RESULTS_DIR,
            title="Gemma 4 FP8 — Context Scaling (KV Cache Reuse)",
        )
        print(f"[report] dashboard: {RESULTS_DIR / 'plots'}")
    except Exception as e:
        print(f"[report] dashboard generation failed: {e}", file=sys.stderr)

    for s in all_summaries:
        status = f"error={s['error']}" if s["error"] else "OK"
        print(
            f"[summary] target={s['target']:>7,}  "
            f"turns={s['turns_completed']:>3}  "
            f"fill={s['final_fill_ratio']:.1%}  "
            f"peak_mem={s['peak_sys_used_gb']:.1f} GB  "
            f"status={status}"
        )

    artifacts = sorted(RESULTS_DIR.glob("*"))
    print(f"\n[done] all results in {RESULTS_DIR}")
    for a in artifacts:
        if a.is_file():
            size = a.stat().st_size
            unit = "KB" if size < 1024 * 1024 else "MB"
            val = size / 1024 if unit == "KB" else size / (1024 * 1024)
            print(f"  {a.name:<35} {val:>8.1f} {unit}")

    print("\n[done] open transcript_*.html in a browser to review the conversation")

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
