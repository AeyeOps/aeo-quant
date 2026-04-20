# Examples

These scripts show how to use `aeo-quant` with a Gemma 4 checkpoint in either
**FP8** or **NVFP4** on NVIDIA Blackwell hardware. Every example is
format-agnostic — switch with one env var, no code changes.

## Setup

Drop a `.env` file in this directory (or the directory you run from):

```
# Pick one. Defaults to fp8 if unset.
QUANT_FORMAT=nvfp4

# Set the matching checkpoint (HF repo ID or local path).
FP8_CHECKPOINT=aeyeops/gemma-4-26b-a4b-it-fp8
NVFP4_CHECKPOINT=aeyeops/gemma-4-26b-a4b-it-nvfp4

HF_TOKEN=hf_your_token_here
```

`quant_env()` routes to the matching `*_CHECKPOINT` and (for nvfp4) auto-applies
the `TRITON_OVERRIDE_ARCH=sm120` Triton quirk. `KV_BITS` defaults to 4 for FP8
and 3 for NVFP4.

Pre-built checkpoints:

- **FP8** — [`aeyeops/gemma-4-26b-a4b-it-fp8`](https://huggingface.co/aeyeops/gemma-4-26b-a4b-it-fp8)
- **NVFP4** — [`aeyeops/gemma-4-26b-a4b-it-nvfp4`](https://huggingface.co/aeyeops/gemma-4-26b-a4b-it-nvfp4)

The scripts read `.env` automatically. No command-line flags, no `source`, no `export` needed.

## The examples

### `quality_check.py` — "Does this checkpoint actually work?"

```bash
uv run python examples/quality_check.py                        # fp8 (default)
QUANT_FORMAT=nvfp4 uv run python examples/quality_check.py     # nvfp4
```

Loads the model, sends three different prompts (a coding task, a plain-English
explanation, and a mixed code+prose question), prints each response so you can
read it, and checks for output quality issues like repetition loops or garbage
tokens. Takes about 5 minutes. Run this first after building or downloading a
checkpoint.

### `parity_check.py` — "Did my code change break the output?"

```bash
uv run python examples/parity_check.py
QUANT_FORMAT=nvfp4 uv run python examples/parity_check.py
```

Generates 50 greedy tokens from a fixed prompt, saves the token IDs, and diffs
against a pinned per-format baseline (`tests/fixtures/parity_baseline_<format>.txt`).
If no baseline exists, the first run pins one. NVFP4 runs additionally report
an informational divergence vs the FP8 baseline (never fails the run).
Fails on >5% token divergence against its own baseline.

This is the regression canary — run it after any change to the model loading,
forward pass, or quantization code. It catches silent quality regressions that
timing-only benchmarks miss.

### `parity_long_check.py` — "Does SWA eviction behave correctly across a sliding window?"

```bash
QUANT_FORMAT=nvfp4 uv run python examples/parity_long_check.py
```

Generates 2000 tokens — long enough to cross Gemma 4's 1024-token sliding
window and exercise eviction on the 25 sliding-attention layers. Fails on
>0.5% token divergence vs the pinned `parity_long_baseline_<format>.txt`. The
50-token `parity_check.py` never crosses the SWA boundary; this one does, so
any sliding-cache bug surfaces here.

### `reasoning_check.py` — "Does it still think clearly under KV compression?"

```bash
uv run python examples/reasoning_check.py
QUANT_FORMAT=nvfp4 uv run python examples/reasoning_check.py
```

Two hard reasoning prompts (Sylow subgroup proof, concurrent LRU cache bug
hunt), 500 tokens each. These stress attention precision harder than
`quality_check.py` — useful for validating that a lower `KV_BITS` value still
produces coherent reasoning. Per-format, per-bits baselines live at
`results/reasoning/baseline_<format>-<bits>bit/`.

### `multi_turn_16k.py` — "How does it hold up in a real conversation?"

```bash
uv run python examples/multi_turn_16k.py
```

Has a multi-turn coding conversation with the model — starts by asking it to
build a task queue system, then progressively asks it to add features, write
tests, fix concurrency bugs, and redesign for scale. Each turn gets harder as
the conversation grows toward 16K tokens.

Measures tok/s, memory, and how much the model "thinks" versus just answering.
The KV cache stays warm across turns via `Gemma4HybridTurboQuantCache` (like a
real chat session — no wasted re-processing, and no wasted dequant on
masked-out sliding-window history).

When it finishes, you get:
- **transcript HTML** — open in a browser to read the full conversation
- **dashboard PNG** — charts showing performance over time
- **metrics JSONL + memory CSV** — raw data if you want to dig deeper
- **a ready-to-paste prompt** you can hand to an LLM to generate an analysis spreadsheet

Runs for about 2 hours depending on hardware.

### `multi_turn_32k.py` — "Same conversation, bigger context window"

```bash
uv run python examples/multi_turn_32k.py
```

Identical to `multi_turn_16k.py` but targets 32K tokens (fills to 80% = ~26K).
More turns, longer run, same metrics. Results land in
`results/multi_turn/<format>-<bits>bit-<timestamp>/` so runs don't clobber each
other. Use both to compare how performance scales across context sizes.

### `profile_generate.py` — "Why is generation slow?"

```bash
uv run python examples/profile_generate.py
```

Diagnostic script for when tok/s numbers look wrong. Loads the same model and
runs a short generation (100 tokens by default) with instrumentation to
pinpoint where time is spent.

Produces two things:

1. **Timing breakdown** — GPU-accurate measurements (via CUDA events, not wall
   clock) of tokenization, prefill, and decode separately. Shows prefill vs
   decode split so you can tell whether slowness is from attention scaling or
   per-token overhead.

2. **Kernel-level profile** — `torch.profiler` trace showing the top 40 CUDA
   operations by time. Tells you exactly how much goes to dtype casts,
   matmuls, attention, etc. Exports a Chrome trace you can open in
   `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/) for a visual
   timeline.

Run it after a benchmark, not during (it needs the GPU). Useful knobs:

```bash
COMPARE_KV=1     uv run python examples/profile_generate.py   # TurboQuant vs native cache
PROFILE_TRACE=1  uv run python examples/profile_generate.py   # include kernel-level trace
GEN_TOKENS=200   uv run python examples/profile_generate.py   # longer measurement
AEO_MOE_TRACE=1  uv run python examples/profile_generate.py   # auto-wrap under nsys with NVTX markers
```

Results go to `results/profiling/<format>-<bits>bit-<timestamp>/` (timing +
stdout log). When `AEO_MOE_TRACE=1` is set, the script auto-wraps itself
under `nsys profile` — the NVTX trace lands in `results/nsys/<timestamp>/`.

### `build_checkpoint.py` — "I want to build my own FP8 checkpoint"

```bash
uv run python examples/build_checkpoint.py
```

Takes a bf16 Gemma 4 model from HuggingFace and quantizes the MoE expert
weights to FP8 (E4M3 with bf16 per-output-channel scales). Reads one
safetensors shard at a time so it never loads the full 52 GB model into
memory (peaks around 18 GB). Writes a drop-in checkpoint you can use with the
other examples.

### `build_checkpoint_nvfp4.py` — "…and NVFP4"

```bash
uv run python examples/build_checkpoint_nvfp4.py
```

The NVFP4 counterpart. Same shard-streaming pipeline, but emits packed E2M1
FP4 bytes with FP8 per-block scales (block_size=16) and FP32 per-tensor
scales. Output artifact is ~19 GB (vs ~27 GB for FP8).

You only need to run either of these if you're building from scratch rather
than using the published checkpoints.

## What you'll need

- An NVIDIA GPU with enough memory (tested on GB10 with 128 GB unified).
  - FP8 path: any Hopper or Blackwell GPU with `torch._scaled_mm` support.
  - NVFP4 path: consumer Blackwell only (`sm_120`/`sm_121`), Triton ≥ 3.5.
- Python 3.12+
- The `aeo-quant` package installed with the bridge stack: `uv pip install -e '.[bridges]'` (includes `turboquant` for KV cache compression)

## Adapting these for other models

The examples are written for Gemma 4, but the toolkit modules they import
(`core.coherence`, `gpu.memory`, `core.writers`, `core.config.quant_env`, etc.)
work with any model. To adapt for a different architecture:

1. Write a new bridge under `src/aeo_quant/bridges/your_model/`
2. Swap the import from `bridges.gemma4.loader` to your loader
3. Update the thinking parser if your model uses different markers
4. Everything else (memory monitoring, coherence checks, transcript recording, plots) stays the same
