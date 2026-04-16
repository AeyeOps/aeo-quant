# Examples

These scripts show how to use `aeo-quant` with a Gemma 4 FP8 checkpoint on NVIDIA hardware. Each one is self-contained — copy it, adapt it, make it yours.

## Setup

Drop a `.env` file in this directory (or the directory you run from):

```
FP8_CHECKPOINT=/path/to/your/fp8/checkpoint
HF_TOKEN=hf_your_token_here
```

That's it. The scripts read `.env` automatically. No command-line flags, no `source`, no `export` needed.

## The examples

### `quality_check.py` — "Does this checkpoint actually work?"

```bash
uv run python examples/quality_check.py
```

Loads the model, sends three different prompts (a coding task, a plain-English explanation, and a mixed code+prose question), prints each response so you can read it, and checks for output quality issues like repetition loops or garbage tokens. Takes about 5 minutes. Run this first after building or downloading a checkpoint.

### `multi_turn_16k.py` — "How does it hold up in a real conversation?"

```bash
uv run python examples/multi_turn_16k.py
```

Has a multi-turn coding conversation with the model — starts by asking it to build a task queue system, then progressively asks it to add features, write tests, fix concurrency bugs, and redesign for scale. Each turn gets harder as the conversation grows toward 16K tokens.

Measures tok/s, memory, and how much the model "thinks" versus just answering. The KV cache stays warm across turns (like a real chat session — no wasted re-processing).

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

Identical to `multi_turn_16k.py` but targets 32K tokens (fills to 80% = ~26K). More turns, longer run, same metrics. Results land in `results/context_scaling_32k/` so they don't clobber the 16K data. Use both to compare how performance scales across context sizes.

### `profile_generate.py` — "Why is generation slow?"

```bash
uv run python examples/profile_generate.py
```

Diagnostic script for when tok/s numbers look wrong. Loads the same model and runs a short generation (100 tokens by default) with instrumentation to pinpoint where time is spent.

Produces two things:

1. **Timing breakdown** — GPU-accurate measurements (via CUDA events, not wall clock) of tokenization, prefill, and decode separately. Shows prefill vs decode split so you can tell whether slowness is from attention scaling or per-token overhead.

2. **Kernel-level profile** — `torch.profiler` trace showing the top 40 CUDA operations by time. Tells you exactly how much goes to dtype casts, matmuls, attention, etc. Exports a Chrome trace you can open in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/) for a visual timeline.

Run it after a benchmark, not during (it needs the GPU). Useful knobs:

```bash
# Compare TurboQuant KV cache vs native cache overhead
COMPARE_KV=1 uv run python examples/profile_generate.py

# Just timing breakdown, skip the profiler trace
PROFILE_TRACE=0 uv run python examples/profile_generate.py

# Generate more tokens for a more stable measurement
GEN_TOKENS=200 uv run python examples/profile_generate.py
```

Results go to `results/profiling/`.

### `build_checkpoint.py` — "I want to build my own FP8 checkpoint"

```bash
uv run python examples/build_checkpoint.py
```

Takes a bf16 Gemma 4 model from HuggingFace and quantizes the MoE expert weights to FP8. Reads one safetensors shard at a time so it never loads the full 52 GB model into memory (peaks around 18 GB). Writes a drop-in checkpoint you can use with the other examples.

You only need to run this if you're building from scratch rather than using a published checkpoint.

## What you'll need

- An NVIDIA GPU with enough memory (tested on GB10 with 128 GB unified)
- Python 3.12+
- The `aeo-quant` package installed with GPU extras: `pip install aeo-quant[bridges]`
- `turboquant` for KV cache compression: `pip install turboquant`

## Adapting these for other models

The examples are written for Gemma 4, but the toolkit modules they import (`core.coherence`, `gpu.memory`, `core.writers`, etc.) work with any model. To adapt for a different architecture:

1. Write a new bridge under `src/aeo_quant/bridges/your_model/`
2. Swap the import from `bridges.gemma4.loader` to your loader
3. Update the thinking parser if your model uses different markers
4. Everything else (memory monitoring, coherence checks, transcript recording, plots) stays the same
