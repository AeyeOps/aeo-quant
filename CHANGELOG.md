# Changelog

All notable changes to this project will be documented in this file.

## [0.1.10] - 2026-04-17

### Added

- **`Gemma4HybridTurboQuantCache`** in `aeo_quant.bridges.gemma4.cache` — a
  SWA-aware KV cache that pre-populates `self.layers` with per-layer backends
  based on `config.layer_types` and `config.sliding_window`.
  - `TurboQuantSlidingLayer(TurboQuantLayer)` caps compressed storage at
    `sliding_window − 1 − residual_len` tokens (895 with defaults) by trimming
    the head of `_key_indices`/`_key_norms`/`_value_indices`/`_value_norms`
    after the parent's `update` runs. The return tensors are untouched so the
    current step's attention mask shape still matches — trimming only shrinks
    storage for the *next* update.
  - `get_mask_sizes(query_length)` on sliding layers mirrors the formula from
    transformers' `DynamicSlidingWindowLayer` verbatim, so
    `transformers.masking_utils` builds the correct sliding-window mask.
  - Full-attention layers (5 of 30 on Gemma 4) stay as vanilla
    `TurboQuantLayer` — unbounded growth is correct for them.
  - Validates `num_kv_shared_layers == 0` and raises `NotImplementedError` on
    unknown `layer_type` values so the next model variant fails loud.
- **`examples/parity_long_check.py`** — 2000-token greedy parity gate. The
  50-token `parity_check.py` never crosses the 1024-token sliding window;
  `parity_long_check.py` does, so any SWA-eviction bug surfaces here.
  First run establishes `tests/fixtures/parity_long_baseline_{fp8,nvfp4}.txt`;
  subsequent runs fail if token-level divergence > 0.5%.

### Changed

- **All four workload modules** (`parity`, `reasoning`, `quality`,
  `multi_turn`) now construct `Gemma4HybridTurboQuantCache(bits=kv_bits,
  config=model.config)` instead of `TurboQuantCache(bits=kv_bits)`. No API
  change visible to callers; existing workload kwargs are preserved.

### Performance (expected, post-validation)

At 16K context, ~80% of the per-step dequant workload on the stock cache is
on sliding-layer history older than the 1024-token window — data the attention
kernel then masks out. `Gemma4HybridTurboQuantCache` eliminates that wasted
work. Cost-model estimates: **2–3× decode tok/s at 16K, 3–5× at 32K**. These
are predictions; measured numbers will be recorded here after the comparator
runs (see `docs/plans/bubbly-humming-deer.md` for the verification protocol).

## [0.1.9] - 2026-04-17

### Added

- **Three new workloads** in `aeo_quant.workloads`, completing the retrofit
  commitment from v0.1.8:
  - `reasoning` — 2 hard prompts (Sylow proof, concurrent LRU bugs), 500
    tokens each. Per-prompt events and per-prompt output records that carry a
    `file` field; baselines live at
    `results/reasoning/baseline_<format>-<bits>bit/` (per-format, per-bits).
  - `quality` — 3 diverse prompts (quicksort, merkle tree, pandas). Workload
    is pure compute; coherence check and the `tok/s >= 3.0` gate run on the
    client so the workload contract stays free of pass/fail semantics.
  - `multi_turn` — long multi-turn conversations with KV cache reuse, target
    context (16K / 32K), per-turn metrics/transcript/memtrail files written to
    a client-provided `out_dir`, `MemoryCapStoppingCriteria` watchdog. Emits
    `turn_start`, `thinking_text`, `thinking_end`, `answer_chunk`,
    `turn_complete`, and `memory_warning` events.
- **`HarnessStreamer`** (`aeo_quant.bridges.gemma4.streamer`) — subclasses
  `LiveStreamer` to inherit its phase machine and TTFT tracking; overrides
  `on_finalized_text` to emit structured events through an `emit` callback
  instead of writing ANSI-decorated text to stderr. Thinking is part of the
  model's output, so it's always streamed as `thinking_text` events (with a
  buffer tail preserved across calls so close markers split at the boundary
  still resolve). Clients render thinking in dim-cyan under a `[thinking]`
  header, answer in bold under `[answer]` — the two phases are visually
  distinct from each other and from user/tool content.

### Changed

- **All five benchmark examples now run via the harness daemon.** Four
  examples retrofitted onto the streaming protocol (the fifth, `parity_check`,
  shipped in v0.1.8):
  - `examples/reasoning_check.py` — harness-only, no in-process model load.
  - `examples/quality_check.py` — harness-only and generalized to
    `QUANT_FORMAT` (reads from `quant_env()` like the others); drops the
    `FP8_CHECKPOINT` hardcoding so NVFP4 quality runs work without a code
    change.
  - `examples/multi_turn_16k.py` and `examples/multi_turn_32k.py` —
    harness-only. The client reconstructs live terminal UX (dim-cyan
    `[thinking]`, bold `[answer]`, carriage-return status line) from streamed
    events, then generates the transcript HTML and dashboard PNG.
- **`stdout.log` scope for retrofitted examples.** Content that previously
  came from the in-process generation loop (per-turn `CudaTimer`,
  `mem_report`, etc.) now originates in the daemon and goes to
  `~/.aeo-quant/harness.log`. Client-side `stdout.log` retains preflight,
  event summaries, and post-run diffs. `tail ~/.aeo-quant/harness.log` for
  full daemon-side detail.
- **`LIVE=0` and `VERBOSE_THINK=1` env vars removed from
  `multi_turn_16k.py` / `multi_turn_32k.py`.** Thinking is part of the model's
  output, so the workload always streams it — no config flag, no alternate
  heartbeat mode. Users who want a silent run can redirect stderr:
  `uv run examples/multi_turn_16k.py 2>/dev/null`.
- **`results_dir()` now stamps the quant format and KV bits into the
  run-stem** when callers pass `format=` and `kv_bits=`. Layout is now
  `results/<category>/<format>-<bits>bit-<timestamp>/` for all four
  retrofitted examples — sortable by time, groupable by quant shape via
  glob (e.g. `ls results/reasoning/nvfp4-3bit-*`). The bare-timestamp
  layout is still the fallback when format/bits aren't supplied, and
  `RESULTS_DIR` env override still bypasses all formatting. The reasoning
  baseline path moved to `results/reasoning/baseline_<format>-<bits>bit/`
  so an FP8 baseline can no longer be silently compared against an NVFP4
  run (which was the source of the 86.8% divergence seen in the v0.1.8
  NVFP4 verification).

## [0.1.8] - 2026-04-16

### Added

- **Harness daemon** (`aeo_quant.harness`) — a long-running UNIX-socket
  service that loads the Gemma 4 model once and serves workload requests
  from multiple clients. First call to an example auto-spawns the daemon
  in the background; subsequent calls connect instantly to the already-
  loaded model, eliminating the ~140s load cost per invocation. Daemon
  is mutually-exclusive on `QUANT_FORMAT` — switching formats requires
  `aeo-harness stop && <new-format example>` which auto-spawns a fresh
  daemon.
  - `aeo-harness start [--format fp8|nvfp4]` — foreground or auto-spawn
  - `aeo-harness status` — current format, uptime, jobs_served, queue_depth
  - `aeo-harness stop` — graceful shutdown, frees ~27 GB
  - Socket at `~/.aeo-quant/harness.sock`, log at `~/.aeo-quant/harness.log`
- **Workloads subsystem** (`aeo_quant.workloads`) — pure-compute functions
  callable either in-process or via the harness. First workload landed:
  `workloads.parity.run()`. Additional workloads (reasoning, quality,
  multi_turn) arrive in a follow-up.
- **Log tailing during daemon spawn** — when an example auto-spawns the
  daemon, the parent streams `~/.aeo-quant/harness.log` to the user's
  terminal during the ~140s model load, so you see live progress (shard
  loading, NVFP4→FP8 conversion, etc.) instead of an opaque
  "waiting..." message.
- **Streaming event protocol** — server can emit zero or more
  `status: event` lines before the terminal `status: ok`/`status: error`
  reply. Clients receive events via a default stdout printer or an
  optional `on_event` callback — no opt-in required. Unlocks long-running
  workloads (multi_turn, future chat) with live progress without adding
  configuration surface.
- **Thread-pool workload execution on the server** — workloads now run
  via `loop.run_in_executor`, keeping the asyncio event loop responsive
  during generate() calls so streaming events flow in real time and
  concurrent `status` probes work.

### Changed

- **`examples/parity_check.py`** — refactored to always use the harness.
  `get_or_start_harness()` replaces the earlier manual load path;
  in-process fallback removed (failing loudly is more honest than
  silently reloading a 27 GB model when the daemon is misconfigured).
  Format mismatch between daemon and requested `QUANT_FORMAT` is a fatal
  error with clear remediation text.
- **`pyproject.toml`** — added `[project.scripts]` entry point for
  `aeo-harness`.

### Fixed

- **`Tee.write` / `Tee.flush` resilient to closed streams.** At interpreter
  shutdown, `atexit` may close the log file before Python's final
  `sys.stdout.flush()` runs, producing `Exception ignored while flushing
  sys.stdout` on `reasoning_check` and `profile_generate`. `Tee` now
  suppresses per-stream `ValueError`/`OSError` so one dead stream doesn't
  kill the other, and shutdown is clean.

## [0.1.7] - 2026-04-16

### Changed

- **`parity_check.py` — per-format baselines, SDK-aligned output path.**
  Replaced the single shared `tests/fixtures/parity_baseline.txt` with
  per-format files `parity_baseline_fp8.txt` and `parity_baseline_nvfp4.txt`.
  Each format compares against its own baseline as the regression gate;
  NVFP4 runs additionally compare against the FP8 baseline as an
  informational quality delta (never fails the run). Fixes the prior
  bug where whichever format ran last silently overwrote the other's
  pinned reference.
- **Parity output now uses the standard `results_dir()` timestamped
  subdirectory pattern** (`results/parity/YYYYMMDD-HHMMSS/output.txt`),
  matching `multi_turn_*`, `profile_generate`, `reasoning_check`,
  `compile_probe`. The inline `datetime.strftime` was the last remaining
  duplication of the SDK's timestamp helper; date/time format now lives
  in one place (`core/config.py:results_dir`).

### Added

- **`docs/plans/2026-04-16-native-nvfp4-matmul.md`** — roadmap spec for
  replacing the NVFP4→FP8 load-time conversion with a native NVFP4
  block-scaled matmul on sm_121. Documents the Triton tutorial-10 base
  kernel, the three required deltas (per-tensor fp32 scale fold-in,
  `matmul_ogs` TMA guard bypass, `sm_121f` compile target), small-M
  vs large-M tile split, and six ordered verification gates. Pre-gated
  by a 20-minute torchao `_addmm_nvfp4_dispatch` probe to rule out a
  zero-code integration path.
- **`kb/nvfp4-blackwell-research.md`** — new section "Native NVFP4
  matmul path on sm_121 — 2026-04-16 survey" with candidate kernel
  inventory, sm_121 shared-memory budget notes (99 KiB vs B200's
  228 KiB), and reference links.
- **`docs/gemma4-fp8-optimization.md`** — roadmap table of next
  decode-speed optimizations (R1 prompt lookup, R2 native NVFP4 matmul,
  R3 assisted decoding, R4 static KV cache) scoped to transformers only.

## [0.1.6] - 2026-04-16

### Added

- **`preflight_memory(min_available_gb, *, label)`** in `gpu/memory.py` —
  fail-fast check that refuses to start heavy workloads when the unified
  memory pool doesn't have enough free headroom. Prevents the "system
  buckling" scenario on shared GB10 where other processes have consumed
  so much memory that our workload's peak would push into swap-thrashing
  or OOM territory. Prints a PASS line on success so operators can see
  the headroom; exits 2 with a clear message on failure.
- All heavy example scripts now call `preflight_memory()` at startup
  with workload-specific minimums (documented as `MIN_FREE_GB` constants):
  - `parity_check.py`, `profile_generate.py`, `quality_check.py`,
    `compile_probe.py`, `cuda_graph_probe.py` — 50 GB
  - `reasoning_check.py` — 55 GB (longer decode)
  - `multi_turn_16k.py`, `multi_turn_32k.py` — 60 GB (KV growth)
  - `build_checkpoint.py`, `build_checkpoint_nvfp4.py` — 60 GB (~53 GB RSS peak)

### Changed

- Removed preexisting unused `pathlib.Path` imports from
  `multi_turn_16k.py`, `multi_turn_32k.py`, and `profile_generate.py`.
- `build_checkpoint*.py` — added `# noqa: SIM118` to `f.keys()` calls
  (safetensors `safe_open` requires `.keys()` for iteration).

## [0.1.5] - 2026-04-16

### Removed

- **`.fp8_cache/` sidecar and all cache save/load code in `load_gemma4_nvfp4()`.**
  The cache was designed to skip a 30–60s NVFP4→FP8 conversion on subsequent
  loads, but the batched-16-experts conversion optimization added in v0.1.4
  reduced conversion cost to ~10s. Measured cache load (~124s of disk I/O
  for 21 GB of shards) was consistently ~114s *slower* than just reconverting.
  Removing the cache simplifies the loader, eliminates stale-cache risk, and
  makes load times predictable. Full write-up in `docs/gemma4-fp8-optimization.md`
  Step 6. **If you have a `.fp8_cache/` directory inside your NVFP4 checkpoint
  dir from v0.1.4, delete it — it is no longer used.**

### Changed

- `load_gemma4_nvfp4()` now always runs the NVFP4→FP8 conversion. Total NVFP4
  load time: ~107s (97s `from_pretrained` + 10s conversion + compile wrap),
  same as before on first load, ~115s faster than the (broken) cached path.
- Docstring and module docstring in `bridges/gemma4/loader.py` updated to
  document why the cache was removed.

### Added

- Diagnostic logging in `_convert_nvfp4_experts_to_fp8()` and
  `load_gemma4_nvfp4()` — per-layer timing, `mem_report()` at key checkpoints,
  helps catch future memory regressions.
- `parity_check.py` now prints stage-by-stage timing and memory reports.
- KB note `kb/nvfp4-blackwell-research.md`: "Conversion Cache Removed" section.

## [0.1.4] - 2026-04-16

### Added

- **NVFP4 quantization pipeline** — full support for NVIDIA's microscaling
  FP4 (E2M1 with two-level FP8 block scales, block_size=16). Stores
  checkpoint at ~23 GB (vs 28.8 GB FP8, -19%); dequants to FP8 at load
  time for the proven `_scaled_mm` inference path.
  - `quantize_3d_to_nvfp4` / `dequant_3d_from_nvfp4` in `gpu/quant.py`
  - `quantize_2d_to_nvfp4` / `dequant_2d_from_nvfp4` (2D variants)
  - `Gemma4TextExpertsNVFP4` in `bridges/gemma4/modeling_nvfp4.py`
  - `load_gemma4_nvfp4()` in `bridges/gemma4/loader.py` — loads NVFP4
    checkpoint, converts to FP8, caches conversion in `.fp8_cache/`
  - `load_gemma4()` dispatcher — routes by `quant_format` parameter
  - `examples/build_checkpoint_nvfp4.py` — shard-streaming builder
- `quant_env()` in `core/config.py` — centralized env var reading for
  `QUANT_FORMAT`, `CHECKPOINT` / `FP8_CHECKPOINT` / `NVFP4_CHECKPOINT`,
  and `KV_BITS` (defaults: fp8/4-bit; nvfp4/3-bit).

### Changed

- **Example scripts unified** — `profile_generate.py`, `parity_check.py`,
  `reasoning_check.py`, `multi_turn_16k.py`, `multi_turn_32k.py` all use
  `quant_env()` + `load_gemma4()`. Set `QUANT_FORMAT=nvfp4` to switch.
  FP8 remains the default; no existing behavior changes.
- `.env` — added `NVFP4_CHECKPOINT` entry.

---

## [0.1.3] - 2026-04-16

### Added

- `results_dir(category, timestamped=True)` in `core/config.py` — SDK
  utility for timestamped results directories. Honors `RESULTS_DIR` env
  var as override. Replaces per-script boilerplate.
- `Tee` class in `core/writers.py` — multi-stream writer for
  stdout + log file tee. Replaces identical `_Tee` classes that were
  copy-pasted across example scripts.
- `openpyxl>=3.1` added to `plots` optional dependency group.

### Changed

- **All example scripts now use `results_dir()`** — timestamped output
  directories created via the SDK instead of per-script `Path()` +
  `mkdir()` + `strftime` boilerplate. Prevents results from being
  overwritten between runs.
- Removed unused `datetime` imports from `compile_probe.py`,
  `profile_generate.py`, and `reasoning_check.py`.

---

## [0.1.2] - 2026-04-16

### Added

- `examples/reasoning_check.py` — two hard reasoning prompts (Sylow
  subgroup proof, concurrent LRU cache bug hunt) that stress attention
  precision under different KV cache bit widths. Parameterized via
  `KV_BITS` env var, default 4.

### Changed

- **`KV_BITS` env var standardized** across all example scripts
  (`parity_check.py`, `profile_generate.py`, `compile_probe.py`,
  `cuda_graph_probe.py`, `quality_check.py`, `multi_turn_16k.py`,
  `multi_turn_32k.py`, `reasoning_check.py`). Replaces the old
  hardcoded `TURBOQUANT_BITS = 4`. Default is 4; users can set
  `KV_BITS=3` (or 2) to experiment with lower-precision KV cache.

### Removed

- `docs/archive/` — session continuation prompts from earlier Claude
  Code sessions. Internal artifacts, not public-facing docs.
- `examples/archive/` — old TRT-era validation scripts
  (`bf16_reference.py`, `nvfp4_validate.py`, `turboquant_validate.py`).
  Superseded by `quality_check.py` and `parity_check.py`.
- `tests/fixtures/parity_baseline.txt` — parity baseline is
  runtime-generated on first run, not a shipped fixture.

### Documented

- **TurboQuant 3-bit KV cache test results.** Tested against 4-bit
  using two reasoning-intensive 500-token prompts. Both produce correct
  reasoning at 3-bit (math proof valid, all 4 LRU bugs found) but
  86–98% token divergence from cascade effect. No decode speedup.
  Decision: 4-bit stays default — no memory pressure at 32K context
  (3.15 GB KV cache fits easily). Revisit 3-bit at 128K+ where
  the 3.15 GB savings becomes meaningful.
- E5M2 (`float8_e5m2`) format rejection: hardware requires E4M3 for
  RowWise `_scaled_mm` on Blackwell; E5M2 has half the mantissa
  precision and its wider range is irrelevant under per-row scaling.

---

## [0.1.1] - 2026-04-16

Building blocks for non-MoE FP8 quantization, plus documentation
improvements based on the initial public release.

### Added

- `quantize_2d_to_fp8` in `gpu/quant.py` — per-output-row FP8
  quantization for standard 2D weight matrices (`nn.Linear`). Companion
  to the existing `quantize_3d_to_fp8` for fused MoE experts.
- `LinearFP8` in `bridges/gemma4/linear_fp8.py` — drop-in `nn.Linear`
  replacement that stores FP8 weights with fp32 per-output-channel
  scales. Shared `fp8_linear()` function usable by both `LinearFP8` and
  the MoE expert path. `from_linear()` classmethod for in-place
  conversion.

### Changed

- **README restructured** — SDK-at-a-glance tree and Mermaid
  architecture diagram up front; full "What it does" details below.
  Hero section highlights the published FP8 checkpoint
  ([aeyeops/gemma-4-26b-a4b-it-fp8](https://huggingface.co/aeyeops/gemma-4-26b-a4b-it-fp8))
  with TurboQuant KV cache.
- **examples/README.md** — added `parity_check.py` documentation,
  `AEO_MOE_TRACE=1` nsys auto-wrap, pointed `.env` at the published
  HF checkpoint.
- **License** — switched from MIT to Apache 2.0 to align with the ML
  tooling ecosystem (transformers, accelerate, vLLM, turboquant).

### Documented

- Non-MoE FP8 quantization investigation results in
  `docs/gemma4-fp8-optimization.md`: 206 Linear modules converted,
  -840 MB VRAM, but 0% decode speedup and 46% parity divergence.
  Rejected — MoE-only FP8 remains the shipped configuration.

---

## [0.1.0] - 2026-04-15

Initial public release. Establishes the package structure, the bridge
pattern for wiring quantized weights into `transformers`, and a complete
FP8 path for Gemma 4 26B-A4B on NVIDIA Blackwell.

### Added

**Package foundation**
- Four-layer architecture: `core` (stdlib), `gpu` (torch + psutil),
  `bridges` (transformers), `plots` (matplotlib). Each layer only
  imports its own dependencies; `import aeo_quant` is always safe.
- `.env`-based configuration with `load_dotenv()` overriding shell env
  vars — single source of truth per checkout.

**Gemma 4 FP8 bridge** (`src/aeo_quant/bridges/gemma4/`)
- `Gemma4TextExpertsFP8` — drop-in subclass of the upstream
  `Gemma4TextExperts` that stores FP8 (`float8_e4m3fn`) expert weights
  with per-output-channel bf16 scales. Forward pass uses
  `torch._scaled_mm` with per-row dynamic input quantization.
- Class-swap loader (`load_gemma4_fp8`) — context manager that
  temporarily replaces the upstream expert class during
  `from_pretrained`, then restores it. No monkey-patching persists.
- Pre-converted fp32 scales at load time — eliminates per-call
  `squeeze/unsqueeze/float/contiguous` in the expert forward path.
  +30 MB memory, measurable prefill improvement.
- Opt-in NVTX trace markers (`AEO_MOE_TRACE=1`) for Nsight Systems
  profiling of the MoE expert path. `profile_generate.py` auto-wraps
  under `nsys` when the flag is set.

**Checkpoint builder** (`examples/build_checkpoint.py`)
- Shard-streaming FP8 quantization of fused 3D MoE experts from a
  bf16 safetensors source. Peaks ~18 GB RSS on a 26B model.
- Required because public Gemma 4 FP8 checkpoints ship with broken
  expert scales (see `docs/gemma4-fp8-results.md`).

**Benchmarking and diagnostics**
- `examples/profile_generate.py` — CUDA-event timing (prefill vs
  decode), optional `torch.profiler` kernel trace, stdout captured to
  timestamped `results/profiling/<ts>/stdout.log`.
- `examples/parity_check.py` — 50-token greedy regression canary
  against a pinned baseline (`tests/fixtures/parity_baseline.txt`).
- `examples/quality_check.py` — three-prompt smoke test for coherence.
- `CudaTimer` context manager and `mem_report()` for memory tracking.

**GPU utilities** (`src/aeo_quant/gpu/`)
- `quantize_3d_to_fp8` — per-expert per-output-channel FP8
  quantization for fused 3D weight tensors.
- Memory reporting with system, process, and torch allocator stats.

**Documentation**
- `docs/gemma4-fp8-results.md` — checkpoint build and validation.
- `docs/gemma4-fp8-optimization.md` — decode optimization log with
  active plan.
- `docs/gemma4-fp8-retrospective.md` — build effort retrospective.
- `docs/turboquant-gemma4-research.md` — TurboQuant KV cache notes.

### Performance (Gemma 4 26B-A4B, GB10, 100 tokens, TurboQuant-4bit KV)

| Metric         | bf16 dequant baseline | v0.1.0 (`_scaled_mm`) |
|----------------|----------------------:|-----------------------:|
| Decode tok/s   |                  7.82 |                   8.96 |
| Prefill (ms)   |                   639 |                    487 |
| Peak VRAM (GB) |                 26.93 |                  26.86 |
