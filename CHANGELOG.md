# Changelog

All notable changes to this project will be documented in this file.

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
