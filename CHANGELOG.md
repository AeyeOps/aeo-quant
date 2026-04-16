# Changelog

All notable changes to this project will be documented in this file.

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
