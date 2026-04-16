# Changelog

All notable changes to this project will be documented in this file.

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
