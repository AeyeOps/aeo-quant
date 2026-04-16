# TurboQuant + Gemma 4 26B-A4B research findings (April 2026)

**Status:** research notes. Prior session already hit these dead ends once — this file exists so future sessions stop repeating the search.

**Goal (user, non-negotiable):** validate TurboQuant KV cache compression specifically on **Gemma 4 26B-A4B MoE**. Not a smaller model, not a dense model.

## The short version

**We're on the wrong stack.** People running Gemma 4 26B-A4B with TurboQuant at long context (up to 262K tokens) are using a **llama.cpp fork with TurboQuant KV kernels**, not `transformers.generate()` + `TurboQuantCache`.

The pip `turboquant==0.2.0` package we've been using is the research reference implementation for the ICLR 2026 paper — correct math, no kernel fusion, naive `dequant-full-KV-per-step` loop. It's meant to prove the distortion bounds, not deliver fast inference.

## Evidence — working setups in the wild

### 1. `conorseabrook/gemma4-turboquant-bench` — our exact use case

- GitHub: `https://github.com/conorseabrook/gemma4-turboquant-bench`
- Claim: **"Gemma 4 26B + TurboQuant: 262K context on a single RTX 4090. Agentic coding benchmark via Claude Code."**
- Stack: **llama.cpp fork** at `TheTom/llama-cpp-turboquant`, branch `feature/turboquant-kv-cache`
- Build: requires compiling from source (`-DCMAKE_CUDA_ARCHITECTURES=89` for 4090 / Ada, 86 for Ampere, 75 for Turing)
- Run mode: **`llama-server`** as a local OpenAI-compatible HTTP endpoint; Claude Code connects to it
- KV quant type: **`turbo3`** (3-bit) preferred over `turbo4` — "faster prefill via tensor-core MMA codepath"
- **`-fa on` (Flash Attention) is REQUIRED** for turbo KV types
- Context: full 262,144 tokens
- Not in mainline llama.cpp or Ollama as of April 2026

### 2. `0xSero/turboquant` — production-adjacent Python

- GitHub: `https://github.com/0xSero/turboquant`
- Claim: "Near-optimal KV cache quantization for LLM inference (3-bit keys, 2-bit values) with **Triton kernels + vLLM integration**"
- Install: `pip install -e .` from source
- Integration: monkey-patches vLLM (`integration/vllm.py`, `vllm_attn_backend.py`)
- **Critical caveat for MoE:** "Only full-attention layers are compressed; linear-attention and Mamba layers remain uncompressed". On MoE with mixed attention types (including Gemma 4's 5 full + 25 sliding layout), savings drop from ~50% to **~30.9%** because the sliding/local layers aren't compressible
- Decode path: "All compressed tokens are expanded to float32" — same naive dequant pattern we saw in pip turboquant, but with Triton kernels making it faster
- On dense Qwen3.5-27B-AWQ: "2.0x context capacity, frees ~30 GB across 4 GPUs"

### 3. `mlx-vlm` (Apple Silicon only — not us)

- CLI: `mlx_vlm.generate --model "mlx-community/gemma-4-26b-a4b-it-4bit" --prompt "..." --kv-bits 3.5 --kv-quant-scheme turboquant`
- Claim: same accuracy as uncompressed, ~4× less active memory, "a lot faster end-to-end"
- Doesn't help us directly (we're on Blackwell/CUDA), but confirms the scheme name `turboquant` is the reference

### 4. Pre-quantized HF model

- `majentik/gemma-4-E4B-turboquant` — an E4B variant, not 26B-A4B. Structural precedent, wrong size.

### 5. `Incept5/gemma4-benchmark` — MLX benchmark

- GitHub: `https://github.com/Incept5/gemma4-benchmark`
- "MLX benchmark: Gemma 4 + Qwen 3.5 on Apple Silicon with TurboQuant KV cache"
- Apple Silicon only.

## Evidence — expected tok/s on GB10

- DGX Spark / GB10 users report **45–60 tok/s** on Gemma 4 26B-A4B
- Source: NVIDIA Developer Forum thread "Someone post this: Gemma 4 26B-A4B MoE running at 45-60 tok/s on DGX Spark"
- URL: `https://forums.developer.nvidia.com/t/someone-post-this-gemma-4-26b-a4b-moe-running-at-45-60-tok-s-on-dgx-spark/365547`
- **Our transformers.generate() + pip turboquant setup is getting 1.8–6.2 tok/s** depending on context. That's 10× off the right ballpark — a strong signal that the stack itself is wrong, not a tuning issue.

## Why our transformers path has the prefill explosion

Observed on this machine (Dell GB10 Max Pro, 128 GB unified LPDDR5x):

| Context | Option B + TurboQuant prefill delta | Option B + TurboQuant peak sys_used |
|---|---|---|
| 4K | +7 GB | 47 GB |
| 8K | +17 GB | 56 GB |
| 16K | +42 GB (killed at 82 GB threshold) | OOMs at 90 GB cap |

- Superlinear scaling (~8 GB more than linear from 8K → 16K) is consistent with **eager attention materialization on the 5 full-attention layers**: `16384² × num_kv_heads=8 × 2 bytes × 5 layers ≈ 21 GB`
- Forcing `attn_implementation="sdpa"` at `from_pretrained()` did NOT change the peak — same 80+ GB at 16K
- Swapping `TurboQuantCache` for `DynamicCache` did NOT change it either — same 80+ GB
- Option B's FP8 forward patch (per-expert dequant) was eliminated as a cause (tried both per-call and batched variants, same peak)
- So the explosion is in the **base transformers Gemma 4 forward path itself**, not in TurboQuant or our patch
- The working stacks (llama.cpp fork, vLLM) all use **paged attention / flash attention** for prefill, which avoids the materialization transformers' Python path hits

## What the pip `turboquant==0.2.0` package actually is

- Source: `.venv/lib/python3.14/site-packages/turboquant/` (inspected)
- Docstring: *"First open-source implementation of Google's TurboQuant (ICLR 2026)"*
- Paper: `arXiv:2504.19874` — Zandieh, Daliri, Hadian, Mirrokni (Google Research)
- Two algorithms:
  - `TurboQuantMSE` (Algorithm 1) — MSE-optimal quantization via random rotation + Lloyd–Max codebook from a Beta distribution
  - `TurboQuantIP` (Algorithm 2) — inner-product optimal via MSE + QJL residual
- `TurboQuantCache` is a `DynamicCache` subclass with `residual_len=128` FP16 window
- `TurboQuantLayer.update()` cat's the new token's KV into the FP16 residual, quantizes overflow, then **dequantizes the entire compressed history to a fresh FP16 tensor on every call** (lines 107–119 in `cache.py`)
- That last step is the O(T)-per-step decode tax we observed — inherent to this implementation, no config knob disables it
- Correct API usage (confirmed from `__init__.py` docstring):
  ```python
  from turboquant import TurboQuantCache
  cache = TurboQuantCache(bits=4)
  outputs = model.generate(..., past_key_values=cache)
  ```
- **We have not been using it wrong.** The slow decode is what this implementation is.

## Practical path forward for the Gemma 4 26B-A4B + TurboQuant validation goal

Ranked by likely-to-work and by how much it hurts our current test harness:

1. **Build `TheTom/llama-cpp-turboquant` for Blackwell SM121** and validate with `llama-server` + `turbo3` + `-fa on`. This is the only documented working path at 262K on Gemma 4 26B. Needs `-DCMAKE_CUDA_ARCHITECTURES=121` (or `121a`), a CUDA 13 toolchain, and the branch `feature/turboquant-kv-cache`. Biggest unknown: does Flash Attention have working Blackwell SM121 kernels in that fork.
2. **Try `0xSero/turboquant` with vLLM on Gemma 4 26B-A4B**. vLLM has its own Gemma 4 support; the monkey-patch route is reported to work on Qwen3.5-27B-AWQ dense. For Gemma 4 MoE expect only ~30% KV savings (only the 5 full-attn layers get compressed) instead of the advertised 4×. Needs the vLLM Blackwell path working.
3. **Stay on pip turboquant + transformers** only as a quality reference at short context (confirmed coherent at ≤8K). Don't chase long context down this path — the prefill explosion is a transformers Gemma 4 issue, not a TurboQuant issue, and the decode cost is the reference implementation's O(T) dequant loop.

## Things NOT to re-research next session

- Don't re-verify that pip `turboquant==0.2.0` is a reference implementation — it says so in its own README and the cache.py naive dequant is plainly visible
- Don't re-run 16K Option B on this transformers path looking for a fix — the explosion is in the base model forward, not anything we can monkey-patch from outside without rewriting attention
- Don't try more `attn_implementation` values on `from_pretrained` — `sdpa` is already the default AND forcing it explicitly did not change peak memory
- Don't pivot to a smaller model to isolate TurboQuant behavior — user explicitly ruled that out, the target IS Gemma 4 26B-A4B MoE specifically
- Don't pivot to DynamicCache to "get decode speed back" — we're validating TurboQuant, not building a production inference server
