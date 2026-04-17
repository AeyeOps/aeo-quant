# Gemma 4 26B-A4B FP8 — Decode Optimization Results

**Date started:** 2026-04-15
**Model:** `google/gemma-4-26B-A4B-it` (self-built FP8 checkpoint)
**Platform:** NVIDIA GB10 (Blackwell sm_121, ARM64)

This doc tracks the runtime optimization work on the FP8 Gemma 4 bridge. The build/validation of the FP8 checkpoint itself is documented separately in [`gemma4-fp8-results.md`](./gemma4-fp8-results.md); that work is complete and unchanged.

See the active plan at [`plans/2026-04-15-fp8-moe-decode-optimization.md`](./plans/2026-04-15-fp8-moe-decode-optimization.md).

---

## Starting point (pre-optimization)

The FP8 bridge shipped with a `Gemma4TextExpertsFP8.forward` that dequantized each selected expert's weights into transient bf16 before a standard `nn.functional.linear`:

```python
gate_up_w = self.gate_up_proj[expert_idx].to(torch.bfloat16) * self.gate_up_proj_scale[expert_idx]
gate, up = nn.functional.linear(current_state, gate_up_w).chunk(2, dim=-1)
```

100-token decode benchmark (TurboQuant-4bit KV, 61-token prompt):

| Metric | Value |
|---|---|
| Decode tok/s | 7.82 |
| Decode time | 12,795 ms |
| Prefill time | 639 ms |
| Total | 13,435 ms |
| Peak VRAM | 26.93 GB |

`torch.profiler` breakdown pointed at:
- `aten::mm` 38.9% CUDA (bf16 matmul post-dequant)
- `aten::copy_` + `aten::to` 40% CUDA (FP8→bf16 dequant traffic)
- `aten::mul` 19.8% CUDA (scale multiply)
- `aten::item` 31.8% CPU, `aten::nonzero` 10.5% CPU (Python expert routing loop)

Self CPU 13.02 s / Self CUDA 8.68 s ≈ 1.5× ratio. Interpretable as "roughly balanced, some overlap."

---

## Attempts and outcomes

### Rejected: batched decode fast path (index_select + bmm)

Hypothesis: gather all k selected expert weights via `index_select`, fuse into a single `bmm`. Eliminates the Python expert loop's `.item()` / `nonzero` host syncs.

Result: **+0% speedup.** Decode went from 7.82 → 7.73 tok/s (within noise).

The `.item()` cost was **latent-overlapped** with GPU compute — the GPU had queued enough work between syncs that the CPU overhead wasn't actually stalling it. Reducing launch count didn't help because the individual matmul sizes were already large enough to saturate.

The worktree branch `feat/cpu-bottleneck` contains a larger, independently developed implementation of the same idea (`moe_decode.py`, `moe_pack.py`) with the same conclusion on the same hardware: **routing batching does not beat eager on GB10 for this model.** The CPU numbers on paper look large but don't translate into wall-clock improvement.

Do not re-implement.

### Shipped: `torch._scaled_mm` FP8-native matmul *(with a correctness bug caught by the parity harness)*

Replaced the `.to(bf16) * scale + linear` pattern with `torch._scaled_mm` on the FP8 weights directly:

```python
fp8_max = 448.0
x_amax = x_bf16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4).to(torch.float32)
x_scale = x_amax / fp8_max                          # (M, 1) fp32
x_fp8 = (x_bf16.to(torch.float32) / x_scale).to(torch.float8_e4m3fn)
return torch._scaled_mm(
    x_fp8, w_fp8.t(),
    scale_a=x_scale, scale_b=scale_w,   # scale_w is (1, out) fp32 from the caller
    out_dtype=torch.bfloat16,
)
```

Key details:
- The weight's `.t()` view on `[out, in]` row-major FP8 is **column-major for free** (`stride(0)==1`) — no copy; that's what `_scaled_mm` needs for B in RowWise mode.
- Per-output-channel bf16 scales from the checkpoint convert to `(1, out) fp32` for `scale_b` (RowWise).
- **Input is dynamically quantized per row:** each row of `x_bf16` gets its own `amax`-based scale so it uses FP8's full range. `_scaled_mm` multiplies the scale back during the matmul.

**Correctness bug found after initial ship (important lesson):**

The first cut of `_fp8_linear` used `scale_a = torch.ones(M, 1)` — i.e., no input scaling, just a direct `x_bf16.to(float8_e4m3fn)`. Post-RMS-norm hidden states are in `|x| < ~3`, which lands on only ~16 distinct FP8 values for that range. Accumulated across 24 MoE layers × 8 experts per token × 2816-element dot products, the quantization error grew large enough to flip `argmax` on the vocab head. The model produced 100% `<pad>` tokens on chat-templated prompts with `enable_thinking=True`.

**`profile_generate.py` did not detect this** because it only reports token *counts*, not content. The initial "+27% decode / +83% prefill" numbers were measured on **garbage output**. Only the `parity_check.py` harness (see `examples/parity_check.py`) caught the regression, by diffing generated token IDs against a pinned baseline.

Fix: the per-row dynamic amax scaling shown above. This is the lesson for the rest of the optimization plan — every change needs a parity gate, not just a timing number.

**Corrected results** (same benchmark, 100 tokens, fix verified by parity check):

| Metric | Pre-_scaled_mm | Post (fixed) | Delta |
|---|---|---|---|
| Decode tok/s | 7.82 | **8.94** | **+14%** |
| Decode time | 12,795 ms | **11,189 ms** | **-13%** |
| Prefill time | 639 ms | **519 ms** | **-19%** |
| Peak VRAM | 26.93 GB | 26.92 GB | unchanged |

Parity: first **29 of 50** greedy-decoded tokens match the bf16-reference exactly (same token IDs). Divergence after position 30 is cascade-driven FP8 precision loss — the FP8 path emits a coherent, on-topic thinking-channel response to the same prompt, just with slightly shorter phrasing on a few decisions. The `tests/fixtures/parity_baseline.txt` reference has been pinned to the FP8 output so subsequent optimization steps can diff byte-for-byte against it.

Commits: `perf: switch gemma4 fp8 experts to torch._scaled_mm` (initial, broken), `fix: dynamic input scaling in gemma4 _scaled_mm path` (correct).

---

## New bottleneck picture (post-`_scaled_mm`)

30-token profiler run (`PROFILE_TRACE=1 uv run examples/profile_generate.py`, taken against the initial *broken* `_scaled_mm` — a re-run against the fixed code is on the Step-1 todo and numbers may shift slightly):

- Self CPU **3.66 s** / Self CUDA **1.54 s** → **2.4× ratio**. The GPU is clearly **starved**. Kernel launch overhead dominates.
- CUDA time distribution:
  - Non-MoE bf16 `aten::mm` — **53%** (attention q/k/v/o projections, non-MoE MLP, LM head). The `[2816, 262144]` LM head alone is **17%**.
  - `_scaled_mm` (MoE experts) — **17%**. No longer the bottleneck.
  - `aten::copy_` / `aten::to` — **17%** (contiguity + dtype).
  - Everything else — ~13%.
- CPU hot path shifted: `aten::item` dropped from 31.8% to **0.8%** (proportional — total CPU shrank). Kernel launch overhead is now the dominant CPU cost, spread across ~250 k kernel launches per 30 tokens (~8.3 k kernels per decoded token).

This is why routing-batching experiments don't move the wall clock: the Python expert loop was never the real bottleneck. The real cost was bandwidth-bound dequant + bf16 matmul. `_scaled_mm` addressed both. What's left is per-kernel launch overhead — which is what graphs and compile are for — and 53% of CUDA time still running in bf16 because only the MoE experts were quantized.

---

## Optimization results (post-`_scaled_mm`)

All numbers: 100-token greedy decode, TurboQuant-4bit KV, 61-token prompt.

### Shipped: NVTX trace markers (Step 1)

Four opt-in named ranges (`fp8_moe_route`, `fp8_moe_gate_up`, `fp8_moe_down`, `fp8_moe_combine`). Gated by `AEO_MOE_TRACE=1` — zero cost when off. When set, `profile_generate.py` auto-wraps under `nsys profile` via `os.execvp`, so the switch is all-or-nothing: markers + collector together. Trace lands in `results/nsys/<timestamp>/`.

NVTX summary from a 100-token run:
- `fp8_moe_gate_up` 41.3% of MoE time (27,732 calls, 120 us avg)
- `fp8_moe_down` 38.7% (27,732 calls, 112 us avg)
- `fp8_moe_route` 8.0% (3,030 calls, 211 us avg)
- `fp8_moe_combine` 7.2% (27,732 calls, 21 us avg)

Gate-up and down (the two `_fp8_linear` calls) together are 80% of MoE time — the targets for the hygiene step.

Decode tok/s: unchanged from baseline (8.92 with markers off).

### Shipped: hot-path hygiene (Step 2)

Pre-convert scale buffers from `(E, out, 1) bf16` to `(E, 1, out) fp32` at load time. Eliminates per-call `squeeze(-1).unsqueeze(0).float().contiguous()` across 55k expert invocations per 100 tokens. bf16 to fp32 is mathematically lossless. Cost: +30 MB.

| Metric | Pre-hygiene | Post-hygiene | Delta |
|---|---|---|---|
| Decode tok/s | 8.92 | 8.96 | +0.4% (noise) |
| Prefill ms | 521 | 487 | -6.5% |
| torch_alloc | 26.83 GB | 26.86 GB | +30 MB |

Parity: 50/50 byte-for-byte match.

### Shipped: `torch.compile(mode="reduce-overhead")` (Step 3b)

CUDA graph capture of the full model forward fails (`cudaErrorStreamCaptureInvalidated`) because the MoE routing loop has data-dependent control flow. `torch.compile` handles this by falling back to eager at graph breaks and compiling the rest.

| Metric | Pre-compile | Post-compile | Delta |
|---|---|---|---|
| Decode tok/s | 8.96 | **10.08** | **+12.5%** |
| Prefill ms | 487 | 482 | -1% |
| Warmup | — | 1.3 s | one-time per process |
| Peak VRAM | 26.86 GB | 26.95 GB | +90 MB |

Parity: 50/50 byte-for-byte match. `torch.compile` preserves semantics exactly.

### Rejected: non-MoE FP8 quantization (Step 4)

Implemented `LinearFP8` (drop-in `nn.Linear` replacement using `torch._scaled_mm`) and `quantize_2d_to_fp8`. Post-load swap of 206 non-MoE Linear modules (attention q/k/v/o, non-MoE MLP gate/up/down, LM head).

Results:
- Decode tok/s: 10.10 (no improvement over 10.08 — `torch.compile` already optimized these matmuls)
- VRAM: -840 MB (26.02 vs 26.86 GB — real savings from halving 206 Linear modules)
- **Parity: 26/50 prefix match, 46% token mismatch** — coherent output but significant divergence

The lack of decode improvement combined with quality divergence made this a net negative. `torch.compile` already addressed the kernel launch overhead that was the non-MoE bottleneck. FP8 quantization on top of that only adds quantization noise through all 24 layers of attention + MLP + LM head, with the error cascading through greedy decode.

The building blocks (`LinearFP8`, `quantize_2d_to_fp8`) are retained in the codebase for future use — they're correct and tested but not wired into the default path.

---

## Final performance summary

| Optimization | Decode tok/s | Cumulative vs baseline |
|---|---|---|
| Baseline (bf16 dequant) | 7.82 | — |
| `torch._scaled_mm` FP8-native matmul | 8.94 | +14% |
| Hot-path hygiene (fp32 scales) | 8.96 | +15% |
| `torch.compile(mode="reduce-overhead")` | **10.08** | **+29%** |

Peak VRAM: 26.95 GB. Parity: byte-for-byte match with FP8 baseline across all shipped steps.

### Rejected: E5M2 as alternative to E4M3 for weight quantization

Researched whether `torch.float8_e5m2` (2 mantissa bits, range [-57344, 57344]) would give better precision than `torch.float8_e4m3fn` (3 mantissa bits, range [-448, 448]) for the non-MoE weights where E4M3 caused quality loss.

Findings:
- **Hardware blocks it.** On Blackwell sm_121, `torch._scaled_mm` with RowWise scaling (our scaling mode) **requires B (weight) matrix to be E4M3**. E5M2 weights are only supported with TensorWise (scalar) scaling, which is strictly worse.
- **Precision is worse.** E5M2 has half the mantissa precision of E4M3 (4 steps vs 8 steps per exponent interval). Empirical round-trip error: E5M2 shows 1.57x higher NRMAE than E4M3 at the same scaling granularity, even on tensors with outlier columns simulating attention weight distributions.
- **Range advantage is irrelevant.** Per-row scaling already normalizes dynamic range into the representable interval. E5M2's 128x wider range provides zero benefit when each row has its own scale.
- **Literature consensus.** E4M3 for inference (weights + activations). E5M2 exists for training gradients where wide range handles gradient spikes. No published work recommends E5M2 for inference weights.

E4M3 is already the right format. The 46% divergence from non-MoE FP8 quantization is a layer-sensitivity problem (cumulative error through 24 layers + LM head argmax sensitivity), not a format problem. Potential future directions: mixed precision (LM head stays bf16), finer-grained scaling (blockwise 1x128 or MXFP8 1x32), or NVFP4.

### Tested: TurboQuant 3-bit vs 4-bit KV cache

TurboQuant supports `bits=1,2,3,4` (its default is actually `bits=3`). We tested 3-bit against 4-bit using two reasoning-intensive prompts designed to stress attention precision:

1. **Math proof** — prove Sylow q-subgroup normality for groups of order p²q. Requires tracking abstract algebraic constraints across the full context.
2. **Concurrent LRU cache** — find 4 interacting bugs (mutable default, race condition, deadlock-prone lock ordering, off-by-one). Requires reasoning about interleaved execution paths and shared mutable state.

Each prompt generated 500 greedy tokens at `bits=4` and `bits=3`. Test harness: `examples/reasoning_check.py` (parameterized via `KV_BITS` env var, default 4).

**Results:**

| | 4-bit KV | 3-bit KV |
|---|---|---|
| Math proof | Correct, rigorous | Correct, rigorous |
| LRU bugs | All 4 found, proper fixes | All 4 found, proper fixes |
| Math tok/s | 9.30 | 9.28 |
| LRU tok/s | 8.56 | 8.53 |
| Math token match vs 4-bit | — | 14% (61-token prefix) |
| LRU token match vs 4-bit | — | 2% (2-token prefix) |

Both outputs are correct and coherent at 3-bit. The massive token divergence (86–98%) is cascade-driven — different phrasing at one point snowballs through greedy decode — not a quality failure. The reasoning and correctness are indistinguishable between 3-bit and 4-bit.

**Memory sizing at 32K context:**

Gemma 4 26B-A4B: 24 layers, 32 KV heads, 128 head dim. Per-token KV in bf16 = 393 KB.

| KV cache | At 32K tokens | At 128K tokens |
|---|---|---|
| bf16 (no quant) | 12.6 GB | 50.3 GB |
| TurboQuant 4-bit | ~3.15 GB | ~12.6 GB |
| TurboQuant 3-bit | ~2.36 GB | ~9.4 GB |
| Savings (3 vs 4) | 790 MB | 3.15 GB |

**Decision: 4-bit stays as default.** At 32K context (our current target), TurboQuant 4-bit KV cache fits easily (~3.15 GB + 26.86 GB model = ~30 GB, well under the ~70 GB available on GB10). 3-bit saves 790 MB but doesn't improve decode speed. The quality is equivalent but the token-level divergence means parity checks would fail, making it harder to gate future optimizations.

All example scripts accept `KV_BITS` as an env var so users can experiment with 3-bit (or 2-bit) at their discretion. Revisit 3-bit when pushing to 128K+ context where the 3.15 GB savings becomes meaningful.

The reasoning check prompts (`examples/reasoning_check.py`) serve as the quality gate for any future KV cache changes. They test attention precision directly — both prompts require the model to hold and reference information from early in the context to produce correct output later.

---

## Step 5: NVFP4 checkpoint (v0.1.4)

**Goal:** Halve expert weight storage via NVFP4 (FP4 E2M1 + two-level microscaling), while keeping the proven `_scaled_mm` FP8 inference path.

**Approach:** Store NVFP4 in the checkpoint, dequant to FP8 at load time. Native FP4 matmul is blocked on sm_121 (CUTLASS/FlashInfer bugs), so we run on the proven pipeline. When native kernels land, the upgrade is localized to the loader.

**NVFP4 format:** Two-level microscaling with block_size=16 (not 32 like MXFP4):
- Level 1: FP8 E4M3 scale per 16-element micro-block
- Level 2: FP32 scalar per tensor
- Storage: uint8 packed (2 nibbles/byte) + fp8 block scales + fp32 tensor scale

**Checkpoint size:**

| Checkpoint | Expert weights | Non-expert (bf16) | Total |
|---|---|---|---|
| FP8 (v0.1.0) | ~12.3 GB | ~16.5 GB | ~28.8 GB |
| NVFP4 (v0.1.4) | ~6.9 GB + scales | ~16.5 GB | ~18 GB |
| Savings | | | **-37%** |

**Double-quantization error:** bf16->NVFP4->bf16->FP8 vs direct bf16->FP8: ~17% mean relative diff. Acceptable for FP4 coarseness.

**Build:** `uv run examples/build_checkpoint_nvfp4.py` (~2 min, shard-streaming, ~53 GB peak RSS)

**Load:** Every load runs the NVFP4→FP8 conversion (~10s). No on-disk conversion cache. See "The FP8 conversion cache that wasn't" below.

**Usage:** `QUANT_FORMAT=nvfp4 uv run examples/profile_generate.py`

## Step 6: The FP8 conversion cache that wasn't (v0.1.5)

**Context:** The NVFP4 plan called for a `.fp8_cache/` sidecar to avoid re-running conversion on every load. First load would do `NVFP4 → bf16 → FP8` and save the converted FP8 tensors; subsequent loads would read them directly, skipping conversion. The plan estimated this would save 25–45s per subsequent load.

### The adventure

**Attempt 1 — single-file cache.** First implementation saved all 120 expert tensors as one `cached.safetensors` via `save_file(tensors_dict)`. The dict-build step copied all 120 tensors to CPU (`.cpu()`) *before* the file write, accumulating ~23 GB of CPU-side copies on top of the already-loaded ~29 GB CUDA model. On a 128 GB unified-memory GB10 with other agents active, this pushed total usage past 120 GB and triggered kernel swap thrashing — the whole system froze, required a long wait to recover.

**Fix 1 — sharded writes.** Rewrote the cache as per-layer shards (30 files, one per MoE layer, ~727 MB each). Peak CPU-side temp dropped to ~762 MB. First cache build ran cleanly in ~40s.

**Attempt 2 — cache reload.** Second invocation took the cache path. Expected: fast reload. Reality: cache load took **124s**, making total load time 221.7s. The cache path was **75 seconds slower than just reconverting**.

### Why the cache failed its premise

The cache's design assumption was that NVFP4→FP8 conversion would be expensive (plan estimate: 30–60s). The batched-16-experts-at-a-time conversion optimization we added reduced conversion to **9.5s**. The cache paid 124s of disk I/O to skip a 9.5s compute step. The math is a net loss of ~114s per load.

On top of that, the cache-hit path still called `from_pretrained` first (to load non-expert weights), which also loads the NVFP4 expert buffers — **paying disk I/O for the expert data twice** (once as NVFP4, once as FP8) before discarding the NVFP4 copy.

### The decision

**Removed the cache entirely in v0.1.5.** Every NVFP4 load reconverts. Current timing:

| Path | Time |
|---|---|
| `from_pretrained` | ~97s |
| NVFP4→FP8 conversion (batched-16) | ~10s |
| `_preconvert_fp8_scales` + `torch.compile` wrap | <1s |
| **Total** | **~107s + torch.compile warmup on first generate** |

Simpler code, no stale-cache risk, no silent freezes, predictable load time.

### The lesson

**When you optimize something, revisit the design decisions that depended on the old cost.** The FP8 cache was a sensible design for a 30–60s conversion. Batching cut conversion to 9.5s. We should have re-evaluated the cache at that point instead of implementing it and only noticing when the runtime measurements didn't match the design's premise.

Two memories saved to help future sessions catch this class of mistake earlier:
- `feedback_batch_expert_conversion.md` — the optimization that changed the cost
- `feedback_ask_before_heavy_gpu.md` — the near-miss with system freeze

### When the cache *would* make sense again

If CUTLASS/FlashInfer sm_121 bugs get fixed and we switch to native FP4 inference, we'd keep NVFP4 weights on CUDA (not convert to FP8). At that point the cache question becomes moot — there's no conversion to cache. Current `_convert_nvfp4_experts_to_fp8` becomes dead code we'd remove.

---

## Roadmap — next decode-speed optimizations

Scope: `transformers.generate()` + our bridge, GB10, no extra memory pressure unless explicitly noted. Baseline to beat: 10.08 tok/s (FP8 and NVFP4, identical compute path).

| # | Optimization | Expected gain | Memory delta | Confidence | Risk / notes |
|---|---|---|---|---|---|
| R1 | Prompt lookup decoding (`generate(prompt_lookup_num_tokens=N)`) | 1.3–2× on repetitive / long-context workloads | ~0 | High | transformers-native, zero new code, A/B via env var; quality is bit-exact (verifier is the same model) |
| R2 | Native NVFP4 matmul (torchao probe first; own Triton kernel if probe fails) | up to ~5× (community NVFP4 ceiling ≈ 52 tok/s on same hw) | ~0 (keeps FP4 in GPU mem) | Medium — no sm_121 drop-in exists per 2026-04-16 survey | Spec: [`plans/2026-04-16-native-nvfp4-matmul.md`](./plans/2026-04-16-native-nvfp4-matmul.md). **Prereq:** 20-min torchao `_addmm_nvfp4_dispatch` probe (`examples/probe_nvfp4_torchao.py`) to decide between zero-code integration vs custom Triton kernel. KB: `kb/nvfp4-blackwell-research.md` § "Native NVFP4 matmul path on sm_121". |
| R3 | Assisted decoding (`generate(assistant_model=small_draft)`) | 2–4× on reasoning-heavy runs | +1–2 GB (draft model) | High value, medium confidence | transformers-native; requires a vocabulary-compatible draft; the one optimization that *adds* memory but the trade is decisively worth it |
| R4 | Static KV cache + CUDA graph decode | 2–4× on standard stacks | ~0 | Low for us — likely blocked | `TurboQuantCache` is a `DynamicCache` subclass with O(T)-per-step dequant; probably can't be graph-captured without dropping TurboQuant |

**Execution order (proposed):** R1 first (cheapest signal), R2 torchao probe in parallel (20 min, one expert, no full-model load), then either wire torchao or commit to the Triton kernel plan. R3 after R2 lands. R4 stays on the list but won't be touched until/unless TurboQuant scope changes.

**Out of scope reminder:** anything that requires switching backends (vLLM, TRT-LLM, llama.cpp, FlashInfer-as-backend) is not on this roadmap. See `feedback_aeo_quant_transformers_only`.

**Harness coverage (2026-04-17):** all five benchmark examples (`parity_check`, `reasoning_check`, `quality_check`, `multi_turn_16k`, `multi_turn_32k`) now run via the `aeo-harness` daemon and share the streaming event protocol. Model load happens once at daemon start; each example dispatches a named workload and reconstructs terminal UX from `STATUS_EVENT` frames. No in-process model load paths remain.
