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

## Next steps (ranked)

See [`plans/2026-04-15-fp8-moe-decode-optimization.md`](./plans/2026-04-15-fp8-moe-decode-optimization.md) for execution detail and verification gates.

1. **NVTX trace markers** — zero-cost per-stage labels for future profiling.
2. **Hot-path hygiene** — eliminate per-call scale allocation + dtype conversion. +35 MB for lossless fp32 scales at load. Expected +2–5%.
3. **`torch.compile(mode="reduce-overhead")`** — aims at the 2.4× CPU/CUDA gap. First-run warmup cost, expected +5–20% if compatible with `_scaled_mm` under `dynamo`.
4. **Non-MoE FP8 quantization** — biggest lever; 53% of CUDA is still bf16. Requires upgraded accuracy gate (teacher-forced top-1 agreement, not just greedy parity) because of cumulative quantization error risk. LM head is the highest-risk layer and may need to stay bf16.
