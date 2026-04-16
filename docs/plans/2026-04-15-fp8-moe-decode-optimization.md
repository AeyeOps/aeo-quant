# FP8 MoE Decode Optimization — Ultra Plan

**Date:** 2026-04-15
**Model:** `google/gemma-4-26B-A4B-it` (FP8 checkpoint)
**Platform:** NVIDIA GB10 (Blackwell sm_121, ARM64, 128 GB unified LPDDR5x)

## Context

A preceding commit (`perf: switch gemma4 fp8 experts to torch._scaled_mm`) lifted decode throughput from 7.82 → 9.94 tok/s (+27%) and prefill from 639 → 108 ms (-83%) by replacing the per-call `FP8→BF16` upcast + scale multiply + bf16 linear with `torch._scaled_mm`. The MoE expert path now consumes only 17% of CUDA time.

A follow-up 30-token profiler run (`PROFILE_TRACE=1 uv run examples/profile_generate.py`) reveals the bottleneck has shifted:

- **CPU 3.66 s vs CUDA 1.54 s (2.4× ratio).** The GPU is starved; PyTorch kernel launch overhead dominates. This is the signature of launch-bound execution.
- **Non-MoE bf16 matmuls are 53% of CUDA time** (attention q/k/v/o, non-MoE MLP, LM head). The `[2816, 262144]` LM-head projection alone is 17%.
- **`_scaled_mm` (MoE experts)** dropped to 17% — no longer the bottleneck.
- **`aten::item` / `aten::nonzero`** in the Python expert loop are now 0.8% / 3.4% CPU. Routing batching is NOT worth pursuing: both the worktree author's implementation (`moe_pack.py`, `moe_decode.py` on `feat/cpu-bottleneck`) and an independent in-session attempt ran slower than eager on GB10. The `.item()` cost was latent-overlapped with GPU compute in the bf16 era, and is proportionally even smaller now.

## Goal

Execute four incremental optimizations, each with a verification gate and an independent git commit. The +35 MB cost of pre-converting FP8 scales from bf16 to fp32 is explicitly approved.

**Ordering rationale:** cheap/diagnostic first (Steps 1–2) to build confidence and unlock better measurement; bigger engineering after (Steps 3–4). Step 4 is where the real remaining headroom lives but carries accuracy risk, so it's last.

## Files touched

| File | Purpose | Steps |
|---|---|---|
| `src/aeo_quant/bridges/gemma4/modeling.py` | NVTX markers, scale cache | 1, 2 |
| `src/aeo_quant/bridges/gemma4/loader.py` | Load-time scale conversion, non-MoE patching, torch.compile wrapper | 2, 3b, 4c |
| `src/aeo_quant/gpu/quant.py` | Add `quantize_2d_to_fp8` | 4a |
| `src/aeo_quant/bridges/gemma4/linear_fp8.py` *(new)* | FP8 drop-in `nn.Linear` | 4b |
| `examples/parity_check.py` *(new, Step 0)* | Regression canary for every step | Step 0 |
| `examples/cuda_graph_probe.py` *(new)* | `_scaled_mm` + `cuda.graph()` feasibility test | 3a |
| `docs/gemma4-fp8-optimization.md` | Running results narrative | all |

## Verification contract (every step)

| Artifact | Command | Pass |
|---|---|---|
| Parity | `uv run examples/parity_check.py` | Token divergence within step-specific threshold |
| Performance | `uv run examples/profile_generate.py` | Decode tok/s meets step-specific target |
| Lint | `uv run ruff check src/ examples/` | Clean |

Failure of any → rollback the increment, investigate, do not advance.

## On quality vs parity (important context for Step 4)

The parity check counts token-level mismatches between a baseline decoded output and a post-change decoded output (50 tokens, greedy, fixed prompt, fixed seed). This is a **canary for silent regressions**, not a quality measure.

For Steps 1–3 the test is meaningful because those changes are mathematically identical (lossless) or deterministic transformations. Expected divergence: zero.

For Step 4 (quantizing non-MoE weights to FP8), the test is insufficient on its own. FP8 introduces real quantization error. Even modest error at any layer can flip an `argmax` tie and cascade into entirely different but still coherent output — a fact about greedy decoding with a long sequence, not a sign of model failure. Before executing Step 4 we will upgrade the check with at least one of:

- Teacher-forced top-1 agreement over a longer passage (measures per-position error without cascade)
- Logit KL divergence at each position
- Coherence read of the generated text

The LM head (`lm_head`, `[2816, 262144]`) is the single highest-risk layer to FP8-ify: its output is compared directly by `argmax`. Step 4's decision tree supports excluding it while shipping the rest.

---

## Step 0 — Precursor commits + parity harness *(DONE at time of writing — included for completeness)*

1. ✅ Commit the `_scaled_mm` swap + `profile_generate.py` cleanups as `perf: switch gemma4 fp8 experts to torch._scaled_mm`.
2. Create plan doc + narrative + `examples/parity_check.py`.
3. Run `parity_check.py` once to produce `results/parity/baseline.txt`.
4. Commit: `docs: add fp8 moe decode optimization plan and parity harness`.

**Exit:** baseline.txt exists; two commits on main.

---

## Step 1 — NVTX trace markers (diagnostic freebie)

**What.** Cherry-pick `moe_trace_range` from `.worktrees/cpu-bottleneck/src/aeo_quant/bridges/gemma4/moe_config.py:21-30`. Drop the env-var mode switching — keep only the trace wrapper. Inline into `modeling.py` (~12 lines).

**Call sites** inside `Gemma4TextExpertsFP8.forward`:
- `fp8_moe_route` — the `one_hot` / `nonzero` preamble.
- `fp8_moe_gate_up` — the `_fp8_linear` gate-up call.
- `fp8_moe_down` — the `_fp8_linear` down call.
- `fp8_moe_combine` — the `index_add_` accumulation.

**Gate:** env var `AEO_MOE_TRACE=1`. Zero cost when off.

**Verification:**
1. `parity_check.py` byte-for-byte match (no env var).
2. `profile_generate.py` decode tok/s within ±1% of pre-Step-1 (no regression from idle wrapper).
3. With `AEO_MOE_TRACE=1 PROFILE_TRACE=1`, profiler table includes the four named ranges.

**Decision:**
- Parity mismatch → rollback.
- ≤1% when off → continue.
- >1% when off → investigate.

**Commit:** `obs: add opt-in nvtx trace markers to gemma4 fp8 forward`

---

## Step 2 — Hot-path hygiene (includes +35 MB fp32 scales)

Kill per-call allocations and dtype conversions in `_fp8_linear` and `forward`.

**Changes:**

1. **Pre-convert scales at load** (`loader.py`):
   - `gate_up_proj_scale: (E, 2I, 1) bf16` → `(E, 1, 2I) float32`
   - `down_proj_scale: (E, H, 1) bf16` → `(E, 1, H) float32`
   - Shape is already RowWise-ready after conversion. `bf16 → fp32` is lossless.
   - Memory cost: +34.6 MB. Approved.

2. **Cache a module-level `scale_a_1x1`** for the M=1 decode path, registered as a non-persistent buffer in `__init__`. `_fp8_linear` uses it when `x_bf16.shape[0] == 1`.

3. **Drop the per-call reshape**: `self.gate_up_proj_scale[expert_idx]` is already `(1, 2I) fp32` — no `squeeze(-1).unsqueeze(0)` needed.

4. **Drop the per-call `.float().contiguous()`**: scales already fp32 + contiguous.

**Verification:**
1. Parity byte-for-byte match (pre-conversion is lossless, matmul unchanged).
2. Profile:
   - Decode tok/s +2–5% over 9.94 target.
   - `torch_alloc` reports +~35 MB.
   - `aten::ones` calls in MoE hot path drop from ~14 k / 30 tokens to ≪100.
   - `aten::_to_copy` / `aten::to` for the scale-conversion paths gone from MoE ops.

**Decision:**
- Parity mismatch → rollback, investigate shape/dtype contract.
- ≥+1% tok/s → continue.
- Neutral (±1%) → continue (cleaner code, reduced kernel count).
- Regression → rollback.

**Commit:** `perf: cache scale_a, pre-convert gemma4 fp8 scales to fp32 at load`

---

## Step 3 — CUDA graph capture (biggest uncertain upside)

The CPU:CUDA 2.4× ratio is the signature of launch-bound execution. Graph replay collapses per-kernel launch overhead. **No precedent in this repo.**

### 3a — Feasibility probe

`examples/cuda_graph_probe.py` (<80 lines): load model, capture a single-token decode under `torch.cuda.graph()`, replay, diff vs eager.

**Question answered:** does `torch._scaled_mm` on sm_121 capture/replay correctly?

**Decision:**
- Probe passes → proceed to 3b.
- Probe fails → document, skip Step 3, jump to Step 4.

**Commit (on keep):** `spike: cuda graph capture probe for _scaled_mm on blackwell`

### 3b — `torch.compile(mode="reduce-overhead")`

This mode auto-uses CUDA graphs and falls back to eager at graph breaks — exactly the pattern our MoE Python loop needs.

**Scope:** in `load_gemma4_fp8`, after pre-conversion and before return, wrap with `torch.compile(model, mode="reduce-overhead", dynamic=False)`. Gate: optional kwarg `compile_decode: bool = True` (removed in follow-up commit if it stays on).

**Verification:**
1. Parity byte-for-byte (`torch.compile` must not change semantics).
2. Decode tok/s ≥ +5% over Step 2 result.
3. CPU/CUDA ratio trending toward 1:1.
4. Stable on 100-token generation.
5. First-run warmup time recorded (acceptable up to ~120 s).

**Decision:**
- Crashes or corrupts → remove wrapper, document failure mode, proceed to Step 4.
- <5% speedup → remove wrapper, document, proceed to Step 4.
- ≥5% speedup → keep; remove the gating kwarg; make default.

**Commit:** `perf: enable torch.compile reduce-overhead on gemma4 decode path`

### 3c — Manual graph capture *(fallback only)*

Only pursued if 3a passed but 3b failed for a specific known `_scaled_mm`+`dynamo` reason. Scope: capture non-MoE segments only; deferred mini-plan.

---

## Step 4 — Quantize non-MoE weights to FP8 (biggest lever, multi-commit, accuracy-gated)

53% of CUDA time is in bf16 matmuls. Quantizing attention + non-MoE MLP + LM head to FP8 could add +30–50%. Memory **decreases**.

### Accuracy gate upgrade (before starting)

Add a teacher-forced top-1 agreement probe to `parity_check.py`:
- Feed a ~500-token passage through both bf16-reference and the FP8-swapped model.
- Compare `argmax` at each position; report `%agree` and mean absolute logit distance on disagreements.
- Also keep the greedy-generation canary as a secondary check.

### 4a — Add `quantize_2d_to_fp8`

Mirror `src/aeo_quant/gpu/quant.py:quantize_3d_to_fp8`:

```python
def quantize_2d_to_fp8(weight_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # weight_bf16: (out, in) bf16
    # returns (weight_fp8: (out, in) fp8_e4m3fn, scale_bf16: (out, 1) bf16)
    # per-output-row max-abs
```

Add `tests/test_quant.py` with a round-trip error budget.

**Commit:** `feat: add quantize_2d_to_fp8 to gpu.quant`

### 4b — `LinearFP8` module

`src/aeo_quant/bridges/gemma4/linear_fp8.py` — drop-in `nn.Linear` replacement.
- Stores `weight: (out, in) fp8_e4m3fn` + `weight_scale: (1, out) fp32` (RowWise-ready).
- Forward uses a shared `_fp8_linear` helper (extract the one from `modeling.py` to module level).
- Handles bias (bf16 add after `_scaled_mm`).

Unit test against `nn.Linear` output within tolerance.

**Commit:** `feat: LinearFP8 drop-in replacement using torch._scaled_mm`

### 4c — Optional non-MoE swap in loader

`load_gemma4_fp8(fp8_non_moe: bool = False)`. When True, after load:

- Walk the model; find linears under:
  - `*.self_attn.{q,k,v,o}_proj`
  - `*.mlp.{gate,up,down}_proj` (non-MoE MLP)
  - `lm_head`
- For each, `quantize_2d_to_fp8(weight)` then install `LinearFP8`.

Default stays OFF.

**Verification (accuracy-led):**
1. Teacher-forced top-1 agreement ≥ 95% on the 500-token passage.
2. Greedy generation still coherent (subjective read, or LLM-judge if automated).
3. Decode tok/s ≥ +15% over Step 3 result.

**Decision tree:**
- All three pass → proceed to 4d.
- Speedup good but top-1 agreement < 95% → per-group probe: swap only attention; only non-MoE MLP; only LM head. Keep groups that pass; exclude those that don't.
- Speedup < +15% → investigate graph breaks (from Step 3 compile interaction) before rolling back.

**Commit:** `feat: optional fp8 quantization of gemma4 non-moe linears at load`

### 4d — Flip default on

Based on 4c evidence:
- All groups pass → `fp8_non_moe=True` default, remove the flag.
- Partial pass → ship with per-group defaults (e.g., `fp8_non_moe={"attn": True, "mlp": True, "lm_head": False}`) if LM head fails.

**Verification:** parity + profile re-run; retained speedup confirmed.

**Commit:** `perf: enable fp8 quantization for gemma4 non-moe linears by default`

---

## Memory updates *(post-Step-4; outside repo)*

Under `~/.claude/projects/-opt-dev-aeo-aeo-quant/memory/`:

1. `feedback_cpu_cuda_ratio_launch_bound.md` — CPU > 1.5–2× CUDA is the launch-bound signature; optimize kernel count, not compute.
2. `project_scaled_mm_rowwise_gotchas.md` — scale_a `(M,1)`, scale_b `(1,N)`, both fp32 contiguous; FP8 `.t()` is free column-major view.
3. `feedback_routing_batching_dead_end.md` — routing batching does NOT beat eager on GB10 Gemma 4 MoE; proven twice.
4. Update `MEMORY.md` index.

---

## Rejected / do not pursue *(record why)*

- **Routing batching** (bmm decode, device-packed sort, grouped-GEMM) — proven dead on GB10 twice.
- **Custom fused FP8 expert kernel** — subsumed by `torch._scaled_mm`.
- **Attention kernel swap** — `_sdpa` already default; manual override doesn't help.
- **Prefill 16 K wall** — in upstream `transformers`, not our patch.
