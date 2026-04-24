# 3D fused-experts NVFP4 kernel — design

**Status:** design complete, pre-implementation. Fail-fast phase order below.

**Date:** 2026-04-17.

## North star (user-directed)

**Objective: 5× the current 6.77 tok/s native NVFP4 — target 52 tok/s (community ceiling on DGX Spark).**

10 tok/s FP8 parity is not the win; it's a proof point that improvement is possible.
The win is 5×. `transformers.generate()` is the fixed substrate — we're not pivoting
to vLLM/llama.cpp.

**Realistic ceiling honesty:** the 52 tok/s community numbers come from llama.cpp / vLLM
stacks with PagedAttention, Flash Attention, and CUDA-native serving loops. Inside
`transformers.generate()` we don't have those. Compounded gains projected:

| Lever | Addresses | Gain | Landing |
|---|---|---|---|
| Graph capture alone | CPU launch overhead only | CPU 234ms→~5ms; bounded by 78ms CUDA | ~12 tok/s |
| 3D fused-experts (this plan) | Launch count + SM overlap | ~1.5–2× | ~14–18 tok/s |
| Both combined | Both, compounding | ~3× | ~20–30 tok/s |
| Community ceiling | +paged KV, FA2, native loop | — | 45–60 tok/s |

**Expected landing: 20–30 tok/s.** Reaching 52 tok/s inside transformers is a stretch
that depends on additional levers (flash attention via `attn_implementation="flash_attention_2"`,
TMA descriptors, autotune per shape). Reassess at each measurement — "is the next
action the most optimal path to 52 tok/s?" Don't sunk-cost on diminishing returns.

## Starting state (entering implementation)

- **Branch:** `main`. Working tree clean (to be verified at session start).
- **Latest commits:** `aa6c042` (tunable MIN_FREE_GB), `72fdffd` (pre-quantize dedup).
- **Today's baseline:** 6.77 tok/s via `parity_check.py` + native harness.
- **Harness state:** may or may not still be running at session start.
  Check with `uv run python -m aeo_quant.harness.cli status`. If running on nvfp4
  with uptime > 0, reuse it (saves ~100 s model load). If not running, start with:
  ```bash
  TRITON_OVERRIDE_ARCH=sm120 QUANT_FORMAT=nvfp4 HARNESS_MIN_FREE_GB=20 \
    nohup uv run python -m aeo_quant.harness.cli start --format nvfp4 \
      > /tmp/aeo-harness.log 2>&1 &
  ```
- **`.env` reverted** to the original 3 lines (HF_TOKEN + 2 checkpoints).
  Runtime env vars live on the command line, not in .env. **Do not** put
  QUANT_FORMAT / TRITON_OVERRIDE_ARCH in .env — the override behavior masks
  shell env and surprised us earlier.
- **Fresh nvfp4 parity baseline** at `tests/fixtures/parity_baseline_nvfp4.txt`
  (established today, post-dedup). Gitignored — don't expect it in git. If the
  file is missing, `parity_check` auto-regenerates on first run.
- **Current parity delta vs FP8:** 29-token prefix match, 42% overall divergence.
  That's the expected FP4-vs-FP8 numerical drift, not a regression signal.

## Why this lever

Profiler: 6.77 tok/s native, 234 ms/token CPU vs 78 ms/token CUDA. Graph capture alone
caps at `1/78ms ≈ 13 tok/s` because it only removes CPU overhead. To reach 52 tok/s
(19 ms/token budget) we must **also** cut CUDA time. The 3D fused-experts kernel is the
only single-change lever that addresses both:

1. **Launch count.** Per-layer launches in MoE: 8 (4 experts × 2 projections) → 2.
   Per-decode-token MoE launches: 240 → 60.
2. **Tensor-core utilization at decode.** Current M=1 with BLOCK_M=16 runs at 1/16 lane
   occupancy per call. Fusing 4 experts into a (E=4, M=1) batched launch doesn't raise
   per-MMA occupancy, but lets the kernel dispatch all 44 tile blocks (11 N-tiles × 4
   experts) concurrently across SMs instead of 4 sequential waves of 11. Better SM
   overlap, smaller per-tile launch cost amortization.

Projected gain: 1.5–2× compounding with subsequent graph capture (additive). Realistic
landing for decode after both: 20–30 tok/s. Gap to 52 tok/s still unknown; reassess at
measurement.

## Scope (first version)

- **Decode path (M=1) only** for the 3D kernel. Prefill keeps the existing per-expert
  Python loop with `nvfp4_linear` / `nvfp4_linear_prequantized`. The MoE forward
  dispatches on `M` at entry.
- **Weights gathered per-call** via `torch.index_select(weights, 0, top_k_expert_ids)`.
  Graph-capturable, avoids the 32× waste of processing all 128 experts.
- **Output fusion not included** — kernel returns `(E, M, N)` bf16; the Python caller
  applies `top_k_weights` and does the combine (`index_add_` equivalent). Can be folded
  into the kernel in a later pass if measurement warrants.

## Checkpoint layout (verified on disk)

```
gate_up_proj:        (num_experts=128, 2*intermediate=1408, hidden/2=1408) uint8
gate_up_proj_scale:  (128, 1408, hidden/16=176)                          float8_e4m3fn
gate_up_proj_scale_2: scalar                                              bfloat16

down_proj:           (128, hidden=2816, intermediate/2=352)               uint8
down_proj_scale:     (128, 2816, intermediate/16=44)                      float8_e4m3fn
down_proj_scale_2:   scalar                                               bfloat16
```

**Key finding:** `*_scale_2` is a single scalar per projection shared across all 128
experts. **One alpha covers all fused experts** — kernel doesn't need a per-expert
alpha tensor.

## Kernel signature

```python
@triton.jit
def _nvfp4_matmul_kernel_3d(
    a_ptr, b_ptr,               # a: (M, K/2) uint8, b: (E, N, K/2) uint8
    a_scale_ptr, b_scale_ptr,   # a_scale: (M, K/16) fp8, b_scale: (E, N, K/16) fp8
    c_ptr,                      # c: (E, M, N) bf16
    alpha,                      # fp32 scalar — folded a_tensor * b_tensor
    M, N, K,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_am_scale, stride_ak_scale,
    stride_be_scale, stride_bn_scale, stride_bk_scale,
    stride_ce, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    ELEM_PER_BYTE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
```

- Activation `a` is **shared across experts** — loaded once per `(pid_m, pid_n)` tile
  regardless of `pid_e`. L1/shared memory caches hit across expert dim.
- Weights `b` index by `pid_e * stride_be + ...` — per-expert slice.
- Output `c` written per-expert with `pid_e * stride_ce + ...`.
- Alpha passed as runtime scalar (same as current kernel). The `.item()` issue from
  Path E still exists at the Python wrapper level — orthogonal to this kernel, fixed
  later.

## Grid

2D launch:

```python
grid = (
    triton.cdiv(M_padded, BLOCK_M) * triton.cdiv(N, BLOCK_N),  # axis 0: tile of C
    E,                                                           # axis 1: expert
)
pid_mn = tl.program_id(axis=0)
pid_e  = tl.program_id(axis=1)
```

For decode (M=1, BLOCK_M=16, N=1408, BLOCK_N=128, E=4): grid = (11, 4) = 44 blocks, one
launch. Compare to current 4 separate launches of 11 blocks each.

## Kernel body

Structurally identical to the existing `_nvfp4_matmul_kernel`:
- Decode pid_mn → pid_m, pid_n (same as today)
- Extra offset on all b pointers: `pid_e * stride_be` (weights), `pid_e * stride_be_scale` (b_scale)
- Extra offset on c pointer: `pid_e * stride_ce`
- Activation load is unchanged (no pid_e dependency)
- Nibble swap, `tl.dot_scaled`, accumulation, alpha fold, bf16 store — all same as current

No new Triton primitives. This is a structural refactor of an existing working kernel.

## Python wrapper

```python
def nvfp4_linear_3d_prequantized(
    a_packed: torch.Tensor,       # (M, K/2) uint8
    a_block_scale: torch.Tensor,  # (M, K/16) fp8_e4m3fn
    a_tensor_scale: torch.Tensor, # scalar fp32
    w_packed: torch.Tensor,       # (E, N, K/2) uint8
    w_block_scale: torch.Tensor,  # (E, N, K/16) fp8_e4m3fn
    w_tensor_scale: torch.Tensor, # scalar fp32 (shared across experts)
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Batched per-expert NVFP4 matmul, E experts sharing one activation.

    Returns (E, M, N) of out_dtype.
    """
```

- Same M-padding path as `nvfp4_linear_prequantized` for M < BLOCK_M.
- Same divisibility checks on N and K.
- Block/num_stages tuning: reuse current adaptive heuristics (small-M lowers BLOCK_M).

The existing `nvfp4_linear` and `nvfp4_linear_prequantized` stay for prefill and tests.

## MoE forward changes

In `Gemma4TextExpertsNVFP4.forward(self, hidden_states, top_k_index, top_k_weights)`:

**Decode branch (M=1):**

```python
# Pre-quant activation (existing dedup optimization preserved)
a_packed, a_block_scale, a_tensor_scale = quantize_2d_to_nvfp4(hidden_states)

# Gather top_k selected experts' weights + scales
expert_ids = top_k_index[0]  # shape (top_k,)

# Gate + up projection (fused)
w_gu_packed     = self.gate_up_proj.index_select(0, expert_ids)        # (k, 2*im, hd/2)
w_gu_block_sc   = self.gate_up_proj_scale.index_select(0, expert_ids)  # (k, 2*im, hd/16)
gate_up = nvfp4_linear_3d_prequantized(
    a_packed, a_block_scale, a_tensor_scale,
    w_gu_packed, w_gu_block_sc, self.gate_up_proj_scale_2,
)  # (k, 1, 2*im)
gate, up = gate_up.chunk(2, dim=-1)
current = self.act_fn(gate) * up  # (k, 1, im)

# Down projection (fused) — per-expert activation, separate activation quant
# current reshape: (k, 1, im) → (k, im) effectively (k, M, im) with M=1
# Need per-expert activation quant here (each expert's gate*up is different)
# Option A: loop of k activation-quants + 3D down kernel
# Option B: per-expert down_proj with existing prequantized kernel
# Choice: Option A — keep one launch per down projection
a_packed_d, a_block_sc_d, a_tensor_sc_d = quantize_2d_to_nvfp4(current.squeeze(1))
# ^ quantize_2d requires 2D input (k, im)

# But a_tensor_scale_d is a SINGLE scalar for the whole (k, im) tensor.
# That's correct math when we're treating the k experts' outputs as one batch,
# but it changes the per-expert quantization relative to the current code
# (which quantizes each expert's activation independently).
# Need to verify parity — see OPEN QUESTION #1 below.

w_d_packed    = self.down_proj.index_select(0, expert_ids)
w_d_block_sc  = self.down_proj_scale.index_select(0, expert_ids)
down_out = nvfp4_linear_3d_prequantized(
    a_packed_d, a_block_sc_d, a_tensor_sc_d,
    w_d_packed, w_d_block_sc, self.down_proj_scale_2,
)  # (k, 1, hd)

# Combine: apply top_k_weights and sum
weights = top_k_weights[0, :, None, None]  # (k, 1, 1)
return (down_out * weights).sum(dim=0)     # (1, hd)
```

**Prefill branch (M>1):** dispatch to existing per-expert loop (unchanged).

Code change footprint in `modeling_nvfp4.py`: ~50 lines replaced, existing loop preserved for prefill.

## Open questions (resolve before implementation)

### 1. Does shared `a_tensor_scale` across the k-batch match the current per-expert quantization?

Current code quantizes each expert's down-proj activation independently, so each
expert has its own `a_tensor_scale` computed from its own slice's `amax`. The 3D
path computes one `a_tensor_scale` over the stacked `(k, im)` tensor — the global
amax. Values are the same when one expert's amax equals the batch amax (it usually
does for one outlier expert); otherwise slightly different numerics.

**Risk:** could cause small parity drift vs today's native baseline.

**Mitigation:** keep the dedup optimization but quantize **per expert** along the
k-dim in the 3D wrapper. Pass a `(k,)` alpha tensor instead of a scalar, and have
the kernel load per-pid_e. Adds minor complexity; worth it for parity.

**Decision:** measure first with scalar-alpha implementation and see if parity holds.
If it drifts, fall back to per-expert alpha.

### 2. `torch.index_select` on (128, N, K/2) uint8 — cost and graph-capturability?

`index_select` is a standard aten op. Gathering 4 rows from (128, 1408, 1408) uint8:
~2.4 MB × 2 (gate_up + down) = ~5 MB of copies per layer. At 30 layers, 150 MB of
memory traffic per decode token. At LPDDR5X bandwidth (~273 GB/s), that's ~0.5 ms.

Compared to 234 ms total per-token CPU overhead, this is negligible. Graph-captured
without issue (fixed shape, fixed dtypes).

### 3. Alpha `.item()` at the wrapper level

The current wrapper does `alpha = (a_tensor_scale * w_tensor_scale).item()`. Forces
CPU sync. For decode parity, this is called **twice per layer** (gate_up + down)
instead of 8× (per expert × per projection) — 4× fewer syncs already. Good
improvement from fusion alone. A proper fix (pass alpha as 0-d tensor, load inside
kernel) is a Path-E-style follow-up, not this lever.

### 4. Output layout: `(E, M, N)` vs `(M, E, N)`

Chose `(E, M, N)` — contiguous per-expert slices match the natural pid_e iteration
and are cache-friendly for the write path. The subsequent combine (`* weights`, then
`sum(dim=0)`) handles the E-dim-outermost layout cleanly.

## Correctness test plan

### Synthetic (no real weights, no GPU model load)

`examples/test_nvfp4_3d_kernel.py` — new file:

1. Create random bf16 weight tensors shape `(k, N, K)`, quantize each per-expert via
   existing `quantize_3d_to_nvfp4`.
2. Create random bf16 activation `(M, K)`, quantize.
3. Call `nvfp4_linear_3d_prequantized` → expected (k, M, N).
4. Reference: loop over k, call existing `nvfp4_linear_prequantized` per expert,
   stack outputs along axis 0.
5. Assert max rel error < 1e-3 (same gate as existing per-expert test).

Shapes tested: decode-critical (M=1, k=4, N=1408, K=2816) and (M=1, k=4, N=2816, K=704).

### Bridge test with real weights

Extend `examples/test_nvfp4_bridge.py` to load one MoE layer's buffers, select a
random 4-expert subset, run both the 3D path and the per-expert reference, assert
agreement.

### End-to-end parity

Via existing nvfp4 harness (reuses the 18 GB already loaded):

```bash
PARITY_MIN_FREE_GB=20 QUANT_FORMAT=nvfp4 uv run python examples/parity_check.py
```

- Must match `tests/fixtures/parity_baseline_nvfp4.txt` within 5% token divergence
  (per existing gate).
- Record tok/s — that's our measurement number.

## Fail-fast phase order (risk-first)

Each phase has a single binary gate. Fail → stop and diagnose. Pass → commit and
continue. No phase blocks on code from later phases. The riskiest foundational
assumption (does `tl.dot_scaled` work with the 3D pointer pattern on sm_121?)
is validated first, for free (no GPU).

| Phase | Action | Cost | Template | Kill gate |
|---|---|---|---|---|
| **0a** | AOT compile probe. Minimal 3D kernel (2 experts, K=128, N=128) with pid_e pointer offset. Compile via `triton.compile(ASTSource, target=GPUTarget('cuda', 120, 32))`. Inspect compiled PTX for `.kind::mxf4nvf4`. | No GPU. ~10 min. | `examples/probe_nvfp4_aot.py` | PTX missing native MMA → whole plan dies; don't write the real kernel |
| **0b** | `index_select` dtype probe. 5 lines — gather 4 rows from a (128, 1408, 1408) uint8 tensor and a (128, 1408, 176) fp8_e4m3fn tensor on GPU. | 2 s GPU. | none — one-off | fp8 gather bugs on Blackwell → switch to advanced indexing or manual cat |
| **1** | Write `_nvfp4_matmul_kernel_3d` + `nvfp4_linear_3d_prequantized`. Write `examples/test_nvfp4_3d_kernel.py` synthetic test at TINY shapes (2 experts, K=128, N=128). | ~hundreds of MB, seconds of compute. | `examples/test_nvfp4_kernel.py` (2D reference) | rel err > 1e-3 → kernel body bug (stride, nibble swap, grid) |
| **2** | Scale synthetic test to Gemma decode shapes: k=4, M=1, K=2816, N=1408 (gate_up) and K=704, N=2816 (down). | Seconds. | same | stride/shape-dependent bug, or tile geometry issue |
| **3** | Extend `examples/test_nvfp4_bridge.py` with 3D-path assertion on one real MoE layer's buffers. | ~500 MB. | existing file | real-checkpoint edge cases (mixed-dtype gather, shape surprises) |
| **4** | Numerics A/B: scalar `a_tensor_scale` (batch amax) vs per-expert `a_tensor_scale` (k-sized tensor). Measure parity drift on real weights for both. Data-driven resolve of Open Question #1. | Minutes. | synthetic + bridge scaffolding | if scalar-alpha drifts > 5%, ship per-expert-alpha variant |
| **5** | Modify `Gemma4TextExpertsNVFP4.forward` to dispatch M=1 → 3D kernel; keep M>1 loop unchanged. | Code-only. | `src/aeo_quant/bridges/gemma4/modeling_nvfp4.py` | code review + next phase's end-to-end parity |
| **6** | Parity_check + tok/s via existing harness (reuse loaded 18 GB). | Reuses harness. | — | > 5% parity drift vs today's nvfp4 baseline → revert and diagnose |
| **7** | Measurement reflection. Did tok/s hit 14–18 tok/s projection? If yes, commit and move to additive levers. If significantly short, investigate before adding layers. If significantly over, update the ceiling model. | — | — | sunk-cost guard: don't start additive work until 3D kernel gain is verified |

**Additive levers (only after phase 7 passes):**
- Path E `.item()` → on-device alpha (wrapper-level, not kernel-level)
- Manual CUDA graph capture (requires verifying `examples/cuda_graph_probe.py` PASSes on FP8 first — never been run to PASS per prior ultrareview)
- TMA scale descriptors
- `triton.autotune` per shape
- Flash attention via `attn_implementation="flash_attention_2"` on `from_pretrained`

**Commit per phase.** Rollback = `git revert`. No big-bang commits.

## Systems guardrails

User has pre-approved GPU time for correctness testing, subject to systems-check discipline.

- **Before every phase that touches GPU**: check unified-memory headroom (`free -g`),
  GPU allocations (`nvidia-smi --query-compute-apps=pid,used_memory`), other user
  processes. Note that `nvidia-smi` returns `[N/A]` for GB10 unified memory fields —
  use process-level `used_memory` instead.
- **Shared resources to respect**: vLLM (`VLLM::EngineCore`) has been running at
  ~52 GB GPU; a qemu VM at ~6.5 GB RSS; multiple claude/codex sessions at ~1 GB each;
  core 19 may be pinned 100% by another agent. Don't disturb.
- **Reuse the loaded harness** when possible instead of spawning a second full-model
  load — it's 18 GB of GPU that's already there.
- **Stop between phases** and reassess systems state before proceeding. If unified
  memory pressure rises or swap grows, pause and report before the next phase.
- **Ask** before any action that would add > 5 GB unified memory beyond what's
  already allocated.

## Risks I'm not controlling for

- **`tl.dot_scaled` on sm_121 under the `TRITON_OVERRIDE_ARCH=sm120` path** has only
  been validated for 2D `a` × 2D `b` so far. The 3D kernel structurally is still a 2D
  dot per program-id (per `(pid_m, pid_n, pid_e)` tuple), so this should work, but it
  hasn't been tested. First-risk check is the AOT synthetic correctness test.

- **SM occupancy at decode**: my claim that 44 concurrent tile blocks on 20 SMs gives
  better overlap than 4× waves of 11 assumes Triton's scheduler issues blocks greedily.
  If there's smem/regs pressure at the larger grid, scheduling may serialize anyway.
  Measurement-dependent.

- **`index_select` on mixed dtypes** (uint8, fp8_e4m3fn) — both are supported, but on
  Blackwell there have been fp8 dtype edge cases. Synthetic test catches this early.

## Commit plan

One commit per phase in the fail-fast table. Each gated by its own binary check.
Revert path clean — per-expert kernel and existing forward are preserved; the 3D
path is additive and opt-in until phase 5. `docs/plans/2026-04-17-nvfp4-3d-fused-experts.md`
is the durable plan — keep it updated with phase outcomes.

## Pointers

**Working-style ground truth** lives in:
- `~/.claude/CLAUDE.md` (global user directives — verbose bash, ask before
  heavy GPU, systems-check, GPU-only inference, etc.)
- `~/.claude/projects/-opt-dev-aeo-aeo-quant/memory/MEMORY.md` (per-project
  feedback/project memory index)

**Reference implementations already in-tree:**
- `examples/probe_nvfp4_aot.py` — AOT compile + PTX inspection. **Phase 0a template.**
- `examples/probe_nvfp4_minimal.py` — tiny live-GPU 2D kernel correctness.
- `examples/test_nvfp4_kernel.py` — multi-shape synthetic correctness. **Phase 1–2 template.**
- `examples/test_nvfp4_bridge.py` — real-weights bridge test. **Phase 3 template.**
- `examples/smoke_nvfp4_native.py` — end-to-end tok/s smoke.
- `examples/parity_check.py` — harness-driven parity + tok/s. **Phase 6 gate.**
- `examples/cuda_graph_probe.py` — FP8 manual graph capture **reference (UNVERIFIED:
  no evidence this has ever PASSed; confirm on FP8 before building on its pattern).**
- `examples/profile_nvfp4_decode.py` — torch.profiler on native decode.

**Key source files:**
- `src/aeo_quant/gpu/nvfp4_matmul.py` — existing 2D kernel + wrappers. Extend here.
- `src/aeo_quant/gpu/quant.py` — `quantize_2d_to_nvfp4`, `quantize_3d_to_nvfp4`. Note
  `tensor_scale` is global `.amax()` (quant.py:146). For dedup dive into decode
  correctness, see commit `72fdffd`.
- `src/aeo_quant/bridges/gemma4/modeling_nvfp4.py` — `Gemma4TextExpertsNVFP4.forward`.
  Phase 5 rewrite target.
- `src/aeo_quant/bridges/gemma4/loader.py` — `load_gemma4_nvfp4`, `torch.compile`
  wrap. Don't touch during phases 0–6.
- `src/aeo_quant/harness/server.py` — harness; `HARNESS_MIN_FREE_GB` env override.
- `src/aeo_quant/workloads/parity.py` — uses `Gemma4HybridTurboQuantCache` (TurboQuant
  KV); our tok/s measurements already include TurboQuant overhead.

**Background context:**
- `kb/nvfp4-blackwell-research.md` — why `TRITON_OVERRIDE_ARCH=sm120` is required on
  sm_121; SASS validation signature `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`.
- `docs/2026-04-17-nvfp4-sm121-breakthrough.md` — today's breakthrough writeup.
- `docs/turboquant-gemma4-research.md` — TurboQuant is a KV cache; orthogonal to
  expert forward.

## Phase outcomes (2026-04-17)

All seven fail-fast phases passed. Commit SHAs in parentheses.

| Phase | Outcome | Evidence |
|---|---|---|
| 0a | PASS (bcf626c) | sm_120 emits 64 `mma.sync.kind::mxf4nvf4` ops for the 3D kernel. sm_121 hard-rejects as expected (handled via `TRITON_OVERRIDE_ARCH=sm120`). |
| 0b | PASS | `index_select` works bit-exactly for both uint8 (0.71 ms for 128→4 rows) and fp8_e4m3fn (0.03 ms) on Blackwell. |
| 1 | PASS (f491418) | 4 tiny synthetic shapes bit-exact (rel_err = 0.0) vs per-expert 2D loop, 1.83-5.91× raw kernel speedup. |
| 2 | PASS (f491418) | Gemma decode shapes (E=4, M=1, gate_up 2816→1408 and down 704→2816) bit-exact, 2.55-4.44× speedup. |
| 3 | PASS (4400fef) | Real-weight bridge test: 4-expert subset via `index_select` from the (128, N, K/2) checkpoint tensors, bit-exact for both gate_up and down. |
| 4 | PASS (405136c) | Scalar `a_tensor_scale` drifts <2% vs per-expert variant across magnitude skews up to 20×. Well under 5% kill gate — ship scalar-alpha. |
| 5 | PASS (9b83ba0) | `Gemma4TextExpertsNVFP4.forward` dispatches M=1 → 3D (two kernel launches per MoE layer), M>1 → unchanged prefill loop. Kernel extended with `stride_ae` so the same kernel handles shared-activation (gate_up) and per-expert-activation (down) layouts. |
| 6 | PASS (20e6505, d6fd8b8) | After fixing a graph-capture regression caused by `os.environ.get` inside `forward` (now resolved at module load): **12.50 tok/s @ 50 tokens, 13.33 tok/s @ 300 tokens** vs 6.77 / 7.07 baselines = **1.80-1.90× speedup**. Output is coherent across 300 tokens (see `parity_check_long.py`). The initial 20% parity divergence was cliff-effect at a 0.125-logit near-tie (`probe_logits_at_divergence.py`); new baselines pinned at 3D path. |
| 7 | REFLECT | Gain ratio (1.80-1.90×) lands at the top of the projected 1.5-2× band — exactly the expected lever size. Absolute tok/s (13.3) is slightly below the projected 14-18 range, tracking the slightly-lower-than-projected 7-tok/s baseline. **Decision: proceed to additive levers.** 13.3 tok/s is 25% of the 52 tok/s north star; the remaining compounding levers are: CUDA graphs (~2×, requires `cuda_graph_probe.py` to PASS on FP8 first), Path-E on-device alpha (removes CPU syncs), flash_attention_2, TMA scale descriptors. |

**Unexpected finding (Phase 6 diagnosis):** `os.environ.get` inside a torch.compile'd
forward kills `reduce-overhead` CUDA graph capture. Dropped tok/s from 6.84 → 1.67
before detection. Fix: resolve env flags at module load into module-level constants.
See commit 20e6505 — added to memory as a general lesson about torch.compile side-effect purity.
