# Native NVFP4 matmul for Gemma 4 26B-A4B on GB10 (sm_121)

**Status:** in progress (2026-04-17). Owner: aeo-quant. Stack: `transformers.generate()` + our bridge — not vLLM/TRT-LLM/llama.cpp.

**Starting point reference:** `kb/nvfp4-blackwell-research.md`, especially the 2026-04-17 re-survey section which corrects several 2026-04-16 assumptions.

**Major corrections to original plan (2026-04-17):**
- `sm_121f` is NOT a real Triton target string. 3.6.0 only emits `sm_121a`. Delta 3 below is obsolete.
- triton#8548 was never fixed. The real blocker is `AccelerateMatmul.cpp`'s `computeCapability != 120 → failure` in the native-FP4 MMAv2 lowering path.
- GB10 has `mma.sync...kind::mxf4nvf4` (NOT `.kind::nvf4` — that name doesn't exist) but NOT `tcgen05.mma.*`. Any tcgen05-path kernel is DOA — we target warp-level MMA only.
- Canonical NVFP4 MMA shape is `m16n8k64`, not `m16n8k32`. PTX ISA 8.8 (CUDA 13+) required.
- torchao's `_addmm_nvfp4_dispatch` routes to `_scaled_mm`, which routes to cuBLAS, which lacks sm_121 FP4 support. The probe is still worth running, but expected-to-fail.
- TMA on sm_121 in Triton 3.6.0 **does** work (PR #8498 shipped). The TMA load path is fine; only MMA lowering is broken.
- **Triton's `tl.inline_asm_elementwise` CANNOT host `mma.sync`** — it's not warp-collective. Inline PTX kernel strategy is dead at the Triton layer; would need CUDA C++ extension as fallback only.
- **Primary strategy is now Path A.5: `TRITON_OVERRIDE_ARCH=sm120`**. The env var causes Triton to stamp "cuda:120" on the module, which makes `ScaledBlockedToMMA` match on sm_121 hardware without any rebuild. Test this FIRST before any source patch. See re-survey "second deep dive" section.
- Fallback Path A: patch `AccelerateMatmul.cpp:665` guard, rebuild Triton.
- Last resort Path B: CUDA C++ kernel with inline PTX via `torch.utils.cpp_extension`.

---

## Goal

Replace the current NVFP4→FP8 dequant at load with a **native NVFP4 block-scaled matmul** called from `Gemma4TextExpertsNVFP4.forward()`. Keep FP4 in GPU memory. This is the only path in our stack that can close the 10 → ~52 tok/s gap to the community NVFP4 ceiling on the same hardware.

**Not a goal:** changing checkpoint format, switching inference backends, or quantizing non-expert weights.

---

## Baseline to beat

- Current decode: 10.08 tok/s (FP8 and NVFP4 — identical compute path after the NVFP4→FP8 load conversion).
- Peak VRAM: 26.95 GB.
- Parity: 50/50 byte-exact vs FP8 baseline.
- Target: ≥ 25 tok/s decode, ≤ current VRAM, parity coherent (some divergence expected from native FP4 accumulation; gated by `reasoning_check.py`, not byte-exact match).

---

## Pre-work gate: 20-minute torchao probe

Before writing any kernel, rule out the zero-code path.

**What:** a standalone script `examples/probe_nvfp4_torchao.py` that:
1. Loads **one** expert weight from the NVFP4 checkpoint (not the full model).
2. Wraps it in `torchao.prototype.mx_formats.NVFP4Tensor` with our block_size=16 and fp8_e4m3 block scale layout.
3. Calls `torchao`'s `_addmm_nvfp4_dispatch` with random bf16 activations at realistic decode shapes `(1, 2816)` and `(8, 2816)`.
4. Prints: does it run / does it error / output norm vs reference bf16 matmul.

**Decision:** 
- Runs clean with reasonable numerical agreement → skip the kernel write entirely, wire torchao into `Gemma4TextExpertsNVFP4.forward()`. R2 becomes a small integration task.
- Errors, returns zeros, or diverges significantly → confirmed we need our own kernel, proceed to Phase 1.

The probe is read-mostly (loads a single tensor, no model). Preflight memory threshold: 5 GB.

---

## Architecture if we write our own

Fork Triton tutorial 10 (`10-block-scaled-matmul.py`) into our tree; don't route through `matmul_ogs`.

Target file: `src/aeo_quant/gpu/nvfp4_matmul.py` — houses the Triton kernel + a thin Python entry point.

```
aeo_quant/gpu/nvfp4_matmul.py
  _nvfp4_matmul_kernel     # @triton.jit — adapted tutorial 10
  nvfp4_linear(            # Python entry; what the forward calls
      x_bf16: Tensor,          # (M, K)
      w_packed: Tensor,        # (N, K//2) uint8
      w_block_scale: Tensor,   # (N, K//16) float8_e4m3fn
      w_tensor_scale: float,   # fp32 scalar, pre-folded at load
      *, out_dtype=bf16,
  ) -> Tensor                  # (M, N)
```

The MoE bridge calls `nvfp4_linear` per selected expert (same call site as today's `_fp8_linear`). A 3D-experts batched variant is a Phase 3 follow-up; start with the per-expert form to match the current loop shape.

---

## The three deltas from the tutorial

### Delta 1 — per-tensor fp32 scale fold-in

The tutorial expects two inputs: FP4 weights and per-block fp8 scales. Our checkpoint also has a per-tensor fp32 global scale (Level 2). Two options:

**Option A — fold at load (simpler, faster runtime):** multiply the fp32 tensor scale into the fp8 block scales once during `_convert_nvfp4_experts_to_fp8` replacement. Risk: any individual `(block_scale × tensor_scale)` product that overflows fp8_e4m3's 448.0 max corrupts that block. Must add an overflow check at fold time and either fall back to Option B for that expert or clamp with a logged warning. From our current checkpoint stats, `tensor_scale` is typically ~0.001–0.01 and `block_scale` typically ≤ 10, so overflow risk is low but must be verified per-tensor before shipping.

**Option B — fold in epilogue (safer, one fmul per output tile):** keep the fp32 tensor scale as a kernel argument; multiply the accumulator by it after the matmul, before the bf16 down-cast. Cost: one vectorized fmul per `(BLOCK_M, BLOCK_N)` tile. Negligible vs the matmul itself.

**Recommendation:** ship Option B first (no overflow risk, correctness-by-construction). Measure. If the epilogue fmul shows up as a real cost in NVTX, add Option A with overflow gate.

### Delta 2 — bypass the `matmul_ogs` TMA guard

The tutorial code is wrapped in a higher-level dispatcher `matmul_ogs.py` which enforces *"must be persistent kernel and TMA-compliant for native MXFP"*. On GB10 that guard throws before any matmul executes (triton#8548). The fix is on Triton main but not in a released wheel as of 2026-04-16.

**Approach:** inline only the `@triton.jit` kernel from the tutorial. Skip the `matmul_ogs.py` wrapper. Write a small Python launcher that:
- computes grid dims,
- handles `tl.make_tensor_descriptor` (TMA descriptor) creation for A, B, and block_scale,
- configures persistent scheduler ourselves (fixed grid × `tl.num_programs(0)` loop is the simplest correct pattern).

TMA descriptors on sm_121 need testing — known working on sm_100. If TMA itself hits a path that needs the fix, fall back to `tl.load` with `block_ptr` (non-TMA). Non-TMA on Blackwell is ~15–25% slower but still beats FP8; it's an acceptable fallback if TMA blocks us.

### Delta 3 — compile target on sm_121 (REVISED 2026-04-17)

**Original plan was wrong.** There is no `sm_121f` target. Triton 3.6.0 maps capability `(12,1)` to `sm_121a` via `sm_arch_from_capability` (verified in installed wheel). `TRITON_OVERRIDE_ARCH=sm_121f` fails the `^sm(\d+)$` regex.

The actual problem: `tl.dot_scaled` on sm_121 hits a dead path in `AccelerateMatmul.cpp`:
```cpp
// lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp (Triton 3.6.0, line ~665)
if (computeCapability != 120) return failure();  // ScaledBlockedToMMA (native FP4)
```
On sm_121, this returns failure and the rewriter falls through to the MMAv5/tcgen05 path, which cannot run on GB10 hardware (tcgen05 instructions don't exist for sm_121).

**Two options for getting native FP4 codegen:**

- **A. Source patch (cleaner):** relax the check to `computeCapability != 120 && computeCapability != 121`, rebuild Triton wheel, install locally.
- **B. Inline PTX (no Triton fork):** write the kernel with `tl.inline_asm_elementwise` emitting `mma.sync.aligned.m16n8k32.row.col.kind::mxf4.f32.e2m1.e2m1.f32` directly. Bypasses Triton's MMA lowering entirely.

Verify post-compile with `cuobjdump --dump-sass` that the issued opcode is `HMMA` with the `mxf4/nvf4` mnemonic (not `QMMA.tcgen05`). Arch header will read `sm_121a`, which is correct — the relevant fact is the MMA variant, not the sm suffix.

---

## Tile and schedule for GB10

- **Shared memory:** 99 KiB/SM on GB10 vs 228 KiB on B200. B200-optimized tiles **will not fit**.
- **Starting tile:** `BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, NUM_STAGES=2`. Budget ~54 KiB for A/B/C slices per stage.
- **For the expert decode path (M=1..8):** use a GEMV-shaped kernel or a small-M specialization — B200 tile shapes assume M≥128. Veitner's NVFP4 GEMV write-up is the reference for the small-M case.
- **For the expert prefill path (M up to 2816):** the larger tile shape above is appropriate.
- **Autotune axes:** `BLOCK_M ∈ {16, 32, 64, 128}`, `BLOCK_N ∈ {64, 128}`, `BLOCK_K ∈ {128, 256}`, `NUM_STAGES ∈ {2, 3}`, `num_warps ∈ {4, 8}`. Cache autotune results per-shape to avoid re-tuning on every load.

---

## Integration into the bridge

1. `bridges/gemma4/modeling_nvfp4.py` — flesh out `Gemma4TextExpertsNVFP4.forward()`:
   - Keep packed uint8, fp8 block scales, fp32 tensor scale as persistent buffers (already the case per current plan).
   - Per expert: call `nvfp4_linear(x, gate_up_packed[e], gate_up_scale[e], gate_up_tensor_scale[e])`, then SiLU gate × up product, then `nvfp4_linear(hidden, down_packed[e], down_scale[e], down_tensor_scale[e])`.
2. `bridges/gemma4/loader.py` — remove the `_convert_nvfp4_experts_to_fp8` call from `load_gemma4_nvfp4()`. The experts stay NVFP4 all the way through. Keep the conversion function around as dead code for one version, delete in the next release.
3. `_preconvert_fp8_scales` no longer applies to expert weights. Non-expert `_scaled_mm` path is unchanged.
4. Add `AEO_NVFP4_NATIVE=1` gate on first ship so we can bisect if parity breaks. Remove the gate once shipped.

---

## Verification gates (ordered, each must pass before moving on)

### Gate 1 — kernel correctness on one expert
- Load one gate_up_proj expert weight.
- Run our kernel on random bf16 activations `(M=1, K=2816)`.
- Compare output to: bf16 reference matmul on the dequant-to-bf16 weight.
- Pass criterion: max relative error < 2× the known bf16→NVFP4→bf16 round-trip error (~20%). If higher, diagnose before proceeding.

### Gate 2 — SASS verification
- `cuobjdump` the compiled kernel.
- Pass criterion: issued ops include `tcgen05.mma.fp4`, arch header `sm_121f`.

### Gate 3 — microbench vs dequant-to-FP8 path
- Same one-expert setup.
- Decode shape `(M=1, K=2816)` and prefill shape `(M=2816, K=2816)`.
- Pass criterion: decode ≥ 1.5× FP8 path, prefill ≥ 1.2× FP8 path. Below that, the kernel isn't worth the complexity; revisit.

### Gate 4 — full-model parity
- `QUANT_FORMAT=nvfp4 AEO_NVFP4_NATIVE=1 uv run examples/parity_check.py`.
- Pass criterion: output is coherent, reasoning_check.py passes on both prompts. Byte-exact match vs FP8 baseline is NOT required (native FP4 accumulates differently than FP8).

### Gate 5 — full-model tok/s
- `QUANT_FORMAT=nvfp4 AEO_NVFP4_NATIVE=1 uv run examples/profile_generate.py`.
- Pass criterion: decode ≥ 20 tok/s (target stretch 30+).

### Gate 6 — reasoning quality
- `QUANT_FORMAT=nvfp4 AEO_NVFP4_NATIVE=1 uv run examples/reasoning_check.py`.
- Pass criterion: both prompts produce correct, rigorous output. Token-level divergence is expected and fine; semantic correctness is the bar.

---

## Risk list

| Risk | Mitigation |
|---|---|
| TMA descriptor path broken on sm_121 | Non-TMA `block_ptr` fallback; eats ~20% speed but still ships |
| Triton `dot_scaled` codegen itself broken on sm_121f | Last resort: emit `tcgen05.mma.fp4` via inline PTX. Deep rabbit hole — reconsider the torchao probe first |
| Option A scale-fold overflows on some expert | Option B epilogue default from day one |
| Autotune finds bad config (e.g., NUM_STAGES=3 at 99 KiB smem fails silently) | Hand-check winning config's smem budget; add assert in launcher |
| Small-M GEMV performance regresses relative to FP8 `_scaled_mm` decode | Keep FP8 path as fallback under env flag; bisect per-shape |
| Parity drift large enough to affect reasoning output | reasoning_check.py is the hard gate; don't ship without it passing |
| CUTLASS/FlashInfer upstream fix lands mid-work | Re-evaluate: if their kernel is Python-callable from transformers, our kernel becomes dead code. That's fine — it was always a transient-stack fix |

---

## Out of scope for this plan

- Switching to vLLM/TRT-LLM/llama.cpp/FlashInfer-backend (violates transformers-only scope — see `feedback_aeo_quant_transformers_only`).
- Quantizing non-expert weights (attention, MLP, LM head) to NVFP4 — deferred per v2 in the NVFP4 plan.
- Batched 3D fused-experts kernel variant — Phase 3 optimization after the per-expert form works.
- Writing a non-Triton kernel (raw CUDA/CUTLASS) — only if Triton `dot_scaled` itself turns out to be broken on sm_121f.

---

## Where to start, literally (REVISED 2026-04-17)

Ordered by risk (cheapest first, rebuild is expensive):

### Step 0 — torchao probe (5 min, documents expected failure)

`examples/probe_nvfp4_torchao.py` under `safe_probe.py`. Expected to fail on `_scaled_mm → cuBLAS`. Record exact rejection. Not a gate — just evidence.

### Step 1 — TRITON_OVERRIDE_ARCH=sm120 compile probe (5 min, decides everything)

Write a minimal nvfp4 matmul with `tl.dot_scaled` at synthetic 128×128×128 shape. Compile twice:

```bash
# default (sm_121a): should hit scaled-dot decomposition fallback
uv run python examples/probe_nvfp4_minimal.py

# override: should match ScaledBlockedToMMA and emit native FP4 MMA
TRITON_OVERRIDE_ARCH=sm120 uv run python examples/probe_nvfp4_minimal.py
```

For each case, extract the compiled cubin and run `cuobjdump --dump-sass`:
- Default: expect decomposed FP16 `FMA` or `HMMA` fallback ops
- Override: expect `HMMA.MXF4` or `HMMA.NVFP4` — native block-scaled FP4 tensor core

Also compare against bf16 reference. Correctness gate: max rel err < 20% (the fp4 round-trip floor).

If override path works → Steps 2–4 in original plan apply without changes.
If override path compiles but outputs garbage → likely sm_121-specific instruction encoding downstream; proceed to Step 1.5.
If override path fails to compile → unexpected; read ptxas error carefully.

### Step 1.5 — Patch Triton source and rebuild (45 min if needed)

Diff the `computeCapability != 120` check to accept 121 per the research doc. Grep `SM120` across the tree for additional asserts. Build Triton from source:

```bash
cd third_party/triton   # or wherever we clone
pip install -e python  # or build wheel
```

### Step 2 — Adapt tutorial 10 kernel for our shapes

Fork `docs/references/triton_tutorial_10_v36.py` into `src/aeo_quant/gpu/nvfp4_matmul.py`. Key adjustments for GB10:
- Strip `matmul_ogs` reference (tutorial doesn't use it anyway — original plan was wrong).
- Shrink default tile: BLOCK_M=BLOCK_N=128, BLOCK_K=128, NUM_STAGES=2 (GB10 has 99 KiB smem vs 228 KiB on B200).
- Add small-M GEMV specialization (M∈{1,8} for decode).
- Drop `supports_block_scaling()` gate (or relax to accept sm_121).
- Per-tensor scale (Option B epilogue): fold as a post-MMA fmul per tile.

### Step 3 — Hook into Gemma 4 bridge

Replace `Gemma4TextExpertsNVFP4.forward()`'s `raise RuntimeError(...)` with a loop that calls our `nvfp4_linear` per selected expert. Keep conversion fallback behind `AEO_NVFP4_NATIVE=0` env flag for bisection.

### Step 4 — Verification gates 1–6

Run in order. Each gates the next. Gates 4–6 require full model load (~27 GB VRAM) — must coordinate with other GPU users on this box.
