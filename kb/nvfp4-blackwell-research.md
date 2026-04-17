# NVFP4 on Blackwell — Research Findings

Verified: 2026-04-16

## NVFP4 Format Spec

**FP4 E2M1** — 4 bits per value: 1 sign, 2 exponent, 1 mantissa.
Representable values: `{0, +/-0.5, +/-1.0, +/-1.5, +/-2.0, +/-3.0, +/-4.0, +/-6.0}` (max = 6.0)

**Two-level microscaling** (NVFP4's key advantage over plain FP4/MXFP4):
- Level 1: FP8 E4M3 scale per micro-block of **16 elements** (not 32 like MXFP4)
- Level 2: FP32 scalar per tensor (global range adjustment)
- Dequant: `value = tensor_scale * block_scale[row, block] * fp4_lookup[nibble]`
- Effective storage: ~4.5 bits per value (FP4 + amortized scale overhead)
- Block size 16 gives ~175x finer granularity than FP8 per-channel scaling

**Nibble encoding** (sign-magnitude): high nibble first when packing two per byte.

## Community Benchmarks

- vLLM on GB10 with Gemma 4 26B-A4B NVFP4: **52 tok/s** (vs our 10 tok/s FP8)
  - vLLM uses native NVFP4 kernels that skip the FP8 dequant step
  - Our ~5x gap is primarily the bandwidth advantage of keeping FP4 in memory

## CUTLASS/FlashInfer sm_121 Bugs

Native FP4 matmul on Blackwell (sm_121) is **broken** as of 2026-04-16:
- CUTLASS: fp4 gemm kernels don't support sm_121 (GB10 = Blackwell mobile)
- FlashInfer: same CUTLASS dependency, same gap
- `torch._scaled_mm`: no FP4 input support
- `torch.float4_e2m1fn_x2`: dtype exists in torch 2.7+ but ops are CPU-only

**Status:** Upstream issue, not fixable by us. When fixed, upgrade is localized
to the loader (swap `_convert_nvfp4_experts_to_fp8` for direct FP4 matmul).

## Dequant-to-FP8 Strategy

Our workaround: **store NVFP4, dequant to FP8 at load, run `_scaled_mm`.**
- Same inference speed as FP8 (proven path, no new kernels)
- Checkpoint 19% smaller (23.4 GB vs 28.8 GB, experts-only)
- Every load reconverts (~10s with batched-16 optimization). No on-disk
  conversion cache — see "Conversion cache removed" below.
- When native kernels land, the checkpoint format is unchanged — only the
  loader's compute path changes

## Double-Quantization Error Budget

bf16 -> NVFP4 -> bf16 -> FP8 vs direct bf16 -> FP8:
- Mean relative diff: ~17% (expected for FP4 coarseness)
- NVFP4's per-16-element block scaling preserves range well enough that
  the FP8 re-quantization doesn't amplify errors excessively

## Conversion Cache Removed (v0.1.5)

Original design included `.fp8_cache/` to skip conversion on subsequent loads.
Batched-16-experts conversion reduced conversion cost from an estimated 30-60s
to 9.5s. Cache load (disk I/O for 21 GB) consistently took ~124s — slower than
just reconverting by ~114s per load.

Removed in v0.1.5. Every NVFP4 load does fresh conversion. Full write-up in
`docs/gemma4-fp8-optimization.md` Step 6.

Lesson: when you optimize a cost, re-evaluate cached bypasses of that cost.

## Native NVFP4 matmul path on sm_121 — 2026-04-16 survey

**Bottom line.** No drop-in NVFP4 block-scaled matmul kernel exists today that we can import, call from our transformers-path `forward()`, and that is verified on sm_121 with our checkpoint layout (packed uint8, fp8_e4m3 per-16 block scale, fp32 per-tensor scale). The viable path is write our own Triton kernel from the Triton `tl.dot_scaled` tutorial as the starting base. Before committing to that, a 20-minute torchao-via-cuBLAS probe can tell us whether a zero-code alternative exists.

### Candidate inventory

| Candidate | Install | CC advertised | Block size | Scale dtypes | Verdict | Notes / evidence |
|---|---|---|---|---|---|---|
| Triton `tl.dot_scaled` (tutorial 10) | `pip install triton` ≥ nightly with PTX 8.7 | sm_100/sm_101 documented; sm_120/121 path hits `matmul_ogs` guard | 16 (NVFP4), 32 (MXFP4) | fp8_e4m3 *and* e8m0 uint8 | **Best base — adapt** | Issues E2M1 unpack + block-scale apply for free via `tcgen05.mma`. Wrapping layer `matmul_ogs` throws *"Must use persistent kernel and be TMA-compliant for native MXFP"* on Grace Blackwell — triton#8548 closed Oct 2025, fix on main, not yet in PyTorch 2.11 bundle. Fork the tutorial kernel directly; don't route through `matmul_ogs`. |
| torchao NVFP4 (`_addmm_nvfp4_dispatch` on `NVFP4Tensor`) | `pip install torchao` | No explicit gate; routes through `torch._scaled_mm` → cuBLAS `fp4_e2m1fn_x2` | 16 | fp8_e4m3 (swizzled or plain) | **Probe first** | Layout matches ours. cuBLAS fp4 path reported working on sm_120 B200 but **not verified on sm_121**. A 20-minute one-expert probe would settle it. If it runs clean, this is R2 with zero kernel code. |
| HF `kernels-community/triton_kernels` | `pip install kernels` + hub load | advertises CC ≥ 9.0 for native FP4 path | 32 (MXFP4 only) | e8m0 uint8 | Not usable | MXFP4 semantics (block_size 32, e8m0 scale), not NVFP4. Same `matmul_ogs` guard. |
| GemLite `A4W4_NVFP_dynamic` | `pip install gemlite` | "focus sm_120"; sm_121 not listed | 16 (claimed NVFP4) | fp8_e4m3 implied | Reference / probe | Dynamic-activation oriented; sm_121 kernel path unverified. |
| Veitner NVFP4 GEMV, advpropsys/fp4-blackwell-bench | git clone | sm_100a only | 16 | fp8_e4m3 | Reference only | Useful for E2M1 unpack pattern and Blackwell tile/schedule shape. |

### sm_121 compile target gotcha

Vanilla `sm_121` and `sm_121a` **reject** the `tcgen05.mma` FP4 instructions. Per the NVIDIA dev forum (sm121 GB10 NVFP4 software-support thread), the required target is **`sm_121f`** (family mode). Any kernel built for this hardware must pass `-arch=sm_121f` — Triton nightly supports this target but the default pipeline often falls back to `sm_100a` and silently mis-lowers FP4 ops. Verify with `cuobjdump` after compile.

### Shared-memory budget

GB10 has **99 KiB smem per SM**, vs 228 KiB on B200. B200 default tiles for Triton NVFP4 kernels (typically 128×128×256 or 128×256×256 at `NUM_STAGES=3`) **will not fit**. Starting tile for GB10: **128×128×128 at `NUM_STAGES=2`**, then tune. Budget: ~54 KiB for A/B/C slices per stage at those dims. Hand-check before first compile.

### Blockers on other stacks (reference, do not chase)

- CUTLASS `BlockScaledMmaOp`: hardcoded `sm_100a` (CUTLASS#2800). C++ API works, Python DSL rejects sm_121.
- FlashInfer `mm_fp4`: returns all zeros on sm_120 with cutlass backend, errors on cudnn/trtllm (FlashInfer#2577, open Feb 2026).
- TensorRT-LLM FP4 GEMM: smem overflow on sm_121 (TRT-LLM#11368).

### Key references

- Triton tutorial 10 — Block Scaled Matmul: https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
- triton#8548 — MXFP4 TMA guard on Grace Blackwell
- torchao NVFP4Tensor source: `pytorch/ao/blob/main/torchao/prototype/mx_formats/nvfp4_tensor.py`
- PyTorch blog — Faster Diffusion on Blackwell (MXFP8/NVFP4 with diffusers + torchao)
- NVIDIA dev forum — "SM121 (GB10) native NVFP4 compute — seeking guidance on software support"
- NVIDIA dev forum — "tcgen05 FP4 support for DGX Spark GB10 sm121"
- Veitner — NVFP4 GEMV: https://veitner.bearblog.dev/nvfp4-gemv/
- advpropsys/fp4-blackwell-bench

---

## Deep re-survey — 2026-04-17 (corrections + new findings)

Originally drafted 2026-04-16; several claims were wrong. This section supersedes.

### 1. `sm_121f` is NOT a Triton target string

Verified in the installed Triton 3.6.0 wheel on this box:
```python
# triton/backends/nvidia/compiler.py
def sm_arch_from_capability(capability: int):
    suffix = "a" if capability >= 90 else ""
    return f"sm_{capability}{suffix}"
# sm_arch_from_capability(121) → "sm_121a"
```
The `_parse_arch` regex is `^sm(\d+)$` — any non-digit suffix is rejected. `TRITON_OVERRIDE_ARCH=sm_121f` fails. `GPUTarget` is a 3-tuple `(backend, arch_int, warp_size)`, not `(backend, cap, variant_char)`.

**Correction to plan:** "Delta 3 — sm_121f compile target" is moot. Triton 3.6.0 already emits `sm_121a`; there is no "f" suffix to force. PR #9734 tried to change this, was reverted in PR #9755 (2026-03-17) per NVIDIA feedback.

### 2. triton#8548 was never fixed

Issue was closed by the original reporter one day after opening, no PR merged. Lezcano's only comment: *"Is this AI generated? Those changes are clearly not correct."* The actual sm_121 enablement attempt was PR #8484 ("sm120/121 via sm80 fallback"), closed unmerged 2025-11-21. Umbrella tracker #8335 (closed) noted masahi's workaround commit 66d39cc was "not landable" because it makes sm_120/121 behave as sm_80.

**Correction to plan:** "Delta 2 — bypass the `matmul_ogs` TMA guard" oversimplifies. The real blocker is one-deeper: Triton's `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp` has a hard `computeCapability != 120 → failure` in the native-FP4 `ScaledBlockedToMMA` path (added PR #8494, merged 2025-10-20). On sm_121, `tl.dot_scaled` with FP4 falls through to the tcgen05/MMAv5 path that **cannot run on GB10 hardware at all**.

### 3. GB10 has `mma.sync...mxf4/nvf4`, NOT `tcgen05.mma.*`

NVIDIA employee `caelunshun` on CUTLASS #2947 (2026-01-11): *"The `tcgen05` instructions do not exist for SM121, so this is not possible to fix. The `mma` instructions with FP4 precision do, however, exist."*

This is the single most important hardware fact. Any code path targeting `tcgen05.mma.fp4` on GB10 is DOA. The correct PTX instruction is warp-level `mma.sync.aligned.{shape}.row.col.kind::mxf4.f32.e2m1.e2m1.f32` (or `kind::nvf4` for the two-level NVFP4 form). Tensor memory (`tmem`) and the `tcgen05` family don't exist on this chip.

### 4. torchao's `_addmm_nvfp4_dispatch` routes through `_scaled_mm` → cuBLAS

Read at `torchao/prototype/mx_formats/nvfp4_tensor.py` line 506:
```python
result = torch._scaled_mm(
    a.qdata.view(torch.float4_e2m1fn_x2),
    b.qdata.view(torch.float4_e2m1fn_x2),
    a_scale_blocked.view(torch.float8_e4m3fn),
    b_scale_blocked.view(torch.float8_e4m3fn),
    ...
)
```
`torch._scaled_mm` with FP4×FP4 routes to cuBLAS; cuBLAS's `fp4_e2m1fn_x2` path is B200 / sm_100 only as of CUDA 13.1. On sm_121 it either errors ("Not implemented") or silently falls back. Confirmed failure in the wild: vLLM #30163 reports *"[FP4 gemm Runner] Failed to run cutlass FP4 gemm on sm120"* on DGX Spark. torchao's own tracker (#3102) states: *"`_addmm_nvfp4_dispatch` only supported on B200 currently."*

**Correction to plan:** The 20-minute probe will likely fail. Still worth running — the exact error string tells us which layer rejected us.

### 5. CUTLASS gap is real and recent

Live open issues as of today:
- CUTLASS #2802 — `BlockScaledMmaOp.admissible_archs = [Arch.sm_100a]` rejects sm_121 hard
- CUTLASS #2800 — "expects arch to be one of ['sm_100a','sm_100f'], but got sm_121a"
- CUTLASS #3100 — `nvidia-cutlass-dsl==4.4.1` has no SM121 SASS; override fails at runtime
- CUTLASS #3144 — `StageCountAutoCarveout` assumes max family SMEM, breaks at 99 KiB/SM
- CUTLASS #3096 (2026-04-14, "With Fix") — SM120 NVFP4 MoE produces garbage; user patched FlashInfer 0.6.5 + CuTe DSL arch whitelist manually, got 39 tok/s

CuTe DSL 4.5 wheel with SM121 SASS is the unreleased upstream gate. Until then, no drop-in kernel path.

### 6. FlashInfer b12x is SM120-only (explicitly)

Merged PRs #3051 (2026-04-14) and #3066 (2026-04-15) add `backend="b12x"` for SM120. Both explicitly: *"SM121 (Spark) is not yet supported pending a nvidia-cutlass-dsl==4.5 wheel release."* vLLM issue #40082 (2026-04-17) tracks vLLM's integration of FlashInfer b12x.

### 7. TMA works on sm_121 in Triton 3.6.0

PR #8498 (merged 2025-10-24, shipped in 3.6.0) enabled TMA scatter/gather4 for sm_121. The TMA path itself is fine — what breaks is the downstream FP4 MMA lowering. This means a tutorial-10-style kernel can load scales via 5D TMA descriptors without issue.

### 8. ptxas version in the 3.6.0 wheel is Blackwell-capable

PR #8941 upgraded bundled ptxas to 13.1 (merged 2025-12-11, in 3.6.0). No need to symlink CUDA 13 ptxas like vLLM 0.11 users hit — our wheel is fine.

### Strategic implications

Two realistic kernel paths from here (ranked by estimated effort):

**Path A — Patch Triton's AccelerateMatmul check and use `tl.dot_scaled`:**
1. Monkey-patch or fork `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp` to accept `computeCapability == 121` for the native-FP4 MMAv2 path.
2. Verify the generated SASS uses `mma.sync...kind::mxf4` (not `tcgen05.mma.*`).
3. If tcgen05 is emitted, we need to force the MMAv2 (warp-level) codegen, not MMAv5 — this likely requires a second patch to the MMA version selection.
4. Build Triton from source (the monkey-patch option doesn't work for C++; we need to either patch-and-rebuild or use a pre-built patched wheel).

**Path B — Inline PTX kernel in Triton:**
1. Write a `@triton.jit` kernel that uses `tl.inline_asm_elementwise` to emit `mma.sync.aligned.m16n8k32.kind::mxf4.f32.e2m1.e2m1.f32` directly.
2. Handle scale application in kernel (fold per-tensor fp32 in epilogue, apply fp8 block scales inside loop).
3. Ship a minimal kernel that doesn't depend on `tl.dot_scaled` lowering at all.
4. Bigger up-front effort, but no Triton fork to maintain.

**Path C (emergency fallback) — CUDA C++ kernel:**
1. Write in plain CUDA with inline PTX, compile with nvcc to a .so, load via torch.ops or cffi.
2. Maximum control but maximum scaffolding.
3. Reserved if A and B both hit walls.

### Sources cited in this section

- triton-lang/triton: PRs #8484, #8494, #8498, #8941, #9734, #9755; Issues #8548, #8335
- pytorch/ao: `nvfp4_tensor.py` source; Issues #3102, #4040; PR #4188
- NVIDIA/cutlass: Issues #2800, #2802, #2947, #3096, #3100, #3144
- flashinfer-ai/flashinfer: PRs #3051, #3066, #3080; Issue #3013
- vllm-project/vllm: Issues #30163, #40082; PR #39920
- Local verification: `uv run python -c "from triton.backends.nvidia.compiler import sm_arch_from_capability; print(sm_arch_from_capability(121))"` → `sm_121a`

---

## Second deep dive — 2026-04-17 PM (revised kernel strategy)

Previous section's Path B ("inline PTX via Triton") is dead. This section supersedes it. Three major corrections.

### Correction 1: PTX instruction names

- `.kind::nvf4` **does not exist**. The PTX ISA has three FP4-related kinds:
  - `.kind::mxf4` — pure MX-FP4 with `.ue8m0` scales, scale_vec::2X only
  - `.kind::mxf4nvf4` — NVFP4 (NVIDIA's global×block FP4), supports `.ue8m0`@2X **or** `.ue4m3`@4X
  - `.kind::mxf8f6f4` — mixed-precision, `.ue8m0`@1X only
- NVFP4 canonical shape is **`m16n8k64`**, not `m16n8k32`. K=64 is fixed for mxf4/mxf4nvf4 (one block per MMA tile).
- Under `.kind::mxf4` / `.kind::mxf4nvf4`, E2M1 is packed **8 values per `.b32` with no padding** — matches our uint8 layout after view.

Verbatim example (PTX ISA 8.7 p. 450):
```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
    {%Rd0..3}, {%Ra0..3}, {%Rb0..1}, {%Rc0..3},
    scaleAData, {bidA, tidA}, scaleBData, {bidB, tidB};
```
For scale_vec::4X, `bidA = bidB = 0` (Table 38).

### Correction 2: sm_121a PTX assembly works

PTX ISA 8.8 (CUDA 13.0+) extended `.kind` / `.block_scale` / `.scale_vec_size` from `sm_120a` to the `sm_12x` **family target `sm_120f`**, which includes `sm_121a` as a later-generation a-target (PTX ISA 8.8 p. 511 — updated gating text).

Chain:
1. sm_121a is a later-generation a-target in the sm_12x family.
2. Later-generation a-targets inherit family features from earlier f-targets.
3. sm_120f (family target) has `.kind::mxf4` / `.kind::mxf4nvf4` / `.kind::mxf8f6f4`.
4. Therefore sm_121a supports all these instructions.

Confirmed empirically by llama.cpp issue #19662 and Triton issue #8539: **sm_121a assembles cleanly under CUDA 13 ptxas**. The bundled ptxas in Triton 3.6.0 is 13.1 (PR #8941), so this works out of the box on our install.

### Correction 3: Triton `inline_asm_elementwise` cannot host `mma.sync`

From Triton docs: *"Each invocation of the inline asm processes `pack` elements at a time. Exactly which set of inputs a block receives is unspecified."* `mma.sync` is **warp-collective** — it requires all 32 lanes to cooperate on a specific tile layout per the PTX ISA fragment tables. Elementwise inline asm provides no such guarantee.

**Path B is dead at the Triton layer.** If we ever need inline PTX, it has to be in a CUDA C++ kernel, compiled with nvcc, loaded via torch.utils.cpp_extension — heavyweight scaffolding. Deferred as emergency-only fallback.

### The actual blocker — `ScaledBlockedToMMA` in `AccelerateMatmul.cpp`

Verified source at v3.6.0:

```cpp
// lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:653
class ScaledBlockedToMMA : public mlir::OpRewritePattern<triton::DotScaledOp> {
  int computeCapability;
public:
  ...
  mlir::LogicalResult matchAndRewrite(triton::DotScaledOp dotOp,
                                      mlir::PatternRewriter &rewriter) const override {
    if (computeCapability != 120)      // <-- line 665
      return failure();
    ...
    auto mmaResult = createMMAEncodingForDot(dotOp, rewriter, computeCapability, 2);  // MMAv2
    ...
    auto ll = triton::gpu::getSM120DotScaledScaleLayout(
        ctx, shape, opIdx, mmaWarps, blocked.getCTALayout());               // helper
    ...
  }
};
```

Plus a secondary check in `mmav2SupportsFp8Operands` (line 911) — only for FP8-in-MMAv2 path, not on our critical NVFP4 route, but good to know:
```cpp
return computeCapability == 89 || computeCapability == 120;
```

**The SM120-specific helper `getSM120DotScaledScaleLayout` is called unconditionally** — if it has a `cc == 120` assert inside, the source patch alone isn't enough. Must grep the tree before rebuilding.

### The cheap path first — `TRITON_OVERRIDE_ARCH=sm120`

`AccelerateMatmul.cpp` reads capability from the `ttg.target` MLIR attribute (string `"cuda:<int>"`), which the Python NVIDIA backend stamps from `target.arch`. That in turn respects `knobs.runtime.override_arch`, which is populated from env var `TRITON_OVERRIDE_ARCH`.

```python
# third_party/nvidia/backend/compiler.py:177
args = {'arch': knobs.runtime.override_arch or f"sm{self.target.arch}"}
```

So `TRITON_OVERRIDE_ARCH=sm120` causes Triton to compile as if the GPU were sm_120. The MMAv2 scaled pattern matches, MLIR lowers to `.kind::mxf4nvf4` PTX, ptxas assembles for sm_120 target, and the driver's runtime JIT loads it onto sm_121 — which is valid because consumer-Blackwell shares the same MMA encoding across sm_120/sm_121.

Risks:
1. Downstream instruction selection may emit sm_120-specific encodings that fail at driver load on sm_121. Empirically test, don't trust reasoning.
2. ptxas target register-file / smem assumptions could differ. Check the cubin post-compile.
3. The sm_121a family feature set is a *superset* of sm_120f, so the reverse (sm_121→sm_120) shouldn't drop required features. But verify with a tiny correctness probe.

**5-minute test plan when GPU frees:**
1. Write a 128×128×128 nvfp4 matmul kernel
2. Compile twice: once with default sm_121a, once with TRITON_OVERRIDE_ARCH=sm120
3. Default case: confirm `tl.dot_scaled` hits the fallback decomposition (slow).
4. Override case: confirm the SM120 MMAv2 scaled pattern matches (fast), SASS shows `HMMA ... MXF4` ops (not fallback `FMA`).

### The full-rebuild path — patching AccelerateMatmul.cpp

If override-arch doesn't work, minimal diff:

```diff
// lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp
-    if (computeCapability != 120)
+    if (computeCapability != 120 && computeCapability != 121)
       return failure();

// Also (for FP8 operands in MMAv2):
-  return computeCapability == 89 || computeCapability == 120;
+  return computeCapability == 89 || computeCapability == 120 ||
+         computeCapability == 121;
```

Plus grep for `cc == 120` / `SM120` in related files. Likely to need adjustment:
- `include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h`
- `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`
- `third_party/nvidia/lib/...` NVGPU→LLVM conversions

Triton build takes ~45 min from source. Do this only if the override doesn't work.

### Revised strategy

1. **Wait for GPU memory to free**, then run `probe_nvfp4_torchao.py` under the harness. Document exact failure. [expected: fail on _scaled_mm]
2. **5-minute probe**: write a minimal nvfp4 matmul with `tl.dot_scaled`, compile under `TRITON_OVERRIDE_ARCH=sm120`, compare SASS to default compile. [expected: override emits HMMA.MXF4, default emits FMA fallback]
3. **If override works**: adapt tutorial 10 kernel for our shapes, integrate into `Gemma4TextExpertsNVFP4.forward()`, run verification gates.
4. **If override doesn't work**: patch Triton source, rebuild locally, test again.
5. **Last resort**: CUDA C++ extension with inline PTX, compile via torch.utils.cpp_extension. Only if Triton source patching hits walls.

### Sources cited in this section

- PTX ISA 8.7: https://docs.nvidia.com/cuda/pdf/ptx_isa_8.7.pdf (sections 9.7.14.3, 9.7.14.5.11, 9.7.14.5.14)
- PTX ISA 8.8: https://docs.nvidia.com/cuda/pdf/ptx_isa_8.8.pdf (section 1.3, 11.1.2)
- Triton v3.6.0 AccelerateMatmul.cpp: https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp
- CUTLASS mma_sm120.hpp: https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm120.hpp
- llama.cpp #19662 (sm_120 base vs sm_120a for mxf4)
- Triton #8539 (sm_121a + CUDA 13 ptxas)
- Triton `tl.inline_asm_elementwise` docs confirming non-warp-collective semantics
- NVIDIA forums: "Run ptx mma.sync.aligned.kind::mxf8f6f4...sm_120a" thread

---

## Offline AOT compile validation — 2026-04-17 PM

Path A.5 is confirmed to work at the Triton/MLIR/PTX level **without
needing a live GPU**.  Using ``triton.compile(ASTSource, target=GPUTarget('cuda', 120, 32))``:

### What compiles

The NVFP4 probe kernel using ``tl.dot_scaled(a, as, "e2m1", b.T, bs, "e2m1", acc)``
with scale_size=16 lowers cleanly under target sm_120.  The resulting
PTX contains 64 native NVFP4 MMA instructions of the form:

```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
    { %r45, %r46, %r47, %r48 },                          // D (accumulator)
    { %r1, %r2, %r3, %r4 },                              // A (lhs e2m1, 4x .b32)
    { %r18, %r19 },                                      // B (rhs e2m1, 2x .b32)
    { %r45, %r46, %r47, %r48 },                          // C (prev accum, same regs as D)
    %r5, { 0, 0 },                                       // scale-A, {byte-id, thread-id}
    %r6, { 0, 0 };                                       // scale-B, {byte-id, thread-id}
```

Matches PTX ISA 8.8 section 9.7.14.5.14 verbatim, including the
`{0, 0}` byte-id/thread-id requirement for `scale_vec::4X`.

The emitted PTX header reads:

```
.version 8.8
.target sm_120a
.address_size 64
```

``sm_120a`` is a later-generation a-target whose features are
inherited by ``sm_121a`` per the sm_12x family relationship (see
PTX ISA 8.8 p. 794).  Driver JIT should load this cubin cleanly on
sm_121 hardware.

### What doesn't compile (default sm_121)

Compiling the same kernel for target sm_121 hits an MLIR assertion:

```
BuiltinAttributes.cpp:1030: ... DenseElementsAttr::get(ShapedType, ArrayRef<APInt>):
    Assertion `type.getElementType().isIntOrIndex()' failed.
```

The assertion is inside the decomposition-fallback path, not the
native path — i.e. when ``ScaledBlockedToMMA`` rejects sm_121, the
pattern the compiler *tries next* is buggy.  Not our problem as long
as we never trigger that path.

### No downstream sm_120-specific asserts found (good news)

The kernel compiles end-to-end for sm_120 without hitting any of the
``getSM120DotScaledScaleLayout``-internal asserts we worried about.
The helper handles the scale layout uniformly; no additional patches
needed.

### What's still unverified

Whether a cubin compiled for ``sm_120a`` executes correctly on sm_121
hardware.  Can only be confirmed with a live GPU run.  Evidence-based
expectation: yes, because
(a) sm_121a is in the sm_12x family tree that inherits from sm_120f;
(b) consumer-Blackwell sm_120 and sm_121 share the same MMA
    instruction set per NVIDIA dev-forum confirmations;
(c) Triton ships ptxas 13.1 which accepts sm_120a cleanly.

The remaining risk surface is in driver JIT behavior, not in the
instruction set itself.

### How to reproduce

```bash
uv run python examples/probe_nvfp4_aot.py --targets 120
```

Output ends with a line count of `kind::mxf4[nvf4]` PTX ops — should
be non-zero for target 120.  Adding ``--dump-ptx`` prints the first
60 lines of PTX for inspection.
