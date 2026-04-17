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
