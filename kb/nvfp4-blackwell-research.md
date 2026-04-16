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
