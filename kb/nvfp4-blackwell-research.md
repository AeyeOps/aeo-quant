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
