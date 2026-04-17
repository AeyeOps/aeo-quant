#!/usr/bin/env python3
"""Minimal NVFP4 matmul probe — does TRITON_OVERRIDE_ARCH=sm120 work on GB10?

Compiles a tiny tl.dot_scaled kernel at synthetic 128x128x128 shape and
tries to launch it. Intended to be run TWICE:

  # Default: sm_121a — expected to hit fallback decomposition (slow, fp16)
  uv run python examples/probe_nvfp4_minimal.py

  # Override: sm_120 coercion — expected to emit native FP4 MMA
  TRITON_OVERRIDE_ARCH=sm120 uv run python examples/probe_nvfp4_minimal.py

After each run the probe dumps:
  - the Triton cache directory (for cuobjdump --dump-sass inspection)
  - the kernel's IR / PTX / cubin hashes
  - max relative error vs bf16 reference matmul

Interpretation: if the override run shows `HMMA` with an MXF4 mnemonic
in the dumped SASS while the default run shows only `HMMA.16816.F32.F16`
(standard fp16 HMMA from the decomposition), then Path A.5 works and
we can drop to the kernel integration step with no Triton rebuild.

Safe: VRAM usage < 10 MB. Runtime < 5 seconds (excluding JIT compile).
"""
from __future__ import annotations

import os
import sys

import torch


# Shape parameters — small enough to keep everything under a few MB.
M, N, K = 128, 128, 128
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
VEC_SIZE = 16  # NVFP4 block size


def _build_kernel():
    """JIT-define the Triton kernel.  Done inside a function so import
    of this module doesn't compile anything — the kernel is only built
    when main() runs.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def _nvfp4_matmul_kernel(
        a_ptr, b_ptr,
        a_scale_ptr, b_scale_ptr,
        c_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_am_scale, stride_ak_scale,
        stride_bn_scale, stride_bk_scale,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        VEC_SIZE: tl.constexpr,
        ELEM_PER_BYTE: tl.constexpr,
    ):
        """One program per output tile.  Single K-iter since K=BLOCK_K=128."""
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        # K indices for packed uint8 (K // ELEM_PER_BYTE bytes per row)
        offs_k_packed = tl.arange(0, BLOCK_K // ELEM_PER_BYTE)
        # K indices for scales (K // VEC_SIZE scales per row)
        offs_k_scale = tl.arange(0, BLOCK_K // VEC_SIZE)

        a_ptrs = (a_ptr + offs_m[:, None] * stride_am
                  + offs_k_packed[None, :] * stride_ak)
        b_ptrs = (b_ptr + offs_n[:, None] * stride_bn
                  + offs_k_packed[None, :] * stride_bk)
        a_scale_ptrs = (a_scale_ptr + offs_m[:, None] * stride_am_scale
                        + offs_k_scale[None, :] * stride_ak_scale)
        b_scale_ptrs = (b_scale_ptr + offs_n[:, None] * stride_bn_scale
                        + offs_k_scale[None, :] * stride_bk_scale)

        a_raw = tl.load(a_ptrs)            # (BLOCK_M, BLOCK_K // 2) uint8
        b_raw = tl.load(b_ptrs)            # (BLOCK_N, BLOCK_K // 2) uint8
        # Our checkpoint pack: (k0 high, k1 low). Triton/NV expects the
        # opposite (k0 low, k1 high). 3-op swap, negligible vs MMA.
        a = ((a_raw & 0xF) << 4) | ((a_raw >> 4) & 0xF)
        b = ((b_raw & 0xF) << 4) | ((b_raw >> 4) & 0xF)
        a_scale = tl.load(a_scale_ptrs)    # (BLOCK_M, BLOCK_K // 16) fp8
        b_scale = tl.load(b_scale_ptrs)    # (BLOCK_N, BLOCK_K // 16) fp8

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc = tl.dot_scaled(a, a_scale, "e2m1", b.T, b_scale, "e2m1", acc)

        c_ptrs = (c_ptr + offs_m[:, None] * stride_cm
                  + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc.to(tl.bfloat16))

    return _nvfp4_matmul_kernel


def _make_inputs():
    """Build synthetic NVFP4 inputs at (M, N, K).

    Returns:
      a_packed: (M, K//2) uint8 — FP4 nibbles of A
      b_packed: (N, K//2) uint8 — FP4 nibbles of B^T (col-major natural for tl.dot_scaled)
      a_scale:  (M, K//16) fp8_e4m3fn
      b_scale:  (N, K//16) fp8_e4m3fn
      ref_bf16: (M, N) bf16 — torch.matmul(a_bf16, b_bf16.T)
    """
    from aeo_quant.gpu.quant import (
        quantize_2d_to_nvfp4,
        dequant_2d_from_nvfp4,
    )

    torch.manual_seed(0)
    device = "cuda"

    # Generate realistic-magnitude bf16 weights
    a_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1
    b_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.1

    # Quantize both to NVFP4
    a_packed, a_scale, a_tensor_scale = quantize_2d_to_nvfp4(a_bf16)
    b_packed, b_scale, b_tensor_scale = quantize_2d_to_nvfp4(b_bf16)

    # Dequantize back to bf16 for reference (so reference matches what
    # the FP4 kernel would produce on exact arithmetic — minus accum order)
    a_bf16_q = dequant_2d_from_nvfp4(a_packed, a_scale, a_tensor_scale)
    b_bf16_q = dequant_2d_from_nvfp4(b_packed, b_scale, b_tensor_scale)
    ref = torch.matmul(
        a_bf16_q.float() * a_tensor_scale,
        (b_bf16_q.float() * b_tensor_scale).T,
    ).to(torch.bfloat16)

    return a_packed, b_packed, a_scale, b_scale, a_tensor_scale, b_tensor_scale, ref


def main() -> int:
    override = os.environ.get("TRITON_OVERRIDE_ARCH", "")
    print(f"=== nvfp4 minimal probe ===")
    print(f"TRITON_OVERRIDE_ARCH = {override or '(not set)'}")

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available")
        return 2

    cc = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")
    print(f"torch: {torch.__version__}")

    import triton
    print(f"triton: {triton.__version__}")

    # Build inputs and reference
    print(f"\n--- building synthetic {M}x{N}x{K} NVFP4 inputs ---")
    try:
        a_packed, b_packed, a_scale, b_scale, a_tscale, b_tscale, ref = _make_inputs()
    except Exception as e:
        print(f"[FAIL] input build: {type(e).__name__}: {e}")
        return 1

    print(f"  a_packed:  {tuple(a_packed.shape)} {a_packed.dtype}")
    print(f"  b_packed:  {tuple(b_packed.shape)} {b_packed.dtype}")
    print(f"  a_scale:   {tuple(a_scale.shape)} {a_scale.dtype}")
    print(f"  b_scale:   {tuple(b_scale.shape)} {b_scale.dtype}")
    print(f"  a_tscale:  {a_tscale.item():.4e}")
    print(f"  b_tscale:  {b_tscale.item():.4e}")
    print(f"  ref norm:  {ref.float().norm().item():.4f}")

    kernel = _build_kernel()
    c = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    print(f"\n--- launching kernel grid={grid} ---")

    try:
        kernel[grid](
            a_packed, b_packed,
            a_scale, b_scale,
            c,
            M, N, K,
            a_packed.stride(0), a_packed.stride(1),
            b_packed.stride(0), b_packed.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            b_scale.stride(0), b_scale.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K,
            VEC_SIZE,
            2,  # ELEM_PER_BYTE: 2 fp4 values per byte
        )
        torch.cuda.synchronize()
        print(f"  [OK] kernel launch+sync")
    except Exception as e:
        print(f"  [FAIL] kernel: {type(e).__name__}: {e}")
        import traceback
        for line in traceback.format_exc().splitlines()[-15:]:
            print(f"    {line}")
        return 1

    # Correctness — both inputs are quantized, so kernel should match
    # our dequant-then-matmul reference modulo FP4 accum order.
    # Note: we don't apply the per-tensor scales here; they're absorbed
    # into the reference computation.
    c_scaled = c.float() * a_tscale * b_tscale
    ref_unscaled = ref.float()
    err_norm = (c_scaled - ref_unscaled).norm().item()
    ref_norm = ref_unscaled.norm().item()
    max_abs = (c_scaled - ref_unscaled).abs().max().item()
    rel = err_norm / (ref_norm + 1e-8)
    print(f"\n--- correctness ---")
    print(f"  kernel_out_scaled_norm: {c_scaled.norm().item():.4f}")
    print(f"  ref_norm:               {ref_norm:.4f}")
    print(f"  max_abs_err:            {max_abs:.4f}")
    print(f"  rel_fro_err:            {rel:.4f}")

    # Dump the Triton cache location for SASS inspection
    cache_root = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    print(f"\n--- post-compile artifacts ---")
    print(f"  cache root: {cache_root}")
    print(f"  inspect SASS with:")
    print(f"    find {cache_root} -name '*.cubin' -newer /tmp | xargs -I{{}} \\")
    print(f"        cuobjdump --dump-sass {{}} | head -80")

    return 0


if __name__ == "__main__":
    sys.exit(main())
