"""Native NVFP4 block-scaled matmul for Gemma 4 MoE experts on GB10.

Exposes :func:`nvfp4_linear`, a drop-in for one expert's gate_up_proj or
down_proj.  Keeps FP4 weights in GPU memory (no dequant-to-FP8 round
trip) by routing through Triton's ``tl.dot_scaled`` primitive.

Critical env var for sm_121 (GB10)::

    TRITON_OVERRIDE_ARCH=sm120

Triton's ``ScaledBlockedToMMA`` MLIR pattern hard-rejects
``computeCapability != 120``, so on sm_121 the default compile falls
through to a slow scaled-dot decomposition.  The override makes Triton
treat the chip as sm_120 for MLIR-pass purposes; consumer-Blackwell
sm_120/sm_121 share the same ``mma.sync...kind::mxf4nvf4`` encoding,
so the resulting PTX loads and runs on sm_121.  See
``kb/nvfp4-blackwell-research.md`` "second deep dive" for the full
story.

Tile budget is sized for GB10's 99 KiB smem/SM (vs 228 KiB on B200):
default ``BLOCK_M=BLOCK_N=128, BLOCK_K=128, NUM_STAGES=2``.  Small-M
(decode) lowers BLOCK_M to match the token count; prefill uses larger
tiles.

Two-level NVFP4 scaling — we handle Level 2 in the epilogue:

* Level 1 (FP8 E4M3 per-16-element block) goes straight into
  ``tl.dot_scaled``.
* Level 2 (FP32 per-tensor) is folded as a single ``fmul`` on the
  accumulator before the bf16 down-cast.  Option B of the plan, see
  ``docs/plans/2026-04-16-native-nvfp4-matmul.md``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Kernel
# -----------------------------------------------------------------------------


@triton.jit
def _swap_nibbles(x):
    """Our NVFP4 checkpoint packs (k0 → high, k1 → low) per byte, but
    Triton's ``tl.dot_scaled`` (and NVIDIA's FP4 MMA hardware) expects
    the opposite: (k0 → low, k1 → high).  Three-op fix: mask, shift,
    OR.  Negligible vs the MMA.

    See ``kb/nvfp4-blackwell-research.md`` and the offline validation
    in ``examples/test_nvfp4_kernel.py`` for the full story.
    """
    return ((x & 0xF) << 4) | ((x >> 4) & 0xF)


@triton.jit
def _nvfp4_matmul_kernel(
    a_ptr, b_ptr,
    a_scale_ptr, b_scale_ptr,
    c_ptr,
    alpha,  # fp32 scalar — folded a_tensor_scale * b_tensor_scale
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_am_scale, stride_ak_scale,
    stride_bn_scale, stride_bk_scale,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    ELEM_PER_BYTE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Block-scaled NVFP4 matmul kernel.

    Shapes in bytes (packed):
      a_ptr:       (M, K // ELEM_PER_BYTE) uint8
      b_ptr:       (N, K // ELEM_PER_BYTE) uint8
      a_scale_ptr: (M, K // VEC_SIZE) float8_e4m3fn
      b_scale_ptr: (N, K // VEC_SIZE) float8_e4m3fn
      c_ptr:       (M, N) bfloat16

    Shapes as the kernel sees them post-fp4x2-expansion (handled by
    `tl.dot_scaled`): a is (BLOCK_M, BLOCK_K) e2m1, b is (BLOCK_N,
    BLOCK_K) e2m1, but stored as (BLOCK_M/N, BLOCK_K // 2) uint8.

    Both a and b are nibble-swapped after load to match Triton's
    (low=k0, high=k1) packing convention.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_k_packed_base = tl.arange(0, BLOCK_K // ELEM_PER_BYTE)
    offs_k_scale_base = tl.arange(0, BLOCK_K // VEC_SIZE)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # No K-masking: the launcher requires K % BLOCK_K == 0.  No M/N
    # masking either: the launcher pads M and N to multiples of
    # BLOCK_M / BLOCK_N and slices the output after.  This keeps the
    # inner loop free of int→fp8 casts that Triton can't handle.
    for k_block in tl.range(0, K // BLOCK_K, num_stages=NUM_STAGES):
        k_packed_start = k_block * (BLOCK_K // ELEM_PER_BYTE)
        k_scale_start = k_block * (BLOCK_K // VEC_SIZE)

        a_ptrs = (a_ptr
                  + offs_m[:, None] * stride_am
                  + (k_packed_start + offs_k_packed_base)[None, :] * stride_ak)
        b_ptrs = (b_ptr
                  + offs_n[:, None] * stride_bn
                  + (k_packed_start + offs_k_packed_base)[None, :] * stride_bk)
        a_scale_ptrs = (a_scale_ptr
                        + offs_m[:, None] * stride_am_scale
                        + (k_scale_start + offs_k_scale_base)[None, :] * stride_ak_scale)
        b_scale_ptrs = (b_scale_ptr
                        + offs_n[:, None] * stride_bn_scale
                        + (k_scale_start + offs_k_scale_base)[None, :] * stride_bk_scale)

        a_raw = tl.load(a_ptrs)
        b_raw = tl.load(b_ptrs)
        a = _swap_nibbles(a_raw)
        b = _swap_nibbles(b_raw)
        a_scale = tl.load(a_scale_ptrs)
        b_scale = tl.load(b_scale_ptrs)

        acc = tl.dot_scaled(a, a_scale, "e2m1", b.T, b_scale, "e2m1", acc)

    # Level-2 per-tensor scale folded here (Option B epilogue)
    acc = acc * alpha

    c_ptrs = (c_ptr
              + offs_m[:, None] * stride_cm
              + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


# -----------------------------------------------------------------------------
# Python entry point
# -----------------------------------------------------------------------------


FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
NVFP4_BLOCK_SIZE = 16


def _quantize_bf16_activation_to_nvfp4(
    x_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row dynamic NVFP4 quantization of bf16 activations.

    Matches the weight-side checkpoint layout: packed uint8 + fp8_e4m3
    block scales + fp32 per-tensor scale.

    Args:
        x_bf16: (M, K) bf16.

    Returns:
        (packed_uint8, block_scale_fp8, tensor_scale_fp32).
    """
    from aeo_quant.gpu.quant import quantize_2d_to_nvfp4

    return quantize_2d_to_nvfp4(x_bf16)


def nvfp4_linear(
    x_bf16: torch.Tensor,
    w_packed: torch.Tensor,
    w_block_scale: torch.Tensor,
    w_tensor_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """One expert's NVFP4 matmul: (..., K) bf16 input × (N, K) NVFP4 weight.

    Flattens leading batch dims, quantizes activations to NVFP4 at
    runtime (same 2-level layout as the weight), runs the block-scaled
    matmul, folds the per-tensor scales in the epilogue.

    Args:
        x_bf16:        (..., K) bf16 activation.
        w_packed:      (N, K // 2) uint8.  Two FP4 nibbles per byte.
        w_block_scale: (N, K // 16) float8_e4m3fn.
        w_tensor_scale: fp32 scalar (0-d tensor).
        out_dtype: defaults to bf16; fp32 also supported.

    Returns:
        (..., N) of ``out_dtype``.
    """
    orig_shape = x_bf16.shape
    if x_bf16.ndim != 2:
        x_bf16 = x_bf16.reshape(-1, orig_shape[-1])

    M, K = x_bf16.shape
    N = w_packed.shape[0]
    assert w_packed.shape[1] == K // 2, (
        f"K mismatch: input K={K} expects w_packed[:,:{K//2}], got {w_packed.shape}"
    )
    assert w_block_scale.shape == (N, K // NVFP4_BLOCK_SIZE), (
        f"w_block_scale shape: expected ({N}, {K // NVFP4_BLOCK_SIZE}), "
        f"got {tuple(w_block_scale.shape)}"
    )

    # Activation quant (per-row dynamic)
    a_packed, a_block_scale, a_tensor_scale = _quantize_bf16_activation_to_nvfp4(x_bf16)

    # Fold the two tensor scales into a single epilogue fmul.
    alpha = (a_tensor_scale.float() * w_tensor_scale.float()).item()

    c = torch.empty((M, N), dtype=out_dtype, device=x_bf16.device)

    # Tile selection.  Decode path (M small) uses a narrow BLOCK_M so we
    # don't waste tensor-core lanes on zero-padded rows.
    if M <= 16:
        BLOCK_M = 16
    elif M <= 32:
        BLOCK_M = 32
    elif M <= 64:
        BLOCK_M = 64
    else:
        BLOCK_M = 128
    BLOCK_N = 128
    # BLOCK_K=64 divides the Gemma 4 K=2880 hidden dim cleanly (45 iters).
    # The native m16n8k64 MMA consumes K=64 per op, so one MMA per iter.
    BLOCK_K = 64
    NUM_STAGES = 2

    # Pad M to a multiple of BLOCK_M if needed.  N and K must be
    # divisible — raise if they aren't (handling that in the kernel
    # requires masked loads that Triton can't cast through fp8 type).
    if N % BLOCK_N != 0:
        raise ValueError(
            f"N={N} must be divisible by BLOCK_N={BLOCK_N}. "
            f"Add padding upstream if needed."
        )
    if K % BLOCK_K != 0:
        raise ValueError(
            f"K={K} must be divisible by BLOCK_K={BLOCK_K}. "
            f"Add padding upstream if needed."
        )

    M_padded = ((M + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    if M_padded != M:
        # Pad the activation + its quantized form to M_padded rows.
        a_packed_padded = torch.zeros(
            M_padded, a_packed.shape[1],
            dtype=a_packed.dtype, device=a_packed.device,
        )
        a_packed_padded[:M] = a_packed
        a_block_scale_padded = torch.zeros(
            M_padded, a_block_scale.shape[1],
            dtype=a_block_scale.dtype, device=a_block_scale.device,
        )
        a_block_scale_padded[:M] = a_block_scale
        a_packed = a_packed_padded
        a_block_scale = a_block_scale_padded
        c_padded = torch.empty(
            (M_padded, N), dtype=out_dtype, device=x_bf16.device,
        )
    else:
        c_padded = c

    grid = (triton.cdiv(M_padded, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _nvfp4_matmul_kernel[grid](
        a_packed, w_packed,
        a_block_scale, w_block_scale,
        c_padded,
        alpha,
        M_padded, N, K,
        a_packed.stride(0), a_packed.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        a_block_scale.stride(0), a_block_scale.stride(1),
        w_block_scale.stride(0), w_block_scale.stride(1),
        c_padded.stride(0), c_padded.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        VEC_SIZE=NVFP4_BLOCK_SIZE,
        ELEM_PER_BYTE=2,
        NUM_STAGES=NUM_STAGES,
    )

    if M_padded != M:
        c = c_padded[:M]
    else:
        c = c_padded

    if out_dtype == torch.float32:
        c = c.float()
    if len(orig_shape) > 2:
        c = c.reshape(*orig_shape[:-1], N)
    return c
