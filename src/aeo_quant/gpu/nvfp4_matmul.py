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

    for k_block in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
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

        mask_m = offs_m[:, None] < M
        mask_n = offs_n[:, None] < N
        # Packed K bound relative to K // ELEM_PER_BYTE
        k_packed_mask = (k_packed_start + offs_k_packed_base)[None, :] < (K // ELEM_PER_BYTE)
        k_scale_mask = (k_scale_start + offs_k_scale_base)[None, :] < (K // VEC_SIZE)

        a = tl.load(a_ptrs, mask=mask_m & k_packed_mask, other=0)
        b = tl.load(b_ptrs, mask=mask_n & k_packed_mask, other=0)
        a_scale = tl.load(a_scale_ptrs, mask=mask_m & k_scale_mask, other=0)
        b_scale = tl.load(b_scale_ptrs, mask=mask_n & k_scale_mask, other=0)

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
    BLOCK_K = 128
    NUM_STAGES = 2

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _nvfp4_matmul_kernel[grid](
        a_packed, w_packed,
        a_block_scale, w_block_scale,
        c,
        alpha,
        M, N, K,
        a_packed.stride(0), a_packed.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        a_block_scale.stride(0), a_block_scale.stride(1),
        w_block_scale.stride(0), w_block_scale.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        VEC_SIZE=NVFP4_BLOCK_SIZE,
        ELEM_PER_BYTE=2,
        NUM_STAGES=NUM_STAGES,
    )

    if out_dtype == torch.float32:
        c = c.float()
    if len(orig_shape) > 2:
        c = c.reshape(*orig_shape[:-1], N)
    return c
