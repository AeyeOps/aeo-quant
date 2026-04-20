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
so the resulting PTX loads and runs on sm_121.

Tile budget is sized for GB10's 99 KiB smem/SM (vs 228 KiB on B200):
default ``BLOCK_M=BLOCK_N=128, BLOCK_K=128, NUM_STAGES=2``.  Small-M
(decode) lowers BLOCK_M to match the token count; prefill uses larger
tiles.

Two-level NVFP4 scaling — Level 2 is handled in the epilogue:

* Level 1 (FP8 E4M3 per-16-element block) goes straight into
  ``tl.dot_scaled``.
* Level 2 (FP32 per-tensor) is folded as a single ``fmul`` on the
  accumulator before the bf16 down-cast.
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
    """
    return ((x & 0xF) << 4) | ((x >> 4) & 0xF)


@triton.jit
def _nvfp4_matmul_kernel_3d_gather(
    a_ptr, b_ptr,
    a_scale_ptr, b_scale_ptr,
    c_ptr,
    alpha_ptr,  # fp32 pointer — folded a_tensor_scale * w_tensor_scale, loaded in-kernel
    expert_ids_ptr,  # int32 (K_SELECTED,) — maps grid pid_e → real expert index in [0, E_FULL)
    M, N, K,
    stride_ae, stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_ae_scale, stride_am_scale, stride_ak_scale,
    stride_be_scale, stride_bn_scale, stride_bk_scale,
    stride_ce, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    ELEM_PER_BYTE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Gather variant of :func:`_nvfp4_matmul_kernel_3d`.

    Identical math to the non-gather kernel. Difference: the weight
    pointers (``b``, ``b_scale``) are offset by a runtime indirection
    ``expert_ids[pid_e]`` instead of ``pid_e`` itself, so the launcher
    can pass the full ``(E_FULL, N, K//EPB)`` weight tensor unmutated
    and let the kernel pick out the 0..K_SELECTED-1 slice via the
    expert_ids table. Saves the per-step ``index_select`` allocation +
    copy on decode.

    Activation and output offsets still use ``pid_e`` (the activation
    is index-aligned with the *gathered* set, and the output is written
    into positions 0..K_SELECTED-1 on the c axis), so the only pointer
    indirection is on the weight side.
    """
    pid_mn = tl.program_id(axis=0)
    pid_e = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid_mn % num_pid_m
    pid_n = pid_mn // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_k_packed_base = tl.arange(0, BLOCK_K // ELEM_PER_BYTE)
    offs_k_scale_base = tl.arange(0, BLOCK_K // VEC_SIZE)

    real_expert = tl.load(expert_ids_ptr + pid_e)

    a_expert_base = a_ptr + pid_e * stride_ae
    a_scale_expert_base = a_scale_ptr + pid_e * stride_ae_scale
    b_expert_base = b_ptr + real_expert * stride_be
    b_scale_expert_base = b_scale_ptr + real_expert * stride_be_scale

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in tl.range(0, K // BLOCK_K, num_stages=NUM_STAGES):
        k_packed_start = k_block * (BLOCK_K // ELEM_PER_BYTE)
        k_scale_start = k_block * (BLOCK_K // VEC_SIZE)

        a_ptrs = (a_expert_base
                  + offs_m[:, None] * stride_am
                  + (k_packed_start + offs_k_packed_base)[None, :] * stride_ak)
        b_ptrs = (b_expert_base
                  + offs_n[:, None] * stride_bn
                  + (k_packed_start + offs_k_packed_base)[None, :] * stride_bk)
        a_scale_ptrs = (a_scale_expert_base
                        + offs_m[:, None] * stride_am_scale
                        + (k_scale_start + offs_k_scale_base)[None, :] * stride_ak_scale)
        b_scale_ptrs = (b_scale_expert_base
                        + offs_n[:, None] * stride_bn_scale
                        + (k_scale_start + offs_k_scale_base)[None, :] * stride_bk_scale)

        a_raw = tl.load(a_ptrs)
        b_raw = tl.load(b_ptrs)
        a = _swap_nibbles(a_raw)
        b = _swap_nibbles(b_raw)
        a_scale = tl.load(a_scale_ptrs)
        b_scale = tl.load(b_scale_ptrs)

        acc = tl.dot_scaled(a, a_scale, "e2m1", b.T, b_scale, "e2m1", acc)

    alpha = tl.load(alpha_ptr)
    acc = acc * alpha

    c_ptrs = (c_ptr
              + pid_e * stride_ce
              + offs_m[:, None] * stride_cm
              + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def _nvfp4_matmul_kernel_3d(
    a_ptr, b_ptr,
    a_scale_ptr, b_scale_ptr,
    c_ptr,
    alpha_ptr,  # fp32 pointer — folded a_tensor_scale * w_tensor_scale, loaded in-kernel
    M, N, K,
    stride_ae, stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_ae_scale, stride_am_scale, stride_ak_scale,
    stride_be_scale, stride_bn_scale, stride_bk_scale,
    stride_ce, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    ELEM_PER_BYTE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Per-expert batched block-scaled NVFP4 matmul.

    Same body as :func:`_nvfp4_matmul_kernel` but with an additional
    ``pid_e`` program axis that offsets the a, b, and c pointers. The
    activation offset uses ``stride_ae`` — pass 0 for a shared
    activation (gate_up decode: all experts see the same token row),
    or a real stride for per-expert activation (down decode: each
    expert's own gate*up output).

    Shapes (either case):
      a:       (E, M, K/ELEM_PER_BYTE) uint8 — or (1, M, K/EPB) expanded
      b:       (E, N, K/ELEM_PER_BYTE) uint8
      a_scale: (E, M, K/VEC_SIZE) fp8_e4m3fn
      b_scale: (E, N, K/VEC_SIZE) fp8_e4m3fn
      c:       (E, M, N) bf16

    Grid: ``(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), E)``.
    Alpha is a scalar — correct when w_tensor_scale is shared across
    experts (as in the Gemma 4 NVFP4 checkpoint: *_scale_2 is a single
    scalar per projection). For per-expert activation, the global
    a_tensor_scale is computed over the whole (E, M, K) tensor; the
    parity drift vs per-expert quantization is < 2%.
    """
    pid_mn = tl.program_id(axis=0)
    pid_e = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid_mn % num_pid_m
    pid_n = pid_mn // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_k_packed_base = tl.arange(0, BLOCK_K // ELEM_PER_BYTE)
    offs_k_scale_base = tl.arange(0, BLOCK_K // VEC_SIZE)

    # Expert base offsets (stride_ae = 0 gives shared activation across experts)
    a_expert_base = a_ptr + pid_e * stride_ae
    a_scale_expert_base = a_scale_ptr + pid_e * stride_ae_scale
    b_expert_base = b_ptr + pid_e * stride_be
    b_scale_expert_base = b_scale_ptr + pid_e * stride_be_scale

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in tl.range(0, K // BLOCK_K, num_stages=NUM_STAGES):
        k_packed_start = k_block * (BLOCK_K // ELEM_PER_BYTE)
        k_scale_start = k_block * (BLOCK_K // VEC_SIZE)

        a_ptrs = (a_expert_base
                  + offs_m[:, None] * stride_am
                  + (k_packed_start + offs_k_packed_base)[None, :] * stride_ak)
        b_ptrs = (b_expert_base
                  + offs_n[:, None] * stride_bn
                  + (k_packed_start + offs_k_packed_base)[None, :] * stride_bk)
        a_scale_ptrs = (a_scale_expert_base
                        + offs_m[:, None] * stride_am_scale
                        + (k_scale_start + offs_k_scale_base)[None, :] * stride_ak_scale)
        b_scale_ptrs = (b_scale_expert_base
                        + offs_n[:, None] * stride_bn_scale
                        + (k_scale_start + offs_k_scale_base)[None, :] * stride_bk_scale)

        a_raw = tl.load(a_ptrs)
        b_raw = tl.load(b_ptrs)
        a = _swap_nibbles(a_raw)
        b = _swap_nibbles(b_raw)
        a_scale = tl.load(a_scale_ptrs)
        b_scale = tl.load(b_scale_ptrs)

        acc = tl.dot_scaled(a, a_scale, "e2m1", b.T, b_scale, "e2m1", acc)

    alpha = tl.load(alpha_ptr)
    acc = acc * alpha

    c_ptrs = (c_ptr
              + pid_e * stride_ce
              + offs_m[:, None] * stride_cm
              + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def _nvfp4_matmul_kernel(
    a_ptr, b_ptr,
    a_scale_ptr, b_scale_ptr,
    c_ptr,
    alpha_ptr,  # fp32 pointer — folded a_tensor_scale * b_tensor_scale, loaded in-kernel
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

    # Level-2 per-tensor scale folded here (Option B epilogue). Loaded
    # from device memory so the Python launcher doesn't need .item() —
    # that host sync would break CUDA graph capture.
    alpha = tl.load(alpha_ptr)
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

    # Activation quant (per-row dynamic)
    a_packed, a_block_scale, a_tensor_scale = _quantize_bf16_activation_to_nvfp4(x_bf16)

    out = nvfp4_linear_prequantized(
        a_packed, a_block_scale, a_tensor_scale,
        w_packed, w_block_scale, w_tensor_scale,
        out_dtype=out_dtype,
    )

    if len(orig_shape) > 2:
        out = out.reshape(*orig_shape[:-1], out.shape[-1])
    return out


def nvfp4_linear_prequantized(
    a_packed: torch.Tensor,
    a_block_scale: torch.Tensor,
    a_tensor_scale: torch.Tensor,
    w_packed: torch.Tensor,
    w_block_scale: torch.Tensor,
    w_tensor_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Same as :func:`nvfp4_linear` but the activation is already quantized.

    Useful when the same activation is reused across multiple experts
    (MoE): quantize the full ``hidden_states`` once at the top of the
    layer's forward, then slice the packed form per expert.  Avoids
    the ~10 small elementwise kernel launches that
    ``_quantize_bf16_activation_to_nvfp4`` does per call.

    Args:
        a_packed:        (M, K // 2) uint8 — packed activation nibbles.
        a_block_scale:   (M, K // 16) fp8_e4m3fn — level-1 scales.
        a_tensor_scale:  fp32 scalar.
        w_packed:        (N, K // 2) uint8.
        w_block_scale:   (N, K // 16) fp8_e4m3fn.
        w_tensor_scale:  fp32 scalar.
        out_dtype:       defaults to bf16.

    Returns:
        (M, N) of ``out_dtype``.
    """
    M = a_packed.shape[0]
    K_half = a_packed.shape[1]
    K = K_half * 2
    N = w_packed.shape[0]

    # Fold the two tensor scales into a single epilogue fmul. Kept as a
    # 0-D device tensor rather than a Python float (.item() would force
    # a host sync per matmul, making the kernel launch uncapturable).
    alpha = (a_tensor_scale.float() * w_tensor_scale.float())

    c = torch.empty((M, N), dtype=out_dtype, device=a_packed.device)

    # Tile selection.  Decode path (M small) lowers BLOCK_M so we don't
    # waste tensor-core lanes on zero-padded rows.
    #
    # NUM_STAGES=3 + num_warps=4 hits 74 TFLOPS at M=1024, K=2816, N=1408,
    # but at M=2880, N=5760 (many programs × ~SMs), NUM_STAGES=3 hurts
    # due to occupancy.  Keep 2 as the safe default; small shapes stay
    # fast even without the extra stage.  Larger prefill shapes would
    # benefit from per-shape autotune once we have the workload data.
    if M <= 16:
        BLOCK_M = 16
    elif M <= 32:
        BLOCK_M = 32
    elif M <= 64:
        BLOCK_M = 64
    else:
        BLOCK_M = 128
    BLOCK_N = 128
    # Prefer BLOCK_K=128 when it divides K (max throughput); fall back to
    # 64 otherwise.  Native MMA K=64 so either maps cleanly.
    BLOCK_K = 128 if K % 128 == 0 else 64
    NUM_STAGES = 2
    NUM_WARPS = 4

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
            (M_padded, N), dtype=out_dtype, device=a_packed.device,
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
        num_warps=NUM_WARPS,
    )

    if M_padded != M:
        c = c_padded[:M]
    else:
        c = c_padded
    return c


def nvfp4_linear_3d_prequantized(
    a_packed: torch.Tensor,
    a_block_scale: torch.Tensor,
    a_tensor_scale: torch.Tensor,
    w_packed: torch.Tensor,
    w_block_scale: torch.Tensor,
    w_tensor_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Batched per-expert NVFP4 matmul for MoE decode.

    Two activation layouts are supported, distinguished by ``ndim``:

    * **Shared (2D)** — ``a_packed`` shape ``(M, K // 2)``. All E experts
      multiply against the same activation rows. Used for gate_up in
      decode where all k experts see the single token's hidden_states.
    * **Per-expert (3D)** — ``a_packed`` shape ``(E, M, K // 2)``. Each
      expert multiplies against its own activation slice. Used for
      down_proj in decode where each expert's gate*up output differs.

    Args:
        a_packed:        ``(M, K // 2)`` or ``(E, M, K // 2)`` uint8.
        a_block_scale:   matching shape with ``K // 16`` fp8_e4m3fn.
        a_tensor_scale:  fp32 scalar.
        w_packed:        ``(E, N, K // 2)`` uint8 — per-expert weights.
        w_block_scale:   ``(E, N, K // 16)`` fp8_e4m3fn.
        w_tensor_scale:  fp32 scalar — shared across experts.
        out_dtype:       bf16 (default) or fp32.

    Returns:
        ``(E, M, N)`` of ``out_dtype``.
    """
    if w_packed.ndim != 3:
        raise ValueError(
            f"w_packed must be 3D (E, N, K//2); got shape {tuple(w_packed.shape)}"
        )
    E, N, K_half_w = w_packed.shape

    # Detect activation layout and canonicalize to 3D (E, M, K/2).
    if a_packed.ndim == 2:
        # Shared activation — add a leading E=1 dim via unsqueeze; the
        # kernel sees stride_ae=0 because the tensor is broadcast-shaped.
        M, K_half_a = a_packed.shape
        a_packed_3d = a_packed.unsqueeze(0)
        a_block_scale_3d = a_block_scale.unsqueeze(0)
        # Force stride_ae=0 so the kernel reads the same row for every
        # expert regardless of broadcasting semantics.
        stride_ae = 0
        stride_ae_scale = 0
        stride_am = a_packed_3d.stride(1)
        stride_ak = a_packed_3d.stride(2)
        stride_am_scale = a_block_scale_3d.stride(1)
        stride_ak_scale = a_block_scale_3d.stride(2)
    elif a_packed.ndim == 3:
        E_a, M, K_half_a = a_packed.shape
        if E_a != E:
            raise ValueError(
                f"per-expert activation E={E_a} must equal weight E={E}"
            )
        stride_ae = a_packed.stride(0)
        stride_am = a_packed.stride(1)
        stride_ak = a_packed.stride(2)
        stride_ae_scale = a_block_scale.stride(0)
        stride_am_scale = a_block_scale.stride(1)
        stride_ak_scale = a_block_scale.stride(2)
        a_packed_3d = a_packed
        a_block_scale_3d = a_block_scale
    else:
        raise ValueError(
            f"a_packed must be 2D (shared) or 3D (per-expert); "
            f"got shape {tuple(a_packed.shape)}"
        )

    if K_half_a != K_half_w:
        raise ValueError(
            f"activation K//2={K_half_a} must equal weight K//2={K_half_w}"
        )
    K = K_half_a * 2

    # 0-D device tensor (see the 2D launcher for why we don't .item() it).
    alpha = (a_tensor_scale.float() * w_tensor_scale.float())

    if M <= 16:
        BLOCK_M = 16
    elif M <= 32:
        BLOCK_M = 32
    elif M <= 64:
        BLOCK_M = 64
    else:
        BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128 if K % 128 == 0 else 64
    NUM_STAGES = 2
    NUM_WARPS = 4

    if N % BLOCK_N != 0:
        raise ValueError(
            f"N={N} must be divisible by BLOCK_N={BLOCK_N}. Pad upstream."
        )
    if K % BLOCK_K != 0:
        raise ValueError(
            f"K={K} must be divisible by BLOCK_K={BLOCK_K}. Pad upstream."
        )

    M_padded = ((M + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    if M_padded != M:
        # Grow the M axis of the canonicalized 3D activation tensor. For
        # the shared path the leading dim stays 1 (stride_ae=0 preserved);
        # for the per-expert path it stays E.
        pad_E = a_packed_3d.shape[0]
        a_packed_padded = torch.zeros(
            pad_E, M_padded, a_packed_3d.shape[2],
            dtype=a_packed_3d.dtype, device=a_packed_3d.device,
        )
        a_packed_padded[:, :M, :] = a_packed_3d
        a_block_scale_padded = torch.zeros(
            pad_E, M_padded, a_block_scale_3d.shape[2],
            dtype=a_block_scale_3d.dtype, device=a_block_scale_3d.device,
        )
        a_block_scale_padded[:, :M, :] = a_block_scale_3d
        a_packed_3d = a_packed_padded
        a_block_scale_3d = a_block_scale_padded
        # Only update stride_ae/stride_ae_scale if this is the per-expert
        # path; keep stride_ae=0 for the shared path (the canonicalized
        # tensor has leading dim 1, so its natural stride(0) would give
        # the right offset only if we actually want per-expert data).
        if stride_ae != 0:
            stride_ae = a_packed_3d.stride(0)
            stride_ae_scale = a_block_scale_3d.stride(0)
        stride_am = a_packed_3d.stride(1)
        stride_ak = a_packed_3d.stride(2)
        stride_am_scale = a_block_scale_3d.stride(1)
        stride_ak_scale = a_block_scale_3d.stride(2)
        c_padded = torch.empty(
            (E, M_padded, N), dtype=out_dtype, device=a_packed_3d.device,
        )
    else:
        c_padded = torch.empty(
            (E, M, N), dtype=out_dtype, device=a_packed_3d.device,
        )

    grid = (triton.cdiv(M_padded, BLOCK_M) * triton.cdiv(N, BLOCK_N), E)

    _nvfp4_matmul_kernel_3d[grid](
        a_packed_3d, w_packed,
        a_block_scale_3d, w_block_scale,
        c_padded,
        alpha,
        M_padded, N, K,
        stride_ae, stride_am, stride_ak,
        w_packed.stride(0), w_packed.stride(1), w_packed.stride(2),
        stride_ae_scale, stride_am_scale, stride_ak_scale,
        w_block_scale.stride(0), w_block_scale.stride(1), w_block_scale.stride(2),
        c_padded.stride(0), c_padded.stride(1), c_padded.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        VEC_SIZE=NVFP4_BLOCK_SIZE,
        ELEM_PER_BYTE=2,
        NUM_STAGES=NUM_STAGES,
        num_warps=NUM_WARPS,
    )

    if M_padded != M:
        return c_padded[:, :M, :]
    return c_padded


def nvfp4_linear_3d_gather(
    a_packed: torch.Tensor,
    a_block_scale: torch.Tensor,
    a_tensor_scale: torch.Tensor,
    w_packed: torch.Tensor,
    w_block_scale: torch.Tensor,
    w_tensor_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode-only batched NVFP4 matmul with in-kernel expert gather.

    Same math as :func:`nvfp4_linear_3d_prequantized`, same shape
    contracts for activations, but weights are passed as the full
    ``(E_FULL, N, K // 2)`` tensor and the kernel picks out each
    selected expert by indirecting through ``expert_ids``. Replaces
    the prior pattern of ``w_packed = w_full.index_select(0, ids)`` +
    `nvfp4_linear_3d_prequantized`, eliminating the Python-level gather
    allocation + copy on every decode step.

    The output shape is ``(K_SELECTED, M, N)`` where
    ``K_SELECTED = expert_ids.numel()`` — i.e. the same shape the
    non-gather launcher would produce if fed the pre-gathered
    ``(K_SELECTED, N, K // 2)`` weight tensor.

    Args:
        a_packed:        ``(M, K // 2)`` for shared activation or
                         ``(K_SELECTED, M, K // 2)`` for per-expert.
        a_block_scale:   matching shape with ``K // 16`` fp8_e4m3fn.
        a_tensor_scale:  fp32 scalar.
        w_packed:        ``(E_FULL, N, K // 2)`` uint8 — full weight
                         tensor, not index_selected.
        w_block_scale:   ``(E_FULL, N, K // 16)`` fp8_e4m3fn — full
                         block-scale tensor.
        w_tensor_scale:  fp32 scalar — shared across experts.
        expert_ids:      ``(K_SELECTED,)`` integer tensor naming the
                         real expert indices in ``[0, E_FULL)``.
                         Coerced to int32 on-device if not already.
        out_dtype:       bf16 (default) or fp32.

    Returns:
        ``(K_SELECTED, M, N)`` of ``out_dtype``.
    """
    if w_packed.ndim != 3:
        raise ValueError(
            f"w_packed must be 3D (E_FULL, N, K//2); "
            f"got shape {tuple(w_packed.shape)}"
        )
    E_full, N, K_half_w = w_packed.shape

    if expert_ids.ndim != 1:
        raise ValueError(
            f"expert_ids must be 1D; got shape {tuple(expert_ids.shape)}"
        )
    K_selected = expert_ids.shape[0]
    if expert_ids.dtype != torch.int32:
        expert_ids = expert_ids.to(torch.int32)
    if not expert_ids.is_contiguous():
        expert_ids = expert_ids.contiguous()

    # Detect activation layout and canonicalize to 3D (K_SELECTED, M, K/2).
    if a_packed.ndim == 2:
        M, K_half_a = a_packed.shape
        a_packed_3d = a_packed.unsqueeze(0)
        a_block_scale_3d = a_block_scale.unsqueeze(0)
        stride_ae = 0
        stride_ae_scale = 0
        stride_am = a_packed_3d.stride(1)
        stride_ak = a_packed_3d.stride(2)
        stride_am_scale = a_block_scale_3d.stride(1)
        stride_ak_scale = a_block_scale_3d.stride(2)
    elif a_packed.ndim == 3:
        E_a, M, K_half_a = a_packed.shape
        if E_a != K_selected:
            raise ValueError(
                f"per-expert activation E={E_a} must equal len(expert_ids)={K_selected}"
            )
        stride_ae = a_packed.stride(0)
        stride_am = a_packed.stride(1)
        stride_ak = a_packed.stride(2)
        stride_ae_scale = a_block_scale.stride(0)
        stride_am_scale = a_block_scale.stride(1)
        stride_ak_scale = a_block_scale.stride(2)
        a_packed_3d = a_packed
        a_block_scale_3d = a_block_scale
    else:
        raise ValueError(
            f"a_packed must be 2D (shared) or 3D (per-expert); "
            f"got shape {tuple(a_packed.shape)}"
        )

    if K_half_a != K_half_w:
        raise ValueError(
            f"activation K//2={K_half_a} must equal weight K//2={K_half_w}"
        )
    K = K_half_a * 2

    alpha = (a_tensor_scale.float() * w_tensor_scale.float())

    if M <= 16:
        BLOCK_M = 16
    elif M <= 32:
        BLOCK_M = 32
    elif M <= 64:
        BLOCK_M = 64
    else:
        BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128 if K % 128 == 0 else 64
    NUM_STAGES = 2
    NUM_WARPS = 4

    if N % BLOCK_N != 0:
        raise ValueError(
            f"N={N} must be divisible by BLOCK_N={BLOCK_N}. Pad upstream."
        )
    if K % BLOCK_K != 0:
        raise ValueError(
            f"K={K} must be divisible by BLOCK_K={BLOCK_K}. Pad upstream."
        )

    M_padded = ((M + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    if M_padded != M:
        pad_E = a_packed_3d.shape[0]
        a_packed_padded = torch.zeros(
            pad_E, M_padded, a_packed_3d.shape[2],
            dtype=a_packed_3d.dtype, device=a_packed_3d.device,
        )
        a_packed_padded[:, :M, :] = a_packed_3d
        a_block_scale_padded = torch.zeros(
            pad_E, M_padded, a_block_scale_3d.shape[2],
            dtype=a_block_scale_3d.dtype, device=a_block_scale_3d.device,
        )
        a_block_scale_padded[:, :M, :] = a_block_scale_3d
        a_packed_3d = a_packed_padded
        a_block_scale_3d = a_block_scale_padded
        if stride_ae != 0:
            stride_ae = a_packed_3d.stride(0)
            stride_ae_scale = a_block_scale_3d.stride(0)
        stride_am = a_packed_3d.stride(1)
        stride_ak = a_packed_3d.stride(2)
        stride_am_scale = a_block_scale_3d.stride(1)
        stride_ak_scale = a_block_scale_3d.stride(2)
        c_padded = torch.empty(
            (K_selected, M_padded, N), dtype=out_dtype, device=a_packed_3d.device,
        )
    else:
        c_padded = torch.empty(
            (K_selected, M, N), dtype=out_dtype, device=a_packed_3d.device,
        )

    grid = (triton.cdiv(M_padded, BLOCK_M) * triton.cdiv(N, BLOCK_N), K_selected)

    _nvfp4_matmul_kernel_3d_gather[grid](
        a_packed_3d, w_packed,
        a_block_scale_3d, w_block_scale,
        c_padded,
        alpha,
        expert_ids,
        M_padded, N, K,
        stride_ae, stride_am, stride_ak,
        w_packed.stride(0), w_packed.stride(1), w_packed.stride(2),
        stride_ae_scale, stride_am_scale, stride_ak_scale,
        w_block_scale.stride(0), w_block_scale.stride(1), w_block_scale.stride(2),
        c_padded.stride(0), c_padded.stride(1), c_padded.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        VEC_SIZE=NVFP4_BLOCK_SIZE,
        ELEM_PER_BYTE=2,
        NUM_STAGES=NUM_STAGES,
        num_warps=NUM_WARPS,
    )

    if M_padded != M:
        return c_padded[:, :M, :]
    return c_padded
