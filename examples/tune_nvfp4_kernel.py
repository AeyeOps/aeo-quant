#!/usr/bin/env python3
"""Hand-tuning probe for the NVFP4 kernel.

Sweeps BLOCK_M × BLOCK_N × NUM_STAGES × num_warps at the Gemma 4
expert prefill shape and reports TFLOPS for each combination.  Run
with ``TRITON_OVERRIDE_ARCH=sm120`` to get the native FP4 MMA path.

Usage::

    TRITON_OVERRIDE_ARCH=sm120 uv run python examples/tune_nvfp4_kernel.py

Output is sorted by TFLOPS.  Best config is a candidate for the
launcher's default tile selection.
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _swap_nibbles(x):
    return ((x & 0xF) << 4) | ((x >> 4) & 0xF)


@triton.jit
def _nvfp4_matmul_tuned_kernel(
    a_ptr, b_ptr,
    a_scale_ptr, b_scale_ptr,
    c_ptr, alpha,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_am_scale, stride_ak_scale,
    stride_bn_scale, stride_bk_scale,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    VEC_SIZE: tl.constexpr, ELEM_PER_BYTE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_packed = tl.arange(0, BLOCK_K // ELEM_PER_BYTE)
    offs_k_scale = tl.arange(0, BLOCK_K // VEC_SIZE)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in tl.range(0, K // BLOCK_K, num_stages=NUM_STAGES):
        kp_start = k_block * (BLOCK_K // ELEM_PER_BYTE)
        ks_start = k_block * (BLOCK_K // VEC_SIZE)
        a_ptrs = (a_ptr + offs_m[:, None] * stride_am
                  + (kp_start + offs_k_packed)[None, :] * stride_ak)
        b_ptrs = (b_ptr + offs_n[:, None] * stride_bn
                  + (kp_start + offs_k_packed)[None, :] * stride_bk)
        as_ptrs = (a_scale_ptr + offs_m[:, None] * stride_am_scale
                   + (ks_start + offs_k_scale)[None, :] * stride_ak_scale)
        bs_ptrs = (b_scale_ptr + offs_n[:, None] * stride_bn_scale
                   + (ks_start + offs_k_scale)[None, :] * stride_bk_scale)

        a_raw = tl.load(a_ptrs)
        b_raw = tl.load(b_ptrs)
        a = _swap_nibbles(a_raw)
        b = _swap_nibbles(b_raw)
        a_scale = tl.load(as_ptrs)
        b_scale = tl.load(bs_ptrs)

        acc = tl.dot_scaled(a, a_scale, "e2m1", b.T, b_scale, "e2m1", acc)

    acc = acc * alpha
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16))


def _time_config(
    a_packed, b_packed, a_scale, b_scale, alpha,
    M, N, K,
    BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, num_warps,
    iters=10,
):
    """Time one config.  Returns (elapsed_ms_mean, tflops) or None on fail."""
    c = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    try:
        # Warmup
        _nvfp4_matmul_tuned_kernel[grid](
            a_packed, b_packed, a_scale, b_scale, c, alpha,
            M, N, K,
            a_packed.stride(0), a_packed.stride(1),
            b_packed.stride(0), b_packed.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            b_scale.stride(0), b_scale.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            VEC_SIZE=16, ELEM_PER_BYTE=2, NUM_STAGES=NUM_STAGES,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()

        t0 = time.monotonic()
        for _ in range(iters):
            _nvfp4_matmul_tuned_kernel[grid](
                a_packed, b_packed, a_scale, b_scale, c, alpha,
                M, N, K,
                a_packed.stride(0), a_packed.stride(1),
                b_packed.stride(0), b_packed.stride(1),
                a_scale.stride(0), a_scale.stride(1),
                b_scale.stride(0), b_scale.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                VEC_SIZE=16, ELEM_PER_BYTE=2, NUM_STAGES=NUM_STAGES,
                num_warps=num_warps,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.monotonic() - t0) * 1000 / iters
        flops = 2.0 * M * N * K
        tflops = flops / (elapsed_ms * 1e-3) / 1e12
        return elapsed_ms, tflops
    except Exception as e:
        return None


def main() -> int:
    from aeo_quant.gpu.quant import quantize_2d_to_nvfp4
    override = os.environ.get("TRITON_OVERRIDE_ARCH", "")
    print(f"TRITON_OVERRIDE_ARCH = {override or '(unset)'}")

    # Gemma 4 26B-A4B real dims (gate_up_proj)
    M = 1024    # prefill-ish
    K = 2816
    N = 1408

    torch.manual_seed(0)
    a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    a_packed, a_scale, a_ts = quantize_2d_to_nvfp4(a_bf16)
    b_packed, b_scale, b_ts = quantize_2d_to_nvfp4(b_bf16)
    alpha = (a_ts.float() * b_ts.float()).item()

    print(f"\n--- tuning at M={M} K={K} N={N} ---\n")
    print(f"{'BLOCK_M':>8}{'BLOCK_N':>8}{'BLOCK_K':>8}{'NUM_ST':>8}{'warps':>7}"
          f"{'ms':>10}{'TFLOPS':>10}")
    print("-" * 60)

    results = []
    block_m_choices = [64, 128, 256]
    block_n_choices = [64, 128, 256]
    block_k_choices = [64, 128]
    stages_choices = [2, 3]
    warps_choices = [4, 8]

    for bm, bn, bk, ns, nw in itertools.product(
        block_m_choices, block_n_choices, block_k_choices,
        stages_choices, warps_choices,
    ):
        # Skip configs that would hit smem limits or misalign
        if M % bm != 0 or N % bn != 0 or K % bk != 0:
            continue
        if bn > 256 and bk > 128:
            continue
        r = _time_config(
            a_packed, b_packed, a_scale, b_scale, alpha,
            M, N, K, bm, bn, bk, ns, nw,
        )
        if r is None:
            print(f"{bm:>8}{bn:>8}{bk:>8}{ns:>8}{nw:>7}{'FAIL':>10}")
            continue
        ms, tf = r
        results.append((tf, ms, bm, bn, bk, ns, nw))
        print(f"{bm:>8}{bn:>8}{bk:>8}{ns:>8}{nw:>7}{ms:>10.3f}{tf:>10.2f}")

    print()
    print("=== top 5 by TFLOPS ===")
    results.sort(reverse=True)
    for tf, ms, bm, bn, bk, ns, nw in results[:5]:
        print(f"  BLOCK_M={bm} BLOCK_N={bn} BLOCK_K={bk} "
              f"NUM_STAGES={ns} num_warps={nw}: "
              f"{ms:.3f}ms → {tf:.2f} TFLOPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
