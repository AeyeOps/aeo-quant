"""Verbatim copy of Triton v3.6.0 block-scaled matmul tutorial.

Source: https://raw.githubusercontent.com/triton-lang/triton/v3.6.0/python/tutorials/10-block-scaled-matmul.py
Tag: v3.6.0
Fetched: 2026-04-17

Kept in-tree for reference while adapting for sm_121 (GB10). Do not
modify this file — edits go in src/aeo_quant/gpu/nvfp4_matmul.py. The
supports_block_scaling() gate at line ~139 hard-rejects cc_major == 12,
which is why this tutorial as-shipped exits immediately on GB10.

License: MIT (inherited from triton-lang/triton).

Key gates to relax when porting:
- supports_block_scaling() (line ~139): accept cc_major >= 10
  (or ignore entirely — the real compatibility is at the MLIR pass
  level, not Python).
- No other Python-side gate; the rest is a straight kernel body.

The real blocker is in Triton's C++: AccelerateMatmul.cpp line 665,
`if (computeCapability != 120) return failure();` in ScaledBlockedToMMA.
Try TRITON_OVERRIDE_ARCH=sm120 first — if it works, no source patch needed.
"""
"""
Block Scaled Matrix Multiplication
==================================
This tutorial demonstrates a Triton implementation of block scaled matrix multiplication
which is generic over FP4 and FP8 formats on NVIDIA and AMD GPUs.
The tutorial supports OCP microscaling formats such as mxfp4 and mxfp8, and NVIDIA's nvfp4
(on NVIDIA GPUs) and mxfp4 (on AMD GPUs). These matrix multiplications are hardware-accelerated
using fifth-generation Tensor Cores on NVIDIA GPUs with compute capability 10, and by the CDNA4
matrix cores on AMD GPUs.
"""

import argparse

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx950'


def supports_block_scaling():
    return (is_cuda() and torch.cuda.get_device_capability()[0] == 10) or is_hip_cdna4()


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 2:
            kernel_name += "_mixed"
        elif args["ELEM_PER_BYTE_A"] == 2 and args["ELEM_PER_BYTE_B"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2.0 * M * N * K
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(
        a_desc, a_scale_desc, b_desc, b_scale_desc, c_desc,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        output_type: tl.constexpr,
        ELEM_PER_BYTE_A: tl.constexpr, ELEM_PER_BYTE_B: tl.constexpr,
        VEC_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        rep_m: tl.constexpr, rep_n: tl.constexpr, rep_k: tl.constexpr,
        NUM_STAGES: tl.constexpr):
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        if MIXED_PREC:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


def block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, dtype_dst, M, N, K, rep_m, rep_n, rep_k, configs):
    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")

    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel[grid](
        a_desc, a_scale_desc, b_desc, b_scale_desc, c_desc,
        M, N, K, dtype_dst,
        configs["ELEM_PER_BYTE_A"], configs["ELEM_PER_BYTE_B"],
        configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"], configs["BLOCK_SIZE_K"],
        rep_m, rep_n, rep_k,
        configs["num_stages"],
    )
    return output


def initialize_block_scaled(M, N, K, block_scale_type="nvfp4", compute_reference=False):
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    assert block_scale_type in ["nvfp4", "mxfp4", "mxfp8", "mixed"]
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2

    device = "cuda"
    a_ref = MXFP4Tensor(size=(M, K), device=device).random()
    b_ref = MXFP4Tensor(size=(N, K), device=device).random()
    if block_scale_type in ["mxfp8", "mixed"]:
        a_ref = a_ref.to(torch.float32)
        a = a_ref.to(torch.float8_e4m3fn)
    else:
        a = a_ref.to_packed_tensor(dim=1)

    if block_scale_type == "mxfp8":
        b_ref = b_ref.to(torch.float32)
        b = b_ref.to(torch.float8_e4m3fn)
    else:
        b = b_ref.to_packed_tensor(dim=1)

    b_ref = b_ref.to(torch.float32).T

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])

    a_scale_shape = [M // 128, K // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [N // 128, K // VEC_SIZE // 4, 32, 16]
    epsilon = 1e-8
    a_scale = torch.rand(a_scale_shape, device=device) + epsilon
    b_scale = torch.rand(b_scale_shape, device=device) + epsilon
    if block_scale_type == "nvfp4":
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
        a_scale_ref = a_scale
        b_scale_ref = b_scale
    elif block_scale_type in ["mxfp4", "mxfp8", "mixed"]:
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data

    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    a_scale_block_shape = [1, rep_m, rep_k, 2, 256]
    b_scale_block_shape = [1, rep_n, rep_k, 2, 256]
    a_scale = a_scale.reshape(1, a_scale_shape[0], a_scale.shape[1], 2, 256)
    b_scale = b_scale.reshape(1, b_scale_shape[0], b_scale.shape[1], 2, 256)
    a_scale_desc = TensorDescriptor.from_tensor(a_scale, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale, block_shape=b_scale_block_shape)

    reference = None
    if compute_reference:
        a_scale_ref = a_scale_ref.to(torch.float32)
        b_scale_ref = b_scale_ref.to(torch.float32)

        def unpack_scale(packed):
            packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
            num_chunk_m, num_chunk_k, _, _, _ = packed.shape
            return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

        a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
        b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
        reference = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref * b_scale_ref)

    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "num_stages": 4,
        "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
        "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
        "VEC_SIZE": VEC_SIZE,
    }
    return a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, reference
