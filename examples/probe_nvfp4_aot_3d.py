#!/usr/bin/env python3
"""Phase 0a — AOT compile probe for the 3D (per-expert batched) NVFP4 kernel.

Identical intent to ``probe_nvfp4_aot.py`` (inspect PTX for
``mma.sync...kind::mxf4nvf4``) but the kernel has an extra ``pid_e``
axis that offsets the b (weight) and c (output) pointers. Purpose is to
confirm that adding the 3D pointer pattern doesn't regress the native
MMA lowering — before we invest in the full kernel + tests.

Usage::

    uv run python examples/probe_nvfp4_aot_3d.py
    uv run python examples/probe_nvfp4_aot_3d.py --dump-ptx

Exits 0 if sm_120 emits ``kind::mxf4nvf4`` (plan viable), non-zero
otherwise (plan dies; don't write the full kernel).
"""
from __future__ import annotations

import argparse
import re
import sys

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
VEC_SIZE = 16
ELEM_PER_BYTE = 2


@triton.jit
def _nvfp4_matmul_probe_3d(
    a_ptr, b_ptr,
    a_scale_ptr, b_scale_ptr,
    c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    ELEM_PER_BYTE: tl.constexpr,
):
    """3D probe — adds pid_e expert offset. One K-iter, no strides, no masks.

    Shapes:
      a: (M, K/ELEM_PER_BYTE) — shared across experts
      b: (E, N, K/ELEM_PER_BYTE) — per-expert weights
      a_scale: (M, K/VEC_SIZE)
      b_scale: (E, N, K/VEC_SIZE)
      c: (E, M, N)
    """
    pid_mn = tl.program_id(axis=0)
    pid_e = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid_mn % num_pid_m
    pid_n = pid_mn // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_packed = tl.arange(0, BLOCK_K // ELEM_PER_BYTE)
    offs_k_scale = tl.arange(0, BLOCK_K // VEC_SIZE)

    # Activation pointers — no pid_e dependency (shared across experts).
    a_ptrs = a_ptr + offs_m[:, None] * (K // ELEM_PER_BYTE) + offs_k_packed[None, :]
    a_scale_ptrs = a_scale_ptr + offs_m[:, None] * (K // VEC_SIZE) + offs_k_scale[None, :]

    # Weight pointers — per-expert offset via pid_e.
    b_expert_offset = pid_e * (N * K // ELEM_PER_BYTE)
    b_scale_expert_offset = pid_e * (N * K // VEC_SIZE)
    b_ptrs = (b_ptr + b_expert_offset
              + offs_n[:, None] * (K // ELEM_PER_BYTE) + offs_k_packed[None, :])
    b_scale_ptrs = (b_scale_ptr + b_scale_expert_offset
                    + offs_n[:, None] * (K // VEC_SIZE) + offs_k_scale[None, :])

    a_raw = tl.load(a_ptrs)
    b_raw = tl.load(b_ptrs)
    a = ((a_raw & 0xF) << 4) | ((a_raw >> 4) & 0xF)
    b = ((b_raw & 0xF) << 4) | ((b_raw >> 4) & 0xF)
    a_scale = tl.load(a_scale_ptrs)
    b_scale = tl.load(b_scale_ptrs)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tl.dot_scaled(a, a_scale, "e2m1", b.T, b_scale, "e2m1", acc)

    # Output pointer — per-expert offset via pid_e.
    c_expert_offset = pid_e * (M * N)
    c_ptrs = c_ptr + c_expert_offset + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16))


def _compile_for_target(cc: int) -> dict:
    src = ASTSource(
        fn=_nvfp4_matmul_probe_3d,
        signature={
            "a_ptr": "*u8",
            "b_ptr": "*u8",
            "a_scale_ptr": "*fp8e4nv",
            "b_scale_ptr": "*fp8e4nv",
            "c_ptr": "*bf16",
            "M": "constexpr",
            "N": "constexpr",
            "K": "constexpr",
            "E": "constexpr",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "BLOCK_K": "constexpr",
            "VEC_SIZE": "constexpr",
            "ELEM_PER_BYTE": "constexpr",
        },
        constexprs={
            "M": 128, "N": 128, "K": 128, "E": 2,
            "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
            "VEC_SIZE": VEC_SIZE, "ELEM_PER_BYTE": ELEM_PER_BYTE,
        },
    )
    tgt = GPUTarget("cuda", cc, 32)
    out = triton.compile(src, target=tgt)
    return {
        "target": tgt,
        "ttir": out.asm.get("ttir", ""),
        "ttgir": out.asm.get("ttgir", ""),
        "llir": out.asm.get("llir", ""),
        "ptx": out.asm.get("ptx", ""),
        "cubin_bytes": len(out.asm.get("cubin", b"")) if isinstance(out.asm.get("cubin"), bytes) else None,
        "metadata": out.metadata,
    }


def _summarize(tag: str, art: dict, dump: bool = False) -> None:
    print(f"\n=== {tag} ===")
    print(f"target: {art['target']}")
    ptx = art["ptx"]
    ttgir = art["ttgir"]

    n_dot_scaled = len(re.findall(r"tt\.dot_scaled", ttgir))
    n_dot_native = len(re.findall(r"triton_nvidia_gpu\.warp_group_dot", ttgir))
    n_mma = len(re.findall(r"mma_v\d|mmav\d", ttgir, re.I))

    print(f"ttgir lines: {ttgir.count(chr(10))}")
    print(f"  dot_scaled residual:  {n_dot_scaled}  (higher = less lowering)")
    print(f"  warp_group_dot:       {n_dot_native}")
    print(f"  mma references:       {n_mma}")

    ptx_lines = ptx.count("\n")
    mxf4_hits = re.findall(r"mma\.sync[^\n]*kind::(mxf4(?:nvf4)?|mxf8f6f4)", ptx)
    mma_hits = re.findall(r"mma\.sync[^\n]+", ptx)
    hmma_bf16 = re.findall(r"\.bf16\.bf16", ptx)
    print(f"ptx lines:             {ptx_lines}")
    print(f"  mma.sync total:      {len(mma_hits)}")
    print(f"  kind::mxf4[nvf4]:    {len(mxf4_hits)}   <-- native FP4 MMA")
    print(f"  bf16.bf16 opcodes:   {len(hmma_bf16)}   <-- fallback decomposition")
    if mxf4_hits:
        print(f"  first native op: {mxf4_hits[0]}")
    if art["cubin_bytes"] is not None:
        print(f"cubin bytes:           {art['cubin_bytes']}")

    if dump:
        print(f"\n--- PTX first 60 lines of {tag} ---")
        for line in ptx.splitlines()[:60]:
            print(f"  {line}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", type=int, default=[121, 120])
    p.add_argument("--dump-ptx", action="store_true")
    ns = p.parse_args()

    print(f"triton: {triton.__version__}")
    print(f"compiling 3D NVFP4 probe for targets: {ns.targets}")

    results = {}
    for cc in ns.targets:
        try:
            results[cc] = _compile_for_target(cc)
            _summarize(f"sm_{cc}", results[cc], dump=ns.dump_ptx)
        except Exception as e:
            print(f"\n=== sm_{cc} === [COMPILE FAILED] {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print("\n=== verdict ===")
    if 120 in results and 121 in results:
        r120 = results[120]
        n_mxf4_120 = len(re.findall(r"kind::mxf4", r120["ptx"]))
        n_mxf4_121 = len(re.findall(r"kind::mxf4", results[121]["ptx"]))
        print(f"  sm_120 native FP4 opcodes: {n_mxf4_120}")
        print(f"  sm_121 native FP4 opcodes: {n_mxf4_121}")
        if n_mxf4_120 > 0:
            print("  → Phase 0a PASS: 3D kernel emits native FP4 MMA on sm_120. Proceed to Phase 0b.")
            return 0
        print("  → Phase 0a FAIL: 3D kernel did not lower to native FP4. Plan dies.")
        return 1
    print("  → Need both targets. Re-run with --targets 120 121.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
