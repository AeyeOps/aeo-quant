#!/usr/bin/env python3
"""Phase 1+2 synthetic test of ``nvfp4_linear_3d_prequantized``.

Compares the 3D batched kernel against a per-expert loop over the 2D
kernel, for a handful of shapes covering tiny (Phase 1) and real
Gemma-decode (Phase 2) geometries.

Safe: VRAM << 500 MB at all shapes. No model load.

Run with ``TRITON_OVERRIDE_ARCH=sm120`` on GB10 (sm_121); warns if unset.

Exit codes:
  0 — all shapes within tolerance (max rel_err < 1e-3 vs per-expert ref)
  1 — at least one shape fails
  2 — CUDA unavailable
"""
from __future__ import annotations

import os
import sys
import time

import torch


def _run_one(E: int, M: int, K: int, N: int, *, verbose: bool = True) -> dict:
    from aeo_quant.gpu.nvfp4_matmul import (
        nvfp4_linear_3d_prequantized,
        nvfp4_linear_prequantized,
    )
    from aeo_quant.gpu.quant import (
        quantize_2d_to_nvfp4,
        quantize_3d_to_nvfp4,
    )

    torch.manual_seed(E * 100003 + M * 397 + K * 13 + N)
    device = "cuda"

    # Activation: shared across experts.
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
    a_packed, a_block_scale, a_tensor_scale = quantize_2d_to_nvfp4(x_bf16)

    # Weights: (E, N, K) bf16, quantized per-3D-tensor → shared tensor_scale
    # across experts, matching Gemma 4 checkpoint layout.
    w_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device=device) * 0.05
    w_packed, w_block_scale, w_tensor_scale = quantize_3d_to_nvfp4(w_bf16)

    # Reference: per-expert 2D matmul, stacked.
    ref_per_expert = []
    for e in range(E):
        out_e = nvfp4_linear_prequantized(
            a_packed, a_block_scale, a_tensor_scale,
            w_packed[e], w_block_scale[e], w_tensor_scale,
        )
        ref_per_expert.append(out_e)
    ref = torch.stack(ref_per_expert, dim=0)  # (E, M, N)

    # Warmup + timing
    torch.cuda.synchronize()
    out = nvfp4_linear_3d_prequantized(
        a_packed, a_block_scale, a_tensor_scale,
        w_packed, w_block_scale, w_tensor_scale,
    )
    torch.cuda.synchronize()

    n_iters = 5 if M <= 8 else 2
    t_start = time.monotonic()
    for _ in range(n_iters):
        out = nvfp4_linear_3d_prequantized(
            a_packed, a_block_scale, a_tensor_scale,
            w_packed, w_block_scale, w_tensor_scale,
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.monotonic() - t_start) * 1000 / n_iters

    # Reference time (for speedup ratio)
    t_start = time.monotonic()
    for _ in range(n_iters):
        ref_outs = []
        for e in range(E):
            ref_outs.append(nvfp4_linear_prequantized(
                a_packed, a_block_scale, a_tensor_scale,
                w_packed[e], w_block_scale[e], w_tensor_scale,
            ))
        _ = torch.stack(ref_outs, dim=0)
    torch.cuda.synchronize()
    ref_ms = (time.monotonic() - t_start) * 1000 / n_iters

    err = (out.float() - ref.float()).abs()
    denom = ref.float().abs().mean().item() + 1e-8
    rel = err.mean().item() / denom
    max_abs = err.max().item()
    ok = rel < 1e-3  # Same gate as per-expert test

    speedup = ref_ms / elapsed_ms if elapsed_ms > 0 else float("inf")

    if verbose:
        status = "OK" if ok else "FAIL"
        print(
            f"  E={E} M={M:>5} N={N:>5} K={K:>5}: "
            f"[{status}] rel_err={rel:.6f} max_abs={max_abs:.6f} "
            f"3d={elapsed_ms:.3f}ms ref={ref_ms:.3f}ms speedup={speedup:.2f}x"
        )
    return {
        "E": E, "M": M, "N": N, "K": K,
        "ok": ok, "rel_err": rel, "max_abs_err": max_abs,
        "elapsed_ms": elapsed_ms, "ref_ms": ref_ms, "speedup": speedup,
    }


def _run_one_per_expert(E: int, M: int, K: int, N: int, *, verbose: bool = True) -> dict:
    """Per-expert activation layout — (E, M, K) input instead of (M, K).

    Reference: loop of 2D matmuls, each with its own a-slice and a-scale.
    """
    from aeo_quant.gpu.nvfp4_matmul import (
        nvfp4_linear_3d_prequantized,
        nvfp4_linear_prequantized,
    )
    from aeo_quant.gpu.quant import (
        quantize_2d_to_nvfp4,
        quantize_3d_to_nvfp4,
    )

    torch.manual_seed(E * 100003 + M * 397 + K * 13 + N + 999983)
    device = "cuda"

    # Per-expert activation (E, M, K) bf16 → quantize_3d (shared tensor_scale
    # across E experts, matching what we'd get from stacking the MoE
    # down-proj inputs and calling quantize_2d on (E*M, K)).
    x_bf16 = torch.randn(E, M, K, dtype=torch.bfloat16, device=device) * 0.1
    a_packed, a_block_scale, a_tensor_scale = quantize_3d_to_nvfp4(x_bf16)

    w_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device=device) * 0.05
    w_packed, w_block_scale, w_tensor_scale = quantize_3d_to_nvfp4(w_bf16)

    # Reference: per-expert 2D matmul. Each expert gets its own
    # a-slice but the SAME a_tensor_scale (shared across experts).
    ref_list = []
    for e in range(E):
        out_e = nvfp4_linear_prequantized(
            a_packed[e], a_block_scale[e], a_tensor_scale,
            w_packed[e], w_block_scale[e], w_tensor_scale,
        )
        ref_list.append(out_e)
    ref = torch.stack(ref_list, dim=0)

    # 3D kernel with per-expert activation (3D input).
    torch.cuda.synchronize()
    out = nvfp4_linear_3d_prequantized(
        a_packed, a_block_scale, a_tensor_scale,
        w_packed, w_block_scale, w_tensor_scale,
    )
    torch.cuda.synchronize()

    n_iters = 5
    t_start = time.monotonic()
    for _ in range(n_iters):
        out = nvfp4_linear_3d_prequantized(
            a_packed, a_block_scale, a_tensor_scale,
            w_packed, w_block_scale, w_tensor_scale,
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.monotonic() - t_start) * 1000 / n_iters

    err = (out.float() - ref.float()).abs()
    denom = ref.float().abs().mean().item() + 1e-8
    rel = err.mean().item() / denom
    max_abs = err.max().item()
    ok = rel < 1e-3

    if verbose:
        status = "OK" if ok else "FAIL"
        print(
            f"  E={E} M={M:>5} N={N:>5} K={K:>5}: "
            f"[{status}] rel_err={rel:.6f} max_abs={max_abs:.6f} "
            f"3d={elapsed_ms:.3f}ms (per-expert activation)"
        )
    return {
        "E": E, "M": M, "N": N, "K": K,
        "ok": ok, "rel_err": rel, "max_abs_err": max_abs,
        "elapsed_ms": elapsed_ms, "variant": "per-expert",
    }


def main() -> int:
    print("=== nvfp4 3D kernel test ===")
    override = os.environ.get("TRITON_OVERRIDE_ARCH", "")
    print(f"TRITON_OVERRIDE_ARCH = {override or '(not set)'}")
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available")
        return 2

    cc = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")
    if cc == (12, 1) and override != "sm120":
        print(
            "\n[WARNING] sm_121 without TRITON_OVERRIDE_ARCH=sm120:"
            " tl.dot_scaled falls through to a slow decomposition."
        )

    import triton
    print(f"triton: {triton.__version__}\n")

    # Phase 1 — tiny shapes (shared activation): single K-tile, small E.
    # Phase 2 — Gemma decode shapes (shared activation):
    #   gate_up: k=4, M=1, K=2816, N=1408  (hidden=2816, moe_intermediate=704, 2*im=1408)
    #   down:    k=4, M=1, K=704,  N=2816
    shapes = [
        # Phase 1 (tiny, Kernel Phase 1 gate)
        (2, 1, 128, 128),
        (2, 8, 128, 128),
        (4, 1, 128, 128),
        (4, 16, 128, 128),
        # Phase 2 (Gemma decode, Kernel Phase 2 gate)
        (4, 1, 2816, 1408),   # gate_up decode
        (4, 1, 704, 2816),    # down decode
    ]

    print("--- shared-activation shapes ---")
    all_ok = True
    results = []
    for E, M, K, N in shapes:
        try:
            r = _run_one(E, M, K, N)
            results.append(r)
            if not r["ok"]:
                all_ok = False
        except Exception as e:
            print(f"  E={E} M={M} K={K} N={N}: [FAIL] {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            results.append({"E": E, "M": M, "K": K, "N": N, "ok": False, "error": str(e)})
            all_ok = False

    # Per-expert activation (3D input) — down_proj decode pattern.
    print("\n--- per-expert-activation shapes ---")
    per_expert_shapes = [
        (4, 1, 128, 128),     # tiny sanity
        (4, 1, 704, 2816),    # down_proj decode
    ]
    for E, M, K, N in per_expert_shapes:
        try:
            r = _run_one_per_expert(E, M, K, N)
            results.append(r)
            if not r["ok"]:
                all_ok = False
        except Exception as e:
            print(f"  E={E} M={M} K={K} N={N}: [FAIL] {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            results.append({"E": E, "M": M, "K": K, "N": N, "ok": False, "error": str(e)})
            all_ok = False

    print("\n--- summary ---")
    total = len(results)
    if all_ok:
        print(f"  all {total} shapes pass")
    else:
        fails = sum(1 for r in results if not r.get("ok"))
        print(f"  FAILED: {fails}/{total}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
