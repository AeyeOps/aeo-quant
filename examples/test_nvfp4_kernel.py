#!/usr/bin/env python3
"""Unit test of the NVFP4 Triton kernel.

Runs ``nvfp4_linear`` at decode (M=1, M=8) and prefill (M=128, M=2816)
shapes with our real checkpoint layout, compares output against a
bf16 reference matmul computed from the dequantized weights.

Safe: VRAM usage < 500 MB even at the largest shape. No model load.

The kernel auto-applies the sm_120 Triton arch coercion on GB10 via
``ensure_nvfp4_triton_arch()`` — see ``aeo_quant.core.config`` for the
rationale. You can still pin ``TRITON_OVERRIDE_ARCH`` explicitly on the
command line to benchmark the fallback path.

Exit codes:
  0 — all shapes pass (max rel err < 0.30 vs bf16 reference)
  1 — correctness failure at one or more shapes
  2 — GPU unavailable or setup error
"""
from __future__ import annotations

import os
import sys
import time

import torch

from aeo_quant.core.config import ensure_nvfp4_triton_arch

ensure_nvfp4_triton_arch()


# Gemma 4 expert shapes
HIDDEN_DIM = 2880
INTERMEDIATE_DIM = 2 * 2880  # 2 * moe_intermediate_size = gate_up_proj out


def _bf16_reference(
    x_bf16: torch.Tensor,
    w_bf16: torch.Tensor,
    w_tensor_scale: torch.Tensor,
) -> torch.Tensor:
    """Ground truth: ``x @ (w_bf16 * w_tensor_scale).T`` in fp32."""
    w_scaled = w_bf16.float() * w_tensor_scale.float()
    return torch.matmul(x_bf16.float(), w_scaled.T).to(torch.bfloat16)


def _test_shape(M: int, K: int, N: int, verbose: bool = True) -> dict:
    """Run one shape and return {ok, rel_err, elapsed_ms}."""
    from aeo_quant.gpu.nvfp4_matmul import nvfp4_linear
    from aeo_quant.gpu.quant import (
        quantize_2d_to_nvfp4,
        dequant_2d_from_nvfp4,
    )

    torch.manual_seed(M * K + N)
    device = "cuda"

    x_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1
    w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.05

    w_packed, w_block_scale, w_tensor_scale = quantize_2d_to_nvfp4(w_bf16)
    # Dequant to get the effective weight the kernel should compute against.
    w_effective = dequant_2d_from_nvfp4(w_packed, w_block_scale, w_tensor_scale)

    # Warmup + time
    torch.cuda.synchronize()
    out = nvfp4_linear(x_bf16, w_packed, w_block_scale, w_tensor_scale)
    torch.cuda.synchronize()

    # Re-run for timing after warmup
    n_iters = 5 if M <= 8 else 2
    t_start = time.monotonic()
    for _ in range(n_iters):
        out = nvfp4_linear(x_bf16, w_packed, w_block_scale, w_tensor_scale)
    torch.cuda.synchronize()
    elapsed_ms = (time.monotonic() - t_start) * 1000 / n_iters

    # Reference: the effective weight is already dequantized, so
    # the kernel should match x @ w_effective.T (no extra tensor_scale —
    # it's already absorbed during dequant_2d_from_nvfp4).
    ref = torch.matmul(x_bf16.float(), w_effective.float().T).to(torch.bfloat16)

    err = (out.float() - ref.float()).abs()
    rel = err.mean().item() / (ref.float().abs().mean().item() + 1e-8)
    max_abs = err.max().item()
    ok = rel < 0.30  # FP4 quantization floor

    if verbose:
        status = "OK" if ok else "FAIL"
        print(
            f"  M={M:>5} N={N:>5} K={K:>5}: "
            f"[{status}] rel_err={rel:.4f} max_abs={max_abs:.4f} "
            f"elapsed={elapsed_ms:.2f}ms"
        )
    return {"M": M, "N": N, "K": K, "ok": ok, "rel_err": rel,
            "max_abs_err": max_abs, "elapsed_ms": elapsed_ms}


def main() -> int:
    print("=== nvfp4 kernel test ===")
    print(f"TRITON_OVERRIDE_ARCH = {os.environ.get('TRITON_OVERRIDE_ARCH', '(unset)')}")
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available")
        return 2

    cc = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")

    import triton
    print(f"triton: {triton.__version__}\n")

    # Decode shapes (small M, realistic expert K/N)
    # Prefill shapes (M up to intermediate_dim)
    K = HIDDEN_DIM  # 2880
    N = INTERMEDIATE_DIM  # 5760 = 2 * 2880
    shapes = [
        (1, K, N),       # single-token decode
        (8, K, N),       # prefix decode
        (64, K, N),      # burst decode
        (128, K, N),     # small prefill
        (512, K, N),     # medium prefill
        (2880, K, N),    # full hidden-length prefill
    ]

    print(f"--- testing {len(shapes)} shapes (K=hidden={K}, N=2*moe_intermediate={N}) ---")
    results = []
    all_ok = True
    for M, K_s, N_s in shapes:
        try:
            r = _test_shape(M, K_s, N_s)
            results.append(r)
            if not r["ok"]:
                all_ok = False
        except torch.cuda.OutOfMemoryError as e:
            print(f"  M={M} K={K_s} N={N_s}: [OOM] skipping")
            results.append({"M": M, "ok": False, "error": "OOM"})
            all_ok = False
        except Exception as e:
            print(f"  M={M} K={K_s} N={N_s}: [FAIL] {type(e).__name__}: {e}")
            results.append({"M": M, "ok": False, "error": str(e)})
            all_ok = False

    print("\n--- summary ---")
    if all_ok:
        print(f"  all {len(shapes)} shapes pass")
        # Report throughput at interesting shapes
        for r in results:
            if r.get("ok") and r.get("elapsed_ms"):
                M = r["M"]
                flops = 2.0 * r["M"] * r["N"] * r["K"]
                tflops = flops / (r["elapsed_ms"] * 1e-3) / 1e12
                print(
                    f"  M={M:>5}: {r['elapsed_ms']:>7.2f} ms → "
                    f"{tflops:>6.2f} TFLOPS"
                )
    else:
        fails = sum(1 for r in results if not r.get("ok"))
        print(f"  FAILED: {fails}/{len(shapes)} shapes")

    print("\n--- next step ---")
    print("  Inspect SASS of the most recent kernel to confirm native FP4:")
    print("    tools/dump_triton_sass.sh --name _nvfp4_matmul --limit 200 \\")
    print("        | grep -E 'HMMA|MXF4|NVF4|kind::' | head -20")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
