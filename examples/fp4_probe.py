#!/usr/bin/env python3
"""NVFP4 feasibility probe for Blackwell (sm_121).

Checks torch FP4 dtype availability, _scaled_mm compatibility, and
basic matmul correctness. No model loading — safe to run anytime.

Usage:
    uv run python examples/fp4_probe.py
"""
from __future__ import annotations

import sys

import torch

def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available")
        return 2

    dev = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"device: {dev} (sm_{cc[0]}{cc[1]})")

    # --- Check FP4 dtype availability ---
    print("\n=== FP4 dtype availability ===")

    fp4_candidates = [
        "float4_e2m1fn",
        "float4_e2m1",
        "float4_e2m1fn_x2",  # packed pair format
    ]
    found_dtypes = {}
    for name in fp4_candidates:
        dt = getattr(torch, name, None)
        status = "FOUND" if dt is not None else "not found"
        print(f"  torch.{name}: {status}")
        if dt is not None:
            found_dtypes[name] = dt

    # Also check for MXFP4 / microscaling support
    mx_candidates = [
        "float4_e2m1fn_x2",  # NV microscaling packed format
    ]
    print("\n=== Microscaling / block scaling ===")
    has_mx = hasattr(torch, "_mx_fp4_matmul") or hasattr(torch, "ops.aten._mx_matmul")
    print(f"  torch._mx_fp4_matmul: {'FOUND' if has_mx else 'not found'}")

    # Check for torchao (quantization toolkit)
    try:
        import torchao
        print(f"  torchao: {torchao.__version__}")
        has_torchao = True
    except ImportError:
        print("  torchao: not installed")
        has_torchao = False

    # Check for float8/scaled_mm support as baseline
    print("\n=== _scaled_mm baseline (FP8) ===")
    try:
        M, K, N = 4, 128, 256
        a_fp8 = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
        b_fp8 = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
        scale_a = torch.ones(M, 1, device="cuda", dtype=torch.float32)
        scale_b = torch.ones(1, N, device="cuda", dtype=torch.float32)
        out = torch._scaled_mm(a_fp8, b_fp8.t(), scale_a=scale_a, scale_b=scale_b,
                               out_dtype=torch.bfloat16)
        print(f"  FP8 _scaled_mm: OK — output shape {tuple(out.shape)}, dtype {out.dtype}")
    except Exception as e:
        print(f"  FP8 _scaled_mm: FAILED — {e}")

    # --- Try FP4 with _scaled_mm if dtype exists ---
    if found_dtypes:
        print("\n=== _scaled_mm with FP4 ===")
        for name, dt in found_dtypes.items():
            try:
                M, K, N = 4, 128, 256
                a = torch.randn(M, K, device="cuda").to(dt)
                b = torch.randn(N, K, device="cuda").to(dt)
                scale_a = torch.ones(M, 1, device="cuda", dtype=torch.float32)
                scale_b = torch.ones(1, N, device="cuda", dtype=torch.float32)
                out = torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b,
                                       out_dtype=torch.bfloat16)
                print(f"  {name} _scaled_mm: OK — output shape {tuple(out.shape)}")
            except Exception as e:
                print(f"  {name} _scaled_mm: FAILED — {e}")

    # --- Check for CUTLASS FP4 kernels ---
    print("\n=== CUTLASS / cuBLAS FP4 support ===")
    try:
        # Check if cublas supports FP4 via capability query
        cublas_version = torch.backends.cuda.preferred_blas_library()
        print(f"  preferred BLAS: {cublas_version}")
    except Exception:
        pass

    # Check CUDA version
    print(f"  CUDA runtime: {torch.version.cuda}")

    # --- Try quantizing a bf16 tensor to FP4 ---
    if found_dtypes:
        print("\n=== FP4 quantization test ===")
        for name, dt in found_dtypes.items():
            try:
                x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
                x_fp4 = x.to(dt)
                x_back = x_fp4.to(torch.bfloat16)
                max_err = (x - x_back).abs().max().item()
                mean_err = (x - x_back).abs().mean().item()
                print(f"  {name} round-trip: max_err={max_err:.4f}, mean_err={mean_err:.4f}")
                print(f"    element_size: {x_fp4.element_size()} bytes")
                print(f"    storage bytes: {x_fp4.nelement() * x_fp4.element_size()}")
                print(f"    vs bf16 bytes: {x.nelement() * x.element_size()}")
            except Exception as e:
                print(f"  {name} round-trip: FAILED — {e}")

    # --- Summary ---
    print("\n=== Summary ===")
    if found_dtypes:
        print(f"  FP4 dtypes available: {', '.join(found_dtypes.keys())}")
        print("  Next step: test _scaled_mm path with real weight shapes")
    else:
        print("  No FP4 dtypes found in torch.")
        print("  Options:")
        print("    1. Check if torchao provides FP4 quantization")
        print("    2. Check if a newer torch nightly has FP4 support")
        print("    3. Use INT4 (GPTQ/AWQ) as alternative 4-bit path")

    if has_torchao:
        print("\n  torchao is installed — checking FP4/INT4 quantization APIs:")
        try:
            from torchao.quantization import quantize_, int4_weight_only
            print("    int4_weight_only: available")
        except ImportError:
            print("    int4_weight_only: not available")
        try:
            from torchao.quantization import float4_weight_only
            print("    float4_weight_only: available")
        except ImportError:
            print("    float4_weight_only: not available")
        try:
            from torchao.dtypes import NF4Tensor
            print("    NF4Tensor: available")
        except ImportError:
            print("    NF4Tensor: not available")

    return 0


if __name__ == "__main__":
    sys.exit(main())
