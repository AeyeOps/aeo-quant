#!/usr/bin/env python3
"""Per plan Gate 0 — torchao NVFP4 dispatch probe.

Loads ONE expert's gate_up_proj weight from the NVFP4 checkpoint (no
full model), wraps it in torchao.prototype.mx_formats.NVFP4Tensor, and
calls the dispatch function with bf16 activations at realistic decode
and prefill shapes.

Expected on sm_121 per 2026-04-17 research (kb/nvfp4-blackwell-research.md):
torchao routes to torch._scaled_mm → cuBLAS, which lacks sm_121 FP4.
A clean pass would let us skip the kernel write entirely; failure
documents the exact cuBLAS/CUTLASS rejection path for our commit log.

Safe: reads a single tensor slice from safetensors, never loads the
full model. Peak VRAM expected under 100 MB.

Usage::

    uv run python examples/safe_probe.py examples/probe_nvfp4_torchao.py

Or directly (bypasses subprocess isolation)::

    uv run python examples/probe_nvfp4_torchao.py
"""
from __future__ import annotations

import os
import sys

import torch


def main() -> int:
    print("=== torchao NVFP4 probe ===")
    print(f"torch: {torch.__version__}")
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available")
        return 2
    cc = torch.cuda.get_device_capability(0)
    dev_name = torch.cuda.get_device_name(0)
    print(f"device: {dev_name} (sm_{cc[0]}{cc[1]})")

    try:
        import torchao
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            NVFP4Tensor,
            _addmm_nvfp4_dispatch,
        )
        print(f"torchao: {torchao.__version__}")
    except ImportError as e:
        print(f"[FATAL] torchao not importable: {e}")
        return 2

    ckpt = os.environ.get("NVFP4_CHECKPOINT", "/opt/dev/aeo/hf-gemma4-nvfp4")
    if not os.path.isdir(ckpt):
        print(f"[FATAL] NVFP4 checkpoint not found at {ckpt}")
        return 2

    # Load one expert slice directly from safetensors — no full model.
    print(f"\n--- loading one expert slice from {ckpt} ---")
    import safetensors.torch as sft
    import json
    from pathlib import Path

    index_path = Path(ckpt) / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)["weight_map"]

    # Find the gate_up_proj expert weight names for the first MoE layer.
    packed_key = scale_key = scale_2_key = None
    for name in index:
        if "experts.gate_up_proj" in name and name.endswith("weight_packed"):
            packed_key = name
            break
        if "experts.gate_up_proj" in name and not name.endswith(("_scale", "_scale_2", "_inv_s")):
            # fallback — some naming schemes differ
            packed_key = name
            break
    # Conservative: scan all keys and pick by heuristic.
    keys_in_first_layer = [k for k in index if ".layers.0." in k and "experts" in k]
    print(f"  first-layer expert keys ({len(keys_in_first_layer)} total, showing first 12):")
    for k in keys_in_first_layer[:12]:
        print(f"    {k}")

    # Pick the exact names the checkpoint uses.
    for k in keys_in_first_layer:
        if "gate_up_proj" in k and k.endswith("gate_up_proj"):
            packed_key = k
        elif "gate_up_proj_scale" == k.split(".")[-1]:
            scale_key = k
        elif "gate_up_proj_scale_2" == k.split(".")[-1]:
            scale_2_key = k

    if not (packed_key and scale_key and scale_2_key):
        print(f"\n[FATAL] could not locate expert keys")
        print(f"  packed={packed_key} scale={scale_key} scale_2={scale_2_key}")
        return 2

    print(f"\n  using:")
    print(f"    packed:   {packed_key} (shard: {index[packed_key]})")
    print(f"    scale:    {scale_key}  (shard: {index[scale_key]})")
    print(f"    scale_2:  {scale_2_key} (shard: {index[scale_2_key]})")

    # Load just those three tensors.
    with sft.safe_open(Path(ckpt) / index[packed_key], framework="pt", device="cuda") as f:
        packed = f.get_tensor(packed_key)
    with sft.safe_open(Path(ckpt) / index[scale_key], framework="pt", device="cuda") as f:
        block_scale = f.get_tensor(scale_key)
    with sft.safe_open(Path(ckpt) / index[scale_2_key], framework="pt", device="cuda") as f:
        tensor_scale = f.get_tensor(scale_2_key)

    print(f"\n  packed shape: {tuple(packed.shape)} dtype: {packed.dtype}")
    print(f"  block_scale shape: {tuple(block_scale.shape)} dtype: {block_scale.dtype}")
    print(f"  tensor_scale shape: {tuple(tensor_scale.shape)} dtype: {tensor_scale.dtype}")

    # Expert 0 weight: shape (2*intermediate, hidden // 2) uint8
    E, out_dim, packed_in = packed.shape
    in_dim = packed_in * 2
    print(f"\n  experts: E={E}, out_dim={out_dim}, in_dim={in_dim}")

    # Take expert 0's weight and scale, make 2D (out, in//2) and (out, in//16).
    w_packed = packed[0].contiguous()            # (out, in//2) uint8
    w_scale = block_scale[0].contiguous()        # (out, in//16) fp8_e4m3fn
    w_tensor_scale = tensor_scale.to(torch.float32)

    print(f"\n  expert-0 w_packed: {tuple(w_packed.shape)} uint8")
    print(f"  expert-0 w_scale:  {tuple(w_scale.shape)} {w_scale.dtype}")
    print(f"  w_tensor_scale:    {w_tensor_scale.item():.6e}")

    # Bf16 reference via our dequant path.
    from aeo_quant.gpu.quant import dequant_3d_from_nvfp4
    w_bf16 = dequant_3d_from_nvfp4(
        packed[:1], block_scale[:1], tensor_scale,
    )[0]  # (out, in) bf16
    print(f"\n  bf16 reference weight: {tuple(w_bf16.shape)} {w_bf16.dtype}")
    print(f"    mean_abs={w_bf16.abs().mean().item():.4f}  max_abs={w_bf16.abs().max().item():.4f}")

    # Build NVFP4Tensor from our packed data.
    # The torchao NVFP4Tensor expects: qdata (packed uint8), scale (fp8 block),
    # block_size=16, orig_dtype, optional per_tensor_scale.
    # Shape convention: torchao stores qdata as (M, K//2) packed where the
    # "outer" dim is the high-precision's outer dim; its tensor shape reports
    # (M, K) after fp4x2-to-hp translation.
    try:
        w_nvfp4 = NVFP4Tensor(
            qdata=w_packed,
            scale=w_scale.contiguous(),
            block_size=16,
            orig_dtype=torch.bfloat16,
            per_tensor_scale=w_tensor_scale,
            is_swizzled_scales=False,
        )
        print(f"\n  NVFP4Tensor constructed: shape={tuple(w_nvfp4.shape)} dtype={w_nvfp4.dtype}")
    except Exception as e:
        print(f"\n[FAIL] NVFP4Tensor constructor: {type(e).__name__}: {e}")
        return 1

    # Roundtrip: torchao.dequantize() vs our bf16 reference.
    try:
        w_back = w_nvfp4.dequantize(torch.bfloat16)
        err = (w_back - w_bf16).abs().max().item()
        rel = (w_back - w_bf16).abs().mean().item() / (w_bf16.abs().mean().item() + 1e-8)
        print(f"  dequant roundtrip vs our ref: max_abs_err={err:.4f}  rel_mean_err={rel:.4f}")
    except Exception as e:
        print(f"  dequant roundtrip: FAILED — {type(e).__name__}: {e}")

    # Now the main event: _addmm_nvfp4_dispatch at decode shapes.
    # The dispatch expects both A (activations) and B (weight) as NVFP4Tensor.
    # We need to quantize a bf16 activation to NVFP4 first.
    for m in (1, 8):
        print(f"\n--- _addmm_nvfp4_dispatch at M={m}, K={in_dim}, N={out_dim} ---")
        x_bf16 = torch.randn(m, in_dim, device="cuda", dtype=torch.bfloat16) * 0.1

        try:
            # Activation quant: per-tensor scale computed from x
            x_amax = x_bf16.abs().max().clamp(min=1e-6).to(torch.float32)
            a_per_tensor_scale = x_amax / (6.0 * 448.0)  # F4_E2M1_MAX * F8E4M3_MAX
            x_nvfp4 = NVFP4Tensor.to_nvfp4(
                x_bf16,
                block_size=16,
                per_tensor_scale=a_per_tensor_scale,
                is_swizzled_scales=False,
            )
            print(f"  x_nvfp4 ready: qdata={tuple(x_nvfp4.qdata.shape)} scale={tuple(x_nvfp4.scale.shape)}")
        except Exception as e:
            print(f"  [FAIL] activation quant: {type(e).__name__}: {e}")
            continue

        # b needs to be (K, N) semantically — torchao's dispatch expects b.t() contiguous.
        # Our w_nvfp4 is (out, in); in matmul terms that's (N, K). So pass as is and
        # let dispatch do the .t() inside.
        try:
            out = _addmm_nvfp4_dispatch(x_nvfp4, w_nvfp4.t(), bias=None)
            ref = torch.matmul(x_bf16, w_bf16.t())
            out_norm = out.norm().item()
            ref_norm = ref.norm().item()
            max_rel = (out - ref).norm().item() / (ref_norm + 1e-8)
            print(f"  [OK] dispatch ran: out={tuple(out.shape)} {out.dtype} "
                  f"norm={out_norm:.4f} ref_norm={ref_norm:.4f} rel_err={max_rel:.4f}")
        except Exception as e:
            print(f"  [FAIL] dispatch: {type(e).__name__}: {e}")
            # Print a short trace for debugging
            import traceback
            tb = traceback.format_exc().splitlines()
            for line in tb[-8:]:
                print(f"    {line}")

    print("\n=== probe done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
