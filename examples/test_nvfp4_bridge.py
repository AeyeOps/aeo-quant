#!/usr/bin/env python3
"""Bridge integration test — load ONE expert from the real NVFP4 checkpoint.

Validates that our kernel matches the checkpoint's actual layout:
packing order, scale dtype, scale layout.  Plan Gate 4 equivalent at
the per-expert level, without needing to load the full 27 GB model.

For each of gate_up_proj and down_proj from expert 0 of layer 0:

1. Load packed uint8 + fp8 block scales + fp32 tensor scale directly
   from safetensors.
2. Dequantize to bf16 (our existing path) — this is the "ground truth".
3. Build a bf16 activation at realistic decode and prefill shapes.
4. Run nvfp4_linear with the packed checkpoint weights + runtime-quantized
   activations.
5. Compare output against bf16 reference matmul.

Usage::

    uv run python examples/test_nvfp4_bridge.py

Safe: peak VRAM under 1 GB (one layer's experts).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

from aeo_quant.core.config import ensure_nvfp4_triton_arch

ensure_nvfp4_triton_arch()


def _load_expert_0(ckpt_dir: str) -> dict:
    """Load layer 0 / expert 0 NVFP4 tensors directly from safetensors."""
    import safetensors.torch as sft

    index = json.load(open(Path(ckpt_dir) / "model.safetensors.index.json"))["weight_map"]

    # Find layer 0 expert tensors
    keys = {}
    for k in index:
        if ".layers.0." not in k or "experts" not in k:
            continue
        leaf = k.split(".")[-1]
        if leaf == "gate_up_proj":
            keys["gate_up_packed"] = k
        elif leaf == "gate_up_proj_scale":
            keys["gate_up_scale"] = k
        elif leaf == "gate_up_proj_scale_2":
            keys["gate_up_scale_2"] = k
        elif leaf == "down_proj":
            keys["down_packed"] = k
        elif leaf == "down_proj_scale":
            keys["down_scale"] = k
        elif leaf == "down_proj_scale_2":
            keys["down_scale_2"] = k

    missing = {"gate_up_packed", "gate_up_scale", "gate_up_scale_2",
               "down_packed", "down_scale", "down_scale_2"} - keys.keys()
    if missing:
        raise KeyError(f"missing keys in checkpoint: {missing}")

    tensors = {}
    for name, key in keys.items():
        shard = Path(ckpt_dir) / index[key]
        with sft.safe_open(shard, framework="pt", device="cuda") as f:
            tensors[name] = f.get_tensor(key)
    return tensors


def _test_projection(name: str, w_packed, w_scale, w_tensor_scale,
                     M: int, K: int, N: int) -> dict:
    """Test one projection (gate_up or down) at one M shape."""
    from aeo_quant.gpu.nvfp4_matmul import nvfp4_linear
    from aeo_quant.gpu.quant import dequant_3d_from_nvfp4

    torch.manual_seed(M + K)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1

    # Reference: dequantize weight, do a plain matmul
    # Expert 0 slice — w_packed is (E, N, K//2). Take expert 0.
    # dequant takes full (E, N, K) batch
    w_bf16_full = dequant_3d_from_nvfp4(
        w_packed[:1], w_scale[:1], w_tensor_scale,
    )
    w_bf16 = w_bf16_full[0]  # (N, K) bf16
    ref = torch.matmul(x.float(), w_bf16.float().T).to(torch.bfloat16)

    # Kernel: call nvfp4_linear with expert 0's slice
    torch.cuda.synchronize()
    t0 = time.monotonic()
    out = nvfp4_linear(
        x, w_packed[0], w_scale[0], w_tensor_scale,
    )
    torch.cuda.synchronize()
    elapsed = (time.monotonic() - t0) * 1000

    err = (out.float() - ref.float()).abs()
    rel = err.mean().item() / (ref.float().abs().mean().item() + 1e-8)
    max_abs = err.max().item()
    return {
        "name": name, "M": M, "K": K, "N": N,
        "rel_err": rel, "max_abs": max_abs,
        "elapsed_ms": elapsed,
    }


def main() -> int:
    ckpt = os.environ.get("NVFP4_CHECKPOINT", "/opt/dev/aeo/hf-gemma4-nvfp4")
    print(f"=== nvfp4 bridge test ===")
    print(f"checkpoint: {ckpt}")
    print(f"TRITON_OVERRIDE_ARCH = {os.environ.get('TRITON_OVERRIDE_ARCH', '(not set)')}")

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available")
        return 2

    print(f"device: {torch.cuda.get_device_name(0)} "
          f"(sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]})")

    print("\n--- loading expert 0 from layer 0 ---")
    t = _load_expert_0(ckpt)
    for name, tensor in t.items():
        print(f"  {name:20s}: {tuple(tensor.shape)} {tensor.dtype}")

    gate_up_N, _, gate_up_K_half = t["gate_up_packed"].shape[:3] if t["gate_up_packed"].ndim == 3 else (None, None, None)

    # Derive dims
    E, N_gu, K_gu_half = t["gate_up_packed"].shape
    K_gu = K_gu_half * 2
    E, N_dn, K_dn_half = t["down_packed"].shape
    K_dn = K_dn_half * 2

    print(f"\n  gate_up_proj: (E={E}, N={N_gu}, K={K_gu})")
    print(f"  down_proj:    (E={E}, N={N_dn}, K={K_dn})")

    # Test gate_up at a few decode/prefill shapes
    print("\n--- gate_up_proj tests ---")
    results = []
    for M in (1, 8, 64, 256, 2880):
        try:
            r = _test_projection(
                "gate_up", t["gate_up_packed"], t["gate_up_scale"],
                t["gate_up_scale_2"], M, K_gu, N_gu,
            )
            results.append(r)
            status = "OK" if r["rel_err"] < 0.30 else "FAIL"
            print(f"  M={M:>5} K={K_gu:>5} N={N_gu:>5}: "
                  f"[{status}] rel_err={r['rel_err']:.4f} max_abs={r['max_abs']:.4f} "
                  f"{r['elapsed_ms']:.2f}ms")
        except Exception as e:
            print(f"  M={M:>5}: [FAIL] {type(e).__name__}: {e}")
            results.append({"ok": False, "error": str(e)})

    print("\n--- down_proj tests ---")
    # down_proj: input is (M, hidden) — wait, no. Gemma MoE:
    # gate_up output: (M, 2*im) ; SwiGLU → (M, im)
    # down input:     (M, im)  → output: (M, hidden)
    # So down projection has K = moe_intermediate = N_dn's input dim = im
    # Here K_dn is that.
    for M in (1, 8, 64, 256, 2880):
        try:
            r = _test_projection(
                "down", t["down_packed"], t["down_scale"],
                t["down_scale_2"], M, K_dn, N_dn,
            )
            results.append(r)
            status = "OK" if r["rel_err"] < 0.30 else "FAIL"
            print(f"  M={M:>5} K={K_dn:>5} N={N_dn:>5}: "
                  f"[{status}] rel_err={r['rel_err']:.4f} max_abs={r['max_abs']:.4f} "
                  f"{r['elapsed_ms']:.2f}ms")
        except Exception as e:
            print(f"  M={M:>5}: [FAIL] {type(e).__name__}: {e}")
            results.append({"ok": False, "error": str(e)})

    # --- 3D batched kernel vs per-expert loop (Phase 3 gate) ---
    print("\n--- 3D kernel: 4-expert subset (decode) ---")
    threed_results = _test_3d_path(
        t["gate_up_packed"], t["gate_up_scale"], t["gate_up_scale_2"],
        t["down_packed"], t["down_scale"], t["down_scale_2"],
        K_gu, N_gu, K_dn, N_dn,
    )
    results.extend(threed_results)

    # Summary
    fails = [
        r for r in results
        if r.get("rel_err") is None or r.get("rel_err", 1.0) >= 0.30
    ]
    print(f"\n--- summary: {len(results) - len(fails)}/{len(results)} pass ---")
    return 0 if not fails else 1


def _test_3d_path(
    gu_packed, gu_scale, gu_scale_2,
    dn_packed, dn_scale, dn_scale_2,
    K_gu: int, N_gu: int, K_dn: int, N_dn: int,
) -> list[dict]:
    """Assert the 3D batched kernel matches the per-expert 2D loop.

    Uses a 4-expert subset (selected via ``index_select``) and real
    Gemma decode shapes (M=1, K_hidden=K_gu for gate_up, K=K_dn for
    down). Tolerance: 1e-3 — the 3D and per-expert paths should be
    bit-exact for a shared scalar w_tensor_scale.
    """
    from aeo_quant.gpu.nvfp4_matmul import (
        nvfp4_linear_3d_prequantized,
        nvfp4_linear_prequantized,
    )
    from aeo_quant.gpu.quant import quantize_2d_to_nvfp4

    out = []
    expert_ids = torch.tensor([3, 17, 45, 91], dtype=torch.long, device="cuda")

    # gate_up: (E=128, N=1408, K=2816) — activation (M=1, K=2816)
    torch.manual_seed(31337)
    x_gu = torch.randn(1, K_gu, dtype=torch.bfloat16, device="cuda") * 0.1
    a_packed, a_block_scale, a_tensor_scale = quantize_2d_to_nvfp4(x_gu)

    w_gu_packed_sub = gu_packed.index_select(0, expert_ids).contiguous()
    w_gu_scale_sub = gu_scale.index_select(0, expert_ids).contiguous()

    ref_gu = torch.stack([
        nvfp4_linear_prequantized(
            a_packed, a_block_scale, a_tensor_scale,
            w_gu_packed_sub[i], w_gu_scale_sub[i], gu_scale_2,
        )
        for i in range(len(expert_ids))
    ], dim=0)
    out_gu = nvfp4_linear_3d_prequantized(
        a_packed, a_block_scale, a_tensor_scale,
        w_gu_packed_sub, w_gu_scale_sub, gu_scale_2,
    )
    err = (out_gu.float() - ref_gu.float()).abs()
    rel = err.mean().item() / (ref_gu.float().abs().mean().item() + 1e-8)
    max_abs = err.max().item()
    ok = rel < 1e-3
    print(f"  gate_up  E=4 M=1 K={K_gu} N={N_gu}: "
          f"[{'OK' if ok else 'FAIL'}] rel_err={rel:.6f} max_abs={max_abs:.6f} "
          f"3d_out={tuple(out_gu.shape)}")
    out.append({"name": "gate_up_3d", "rel_err": rel, "max_abs": max_abs, "M": 1})

    # down: (E=128, N=2816, K=704) — activation (M=1, K=704). Fake input from
    # a random bf16 (real flow would be the gate*up result; numerics are
    # about kernel correctness, not end-to-end MoE math).
    torch.manual_seed(31338)
    x_dn = torch.randn(1, K_dn, dtype=torch.bfloat16, device="cuda") * 0.1
    a_packed_dn, a_bs_dn, a_ts_dn = quantize_2d_to_nvfp4(x_dn)

    w_dn_packed_sub = dn_packed.index_select(0, expert_ids).contiguous()
    w_dn_scale_sub = dn_scale.index_select(0, expert_ids).contiguous()

    ref_dn = torch.stack([
        nvfp4_linear_prequantized(
            a_packed_dn, a_bs_dn, a_ts_dn,
            w_dn_packed_sub[i], w_dn_scale_sub[i], dn_scale_2,
        )
        for i in range(len(expert_ids))
    ], dim=0)
    out_dn = nvfp4_linear_3d_prequantized(
        a_packed_dn, a_bs_dn, a_ts_dn,
        w_dn_packed_sub, w_dn_scale_sub, dn_scale_2,
    )
    err = (out_dn.float() - ref_dn.float()).abs()
    rel = err.mean().item() / (ref_dn.float().abs().mean().item() + 1e-8)
    max_abs = err.max().item()
    ok = rel < 1e-3
    print(f"  down     E=4 M=1 K={K_dn} N={N_dn}: "
          f"[{'OK' if ok else 'FAIL'}] rel_err={rel:.6f} max_abs={max_abs:.6f} "
          f"3d_out={tuple(out_dn.shape)}")
    out.append({"name": "down_3d", "rel_err": rel, "max_abs": max_abs, "M": 1})

    return out


if __name__ == "__main__":
    sys.exit(main())
