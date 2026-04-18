#!/usr/bin/env python3
"""Phase 4 — A/B numerics: scalar vs per-expert ``a_tensor_scale``.

In the MoE down-projection at decode, we have k different expert
outputs stacked into (k, im). The 3D kernel consumes a single scalar
``a_tensor_scale``, but the per-expert baseline quantizes each
expert's activation independently (producing k distinct scales).

Open Question #1 (plan): does the shared-scale variant drift enough
to matter? Kill gate: > 5% drift vs the per-expert variant on real
weights.

Measurement:
  - Variant A (scalar alpha): stack (k, im), quantize_2d_to_nvfp4
    once → one tensor_scale; run 3D kernel.
  - Variant B (per-expert alpha): quantize each (1, im) slice
    independently → k tensor_scales; call per-expert 2D kernel.
  - Reference: full-precision bf16 matmul against dequantized
    weights (no activation quantization at all).

Exit:
  0 — scalar variant's drift vs per-expert variant is < 5%
  1 — drift > 5% (plan says ship per-expert-alpha variant)
  2 — setup error
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from aeo_quant.core.config import ensure_nvfp4_triton_arch

ensure_nvfp4_triton_arch()


def main() -> int:
    ckpt = os.environ.get("NVFP4_CHECKPOINT", "/opt/dev/aeo/hf-gemma4-nvfp4")
    print("=== nvfp4 3D A/B alpha test ===")
    print(f"checkpoint: {ckpt}")
    print(f"TRITON_OVERRIDE_ARCH = {os.environ.get('TRITON_OVERRIDE_ARCH', '(unset)')}")

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available")
        return 2

    import json
    import safetensors.torch as sft
    from aeo_quant.gpu.nvfp4_matmul import (
        nvfp4_linear_3d_prequantized,
        nvfp4_linear_prequantized,
    )
    from aeo_quant.gpu.quant import (
        dequant_3d_from_nvfp4,
        quantize_2d_to_nvfp4,
    )

    # Load layer 0 down_proj tensors from the NVFP4 checkpoint.
    index = json.load(open(Path(ckpt) / "model.safetensors.index.json"))["weight_map"]
    keys = {"down_packed": None, "down_scale": None, "down_scale_2": None}
    for k in index:
        if ".layers.0." not in k or "experts" not in k:
            continue
        leaf = k.split(".")[-1]
        if leaf == "down_proj":
            keys["down_packed"] = k
        elif leaf == "down_proj_scale":
            keys["down_scale"] = k
        elif leaf == "down_proj_scale_2":
            keys["down_scale_2"] = k
    missing = [k for k, v in keys.items() if v is None]
    if missing:
        print(f"[FATAL] missing checkpoint keys: {missing}")
        return 2

    tensors = {}
    for name, key in keys.items():
        shard = Path(ckpt) / index[key]
        with sft.safe_open(shard, framework="pt", device="cuda") as f:
            tensors[name] = f.get_tensor(key)

    w_packed = tensors["down_packed"]       # (128, 2816, 352)
    w_scale = tensors["down_scale"]         # (128, 2816, 44)
    w_tensor_scale = tensors["down_scale_2"]  # scalar bf16

    E_all, N_dn, K_dn_half = w_packed.shape
    K_dn = K_dn_half * 2
    print(f"down_proj: (E={E_all}, N={N_dn}, K={K_dn})")

    k = 4
    expert_ids = torch.tensor([3, 17, 45, 91], dtype=torch.long, device="cuda")
    w_sub_packed = w_packed.index_select(0, expert_ids).contiguous()
    w_sub_scale = w_scale.index_select(0, expert_ids).contiguous()

    # Simulate realistic per-expert activation magnitude variation by
    # scaling each expert's activation differently (matches the real
    # post-SwiGLU distribution where different experts produce outputs
    # of different scales).
    print("\n--- scenarios (each row = a seed/magnitude pattern) ---")
    print(
        f"{'scenario':<24} {'A_rel(vs ref)':>14} {'B_rel(vs ref)':>14} "
        f"{'A-B drift':>12}"
    )

    scenarios = [
        ("uniform (same magn)",    [1.0, 1.0, 1.0, 1.0]),
        ("mild skew (2x outlier)", [0.5, 1.0, 1.0, 2.0]),
        ("strong skew (5x)",       [0.2, 0.5, 1.0, 5.0]),
        ("extreme (20x)",          [0.1, 0.5, 1.0, 20.0]),
    ]

    max_drift = 0.0
    for name, magnitudes in scenarios:
        torch.manual_seed(hash(name) & 0xFFFFFFFF)
        # k separate activations (1, K) with specified magnitudes
        per_expert_x = []
        for m in magnitudes:
            x = torch.randn(1, K_dn, dtype=torch.bfloat16, device="cuda") * 0.1 * m
            per_expert_x.append(x)

        # Reference: full-precision bf16 matmul
        # Dequant the k-expert weight slab
        w_bf16 = dequant_3d_from_nvfp4(w_sub_packed, w_sub_scale, w_tensor_scale)
        ref = torch.stack([
            torch.matmul(per_expert_x[i].float(), w_bf16[i].float().T).to(torch.bfloat16)
            for i in range(k)
        ], dim=0)  # (k, 1, N)

        # Variant A: scalar alpha — stack then quantize once
        x_stacked = torch.cat(per_expert_x, dim=0)  # (k, K)
        a_packed_A, a_bs_A, a_ts_A = quantize_2d_to_nvfp4(x_stacked)
        out_A = nvfp4_linear_3d_prequantized(
            a_packed_A, a_bs_A, a_ts_A,
            w_sub_packed, w_sub_scale, w_tensor_scale,
        )  # (k, 1, N) — but a_stacked has M=k rows so kernel sees M=k per expert
        # We need the diagonal: out_A[e, e, :] since each expert should see its own row.
        # Actually the kernel computes a full (k, M=k, N) because the
        # activation is (k, K). We want each expert to operate on its
        # OWN row — so we should run the kernel k times with M=1 each,
        # OR use a_stacked[e:e+1] per expert.
        #
        # For the A/B test, quantize_once-then-per-expert-slice is what
        # we actually care about: one quantize → the scalar is the
        # global-batch scale, then slice (1, K) for each expert.
        refined_out_A = []
        for e in range(k):
            out_e = nvfp4_linear_3d_prequantized(
                a_packed_A[e:e+1], a_bs_A[e:e+1], a_ts_A,
                w_sub_packed[e:e+1], w_sub_scale[e:e+1], w_tensor_scale,
            )  # (1, 1, N)
            refined_out_A.append(out_e[0])
        out_A = torch.stack(refined_out_A, dim=0)  # (k, 1, N)

        # Variant B: per-expert alpha — quantize each (1, K) separately
        out_B = []
        for e in range(k):
            a_packed_e, a_bs_e, a_ts_e = quantize_2d_to_nvfp4(per_expert_x[e])
            out_e = nvfp4_linear_prequantized(
                a_packed_e, a_bs_e, a_ts_e,
                w_sub_packed[e], w_sub_scale[e], w_tensor_scale,
            )
            out_B.append(out_e)
        out_B = torch.stack(out_B, dim=0)  # (k, 1, N)

        ref_norm = ref.float().abs().mean().item() + 1e-8
        rel_A = (out_A.float() - ref.float()).abs().mean().item() / ref_norm
        rel_B = (out_B.float() - ref.float()).abs().mean().item() / ref_norm

        drift_ab = abs(rel_A - rel_B) / (rel_B + 1e-8)
        max_drift = max(max_drift, drift_ab)

        print(f"{name:<24} {rel_A:>14.6f} {rel_B:>14.6f} {drift_ab*100:>10.2f} %")

    print(f"\nmax A/B drift across scenarios: {max_drift*100:.2f}%  (gate: 5%)")
    if max_drift < 0.05:
        print("→ Phase 4 PASS: scalar alpha is within 5% of per-expert alpha. "
              "Ship scalar-alpha variant.")
        return 0
    print("→ Phase 4 FAIL: scalar alpha drifts > 5%. Ship per-expert-alpha variant.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
