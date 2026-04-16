"""Per-(expert, output channel) max-abs FP8 quantization for fused 3D MoE weights.

This module implements the pure-tensor-math side of the self-built Gemma 4 FP8
checkpoint path: given a fused 3D expert weight tensor of shape
``(num_experts, out_dim, in_dim)`` in bfloat16, it produces a float8_e4m3fn
weight tensor with the same shape and a bfloat16 per-(expert, output_channel)
scale tensor of shape ``(num_experts, out_dim, 1)``.

The FP8 weight bytes match the LargitData ``gemma-4-26b-a4b-it-fp8`` checkpoint
layout, so the weights themselves are interchangeable. The scale tensor naming
intentionally differs: we use flat keys like ``gate_up_proj_scale`` rather than
LargitData's dotted ``gate_up_proj.weight_scale``, because the dotted form
cannot bind to a Parameter that is itself named ``gate_up_proj`` (PyTorch
cannot simultaneously treat ``gate_up_proj`` as both a Parameter and a
sub-module). Flat buffer names let ``state_dict`` load the scales via the
standard mechanism without any custom ``_load_from_state_dict`` override.
"""
from __future__ import annotations

import torch

FP8_E4M3_MAX = 448.0  # max representable in torch.float8_e4m3fn


def quantize_3d_to_fp8(weight_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(expert, output_channel) max-abs FP8 quant of a 3D MoE weight tensor.

    Args:
        weight_bf16: shape ``(num_experts, out_dim, in_dim)``, dtype ``bfloat16``.

    Returns:
        A tuple ``(weight_fp8, scale_bf16)`` where:
          - ``weight_fp8``: shape ``(num_experts, out_dim, in_dim)``, dtype ``float8_e4m3fn``
          - ``scale_bf16``: shape ``(num_experts, out_dim, 1)``, dtype ``bfloat16``

    The intermediate divide runs in float32 on purpose to avoid bf16 underflow
    before the cast to float8_e4m3fn.
    """
    if weight_bf16.ndim != 3:
        raise ValueError(
            f"quantize_3d_to_fp8 expects a 3D tensor (num_experts, out_dim, in_dim); "
            f"got shape {tuple(weight_bf16.shape)}"
        )
    max_abs = weight_bf16.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / FP8_E4M3_MAX).clamp(min=1e-8).to(torch.bfloat16)
    weight_fp8 = (
        weight_bf16.to(torch.float32) / scale.to(torch.float32)
    ).to(torch.float8_e4m3fn)
    return weight_fp8, scale


def dequantize_3d_from_fp8(
    weight_fp8: torch.Tensor, scale_bf16: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`quantize_3d_to_fp8` for testing/validation only."""
    return weight_fp8.to(torch.bfloat16) * scale_bf16
