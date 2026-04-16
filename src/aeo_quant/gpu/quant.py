"""Per-output-channel max-abs FP8 quantization utilities.

Two entry points:
- ``quantize_3d_to_fp8``: fused 3D MoE expert tensors ``(E, out, in)``
- ``quantize_2d_to_fp8``: standard 2D weight matrices ``(out, in)``

Both produce float8_e4m3fn weights with per-output-row bfloat16 scales,
suitable for ``torch._scaled_mm`` with RowWise scaling.
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


def quantize_2d_to_fp8(weight_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row max-abs FP8 quant of a 2D weight matrix.

    Args:
        weight_bf16: shape ``(out_dim, in_dim)``, dtype ``bfloat16``.

    Returns:
        A tuple ``(weight_fp8, scale_bf16)`` where:
          - ``weight_fp8``: shape ``(out_dim, in_dim)``, dtype ``float8_e4m3fn``
          - ``scale_bf16``: shape ``(out_dim, 1)``, dtype ``bfloat16``
    """
    if weight_bf16.ndim != 2:
        raise ValueError(
            f"quantize_2d_to_fp8 expects a 2D tensor (out_dim, in_dim); "
            f"got shape {tuple(weight_bf16.shape)}"
        )
    max_abs = weight_bf16.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / FP8_E4M3_MAX).clamp(min=1e-8).to(torch.bfloat16)
    weight_fp8 = (
        weight_bf16.to(torch.float32) / scale.to(torch.float32)
    ).to(torch.float8_e4m3fn)
    return weight_fp8, scale


def dequantize_2d_from_fp8(
    weight_fp8: torch.Tensor, scale_bf16: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`quantize_2d_to_fp8` for testing/validation only."""
    return weight_fp8.to(torch.bfloat16) * scale_bf16
