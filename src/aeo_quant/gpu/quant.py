"""FP8 and NVFP4 quantization utilities.

FP8 (per-output-channel max-abs):
- ``quantize_3d_to_fp8``: fused 3D MoE expert tensors ``(E, out, in)``
- ``quantize_2d_to_fp8``: standard 2D weight matrices ``(out, in)``

NVFP4 (two-level microscaling, E2M1 with FP8 block scales):
- ``quantize_3d_to_nvfp4`` / ``dequant_3d_from_nvfp4``: 3D MoE expert tensors
- ``quantize_2d_to_nvfp4`` / ``dequant_2d_from_nvfp4``: 2D weight matrices
"""
from __future__ import annotations

import torch

FP8_E4M3_MAX = 448.0  # max representable in torch.float8_e4m3fn
FP4_E2M1_MAX = 6.0    # max representable in FP4 E2M1
NVFP4_BLOCK_SIZE = 16  # NVFP4 micro-block size (NOT 32 like MXFP4)

# Midpoints between adjacent FP4 E2M1 magnitudes — boundaries for nearest rounding.
# The 8 positive magnitudes are {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
_FP4_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

# Nibble-to-float lookup table (sign-magnitude E2M1 encoding).
_FP4_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,    # nibbles 0-7  (positive)
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # nibbles 8-15 (negative)
])

# Per-device caches of the bounds/LUT tensors. Host→device copies at call
# time would (a) waste ~10µs each on a cudaMemcpy and (b) make the quant
# path uncapturable in a CUDA graph, which disallows unpinned host copies
# mid-capture. Populated lazily on first use per device.
_FP4_BOUNDS_ON_DEVICE: dict[str, torch.Tensor] = {}
_FP4_LUT_ON_DEVICE: dict[str, torch.Tensor] = {}


def _fp4_bounds_for(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _FP4_BOUNDS_ON_DEVICE:
        _FP4_BOUNDS_ON_DEVICE[key] = _FP4_BOUNDS.to(device)
    return _FP4_BOUNDS_ON_DEVICE[key]


def _fp4_lut_for(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _FP4_LUT_ON_DEVICE:
        _FP4_LUT_ON_DEVICE[key] = _FP4_LUT.to(device)
    return _FP4_LUT_ON_DEVICE[key]


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


# ---------------------------------------------------------------------------
# NVFP4 (two-level microscaling: FP4 E2M1 values + FP8 block scales)
# ---------------------------------------------------------------------------


def _round_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round float values to nearest FP4 E2M1, return as uint8 nibbles 0-15.

    Uses searchsorted on 7 midpoint boundaries for O(1) per-element binning.
    Bit 3 encodes sign (sign-magnitude), bits 0-2 encode magnitude index.
    """
    sign = (x < 0).to(torch.uint8) << 3
    mag = x.abs()
    idx = torch.searchsorted(_fp4_bounds_for(mag.device), mag)  # 0..7
    return sign | idx.to(torch.uint8)


def _nibble_to_fp4(nibbles: torch.Tensor) -> torch.Tensor:
    """Convert uint8 nibbles (0-15) to FP4 E2M1 float values via lookup."""
    return _fp4_lut_for(nibbles.device)[nibbles.long()]


def quantize_3d_to_nvfp4(
    weight_bf16: torch.Tensor, block_size: int = NVFP4_BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-level microscaling NVFP4 quantization of a 3D MoE weight tensor.

    Args:
        weight_bf16: shape ``(num_experts, out_dim, in_dim)``, dtype ``bfloat16``.
        block_size: micro-block size for level-1 scaling (default 16).

    Returns:
        A tuple ``(packed_uint8, block_scale_fp8, tensor_scale_fp32)`` where:
          - ``packed_uint8``: shape ``(E, out_dim, in_dim // 2)``, dtype ``uint8``
            — two FP4 nibbles per byte, high nibble first
          - ``block_scale_fp8``: shape ``(E, out_dim, in_dim // block_size)``,
            dtype ``float8_e4m3fn`` — level-1 per-block scale
          - ``tensor_scale_fp32``: scalar ``float32`` — level-2 per-tensor scale
    """
    if weight_bf16.ndim != 3:
        raise ValueError(
            f"quantize_3d_to_nvfp4 expects a 3D tensor (E, out_dim, in_dim); "
            f"got shape {tuple(weight_bf16.shape)}"
        )
    E, out_dim, in_dim = weight_bf16.shape
    if in_dim % block_size != 0:
        raise ValueError(
            f"in_dim ({in_dim}) must be divisible by block_size ({block_size})"
        )

    # Level 2: per-tensor global scale (fp32 scalar)
    tensor_amax = weight_bf16.abs().amax()
    tensor_scale = (tensor_amax / (FP4_E2M1_MAX * FP8_E4M3_MAX)).clamp(min=1e-12)
    w_prescaled = weight_bf16.float() / tensor_scale.float()

    # Level 1: per-block micro-scale (FP8 E4M3, max=448.0)
    w = w_prescaled.reshape(E, out_dim, -1, block_size)
    block_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    block_scale = (block_max / FP4_E2M1_MAX).to(torch.float8_e4m3fn)

    # Quantize to FP4 range
    w_norm = (w / block_scale.float()).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
    nibbles = _round_to_fp4_e2m1(w_norm)

    # Pack two nibbles per byte, high nibble first
    nibbles = nibbles.reshape(E, out_dim, -1, block_size // 2, 2)
    packed = ((nibbles[..., 0] << 4) | nibbles[..., 1]).to(torch.uint8)
    packed = packed.reshape(E, out_dim, in_dim // 2)

    return packed, block_scale.squeeze(-1), tensor_scale


def dequant_3d_from_nvfp4(
    packed_uint8: torch.Tensor,
    block_scale_fp8: torch.Tensor,
    tensor_scale_fp32: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Inverse of :func:`quantize_3d_to_nvfp4`.

    Returns a bf16 tensor of shape ``(E, out_dim, in_dim)``.
    """
    E, out_dim, packed_in = packed_uint8.shape
    in_dim = packed_in * 2

    # Unpack nibbles
    high = (packed_uint8 >> 4).to(torch.uint8)
    low = (packed_uint8 & 0x0F).to(torch.uint8)
    nibbles = torch.stack([high, low], dim=-1).reshape(E, out_dim, in_dim)

    # Convert nibbles to float
    values = _nibble_to_fp4(nibbles)

    # Apply two-level dequant: value * block_scale * tensor_scale
    values = values.reshape(E, out_dim, -1, block_size)
    result = values * block_scale_fp8.unsqueeze(-1).float() * tensor_scale_fp32.float()
    return result.reshape(E, out_dim, in_dim).to(torch.bfloat16)


def quantize_2d_to_nvfp4(
    weight_bf16: torch.Tensor, block_size: int = NVFP4_BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-level microscaling NVFP4 quantization of a 2D weight matrix.

    Delegates to :func:`quantize_3d_to_nvfp4` with an unsqueezed leading dim.
    """
    if weight_bf16.ndim != 2:
        raise ValueError(
            f"quantize_2d_to_nvfp4 expects a 2D tensor (out_dim, in_dim); "
            f"got shape {tuple(weight_bf16.shape)}"
        )
    packed, block_scale, tensor_scale = quantize_3d_to_nvfp4(
        weight_bf16.unsqueeze(0), block_size
    )
    return packed.squeeze(0), block_scale.squeeze(0), tensor_scale


def dequant_2d_from_nvfp4(
    packed_uint8: torch.Tensor,
    block_scale_fp8: torch.Tensor,
    tensor_scale_fp32: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Inverse of :func:`quantize_2d_to_nvfp4`.

    Returns a bf16 tensor of shape ``(out_dim, in_dim)``.
    """
    return dequant_3d_from_nvfp4(
        packed_uint8.unsqueeze(0), block_scale_fp8.unsqueeze(0),
        tensor_scale_fp32, block_size
    ).squeeze(0)
