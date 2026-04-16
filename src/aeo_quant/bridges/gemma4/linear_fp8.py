"""LinearFP8: drop-in FP8 replacement for nn.Linear using torch._scaled_mm.

Stores weight as (out, in) float8_e4m3fn with a (1, out) fp32 scale buffer.
Forward runs through the same _fp8_linear path as the MoE experts — per-row
dynamic input quantization, RowWise weight scaling, bf16 output.
"""
from __future__ import annotations

import torch
import torch.nn as nn

FP8_MAX = 448.0


def fp8_linear(x_bf16: torch.Tensor, w_fp8: torch.Tensor,
               scale_w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """FP8-native matmul with per-row dynamic input quantization.

    Args:
        x_bf16:  (..., K) bf16 input — any leading batch dims are flattened.
        w_fp8:   (N, K) fp8 weight. .t() gives (K, N) column-major for _scaled_mm.
        scale_w: (1, N) fp32 per-output-channel weight scale (RowWise).
        bias:    optional (N,) bf16 bias, added after matmul.

    Returns:
        (..., N) bf16 output with the same leading dimensions as input.
    """
    orig_shape = x_bf16.shape
    if x_bf16.ndim > 2:
        x_bf16 = x_bf16.reshape(-1, orig_shape[-1])

    x_amax = x_bf16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4).to(torch.float32)
    x_scale = x_amax / FP8_MAX
    x_fp8 = (x_bf16.to(torch.float32) / x_scale).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(
        x_fp8, w_fp8.t(), scale_a=x_scale, scale_b=scale_w,
        out_dtype=torch.bfloat16,
    )
    if bias is not None:
        out = out + bias

    if len(orig_shape) > 2:
        out = out.reshape(*orig_shape[:-1], out.shape[-1])
    return out


class LinearFP8(nn.Module):
    """Drop-in replacement for nn.Linear that stores FP8 weights.

    Quantizes a bf16 Linear's weight at construction time via
    ``quantize_2d_to_fp8`` and stores the result as a non-grad Parameter
    plus a fp32 scale buffer.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(1, out_features, dtype=torch.float32, device=device),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16, device=device),
                requires_grad=False,
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> LinearFP8:
        """Quantize an existing nn.Linear in-place."""
        from aeo_quant.gpu.quant import quantize_2d_to_fp8

        has_bias = linear.bias is not None
        mod = cls(linear.in_features, linear.out_features,
                  bias=has_bias, device=linear.weight.device)

        w_fp8, scale_bf16 = quantize_2d_to_fp8(linear.weight.data)
        mod.weight = nn.Parameter(w_fp8, requires_grad=False)
        # scale: (out, 1) bf16 from quant → (1, out) fp32 for RowWise _scaled_mm
        mod.weight_scale = scale_bf16.squeeze(-1).unsqueeze(0).float().contiguous()

        if has_bias:
            mod.bias = nn.Parameter(linear.bias.data.to(torch.bfloat16), requires_grad=False)

        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp8_linear(x, self.weight, self.weight_scale, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, dtype=float8_e4m3fn"
        )
