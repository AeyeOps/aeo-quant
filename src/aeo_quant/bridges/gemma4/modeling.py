"""Gemma4TextExpertsFP8: drop-in FP8 replacement for transformers' Gemma4TextExperts.

Subclasses Gemma4TextExperts so the isinstance check in Gemma4PreTrainedModel._init_weights
(modeling_gemma4.py:1481) still matches. __init__ calls nn.Module.__init__ directly to
skip the parent's bf16 Parameter allocation, which would otherwise briefly double memory
before we delete and replace it. Forward uses torch._scaled_mm for FP8-native matmul
with per-output-channel (RowWise) weight scaling and per-row dynamic input quantization.
"""
from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

# Opt-in NVTX range markers for profiling. Zero cost when AEO_MOE_TRACE is unset:
# _moe_range returns contextlib.nullcontext, which has no CUDA or host overhead.
_MOE_TRACE = os.environ.get("AEO_MOE_TRACE") == "1"


def _moe_range(name: str):
    if _MOE_TRACE:
        return torch.cuda.nvtx.range(name)
    return nullcontext()


class Gemma4TextExpertsFP8(Gemma4TextExperts):
    """Drop-in FP8 replacement for Gemma4TextExperts.

    Same external API (forward signature, attribute names used by parent
    classes), but `gate_up_proj` and `down_proj` are float8_e4m3fn Parameters
    and there are two new bf16 scale buffers. Forward dequantizes per-expert
    per-call into transient bf16 weights.

    Scale buffer naming is flat (`gate_up_proj_scale`, not `gate_up_proj.weight_scale`)
    to avoid colliding with the gate_up_proj Parameter's name.
    """

    def __init__(self, config):
        # Skip parent __init__'s bf16 Parameter allocation by calling
        # nn.Module.__init__ directly. We re-implement only the attributes
        # forward() actually touches.
        nn.Module.__init__(self)
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_activation]

        ne = self.num_experts
        im = self.intermediate_dim
        hd = self.hidden_dim

        # FP8 weight Parameters -- loaded from checkpoint by from_pretrained.
        self.gate_up_proj = nn.Parameter(
            torch.empty(ne, 2 * im, hd, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(ne, hd, im, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )

        # bf16 per-output-channel scale buffers -- loaded from checkpoint via
        # the standard state_dict mechanism (persistent=True is the default but
        # we set it explicitly for clarity). Flat naming so they don't collide
        # with the gate_up_proj / down_proj Parameter names.
        self.register_buffer(
            "gate_up_proj_scale",
            torch.empty(ne, 2 * im, 1, dtype=torch.bfloat16),
            persistent=True,
        )
        self.register_buffer(
            "down_proj_scale",
            torch.empty(ne, hd, 1, dtype=torch.bfloat16),
            persistent=True,
        )

    @staticmethod
    def _fp8_linear(x_bf16, w_fp8, scale_w_1xN):
        """FP8-native matmul with per-row dynamic input quantization.

        x_bf16:     (M, K) bf16 input
        w_fp8:      (N, K) fp8 weight (row-major). .t() gives (K, N) column-major
                    (stride(0)==1), which _scaled_mm requires for B.
        scale_w_1xN: (1, N) fp32 per-output-channel weight scale (RowWise scale_b).

        Input is quantized per-row against its own amax so each row uses FP8's
        full range. scale_a=(M, 1) fp32.
        """
        fp8_max = 448.0
        x_amax = x_bf16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4).to(torch.float32)
        x_scale = x_amax / fp8_max  # (M, 1) fp32
        x_fp8 = (x_bf16.to(torch.float32) / x_scale).to(torch.float8_e4m3fn)
        return torch._scaled_mm(
            x_fp8, w_fp8.t(), scale_a=x_scale, scale_b=scale_w_1xN,
            out_dtype=torch.bfloat16,
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with _moe_range("fp8_moe_route"), torch.no_grad():
            expert_mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            with _moe_range("fp8_moe_gate_up"):
                gate_up = self._fp8_linear(
                    current_state,
                    self.gate_up_proj[expert_idx],
                    self.gate_up_proj_scale[expert_idx],
                )
                gate, up = gate_up.chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up

            with _moe_range("fp8_moe_down"):
                current_hidden_states = self._fp8_linear(
                    current_hidden_states,
                    self.down_proj[expert_idx],
                    self.down_proj_scale[expert_idx],
                )

            with _moe_range("fp8_moe_combine"):
                current_hidden_states *= top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(
                    0, token_idx, current_hidden_states.to(final_hidden_states.dtype),
                )

        return final_hidden_states
