"""Gemma4TextExpertsNVFP4: NVFP4 expert path for Gemma 4 MoE.

Two roles:

1. **Load-time buffer holder.** Subclasses ``Gemma4TextExperts`` so the
   ``isinstance`` check in ``Gemma4PreTrainedModel._init_weights`` still
   matches. All NVFP4 tensors are registered as persistent buffers
   (not Parameters) because uint8 packed weights are non-float and
   incompatible with ``nn.Parameter``.

2. **Inference forward.** ``forward()`` dispatches on
   ``hidden_states.shape[0]``: M=1 decode routes through the 3D batched
   NVFP4 kernel (``nvfp4_linear_3d_prequantized``); M>1 prefill routes
   through the per-expert 2D loop (``nvfp4_linear_prequantized``). Both
   keep FP4 weights packed on GPU — no dequant round trip.

Required env var for GB10 (sm_121)::

    TRITON_OVERRIDE_ARCH=sm120

Without this, Triton's ``ScaledBlockedToMMA`` MLIR pattern hard-rejects
the compute capability and ``tl.dot_scaled`` falls through to a slow
decomposition.
"""
from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

from aeo_quant.gpu.quant import NVFP4_BLOCK_SIZE


_MOE_TRACE = os.environ.get("AEO_MOE_TRACE") == "1"


def _moe_range(name: str):
    if _MOE_TRACE:
        return torch.cuda.nvtx.range(name)
    return nullcontext()


class Gemma4TextExpertsNVFP4(Gemma4TextExperts):
    """Load-time container + NVFP4 forward for Gemma 4 experts.

    Registers persistent buffers with the right shapes and dtypes so
    that ``from_pretrained`` can load the NVFP4 checkpoint tensors.
    ``forward()`` dispatches decode vs prefill on input shape; both
    paths run the FP4 weights through our Triton ``tl.dot_scaled``
    kernel with no dequantization round trip.
    """

    def __init__(self, config):
        nn.Module.__init__(self)
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_activation]

        ne = self.num_experts
        im = self.intermediate_dim
        hd = self.hidden_dim
        bs = NVFP4_BLOCK_SIZE

        # Packed uint8 weights (two FP4 nibbles per byte)
        self.register_buffer(
            "gate_up_proj",
            torch.empty(ne, 2 * im, hd // 2, dtype=torch.uint8),
            persistent=True,
        )
        self.register_buffer(
            "down_proj",
            torch.empty(ne, hd, im // 2, dtype=torch.uint8),
            persistent=True,
        )

        # FP8 E4M3 block scales (one per micro-block of 16 elements)
        self.register_buffer(
            "gate_up_proj_scale",
            torch.empty(ne, 2 * im, hd // bs, dtype=torch.float8_e4m3fn),
            persistent=True,
        )
        self.register_buffer(
            "down_proj_scale",
            torch.empty(ne, hd, im // bs, dtype=torch.float8_e4m3fn),
            persistent=True,
        )

        # FP32 per-tensor global scales (scalars)
        self.register_buffer(
            "gate_up_proj_scale_2",
            torch.tensor(0.0, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "down_proj_scale_2",
            torch.tensor(0.0, dtype=torch.float32),
            persistent=True,
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """Native NVFP4 expert forward.

        Shape-regime dispatch: M=1 (decode) runs through the 3D
        batched kernel; M>1 (prefill) stays on the per-expert 2D loop.
        Both paths are functionally equivalent and produce bit-exact
        output; the 3D path trades k per-expert launches for one
        fused launch per projection at decode shapes where launch
        overhead dominates.
        """
        if hidden_states.shape[0] == 1:
            return self._forward_decode_3d(
                hidden_states, top_k_index, top_k_weights,
            )
        return self._forward_prefill(
            hidden_states, top_k_index, top_k_weights,
        )

    def _forward_decode_3d(self, hidden_states, top_k_index, top_k_weights):
        """M=1 decode path using the 3D batched NVFP4 kernel.

        gate_up: shared activation (1, hidden) across k experts → one
        kernel launch producing (k, 1, 2*im).
        down:    per-expert activation (k, im) → one kernel launch
        producing (k, 1, hidden).

        Combine applies top_k_weights and sums along the k axis.
        """
        from aeo_quant.gpu.nvfp4_matmul import nvfp4_linear_3d_prequantized
        from aeo_quant.gpu.quant import quantize_2d_to_nvfp4

        # (1, top_k) → (top_k,). For M=1 there is exactly one row of
        # routing information; we operate on it directly rather than
        # building the full expert_mask / expert_hit scaffolding.
        expert_ids = top_k_index[0]  # (top_k,)

        with _moe_range("nvfp4_moe_3d_prequant_hidden"):
            h_packed, h_block_scale, h_tensor_scale = quantize_2d_to_nvfp4(
                hidden_states,
            )

        with _moe_range("nvfp4_moe_3d_gate_up"):
            w_gu_packed = self.gate_up_proj.index_select(0, expert_ids)
            w_gu_bs = self.gate_up_proj_scale.index_select(0, expert_ids)
            gate_up = nvfp4_linear_3d_prequantized(
                h_packed, h_block_scale, h_tensor_scale,
                w_gu_packed, w_gu_bs, self.gate_up_proj_scale_2,
            )  # (k, 1, 2*im)
            gate, up = gate_up.chunk(2, dim=-1)
            current = self.act_fn(gate) * up  # (k, 1, im)

        with _moe_range("nvfp4_moe_3d_prequant_down"):
            # (k, 1, im) → (k, im) for quantize_2d; then unsqueeze to
            # (k, 1, im//2) per-expert activation layout for the kernel.
            current_2d = current.squeeze(1)
            d_packed_2d, d_block_scale_2d, d_tensor_scale = quantize_2d_to_nvfp4(
                current_2d,
            )
            d_packed = d_packed_2d.unsqueeze(1)          # (k, 1, im//2)
            d_block_scale = d_block_scale_2d.unsqueeze(1)  # (k, 1, im//16)

        with _moe_range("nvfp4_moe_3d_down"):
            w_d_packed = self.down_proj.index_select(0, expert_ids)
            w_d_bs = self.down_proj_scale.index_select(0, expert_ids)
            down_out = nvfp4_linear_3d_prequantized(
                d_packed, d_block_scale, d_tensor_scale,
                w_d_packed, w_d_bs, self.down_proj_scale_2,
            )  # (k, 1, hidden)

        with _moe_range("nvfp4_moe_3d_combine"):
            # top_k_weights: (1, k) → (k, 1, 1) to broadcast over down_out.
            weights = top_k_weights[0, :, None, None]
            out = (down_out * weights).sum(dim=0)  # (1, hidden)

        return out.to(hidden_states.dtype)

    def _forward_prefill(self, hidden_states, top_k_index, top_k_weights):
        """Per-expert 2D path used for prefill (M > 1).

        Optimization: the full ``hidden_states`` is quantized once
        up-front, and each expert slices the packed form rather than
        re-quantizing its own token slice.  Saves roughly (top_k - 1)
        activation-quant passes per layer per token — with 10+ small
        elementwise launches each, meaningful in the decode path.
        """
        from aeo_quant.gpu.nvfp4_matmul import (
            nvfp4_linear,
            nvfp4_linear_prequantized,
        )
        from aeo_quant.gpu.quant import quantize_2d_to_nvfp4

        with _moe_range("nvfp4_moe_prequant"):
            h_packed, h_block_scale, h_tensor_scale = quantize_2d_to_nvfp4(
                hidden_states
            )

        final_hidden_states = torch.zeros_like(hidden_states)
        with _moe_range("nvfp4_moe_route"), torch.no_grad():
            expert_mask = nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            with _moe_range("nvfp4_moe_gate_up"):
                gate_up = nvfp4_linear_prequantized(
                    h_packed[token_idx],
                    h_block_scale[token_idx],
                    h_tensor_scale,
                    self.gate_up_proj[expert_idx],
                    self.gate_up_proj_scale[expert_idx],
                    self.gate_up_proj_scale_2,
                )
                gate, up = gate_up.chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up

            with _moe_range("nvfp4_moe_down"):
                # down_proj input is per-expert (distinct gate*up output
                # per expert), so this still uses the runtime-quant path.
                current_hidden_states = nvfp4_linear(
                    current_hidden_states,
                    self.down_proj[expert_idx],
                    self.down_proj_scale[expert_idx],
                    self.down_proj_scale_2,
                )

            with _moe_range("nvfp4_moe_combine"):
                current_hidden_states *= top_k_weights[
                    token_idx, top_k_pos, None
                ]
                final_hidden_states.index_add_(
                    0, token_idx,
                    current_hidden_states.to(final_hidden_states.dtype),
                )

        return final_hidden_states
