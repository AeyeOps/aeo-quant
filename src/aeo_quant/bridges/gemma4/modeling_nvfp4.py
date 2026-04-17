"""Gemma4TextExpertsNVFP4: native-NVFP4 expert path for Gemma 4 MoE.

Two roles:

1. **Load-time buffer holder.** Subclasses ``Gemma4TextExperts`` so the
   ``isinstance`` check in ``Gemma4PreTrainedModel._init_weights`` still
   matches.  All NVFP4 tensors are registered as persistent buffers
   (not Parameters) because uint8 packed weights are non-float and
   incompatible with ``nn.Parameter``.

2. **Inference path when ``AEO_NVFP4_NATIVE=1``.**  ``forward()`` routes
   each selected expert through :func:`aeo_quant.gpu.nvfp4_matmul.nvfp4_linear`,
   keeping FP4 weights in GPU memory (no dequant-to-FP8 round trip).

When ``AEO_NVFP4_NATIVE`` is unset or 0, the loader converts NVFP4
buffers to FP8 Parameters after ``from_pretrained`` and swaps
``__class__`` to ``Gemma4TextExpertsFP8`` — ``forward()`` here is
never called in that case.

Critical env var for GB10 (sm_121)::

    TRITON_OVERRIDE_ARCH=sm120

Without this, Triton's ``ScaledBlockedToMMA`` MLIR pattern hard-rejects
the compute capability and ``tl.dot_scaled`` falls through to a slow
decomposition.  See ``kb/nvfp4-blackwell-research.md`` "second deep dive".
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
    """Load-time container + native-NVFP4 forward for Gemma 4 experts.

    Registers persistent buffers with the right shapes and dtypes so
    that ``from_pretrained`` can load the NVFP4 checkpoint tensors.
    ``forward()`` runs the per-expert NVFP4 matmul through our Triton
    kernel when this class is still installed at inference time —
    i.e., when the loader was instructed (via ``AEO_NVFP4_NATIVE=1``)
    to skip the convert-to-FP8 step.
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

        Routes each selected expert's gate_up_proj and down_proj through
        :func:`nvfp4_linear`.  Mirrors the dispatch and combine logic of
        :class:`Gemma4TextExpertsFP8.forward`.

        Optimization: the full ``hidden_states`` is quantized once
        up-front, and each expert slices the packed form rather than
        re-quantizing its own token slice.  Saves roughly (top_k - 1)
        activation-quant passes per layer per token — with 10+ small
        elementwise launches each, meaningful in the decode path.
        """
        # Lazy imports so the load-only path (no native inference) doesn't
        # need Triton at import time.
        from aeo_quant.gpu.nvfp4_matmul import (
            nvfp4_linear,
            nvfp4_linear_prequantized,
        )
        from aeo_quant.gpu.quant import quantize_2d_to_nvfp4

        # Pre-quantize hidden_states once (reused across every selected
        # expert's gate_up_proj).  Per-expert token slicing then works
        # on the packed form without any more activation-quant work.
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
