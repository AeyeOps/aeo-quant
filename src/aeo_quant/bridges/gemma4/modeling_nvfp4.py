"""Gemma4TextExpertsNVFP4: load-time container for NVFP4 checkpoint weights.

Subclasses Gemma4TextExperts so the isinstance check in
Gemma4PreTrainedModel._init_weights still matches. All tensors are
registered as persistent buffers (not Parameters) because uint8 packed
weights are non-float and incompatible with nn.Parameter.

This class is NOT used for inference — the loader converts NVFP4 buffers
to FP8 Parameters after from_pretrained, then swaps __class__ to
Gemma4TextExpertsFP8 for the proven _scaled_mm inference path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

from aeo_quant.gpu.quant import NVFP4_BLOCK_SIZE


class Gemma4TextExpertsNVFP4(Gemma4TextExperts):
    """Load-time container for NVFP4 checkpoint weights.

    Registers persistent buffers with the right shapes and dtypes so that
    ``from_pretrained`` can load the NVFP4 checkpoint tensors. Forward is
    intentionally not implemented — the loader converts to FP8 after load.
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
        raise RuntimeError(
            "Gemma4TextExpertsNVFP4.forward() should never be called — "
            "the loader must convert to FP8 before inference."
        )
