"""Class-swap loader for the self-built FP8 Gemma 4 model.

Provides a context manager that temporarily swaps
``transformers.models.gemma4.modeling_gemma4.Gemma4TextExperts`` for our
``Gemma4TextExpertsFP8`` subclass during ``from_pretrained``, then restores
the original class on exit (even if loading raises).

We use a context manager rather than a permanent monkey-patch because the
swap is needed only during model construction (specifically before
``Gemma4TextDecoderLayer.__init__`` runs at modeling_gemma4.py:1349); leaving
it patched globally would affect any other code that imports
``Gemma4TextExperts`` later in the same process.
"""
from __future__ import annotations

import contextlib

import torch
from transformers import AutoModelForCausalLM
from transformers.models.gemma4 import modeling_gemma4

from .modeling import Gemma4TextExpertsFP8


@contextlib.contextmanager
def gemma4_experts_fp8_class_swap():
    """Temporarily swap ``Gemma4TextExperts`` with ``Gemma4TextExpertsFP8``.

    The original class is restored in a ``finally`` block so that the global
    state is recovered even if the wrapped code raises.
    """
    original = modeling_gemma4.Gemma4TextExperts
    modeling_gemma4.Gemma4TextExperts = Gemma4TextExpertsFP8
    try:
        yield
    finally:
        modeling_gemma4.Gemma4TextExperts = original


def _preconvert_fp8_scales(model):
    """Convert FP8 expert scale buffers to RowWise fp32 layout in-place.

    Checkpoint stores scales as ``(E, out_dim, 1) bf16``.
    ``_scaled_mm`` needs ``scale_b`` as ``(1, out_dim) fp32`` per-expert slice.
    Pre-conversion reshapes to ``(E, 1, out_dim) fp32`` so forward() can index
    directly without per-call squeeze/unsqueeze/float/contiguous.

    bf16 → fp32 is mathematically lossless.  +35 MB for a 24-layer model.
    """
    for module in model.modules():
        if isinstance(module, Gemma4TextExpertsFP8):
            module.gate_up_proj_scale = (
                module.gate_up_proj_scale.squeeze(-1).unsqueeze(1).float().contiguous()
            )
            module.down_proj_scale = (
                module.down_proj_scale.squeeze(-1).unsqueeze(1).float().contiguous()
            )


def load_gemma4_fp8(model_id_or_path, **from_pretrained_kwargs):
    """Load a self-built FP8 Gemma 4 checkpoint with the class swap active.

    Defaults ``dtype=torch.bfloat16`` and ``device_map="cuda"`` unless the
    caller overrides them via ``from_pretrained_kwargs``.  Wraps the model
    with ``torch.compile(mode="reduce-overhead")`` to reduce kernel launch
    overhead (+12% decode throughput, ~1 s one-time warmup).
    """
    from_pretrained_kwargs.setdefault("dtype", torch.bfloat16)
    from_pretrained_kwargs.setdefault("device_map", "cuda")
    with gemma4_experts_fp8_class_swap():
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path, **from_pretrained_kwargs
        )
    _preconvert_fp8_scales(model)
    return torch.compile(model, mode="reduce-overhead", dynamic=False)
