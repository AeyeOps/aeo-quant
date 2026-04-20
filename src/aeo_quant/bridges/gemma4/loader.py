"""Class-swap loader for the self-built FP8 and NVFP4 Gemma 4 models.

Provides context managers that temporarily swap
``transformers.models.gemma4.modeling_gemma4.Gemma4TextExperts`` for our
quantized subclasses during ``from_pretrained``, then restore the original
class on exit (even if loading raises).

FP8: loads the FP8 checkpoint into ``Gemma4TextExpertsFP8`` Parameters,
pre-converts block scales to the RowWise fp32 layout ``_scaled_mm``
expects.

NVFP4: loads the NVFP4 checkpoint into ``Gemma4TextExpertsNVFP4``
buffers (uint8 packed weights + fp8 block scales + fp32 tensor scale).
Inference routes through our Triton ``tl.dot_scaled`` kernel
(``_nvfp4_matmul_kernel_3d`` for decode, ``_nvfp4_matmul_kernel`` for
prefill). Requires ``TRITON_OVERRIDE_ARCH=sm120`` on sm_121 (GB10) so
the kernel lowers to ``mma.sync...kind::mxf4nvf4`` instead of
decomposition.
"""
from __future__ import annotations

import contextlib

import torch
from transformers import AutoModelForCausalLM
from transformers.models.gemma4 import modeling_gemma4

from .modeling import Gemma4TextExpertsFP8
from .modeling_nvfp4 import Gemma4TextExpertsNVFP4


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


def load_gemma4(model_id_or_path, *, quant_format="fp8", **from_pretrained_kwargs):
    """Load a quantized Gemma 4 model, dispatching by format.

    Args:
        model_id_or_path: path to the checkpoint directory.
        quant_format: ``"fp8"`` (default) or ``"nvfp4"``.
    """
    if quant_format == "nvfp4":
        return load_gemma4_nvfp4(model_id_or_path, **from_pretrained_kwargs)
    return load_gemma4_fp8(model_id_or_path, **from_pretrained_kwargs)


def load_gemma4_fp8(model_id_or_path, **from_pretrained_kwargs):
    """Load a self-built FP8 Gemma 4 checkpoint with the class swap active.

    Defaults ``dtype=torch.bfloat16`` and ``device_map="cuda"`` unless the
    caller overrides them via ``from_pretrained_kwargs``.
    """
    from_pretrained_kwargs.setdefault("dtype", torch.bfloat16)
    from_pretrained_kwargs.setdefault("device_map", "cuda")
    with gemma4_experts_fp8_class_swap():
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path, **from_pretrained_kwargs
        )
    _preconvert_fp8_scales(model)
    return model


# ---------------------------------------------------------------------------
# NVFP4 loader
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _gemma4_experts_nvfp4_class_swap():
    """Temporarily swap ``Gemma4TextExperts`` with ``Gemma4TextExpertsNVFP4``."""
    original = modeling_gemma4.Gemma4TextExperts
    modeling_gemma4.Gemma4TextExperts = Gemma4TextExpertsNVFP4
    try:
        yield
    finally:
        modeling_gemma4.Gemma4TextExperts = original


def load_gemma4_nvfp4(model_id_or_path, **from_pretrained_kwargs):
    """Load a self-built NVFP4 Gemma 4 checkpoint.

    Keeps FP4 weights packed in GPU memory for the lifetime of the model.
    Inference routes through :func:`aeo_quant.gpu.nvfp4_matmul.nvfp4_linear`
    (prefill) and :func:`aeo_quant.gpu.nvfp4_matmul.nvfp4_linear_3d_prequantized`
    (decode), both backed by Triton ``tl.dot_scaled``.

    ``TRITON_OVERRIDE_ARCH=sm120`` is required on sm_121 (GB10) — without
    it the kernel falls through to a slow scaled-dot decomposition.
    """
    import time

    from aeo_quant.gpu.memory import mem_report

    from_pretrained_kwargs.setdefault("dtype", torch.bfloat16)
    from_pretrained_kwargs.setdefault("device_map", "cuda")

    mem_report("nvfp4_load:start")
    t_load = time.time()
    with _gemma4_experts_nvfp4_class_swap():
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path, **from_pretrained_kwargs
        )
    print(f"[nvfp4] from_pretrained done in {time.time() - t_load:.1f}s", flush=True)
    mem_report("nvfp4_load:done")

    return model
