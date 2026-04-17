"""Class-swap loader for the self-built FP8 and NVFP4 Gemma 4 models.

Provides context managers that temporarily swap
``transformers.models.gemma4.modeling_gemma4.Gemma4TextExperts`` for our
quantized subclasses during ``from_pretrained``, then restore the original
class on exit (even if loading raises).

For NVFP4: loads the checkpoint into ``Gemma4TextExpertsNVFP4`` buffers, then
converts expert weights NVFP4 -> bf16 -> FP8 in-place every load. Inference
uses the identical ``_scaled_mm`` path as the FP8 loader.

There is intentionally no FP8 conversion cache. An earlier design saved the
converted FP8 weights to ``.fp8_cache/`` to skip conversion on subsequent
loads, but the batched conversion optimization made conversion so cheap
(~10s) that cache load (~124s of disk I/O) was always slower than just
reconverting. See ``docs/gemma4-fp8-optimization.md`` for the full story.
"""
from __future__ import annotations

import contextlib
import os

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


def _convert_nvfp4_experts_to_fp8(model):
    """Walk all NVFP4 expert modules, convert to FP8 in-place.

    Processes 16 experts at a time to balance memory safety against kernel
    launch overhead.  Full 128-expert dequant creates ~6.6 GB of int64
    intermediates (nibbles.long() for LUT indexing); batching by 16 keeps
    peak temporary at ~830 MB while reducing CUDA kernel dispatches 16×
    vs per-expert iteration.

    For each Gemma4TextExpertsNVFP4 module:
      1. Batched: dequant NVFP4 -> bf16 -> FP8 + per-channel scale
      2. Replace buffers with FP8 Parameters + scale buffers
      3. Swap __class__ to Gemma4TextExpertsFP8
    """
    import time

    from aeo_quant.gpu.memory import mem_report
    from aeo_quant.gpu.quant import (
        dequant_3d_from_nvfp4,
        quantize_3d_to_fp8,
    )

    batch = 16  # 16 experts at a time: ~830 MB peak temporary

    mem_report("convert:start")
    t_start = time.time()
    layer_idx = 0
    for name, module in model.named_modules():
        if not isinstance(module, Gemma4TextExpertsNVFP4):
            continue

        layer_idx += 1
        t_layer = time.time()

        for proj_name in ("gate_up_proj", "down_proj"):
            packed = getattr(module, proj_name)
            block_scale = getattr(module, f"{proj_name}_scale")
            tensor_scale = getattr(module, f"{proj_name}_scale_2")

            E, out_dim, packed_in = packed.shape
            in_dim = packed_in * 2

            # Pre-allocate output tensors
            w_fp8 = torch.empty(
                E, out_dim, in_dim,
                dtype=torch.float8_e4m3fn, device=packed.device,
            )
            scale_bf16 = torch.empty(
                E, out_dim, 1,
                dtype=torch.bfloat16, device=packed.device,
            )

            # Batched conversion: 16 experts per iteration (16× fewer
            # kernel dispatches than per-expert, ~830 MB peak vs ~6.6 GB
            # for all 128 at once)
            for e in range(0, E, batch):
                end = min(e + batch, E)
                w_chunk = dequant_3d_from_nvfp4(
                    packed[e:end], block_scale[e:end], tensor_scale,
                )
                fp8_chunk, s_chunk = quantize_3d_to_fp8(w_chunk)
                w_fp8[e:end] = fp8_chunk
                scale_bf16[e:end] = s_chunk
                del w_chunk, fp8_chunk, s_chunk

            # Replace: packed buffer -> FP8 Parameter
            delattr(module, proj_name)
            setattr(module, proj_name, torch.nn.Parameter(w_fp8, requires_grad=False))

            # Replace: block_scale buffer -> per-channel scale buffer
            delattr(module, f"{proj_name}_scale")
            module.register_buffer(
                f"{proj_name}_scale", scale_bf16, persistent=True,
            )

            # Remove tensor_scale (not needed for FP8)
            delattr(module, f"{proj_name}_scale_2")

            del packed, block_scale, tensor_scale

        module.__class__ = Gemma4TextExpertsFP8
        torch.cuda.empty_cache()
        elapsed = time.time() - t_layer
        print(
            f"  [nvfp4→fp8] layer {layer_idx:>2}/30 done in {elapsed:>4.1f}s ({name})",
            flush=True,
        )
        if layer_idx % 10 == 0:
            mem_report(f"convert:after layer {layer_idx}")
    print(
        f"  [nvfp4→fp8] conversion complete in {time.time() - t_start:.1f}s",
        flush=True,
    )
    mem_report("convert:done")


def load_gemma4_nvfp4(model_id_or_path, **from_pretrained_kwargs):
    """Load a self-built NVFP4 Gemma 4 checkpoint.

    Two modes, controlled by the ``AEO_NVFP4_NATIVE`` env var:

    * Unset or ``0`` (default): converts NVFP4 -> bf16 -> FP8 in-place at
      load and inference runs through the proven ``_scaled_mm`` FP8 path.
      Takes ~10 s on GB10 with batched 16-experts-at-a-time dequant.

    * ``1``: keeps FP4 weights in GPU memory for the lifetime of the
      model.  Inference routes through
      :func:`aeo_quant.gpu.nvfp4_matmul.nvfp4_linear`, a Triton
      ``tl.dot_scaled`` kernel.  Also requires
      ``TRITON_OVERRIDE_ARCH=sm120`` on sm_121 (GB10), per
      ``kb/nvfp4-blackwell-research.md``.

    An earlier version always did the FP8 conversion and cached the
    result on disk; the batching optimization made conversion so fast
    that the cache was always slower than reconverting.  See
    docs/gemma4-fp8-optimization.md.
    """
    import time

    from aeo_quant.gpu.memory import mem_report

    from_pretrained_kwargs.setdefault("dtype", torch.bfloat16)
    from_pretrained_kwargs.setdefault("device_map", "cuda")

    native = os.environ.get("AEO_NVFP4_NATIVE", "0") == "1"

    mem_report("nvfp4_load:start")
    t_load = time.time()
    with _gemma4_experts_nvfp4_class_swap():
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path, **from_pretrained_kwargs
        )
    print(f"[nvfp4] from_pretrained done in {time.time() - t_load:.1f}s", flush=True)
    mem_report("nvfp4_load:after from_pretrained")

    if native:
        print(
            "[nvfp4] AEO_NVFP4_NATIVE=1 — keeping FP4 weights for native matmul",
            flush=True,
        )
        if os.environ.get("TRITON_OVERRIDE_ARCH") != "sm120":
            print(
                "[nvfp4] WARNING: TRITON_OVERRIDE_ARCH is not 'sm120'. "
                "On sm_121 (GB10) this is required for tl.dot_scaled to "
                "route to native FP4 MMA; without it the kernel falls "
                "through to a slow decomposition. "
                "See kb/nvfp4-blackwell-research.md.",
                flush=True,
            )
        mem_report("nvfp4_load:done (native)")
        # torch.compile the model — same wrap as the FP8 path.  The
        # nvfp4 forward stays on its native class; no preconvert step.
        return torch.compile(model, mode="reduce-overhead", dynamic=False)

    print("[nvfp4] converting NVFP4 -> bf16 -> FP8", flush=True)
    _convert_nvfp4_experts_to_fp8(model)

    t_compile = time.time()
    _preconvert_fp8_scales(model)
    compiled = torch.compile(model, mode="reduce-overhead", dynamic=False)
    print(f"[nvfp4] preconvert+compile wrap done in {time.time() - t_compile:.1f}s", flush=True)
    mem_report("nvfp4_load:done")
    return compiled
