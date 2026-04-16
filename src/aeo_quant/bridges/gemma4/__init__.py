"""Gemma 4 FP8/NVFP4 bridge — class-swap loader for fused 3D MoE expert quantization."""

from aeo_quant._lazy import require as _require

_require("torch", "bridges")
_require("transformers", "bridges")
