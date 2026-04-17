"""SWA-aware TurboQuant KV cache for Gemma 4.

Gemma 4 mixes 25 sliding-attention layers with 5 full-attention layers. Stock
``TurboQuantCache`` stores unbounded history on every layer and dequantizes the
full stored tensor on every decode step, even on sliding layers where attention
masks out everything older than ``sliding_window=1024``. This module swaps in a
``TurboQuantSlidingLayer`` for sliding-attention layers that caps its compressed
storage at the window size, so per-step dequant cost on those layers becomes
fixed instead of growing with conversation length.

The class is kept Gemma-4-specific by name for v1. The underlying logic (read
``layer_types`` + ``sliding_window`` from config, dispatch to sliding vs full
layers) generalizes cleanly — when a second model family needs this, rename to
``HybridTurboQuantCache``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from turboquant import TurboQuantCache
from turboquant.cache import TurboQuantLayer


class TurboQuantSlidingLayer(TurboQuantLayer):
    """TurboQuant-compressed KV layer that trims storage to ``sliding_window``.

    Mirrors ``transformers.cache_utils.DynamicSlidingWindowLayer`` but on the
    compressed + residual storage from ``TurboQuantLayer``. After the parent's
    ``update`` runs, the compressed index/norm arrays are trimmed from the head
    (dropping the oldest quantized tokens) so next-step dequant touches at most
    ``sliding_window - 1 - residual_len`` compressed tokens plus ``residual_len``
    FP16 residual tokens — a fixed cost per layer, independent of how many
    tokens the conversation has accumulated.
    """

    is_sliding = True

    def __init__(self, bits: int = 3, residual_len: int = 128, sliding_window: int = 1024):
        super().__init__(bits=bits, residual_len=residual_len)
        self.sliding_window = sliding_window
        # Match DynamicSlidingWindowLayer: store sliding_window - 1 total tokens
        # (the current token fills the missing slot during attention compute).
        self._compressed_cap = sliding_window - 1 - residual_len
        if self._compressed_cap < 0:
            raise ValueError(
                f"sliding_window ({sliding_window}) must exceed residual_len "
                f"({residual_len}) + 1 so compressed storage has at least one slot"
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = super().update(key_states, value_states, cache_kwargs)

        # Trim the head (oldest) of the compressed arrays so next-step dequant
        # starts from a window-bounded state. The returned tensors are NOT
        # touched — they already carry the pre-trim view (length
        # ``window - 1 + query_length`` when steady-state full) that matches
        # ``get_mask_sizes(query_length)`` below.
        #
        # ``_key_indices`` is a 1-D empty placeholder from lazy_initialization
        # until the first overflow-quantize pass turns it 4-D, so gate on
        # ``numel() > 0`` (the same check the parent uses at cache.py:108)
        # before touching ``shape[-2]``.
        if (
            self._key_indices is not None
            and self._key_indices.numel() > 0
            and self._key_indices.shape[-2] > self._compressed_cap
        ):
            cap = self._compressed_cap
            self._key_indices = self._key_indices[..., -cap:, :].contiguous()
            self._key_norms = self._key_norms[..., -cap:, :].contiguous()
            self._value_indices = self._value_indices[..., -cap:, :].contiguous()
            self._value_norms = self._value_norms[..., -cap:, :].contiguous()

        return keys, values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # Verbatim from transformers.cache_utils.DynamicSlidingWindowLayer.get_mask_sizes
        # so mask builders see the same contract they'd see from a stock sliding cache.
        cumulative_length = self._total_len
        is_full = cumulative_length >= self.sliding_window
        kv_offset = max(cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            kv_length = self.sliding_window - 1 + query_length
        else:
            kv_length = cumulative_length + query_length
        return kv_length, kv_offset

    def get_max_cache_shape(self) -> int:
        return self.sliding_window


class Gemma4HybridTurboQuantCache(TurboQuantCache):
    """TurboQuant cache with per-layer SWA routing for Gemma 4.

    Reads ``layer_types`` and ``sliding_window`` from the model config and
    pre-populates ``self.layers`` with ``TurboQuantSlidingLayer`` for sliding
    layers and ``TurboQuantLayer`` for full-attention layers. The stock
    ``TurboQuantCache.update`` dispatches by ``layer_idx`` without knowing the
    layer type, so the pre-populated list is what makes the hybrid behavior
    work — ``Cache.get_seq_length`` / ``Cache.get_mask_sizes`` inherit the
    per-layer delegation that the base class already provides.
    """

    def __init__(
        self,
        bits: int = 3,
        residual_len: int = 128,
        *,
        config=None,
    ):
        if config is None:
            raise ValueError(
                "Gemma4HybridTurboQuantCache requires a `config=` to discover "
                "layer_types and sliding_window. Pass `config=model.config`."
            )

        super().__init__(bits=bits)

        decoder_config = config.get_text_config(decoder=True)
        layer_types = getattr(decoder_config, "layer_types", None)
        sliding_window = getattr(decoder_config, "sliding_window", None)
        n_shared = getattr(decoder_config, "num_kv_shared_layers", 0)

        if layer_types is None:
            raise ValueError(
                "Gemma4HybridTurboQuantCache requires `config.layer_types`; "
                "model config exposes none (is this actually Gemma 4?)."
            )
        if sliding_window is None:
            raise ValueError(
                "Gemma4HybridTurboQuantCache requires `config.sliding_window`; "
                "model config exposes none (is this actually Gemma 4?)."
            )
        if n_shared > 0:
            raise NotImplementedError(
                f"num_kv_shared_layers={n_shared}: cross-layer KV sharing is "
                "not supported by this cache. Gemma 4 sets this to 0; if a "
                "future model variant needs it, extend the cache."
            )

        layers = []
        for idx, lt in enumerate(layer_types):
            if lt == "sliding_attention":
                layers.append(TurboQuantSlidingLayer(
                    bits=bits, residual_len=residual_len, sliding_window=sliding_window
                ))
            elif lt == "full_attention":
                layers.append(TurboQuantLayer(bits=bits, residual_len=residual_len))
            else:
                raise NotImplementedError(
                    f"Layer {idx} has unsupported layer_type={lt!r}. "
                    "This cache handles 'sliding_attention' and 'full_attention' only."
                )

        # Replace the empty list that TurboQuantCache.__init__ left behind.
        self.layers = layers
        self._residual_len = residual_len
        self._sliding_window = sliding_window

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Defensive: inherited TurboQuantCache.update lazy-appends a vanilla
        # TurboQuantLayer when layer_idx is out of range, which would silently
        # break SWA semantics for the rest of the run. Raise instead.
        if layer_idx >= len(self.layers):
            raise IndexError(
                f"layer_idx={layer_idx} exceeds pre-populated layer count "
                f"({len(self.layers)}). This cache is shape-locked to the "
                "model config passed at construction."
            )
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
