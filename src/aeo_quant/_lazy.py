"""Guarded optional imports with clear install instructions."""

from __future__ import annotations

import importlib


def require(module_name: str, extra: str):
    """Import and return a module, raising a clear error if missing."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"{module_name} is required for this feature. "
            f"Install it with: pip install aeo-quant[{extra}]"
        ) from None
