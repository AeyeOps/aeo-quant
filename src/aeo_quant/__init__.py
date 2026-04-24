"""AeyeOps quantization toolkit — model evaluation, benchmarking, and quantization bridges."""

# Single source of truth for the version is `pyproject.toml`. Reading through
# installed package metadata keeps __version__ from drifting out of sync with
# the declared version. The PackageNotFoundError fallback handles the corner
# case of importing from a source tree that was never installed (e.g. raw
# `python -c "import aeo_quant"` with just the repo on sys.path).
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("aeo-quant")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Compat shim: turboquant 0.2.0 uses np.trapz, removed in numpy 2.x
try:
    import numpy as np
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
except ImportError:
    pass  # numpy not installed

# Core re-exports (stdlib only — always available)
from aeo_quant.core.coherence import check_output_coherent as check_output_coherent
from aeo_quant.core.context import trim_history_to_budget as trim_history_to_budget
from aeo_quant.core.writers import CSVWriter as CSVWriter
from aeo_quant.core.writers import TranscriptWriter as TranscriptWriter
