"""AeyeOps quantization toolkit — model evaluation, benchmarking, and quantization bridges."""

__version__ = "0.1.15"

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
