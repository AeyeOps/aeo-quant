"""GPU utilities — requires torch and psutil.

Install: pip install aeo-quant[gpu]
"""

from aeo_quant._lazy import require as _require

_require("torch", "gpu")
_require("psutil", "gpu")
