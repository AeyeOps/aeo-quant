"""Configuration helpers — .env loading, environment setup, results paths.

Stdlib only — no third-party dependencies.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def _resolve_dotenv_path(path: str | Path) -> Path | None:
    """Find a .env file by walking upward from the current directory.

    Relative paths are resolved against the current working directory first,
    then each parent directory. Absolute paths are used as-is.
    """
    p = Path(path)
    if p.is_absolute():
        return p if p.exists() else None

    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        candidate = base / p
        if candidate.exists():
            return candidate
    return None


def load_dotenv(path: str | Path = ".env", *, override: bool = True) -> int:
    """Load environment variables from a .env file.

    Args:
        path: Path to the .env file. Defaults to ".env" in the current directory.
        override: If True (default), .env values override existing env vars.
            If False, existing env vars take precedence.

    Returns:
        Number of variables loaded.

    The parser handles:
    - Comments (lines starting with #)
    - Quoted values: FOO="bar" and FOO='bar' strip the quotes
    - Inline comments: FOO=bar # comment
    - Empty values: FOO= sets FOO to ""
    - Whitespace around key/value is stripped
    """
    p = _resolve_dotenv_path(path)
    if p is None:
        return 0

    count = 0
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        # Strip inline comments (only if not inside quotes)
        if value and value[0] not in ('"', "'") and " #" in value:
            value = value.split(" #", 1)[0].strip()

        # Strip surrounding quotes
        if len(value) >= 2:
            if (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"):
                value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value
            count += 1

    return count


def setup_cuda_allocator(config: str = "expandable_segments:True") -> None:
    """Set PYTORCH_CUDA_ALLOC_CONF if not already set."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", config)


def quant_env() -> tuple[str, Path, int]:
    """Read quantization config from environment.

    Returns:
        ``(quant_format, checkpoint_path, kv_bits)`` where:
          - ``quant_format``: ``"fp8"`` (default) or ``"nvfp4"``
          - ``checkpoint_path``: resolved from ``CHECKPOINT``, ``FP8_CHECKPOINT``,
            or ``NVFP4_CHECKPOINT`` depending on format
          - ``kv_bits``: from ``KV_BITS`` env var (default 4 for fp8, 3 for nvfp4)

    Calls ``sys.exit(1)`` if no checkpoint path is found.
    """
    import sys

    fmt = os.environ.get("QUANT_FORMAT", "fp8")
    ckpt = os.environ.get("CHECKPOINT") or os.environ.get(
        "NVFP4_CHECKPOINT" if fmt == "nvfp4" else "FP8_CHECKPOINT"
    )
    if not ckpt:
        print(
            f"[FATAL] No checkpoint set for QUANT_FORMAT={fmt}. Set CHECKPOINT "
            f"or {'NVFP4_CHECKPOINT' if fmt == 'nvfp4' else 'FP8_CHECKPOINT'} "
            f"in .env or as an env var.",
            file=sys.stderr,
        )
        sys.exit(1)
    kv_bits = int(os.environ.get("KV_BITS", "3" if fmt == "nvfp4" else "4"))
    return fmt, Path(ckpt), kv_bits


def results_dir(category: str, *, timestamped: bool = True) -> Path:
    """Create and return a results directory.

    Default layout: ``results/{category}/{YYYYMMDD-HHMMSS}/``.
    Honors the ``RESULTS_DIR`` env var as an override.
    """
    override = os.environ.get("RESULTS_DIR")
    if override:
        d = Path(override)
    else:
        d = Path("results") / category
        if timestamped:
            d = d / datetime.now().strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d
