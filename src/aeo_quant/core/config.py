"""Configuration helpers — .env loading and environment setup.

Stdlib only — no third-party dependencies.
"""

from __future__ import annotations

import os
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
