#!/usr/bin/env bash
# Dump SASS from the most recently compiled Triton kernel.
#
# Triton caches compiled artifacts under $TRITON_CACHE_DIR (default
# ~/.triton/cache). Each compile writes a .cubin that we can disassemble
# with cuobjdump.
#
# Usage:
#   tools/dump_triton_sass.sh                          # most recent
#   tools/dump_triton_sass.sh --name _nvfp4_matmul     # filter by kernel name
#   tools/dump_triton_sass.sh --limit 200              # limit output lines
#
# Output: SASS to stdout. Compile metadata (arch, smem, regs) to stderr.

set -euo pipefail

CACHE="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
NAME_FILTER=""
LIMIT=300

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)  NAME_FILTER="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *)       echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -d "$CACHE" ]]; then
    echo "[FATAL] Triton cache not found at $CACHE" >&2
    exit 2
fi

# Find most recent .cubin, optionally name-filtered
if [[ -n "$NAME_FILTER" ]]; then
    CUBIN=$(find "$CACHE" -name "*.cubin" -path "*${NAME_FILTER}*" -printf "%T@ %p\n" \
            | sort -nr | head -1 | awk '{print $2}')
else
    CUBIN=$(find "$CACHE" -name "*.cubin" -printf "%T@ %p\n" \
            | sort -nr | head -1 | awk '{print $2}')
fi

if [[ -z "$CUBIN" ]]; then
    echo "[FATAL] no .cubin found under $CACHE" >&2
    exit 2
fi

echo "[sass] cubin: $CUBIN" >&2
echo "[sass] mtime: $(stat -c '%y' "$CUBIN")" >&2

# Sibling JSON has compile metadata
DIR=$(dirname "$CUBIN")
if ls "$DIR"/*.json >/dev/null 2>&1; then
    JSON=$(ls "$DIR"/*.json | head -1)
    echo "[sass] meta: $JSON" >&2
    # Pretty-print the interesting fields if jq is installed
    if command -v jq >/dev/null 2>&1; then
        jq -r '. | "  name=\(.name) arch=\(.target // .arch // "?") n_regs=\(.n_regs // "?") shared=\(.shared // "?")"' "$JSON" >&2 || true
    fi
fi

# Dump SASS — cuobjdump ships with the CUDA toolkit
if ! command -v cuobjdump >/dev/null 2>&1; then
    echo "[FATAL] cuobjdump not in PATH" >&2
    echo "        try: /usr/local/cuda/bin/cuobjdump" >&2
    exit 2
fi

cuobjdump --dump-sass "$CUBIN" 2>&1 | head -n "$LIMIT"
