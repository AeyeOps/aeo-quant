#!/usr/bin/env python3
"""Build an FP8 checkpoint from a bf16 Gemma 4 model via shard streaming.

Reads the source model's safetensors shards one at a time, quantizes fused
3D MoE expert weights to FP8 in-flight, and writes sharded output. The full
bf16 model is never loaded into memory — peak usage is ~18 GB.

Usage:
    uv run python examples/build_checkpoint.py

Set SOURCE_MODEL in .env to override the base model (default: google/gemma-4-26B-A4B-it).
Set FP8_CHECKPOINT to control where the output checkpoint is written.
"""
from __future__ import annotations

import gc
import json
import os
import re
import shutil
import sys
from pathlib import Path

import psutil
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

import aeo_quant  # noqa: F401 — triggers np.trapz compat shim before numpy is used
from aeo_quant.core.config import load_dotenv, setup_cuda_allocator
from aeo_quant.gpu.memory import _GB, enforce_cap, gb, mem_report, preflight_memory
from aeo_quant.gpu.quant import quantize_3d_to_fp8

load_dotenv()
setup_cuda_allocator()

# Memory budget (unified LPDDR5X on GB10): shard streaming peaks ~53 GB RSS.
# Need enough headroom to load one source shard (~5 GB) and process it.
MIN_FREE_GB = 60.0

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VRAM_CAP_GB = float(os.environ.get("VRAM_CAP_GB", "90.0"))
SOURCE_MODEL_ID = os.environ.get("SOURCE_MODEL", "google/gemma-4-26B-A4B-it")
SHARD_SIZE_BYTES = 5 * 1024**3  # 5 GB target shard size

_OUTPUT_DIR_ENV = os.environ.get("FP8_CHECKPOINT")
if not _OUTPUT_DIR_ENV:
    print(
        "[FATAL] FP8_CHECKPOINT not set. Add it to .env — this is where the "
        "output checkpoint will be written.",
        file=sys.stderr,
    )
    sys.exit(1)
OUTPUT_DIR: Path = Path(_OUTPUT_DIR_ENV)

# Fused 3D expert weight pattern
_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.experts\.(gate_up_proj|down_proj)$"
)


def main() -> None:
    preflight_memory(MIN_FREE_GB, label="build_checkpoint")
    mem_report("start")

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available.", file=sys.stderr)
        sys.exit(1)

    vm = psutil.virtual_memory()
    print(f"[preflight] unified mem available: {gb(vm.available)}")
    print(f"[preflight] source model: {SOURCE_MODEL_ID}")
    print(f"[preflight] output dir: {OUTPUT_DIR}")
    enforce_cap("preflight", VRAM_CAP_GB)

    # Download source model (or use cache)
    print(f"[download] {SOURCE_MODEL_ID}")
    src_dir = Path(snapshot_download(
        SOURCE_MODEL_ID, allow_patterns=["*.safetensors", "*.json", "*.jinja"],
    ))
    print(f"[download] cached at {src_dir}")
    mem_report("after download")

    # Find source shards
    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"[FATAL] no model.safetensors.index.json in {src_dir}", file=sys.stderr)
        sys.exit(1)

    with open(index_path) as f:
        src_index = json.load(f)
    shard_files = sorted(set(src_index["weight_map"].values()))
    print(f"[build] {len(shard_files)} source shards, {len(src_index['weight_map'])} keys")

    # Prepare output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_weight_map: dict[str, str] = {}
    out_shard_idx = 0
    out_shard_tensors: dict[str, torch.Tensor] = {}
    out_shard_bytes = 0
    total_quantized = 0
    total_passthrough = 0

    def flush_shard():
        nonlocal out_shard_idx, out_shard_tensors, out_shard_bytes
        if not out_shard_tensors:
            return
        out_shard_idx += 1
        n_shards_expected = len(shard_files) + 1  # may be more due to scale tensors
        shard_name = f"model-{out_shard_idx:05d}-of-{n_shards_expected:05d}.safetensors"
        out_path = OUTPUT_DIR / shard_name
        save_file(out_shard_tensors, str(out_path))
        for key in out_shard_tensors:
            out_weight_map[key] = shard_name
        n_tensors = len(out_shard_tensors)
        print(f"  wrote {shard_name} ({n_tensors} tensors, {out_shard_bytes / _GB:.2f} GB)")
        out_shard_tensors = {}
        out_shard_bytes = 0

    def add_tensor(key: str, tensor: torch.Tensor):
        nonlocal out_shard_bytes
        nbytes = tensor.nelement() * tensor.element_size()
        if out_shard_bytes + nbytes > SHARD_SIZE_BYTES and out_shard_tensors:
            flush_shard()
        out_shard_tensors[key] = tensor
        out_shard_bytes += nbytes

    # Process source shards one at a time
    for shard_file in shard_files:
        shard_path = src_dir / shard_file
        print(f"\n[shard] {shard_file}")
        mem_report(f"before {shard_file}")
        enforce_cap(f"before {shard_file}", VRAM_CAP_GB)

        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 — safe_open requires .keys()
                tensor = f.get_tensor(key)
                m = _EXPERT_RE.match(key)
                if m:
                    # Quantize fused 3D expert weights
                    weight_fp8, scale = quantize_3d_to_fp8(tensor)
                    add_tensor(key, weight_fp8)
                    proj_name = m.group(2)  # gate_up_proj or down_proj
                    scale_key = key.rsplit(".", 1)[0] + f".{proj_name}_scale"
                    add_tensor(scale_key, scale.squeeze(-1))
                    total_quantized += 1
                    print(f"    FP8: {key} {tuple(tensor.shape)}")
                else:
                    # Pass through unchanged
                    add_tensor(key, tensor)
                    total_passthrough += 1

                del tensor
                gc.collect()

        mem_report(f"after {shard_file}")

    # Flush remaining tensors
    flush_shard()

    # Fix shard naming (now we know the actual count)
    actual_shards = out_shard_idx
    for i in range(1, actual_shards + 1):
        old_name = f"model-{i:05d}-of-{len(shard_files) + 1:05d}.safetensors"
        new_name = f"model-{i:05d}-of-{actual_shards:05d}.safetensors"
        if old_name != new_name:
            (OUTPUT_DIR / old_name).rename(OUTPUT_DIR / new_name)
            for k, v in out_weight_map.items():
                if v == old_name:
                    out_weight_map[k] = new_name

    # Write index
    out_index = {
        "metadata": src_index.get("metadata", {}),
        "weight_map": out_weight_map,
    }
    with open(OUTPUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(out_index, f, indent=2)

    # Copy config, tokenizer, and template files
    for pattern in ["config.json", "generation_config.json", "tokenizer*.json",
                     "processor_config.json", "chat_template.jinja"]:
        for src_file in src_dir.glob(pattern):
            shutil.copy2(src_file, OUTPUT_DIR / src_file.name)

    mem_report("done")
    print(f"\n[done] {total_quantized} expert tensors quantized to FP8")
    print(f"[done] {total_passthrough} tensors passed through unchanged")
    print(f"[done] {actual_shards} output shards written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
