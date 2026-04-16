#!/usr/bin/env python3
"""Validate TurboQuant KV cache with Gemma 4 26B-A4B on Dell GB10 Max Pro.

Loads google/gemma-4-26B-A4B-it at bf16, runs a short generation with
TurboQuantCache(bits=4), and enforces a hard 90 GB unified-memory cap at
every phase boundary. Fails fast on CUDA absence per project rules.

Usage: python scripts/test_turboquant_gemma4.py
Requires: pip install turboquant, HF_TOKEN in env, Gemma 4 license accepted.
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import psutil
import torch

# turboquant 0.2.0 (latest on PyPI, 2026-03-27) calls np.trapz which was removed
# in numpy 2.x. Upstream hasn't patched yet. Alias it to the modern replacement.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

VRAM_CAP_GB = 90.0
MIN_AVAIL_GB = 70.0
# FP8 compressed-tensors checkpoint with the FUSED MoE layout
# (layers.N.experts.gate_up_proj / experts.down_proj) that matches
# transformers 5.5.3. RedHatAI/*-FP8-Dynamic used the unfused per-expert
# layout from the older modelopt plugin, producing MISSING warnings and
# random-init MoE weights → garbage output. LargitData's checkpoint uses
# the fused layout.
MODEL_ID = "LargitData/gemma-4-26b-a4b-it-fp8"
# LargitData ships a broken/stub tokenizer (bogus chat template, ~10-token
# vocab) that maps the whole prompt to special tokens. Use the base model's
# tokenizer instead — same architecture, fully compatible with the weights.
TOKENIZER_ID = "google/gemma-4-26B-A4B-it"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
TURBOQUANT_BITS = 4
PROMPT = "Write a Python quicksort function and briefly explain how it works."
# LONG_CTX_TOKENS=N pads the user prompt with filler until tokenized input
# reaches approximately N tokens. Used to validate long-context memory behavior
# under the 90 GB cap with Option B + TurboQuant.
LONG_CTX_TOKENS = int(os.environ.get("LONG_CTX_TOKENS", "0"))

# Three mutually-exclusive load paths for Gemma 4 26B-A4B experts:
#   OPTION_A (default): load LargitData FP8, dequant experts to bf16 at load time
#                       via dequantize_gemma4_experts_inplace().
#   OPTION_B=1:         load LargitData FP8, keep experts as FP8 Parameters and
#                       patch Gemma4TextExperts.forward with a runtime FP8 dequant
#                       shim. Saves ~23 GB persistent memory at short context.
#   OPTION_C=1:         load the self-built FP8 checkpoint at
#                       models/gemma-4-26b-a4b-it-fp8/ via the class-swap loader
#                       (scripts/gemma4_fp8/loader.py). No runtime patching needed —
#                       the subclass already materializes FP8 params + bf16 scale
#                       buffers. This is the durable path; A and B are legacy.
USE_OPTION_B = os.environ.get("OPTION_B", "").strip() not in ("", "0", "false", "False")
USE_OPTION_C = os.environ.get("OPTION_C", "").strip() not in ("", "0", "false", "False")
if USE_OPTION_B and USE_OPTION_C:
    print("[FATAL] OPTION_B and OPTION_C are mutually exclusive.", file=sys.stderr)
    sys.exit(1)

# Self-built FP8 checkpoint location (for OPTION_C).
OPTION_C_FP8_PATH = (
    __import__("pathlib").Path(__import__("os").environ["FP8_CHECKPOINT"])
    if __import__("os").environ.get("FP8_CHECKPOINT") else None
)

_GB = 1024**3


def gb(n_bytes: int) -> str:
    return f"{n_bytes / _GB:6.2f} GB"


def mem_report(label: str) -> None:
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    t_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    t_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    print(
        f"[mem] {label:<22} "
        f"sys_used={gb(vm.used)}  "
        f"sys_avail={gb(vm.available)}  "
        f"proc_rss={gb(rss)}  "
        f"torch_alloc={gb(t_alloc)}  "
        f"torch_peak={gb(t_peak)}",
        flush=True,
    )


def enforce_cap(label: str) -> None:
    vm = psutil.virtual_memory()
    if vm.used > VRAM_CAP_GB * _GB:
        print(
            f"\n[FATAL] unified memory cap exceeded at '{label}': "
            f"sys_used={gb(vm.used)} > cap={VRAM_CAP_GB:.0f} GB",
            file=sys.stderr,
        )
        sys.exit(2)


def dequantize_gemma4_experts_inplace(model, model_id: str) -> int:
    """Fix garbage Gemma4TextExperts weights after loading an FP8 checkpoint.

    transformers 5.5.3 stores Gemma 4 experts as fused 3D nn.Parameter tensors
    that compressed-tensors 0.15.0.1 cannot quantize (only nn.Linear/nn.Embedding
    supported). Result: FP8 bytes load as bf16 values without scale applied.

    Re-reads the expert weights and their weight_scale tensors from the cached
    safetensors shards, computes bf16 = fp8.to(bf16) * scale, and copies the
    result into the model's existing Parameter storage in place.
    """
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    print(f"[fix] dequantizing Gemma4TextExperts weights from {model_id}", flush=True)
    cache_dir = Path(snapshot_download(model_id, allow_patterns=["*.safetensors*"]))
    shards = sorted(cache_dir.glob("model*.safetensors"))
    print(f"[fix] {len(shards)} safetensors shards in {cache_dir}", flush=True)

    patched = 0
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            keys = list(f.keys())
            weight_keys = [
                k for k in keys
                if k.endswith(".experts.gate_up_proj") or k.endswith(".experts.down_proj")
            ]
            for wkey in weight_keys:
                skey = f"{wkey}.weight_scale"
                if skey not in keys:
                    print(f"[fix] WARN no scale for {wkey}", flush=True)
                    continue

                weight_fp8 = f.get_tensor(wkey)
                scale = f.get_tensor(skey)

                weight_bf16 = weight_fp8.to(torch.bfloat16) * scale

                parts = wkey.split(".")
                obj = model
                for p in parts:
                    obj = obj[int(p)] if p.isdigit() else getattr(obj, p)

                with torch.no_grad():
                    obj.copy_(weight_bf16.to(obj.device))
                patched += 1
                del weight_fp8, scale, weight_bf16
    return patched


def _patched_gemma4_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Gemma4TextExperts.forward mirrored from transformers 5.5.3 modeling_gemma4.py
    lines 1262-1286, with per-expert FP8→bf16 dequant inlined before each linear.

    Tried a batched-dequant variant but it regressed decode tok/s ~3x without
    fixing the 16K prefill memory explosion, so reverted.

    Requires: self.gate_up_proj and self.down_proj are float8_e4m3fn Parameters,
    and self.gate_up_proj_scale / self.down_proj_scale are bf16 tensors with
    shape (num_experts, out_dim, 1) registered as module attributes.
    """
    import torch.nn as nn

    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        gate_up_w = self.gate_up_proj[expert_idx].to(torch.bfloat16) * self.gate_up_proj_scale[expert_idx]
        gate, up = nn.functional.linear(current_state, gate_up_w).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up

        down_w = self.down_proj[expert_idx].to(torch.bfloat16) * self.down_proj_scale[expert_idx]
        current_hidden_states = nn.functional.linear(current_hidden_states, down_w)
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def reparameterize_gemma4_experts_to_fp8(model, model_id: str) -> int:
    """Replace bf16 Gemma4TextExperts Parameters with FP8 ones and register scales.

    Walks every Gemma4TextExperts submodule; for gate_up_proj and down_proj, loads
    the FP8 weight + bf16 weight_scale from the cached safetensors shards,
    deletes the old bf16 Parameter (critical for real memory release), and
    registers the FP8 Parameter in its place. The scale is set as a plain
    attribute, not a Parameter, to avoid save/load name collisions.
    """
    import torch.nn as nn
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    print(f"[fix] reparameterizing Gemma4TextExperts to FP8 from {model_id}", flush=True)
    cache_dir = Path(snapshot_download(model_id, allow_patterns=["*.safetensors*"]))
    shards = sorted(cache_dir.glob("model*.safetensors"))

    key_to_shard: dict[str, Path] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                key_to_shard[k] = shard

    patched = 0
    for mod_name, mod in model.named_modules():
        if type(mod).__name__ != "Gemma4TextExperts":
            continue
        for proj in ("gate_up_proj", "down_proj"):
            wkey = f"{mod_name}.{proj}"
            skey = f"{wkey}.weight_scale"
            if wkey not in key_to_shard or skey not in key_to_shard:
                print(f"[fix] WARN missing key: {wkey} or {skey}", flush=True)
                continue
            with safe_open(key_to_shard[wkey], framework="pt") as f:
                weight_fp8 = f.get_tensor(wkey)
            with safe_open(key_to_shard[skey], framework="pt") as f:
                scale = f.get_tensor(skey)

            device = next(mod.parameters()).device

            # delattr releases the old bf16 Parameter before the FP8 replacement
            # is registered, so memory actually drops.
            delattr(mod, proj)
            mod.register_parameter(
                proj, nn.Parameter(weight_fp8.to(device), requires_grad=False)
            )
            setattr(mod, f"{proj}_scale", scale.to(device))
            patched += 1

            del weight_fp8, scale
            gc.collect()
            torch.cuda.empty_cache()
    return patched


def preflight() -> None:
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available — GPU-only per project rules.", file=sys.stderr)
        sys.exit(1)

    dev_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    vm = psutil.virtual_memory()
    avail_gb = vm.available / _GB

    print(f"[preflight] device: {dev_name} (sm_{cc_major}{cc_minor})")
    print(f"[preflight] torch: {torch.__version__}")
    print(f"[preflight] unified mem total:     {gb(vm.total)}")
    print(f"[preflight] unified mem available: {gb(vm.available)}")
    print(f"[preflight] safety cap: {VRAM_CAP_GB:.0f} GB  |  min available to start: {MIN_AVAIL_GB:.0f} GB")

    if avail_gb < MIN_AVAIL_GB:
        print(
            f"[FATAL] need {MIN_AVAIL_GB:.0f} GB available, only {avail_gb:.1f} GB. "
            f"Unload Ollama or other processes first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print(
            "[WARN] neither HF_TOKEN nor HUGGING_FACE_HUB_TOKEN set in env. "
            "Gemma 4 is gated — download will 401 if the token isn't loaded. "
            "Source your .env first.",
            file=sys.stderr,
        )


def main() -> None:
    mem_report("start")
    preflight()
    enforce_cap("preflight")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from turboquant import TurboQuantCache
    except ImportError:
        print(
            "[FATAL] turboquant not installed. Run: uv pip install turboquant",
            file=sys.stderr,
        )
        sys.exit(1)

    mem_report("imports loaded")

    if USE_OPTION_C:
        print(f"[fix] OPTION_C=1 → using self-built FP8 via load_gemma4_fp8 ({OPTION_C_FP8_PATH})")
        # sys.path fix so `from scripts.gemma4_fp8.loader import ...` resolves
        # when this script is invoked as `uv run python scripts/test_turboquant_gemma4.py`.
        repo_root = __import__("pathlib").Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        if not OPTION_C_FP8_PATH.exists():
            print(
                f"[FATAL] OPTION_C artifact missing at {OPTION_C_FP8_PATH}. "
                f"Run scripts/build_gemma4_fp8_checkpoint.py first.",
                file=sys.stderr,
            )
            sys.exit(1)
    elif USE_OPTION_B:
        print("[fix] OPTION_B=1 → installing patched Gemma4TextExperts.forward")
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
        Gemma4TextExperts.forward = _patched_gemma4_experts_forward
    else:
        print("[fix] OPTION_A (default) → will dequant experts to bf16 after load")

    print(f"[load] tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    mem_report("tokenizer loaded")
    enforce_cap("after tokenizer")

    attn_impl = os.environ.get("ATTN_IMPL", "sdpa")
    print(f"[load] attn_implementation={attn_impl}")
    t0 = time.time()
    if USE_OPTION_C:
        print(f"[load] model: self-built FP8 at {OPTION_C_FP8_PATH}")
        from scripts.gemma4_fp8.loader import load_gemma4_fp8
        model = load_gemma4_fp8(
            str(OPTION_C_FP8_PATH),
            attn_implementation=attn_impl,
        )
    else:
        print(f"[load] model: {MODEL_ID} (bf16)  — this downloads ~52 GB on first run")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation=attn_impl,
        )
    print(f"[load] model loaded in {time.time() - t0:.1f}s")
    mem_report("model loaded")
    enforce_cap("after model load")

    if USE_OPTION_C:
        print("[fix] OPTION_C → no post-load patch (weights are already FP8 + bf16 scale buffers)")
    elif USE_OPTION_B:
        print("[fix] running Gemma4TextExperts FP8 reparameterize (Option B)...")
        t0 = time.time()
        n_patched = reparameterize_gemma4_experts_to_fp8(model, MODEL_ID)
        print(f"[fix] reparameterized {n_patched} tensors in {time.time()-t0:.1f}s")
        mem_report("after FP8 reparameterize")
        enforce_cap("after FP8 reparameterize")
    else:
        print("[fix] running Gemma4TextExperts dequant shim (Option A)...")
        t0 = time.time()
        n_patched = dequantize_gemma4_experts_inplace(model, MODEL_ID)
        print(f"[fix] patched {n_patched} tensors in {time.time()-t0:.1f}s")
        mem_report("after dequant shim")
        enforce_cap("after dequant shim")

    if LONG_CTX_TOKENS > 0:
        print(f"[gen] padding prompt toward {LONG_CTX_TOKENS} tokens (long-context validation)")
        filler_sentence = (
            "The quick brown fox jumps over the lazy dog while reviewing the "
            "quarterly earnings report and debating the merits of various "
            "sorting algorithms used in distributed database systems. "
        )
        base_ids = tokenizer(filler_sentence, add_special_tokens=False)["input_ids"]
        tokens_per_sentence = max(len(base_ids), 1)
        reps = max(1, (LONG_CTX_TOKENS - 100) // tokens_per_sentence + 1)
        filler = filler_sentence * reps
        filler_ids = tokenizer(filler, add_special_tokens=False)["input_ids"]
        keep = max(0, LONG_CTX_TOKENS - 100)
        filler_truncated = tokenizer.decode(filler_ids[:keep], skip_special_tokens=True)
        padded_user = (
            f"Context document:\n\n{filler_truncated}\n\n"
            f"End of document.\n\nTask: {PROMPT}"
        )
        messages = [{"role": "user", "content": padded_user}]
    else:
        messages = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if LONG_CTX_TOKENS > 0:
        print(f"[gen] padded prompt length (chars): {len(prompt)}")
    else:
        print(f"[gen] prompt after chat template: {prompt!r}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_input_tokens = inputs["input_ids"].shape[-1]
    print(f"[gen] input tokens: {n_input_tokens}")
    if n_input_tokens <= 64:
        print(f"[gen] input token ids: {inputs['input_ids'][0].tolist()}")
    else:
        first_ids = inputs['input_ids'][0][:20].tolist()
        last_ids = inputs['input_ids'][0][-20:].tolist()
        print(f"[gen] first 20 ids: {first_ids}")
        print(f"[gen] last 20 ids:  {last_ids}")
    print(f"[gen] model.config.eos_token_id: {getattr(model.config, 'eos_token_id', None)}")
    print(f"[gen] model.config.pad_token_id: {getattr(model.config, 'pad_token_id', None)}")
    print(f"[gen] tokenizer.eos_token_id:    {tokenizer.eos_token_id}")
    print(f"[gen] tokenizer.pad_token_id:    {tokenizer.pad_token_id}")
    mem_report("inputs prepared")
    enforce_cap("after inputs")

    if os.environ.get("NO_TURBOQUANT", "").strip() not in ("", "0", "false", "False"):
        from transformers import DynamicCache
        print("[gen] NO_TURBOQUANT=1 → using default DynamicCache (bisect diagnostic)")
        cache = DynamicCache()
    else:
        print(f"[gen] TurboQuantCache(bits={TURBOQUANT_BITS})")
        cache = TurboQuantCache(bits=TURBOQUANT_BITS)

    print(f"[gen] generating up to {MAX_NEW_TOKENS} tokens (greedy)")
    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            past_key_values=cache,
            use_cache=True,
            do_sample=False,
        )
    gen_time = time.time() - t0
    enforce_cap("after generate")
    mem_report("after generate")

    n_new_tokens = int(outputs.shape[-1] - n_input_tokens)
    tok_per_s = n_new_tokens / gen_time if gen_time > 0 else 0.0
    print(f"[gen] generated {n_new_tokens} tokens in {gen_time:.1f}s ({tok_per_s:.1f} tok/s)")

    new_ids = outputs[0][n_input_tokens:].tolist()
    print(f"[gen] first 20 new token ids: {new_ids[:20]}")
    unique_ids = set(new_ids)
    print(f"[gen] unique token ids in output: {len(unique_ids)}")

    decoded_clean = tokenizer.decode(outputs[0][n_input_tokens:], skip_special_tokens=True)
    decoded_raw = tokenizer.decode(outputs[0][n_input_tokens:], skip_special_tokens=False)
    print("\n=== MODEL OUTPUT (skip_special_tokens=True) ===")
    print(decoded_clean if decoded_clean else "<EMPTY>")
    print("=== END ===\n")
    print("=== MODEL OUTPUT (skip_special_tokens=False) ===")
    print(decoded_raw if decoded_raw else "<EMPTY>")
    print("=== END ===\n")

    peak_torch = torch.cuda.max_memory_allocated()
    del model, tokenizer, cache, outputs, inputs
    gc.collect()
    torch.cuda.empty_cache()
    mem_report("cleaned up")

    print(f"[summary] peak torch allocated: {gb(peak_torch)}")
    print(f"[summary] status: OK  |  {n_new_tokens} tokens at {tok_per_s:.1f} tok/s with TurboQuant {TURBOQUANT_BITS}-bit KV cache")


if __name__ == "__main__":
    main()
