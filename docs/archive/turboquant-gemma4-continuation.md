# TurboQuant + Gemma 4 26B-A4B on Dell GB10 Max Pro — continuation prompt

**Read this file first, then execute the plan. Everything you need is here.**

## Why this exists

A long prior Claude Code session established the environment, fixed several load failures, identified the root cause of garbage output, and designed two fixes (Option A and Option B). The prior session's context was getting long, so work was paused here to continue in a fresh session.

## Goal

Validate **TurboQuant 4-bit KV cache compression on Gemma 4 26B-A4B** running via transformers on Dell GB10 Max Pro (NVIDIA GB10 Grace Blackwell ARM64, 128 GB unified memory). The user's actual target is **long-context coding workloads at 128K–256K tokens**, which is why Option B's memory savings matter — short-context tests aren't the goal.

Two-stage plan:
1. **Option A (do first, ~45 min)** — Dequantize-at-load shim. Produces correct bf16 weights, same memory footprint as bf16 base. Purpose: prove TurboQuant mechanics work end-to-end with coherent output.
2. **Option B (do after A is green, ~90 min)** — Replace expert Parameters with FP8 + runtime dequant in forward. Saves ~23 GB persistent memory. Unlocks 256K context under the 90 GB cap.

## Environment (already configured — don't reinstall)

- **Project root:** `<dev-root>/trt` (work here; CLAUDE.md has project rules)
- **Python venv:** `.venv`, managed by `uv`. **Always use `uv run python ...`**, never activate manually
- **Python version:** 3.14.2
- **Key installed packages:**
  - `torch==2.11.0+cu130` (CUDA 13.0, aarch64, Blackwell SM121)
  - `transformers==5.5.3`
  - `compressed-tensors==0.15.0.1`
  - `accelerate==1.13.0`
  - `turboquant==0.2.0`
  - `psutil==7.2.2`
  - `safetensors` (from transformers)
  - `huggingface_hub`
- **pyproject.toml** has all of the above pinned; a `[[tool.uv.index]] pytorch-cu130` block pins torch to the CUDA 13.0 aarch64 index
- **`.env`** has `HF_TOKEN` — source it before running anything: `bash -c 'set -a; source .env; set +a; uv run python ...'`
- **HF cache** is at `$HF_HOME` via `HF_HOME` — NOT `~/.cache`
- **Cached models already downloaded** (don't re-download):
  - `models--google--gemma-4-26B-A4B-it` — 49 GB bf16, **license already accepted**
  - `models--LargitData--gemma-4-26b-a4b-it-fp8` — 27 GB FP8 (compressed-tensors FP8_DYNAMIC format)

## Hardware

- **Dell GB10 Max Pro** (Dell's whitelabel of NVIDIA GB10 reference / DGX Spark-class)
- 128 GB unified LPDDR5x, 273 GB/s bandwidth, Blackwell SM121 GPU
- **`nvidia-smi` memory queries return N/A** on GB10 — unified memory has no separate VRAM pool
- Use `free -h`, `psutil.virtual_memory()`, `torch.cuda.mem_get_info()` instead
- Use `nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv` to see WHO is holding GPU memory (Ollama and stt-service are notorious squatters)
- **Hard memory cap: 90 GB sys_used.** User's rule: "GB10 has no OOM guardrails and will blow up." Enforce this at every phase boundary, abort on violation.

### Before running anything, check for memory squatters

```bash
ollama ps  # if anything loaded, ask user to stop: ollama stop <model>
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
free -h
```

Baseline should be ~14-16 GB used. If it's higher, identify and clear.

## Current script state

**File:** `<repo-root>/scripts/test_turboquant_gemma4.py`

Already has:
- `numpy.trapz = numpy.trapezoid` monkey-patch (turboquant 0.2.0 bug with numpy 2.4.1)
- `VRAM_CAP_GB = 90.0`, `MIN_AVAIL_GB = 70.0`
- `MODEL_ID = "LargitData/gemma-4-26b-a4b-it-fp8"`
- `TOKENIZER_ID = "google/gemma-4-26B-A4B-it"` (LargitData's tokenizer is a broken stub)
- `enforce_cap()` check at phase boundaries
- `mem_report()` at every phase
- Diagnostic output: prompt after chat template, input token IDs, eos/pad ids
- **Currently loads successfully, runs generation, but output is garbage (multilingual token soup).** Peak sys_used ~65 GB.

Read the file first before editing. Don't rewrite — add the shim as a new function.

## Root cause (verified from source, don't re-research)

**`compressed-tensors 0.15.0.1` cannot quantize Gemma 4's fused MoE experts.**

Verified from source:

1. **`transformers/models/gemma4/modeling_gemma4.py` lines 1250-1286:** `Gemma4TextExperts` stores experts as **fused 3D `nn.Parameter`**:
   ```python
   self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2*intermediate_dim, hidden_dim))  # (128, 1408, 2816)
   self.down_proj    = nn.Parameter(torch.empty(num_experts, hidden_dim, intermediate_dim))    # (128, 2816, 704)
   # forward() slices with [expert_idx] and calls nn.functional.linear
   ```

2. **`compressed_tensors/quantization/lifecycle/initialize.py` line 116-117:**
   ```python
   elif isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
       # register weight_scale as nn.Parameter, hook quantized forward
       ...
   else:
       raise ValueError(f"Quantization of module type {type(module)} is not supported")
   ```
   Only nn.Linear / nn.Embedding / attention modules are supported. Gemma4TextExperts is none of these.

3. **LargitData's `config.json` quantization_config:** `targets = ["Linear"]`, so `match_named_modules` walks the model, finds Gemma4TextExperts is not Linear, skips it entirely. No scale parameters are registered on the experts module.

4. **Load consequences:**
   - `experts.gate_up_proj` FP8 bytes load into the existing bf16 nn.Parameter via `.to(bfloat16)` conversion (PyTorch handles FP8→bf16 numerically). The dtype is bf16, values are FP8 magnitudes.
   - `experts.gate_up_proj.weight_scale` has no corresponding nn.Parameter in the state_dict → loader emits `UNEXPECTED` warning → scale is **silently dropped on the floor**
   - Forward uses the un-scaled weights → matmul produces wrong magnitudes → multilingual token soup

5. **No upstream PR fixes this as of April 12, 2026.** Closest-adjacent work (none solve our case):
   - compressed-tensors #608/#609 (March 2026): fixed `pack_to_int32` for 3D MoE on the **save** side
   - compressed-tensors #641 (April 9, 2026, merged): added offline FP8_BLOCK dequantizer via `convert_checkpoint` entrypoint — FP8_BLOCK not FP8_DYNAMIC, offline not runtime, not yet released to PyPI
   - transformers #43478 (closed, not merged): BatchLinear proposal
   - compressed-tensors main branch initialize.py (commit `077e752b`): still has the same ValueError

## Checkpoint structure (verified empirically from safetensors)

Ran a probe on LargitData's shards:

```
model.language_model.layers.N.experts.gate_up_proj        shape=(128, 1408, 2816) dtype=torch.float8_e4m3fn
model.language_model.layers.N.experts.gate_up_proj.weight_scale   shape=(128, 1408, 1) dtype=torch.bfloat16
model.language_model.layers.N.experts.down_proj           shape=(128, 2816,  704) dtype=torch.float8_e4m3fn
model.language_model.layers.N.experts.down_proj.weight_scale      shape=(128, 2816, 1) dtype=torch.bfloat16
```

- 2 safetensors shards total
- Weights are **`torch.float8_e4m3fn`** (E4M3 FN variant)
- Scales are **`torch.bfloat16`**, per-expert per-output-channel (last dim is 1 for broadcast)
- Dequant math: `weight_bf16 = weight_fp8.to(torch.bfloat16) * scale` — scale broadcasts on the last dim

Note: the weight tensor name has NO `.weight` suffix (because Gemma4TextExperts stores the Parameter directly as `gate_up_proj`, not as a submodule). But the scale tensor name DOES have `.weight_scale` suffix. So you get the asymmetric pair:
- `experts.gate_up_proj` (the weight)
- `experts.gate_up_proj.weight_scale` (the scale)

## Memory budget math

Gemma 4 26B-A4B attention pattern: **5 sliding + 1 full**, sliding_window=1024, num_kv_heads=8.
- Full-attn layers (5 of 30): head_dim=512 → 8 * 512 * 2(K+V) * 2(bf16) = **16 KB/token**
- SWA layers (25 of 30): capped at 1024 tokens, ~200 MB total, doesn't scale
- **Full-attn KV scales at 80 KB/token**: 5 layers * 16 KB

| Context | Uncompressed KV | TurboQuant bits=4 (4×) | bits=3 (~6×) |
|---|---|---|---|
| 32K  | 2.6 GB  | 0.65 GB | 0.4 GB |
| 128K | 10.5 GB | 2.6 GB  | 1.7 GB |
| 256K | **21 GB** | **5.25 GB** | **3.5 GB** |

| Config | Weights | KV@256K (TQ4) | Overhead | **Peak** | Headroom |
|---|---|---|---|---|---|
| Option A (bf16 experts) | ~52 GB | 5.25 | ~15 | **72 GB** | 18 GB (tight) |
| Option B (FP8 experts runtime) | **~31 GB** | 5.25 | ~15 | **51 GB** | **39 GB** |

Option B saves ~21 GB persistent — the whole point of this experiment is memory relief for long context, so Option B is ultimately required.

---

# TASK PART 1 — OPTION A (dequant shim)

**Goal:** Overwrite Gemma4TextExperts' garbage bf16 weights in-place with correctly dequantized bf16 values, then run generation and confirm coherent output. No upstream changes. Pure user-space. ~50 lines added to the test script.

## Implementation

Add this function to `scripts/test_turboquant_gemma4.py`:

```python
def dequantize_gemma4_experts_inplace(model, model_id: str) -> int:
    """Fix garbage Gemma4TextExperts weights after loading an FP8 checkpoint.

    transformers 5.5.3 stores Gemma 4 experts as fused 3D nn.Parameter tensors
    that compressed-tensors 0.15.0.1 cannot quantize (only nn.Linear/nn.Embedding
    supported). Result: FP8 bytes load as bf16 values without scale applied.

    This function re-reads the expert weights AND their weight_scale tensors from
    the cached safetensors shards, computes `bf16 = fp8.to(bf16) * scale`, and
    copies the result into the model's existing Parameter storage in place.

    Returns the number of tensors patched. Expected: 60 (30 layers * 2 projections).
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

                weight_fp8 = f.get_tensor(wkey)  # (128, out, in) float8_e4m3fn
                scale = f.get_tensor(skey)       # (128, out, 1)  bfloat16

                # Elementwise dequant via broadcast
                weight_bf16 = weight_fp8.to(torch.bfloat16) * scale

                # Navigate wkey like "model.language_model.layers.15.experts.gate_up_proj"
                parts = wkey.split(".")
                obj = model
                for p in parts:
                    obj = obj[int(p)] if p.isdigit() else getattr(obj, p)

                with torch.no_grad():
                    obj.copy_(weight_bf16.to(obj.device))
                patched += 1
                del weight_fp8, scale, weight_bf16
    return patched
```

**Call site:** in `main()`, insert after `mem_report("model loaded")` and `enforce_cap("after model load")`, before `inputs = tokenizer(...)`:

```python
print("[fix] running Gemma4TextExperts dequant shim...")
t0 = time.time()
n_patched = dequantize_gemma4_experts_inplace(model, MODEL_ID)
print(f"[fix] patched {n_patched} tensors in {time.time()-t0:.1f}s")
mem_report("after dequant shim")
enforce_cap("after dequant shim")
```

## Run Option A

```bash
cd <dev-root>/trt && rm -f /tmp/turboquant_test.log && \
  bash -c 'set -a; source .env; set +a; uv run python scripts/test_turboquant_gemma4.py' \
  > /tmp/turboquant_test.log 2>&1
```

Run in background via Bash tool `run_in_background: true`. Model load takes ~30s from cache, dequant shim ~30-60s, generation ~30s. Total ~2-3 minutes.

Check progress:
```bash
cat /tmp/turboquant_test.log | tr '\r' '\n' | grep -E "\[mem\]|\[gen\]|\[load\]|\[fix\]|\[summary\]|MODEL OUTPUT|Traceback"
free -h | head -2
```

## Option A success criteria

1. ✅ `[fix] patched 60 tensors` in log
2. ✅ Peak `sys_used` stays under 70 GB (shim adds ~2 GB transient, baseline ~65 GB)
3. ✅ `enforce_cap` never trips
4. ✅ **Decoded output is coherent English / Python code**, not multilingual token soup. The prompt is `"Write a Python quicksort function and briefly explain how it works."` — expect a real quicksort function and a short explanation.
5. ✅ Exit code 0

If output is STILL garbage after the shim, the dequant math is wrong or the wrong Parameters are being written. Double-check by printing `obj.shape`, `obj.dtype`, `weight_bf16.shape`, `weight_bf16.dtype`, a small slice of values before/after.

## After Option A passes

Don't delete the shim — **leave it in place as the baseline**, add Option B as a follow-up path gated by an env var or CLI flag. The user wants to compare Option A's output quality to Option B's output quality (should be identical since it's the same dequant math, just at different times).

---

# TASK PART 2 — OPTION B (FP8 runtime + forward patch)

**Goal:** Replace the bf16 expert Parameters with FP8 Parameters, install a monkey-patched `Gemma4TextExperts.forward` that dequantizes on-the-fly during matmul. Save ~23 GB persistent memory. Unlock 256K context under the 90 GB cap.

## Three pieces needed

### 1. Patched forward (copy original, substitute 2 linear calls)

Source to mirror is at `.venv/lib/python3.14/site-packages/transformers/models/gemma4/modeling_gemma4.py` lines **1262-1286**:

```python
# Original:
for expert_idx in expert_hit:
    expert_idx = expert_idx[0]
    if expert_idx == self.num_experts:
        continue
    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
    current_state = hidden_states[token_idx]
    gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
    current_hidden_states = self.act_fn(gate) * up
    current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
    current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
    final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
```

**Patched version** (copies the two linear calls, substitutes dequantized weights):

```python
def _patched_gemma4_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    import torch
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
        # --- DEQUANT HERE ---
        gate_up_w = self.gate_up_proj[expert_idx].to(torch.bfloat16) * self.gate_up_proj_scale[expert_idx]
        gate, up = nn.functional.linear(current_state, gate_up_w).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        # --- DEQUANT HERE ---
        down_w = self.down_proj[expert_idx].to(torch.bfloat16) * self.down_proj_scale[expert_idx]
        current_hidden_states = nn.functional.linear(current_hidden_states, down_w)
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states
```

Install class-wide **before model load**:
```python
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
Gemma4TextExperts.forward = _patched_gemma4_experts_forward
```

### 2. Post-load parameter swap (replaces the dequant shim)

After `from_pretrained` returns, walk the model, and for each Gemma4TextExperts submodule:

```python
def reparameterize_gemma4_experts_to_fp8(model, model_id: str) -> int:
    """Replace bf16 Gemma4TextExperts parameters with FP8 versions and register scales."""
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    cache_dir = Path(snapshot_download(model_id, allow_patterns=["*.safetensors*"]))
    shards = sorted(cache_dir.glob("model*.safetensors"))

    # Build a flat map: key -> shard
    key_to_shard = {}
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
                continue
            with safe_open(key_to_shard[wkey], framework="pt") as f:
                weight_fp8 = f.get_tensor(wkey)  # float8_e4m3fn
            with safe_open(key_to_shard[skey], framework="pt") as f:
                scale = f.get_tensor(skey)  # bfloat16

            device = next(mod.parameters()).device

            # Delete old bf16 Parameter first so its memory is freed BEFORE allocating the FP8 replacement
            delattr(mod, proj)
            mod.register_parameter(proj, nn.Parameter(weight_fp8.to(device), requires_grad=False))

            # Register the scale as an attribute (not a Parameter, to avoid save/load name collision)
            setattr(mod, f"{proj}_scale", scale.to(device))
            patched += 1

            del weight_fp8, scale
            gc.collect()
            torch.cuda.empty_cache()
    return patched
```

**Critical detail:** `delattr(mod, proj)` before the re-register is essential — without it, pytorch keeps the old bf16 Parameter alive and you don't see the memory drop.

### 3. Integration into main()

```python
# BEFORE loading the model
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
Gemma4TextExperts.forward = _patched_gemma4_experts_forward

# ... existing preflight, tokenizer load ...

# Model load
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
mem_report("model loaded")
enforce_cap("after model load")

# Replace the dequant shim with the reparameterize shim
n_patched = reparameterize_gemma4_experts_to_fp8(model, MODEL_ID)
print(f"[fix] reparameterized {n_patched} expert tensors to FP8")
mem_report("after FP8 reparameterize")
enforce_cap("after FP8 reparameterize")

# ... continue to generation ...
```

## Option B success criteria

1. ✅ `torch_alloc` drops from ~47 GB to **~24 GB** after reparameterize (saved ~23 GB)
2. ✅ `sys_used` drops from ~65 GB to **~42 GB**
3. ✅ Decoded output is **coherent Python quicksort code** (same quality as Option A)
4. ✅ Generation tok/s may be similar or slightly lower than Option A (extra dequant op per layer per token; the `.to(bf16)` + multiply is cheap relative to the matmul)
5. ✅ Long-context validation: bump `MAX_NEW_TOKENS` to 512 and pad the prompt to ~128K tokens (e.g., repeat a long context, or use a dummy long document) — verify peak `sys_used` stays well under 90 GB

## Watch out for

- `torch.cuda.empty_cache()` + explicit `del` between param swaps is necessary to see the memory drop in real time. Without it, pytorch holds freed memory in its allocator cache and you'll see no change.
- `nn.Parameter(fp8_tensor, requires_grad=False)` — `requires_grad=False` is important because FP8 doesn't support autograd.
- The forward patch uses `.to(torch.bfloat16)` which does the FP8→bf16 numerical conversion; confirm with a tiny probe before trusting at scale.
- If generation time roughly doubles, the dequant is dominating — it shouldn't for bits=4 but if it does, consider caching the last-used expert dequant.
- When running repeatedly, the `uv run` wrapper does a tiny sync check each time ("Uninstalled 1 package / Installed 1 package") — this is normal, not a memory issue.

---

## Things NOT to do

- ❌ **Do not write a new "FP8 is broken upstream" memory file.** We already have the research findings — this is execution work, not another root-cause investigation.
- ❌ **Do not try RedHatAI or protoLabsAI or Firworks or BCCard FP8 checkpoints.** They all have either the unfused legacy layout OR the same fused-MoE issue. LargitData is the confirmed good one for layout.
- ❌ **Do not reinstall compressed-tensors, transformers, or torch** — the versions in pyproject.toml are deliberate. `torch==2.11.0+cu130` is the only CUDA+Blackwell+aarch64 combo that works.
- ❌ **Do not modify pyproject.toml** unless you need a new small dep (unlikely for this task).
- ❌ **Do not stop the Ollama service or stt-service** without user permission — they're shared services.
- ❌ **Do not alter `VRAM_CAP_GB = 90.0`** — that's a hard user rule.
- ❌ **Do not add FP8 fallbacks or "guardrails"** that silently degrade — fail loud per CLAUDE.md.

## Memory files to check

`~/.claude/projects/-opt-dev-trt/memory/MEMORY.md` has five entries relevant to this work:
- `hardware_dell_gb10_max_pro.md` — correct product name
- `turboquant_python_package.md` — `pip install turboquant` API
- `feedback_research_before_pessimism.md` — WebSearch before asserting limits on Blackwell/CUDA/ARM64
- `reference_pytorch_gb10_install.md` — PyTorch GPU install path for GB10, SM121 confirmed working
- `project_gb10_memory_monitoring.md` — use psutil/free, not nvidia-smi, for memory on GB10

**Do not update memory until both Option A and Option B are validated end-to-end with coherent output and measured memory savings.** The prior session learned the hard way that premature memory saves become diary entries that mislead future sessions.

## Completion signal

When Option B is validated with coherent output and measured ~23 GB memory savings at short context, AND a 128K long-context test has run successfully, write ONE reference memory file at:

`~/.claude/projects/-opt-dev-trt/memory/reference_gemma4_moe_fp8_runtime_shim.md`

Summarizing:
- The upstream gap in compressed-tensors 0.15.0.1 (fused MoE not supported)
- The working user-space monkey-patch pattern (forward patch + parameter swap)
- Measured memory savings on Dell GB10 Max Pro
- The checkpoint that works (LargitData/gemma-4-26b-a4b-it-fp8) + tokenizer caveat

Update `MEMORY.md` index to include the new entry. Keep it terse and factual.
