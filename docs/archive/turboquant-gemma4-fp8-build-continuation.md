# Continuation: self-built FP8 quant of Gemma 4 26B-A4B (April 12, 2026)

**Read this first. Then check `TaskList`. Then check `~/.claude/plans/shiny-strolling-newt.md`. Then act.**

## State at handoff

- The approved plan lives at `~/.claude/plans/shiny-strolling-newt.md` — full Phase 0–8 design with file layouts, code, acceptance criteria, risks. **Read it.**
- **Tasks do NOT survive `/clear`.** You must recreate them at the start of the session using the "Task list to recreate" section at the bottom of this file. Do this BEFORE starting any actual work.
- Phases 0, 1, 2, 3, 4 are **completed**. The `scripts/gemma4_fp8/` package exists with `quant.py`, `modeling.py`, `loader.py`, `__init__.py`, all validated in isolation (no model load).
- Phase 5 (the FP8 build) is **in_progress but the build process died** — see "What went wrong" below. Task #11 needs a retry.
- Phases 6, 7, 8 are **pending and blocked by Phase 5**.
- The session that produced this state ran out of context before retrying Phase 5.

## What went wrong with the Phase 5 attempt

1. **First attempt:** subagent wrote `scripts/build_gemma4_fp8_checkpoint.py`, hit a `ModuleNotFoundError: No module named 'scripts'` at the lazy import in `main()`, fixed it by inserting the repo root into `sys.path` before the lazy import. Build never started because vLLM (`saricles/Qwen3-Coder-Next-NVFP4-GB10`, PID 1417896 / 1418624) had come up between baseline check and launch and was holding 44 GB.

2. **vLLM stopped:** the user authorized killing vLLM. SIGTERM was ignored, SIGKILL via `sudo -n kill -KILL 1417896 1418624` succeeded. Baseline restored to ~10 GB used / 111 GB available.

3. **Second attempt:** dispatched the build via a fresh Opus subagent. The subagent launched the build via `Bash(run_in_background=True)` and started a Monitor. The build got through preflight, started loading bf16 weights, reached "Loading weights: 4%|▍ | 42/1013" with sys_used=63.8 GB (Monitor sample) — then **the process died silently** with no traceback in `/tmp/build_gemma4_fp8.log`.

4. **dmesg shows NVRM Out of memory at 13:09:50** — but that's BEFORE the second attempt launched (which was around 13:11+). Those errors are stale, from earlier in the session when the build was attempted under vLLM contention. Not the cause of this death.

5. **Most likely cause of the silent death:** the subagent that launched the build via `run_in_background=True` finished its task and returned to the coordinator. Background processes spawned from a subagent's bash shell appear to be reaped when the subagent exits, even with `&` backgrounding. The Monitor task `b56k3fnn0` correctly noticed the process was gone (pgrep stopped matching) and exited.

**Lesson for the retry: the coordinator must launch the build DIRECTLY via `Bash(run_in_background=True)`, NOT delegate to a subagent.** The coordinator persists for the entire conversation; subagents do not.

## Current physical state

- `$FP8_CHECKPOINT/` does NOT exist yet (no successful build)
- `/tmp/build_gemma4_fp8.log` has 11 lines from the dead attempt — safe to delete or overwrite
- Memory baseline at handoff: ~9.5 GB used / 112 GB available
- vLLM is gone. Only stt-service (~3.5 GB GPU) is running
- The build script `<repo-root>/scripts/build_gemma4_fp8_checkpoint.py` exists with the sys.path fix

## What to do in the next session

1. **`TaskList`** — confirm the task graph (Phases 0–4 completed, Phase 5 in_progress, 6/7/8 pending blocked).
2. **Read the plan**: `Read ~/.claude/plans/shiny-strolling-newt.md`. Skip to "Phase 5".
3. **Read this file** end-to-end (you're here).
4. **Verify baseline** before doing anything:
   ```bash
   free -h | head -2  # expect <15 GB used
   nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv  # expect only stt-service
   ollama ps  # expect empty
   ```
   If sys_used > 20 GB or another GPU consumer is running, **STOP and ask the user** before proceeding.
5. **Verify the build script is intact:**
   ```bash
   ls -la <repo-root>/scripts/build_gemma4_fp8_checkpoint.py
   head -30 <repo-root>/scripts/build_gemma4_fp8_checkpoint.py  # confirm sys.path fix is at top of main()
   ```
6. **Launch the build DIRECTLY from the coordinator** (NOT via a subagent). Use the Bash tool with `run_in_background=true`:
   ```bash
   rm -f /tmp/build_gemma4_fp8.log && \
     bash -c 'set -a; source .env; set +a; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/build_gemma4_fp8_checkpoint.py' \
     > /tmp/build_gemma4_fp8.log 2>&1
   ```
7. **Start a Monitor** (also from the coordinator) at 2-second cadence to stream memory + log events. Use this exact command (the `[b]` trick avoids self-match):
   ```bash
   LOG=/tmp/build_gemma4_fp8.log; LAST=0; TICK=0; PAT="[b]uild_gemma4_fp8_checkpoint.py"; \
   while pgrep -f "$PAT" >/dev/null 2>&1; do \
     mem=$(awk '/MemTotal:/{t=$2} /MemAvailable:/{a=$2} END{printf "%.1f", (t-a)/1024/1024}' /proc/meminfo); \
     echo "[mem] t=${TICK}s sys_used=${mem}GB"; \
     if [ -f "$LOG" ] && [ -s "$LOG" ]; then \
       sz=$(stat -c %s "$LOG" 2>/dev/null || echo 0); \
       if [ "$sz" -gt "$LAST" ]; then \
         tail -c +$((LAST+1)) "$LOG" 2>/dev/null | tr '\r' '\n' | grep -E "^\[(mem|preflight|summary)\]|quantized layer|safetensors saved|Traceback|FATAL|Error|model loaded" || true; \
         LAST=$sz; \
       fi; \
     fi; \
     TICK=$((TICK+2)); sleep 2; \
   done; \
   mem=$(awk '/MemTotal:/{t=$2} /MemAvailable:/{a=$2} END{printf "%.1f", (t-a)/1024/1024}' /proc/meminfo); \
   echo "[watch] done, final sys_used=${mem}GB"
   ```
   Use the `Monitor` tool with `persistent: false` and `timeout_ms: 1500000` (25 min budget).
8. **Watch the events.** Hard kill threshold: **82 GB sys_used.** If any sample crosses 82 GB:
   ```bash
   pkill -9 -f "[b]uild_gemma4_fp8_checkpoint.py"
   ```
   then verify `free -h` shows recovery, dump the tail of the log, report the failure to the user, and **do not retry without their direction**.
9. **On clean completion** (background bash task fires its completion notification, log shows `[summary] FP8 checkpoint built at ...`):
   - Read tail of `/tmp/build_gemma4_fp8.log` (last ~80 lines)
   - Run the structural verification script from the plan's Phase 5 step 3 (also pasted below)
   - Mark task #11 completed via `TaskUpdate`
   - Proceed to Phase 6 (smoke test) by dispatching task #12 — this one CAN be delegated to a subagent because it's shorter
10. **Phase 6/7/8** follow per the plan file. Phase 8's commit step is critical — it locks in the work.

## Structural verification script (paste verbatim after build completes)

```bash
bash -c 'set -a; source .env; set +a; cd <dev-root>/trt && uv run python -c "
import json
from pathlib import Path
from safetensors import safe_open
import torch

out = Path(\"$FP8_CHECKPOINT\")
idx = json.loads((out / \"model.safetensors.index.json\").read_text())
weight_map = idx[\"weight_map\"]
print(f\"total keys: {len(weight_map)}\")
print(f\"shards: {len(set(weight_map.values()))}\")

expert_keys = [k for k in weight_map if \".experts.\" in k]
print(f\"expert keys: {len(expert_keys)}\")
assert len(expert_keys) == 30 * 4, f\"expected 120 expert keys, got {len(expert_keys)}\"

sample = {}
for needle in [\"experts.gate_up_proj\", \"experts.gate_up_proj_scale\", \"experts.down_proj\", \"experts.down_proj_scale\"]:
    matching = [k for k in expert_keys if k.endswith(needle)]
    assert matching, f\"no key ending with {needle}\"
    k = matching[0]
    shard = weight_map[k]
    with safe_open(out / shard, framework=\"pt\") as f:
        t = f.get_tensor(k)
        sample[needle] = (t.dtype, tuple(t.shape))

assert sample[\"experts.gate_up_proj\"][0] == torch.float8_e4m3fn
assert sample[\"experts.gate_up_proj_scale\"][0] == torch.bfloat16
assert sample[\"experts.down_proj\"][0] == torch.float8_e4m3fn
assert sample[\"experts.down_proj_scale\"][0] == torch.bfloat16
for k, v in sample.items(): print(f\"  {k}: dtype={v[0]} shape={v[1]}\")
print(\"OK: checkpoint has correct expert key dtypes and shapes\")
"'
```

## Files in this repo that matter (committed at the handoff commit, run `git log --oneline -5`)

Read these to understand what's already built:

| File | Purpose |
|---|---|
| `~/.claude/plans/shiny-strolling-newt.md` | The full approved plan. Read this. |
| `docs/kb/turboquant-gemma4-continuation.md` | Prior session's handoff for Option A/B/forward-patch work |
| `docs/kb/turboquant-gemma4-research-apr2026.md` | This session's research findings: working external stacks, why pip turboquant 0.2.0 is reference-only, the 16K prefill explosion analysis |
| `docs/kb/turboquant-gemma4-fp8-build-continuation.md` | This file |
| `scripts/test_turboquant_gemma4.py` | Existing harness with Option A (dequant shim) and Option B (FP8 reparameterize + monkey-patch forward). Phase 8 will add Option C (uses the new clean loader) |
| `scripts/test_nvfp4_gemma4.py` | NVFP4 attempt (failed — bg-digitalservices checkpoint requires vLLM patches, won't load on transformers). Kept for history. |
| `scripts/build_gemma4_fp8_checkpoint.py` | The build script for Phase 5. sys.path fix already applied. |
| `scripts/gemma4_fp8/__init__.py` | Public exports |
| `scripts/gemma4_fp8/quant.py` | Per-channel max-abs FP8 quantizer, round-trip validated, error 0.0938 |
| `scripts/gemma4_fp8/modeling.py` | `Gemma4TextExpertsFP8` subclass, validated against real config |
| `scripts/gemma4_fp8/loader.py` | `load_gemma4_fp8` + `gemma4_experts_fp8_class_swap` context manager, validated |
| `references/bg-digitalservices/quantize_gemma4_moe.py` | THIRD PARTY reference, gitignored. Their `_QuantGemma4TextExperts._setup()` unfuse pattern. NVFP4 + vLLM, not directly applicable but useful reading. |
| `references/bg-digitalservices/gemma4_patched.py` | THIRD PARTY reference, gitignored. Their vLLM `_weight_iterator` 3D-to-2D exploder. |
| `~/.claude/projects/-opt-dev-trt/memory/feedback_fp8_checkpoint_layouts.md` | Saved memory: vendor-to-defect map for FP8/NVFP4 Gemma 4 checkpoints |
| `~/.claude/projects/-opt-dev-trt/memory/project_gemma4_fp8_self_build.md` | Saved memory: this self-built path |

## Hardware / environment context (do NOT re-research)

- **Dell GB10 Max Pro** (Dell whitelabel of NVIDIA GB10 reference / DGX Spark-class)
- **128 GB unified LPDDR5x**, 273 GB/s bandwidth, **Blackwell SM121** GPU
- **No swap**, no OOM guardrails — process kills propagate to other sessions if cap is breached
- `nvidia-smi` returns N/A on unified memory; use `free -h`, `psutil`, `/proc/meminfo`
- Memory cap: **90 GB sys_used HARD**. Kill threshold: **82 GB**. NON-NEGOTIABLE.
- Other persistent processes: `stt-service` (~3.5 GB GPU). Do NOT stop without user permission.
- Python venv: `.venv` managed by `uv`. Always `uv run python ...`, never activate manually.
- Python version: 3.14.2.
- Key packages pinned in pyproject.toml: torch==2.11.0+cu130, transformers==5.5.3, compressed-tensors==0.15.0.1, accelerate==1.13.0, turboquant==0.2.0, psutil==7.2.2.
- HF cache: `$HF_HOME/`. NOT `~/.cache`.
- Source model `google/gemma-4-26B-A4B-it` is already downloaded (~49 GB bf16, license accepted).
- `.env` has `HF_TOKEN`. Source it before running anything: `bash -c 'set -a; source .env; set +a; uv run python ...'`.

## Things NOT to do

- Do **NOT** re-investigate which public FP8/NVFP4 checkpoint loads cleanly. None do. Confirmed via direct testing this session: `LargitData/gemma-4-26b-a4b-it-fp8` (fused-3D, scales drop), `protoLabsAI/gemma-4-26B-A4B-it-FP8` (legacy unfused, garbage), `RedHatAI/*-FP8-Dynamic` (legacy unfused, garbage), `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` (missing experts + dim mismatch + vLLM-only loader). Building our own is the only path.
- Do **NOT** try to fix the 16K prefill memory explosion as part of this work. It's a transformers Gemma 4 base-forward issue, not weight-related. Out of scope. The plan file says so explicitly.
- Do **NOT** try to install modelopt or llm-compressor for the quantization. We're rolling our own ~20-line per-channel max-abs FP8 quant in `scripts/gemma4_fp8/quant.py`, already validated.
- Do **NOT** delegate the long-running build to a subagent — they get reaped when the agent exits. Run via the coordinator's own `Bash(run_in_background=True)`.
- Do **NOT** use `trust_remote_code` or `auto_map`. The class-swap loader pattern is the chosen design.
- Do **NOT** modify `VRAM_CAP_GB = 90.0` anywhere. Do not skip `enforce_cap` calls.
- Do **NOT** stop `stt-service` without user permission.
- Do **NOT** commit the build artifact (`$FP8_CHECKPOINT/`) — it's outside the repo and should stay there.
- Do **NOT** start the build under contention. If sys_used > 20 GB at baseline OR another big GPU consumer is present, ASK THE USER first.
- Do **NOT** add Co-Authored-By lines for Claude in any commit message.

## Quick recovery cheatsheet

If you only have a few minutes and need to make progress:

1. `git log --oneline -5` — confirm commit `3f92165` (the handoff commit) is in history
2. `ls -la $FP8_CHECKPOINT/` — does the artifact exist?
   - If yes → skip Phase 5, go straight to Phase 6
   - If no → run Phase 5 (steps 4–10 above)
3. Recreate the task list from the section below
4. Start executing from the first `pending` task

## Task list to recreate (tasks do NOT survive /clear — recreate them here)

Run these `TaskCreate` calls at the start of the new session, then immediately `TaskUpdate` the phases 0–4 ones to `completed` and Phase 5 to `in_progress` (or `pending` if the build artifact doesn't yet exist when you check). Use `TaskUpdate addBlockedBy` for the dependency chain.

**Task: Phase 0 — Save FP8 institutional memory** (STATUS: completed in prior session, just recreate and mark completed)
- Description: Save two auto-memory entries at `~/.claude/projects/-opt-dev-trt/memory/`: `feedback_fp8_checkpoint_layouts.md` (vendor-to-defect map for FP8/NVFP4 Gemma 4 checkpoints) and `project_gemma4_fp8_self_build.md` (this self-built path). Add one-line entries to `MEMORY.md` index.
- Both files already exist from the prior session — verify with `ls ~/.claude/projects/-opt-dev-trt/memory/feedback_fp8_checkpoint_layouts.md` and mark completed.

**Task: Phase 1 — Download bg-digitalservices reference scripts** (STATUS: completed, recreate and mark completed)
- Description: Download `quantize_gemma4_moe.py` (10.5 KB) and `gemma4_patched.py` (51 KB) from `https://huggingface.co/bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16/raw/main/` into `<repo-root>/references/bg-digitalservices/`. Add `references/` to `.gitignore`.
- Already done. Verify `ls <repo-root>/references/bg-digitalservices/` shows both files, mark completed.

**Task: Phase 2 — Implement `scripts/gemma4_fp8/quant.py`** (STATUS: completed, recreate and mark completed)
- Description: Per-channel max-abs FP8 quantization utility for 3D MoE expert tensors. Pure tensor math, no transformers, no model load. Round-trip self-test with `python -m scripts.gemma4_fp8.quant` should print `OK: quantize_3d_to_fp8 round-trip passes`.
- Already done. Verify the file exists, run the self-test, mark completed.

**Task: Phase 3 — Implement `scripts/gemma4_fp8/modeling.py`** (STATUS: completed, recreate and mark completed)
- Description: `Gemma4TextExpertsFP8` subclass with FP8 weight Parameters and bf16 scale buffers. Calls `nn.Module.__init__(self)` directly (not `super().__init__`) to skip the parent's bf16 Parameter allocation. Forward does per-call dequant.
- Already done. Verify by instantiating under `init_empty_weights` with the real config, mark completed.

**Task: Phase 4 — Implement `scripts/gemma4_fp8/loader.py`** (STATUS: completed, recreate and mark completed)
- Description: `gemma4_experts_fp8_class_swap()` contextmanager + `load_gemma4_fp8(model_id, **kwargs)` helper that monkey-patches `transformers.models.gemma4.modeling_gemma4.Gemma4TextExperts` → `Gemma4TextExpertsFP8` for the duration of `from_pretrained`.
- Already done. Verify by running the class-swap self-test, mark completed.

**Task: Phase 5 — Build the FP8 checkpoint** (STATUS: in_progress — NEEDS RETRY, script exists at `scripts/build_gemma4_fp8_checkpoint.py`)
- Description: Load `google/gemma-4-26B-A4B-it` bf16, quantize all Gemma4TextExperts to FP8 + bf16 scales, save sharded safetensors checkpoint at `$FP8_CHECKPOINT/`. 90 GB memory cap, kill at 82 GB. **MUST be launched directly from the coordinator** via `Bash(run_in_background=True)` — NOT via a subagent (subagents get reaped, taking their background children with them).
- Execution steps are in the "What to do in the next session" section above (steps 4–10). Also in the plan file's Phase 5 section.

**Task: Phase 6 — Smoke-test FP8 checkpoint load and short-context generation** (STATUS: pending, BLOCKED BY Phase 5)
- Description: Create `scripts/test_gemma4_fp8_load.py`. Uses `load_gemma4_fp8()` to load the Phase 5 artifact, verifies no UNEXPECTED/MISSING/MISMATCH in the load report, checks expert dtypes are `float8_e4m3fn` and scale buffers are `bfloat16`, runs a short quicksort prompt with `TurboQuantCache(bits=4)`, confirms coherent output. Peak `sys_used` ≤ 50 GB. `tok/s ≥ 4.0`. Full details in plan file Phase 6.
- `addBlockedBy: [Phase 5 task id]`

**Task: Phase 7 — Quality compare FP8 vs bf16 reference** (STATUS: pending, BLOCKED BY Phase 6)
- Description: Run same prompt through bf16 Option A path (`scripts/test_turboquant_gemma4.py` with no OPTION_B) AND through the new FP8 path (`scripts/test_gemma4_fp8_load.py`). Compare first 50 decoded token IDs. Identical or diverges late = pass. Diverges in first 10 tokens = fail (FP8 quant quality bug), stop and report. Write a short summary at `/tmp/quality_compare_summary.md` for the Phase 8 KB doc.
- `addBlockedBy: [Phase 6 task id]`

**Task: Phase 8 — Document, integrate, commit** (STATUS: pending, BLOCKED BY Phase 7)
- Description:
  1. Write `docs/kb/gemma4-fp8-self-built-howto.md` (why, how to build, how to load, storage format, validation results, open issues, when to use vs Option A/B)
  2. Add `OPTION_C=1` mode to `scripts/test_turboquant_gemma4.py` that uses the new loader instead of Option A/B runtime shims
  3. Smoke-test OPTION_C runs end-to-end with active memory monitoring
  4. `git commit` everything with the message in the plan file's Phase 8 section (no Co-Authored-By)
- `addBlockedBy: [Phase 7 task id]`

## Task dependency summary

```
Phase 0 ────┐
Phase 1 ────┤
Phase 2 ────┤  (all parallel, all completed)
Phase 3 ────┤
Phase 4 ────┘
            │
            ▼
          Phase 5 ──(BLOCKED until build succeeds)──▶ Phase 6 ──▶ Phase 7 ──▶ Phase 8
```
