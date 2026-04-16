# Gemma 4 26B-A4B FP8 Self-Build — Retrospective

**Date:** 2026-04-13
**Effort window:** 2026-04-10 → 2026-04-13 (~4 days, spread across several sessions)
**Outcome:** ✓ Ship. Self-built FP8 checkpoint validated at 99.2% token match vs native bf16. Artifact committed in-repo under `models/gemma-4-26b-a4b-it-fp8/` via Git LFS, awaiting pre-publish test gate before HF upload.

This is a retrospective, not a results doc. For the validation numbers and
reproduction steps, see `gemma4-fp8-self-built-results.md` in the same
directory. This file is about what we learned doing the work — what worked,
what didn't, and what I'd change next time.

---

## The project in one paragraph

TurboQuant KV cache compression needed a working Gemma 4 26B-A4B MoE
checkpoint on GB10. Every public FP8/NVFP4 release for this model failed to
load cleanly on `transformers 5.5.3` + `compressed-tensors 0.15.0.1` —
the fused 3D `Gemma4TextExperts` layout is incompatible with every public
quantization tool (modelopt, llm-compressor, AutoFP8), which all silently
skip the experts (91% of the model). So we built our own FP8 quant with a
custom class-swap loader. Plan of record:
`~/.claude/plans/shiny-strolling-newt.md` (Phases 0–8). Phase 5's build
strategy was rewritten mid-effort per
`~/.claude/plans/velvet-wiggling-toast.md` after the first approach hit
a hard memory wall.

---

## What worked

### 1. Phases 0–4 were textbook

Scaffolding (memory entries → quant utility → modeling subclass → loader)
landed in order, each phase independently testable, no rework. The round-
trip self-test on `quantize_3d_to_fp8()` caught zero bugs because the math
is trivial — but that's the point. Keeping the quant utility *pure* (3D
tensor in, 3D tensor + scale out, no globals, no model objects) meant we
could verify it with a three-line test and then trust it for the whole
build.

Lesson: pure functions that handle raw tensors (not model objects) can be
validated in isolation and then consumed by any build strategy. This is
what let us swap Phase 5 from "full-load + named_modules walk" to
"shard-streaming" without touching the quant math.

### 2. Class-swap loader pattern

`scripts/gemma4_fp8/loader.py` installs a context manager that replaces
`Gemma4TextExperts` with `Gemma4TextExpertsFP8` during `from_pretrained`
only, then restores the original class on exit. No `trust_remote_code`, no
`auto_map`, no runtime monkey-patching of a loaded model, no re-export
dance in `transformers`. The state-dict path populates our FP8 parameters
and bf16 scale buffers without any custom `_load_from_state_dict` override.

The key design call was making the scale buffers *flat* named
(`gate_up_proj_scale`) instead of nested (`gate_up_proj.weight_scale`) so
they don't collide with the parameter namespace. That's mundane but it's
the reason we got a `0/0/0/0` load report on the first try with no
remapping logic.

Lesson: when extending a framework class, read the parent's
`__init__`/`state_dict` code path before you design your attribute names.
Names are API.

### 3. Shard-streaming build rewrite

The full-load strategy (Phase 5 as originally planned) couldn't finish.
The shard-streaming rewrite dropped peak CPU memory from 82+ GB (deadly)
to ~16 GB (trivial), preserving the quant math byte-for-byte. The build
is now effectively free — 68 seconds, never touches the GPU, and doesn't
care about unified-memory contention.

Lesson: when your "load everything, transform, save" pipeline doesn't fit
memory, *stream at the file format boundary*. safetensors is mmap-backed;
you can `safe_open` a 47 GB shard without paying 47 GB of RSS, and
`get_tensor(key)` materializes one tensor at a time. This pattern
generalizes well — same trick works for HF Datasets (Arrow streaming),
Parquet, and any file format whose readers expose lazy/random access.

### 4. Memory discipline (90 GB cap, `mem_report` everywhere)

Every script emits structured `[mem]` lines at every phase boundary, and
`enforce_cap()` kills the process at 90 GB `sys_used`. This turned every
OOM into a clean failure with a timeline rather than a crashed interpreter
with no trail. During the Phase 5 incident the memory timeline showed us
the exact tensor where the build died (314/1013), and the timeline showed
*linear* growth — not a spike — which is how we proved the issue was
cumulative memory pressure rather than a single bad allocation.

Lesson: always instrument memory in long-running scripts on a constrained
platform. The cost is ~10 lines of Python; the return is "I know what
killed it" every time.

---

## What didn't work (and what we changed)

### 1. Original plan's memory estimate was wrong

`shiny-strolling-newt.md` Phase 5 estimated ~62 GB peak for the full-load
approach. The real number on `transformers 5.5.3` with `device_map="cuda"`
is well over 90 GB at 100% load. The plan wasn't doing arithmetic on the
wrong model — it just underestimated the transformers loader's transient
overhead (intermediate dtype conversions, double-buffering during
`named_modules` walks, etc.) for a 26B parameter MoE on a unified-memory
system.

What I'd do differently: **never trust a memory estimate derived from
"params × bytes/param"**. On unified memory especially, transient load
overhead can easily be 2x the steady-state weight size. For GB10, always
build a small pilot that runs the same loader path on a *tenth* of the
weights and extrapolate from observed peaks, not from parameter counts.

### 2. First build attempt: vLLM contention misdiagnosis

The first Phase 5 launch died at 82.8 GB `sys_used`. Initial reaction was
"the plan's estimate was off" — almost triggered a refactor. Actual cause:
a vLLM server loaded in a parallel Claude session was holding 44 GB of
unified memory. On GB10 that memory is our budget too. The misdiagnosis
came within minutes of causing a speculative plan rewrite, which would
have cost a day of wasted work for no reason.

Root fix: added `feedback_preflight_checks_vital.md` to memory as an
operator habit rule — *run `free -h` / `nvidia-smi --query-compute-apps` /
`pgrep vllm|trtllm|ollama|sglang|tgi` before any >20 GB GB10 op, and
re-check right before launch*. Parallel sessions can start contenders
between baseline and action.

Second-order lesson: when a build fails at an unexpected memory point,
**the first hypothesis should always be "what else is running?"**, not
"my estimate was wrong". External contention is cheap to rule out and
expensive to miss.

### 3. I almost wrote the pre-flight check as code

After the vLLM incident I started adding a 150-line
`check_unexpected_consumers()` function to the build script. User caught
it: *"You may be taking this a little too literally, creating code for
it."* The right answer was a two-line operator habit (run three shell
commands before launch), not a policy-engine-in-every-script.

Lesson: not every lesson becomes code. Habit-level guidance belongs in
memory / runbooks, not in every script's preamble. Reserve code for things
the machine can do better than the operator; leave everything else to
operator discipline.

### 4. Second build attempt: deterministic 82 GB OOM in pristine environment

After clearing vLLM, the *same* build script died at *exactly* the same
tensor (314/1013) at the same memory level. This ruled out contention and
confirmed the full-load approach genuinely couldn't fit. That's when we
replaced the whole Phase 5 strategy with shard-streaming via
`velvet-wiggling-toast.md`.

Lesson: *determinism is diagnostic*. If a failure repeats at exactly the
same place across clean runs, it's not transient — it's structural, and
you need a different approach, not a retry.

### 5. Ultrareview caught three real issues in the rewrite plan

Before executing `velvet-wiggling-toast.md`, ultrareview flagged:
1. The expert key regex was based on a wrong guess — actual pattern is
   `model.language_model.layers.N.experts.{gate_up_proj, down_proj}`,
   not `language_model.model.layers.N.mlp.experts.*`.
2. The rewritten `main()` dropped the `sys.path` fix at the top of the
   function, which would cause `ModuleNotFoundError: No module named
   'scripts'` when invoked via `uv run python`.
3. Several imports (`time`, `AutoConfig`, `AutoModelForCausalLM`, `math`)
   would become dead after the rewrite and needed to be removed.

All three were fixed in the plan patch *before* execution, saving an
estimated ~30 minutes of debug-and-retry.

Lesson: *ultrareview earns its cost*. Plan → ultrareview → patch → execute
caught all three issues before any compute was spent. Running the plan
unvalidated would have burned time on known-preventable failures.

---

## Decisions worth recording

### D1. FP8 over INT4/NVFP4

FP8 was chosen because (a) we already had FP8 working in the Option B
runtime reparameterize shim, (b) the bf16 scale recovery is trivial
(multiply, no dequant table), (c) `torch.float8_e4m3fn` is a first-class
PyTorch dtype on Blackwell with native storage semantics, and (d) we
didn't need the extra compression ratio from NVFP4 — 27 GB vs 17 GB matters
less than "it works on the first try."

The right time to revisit NVFP4: if a future workload needs ≤ 20 GB
weight footprint on GB10. Not before.

### D2. Per-(expert, output-channel) max-abs scale, no calibration

We deliberately avoided calibration data, activation statistics, and
outlier handling beyond a 1e-8 clamp. The assumption was that Gemma 4's
bf16 MoE experts don't have pathological outliers that would require
AWQ/SmoothQuant-style handling. The Phase 7 quality compare validated
this assumption: 99.2% token match vs bf16 reference is as good as FP8
realistically gets on greedy decoding, and the single divergence was a
benign near-tied-logit flip at token 4 that immediately re-converged.

If quality regression *had* shown up (output drift, repetition loops,
gibberish), the fallback was per-tensor scale + saturation clamp, then
stochastic rounding, then AWQ. We never needed any of those.

### D3. Commit the artifact into the project, not the HF cache

The original Phase 8 guidance was "don't commit the build artifact, it
lives in `$MODELS_CACHE/...` outside the repo." The user changed
this mid-Phase 8: commit the tensors into the project under
`models/gemma-4-26b-a4b-it-fp8/`, Git LFS for the shards,
HF-conformant structure so the directory can be uploaded as-is when the
time comes.

Consequence: the repo now has a hard dependency on Git LFS for any clone
that wants the weights, and `git push` will incur LFS bandwidth charges.
Both are acceptable — the alternative (regenerate from build script every
clone) is cheaper in LFS bills but erodes reproducibility when the build
script, transformers version, or base model change.

### D4. Publishing gated on "fully tested"

The user drew an explicit gate: *confirm the name, commit locally, but do
not publish to HF until we are fully tested*. That gate stands. The
pre-publish test matrix is in the next section; nothing uploads until it
passes.

### D5. HF naming: no prefix, HF-canonical hyphens

`aeyeops/gemma-4-26b-a4b-it-fp8`, following the convention of
`RedHatAI/gemma-2-9b-it-FP8-dynamic` etc. — base model name + quant
suffix, no redundant org prefix (the namespace already carries the org).
The `-it` suffix is Google's shorthand for "instruction-tuned" and we
keep it because dropping it would misrepresent which base we quantized
from.

### D6. Loader lives in the source repo, not the model repo

`scripts/gemma4_fp8/{quant,modeling,loader}.py` stay in the `trt` source
repo. The HF model repo ships only the weights, config, tokenizer, and
README. Users wanting to load the checkpoint need to clone `trt` for the
loader or copy the three files. Alternative would be to ship the loader
inside the model repo via `trust_remote_code`, but `trust_remote_code` is
a security hazard and we'd rather not normalize it for anyone downstream.

---

## Pre-publish test gate ("fully tested")

Before anything pushes to `aeyeops/gemma-4-26b-a4b-it-fp8` on
HuggingFace, all of these must pass:

1. **Phase 6 smoke test re-run** from a fresh Python process on the
   current repo state. Load report `0/0/0/0`, 30 verified modules, tok/s ≥ 4.0.
2. **OPTION_C run through `scripts/test_turboquant_gemma4.py`** — the
   canonical TurboQuant harness, so the FP8 self-built path is validated
   through the same code path as Options A and B. Peak `sys_used` ≤ 50 GB,
   exit code 0, coherent output.
3. **Longer output budget:** re-run Phase 6 with `MAX_NEW_TOKENS=512` and
   verify the output remains coherent (no repetition loops, no gibberish,
   no premature EOS). FP8 quant drift sometimes shows up only after a
   long decode horizon; 128 tokens is not enough to be sure.
4. **Prompt diversity:** run with at least three different prompts —
   the quicksort code prompt (current), one pure natural-language Q&A
   prompt, and one prompt with a mixed code+prose answer. Confirm coherent
   output for each.
5. **Long-context handling up to ~4K tokens:** pad the input prompt to
   4K tokens and verify the FP8 path handles it without OOM. We know
   ~16K is the upstream `modeling_gemma4` wall; 4K should be safe and
   gives us a real "not-smoke-test" context length.
6. **Cross-session reproducibility:** full build → load → generate from
   a clean environment, by a human operator, following the reproduction
   steps in `gemma4-fp8-self-built-results.md`. Catches doc drift.
7. **Clean `git status` after the commit,** with the artifact tracked
   via LFS and no accidental secrets in the staged set (check
   `.env`, `api_keys.env` are not staged).

When all seven pass, the gate is open and HF push is authorized. Until
then, the repo stays local.

### Gate status: PASSED on 2026-04-13

All seven items verified on GB10 Max Pro under the 90 GB cap:
- Item 1 (smoke re-run): PASS (8.0 tok/s, peak sys_used 41.85 GB)
- Item 2 (OPTION_C short): PASS (exit 0, 8.3 tok/s, peak torch 26.97 GB)
- Item 3 (MAX_NEW_TOKENS=512): PASS (no repetition, coherent across 512-token horizon)
- Item 4 (prompt diversity ×3): PASS (code/NL/mixed — all 5 coherence checks passed)
- Item 5 (4K context): PASS (4128 input tokens, 3.5 tok/s, peak torch 32.04 GB)
- Item 6 (reproducibility walk-through): PASS (no doc drift, 99.2% token match vs bf16)
- Item 7 (git hygiene): PASS (clean, no secret leaks, 7 LFS objects intact)

Authorized for HF publish in a subsequent session.

### What we're *not* testing before publish

- **>16K context** — known upstream issue in `modeling_gemma4`, unrelated
  to quantization, deferred.
- **Throughput parity with bf16** — we already know it's ~30% slower and
  that's documented in the model card as a known limitation.
- **Multi-GPU / distributed inference** — GB10 is single-GPU, not in scope.
- **Fine-tuning / continued training** — this is an inference-only build.

---

## What I'd do differently next time

1. **Start with shard-streaming.** The "load everything, walk, save"
   pattern should be reserved for models that clearly fit in memory with
   a large margin. Anything within 2x of unified memory should default to
   streaming at the file boundary.
2. **Estimate memory from a pilot, not from `params × 2`.** Build a
   1/10-scale pilot on fewer layers, measure peak, scale linearly.
   Saves a whole class of plan-estimate disasters.
3. **Bake the pre-flight check into the operator runbook from day one.**
   It's a 10-second habit that would have saved us the vLLM misdiagnosis
   incident entirely.
4. **Plan the directory name before you write the model card.** I wrote
   `models/gemma4-26b-a4b-it-fp8-selfbuilt/README.md` with the loader
   example pointing at the wrong path twice (first path change, then
   directory rename). Naming is cheap to get right up front and
   annoying to chase through every reference.
5. **Write the retrospective live.** This doc would have been cleaner if
   I'd jotted notes during the work instead of reconstructing from logs
   and memory at the end. Next time: one "decisions.md" scratch file at
   the start of the effort.

---

## Referenced artifacts

- **Plan of record:** `~/.claude/plans/shiny-strolling-newt.md` (Phases 0–8)
- **Phase 5 rewrite plan:** `~/.claude/plans/velvet-wiggling-toast.md` (shard-streaming)
- **Results doc:** `docs/kb/gemma4-fp8-self-built-results.md`
- **Historical handoff:** `docs/kb/turboquant-gemma4-fp8-build-continuation.md` (frozen, describes the pre-shard-streaming state)
- **Vendor defect map:** `~/.claude/projects/-opt-dev-trt/memory/feedback_fp8_checkpoint_layouts.md`
- **Pre-flight rule:** `~/.claude/projects/-opt-dev-trt/memory/feedback_preflight_checks_vital.md`
- **Project memory:** `~/.claude/projects/-opt-dev-trt/memory/project_gemma4_fp8_self_build.md`
- **Source code:** `scripts/gemma4_fp8/{quant,modeling,loader}.py`, `scripts/build_gemma4_fp8_checkpoint.py`, `scripts/test_gemma4_fp8_load.py`, `scripts/test_gemma4_bf16_reference.py`
- **Build artifact:** `models/gemma-4-26b-a4b-it-fp8/` (Git LFS)
- **Intended HF location:** `aeyeops/gemma-4-26b-a4b-it-fp8` (not yet published)
