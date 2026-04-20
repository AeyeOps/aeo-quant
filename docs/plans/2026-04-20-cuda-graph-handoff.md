# 2026-04-20 — CUDA graph capture handoff

Continuation of the 2026-04-18 round. That plan queued "CUDA graph
capture for the decode step" as the highest-leverage remaining lever
inside `transformers.generate()`. This session executed the spike,
scoped the win, and hit a blocker that warrants a fresh session to
resolve rather than more unstructured debugging in this one.

Nothing from this session shipped. The capturable-cache code and all
spike scripts live in `tmp/` (gitignored); `src/` is untouched.

## What we proved

1. **CUDA graph capture is mechanically compatible with Gemma 4 NVFP4
   decode through our own cache.** A fixed-buffer cache modeled on
   `transformers.cache_utils.StaticLayer` (pre-allocated K/V buffers,
   tensor-driven `cumulative_length`) captures cleanly, replays
   correctly, and does not crash under HF's attention/mask code paths.

2. **Short-context (cum_len ≤ ~70 on this model) parity with production
   holds byte-for-byte under capture+replay.** Spike v2 measured 0/30
   mismatches on the original 30-token harness.

3. **On raw eager + capturable cache + explicit `torch.cuda.graph()`,
   captured decode runs at 19.19 tok/s vs the raw-eager baseline of
   15.23 tok/s — a 1.26× speedup** on Gemma 4 NVFP4, prompt_len=41,
   30 token generation. Bit-for-bit identical output in that window.

## What we disproved

1. **The loader's `torch.compile(mode="reduce-overhead")` is actively
   harmful, not a no-op.** It's only inert when HF's generate() bypasses
   `OptimizedModule.__call__`. Direct `model(...)` calls (needed for any
   capture pipeline) engage compile's internal cudagraph_trees, which
   conflict with Gemma4HybridTurboQuantCache's `torch.cat` growth
   (`RuntimeError: accessing tensor output of CUDAGraphs that has been
   overwritten by a subsequent run`) and also fights explicit
   `torch.cuda.graph()` capture (`AssertionError` in
   `cudagraph_trees.dealloc_current_path_weakrefs`). See the matrix
   spike result below.

2. **The single-commit 0.1.13 path the user asked for is not
   achievable yet.** It required a capturable cache that matches
   production token-for-token across realistic generation lengths, and
   we don't have that.

## The blocker

### Observation

Even the simplest possible capturable cache — V1, no quantize path, just
`index_copy_` into a fixed buffer with a tensor cumulative_length —
**deterministically diverges from `Gemma4HybridTurboQuantCache` at decode
step 49 / cumulative_length ≈ 90 under pure eager decode of Gemma 4
NVFP4**. Prefill + first 49 decode tokens match byte-for-byte; step 50
onwards is garbage (101/150 mismatches).

### What we ruled out

* **Quantize-path drift.** V2 with always-on quantize-and-blend was
  implemented and tested; V1 (no quantize) fails the same way, so the
  quantize path is not the source of divergence.

* **`is_sliding` routing in HF masking_utils.** Originally all layers
  reported `is_sliding=False`, forcing HF's
  `create_sliding_window_causal_mask` to pick layer 0 for both mask
  variants. Fixed to per-layer-type (True for Gemma 4's 25 sliding
  layers, False for the 5 full layers). No change in divergence step.

* **`get_seq_length` returning tensor vs Python int.** HF's
  `_preprocess_mask_arguments` handles both; comment at
  `masking_utils.py:848` specifically calls out StaticLayer returning
  a tensor. Tested both; V1 diverges identically at step 49 either way.

* **Cross-layer advance of `cumulative_length`.** Each layer owns its
  own; advancing one doesn't affect others. Verified by code reading.

### Leading hypotheses, ranked

1. **Attention-kernel numerical drift from the zero-padded unused view
   slots.** My cache returns K/V shape `(B, H, max_cache_len, D)` with
   valid data in `[0, cum_len)` and zeros in `[cum_len, max_cache_len)`.
   Production returns `(B, H, cum_len, D)` — no padding. SDPA (or
   whatever attn_implementation resolves to on this build) likely
   selects a different kernel / different fp32-accumulation order for
   the larger K tensor. Over ~50 decode steps the drift compounds enough
   that greedy argmax crosses a top-1 vs top-2 boundary at step 49.
   **Not yet verified.** Verification path: capture the per-step logit
   diff between production and capturable on CUDA, plot max |Δlogit|
   over step index. If it grows smoothly from step 0 and crosses the
   top-1 flip threshold at step ~49, this is the mechanism.

2. **HF's built-in StaticCache produces this same divergence with
   Gemma 4.** If `StaticCache(config=model.config,
   max_cache_len=199)` used as a drop-in in the same test diverges from
   `Gemma4HybridTurboQuantCache` at the same step, the issue is
   fundamental to fixed-buffer approaches on this model, not specific
   to my implementation. If StaticCache converges, something in my
   cache's implementation is wrong and worth hunting. **This is the
   single highest-value next experiment** — one ~2-minute model load.

3. **Rotary-embedding or attention-layer interaction specific to
   Gemma 4 at certain sequence length.** Gemma 4 has hybrid attention
   with `sliding_window=1024` and `max_position_embeddings=262144`. The
   divergence at exactly position 90 doesn't match any config threshold
   we inspected, but the hybrid dispatch + rotary application may have
   a subtle path we haven't audited. Less likely than 1 and 2.

## Measured numbers

### Matrix spike (single model load, 4 configurations, Gemma 4 NVFP4, prompt_len=41, 30 tokens)

```
config                   tok/s      s  tokens  parity (vs eager_raw)
-----------------------------------------------------
eager_raw                15.23   1.90      29  OK               (reference)
eager_compiled               -      -       -  ERROR: RuntimeError
graphs_raw               19.19   1.35      26  OK               (+26% over eager_raw)
graphs_compiled              -      -       -  ERROR: AssertionError
```

The `graphs_raw` 1.26× applies only in the short-context window where
parity held. It is not a shippable measurement on its own.

### Long-gen parity (V2 cache, 200-token gen, same prompt/model)

```
first mismatch at decode step: 49 (abs seq position 90, residual NOT crossed)
parity: 151/200 mismatches, max matching prefix = 49
```

### Long-gen parity (V1 cache, same setup, 150-token gen)

```
first mismatch at decode step: 49 (abs seq position 90, residual NOT crossed)
parity: 101/150 mismatches, max matching prefix = 49
```

V1 and V2 diverge at the same step → not a quantize-path issue.

## Artifacts (all in `tmp/`, gitignored)

* `tmp/capturable_turboquant_cache.py` — V2 cache with fixed-buffer
  quantize-and-blend path preserving TurboQuant semantics (tensor-gated
  promotion). Untested against model beyond the initial parity spike.
* `tmp/capturable_cache_v1_only.py` — V1 cache, no quantize path,
  identical fixed-buffer structure. Reference for isolating bugs.
* `tmp/cuda_graph_spike.py` — original 30-token decode spike (eager vs
  captured). Passes within the 30-token window.
* `tmp/cuda_graph_matrix.py` — compile × graphs matrix spike. Source of
  the table above.
* `tmp/parity_capturable_vs_production.py` — synthetic update-only
  parity (no model); shows ~0.004 bf16 LSB drift past residual window.
* `tmp/long_gen_token_parity.py` — real-model long-gen parity for V2.
* `tmp/long_gen_v1_test.py` — real-model long-gen parity for V1 (the
  blocker).

## Concrete next-session starting points, in order

1. **Run HF StaticCache against production on the same long-gen test.**
   Drop-in replacement in `tmp/long_gen_v1_test.py`:
   `cache = StaticCache(config=m.config, max_cache_len=199)`. Check
   whether divergence happens at the same step. This decides hypothesis
   1 vs 2 in one run.

2. **If StaticCache diverges too:** the issue is fundamental to
   fixed-buffer caching on Gemma 4 inside HF's attention path. The
   cache-level graph-capture lever is blocked on this model. Pivot to
   attention-kernel work (NVFP4 q/k/v — Path B from the 2026-04-18 plan).

3. **If StaticCache converges:** my capturable cache has a specific
   bug. Diff my cache's state against StaticCache's state at each
   update. Likely suspects worth auditing first:
     * Buffer allocation order vs. `mark_static_address` timing.
     * Non-contiguous `key_states` stride handling (Gemma 4's
       `key_states.transpose(1, 2)` before update).
     * `is_initialized=True` set eagerly vs. lazy pattern HF expects.

4. **Only after the cache works on long-gen parity**, extend with the
   quantize path and sliding-window semantics. The V2 code in `tmp/`
   is untested but compiles; use it as a starting point or a cautionary
   tale depending on whether Step 3 yields a clean fix.

5. **Ship plan, if/when cache is solved:**
   * Single commit 0.1.13 still feasible: strip compile from loader,
     add capturable cache to `src/aeo_quant/bridges/gemma4/`, wire
     opt-in fast-path, parity gauntlet (`examples/parity_check.py`,
     `examples/reasoning_check.py`, at least one multi-turn-16k turn).

## 2026-04-20 follow-up — StaticCache parity run

Ran the Step 1 experiment: HF's built-in `StaticCache` as a drop-in
for the V1 cache in `tmp/long_gen_v1_test.py`, same prompt/model/gen
length, same production reference.

```
V1 capturable:  first mismatch decode step 49, abs pos 90, 101/150 mismatches
StaticCache:    first mismatch decode step 49, abs pos 90, 101/150 mismatches
```

Identical failure. Hypothesis 2 resolved: **the divergence is not a
bug in the V1 cache — fixed-buffer / zero-padded-view K/V caching is
fundamentally non-equivalent to `Gemma4HybridTurboQuantCache` on long
context for this model.** Both caches depart from production at the
same decode step.

Implication: the cache-level lever (Step 3) is closed. Next move is
the attention-kernel path (lever 3 in `docs/plans/2026-04-18-nvfp4-next-round.md`):
profile a decode step, decide between `flash_attention_2` swap (cheap)
and NVFP4 q/k/v projections (expensive).

Standalone pure-cleanup still valid independent of all this: strip
`torch.compile(mode="reduce-overhead")` from `load_gemma4_fp8` and
`load_gemma4_nvfp4` in the loader. This session proved it crashes on
direct `model(...)` calls and contributes nothing via `generate()`.

Gotcha surfaced during the run: `Gemma4TextConfig.num_kv_shared_layers`
defaults to `0`, which triggers a `layer_types[:0] = []` truncation bug
in `transformers.StaticCache.__init__`. Any future experiment using
`StaticCache` on Gemma 4 needs to strip the attribute at both instance
and class scope (see the inline workaround in `tmp/long_gen_v1_test.py`)
or upstream a fix to transformers.

## 2026-04-20 second follow-up — Commit 2 attempt (NVFP4 attention projections)

After shipping Commit 1 (0.1.14 gather kernel, +15% decode), the next
lever per the plan was the 58% bf16 matmul bucket — attention Q/K/V/O
projections plus `lm_head`. We executed the plan fully:

* `LinearNVFP4` module + `convert_linears_to_nvfp4` post-load walker,
  mirroring the existing `Gemma4TextExpertsNVFP4` buffer layout (uint8
  packed, fp8 block scales, fp32 tensor scale).
* Loader hooked with `quantize_linears: bool = True` default.
* Walker replaced 116 modules (30 q_proj + 30 k_proj + 25 v_proj under
  the `attention_k_eq_v=True` config, 30 o_proj, 1 lm_head) in 2.1 s.
* Quantization ran without errors; per-module load-smoke verified type,
  shape, dtype, and single-forward finiteness.

`parity_check.py` triggered the fast-fail rule immediately:
**44/50 mismatches (88% divergence), max matching prefix 5 tokens**,
80% divergence vs FP8 baseline too. Worse than the prior FP8-on-206-Linears
attempt (46% divergence). Output was coherent Python-engineering prose —
model didn't break, it chose a different path at every decode step.

### Root cause

Not a bug. Confirmed by three independent measurements:

1. **Weight round-trip error on a single Linear:** 9.5% RMS on
   (4096, 2816), (2816, 4096), (262144, 2816) shapes with Gaussian
   weights at std 0.02 — same distribution as real attention weights.
2. **2D quantizer vs 3D quantizer on identical input:** byte-identical
   packed uint8, byte-identical fp8 block scales, byte-identical fp32
   tensor scale. Not a 2D-path bug.
3. **Per-matmul output error with activation quant:** 13% RMS, cosine
   0.991. Theory predicts `sqrt(K) × w_std × 2^-4 / output_std ≈
   sqrt(2816) × 0.02 × 0.0625 / 0.53 ≈ 12.5%` — matches measurement.

Compounded across 30 layers × 4 projections on Gemma 4 (262K vocab,
greedy decode), the top-1 vs top-2 logit gap is typically under 5% of
top-1 magnitude, so a 13% per-matmul noise floor flips argmax at high
rate. 88% divergence at token 5 is consistent with this math, not
with a code defect.

### Why MoE works and attention doesn't

MoE experts have the same 9.5% per-matmul weight error — but the NVFP4
parity baseline was *established with* NVFP4 MoE, so there is no
bf16-MoE vs NVFP4-MoE reference delta in parity_check. Attention was
unquantized at baseline time; introducing NVFP4 attention is a pure
A-vs-B perturbation against a bf16 reference, and FP4 numerics don't
survive that comparison.

### Why narrow-scope variants also fail the product mission

* **`lm_head`-only:** `lm_head` is tied to `embed_tokens`. Quantizing
  it breaks the tie and costs ~370 MB of *additional* resident memory
  for an NVFP4 buffer set, while `embed_tokens` stays bf16. That's a
  net memory increase for a single-digit % potential speedup, and
  aeo-quant's README-stated product is turbo-quant Gemma on **low
  memory** with reasonable speed. Wrong direction on memory.
  Additionally, parity on lm_head alone is not guaranteed — one-shot
  13% logit noise in a 262K vocab can still flip top-1 on tight-gap
  tokens.
* **Attention subset (e.g. sliding layers only, or skip `o_proj`):**
  the compounding problem doesn't cleanly partition. Any layer with
  NVFP4 projections introduces 13% noise into the residual stream,
  which perturbs the next layer's inputs. No clean floor.
* **FP8 for attention:** the prior FP8-on-206-Linears attempt was
  rejected for 0% speedup + 46% divergence. Per-row scale was the
  quality failure mode; per-block would help but we'd be back to the
  same fundamental precision question (FP8 E4M3 has ~3 bits of
  mantissa within a block, still coarse on attention).

### Verdict

**NVFP4 quantization of Gemma 4 attention projections is closed as a
decode-speedup lever on this model.** Same disposition as the
fixed-buffer capturable cache from earlier this week: the math is
fundamental, not a bug, and narrowing scope doesn't recover the
product mission. Commit 2's working-directory changes reverted; the
documentation record for the attempt lives here and in the memory
entry `project_nvfp4_attention_infeasible.md`.

### Remaining levers inside the transformers substrate

With lever 1 (CUDA graph via cache) closed, lever 3 (attention kernel)
a non-target at 0.29% CUDA time, lever 4 (alpha folding) shipped in
0.1.12, the gather lever shipped in 0.1.14, and attention-projection
NVFP4 closed here — the inventory is down to:

* **Lever 2 (per-shape prefill autotune):** TTFT lever, not decode.
* **Router NVFP4:** (128, 2816), 30× per step. Small matmul; at M=1
  the per-call activation quant overhead will likely dominate the
  bf16 GEMV. Low-probability win.
* **Option B from Commit 2 plan (class-swap with shared activation
  quant):** depended on attention projections being viable targets;
  moot now.

Current 18.74 tok/s is 2.77× the 6.77 baseline. Per
`feedback_nvfp4_5x_objective.md`, realistic ceiling inside
`transformers.generate()` is 20–30 tok/s. We're at the low end of that
envelope. The remaining gap to the north-star 5× target (52 tok/s)
lives in a substrate explicitly out of scope for aeo-quant.

## Scope-guard reminder

aeo-quant is Gemma + NVFP4 + TurboQuant. A cache that doesn't preserve
TurboQuant's compression under long context is not a valid product for
multi_turn_16k-style workloads. The Phase 1 short-context spike never
triggered promotion; the Phase 2 work needs bit-exact parity against
`Gemma4HybridTurboQuantCache` on runs that do. Any "it works on short
context" measurement is insufficient to ship.

## Also useful for fresh session

The loader currently ships `torch.compile(mode="reduce-overhead")` at
module scope on both `load_gemma4_fp8` and `load_gemma4_nvfp4`. Session
memory asserted this is a no-op for `generate()`; this session proved it
crashes on direct `model(...)` calls. It contributes no speedup in
either path. Stripping it is a pure-cleanup 1-line win, independent of
the graph-capture work.
