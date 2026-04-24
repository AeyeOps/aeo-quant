# 2026-04-23 — In-transformers NVFP4 optimization: envelope closed

Closing note for the throughput work that started 2026-04-15. This
session ran a refreshed `profile_generate.py` against the post-0.1.14
NVFP4 path, compared the bucket shape against the closed-lever inventory
in `2026-04-20-cuda-graph-handoff.md`, and confirms no new bucket worth
chasing inside `transformers.generate()`.

Nothing shipped this session. `src/` untouched.

## Refreshed trace

Run: `QUANT_FORMAT=nvfp4 PROFILE_TRACE=1 GEN_TOKENS=200 uv run python
examples/profile_generate.py`. Clean GPU (vLLM stopped for this run).
Output at `results/profiling/nvfp4-3bit-20260423-184253/`.

### Stage-1 timing

```
prompt tokens:    61
generated tokens: 200
tokenize:           0.3 ms
prefill:          723.9 ms   (5.5%)
decode:        12,460.2 ms   (94.5%)
total:         13,184.0 ms
decode tok/s:  16.05
```

Headline of 16.05 tok/s vs the 18.74 "quiet-box" figure is explained by
prompt: `profile_generate.py` uses the 61-token coding prompt with
200-token generation, producing a longer average attention context per
decode step than `parity_check.py`'s shorter run. Both numbers are
inside the 16–19 tok/s envelope for this model on this hardware.

### CUDA-time breakdown (200 decode tokens, `Self CUDA time total`: 9.314 s)

| Bucket | CUDA % | Detail |
|---|---|---|
| `lm_head` matmul, bf16 GEMV `[1,2816] × [2816,262144]` | **18.19%** (1.694 s / 200 calls @ 8.47 ms) | single biggest bucket |
| Attention projections (bf16 GEMVs, all shapes combined) | **~41%** | 9.34 + 7.59 + 7.46 + 7.36 + 4.62 + 2.83 + 2.12% across q/k/v/o/gate/up/down |
| `_nvfp4_matmul_kernel_3d_gather` (MoE experts) | **12.17%** (1.134 s / 11940 calls @ 94.9 μs) | irreducible matmul work |
| Activation-quant elementwise (`mul+pow+abs+amax+div+clamp`) | **~8%** (≈ 743 ms) | matches the prior writeup's 8.6% |
| KV-cache (`aten::cat`, `aten::copy_`) | **~9%** | maintenance |
| SDPA / attention kernel (flash + math) | **~1.5%** | non-target |
| `_nvfp4_matmul_kernel` (2D, router-shaped) | **1.12%** (104 ms / 3994 calls) | router already in NVFP4 where it fits |

Shape of the trace is the same as the 2026-04-20 writeup predicted.
No new bucket.

## Remaining "probe-first" levers — disposition

From `2026-04-20-cuda-graph-handoff.md § "Remaining levers inside the
transformers substrate"`:

1. **Router NVFP4.** Trace shows the 2D `_nvfp4_matmul_kernel` already
   accounts for 1.12% of CUDA (3994 calls). The remaining bf16 router
   work is not visible as a distinct bucket in the top-40. Upper bound
   on gain is well under 1% wall-clock. **Closed — not worth the
   commit.**

2. **Per-shape prefill autotune.** Prefill is 5.5% of total wall-clock.
   A 3× prefill speedup at constant decode nets ~0.55 tok/s. Does not
   move the decode headline. **Closed — TTFT-only, not the mission.**

3. **MoE activation-quant elementwise cleanup.** ~8% of CUDA time in
   the elementwise smear. Fusion best case per the handoff doc is 2–3%
   wall-clock → ~0.3–0.5 tok/s at the current rate. Inside envelope
   noise. **Closed — at the noise floor, doesn't justify the
   engineering spend.**

## Why the big buckets stay closed

The 18.19% `lm_head` bucket and the 41% attention-projection bucket
sum to roughly 60% of CUDA time — the pool where any real decode
speedup inside `transformers` would live. Both are closed by prior
sessions on grounds independent of this trace:

- **Attention projections NVFP4** (`project_nvfp4_attention_infeasible.md`,
  written 2026-04-20): 88% divergence at token 5. FP4 per-matmul noise
  floor ≈ 13% RMS compounds past argmax tolerance on Gemma 4's 262K
  vocab. Not a bug — the math is fundamental.

- **`lm_head`-only NVFP4:** tied to `embed_tokens`; quantizing breaks
  the tie and costs ~370 MB additional resident memory for a single-
  digit % potential speedup. Wrong direction for aeo-quant's
  low-memory product positioning. Parity on the 262K-vocab output not
  guaranteed even in isolation.

- **Fixed-buffer capturable cache** (`project_fixed_buffer_cache_blocked.md`):
  `StaticCache` and hand-written V1 both diverge identically at decode
  step 49. Fixed-buffer / zero-padded-view K/V is not numerically
  equivalent to `Gemma4HybridTurboQuantCache` on long context under
  HF's attention path.

The refreshed trace does not change the calculus on any of these.

## Final disposition

**The in-`transformers.generate()` NVFP4 throughput optimization is
complete at 16–19 tok/s depending on prompt.** Published release
`0.1.15` is the shipping point.

Per `feedback_nvfp4_5x_objective.md` (session memory): target is 52
tok/s (5× the 6.77 baseline); realistic ceiling inside `transformers`
is 20–30 tok/s. We are at the low end of that envelope, consistent
with Gemma 4's hybrid attention + 262K vocab + MoE decode bandwidth on
GB10 unified LPDDR5X.

The remaining gap to 52 tok/s lives in a substrate explicitly out of
scope per `feedback_aeo_quant_transformers_only.md`: vLLM, TRT-LLM,
Marlin, FlashInfer, llama.cpp. Any future 5× work would be a separate
project built on a different substrate, not a continuation of this
track.

## What to do with the results directory

The profile at `results/profiling/nvfp4-3bit-20260423-184253/` is the
canonical post-0.1.14 reference trace. Keep it. If a future session
makes changes it believes should speed up decode, re-running
`profile_generate.py` against this baseline produces a direct
before/after comparison on the same prompt.

## Future-session guardrail

If a fresh session opens this thread with "let me re-investigate the
attention NVFP4 question" or "let me try another capturable cache
variant," this record plus the two `project_nvfp4_*` memory entries
are the reason not to. The math closing those levers is fundamental to
Gemma 4 NVFP4 on this stack, not session-specific. Re-validating them
burns the GPU and re-reaches the same conclusion.
