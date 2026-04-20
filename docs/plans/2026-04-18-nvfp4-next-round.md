# 2026-04-18 — NVFP4 next round toward 5× target

Handoff doc. Written at the end of a session that spent most of its budget on governance (architectural rule, examples/ reorg, `tmp/` policy) rather than kernel work. The kernel ideas below are what was queued up but not yet started.

## Where we are

- **Baseline:** 6.77 tok/s decode (pre-optimization)
- **Today:** 12.5–13.3 tok/s decode — 3D fused-experts kernel + `torch.compile(mode="reduce-overhead")` — **~1.9× cumulative**
- **North-star target:** 52 tok/s (5× the 6.77 baseline)
- **Realistic ceiling inside `transformers.generate()`:** 20–30 tok/s — the rest lives in a substrate we've ruled out of scope. See `feedback_nvfp4_5x_objective.md`.
- **Remaining headroom inside our substrate:** roughly 1.5–2.2× more from where we are.

## Carryover: validate the last shipped fix

The harness daemon on the box at end-of-session was running pre-`e7203a9` bytecode — it crashes with `RuntimeError: PassManager::run failed` on the first NVFP4 workload because its in-memory module predates `ensure_nvfp4_triton_arch()`. Fix is operational, not code:

```
aeo-harness stop
QUANT_FORMAT=nvfp4 aeo-harness start
uv run examples/parity_check.py     # confirms nvfp4 path is live
uv run examples/reasoning_check.py   # any additional cheap validation
```

Then swap to fp8 and re-run the cheap gauntlet for parity. Do this *before* kernel work — regressions are cheaper to catch now.

## Candidate levers, ranked by expected leverage

### 1. CUDA graph capture for the decode step  (highest leverage)

After the 3D kernel, decode is ~75 ms/token and the remaining CPU–CUDA gap in the profile is pure Python/kernel-launch overhead. `torch.compile(mode="reduce-overhead")` gets partial capture, but wrapping the full post-prefill decode forward in an explicit `torch.cuda.graph()` should close most of that gap.

- **Prerequisite already shipped:** removed `os.environ.get()` from inside the compiled forward (see `feedback_torch_compile_env_var.md`) — graphs require side-effect-free capture. The previous `cuda_graph_probe.py` bring-up attempt predates that fix; it's in `tmp/` if you want to see the earlier attempt, but expect to rewrite it from scratch.
- **Risk:** cache state (KV cache, rotary position) mutating across capture boundary. `Gemma4HybridTurboQuantCache` with pre-populated `layers` should be graph-compatible but hasn't been tested under capture.
- **Expected win:** +2–4 tok/s (best guess from the CPU–CUDA gap in the last profile).

### 2. Per-shape autotune for the prefill 2D kernel

Current tile selection in `_nvfp4_matmul_kernel` is static: `BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, NUM_STAGES=2` with a small-M override. Decode (M=1, 8) works well; prefill (M=2816 for 16K-context long prompts) was never tuned individually.

- We now have enough workload data to tune per-shape without overfitting: `parity_long_check.py` + `multi_turn_16k/32k.py` exercise the real prefill geometries.
- Small decode speedup, larger prefill speedup. Main value: prefill time is user-visible as TTFT.
- Previous `tune_nvfp4_kernel.py` sweep scaffold is in `tmp/` — probably faster to rewrite than to rehabilitate.

### 3. Attention bottleneck audit  (check before committing to 4)

The 3D MoE kernel moved the bottleneck. With experts at ~1.9×, attention may now be the tall pole — it's still bf16, not NVFP4, and uses transformers' default SDPA path. Before doing more MoE work or committing to lever 4, confirm where time is actually going:

- Profile one decode step with `torch.profiler`.
- If attention CUDA time is >30% of the step, attention is the next target, not MoE.
- Cheap mitigation if it is: `attn_implementation="flash_attention_2"` at load time (no kernel work).
- Expensive mitigation: NVFP4 q/k/v projections (substantial — defer).

### 4. On-device alpha folding  (micro-win, good filler)

`alpha` in `nvfp4_linear_prequantized` is computed as `(a_tensor_scale.float() * w_tensor_scale.float()).item()` — host sync per matmul. Making this device-side removes one sync. Low magnitude (<0.5 tok/s guess), low risk. Good background task while a bigger change compiles.

## Suggested order for the next session

1. Daemon restart + cheap gauntlet on both formats (above).
2. Profile one decode step (write a fresh profiler in `tmp/` — don't resurrect the old one).
3. Decide between lever 1 (if the gap is CPU-bound) or lever 3 (if attention is now hot).
4. Land that one lever cleanly, bump version, update CHANGELOG. Then re-evaluate.

Don't stack multiple levers in one change. Every win needs a parity-gated commit so regressions are bisectable.

## Scope guardrails (inherited from CLAUDE.md + memories)

- Transformers is the fixed substrate. No vLLM, TRT-LLM, Marlin, llama.cpp, FlashInfer-backend discussions. We target inside `transformers.generate()`.
- Probes, profilers, tuning sweeps → `tmp/` from the start. Never `examples/` or `tests/`.
- No test harness ambitions. The surface is still moving.
- Ask before heavy GPU loads — GB10 is shared.
- Verify output content, not just token counts — parity_check is the canary.
- Batch expert conversions in 16s if touching that path.
