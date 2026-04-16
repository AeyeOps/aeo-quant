# Model Optimization Learnings — Gemma 4 on GB10

Consolidated findings from the FP8 and NVFP4 optimization work.
Updated: 2026-04-16

## Bandwidth Wall

- GB10 has 273 GB/s LPDDR5X (unified memory, not HBM)
- Gemma 4 26B-A4B: ~27 GB of weight reads per token at FP8
- Theoretical ceiling: 273 / 27 = ~10 tok/s
- Measured: 10.08 tok/s (close to theoretical max)
- Only path to higher throughput: reduce bytes read per token

## FP8 _scaled_mm RowWise Contract

The Blackwell `_scaled_mm` kernel has strict requirements:
- Weight must be `float8_e4m3fn`, shape `(N, K)` row-major
- `.t()` gives `(K, N)` column-major (stride(0)==1), required for B operand
- `scale_b` must be `(1, N)` fp32 for RowWise (per-output-channel) scaling
- `scale_a` must be `(M, 1)` fp32 for per-row dynamic input quantization
- Pre-convert scales at load time to avoid per-call reshape overhead

## torch.compile Trade-offs

- `mode="reduce-overhead"`: +12% decode throughput, ~1s warmup
- Memory cost: ~39 GB peak during compilation (drops after)
- `dynamic=False` required for stable CUDA graph capture

## Non-MoE FP8: Rejected

Attempted FP8 quantization of all 206 nn.Linear modules:
- -840 MB VRAM savings
- 0% decode speedup (these modules aren't bandwidth-bound)
- 46% token divergence from FP8 per-channel being too coarse for
  attention q/k/v projections
- **Lesson:** per-channel max-abs is fine for large expert tensors
  (128 experts dilute error) but too coarse for smaller modules

## NVFP4 Block Scaling: Why It Works Where FP8 Didn't

NVFP4's per-16-element block scaling provides ~175x finer granularity
than FP8's per-output-channel scaling. This is why NVIDIA reports <1%
accuracy loss even on non-MoE layers:
- FP8 per-channel: one scale per row (e.g., 2816 elements share one scale)
- NVFP4 per-block: one scale per 16 elements (176 scales per row of 2816)

For our V1 (experts-only), this means:
- Expert weights that were already fine at FP8 are even better at NVFP4
- The additional FP4->FP8 double-quant error is modest (~17% mean relative)

## CPU:CUDA Ratio

When CPU:CUDA time ratio > 1.5x, the GPU is starved by kernel launch
overhead. Fix with torch.compile or CUDA graphs, not compute optimization.
This was the key insight that led to the +12% torch.compile win.
