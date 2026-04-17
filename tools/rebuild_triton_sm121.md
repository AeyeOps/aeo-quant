# Rebuild Triton from source with the sm_121 patch (Path A)

Use when `TRITON_OVERRIDE_ARCH=sm120` doesn't produce correct output
(Path A.5 fails).  Expected build time: 30–45 min on GB10.

## When to do this

Only if `examples/test_nvfp4_kernel.py` shows correctness failures on
every shape with `TRITON_OVERRIDE_ARCH=sm120`.  If correctness passes
but performance is off, that's a tuning issue, not a build issue.

## The patch

Target file: `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`.

```diff
     matchAndRewrite(triton::DotScaledOp dotOp,
                     mlir::PatternRewriter &rewriter) const override {
-      if (computeCapability != 120)
+      if (computeCapability != 120 && computeCapability != 121)
         return failure();
```

And (only if your workload hits FP8-operand decomposition):

```diff
 static bool mmav2SupportsFp8Operands(int computeCapability) {
-  return computeCapability == 89 || computeCapability == 120;
+  return computeCapability == 89 || computeCapability == 120 ||
+         computeCapability == 121;
 }
```

## Before rebuilding — grep for hidden sm_120 asserts

The helper `getSM120DotScaledScaleLayout` is called unconditionally.
If it has a `cc == 120` assert inside, the primary patch alone isn't
enough.

```bash
cd /opt/dev/aeo/third_party/triton  # once you clone it
grep -rn "computeCapability == 120\|cc == 120\|SM120" include/ lib/ third_party/nvidia/ \
    | grep -vE '(test|\.pyc|\.md)'
```

Relax each site that's truly capability-gated (not name-only).

## Build

Triton 3.6.0's build dir layout (verify on your clone):

```bash
git clone -b v3.6.0 https://github.com/triton-lang/triton
cd triton
# apply the diff above via git apply or manual edit
pip install -e python --verbose
```

If the build errors on missing deps, install `cmake` ≥ 3.22, `ninja`,
`libstdc++-dev`, and `libssl-dev`.  Triton ships its own LLVM build.

## Post-build verification

```bash
uv run python -c "
import triton
print('triton:', triton.__version__, 'from', triton.__file__)
"
# Should point to your local clone, not the PyPI wheel.
```

Run the kernel test without the override:

```bash
uv run python examples/test_nvfp4_kernel.py
# The ScaledBlockedToMMA pattern should now match on sm_121 natively.
```

And compare SASS:

```bash
tools/dump_triton_sass.sh --name _nvfp4_matmul --limit 200 \
    | grep -E 'HMMA|MXF4|NVF4|kind::'
```

Expect `HMMA` with an MXF4 mnemonic, not `FMA` fallbacks.

## If the patch itself doesn't compile

Triton's MLIR pattern registration has moved between versions.  If
`AccelerateMatmul.cpp` looks different from what's in
`kb/nvfp4-blackwell-research.md`, check the current file against
`docs/references/triton_tutorial_10_v36.py` neighbors and adjust.
Line numbers drift; semantics are stable across patch releases.
