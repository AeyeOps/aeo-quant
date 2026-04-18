# examples/ — SDK surface, not a scratchpad

This directory **ships as part of the `aeo-quant` product**. Every file here is read by users who are learning how to use the SDK. Treat it the same way you'd treat the public API: changes are user-visible, additions are a commitment.

## What belongs in `examples/`

A file belongs here if a user copying it into their own project would learn how to use `aeo-quant` against a real checkpoint. Concretely:

- **Format-agnostic.** Reads `QUANT_FORMAT` via `quant_env()` — works on fp8 and nvfp4 without edits. A script that hard-codes a format (`if fmt != "nvfp4": return 2`) does not belong here.
- **Uses the harness.** Dispatches work via `get_or_start_harness()` + `client.run_workload(...)`. In-process loads (`load_gemma4(...)`) are for the loader itself; examples should show users the efficient path.
- **Single clear purpose.** The Usage docstring reads like product documentation. A reader should know what it demonstrates in one sentence.
- **Listed in `examples/README.md`** with a one-line pitch, a runnable command, and an expected runtime.

Current canonical set: `parity_check.py`, `parity_long_check.py`, `quality_check.py`, `reasoning_check.py`, `multi_turn_16k.py`, `multi_turn_32k.py`, `profile_generate.py`, `build_checkpoint.py`, `build_checkpoint_nvfp4.py`.

## What does NOT belong in `examples/`

Do not land files here that fall into any of these categories. They have homes elsewhere:

| Kind of script | Goes in | Examples |
|---|---|---|
| Bring-up smoke / "did we wire it up?" one-offs | `tmp/` (gitignored) | Anything named `smoke_*`, `*_probe`, `probe_*` |
| Kernel unit tests with numeric tolerance gates | `tests/` | `test_nvfp4_kernel.py`, `test_*_bridge.py` |
| Tile-size / autotune sweeps | `tmp/` (gitignored) | `tune_*` |
| Kernel-level profilers (not whole-model) | `tmp/` (gitignored) | kernel-level profilers vs `profile_generate.py` (SDK-level, stays) |
| Debug probes for a specific bug or feature flag | `tmp/` (gitignored) | `probe_logits_at_divergence.py`, `probe_nvfp4_aot.py` |
| Format-specific demos that duplicate format-agnostic examples | nowhere — delete or fold into the agnostic version | The old `smoke_nvfp4.py` duplicated `quality_check.py`'s function |

See `tmp/CLAUDE.md` for the scratch-space policy — those files are explicitly not maintained and not referenced from documentation.

If in doubt, ask: **"would I be proud to point a new user at this file as their introduction to the SDK?"** If the answer is no, it doesn't go here.

## Editing rules for agents

- Adding a new example is a product decision. It needs a README entry in the same change, or the change is incomplete.
- Renaming or removing an example is a breaking change for users who script against these paths. Do it deliberately, in a single clean commit, with the README updated.
- If you need a throwaway script to validate your work, put it in `tmp/` from the start (it's gitignored and explicitly not maintained). Don't land it in `examples/` with intent to move it later — cleanup rarely happens.
- The test scripts in `tests/` are for the kernel and bridge layers. End-to-end generation regressions go through `parity_check.py` / `parity_long_check.py` / `reasoning_check.py`, which double as both SDK examples and regression canaries because they diff against pinned baselines.

## The `QUANT_FORMAT` contract

Every example here must support:

```bash
uv run python examples/<name>.py                          # defaults to fp8
QUANT_FORMAT=nvfp4 uv run python examples/<name>.py       # nvfp4 path
```

`quant_env()` handles format resolution, checkpoint dispatch, and (for nvfp4) auto-applies the `TRITON_OVERRIDE_ARCH=sm120` Triton quirk via `ensure_nvfp4_triton_arch()`. Do not replicate that logic in example scripts — import `quant_env()` once at module load and let it do its job.
