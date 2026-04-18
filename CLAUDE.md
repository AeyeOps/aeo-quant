# aeo-quant — architecture principles and repo map

## The one rule

**`src/` is the SDK. It is the product. It is the only thing that exists from an architectural standpoint.**

Everything else in this repo exists to support, demonstrate, document, or test `src/` — and references go **one way only**: from those supporting directories *to* `src/`. Never the reverse.

Concretely:

- `src/**/*.py` imports from `examples/`, `tests/`, `tools/`, or anywhere outside `src/` → forbidden
- `src/**/*.py` docstrings or comments reference `examples/foo.py`, `kb/something.md`, `docs/plan.md`, or any non-`src/` path → forbidden
- `examples/foo.py` imports from `aeo_quant.*` → yes, that's the whole point
- `docs/foo.md` cites `src/aeo_quant/gpu/nvfp4_matmul.py` → yes, outside → inside is fine
- `tmp/` does whatever it wants, but nothing else references it (see `tmp/CLAUDE.md`)

**Why the docstring/comment rule is strict, not advisory:** anything outside `src/` may not exist in the future. `examples/` gets pruned as the canonical set evolves. `kb/` is research, dated and sometimes wrong. `docs/plans/*.md` are historical snapshots. `tmp/` is explicitly ephemeral. When `src/` points at them, the SDK takes an implicit dependency on artifacts it does not own — and the moment any of those get renamed, moved, deleted, or retired, the comment lies. Core code must stand alone: any context a reader of `src/` needs should be *in the source* or nowhere. If the explanation is long enough that it feels like it "belongs in a doc," that's a signal the docstring is trying to do too much — shorten the docstring, not outsource it.

## Directory map

| Dir | Purpose | Governance |
|---|---|---|
| `src/aeo_quant/` | The SDK. The only part that matters architecturally. | this file |
| `examples/` | Ships with the SDK. Demonstrates canonical use. | `examples/CLAUDE.md` |
| `tests/` | Regression coverage. No harness by design — see below. | — |
| `tmp/` | Gitignored scratch space. Not maintained, not referenced. | `tmp/CLAUDE.md` |
| `docs/` | Reference material, dated plans, writeups. Historical snapshots stay historical — don't retrofit. | — |
| `kb/` | Research notes and deep dives. Often date-stamped. | — |
| `tools/` | Repo-admin scripts (rebuilds, env setup). | — |

## Operating principles

### Shrink the tracked surface before polishing references
When a small code change cascades into updating 10+ files, the reference graph is the bug — not the incompleteness of the retrofit. Cut scope (move to `tmp/`, delete, or stop tracking) before you coordinate more updates. "Reference integrity" is not an end in itself; if maintaining it is eating the budget for the actual work, something in the graph should not exist.

### Iteration artifacts are not tests, examples, or SDK surface
A script named `test_foo.py` written mid-debug is not a test suite. A demo hardcoded to one format is not an SDK example. Both belong in `tmp/` or get deleted. The `test_` prefix and the `examples/` directory are commitments — don't make them accidentally.

### No test harness ambitions until the design settles
The software is still evolving fast. Tests written against a moving target become the next retrofit. When the functional surface has stabilized, the user will call for coverage — don't initiate it pre-emptively. Probe scripts go to `tmp/`, not `tests/`.

### Historical documents stay historical
Dated writeups (`docs/2026-04-17-*.md`, dated plans, `CHANGELOG.md` entries) record what was true at the time they were written. Do not retrofit their file references when you move code around. If they point at a path that no longer exists, that's a correct historical snapshot — rewriting it would falsify the record.

### The `QUANT_FORMAT` contract
Every SDK example and workload path honors `QUANT_FORMAT=fp8|nvfp4`. Format resolution, checkpoint dispatch, and the nvfp4 Triton-arch quirk (`TRITON_OVERRIDE_ARCH=sm120`) all live in `aeo_quant.core.config.quant_env()`. Don't replicate that logic elsewhere, don't hardcode a format in an example.
