"""Output coherence checks for quantized model validation.

Merged from two independent implementations (context-scaling and prompt-matrix)
into a unified checker with configurable thresholds.

Stdlib only — no third-party dependencies.
"""

from __future__ import annotations


def check_output_coherent(
    decoded_text: str,
    new_ids: list[int],
    min_unique: int = 20,
    max_consecutive_run: int = 50,
    min_printable_ratio: float = 0.80,
) -> list[str]:
    """Check whether generated output looks coherent.

    Returns a list of failure reasons. An empty list means the output passed
    all checks.

    Args:
        decoded_text: The decoded text output from the model.
        new_ids: The list of generated token IDs.
        min_unique: Minimum number of unique token IDs required.
            Context-scaling default: 20, prompt-matrix default: 30.
        max_consecutive_run: Maximum allowed length of a single repeated
            token run before flagging as degenerate. Both sources use 50.
        min_printable_ratio: Minimum fraction of characters that must be
            printable ASCII (0x20-0x7E) or whitespace (\\n, \\t).
            Prompt-matrix default: 0.80, context-scaling: not checked.
    """
    failures: list[str] = []

    if not decoded_text.strip():
        failures.append("empty decoded text")

    unique = len(set(new_ids))
    if unique < min_unique:
        failures.append(f"only {unique} unique token ids (expected >= {min_unique})")

    # Check for degenerate repetition
    max_run = 1
    cur = 1
    for i in range(1, len(new_ids)):
        if new_ids[i] == new_ids[i - 1]:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 1
    if max_run > max_consecutive_run:
        failures.append(
            f"max consecutive-token run = {max_run} (> {max_consecutive_run})"
        )

    # Check printable ASCII ratio
    if decoded_text:
        printable = sum(
            1 for c in decoded_text if 32 <= ord(c) < 127 or c in "\n\t"
        )
        frac = printable / max(1, len(decoded_text))
        if frac < min_printable_ratio:
            failures.append(
                f"printable ASCII fraction = {frac:.1%} (< {min_printable_ratio:.0%})"
            )

    return failures
