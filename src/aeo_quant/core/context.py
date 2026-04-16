"""Context-window budget management.

Stdlib only — no third-party dependencies.
"""

from __future__ import annotations


def trim_history_to_budget(history: list[dict], budget_tokens: int) -> list[dict]:
    """Estimate token count as sum(len(m["content"]) // 4) and drop oldest
    user/assistant pairs (preserving index 0 = system prompt and the most
    recent pairs) until within budget."""

    def _estimate_tokens(msgs: list[dict]) -> int:
        return sum(len(m["content"]) // 4 for m in msgs)

    if _estimate_tokens(history) <= budget_tokens:
        return history

    # history[0] is the system prompt; pairs start at index 1
    # We drop from the oldest pairs (index 1,2 then 3,4 etc.)
    result = list(history)
    while _estimate_tokens(result) > budget_tokens and len(result) > 3:
        # Drop the oldest user/assistant pair after the system prompt
        # result[1] should be user, result[2] should be assistant
        del result[1:3]

    return result
