"""Topic-agnostic progressive follow-up prompts for multi-turn evaluation.

Each prompt deepens the conversation without assuming a specific domain,
making them suitable for any topic seed.

Stdlib only -- no external dependencies.
"""
from __future__ import annotations

PROGRESSIVE_FOLLOWUPS: list[str] = [
    "Deepen the most important insight from your previous answer. What's the "
    "failure mode that would bite someone who missed it?",
    "Challenge the two most load-bearing assumptions you just made and argue "
    "the counter-position for each.",
    "Translate the concept you just explained into production-quality code in "
    "whatever language or tooling is natural to the topic.",
    "Now apply this at 100x scale. What changes? What breaks first?",
    "Add observability: what would you instrument, and how would you know "
    "something is wrong before a user complains?",
    "Write the runbook entry an on-call engineer would need if this system "
    "failed at 3 AM on a holiday.",
    "Pick a less-common edge case you didn't address. Walk through how it "
    "interacts with what you've built so far.",
    "Refactor your previous response for clarity. Name what's redundant, "
    "what's missing, what's wrong.",
    "Add a testing strategy. What tests would verify the claims in your "
    "previous answer, and how would you structure them?",
    "Compare your approach to the obvious alternative. What are the "
    "trade-offs, and when would you pick the other one?",
    "Steelman the position of someone who disagrees with your previous "
    "answer, then respond to their strongest objection.",
    "Add error handling and failure modes. What can go wrong, and how does "
    "the system detect and respond?",
    "Operationalize this: what configuration, deployment, and ops concerns "
    "would a team inherit on day one?",
    "Write the one-paragraph executive summary. Your reader has 30 seconds.",
    "One thing you'd change about your previous answer if you had to defend "
    "it in a code review - what and why?",
]
