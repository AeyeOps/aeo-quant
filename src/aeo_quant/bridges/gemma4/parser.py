"""Gemma 4 stream parser — binds Gemma 4 channel markers to the core parser.

This is the single source of truth for Gemma 4 marker strings.
Both the example benchmark scripts and :class:`LiveStreamer` import from
here so markers are defined once.
"""

from __future__ import annotations

from aeo_quant.core.segments import MarkerSpec, MarkerStreamParser

GEMMA4_PARSER = MarkerStreamParser(
    markers=[
        MarkerSpec(
            start="<|channel>thought\n",
            end=("\n<channel|>", "<channel|>"),
            type="thinking",
            metadata={"channel": "thought"},
        ),
        # Future channels (tool_call, tool_result, etc.) go here.
    ],
    default_type="assistant",
    strip_trailing=["<turn|>"],
)
