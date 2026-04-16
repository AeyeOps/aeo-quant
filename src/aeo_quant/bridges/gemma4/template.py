"""Gemma 4 chat template helpers.

Output parsing lives in :mod:`aeo_quant.bridges.gemma4.parser` (``GEMMA4_PARSER``).
This module only builds prompt text for new turns.
"""
from __future__ import annotations


def incremental_turn_tokens(prompt_text: str, prev_eos_hit: bool = True) -> str:
    """Build the incremental token string for a new user turn.

    When reusing KV cache across turns, only the new user message + generation
    prompt needs to be tokenized and fed to the model. The format matches
    Gemma 4's chat template structure.
    """
    if prev_eos_hit:
        return f"\n<|turn>user\n{prompt_text}<turn|>\n<|turn>model\n"
    else:
        return f"<turn|>\n<|turn>user\n{prompt_text}<turn|>\n<|turn>model\n"
