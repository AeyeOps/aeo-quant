"""Live terminal streamer for Gemma 4 generation with thinking/answer phases.

Hooks into transformers' streamer protocol so tokens render to the terminal
as they are generated, instead of waiting 10+ minutes in silence.

Usage::

    from aeo_quant.bridges.gemma4.streamer import LiveStreamer

    streamer = LiveStreamer(tokenizer)
    model.generate(..., streamer=streamer)
"""
from __future__ import annotations

import sys
import time

from transformers import TextStreamer

from aeo_quant.bridges.gemma4.parser import GEMMA4_PARSER


class LiveStreamer(TextStreamer):
    """Stream generation output to stderr with thinking/answer phase awareness.

    During thinking: updates a single status line with token count, tok/s,
    and elapsed time (overwritten in-place via carriage return).

    During answer: streams the decoded text verbatim to the terminal.

    Set ``verbose_think=True`` to also stream thinking text (dimmed).

    Marker strings come from ``GEMMA4_PARSER`` so phase detection here and
    structured parsing in the example scripts share a single source of truth.
    """

    # ANSI
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    RESET = "\033[0m"
    CLEAR_LINE = "\r\033[K"

    def __init__(self, tokenizer, *, verbose_think: bool = False, **kwargs):
        kwargs.setdefault("skip_prompt", True)
        super().__init__(tokenizer, **kwargs)
        self.verbose_think = verbose_think

        thinking_spec = next(m for m in GEMMA4_PARSER.markers if m.type == "thinking")
        self._think_start: str = thinking_spec.start
        self._think_end: tuple[str, ...] = (
            (thinking_spec.end,)
            if isinstance(thinking_spec.end, str)
            else tuple(thinking_spec.end)
        )

        self._phase: str = "prefix"  # prefix → thinking → answer
        self._buf: str = ""
        self._n_tokens: int = 0
        self._think_end_at: int = 0  # _n_tokens when thinking ended
        self._t0: float = time.monotonic()
        self._last_status_t: float = 0.0

        # TTFT: set on prompt put(), first-token put() respectively
        self._t_prompt: float | None = None
        self._t_first_token: float | None = None

    # -- token counting + TTFT (called once per generated token) ---------------

    def put(self, value):
        was_prompt = self.skip_prompt and self.next_tokens_are_prompt
        if was_prompt:
            self._t_prompt = time.monotonic()
        super().put(value)
        if not was_prompt:
            if self._t_first_token is None:
                self._t_first_token = time.monotonic()
            self._n_tokens += 1

    @property
    def ttft(self) -> float | None:
        """Time to first token (seconds): prefill latency.

        Measured from prompt ingestion to first generated token.
        Returns None if generation hasn't produced a token yet.
        """
        if self._t_prompt is not None and self._t_first_token is not None:
            return self._t_first_token - self._t_prompt
        return None

    # -- display ---------------------------------------------------------------

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self._buf += text

        # --- prefix: waiting for thinking to start ----------------------------
        if self._phase == "prefix":
            pos = self._buf.find(self._think_start)
            if pos != -1:
                self._phase = "thinking"
                self._buf = self._buf[pos + len(self._think_start) :]
                self._w(
                    f"\n{self.DIM}{self.CYAN}[thinking]{self.RESET}{self.DIM} "
                )
            elif stream_end:
                # no thinking phase at all — dump as answer
                self._w(f"\n{self.BOLD}[answer]{self.RESET} {self._buf}")
                self._buf = ""
                self._print_summary()
                return

        # --- thinking ---------------------------------------------------------
        if self._phase == "thinking":
            for marker in self._think_end:
                pos = self._buf.find(marker)
                if pos != -1:
                    remainder = self._buf[pos + len(marker) :]
                    self._buf = ""
                    self._phase = "answer"
                    self._think_end_at = self._n_tokens

                    elapsed = time.monotonic() - self._t0
                    rate = self._n_tokens / elapsed if elapsed > 0 else 0
                    self._w(
                        f"{self.CLEAR_LINE}"
                        f"{self.DIM}{self.CYAN}[thinking]{self.RESET}"
                        f"{self.DIM} {self._n_tokens} tokens in {elapsed:.0f}s "
                        f"({rate:.1f} tok/s){self.RESET}\n"
                        f"{self.BOLD}[answer]{self.RESET} "
                    )
                    if remainder.strip():
                        self._w(remainder)
                    break
            else:
                # still thinking
                if self.verbose_think:
                    self._w(text)
                else:
                    now = time.monotonic()
                    if now - self._last_status_t >= 1.0:
                        elapsed = now - self._t0
                        rate = self._n_tokens / elapsed if elapsed > 0 else 0
                        self._w(
                            f"{self.CLEAR_LINE}"
                            f"{self.DIM}{self.CYAN}[thinking]{self.RESET}"
                            f"{self.DIM} {self._n_tokens} tokens | "
                            f"{rate:.1f} tok/s | "
                            f"{elapsed:.0f}s{self.RESET}"
                        )
                        self._last_status_t = now
                # trim buffer — only need tail for cross-call marker detection
                max_m = max(len(m) for m in self._think_end)
                if len(self._buf) > max_m * 3:
                    self._buf = self._buf[-max_m * 2 :]

        # --- answer -----------------------------------------------------------
        elif self._phase == "answer":
            self._w(text)

        if stream_end:
            self._print_summary()

    # -- helpers ---------------------------------------------------------------

    def _print_summary(self) -> None:
        elapsed = time.monotonic() - self._t0
        rate = self._n_tokens / elapsed if elapsed > 0 else 0
        think_n = self._think_end_at
        answer_n = self._n_tokens - think_n
        self._w(
            f"\n{self.DIM}[stream] "
            f"{self._n_tokens} tokens "
            f"(think={think_n}, answer={answer_n}) "
            f"in {elapsed:.0f}s ({rate:.1f} tok/s){self.RESET}\n"
        )

    def _w(self, s: str) -> None:
        sys.stderr.write(s)
        sys.stderr.flush()
