"""Generic marker-based stream parser for model output.

Produces an ordered list of typed ``Segment`` objects from raw decoded text.
Models with channel-style markers (Gemma 4, Claude, etc.) configure a
``MarkerStreamParser`` instance with their marker set; exotic models subclass.

The parser guarantees complete coverage: every non-whitespace byte of input
is accounted for in some segment, including an ``"unknown"`` catch-all for
content inside an unclosed marker pair. This means no model output is ever
silently dropped.

Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from .types import Segment


@dataclass
class MarkerSpec:
    """One marker pair that delimits a typed segment in raw output.

    start: opening marker string (e.g., ``"<|channel>thought\\n"``).
    end:   single string OR tuple of acceptable end-variants (first match wins).
    type:  ``Segment.type`` to produce for content between the markers.
    metadata: fixed metadata dict attached to every segment of this type.
    """
    start: str
    end: Union[str, tuple[str, ...]]
    type: str
    metadata: dict = field(default_factory=dict)


class MarkerStreamParser:
    """Parses raw model output into an ordered ``Segment`` list via marker pairs.

    Algorithm (left-to-right scan):

    1. Find the earliest marker start at or after the current position.
    2. Emit a default-type segment for content before that start (if non-empty).
    3. Find a matching end marker (trying each variant).
    4. Emit a typed segment for content between the markers.
    5. If no end marker is found, emit the leftover as ``"unknown"``.
    6. Drop segments whose content is empty/whitespace-only.
    7. Strip trailing markers (e.g. ``<turn|>``) from the last segment.
    """

    def __init__(
        self,
        markers: list[MarkerSpec],
        *,
        default_type: str = "assistant",
        strip_trailing: Optional[list[str]] = None,
    ) -> None:
        self.markers = list(markers)
        self.default_type = default_type
        self.strip_trailing = list(strip_trailing) if strip_trailing else []

    def parse(self, raw_text: str) -> list[Segment]:
        segments: list[Segment] = []
        pos = 0
        n = len(raw_text)

        while pos < n:
            next_marker: Optional[MarkerSpec] = None
            next_start_pos = n
            for m in self.markers:
                idx = raw_text.find(m.start, pos)
                if idx != -1 and idx < next_start_pos:
                    next_start_pos = idx
                    next_marker = m

            if next_start_pos > pos:
                default_text = raw_text[pos:next_start_pos]
                if default_text.strip():
                    segments.append(
                        Segment(type=self.default_type, content=default_text, metadata={})
                    )

            if next_marker is None:
                break

            content_start = next_start_pos + len(next_marker.start)
            end_variants: tuple[str, ...] = (
                (next_marker.end,)
                if isinstance(next_marker.end, str)
                else tuple(next_marker.end)
            )
            content_end = -1
            end_marker_len = 0
            for variant in end_variants:
                idx = raw_text.find(variant, content_start)
                if idx != -1:
                    content_end = idx
                    end_marker_len = len(variant)
                    break

            if content_end == -1:
                unknown_text = raw_text[content_start:]
                if unknown_text.strip():
                    segments.append(
                        Segment(
                            type="unknown",
                            content=unknown_text,
                            metadata={
                                "expected_type": next_marker.type,
                                "reason": "missing_end_marker",
                            },
                        )
                    )
                pos = n
            else:
                content = raw_text[content_start:content_end]
                if content.strip():
                    segments.append(
                        Segment(
                            type=next_marker.type,
                            content=content,
                            metadata=dict(next_marker.metadata),
                        )
                    )
                pos = content_end + end_marker_len

        if segments and self.strip_trailing:
            last = segments[-1]
            cleaned = last.content
            for trail in self.strip_trailing:
                cleaned = cleaned.replace(trail, "")
            last.content = cleaned.rstrip()
            if not last.content.strip():
                segments.pop()

        return segments
