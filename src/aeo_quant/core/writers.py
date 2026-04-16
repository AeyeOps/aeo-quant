"""Thread-safe CSV and JSONL transcript writers.

Stdlib only — no third-party dependencies.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path
from typing import Optional

from .types import Segment, iso


def _extract_assistant_text(segments: Optional[list[Segment]]) -> Optional[str]:
    """Concatenate content from all assistant-typed segments.

    Used to populate the legacy top-level ``assistant`` field on turn records
    so that consumers who only read that field still see the answer text.
    """
    if not segments:
        return None
    parts = [s.content for s in segments if s.type == "assistant"]
    if not parts:
        return None
    return "".join(parts)


class JSONLWriter:
    """Simple JSONL append writer. Opens in append mode, flushes on every write.

    For cases where you just need to append JSON records to a file without
    the structured metadata of TranscriptWriter. Not thread-safe — use
    TranscriptWriter if you need concurrent writes.
    """

    def __init__(self, path: Path):
        self.path = path

    def write(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class CSVWriter:
    """Thread-safe CSV writer that flushes after every row."""

    def __init__(self, path: Path, header: list[str]):
        self.path = path
        self.lock = threading.Lock()
        self.fh = path.open("w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=header)
        self.writer.writeheader()
        self.fh.flush()

    def write(self, row: dict) -> None:
        with self.lock:
            self.writer.writerow(row)
            self.fh.flush()

    def close(self) -> None:
        with self.lock:
            try:
                self.fh.flush()
                self.fh.close()
            except Exception:
                pass


class TranscriptWriter:
    """Thread-safe JSONL transcript writer. One JSON object per line, flushed on every write.

    The first line is a metadata record with the system prompt (written once).
    Each subsequent line is a turn record: session_id, turn_index, user message,
    assistant response, status, and per-turn metrics.

    IO errors are caught and warned about -- they never propagate into the worker.
    CSV is the authoritative artifact; transcript is the conversation record.
    """

    def __init__(self, path: Path, system_prompt: str, config: Optional[dict] = None):
        self.path = path
        self.lock = threading.Lock()
        self._fh = path.open("w")
        # Write system prompt once as the first record
        metadata: dict = {
            "type": "metadata",
            "ts": iso(time.time()),
            "system_prompt": system_prompt,
        }
        if config is not None:
            metadata["config"] = config
        self._write_record(metadata)

    def _write_record(self, record: dict) -> None:
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception as e:
            print(f"!! transcript write error (non-fatal): {e}", flush=True)

    def write_turn(
        self,
        session_id: int,
        session_topic: str,
        turn_index: int,
        user_msg: str,
        assistant_msg: str = "",
        status: str = "ok",
        wall: float = 0.0,
        ttft: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        segments: Optional[list[Segment]] = None,
        raw_output: Optional[str] = None,
        raw_usage: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> None:
        with self.lock:
            # Synthesize a single assistant segment if caller passed only assistant_msg.
            if segments is None and assistant_msg and status == "ok":
                segments = [Segment(type="assistant", content=assistant_msg)]

            persist_segments = segments if (status == "ok" and segments) else None
            persist_raw_output = raw_output if status == "ok" else None

            assistant_text = (
                _extract_assistant_text(persist_segments)
                if persist_segments
                else (assistant_msg if status == "ok" else None)
            )

            record: dict = {
                "type": "turn",
                "ts": iso(time.time()),
                "session_id": session_id,
                "session_topic": session_topic,
                "turn_index": turn_index,
                "user": user_msg,
                "segments": (
                    [s.to_dict() for s in persist_segments] if persist_segments else None
                ),
                "raw_output": persist_raw_output,
                "assistant": assistant_text,
                "status": status,
                "metrics": {
                    "wall_s": round(wall, 3),
                    "ttft_s": round(ttft, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            }
            # Pass through the full usage dict (may contain fields
            # beyond prompt_tokens/completion_tokens that we don't parse yet)
            if raw_usage:
                record["raw_usage"] = raw_usage
            # Catch-all for anything extra that callers want to attach
            if extra:
                record["extra"] = extra
            self._write_record(record)

    def close(self) -> None:
        with self.lock:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
