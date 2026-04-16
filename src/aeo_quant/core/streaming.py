"""HTTP streaming helpers for OpenAI-compatible chat completion APIs.

Stdlib only — uses urllib.request, no third-party dependencies.
"""

from __future__ import annotations

import json
import time
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def chat_completion_streaming(
    endpoint: str,
    model_id: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> tuple[str, dict, float]:
    """Send a streaming chat completion request and collect the response.

    Returns:
        (content_text, usage_dict, ttft_seconds)
    """
    body = json.dumps(
        {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    req = Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    t0 = time.monotonic()
    ttft: Optional[float] = None
    content_parts: list[str] = []
    usage: dict = {}

    with urlopen(req, timeout=timeout_s) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                evt = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "usage" in evt and evt["usage"]:
                usage = evt["usage"]
            choices = evt.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                if ttft is None:
                    ttft = time.monotonic() - t0
                content_parts.append(piece)

    if ttft is None:
        ttft = time.monotonic() - t0
    return "".join(content_parts), usage, ttft


def discover_model_id(models_endpoint: str) -> str:
    """Query the /v1/models endpoint and return the first model's ID."""
    req = Request(models_endpoint, headers={"Accept": "application/json"})
    with urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"no models served: {data}")
    return models[0]["id"]


def wait_for_health(health_endpoint: str, timeout_s: int) -> None:
    """Block until the health endpoint returns 200, or raise after timeout_s."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(Request(health_endpoint), timeout=5) as resp:
                if resp.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise RuntimeError(f"health check did not pass within {timeout_s}s")
