"""Self-contained HTML viewer generator for JSONL run transcripts.

Reads a JSONL transcript file (produced by TranscriptWriter) and generates
a single-file HTML viewer with embedded CSS and JS. Works offline.

Stdlib only — no third-party dependencies (marked.js is optionally inlined
from a vendor/ directory next to the caller, or a stub is used).
"""

from __future__ import annotations

import html as html_mod
import json
import os
import sys
from pathlib import Path

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
  :root {
    --bg: #0d1117; --bg-card: #161b22; --bg-hover: #1c2128;
    --border: #30363d; --text: #e6edf3; --text-dim: #8b949e;
    --accent: #58a6ff; --accent-dim: #1f6feb;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --user-bg: #1c2128; --assistant-bg: #0d1117;
    --code-bg: #1c2128;
    --s1: #58a6ff; --s2: #3fb950; --s3: #d29922; --s4: #bc8cff;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6; }

  /* Header */
  .header { padding: 20px 24px; border-bottom: 1px solid var(--border); }
  .header h1 { font-size: 20px; font-weight: 600; margin-bottom: 8px; }
  .header .meta { font-size: 13px; color: var(--text-dim); display: flex; gap: 20px; flex-wrap: wrap; }
  .header .meta span { display: inline-flex; align-items: center; gap: 4px; }

  /* System prompt */
  .system-prompt { margin: 12px 24px; padding: 12px 16px; background: var(--bg-card);
    border: 1px solid var(--border); border-radius: 6px; font-size: 13px; color: var(--text-dim); }
  .system-prompt summary { cursor: pointer; font-weight: 600; color: var(--text); }

  /* Tab bar */
  .tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border);
    padding: 0 24px; position: sticky; top: 0; background: var(--bg); z-index: 10; }
  .tab { padding: 10px 20px; font-size: 14px; font-weight: 500; cursor: pointer;
    border-bottom: 2px solid transparent; color: var(--text-dim); transition: all 0.15s;
    display: flex; align-items: center; gap: 8px; }
  .tab:hover { color: var(--text); background: var(--bg-hover); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab .dot { width: 8px; height: 8px; border-radius: 50%; }
  .tab[data-session="0"] .dot { background: var(--s1); }
  .tab[data-session="1"] .dot { background: var(--s2); }
  .tab[data-session="2"] .dot { background: var(--s3); }
  .tab[data-session="3"] .dot { background: var(--s4); }
  .tab[data-session="all"] .dot { background: var(--text-dim); }

  /* Session panel */
  .panel { display: none; padding: 16px 24px; }
  .panel.active { display: block; }

  /* Turn card */
  .turn { margin-bottom: 12px; border: 1px solid var(--border); border-radius: 8px;
    overflow: hidden; background: var(--bg-card); }
  .turn-header { padding: 12px 16px; cursor: pointer; display: flex;
    justify-content: space-between; align-items: center; transition: background 0.1s; }
  .turn-header:hover { background: var(--bg-hover); }
  .turn-header .left { display: flex; align-items: center; gap: 12px; }
  .turn-header .turn-num { font-weight: 600; font-size: 14px; min-width: 60px; }
  .turn-header .turn-status { font-size: 12px; padding: 2px 8px; border-radius: 10px;
    font-weight: 500; }
  .turn-header .turn-status.ok { background: rgba(63,185,80,0.15); color: var(--green); }
  .turn-header .turn-status.error { background: rgba(248,81,73,0.15); color: var(--red); }
  .turn-header .badges { display: flex; gap: 8px; font-size: 12px; color: var(--text-dim); }
  .turn-header .badge { background: var(--bg); padding: 2px 8px; border-radius: 4px;
    font-family: 'SF Mono', Menlo, monospace; }
  .turn-header .chevron { transition: transform 0.2s; color: var(--text-dim); font-size: 18px; }
  .turn.expanded .turn-header .chevron { transform: rotate(90deg); }

  /* Turn body */
  .turn-body { display: none; border-top: 1px solid var(--border); }
  .turn.expanded .turn-body { display: block; }

  .message { padding: 16px; }
  .message.user { background: var(--user-bg); border-bottom: 1px solid var(--border); }
  .message.assistant { background: var(--assistant-bg); }
  .message-label { font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 8px; }
  .message.user .message-label { color: var(--accent); }
  .message.assistant .message-label { color: var(--green); }
  .message-content { font-size: 14px; word-break: break-word;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; }
  .message-content > *:first-child { margin-top: 0; }
  .message-content > *:last-child { margin-bottom: 0; }

  /* Rendered markdown elements */
  .message-content h1, .message-content h2, .message-content h3,
  .message-content h4, .message-content h5, .message-content h6 {
    margin: 16px 0 8px 0; font-weight: 600; line-height: 1.3; }
  .message-content h1 { font-size: 1.4em; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
  .message-content h2 { font-size: 1.25em; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
  .message-content h3 { font-size: 1.1em; }
  .message-content h4 { font-size: 1em; }
  .message-content p { margin: 8px 0; }
  .message-content ul, .message-content ol { margin: 8px 0; padding-left: 24px; }
  .message-content li { margin: 4px 0; }
  .message-content li > ul, .message-content li > ol { margin: 2px 0; }

  /* Code */
  .message-content code { background: var(--code-bg); padding: 1px 5px; border-radius: 3px;
    font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 13px; }
  .message-content pre { background: var(--code-bg); padding: 12px 16px; border-radius: 6px;
    overflow-x: auto; margin: 8px 0; border: 1px solid var(--border); }
  .message-content pre code { background: none; padding: 0; font-size: 13px; display: block;
    white-space: pre; }

  /* Tables */
  .message-content table { border-collapse: collapse; margin: 8px 0; width: auto;
    font-size: 13px; }
  .message-content th, .message-content td { border: 1px solid var(--border);
    padding: 6px 12px; text-align: left; }
  .message-content th { background: var(--bg-hover); font-weight: 600; }
  .message-content tr:nth-child(even) { background: rgba(255,255,255,0.02); }

  /* Blockquotes */
  .message-content blockquote { border-left: 3px solid var(--accent-dim); margin: 8px 0;
    padding: 4px 16px; color: var(--text-dim); }
  .message-content blockquote p { margin: 4px 0; }

  /* Horizontal rules */
  .message-content hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }

  /* Links */
  .message-content a { color: var(--accent); text-decoration: none; }
  .message-content a:hover { text-decoration: underline; }

  /* Strong / em */
  .message-content strong { font-weight: 600; }
  .message-content em { font-style: italic; }

  /* Thinking segment (collapsible, dimmed italic) */
  .thinking-block { background: var(--bg); border-bottom: 1px solid var(--border);
    padding: 12px 16px; color: var(--text-dim); }
  .thinking-block[open] { padding-bottom: 16px; }
  .thinking-summary { cursor: pointer; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em; color: var(--s4); }
  .thinking-content { margin-top: 10px; font-style: italic; font-size: 13px;
    color: var(--text-dim); }

  /* Tool call / tool result */
  .message.tool-call { background: var(--bg); border-bottom: 1px solid var(--border);
    border-left: 3px solid var(--yellow); }
  .message.tool-result { background: var(--bg); border-bottom: 1px solid var(--border);
    border-left: 3px solid var(--green); }
  .message.tool-call .message-label { color: var(--yellow); }
  .message.tool-result .message-label { color: var(--green); }
  .tool-content { background: var(--code-bg); padding: 10px 12px; border-radius: 4px;
    font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 12px;
    white-space: pre-wrap; overflow-x: auto; border: 1px solid var(--border); }

  /* System message (dimmed box) */
  .message.system { background: var(--bg); border-bottom: 1px solid var(--border);
    border-left: 3px solid var(--text-dim); color: var(--text-dim); }
  .message.system .message-label { color: var(--text-dim); }

  /* Unknown segment catch-all */
  .unknown-block { background: var(--bg); border-bottom: 1px solid var(--border);
    border-left: 3px solid var(--red); padding: 12px 16px; }
  .unknown-summary { cursor: pointer; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em; color: var(--red); }

  /* Raw output ground-truth catch-all */
  .raw-block { background: var(--bg); border-top: 1px solid var(--border);
    padding: 12px 16px; }
  .raw-summary { cursor: pointer; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim); }
  .raw-content { margin-top: 10px; background: var(--code-bg); padding: 10px 12px;
    border-radius: 4px; font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-size: 12px; white-space: pre-wrap; overflow-x: auto;
    border: 1px solid var(--border); color: var(--text-dim); }

  /* Metrics bar */
  .metrics-bar { padding: 8px 16px; background: var(--bg); border-top: 1px solid var(--border);
    display: flex; gap: 16px; font-size: 12px; color: var(--text-dim);
    font-family: 'SF Mono', Menlo, monospace; }

  /* Timeline tab */
  .timeline-turn { display: flex; align-items: flex-start; gap: 12px; padding: 8px 0;
    border-bottom: 1px solid var(--border); cursor: pointer; }
  .timeline-turn:hover { background: var(--bg-hover); }
  .timeline-turn .tl-time { font-size: 12px; color: var(--text-dim);
    font-family: 'SF Mono', Menlo, monospace; min-width: 70px; }
  .timeline-turn .tl-session { font-size: 12px; font-weight: 600; min-width: 30px; }
  .timeline-turn .tl-desc { font-size: 13px; flex: 1; }
  .timeline-turn .tl-wall { font-size: 12px; color: var(--text-dim);
    font-family: 'SF Mono', Menlo, monospace; }

  /* Expand/collapse all */
  .toolbar { padding: 8px 24px; display: flex; gap: 8px; }
  .toolbar button { background: var(--bg-card); border: 1px solid var(--border);
    color: var(--text); padding: 4px 12px; border-radius: 4px; cursor: pointer;
    font-size: 12px; }
  .toolbar button:hover { background: var(--bg-hover); }
</style>
</head>
<body>

<div class="header">
  <h1>{{TITLE}}</h1>
  <div class="meta">
    <span>{{STARTED}}</span>
    <span>{{NUM_SESSIONS}} sessions</span>
    <span>{{TURNS_PER_SESSION}} turns/session</span>
    <span>max_tokens={{MAX_TOKENS}}</span>
    <span>temp={{TEMPERATURE}}</span>
  </div>
</div>

<details class="system-prompt">
  <summary>System Prompt</summary>
  <p style="margin-top:8px; white-space: pre-wrap;">{{SYSTEM_PROMPT}}</p>
</details>

<div class="tabs" id="tabs">
  <div class="tab active" data-session="all" onclick="switchTab('all')">
    <span class="dot"></span> Timeline
  </div>
  {{SESSION_TABS}}
</div>

<div class="toolbar" id="toolbar">
  <button onclick="expandAll()">Expand All</button>
  <button onclick="collapseAll()">Collapse All</button>
</div>

<div id="panels">
  <div class="panel active" data-session="all" id="panel-all">
    {{TIMELINE_HTML}}
  </div>
  {{SESSION_PANELS}}
</div>

{{MARKED_LIB}}
<script>
const DATA = {{JSON_DATA}};

marked.use({ gfm: true, breaks: false });

function switchTab(session) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`.tab[data-session="${session}"]`).classList.add('active');
  document.getElementById(`panel-${session}`).classList.add('active');
}

function toggleTurn(el) {
  el.closest('.turn').classList.toggle('expanded');
}

function expandAll() {
  const panel = document.querySelector('.panel.active');
  panel.querySelectorAll('.turn').forEach(t => t.classList.add('expanded'));
}

function collapseAll() {
  const panel = document.querySelector('.panel.active');
  panel.querySelectorAll('.turn').forEach(t => t.classList.remove('expanded'));
}

// Render all message contents with marked.js
document.querySelectorAll('.message-content').forEach(el => {
  el.innerHTML = marked.parse(el.textContent);
});
</script>
</body>
</html>"""


def _escape(s: str) -> str:
    """HTML-escape a string."""
    return html_mod.escape(s, quote=True)


def _format_wall(secs: float) -> str:
    if secs < 0:
        return "--"
    if secs < 60:
        return f"{secs:.1f}s"
    m = int(secs) // 60
    s = secs - m * 60
    return f"{m}m{s:.0f}s"


def _render_segment(seg: dict) -> str:
    """Render one typed segment dict into an HTML block.

    Dispatches on ``seg["type"]``. Unknown types fall through to a visible
    gray-bordered block so new content types never disappear silently.
    """
    t = seg.get("type", "unknown")
    content = _escape(seg.get("content", ""))
    if t == "thinking":
        return (
            '<details class="thinking-block">'
            '<summary class="thinking-summary">Thinking</summary>'
            f'<div class="message-content thinking-content">{content}</div>'
            '</details>'
        )
    if t == "assistant":
        return (
            '<div class="message assistant">'
            '<div class="message-label">Assistant</div>'
            f'<div class="message-content">{content}</div>'
            '</div>'
        )
    if t == "tool_call":
        return (
            '<div class="message tool-call">'
            '<div class="message-label">Tool Call</div>'
            f'<pre class="tool-content">{content}</pre>'
            '</div>'
        )
    if t == "tool_result":
        return (
            '<div class="message tool-result">'
            '<div class="message-label">Tool Result</div>'
            f'<pre class="tool-content">{content}</pre>'
            '</div>'
        )
    if t == "system":
        return (
            '<div class="message system">'
            '<div class="message-label">System</div>'
            f'<div class="message-content">{content}</div>'
            '</div>'
        )
    # Unknown — catch-all: visible, clearly flagged, default expanded
    label = _escape(t)
    return (
        '<details class="unknown-block" open>'
        f'<summary class="unknown-summary">Unknown segment (type={label})</summary>'
        f'<pre class="raw-content">{content}</pre>'
        '</details>'
    )


def _build_turn_card(turn: dict, idx: int) -> str:
    sid = turn["session_id"]
    tidx = turn["turn_index"]
    status = turn["status"]
    m = turn.get("metrics", {})
    wall = m.get("wall_s", 0)
    ttft = m.get("ttft_s", 0)
    pt = m.get("prompt_tokens", 0)
    ct = m.get("completion_tokens", 0)

    status_cls = "ok" if status == "ok" else "error"
    color_var = f"var(--s{sid + 1})"

    user_content = _escape(turn.get("user", ""))

    body_parts: list[str] = [
        '<div class="message user">'
        '<div class="message-label">User</div>'
        f'<div class="message-content">{user_content}</div>'
        '</div>'
    ]

    segments = turn.get("segments")
    if segments:
        for seg in segments:
            body_parts.append(_render_segment(seg))
    else:
        # Legacy transcripts have no segments array — render the top-level
        # assistant field as a single bubble so old runs still view correctly.
        assistant_content = _escape(turn.get("assistant", "") or f"[{status}]")
        body_parts.append(
            '<div class="message assistant">'
            '<div class="message-label">Assistant</div>'
            f'<div class="message-content">{assistant_content}</div>'
            '</div>'
        )

    raw_output = turn.get("raw_output")
    if raw_output:
        raw_escaped = _escape(raw_output)
        body_parts.append(
            '<details class="raw-block">'
            '<summary class="raw-summary">Raw output (ground truth)</summary>'
            f'<pre class="raw-content">{raw_escaped}</pre>'
            '</details>'
        )

    body_html = "\n    ".join(body_parts)

    return f"""<div class="turn" id="turn-{sid}-{tidx}">
  <div class="turn-header" onclick="toggleTurn(this)">
    <div class="left">
      <span class="turn-num" style="color:{color_var}">Turn {tidx + 1}</span>
      <span class="turn-status {status_cls}">{status}</span>
    </div>
    <div class="badges">
      <span class="badge">wall {_format_wall(wall)}</span>
      <span class="badge">ttft {_format_wall(ttft)}</span>
      <span class="badge">p={pt:,}</span>
      <span class="badge">c={ct:,}</span>
    </div>
    <span class="chevron">&#9654;</span>
  </div>
  <div class="turn-body">
    {body_html}
    <div class="metrics-bar">
      <span>wall={wall:.1f}s</span>
      <span>ttft={ttft:.3f}s</span>
      <span>prompt={pt:,}</span>
      <span>completion={ct:,}</span>
      <span>total={pt + ct:,}</span>
    </div>
  </div>
</div>"""


def _build_timeline_entry(turn: dict) -> str:
    sid = turn["session_id"]
    tidx = turn["turn_index"]
    m = turn.get("metrics", {})
    wall = m.get("wall_s", 0)
    ct = m.get("completion_tokens", 0)
    color_var = f"var(--s{sid + 1})"
    ts = turn.get("ts", "")
    # Show just the time portion
    time_part = ts.split("T")[1][:8] if "T" in ts else ts

    return f"""<div class="timeline-turn" onclick="switchTab('{sid}');
      setTimeout(()=>{{const e=document.getElementById('turn-{sid}-{tidx}');
      e.classList.add('expanded');e.scrollIntoView({{behavior:'smooth',block:'center'}})}},100)">
  <span class="tl-time">{time_part}</span>
  <span class="tl-session" style="color:{color_var}">S{sid + 1}</span>
  <span class="tl-desc">Turn {tidx + 1} &mdash; {ct:,} tokens</span>
  <span class="tl-wall">{_format_wall(wall)}</span>
</div>"""


def generate_html(
    jsonl_path: str | Path,
    output_path: str | Path | None = None,
    open_browser: bool = False,
    vendor_dir: Path | None = None,
) -> Path:
    """Generate a self-contained HTML viewer from a transcript JSONL file.

    Args:
        jsonl_path: Path to the transcript .jsonl file.
        output_path: Output HTML path. Defaults to same directory with .html extension.
        open_browser: If True, open the generated file in the default browser.
        vendor_dir: Directory containing marked.umd.js. If None, falls back to
            a stub that does basic HTML escaping (no markdown rendering).

    Returns:
        The Path to the generated HTML file.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found")

    out_path = Path(output_path) if output_path else jsonl_path.with_suffix(".html")

    # Parse JSONL
    metadata: dict | None = None
    turns: list[dict] = []
    # utf-8-sig transparently strips a BOM if present (common on Windows)
    with open(jsonl_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "metadata":
                metadata = obj
            elif obj.get("type") == "turn":
                turns.append(obj)

    if not metadata:
        metadata = {"system_prompt": "(not recorded)", "config": {}, "ts": ""}

    config = metadata.get("config", {})
    num_sessions = config.get("num_sessions", len(set(t["session_id"] for t in turns)))
    turns_per_session = config.get("turns_per_session", 9)
    max_tokens = config.get("max_tokens", "?")
    temperature = config.get("temperature", "?")

    # Group by session
    by_session: dict[int, list[dict]] = {}
    for t in turns:
        by_session.setdefault(t["session_id"], []).append(t)
    for sid in by_session:
        by_session[sid].sort(key=lambda t: t["turn_index"])

    # Derive title from filename
    stem = jsonl_path.stem.replace("_transcript", "").replace("_", " ").title()
    title = f"{stem} -- Conversation Viewer"
    started = metadata.get("ts", "")

    # Build session tabs
    session_tabs: list[str] = []
    for sid in sorted(by_session.keys()):
        topic = by_session[sid][0].get("session_topic", f"Session {sid + 1}")
        short_topic = topic[:40] + ("..." if len(topic) > 40 else "")
        ok_count = sum(1 for t in by_session[sid] if t["status"] == "ok")
        total = len(by_session[sid])
        session_tabs.append(
            f'<div class="tab" data-session="{sid}" onclick="switchTab(\'{sid}\')">'
            f'<span class="dot"></span> S{sid + 1}: {_escape(short_topic)} ({ok_count}/{total})</div>'
        )

    # Build session panels
    session_panels: list[str] = []
    for sid in sorted(by_session.keys()):
        cards = "\n".join(_build_turn_card(t, i) for i, t in enumerate(by_session[sid]))
        session_panels.append(
            f'<div class="panel" data-session="{sid}" id="panel-{sid}">\n{cards}\n</div>'
        )

    # Build timeline (all turns sorted by timestamp)
    sorted_turns = sorted(turns, key=lambda t: t.get("ts", ""))
    timeline_html = "\n".join(_build_timeline_entry(t) for t in sorted_turns)

    # Load marked.js for markdown rendering
    marked_lib = (
        "<script>window.marked={parse:function(t){"
        "return t.replace(/&/g,'&amp;').replace(/</g,'&lt;')"
        ".replace(/>/g,'&gt;').replace(/\\n/g,'<br>')"
        "},use:function(){}};</script>"
    )
    if vendor_dir is not None:
        marked_path = vendor_dir / "marked.umd.js"
        if marked_path.exists():
            marked_js = marked_path.read_text(encoding="utf-8")
            marked_lib = f"<script>{marked_js}<" + "/script>"

    # Assemble
    page = _HTML_TEMPLATE
    page = page.replace("{{TITLE}}", _escape(title))
    page = page.replace("{{STARTED}}", _escape(started))
    page = page.replace("{{NUM_SESSIONS}}", str(num_sessions))
    page = page.replace("{{TURNS_PER_SESSION}}", str(turns_per_session))
    page = page.replace("{{MAX_TOKENS}}", str(max_tokens))
    page = page.replace("{{TEMPERATURE}}", str(temperature))
    page = page.replace("{{SYSTEM_PROMPT}}", _escape(metadata.get("system_prompt", "")))
    page = page.replace("{{SESSION_TABS}}", "\n  ".join(session_tabs))
    page = page.replace("{{SESSION_PANELS}}", "\n".join(session_panels))
    page = page.replace("{{TIMELINE_HTML}}", timeline_html)
    page = page.replace("{{MARKED_LIB}}", marked_lib)
    # Escape </script> sequences inside the JSON blob so they don't break
    # the HTML parser
    json_blob = json.dumps(turns, ensure_ascii=False).replace("</", "<\\/")
    page = page.replace("{{JSON_DATA}}", json_blob)

    out_path.write_text(page, encoding="utf-8")

    # os.path.abspath preserves mapped drive letters on Windows
    abs_path = Path(os.path.abspath(out_path))
    print(f"Viewer saved to {abs_path}")
    print(f"  {len(turns)} turns across {num_sessions} sessions")

    if open_browser:
        import webbrowser

        posix_path = abs_path.as_posix()
        if posix_path.startswith("//"):
            file_url = f"file:{posix_path}"
        elif not posix_path.startswith("/"):
            file_url = f"file:///{posix_path}"
        else:
            file_url = f"file://{posix_path}"
        webbrowser.open(file_url)

    return abs_path


def main() -> int:
    """CLI entry point for standalone usage."""
    import argparse

    ap = argparse.ArgumentParser(description="Generate HTML viewer from transcript JSONL")
    ap.add_argument("jsonl", type=Path, help="Path to the transcript .jsonl file")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output HTML path (default: same dir, .html extension)")
    ap.add_argument("--open", action="store_true",
                    help="Open the generated HTML in the default browser")
    ap.add_argument("--vendor-dir", type=Path, default=None,
                    help="Directory containing marked.umd.js for markdown rendering")
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"Error: {args.jsonl} not found", file=sys.stderr)
        return 1

    if args.jsonl.suffix not in (".jsonl", ".json", ".ndjson"):
        print(
            f"Error: expected a .jsonl file, got {args.jsonl.suffix!r}\n"
            f"  Usage: python -m aeo_quant.core.viewer path/to/transcript.jsonl",
            file=sys.stderr,
        )
        return 1

    try:
        out = generate_html(
            args.jsonl,
            output_path=args.output,
            open_browser=args.open,
            vendor_dir=args.vendor_dir,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Print file URL
    posix_path = out.as_posix()
    if posix_path.startswith("//"):
        file_url = f"file:{posix_path}"
    elif not posix_path.startswith("/"):
        file_url = f"file:///{posix_path}"
    else:
        file_url = f"file://{posix_path}"
    print(f"  {file_url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
