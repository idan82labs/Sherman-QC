"""Stdout protocol shared by every supervised step (E1 compile, E3 inspect, C06 render).

One record per line, tab separated: ``PROGRESS\\t{json}``, ``RESULT\\t{json}``
or ``ERROR\\t{json}``. Anything else is free-form log text. This mirrors the
legacy worker emitter (``backend/bend_analysis_worker.py:84-85``) and reader
(``backend/_server_legacy_impl.py:971-995``) without importing either.

ERROR payloads come in two shapes: the production steps write
``{"code", "message"}`` (ENGINE_IO §3/§4) and the legacy worker writes
``{"error"}``. :func:`error_message` reads both.
"""
from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["PROTOCOL_TAGS", "parse_protocol_line", "sanitize_message", "error_message", "format_protocol_line"]

PROTOCOL_TAGS: frozenset[str] = frozenset({"PROGRESS", "RESULT", "ERROR"})
_ANSI = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
_MAX_MESSAGE = 2000


def sanitize_message(message: Any) -> str:
    """Strip ANSI escapes, collapse whitespace, cap at 2000 characters (mirrors
    ``_sanitize_error_message`` at backend/bend_analysis_worker.py:43-47)."""
    text = str(message) if message is not None else ""
    text = _ANSI.sub("", text)
    text = " ".join(text.split())
    return text.strip()[:_MAX_MESSAGE]


def parse_protocol_line(line: str) -> tuple[str | None, dict[str, Any] | None]:
    """Return ``(tag, payload)`` for a protocol record, else ``(None, None)``.

    The payload must be a JSON object; a known tag with a malformed or
    non-object payload is treated as free-form text.
    """
    text = line.strip()
    if "\t" not in text:
        return None, None
    tag, _, raw = text.partition("\t")
    if tag not in PROTOCOL_TAGS:
        return None, None
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    return tag, payload


def error_message(payload: dict[str, Any] | None) -> tuple[str | None, str]:
    """``(code, message)`` from an ERROR payload of either shape."""
    if not payload:
        return None, ""
    code = payload.get("code")
    message = payload.get("message", payload.get("error", ""))
    return (str(code) if isinstance(code, str) and code else None), sanitize_message(message)


def format_protocol_line(tag: str, payload: dict[str, Any]) -> str:
    """Render one protocol line (used by tests and fake steps)."""
    if tag not in PROTOCOL_TAGS:
        raise ValueError(f"unknown protocol tag {tag!r}")
    return f"{tag}\t{json.dumps(payload, ensure_ascii=False)}"
