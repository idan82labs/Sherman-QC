"""TypeSafe System One request bodies for the bend-candidate harness (C10, v1).

Request shape (``POST https://api.typesafe.ai/v1/systemone``, docs verified
2026-10-17 at https://docs.typesafe.ai/api.md)::

    {"state": <object>, "model": "<model id>",
     "questions": {"<id>": {"type": "noul"|"choice"|"score",
                            "instructions": <str|object|array>,
                            "criteria": ...}}}

:func:`validate_request` enforces the documented request rules, conservatively:

- body at most 32 MiB, UTF-8 JSON without BOM, finite numbers only;
- at most 64 nested levels in each of ``state``, ``instructions``, ``criteria``;
- Noul criteria only ``true``/``false``; Choice 1..255 options (a map);
  Score 2..10 levels (an array);
- context limits: 64k tokens per request and 32k per question (state plus one
  question), estimated as ``ceil(len(json) / 3)``. The real tokenizer usually
  gives fewer tokens; the estimate is deliberately pessimistic.

:func:`body_bytes` is the exact byte string that is cached, hashed and sent.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from research.jev.state_v1 import check_state

__all__ = [
    "MAX_BODY_BYTES",
    "MAX_NESTING",
    "MAX_CHOICE_OPTIONS",
    "SCORE_LEVELS",
    "MAX_REQUEST_TOKENS",
    "MAX_QUESTION_TOKENS",
    "RequestInvalid",
    "build_request",
    "validate_request",
    "body_bytes",
    "body_sha256",
    "estimate_tokens",
]

MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_NESTING = 64
MAX_CHOICE_OPTIONS = 255
SCORE_LEVELS = (2, 10)
MAX_REQUEST_TOKENS = 64_000
MAX_QUESTION_TOKENS = 32_000
QUESTION_TYPES = ("noul", "choice", "score")


class RequestInvalid(ValueError):
    """The body breaks a request rule; ``problems`` lists every violation found."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def body_bytes(body: Mapping[str, Any]) -> bytes:
    """Canonical UTF-8 JSON bytes of a request body (no BOM, sorted keys)."""
    return _dumps(body).encode("utf-8")


def body_sha256(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(body_bytes(body)).hexdigest()


def estimate_tokens(obj: Any) -> int:
    """Conservative token estimate: ``ceil(len(json_text) / 3)``."""
    return math.ceil(len(_dumps(obj)) / 3)


def _depth(obj: Any) -> int:
    """Nesting depth of objects/arrays (a scalar is 0, ``{}`` is 1)."""
    if isinstance(obj, Mapping):
        return 1 + max((_depth(v) for v in obj.values()), default=0)
    if isinstance(obj, (list, tuple)):
        return 1 + max((_depth(v) for v in obj), default=0)
    return 0


def _non_finite(obj: Any, path: str, out: list[str]) -> None:
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):
            out.append(f"{path}: non-finite number")
        return
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if not isinstance(k, str):
                out.append(f"{path}: non-string key")
            _non_finite(v, f"{path}.{k}", out)
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _non_finite(v, f"{path}[{i}]", out)
        return
    out.append(f"{path}: unsupported type {type(obj).__name__}")


def _instruction_ok(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, (Mapping, list, tuple)) and bool(value)


def _check_question(qid: str, q: Any, problems: list[str]) -> None:
    path = f"questions.{qid}"
    if not isinstance(q, Mapping):
        problems.append(f"{path}: must be an object")
        return
    qtype = q.get("type")
    if qtype not in QUESTION_TYPES:
        problems.append(f"{path}.type must be one of {QUESTION_TYPES}")
        return
    extra = sorted(set(q) - {"type", "instructions", "criteria"})
    if extra:
        problems.append(f"{path}: unexpected keys {extra}")
    if not _instruction_ok(q.get("instructions")):
        problems.append(f"{path}.instructions is required and must be non-empty")
    for part in ("instructions", "criteria"):
        if part in q and _depth(q[part]) > MAX_NESTING:
            problems.append(f"{path}.{part}: nesting deeper than {MAX_NESTING}")
    criteria = q.get("criteria")
    if qtype == "noul":
        if criteria is not None:
            if not isinstance(criteria, Mapping) or not set(criteria) <= {"true", "false"} or not criteria:
                problems.append(f"{path}.criteria for a noul is an object with only 'true'/'false'")
    elif qtype == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            problems.append(f"{path}.criteria for a choice is a non-empty option map")
        elif len(criteria) > MAX_CHOICE_OPTIONS:
            problems.append(f"{path}.criteria has {len(criteria)} options (max {MAX_CHOICE_OPTIONS})")
        elif any(not isinstance(k, str) or not k for k in criteria):
            problems.append(f"{path}.criteria option names must be non-empty strings")
    else:  # score
        lo, hi = SCORE_LEVELS
        if not isinstance(criteria, (list, tuple)):
            problems.append(f"{path}.criteria for a score is an ordered array of levels")
        elif not lo <= len(criteria) <= hi:
            problems.append(f"{path}.criteria has {len(criteria)} levels (allowed {lo}..{hi})")


def validate_request(body: Any, *, check_state_rules: bool = True) -> dict[str, int]:
    """Raise :class:`RequestInvalid` on any violation; return ``{bytes, est_tokens}`` otherwise.

    With ``check_state_rules`` (default) the state must also pass the C10
    deny-list (:func:`research.jev.state_v1.check_state`).
    """
    problems: list[str] = []
    if not isinstance(body, Mapping):
        raise RequestInvalid(["body must be a JSON object"])
    missing = [k for k in ("state", "model", "questions") if k not in body]
    if missing:
        problems.append(f"missing required fields {missing}")
    extra = sorted(set(body) - {"state", "model", "questions"})
    if extra:
        problems.append(f"unexpected top-level fields {extra}")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        problems.append("model must be a non-empty string")
    state = body.get("state")
    if state is not None and not isinstance(state, (str, Mapping, list, tuple)):
        problems.append("state must be a string, object or array")
    if state is not None and _depth(state) > MAX_NESTING:
        problems.append(f"state: nesting deeper than {MAX_NESTING}")
    questions = body.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        problems.append("questions must be a non-empty map")
        questions = {}
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid:
            problems.append("question ids must be non-empty strings")
            continue
        _check_question(qid, q, problems)
    _non_finite(body, "body", problems)
    if problems:
        raise RequestInvalid(problems)

    if check_state_rules:
        try:
            check_state(state)
        except ValueError as exc:
            raise RequestInvalid([f"state: {exc}"]) from exc

    raw = body_bytes(body)
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append("body must not start with a UTF-8 BOM")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        problems.append("body is not valid UTF-8")
    if len(raw) > MAX_BODY_BYTES:
        problems.append(f"body is {len(raw)} bytes (max {MAX_BODY_BYTES})")
    est_tokens = estimate_tokens(body)
    if est_tokens > MAX_REQUEST_TOKENS:
        problems.append(f"estimated {est_tokens} tokens per request (max {MAX_REQUEST_TOKENS})")
    state_tokens = estimate_tokens(state)
    for qid, q in questions.items():
        per_question = state_tokens + estimate_tokens(q)
        if per_question > MAX_QUESTION_TOKENS:
            problems.append(f"questions.{qid}: estimated {per_question} tokens with the state (max {MAX_QUESTION_TOKENS})")
    if problems:
        raise RequestInvalid(problems)
    return {"bytes": len(raw), "est_tokens": est_tokens}


def build_request(state: Mapping[str, Any], model: str, questions: Mapping[str, Any]) -> dict[str, Any]:
    """Build and validate one request body."""
    body = {"state": dict(state), "model": model, "questions": dict(questions)}
    validate_request(body)
    return body
