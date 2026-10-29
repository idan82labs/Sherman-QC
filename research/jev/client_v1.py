"""Dry-run-by-default TypeSafe System One client for the Jev harness (C10, v1).

Nothing is sent unless **both** hold: the caller passes ``live=True`` (CLI
``live --live``) and ``TYPESAFE_API_KEY`` is set in the environment. Otherwise
:class:`LiveNotAuthorized` is raised (CLI exit 2). The key is read from the
environment at call time only and never written, logged or echoed.

Dry run (:func:`dry_run`) writes ``requests/<record_id>.json`` (the exact bytes
that would be sent) and ``summary.json`` with the request count, the estimated
tokens (``ceil(len(json)/3)`` per request) and the estimated cost at
:data:`PRICE_USD_PER_MTOK` (USD per million input tokens; output tokens are free).

Live (:func:`live_run`), per the docs verified 2026-10-17:

- ``POST https://api.typesafe.ai/v1/systemone`` with ``Authorization: Bearer``;
- 10 s timeout per attempt;
- 429 and 529 are retried with exponential backoff, honouring ``Retry-After``;
  other 5xx and transport failures (timeouts, connection errors:
  ``httpx.TransportError``, surfaced as :class:`TransportFailure`) are retried the
  same way; every other 4xx (400, 401, 413, 415, 422, ...) fails fast. Exhausted
  retries raise :class:`ApiError` (CLI exit 3) after ``answers.jsonl`` and
  ``summary.json`` are written;
- responses are cached by ``sha256(body)``, so a re-run never re-sends a body it
  already has an answer for;
- ``answers.jsonl`` records, per record, the response ``model`` (the versioned id
  that answered), the answers, ``usage`` and the request identity
  (:func:`request_identity`: body sha256, request model, question variant, ids and
  set sha256, ablation), so the evaluator can refuse answers that do not belong to
  the current states and question set.

Model ids: only versioned ids (``jev-1.13.0``) are accepted. The aliases
``jev-latest`` (-> ``jev-1.13.0``) and ``jev-preview`` (-> ``jev-1.14.0``) move
when a release ships, so they are refused unless ``allow_alias``.

Splits: the live client sends only ``dev`` and ``calib`` records and needs a
split assignment to know which those are; ``eval`` is sent only with
``include_eval``. ``excluded`` is never sent. A dry run without a split
assignment writes every record (nothing leaves the machine).
"""
from __future__ import annotations

import email.utils
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from research.jev.labels_v1 import LabelRow, read_jsonl, write_jsonl
from research.jev.questions_v1 import DEFAULT_QUESTIONS, question_set_sha256, select_questions
from research.jev.request_v1 import body_bytes, body_sha256, build_request, estimate_tokens
from research.jev.splits_v1 import SENDABLE_SPLITS, SplitManifest, load_split_manifest
from research.jev.state_v1 import ablate

__all__ = [
    "ENDPOINT",
    "API_KEY_ENV",
    "MODEL_ALIASES",
    "PRICE_USD_PER_MTOK",
    "TIMEOUT_S",
    "ClientError",
    "LiveNotAuthorized",
    "ModelNotVersioned",
    "ApiError",
    "TransportFailure",
    "Transport",
    "request_identity",
    "HttpxTransport",
    "resolve_model",
    "load_states",
    "dry_run",
    "live_run",
    "validate_response",
    "parse_retry_after",
]

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
API_KEY_ENV = "TYPESAFE_API_KEY"
#: Alias table as documented on 2026-10-17 (docs.typesafe.ai/models.md).
MODEL_ALIASES = {"jev-latest": "jev-1.13.0", "jev-preview": "jev-1.14.0"}
_VERSIONED = re.compile(r"^jev-\d+\.\d+\.\d+$")
PRICE_USD_PER_MTOK = 0.042
TIMEOUT_S = 10.0
RETRY_STATUSES = frozenset({429, 529})
MAX_ATTEMPTS = 6
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0
RETRY_AFTER_CAP_S = 300.0


class ClientError(RuntimeError):
    """Usage or policy problem (CLI exit 2); :class:`ApiError` is the exit-3 case."""

    exit_code = 2


class LiveNotAuthorized(ClientError):
    """Live mode was not requested explicitly, or the API key is not in the environment."""

    exit_code = 2


class ModelNotVersioned(ClientError):
    exit_code = 2


class ApiError(ClientError):
    """The API refused or failed (CLI exit 3)."""

    exit_code = 3

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class TransportFailure(RuntimeError):
    """The request did not produce an HTTP response (timeout, connection error)."""


#: HTTP status recorded for a transport failure (no response was received).
TRANSPORT_FAILURE_STATUS = 0


def _transport_errors() -> tuple[type[BaseException], ...]:
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a dependency
        return (TransportFailure,)
    return (TransportFailure, httpx.TransportError)


class Transport(Protocol):
    def post(self, body: bytes) -> tuple[int, Mapping[str, str], bytes]:
        """Send one body; return ``(status, headers, payload)``."""


class HttpxTransport:
    """The real transport (httpx). Holds the key in memory only."""

    def __init__(self, api_key: str, *, endpoint: str = ENDPOINT, timeout_s: float = TIMEOUT_S):
        import httpx

        if not api_key:
            raise LiveNotAuthorized(f"{API_KEY_ENV} is empty")
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._endpoint = endpoint
        self._client = httpx.Client(timeout=timeout_s)

    def post(self, body: bytes) -> tuple[int, Mapping[str, str], bytes]:
        import httpx

        try:
            response = self._client.post(self._endpoint, content=body, headers=self._headers)
        except httpx.TransportError as exc:
            # Only the exception type and message (never headers) are kept.
            raise TransportFailure(f"{type(exc).__name__}: {exc}") from None
        return response.status_code, dict(response.headers), response.content

    def close(self) -> None:
        self._client.close()

    def __repr__(self) -> str:  # never show the key
        return f"HttpxTransport(endpoint={self._endpoint!r})"


def resolve_model(model: str, *, allow_alias: bool = False) -> tuple[str, bool]:
    """``(model_to_send, is_alias)``. Unversioned names are refused unless ``allow_alias``."""
    if _VERSIONED.fullmatch(model):
        return model, False
    if model in MODEL_ALIASES:
        if not allow_alias:
            raise ModelNotVersioned(
                f"{model!r} is an alias (currently {MODEL_ALIASES[model]}); pass a versioned id or --allow-alias"
            )
        return model, True
    raise ModelNotVersioned(f"{model!r} is not a versioned Jev model id (jev-X.Y.Z)")


def load_states(path: Path) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for row in read_jsonl(Path(path)):
        rid, state = row.get("record_id"), row.get("state")
        if not isinstance(rid, str) or not isinstance(state, dict):
            raise ClientError(f"{path}: rows need record_id and state")
        if rid in seen:
            raise ClientError(f"{path}: duplicate record_id {rid}")
        seen.add(rid)
        out.append((rid, state))
    return out


def _assignments(splits: Path | Mapping[str, Any] | SplitManifest | None,
                 labels: Path | None) -> tuple[dict[str, str] | None, str | None]:
    if splits is None:
        return None, None
    if labels is None:
        raise ClientError("a split manifest needs the labels.jsonl that names each record's SKU")
    manifest = splits if isinstance(splits, SplitManifest) else load_split_manifest(splits)  # type: ignore[arg-type]
    rows = [LabelRow.from_json(r) for r in read_jsonl(Path(labels))]
    return manifest.assign(rows), manifest.sha256


def _select(states: Sequence[tuple[str, dict[str, Any]]], assignment: Mapping[str, str] | None,
            include_eval: bool) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, int]]:
    if assignment is None:
        return list(states), {}
    allowed = set(SENDABLE_SPLITS) | ({"eval"} if include_eval else set())
    chosen, skipped = [], {}
    for rid, state in states:
        split = assignment.get(rid, "excluded")
        if split in allowed:
            chosen.append((rid, state))
        else:
            skipped[split] = skipped.get(split, 0) + 1
    return chosen, skipped


@dataclass(frozen=True)
class _Prepared:
    record_id: str
    body: dict[str, Any]
    raw: bytes
    sha256: str
    est_tokens: int


def _prepare(states: Iterable[tuple[str, dict[str, Any]]], *, model: str, questions: Mapping[str, Any],
             cad_free: bool, context_free: bool) -> list[_Prepared]:
    out = []
    for rid, state in states:
        body = build_request(ablate(state, cad_free=cad_free, context_free=context_free), model, questions)
        out.append(_Prepared(rid, body, body_bytes(body), body_sha256(body), estimate_tokens(body)))
    return out


def request_identity(state: Mapping[str, Any], *, model: str, variant: str = "v1",
                     question_names: Sequence[str] = DEFAULT_QUESTIONS, cad_free: bool = False,
                     context_free: bool = False) -> dict[str, Any]:
    """The identity fields recorded on every ``answers.jsonl`` row for ``state``.

    ``body_sha256`` is the sha of the exact body that is (or would be) sent; the other
    fields say how to rebuild it, so :mod:`.evaluate_v1` can recompute and compare.
    """
    questions = select_questions(variant, question_names)
    body = build_request(ablate(state, cad_free=cad_free, context_free=context_free), model, questions)
    return {
        "body_sha256": body_sha256(body),
        "request_model": model,
        "question_variant": variant,
        "question_ids": list(questions),
        "question_set_sha256": question_set_sha256(questions, variant),
        "ablation": {"cad_free": cad_free, "context_free": context_free},
    }


def _new_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ClientError(f"{path} exists and is not empty")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _summary(prepared: Sequence[_Prepared], **extra: Any) -> dict[str, Any]:
    est = sum(p.est_tokens for p in prepared)
    return {
        "schema": "jev_request_summary_v1",
        "count": len(prepared),
        "est_tokens": est,
        "est_cost_usd": round(est * PRICE_USD_PER_MTOK / 1_000_000, 8),
        "price_usd_per_mtok": PRICE_USD_PER_MTOK,
        "token_estimate": "ceil(len(canonical_json)/3) per request (conservative)",
        **extra,
    }


def dry_run(
    states_path: Path,
    out_dir: Path,
    *,
    model: str = "jev-1.13.0",
    allow_alias: bool = False,
    variant: str = "v1",
    question_names: Sequence[str] = DEFAULT_QUESTIONS,
    splits: Path | Mapping[str, Any] | None = None,
    labels: Path | None = None,
    include_eval: bool = False,
    cad_free: bool = False,
    context_free: bool = False,
) -> dict[str, Any]:
    """Write the request bodies and a cost summary. No network access of any kind."""
    sent_model, is_alias = resolve_model(model, allow_alias=allow_alias)
    questions = select_questions(variant, question_names)
    assignment, split_sha = _assignments(splits, labels)
    chosen, skipped = _select(load_states(states_path), assignment, include_eval)
    prepared = _prepare(chosen, model=sent_model, questions=questions, cad_free=cad_free, context_free=context_free)
    out = _new_dir(out_dir)
    req_dir = out / "requests"
    req_dir.mkdir()
    for p in prepared:
        (req_dir / f"{p.record_id}.json").write_bytes(p.raw)
    summary = _summary(
        prepared,
        mode="dry_run",
        network="none",
        model=sent_model,
        model_is_alias=is_alias,
        question_variant=variant,
        question_ids=list(questions),
        question_set_sha256=question_set_sha256(questions, variant),
        ablation={"cad_free": cad_free, "context_free": context_free},
        split_filter="none (dry run, every record)" if assignment is None else
        ("dev+calib+eval" if include_eval else "dev+calib"),
        split_manifest_sha256=split_sha,
        skipped_by_split=skipped,
        states_sha256=hashlib.sha256(Path(states_path).read_bytes()).hexdigest(),
    )
    _write_json(out / "summary.json", summary)
    return summary


# --------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header (delta-seconds or HTTP-date)."""
    if value is None:
        return None
    value = value.strip()
    try:
        seconds = float(value)
        return max(0.0, min(seconds, RETRY_AFTER_CAP_S)) if math.isfinite(seconds) else None
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    current = time.time() if now is None else now
    return max(0.0, min(when.timestamp() - current, RETRY_AFTER_CAP_S))


def validate_response(payload: Any, questions: Mapping[str, Any]) -> None:
    """Raise :class:`ApiError` unless ``payload`` has the documented answer shape."""
    if not isinstance(payload, Mapping):
        raise ApiError(200, "response is not an object")
    model = payload.get("model")
    if not isinstance(model, str) or not _VERSIONED.fullmatch(model):
        raise ApiError(200, "response.model is not a versioned id")
    answers = payload.get("answers")
    if not isinstance(answers, Mapping) or set(answers) != set(questions):
        raise ApiError(200, "response.answers does not match the question ids")
    for qid, q in questions.items():
        a = answers[qid]
        if not isinstance(a, Mapping) or a.get("type") != q["type"]:
            raise ApiError(200, f"answer {qid} has the wrong type")
        if q["type"] == "noul":
            v = a.get("noul")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
                raise ApiError(200, f"answer {qid}.noul must be a number in [0, 1]")
        else:
            probs = a.get("probabilities")
            if not isinstance(probs, Mapping) or not probs:
                raise ApiError(200, f"answer {qid}.probabilities missing")
            if abs(sum(float(p) for p in probs.values()) - 1.0) > 1e-3:
                raise ApiError(200, f"answer {qid}.probabilities do not sum to 1")
            if q["type"] == "choice" and a.get("choice") not in q["criteria"]:
                raise ApiError(200, f"answer {qid}.choice is not an option")
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        raise ApiError(200, "response.usage missing")


def _send(transport: Transport, raw: bytes, *, sleep: Callable[[float], None],
          max_attempts: int = MAX_ATTEMPTS) -> dict[str, Any]:
    delay = BACKOFF_BASE_S
    last = ""
    status = TRANSPORT_FAILURE_STATUS
    transport_errors = _transport_errors()
    for attempt in range(1, max_attempts + 1):
        try:
            status, headers, payload = transport.post(raw)
        except transport_errors as exc:
            status, last = TRANSPORT_FAILURE_STATUS, f"transport failure: {type(exc).__name__}: {exc}"[:300]
            if attempt == max_attempts:
                break
            sleep(min(BACKOFF_CAP_S, delay))
            delay *= 2
            continue
        if status == 200:
            try:
                return json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ApiError(status, "response is not JSON") from exc
        text = payload[:300].decode("utf-8", "replace")
        last = text
        retryable = status in RETRY_STATUSES or 500 <= status < 600
        if not retryable:
            raise ApiError(status, text)
        if attempt == max_attempts:
            break
        lowered = {k.lower(): v for k, v in headers.items()}
        wait = parse_retry_after(lowered.get("retry-after"))
        sleep(min(BACKOFF_CAP_S, delay) if wait is None else wait)
        delay *= 2
    raise ApiError(status, f"gave up after {max_attempts} attempts: {last}")


def live_run(
    states_path: Path,
    out_dir: Path,
    *,
    live: bool,
    splits: Path | Mapping[str, Any],
    labels: Path,
    model: str = "jev-1.13.0",
    allow_alias: bool = False,
    variant: str = "v1",
    question_names: Sequence[str] = DEFAULT_QUESTIONS,
    include_eval: bool = False,
    cad_free: bool = False,
    context_free: bool = False,
    env: Mapping[str, str] | None = None,
    transport: Transport | None = None,
    sleep: Callable[[float], None] | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Send the selected records to System One, caching by body sha; returns the summary."""
    env = os.environ if env is None else env
    sleep = time.sleep if sleep is None else sleep
    if not live:
        raise LiveNotAuthorized("live calls need the explicit --live flag")
    api_key = env.get(API_KEY_ENV, "")
    if not api_key:
        raise LiveNotAuthorized(f"live calls need {API_KEY_ENV} in the environment")
    if splits is None or labels is None:
        raise LiveNotAuthorized("live calls need a split manifest and labels.jsonl (only dev/calib are sent)")
    sent_model, is_alias = resolve_model(model, allow_alias=allow_alias)
    questions = select_questions(variant, question_names)
    qset_sha = question_set_sha256(questions, variant)
    ablation = {"cad_free": cad_free, "context_free": context_free}
    assignment, split_sha = _assignments(splits, labels)
    chosen, skipped = _select(load_states(states_path), assignment, include_eval)
    if max_records is not None:
        chosen = chosen[:max_records]
    prepared = _prepare(chosen, model=sent_model, questions=questions, cad_free=cad_free, context_free=context_free)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    owns_transport = transport is None
    transport = transport or HttpxTransport(api_key)
    sent = cached = 0
    error: str | None = None
    error_status = 0
    answers: list[dict[str, Any]] = []
    try:
        for p in prepared:
            cache_file = cache / f"{p.sha256}.json"
            if cache_file.is_file():
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                was_cached = True
                cached += 1
            else:
                payload = _send(transport, p.raw, sleep=sleep)
                validate_response(payload, questions)
                _write_json(cache_file, payload)
                was_cached = False
                sent += 1
            answers.append({"record_id": p.record_id, "body_sha256": p.sha256,
                            "request_model": sent_model, "question_variant": variant,
                            "question_ids": list(questions), "question_set_sha256": qset_sha,
                            "ablation": ablation,
                            "response_model": payload["model"], "answers": payload["answers"],
                            "usage": payload["usage"], "cached": was_cached})
    except ApiError as exc:
        error = str(exc)
        error_status = exc.status
    finally:
        if owns_transport and hasattr(transport, "close"):
            transport.close()  # type: ignore[union-attr]
    write_jsonl(out / "answers.jsonl", answers)
    summary = _summary(
        prepared,
        mode="live",
        model=sent_model,
        model_is_alias=is_alias,
        response_models=sorted({a["response_model"] for a in answers}),
        question_variant=variant,
        question_ids=list(questions),
        question_set_sha256=qset_sha,
        ablation=ablation,
        split_filter="dev+calib+eval" if include_eval else "dev+calib",
        split_manifest_sha256=split_sha,
        skipped_by_split=skipped,
        sent=sent,
        cached=cached,
        answered=len(answers),
        usage_input_tokens=sum(int(a["usage"].get("input_tokens", 0)) for a in answers),
        error=error,
    )
    _write_json(out / "summary.json", summary)
    if error is not None:
        raise ApiError(error_status, f"stopped after {len(answers)} answers: {error}")
    return summary
