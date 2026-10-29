"""Dry-run-by-default, cached, concurrent Jev runner for v2 states.

Reuses the v1 client's transport, retry/backoff (``client_v1._send``) and
response validation. Nothing leaves the machine unless ``live=True`` *and*
``TYPESAFE_API_KEY`` is in the environment; the key is read at call time and
never written or echoed. Every response is cached under ``cache/<body sha>.json``
so a re-run never re-sends a body. A ``budget_usd`` cap is checked against the
pessimistic token estimate *before* any call is made, and against the reported
usage while running (stop when exceeded).
"""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from research.jev.client_v1 import (
    API_KEY_ENV,
    PRICE_USD_PER_MTOK,
    ApiError,
    HttpxTransport,
    LiveNotAuthorized,
    _send,
    resolve_model,
    validate_response,
)
from research.jev.questions_v2 import question_set_sha256, select_questions_v2
from research.jev.request_v1 import body_bytes, body_sha256, estimate_tokens, validate_request
from research.jev.states_v2 import check_state_v2

__all__ = ["prepare", "dry_run", "live_run"]


def prepare(states: Sequence[tuple[str, dict[str, Any]]], *, model: str, variant: str) -> list[dict[str, Any]]:
    questions = select_questions_v2(variant)
    out = []
    for rid, state in states:
        check_state_v2(state)
        body = {"state": state, "model": model, "questions": questions}
        validate_request(body, check_state_rules=False)
        out.append({"record_id": rid, "body": body, "raw": body_bytes(body), "sha": body_sha256(body),
                    "est_tokens": estimate_tokens(body)})
    return out


def _cost(tokens: int) -> float:
    return tokens / 1e6 * PRICE_USD_PER_MTOK


def dry_run(states, out_dir: Path, *, model: str, variant: str, sample_bodies: int = 3) -> dict[str, Any]:
    prepared = prepare(states, model=model, variant=variant)
    out = Path(out_dir)
    (out / "requests").mkdir(parents=True, exist_ok=True)
    for p in prepared[:sample_bodies]:
        (out / "requests" / f"{p['record_id']}.json").write_bytes(p["raw"])
    cache = out.parent / "cache"
    uncached = [p for p in prepared if not (cache / f"{p['sha']}.json").is_file()]
    tok = sum(p["est_tokens"] for p in uncached)
    summary = {"mode": "dry_run", "model": model, "variant": variant,
               "question_set_sha256": question_set_sha256(select_questions_v2(variant), variant),
               "records": len(prepared), "uncached_records": len(uncached), "est_tokens_uncached": tok,
               "est_cost_usd_uncached": round(_cost(tok), 4), "price_usd_per_mtok": PRICE_USD_PER_MTOK}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def live_run(states, cache_dir: Path, *, live: bool, model: str, variant: str, budget_usd: float,
             workers: int = 8, env: Mapping[str, str] | None = None, transport=None) -> dict[str, Any]:
    env = os.environ if env is None else env
    if not live:
        raise LiveNotAuthorized("live calls need the explicit --live flag")
    key = env.get(API_KEY_ENV, "")
    if not key and transport is None:
        raise LiveNotAuthorized(f"live calls need {API_KEY_ENV} in the environment")
    model, _alias = resolve_model(model)
    prepared = prepare(states, model=model, variant=variant)
    questions = select_questions_v2(variant)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    todo = [p for p in prepared if not (cache / f"{p['sha']}.json").is_file()]
    est = _cost(sum(p["est_tokens"] for p in todo))
    if est > budget_usd:
        raise LiveNotAuthorized(f"estimated ${est:.2f} for {len(todo)} uncached bodies exceeds budget ${budget_usd:.2f}")
    transport = transport or HttpxTransport(key)
    lock = threading.Lock()
    used = {"tokens": 0, "sent": 0}
    stop = threading.Event()

    def one(p):
        if stop.is_set():
            return None
        payload = _send(transport, p["raw"], sleep=__import__("time").sleep)
        validate_response(payload, questions)
        (cache / f"{p['sha']}.json").write_text(json.dumps(payload, sort_keys=True))
        with lock:
            used["tokens"] += int(payload["usage"].get("input_tokens", 0))
            used["sent"] += 1
            if _cost(used["tokens"]) > budget_usd:
                stop.set()
        return p["sha"]

    error = None
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, p) for p in todo]
            for f in as_completed(futs):
                try:
                    f.result()
                except ApiError as exc:  # keep what we have; report
                    error = str(exc)
                    stop.set()
    finally:
        if hasattr(transport, "close"):
            transport.close()
    return {"records": len(prepared), "sent": used["sent"], "cached_before": len(prepared) - len(todo),
            "usage_input_tokens": used["tokens"], "usage_cost_usd": round(_cost(used["tokens"]), 4),
            "stopped_on_budget": stop.is_set() and error is None, "error": error}


def load_answers(states, cache_dir: Path, *, model: str, variant: str) -> dict[str, dict[str, Any]]:
    """``{record_id: answers}`` for every record whose exact body has a cached response."""
    out = {}
    for p in prepare(states, model=resolve_model(model)[0], variant=variant):
        f = Path(cache_dir) / f"{p['sha']}.json"
        if f.is_file():
            payload = json.loads(f.read_text())
            out[p["record_id"]] = {"answers": payload["answers"], "usage": payload.get("usage", {}),
                                   "model": payload.get("model")}
    return out
