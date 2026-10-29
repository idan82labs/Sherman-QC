"""``/api/line`` router (C07): station feed, events, jobs, artifacts, quarantine,
devices, status, readiness and metrics.

Endpoints (CONTRACTS §9; capabilities from C04 ``principals``):

====================================  ===============  =========================================
Route                                 Capability       Notes
====================================  ===============  =========================================
GET  /jobs                            line.read        ``station_id, status, sku, kind, limit,
                                                       before`` (``created_at`` cursor)
GET  /jobs/{job_id}                   line.read        device: own station only (else 404)
GET  /jobs/{job_id}/result            line.read        validated ShopfloorInspectionResultV1;
                                                       409 not completed, 404 not present
GET  /jobs/{job_id}/zeroshot-overlay  line.read        ``sherman_zeroshot_overlay_v1`` (validated,
                                                       identity-checked); 404 OVERLAY_MISSING
GET  /jobs/{job_id}/artifacts/{name}  line.read        ``layout.ARTIFACT_ALLOWLIST`` only
POST /jobs/{job_id}/retry             line.operate     JOB_RETRY_REQUESTED (store)
POST /jobs/{job_id}/cancel            line.operate     JOB_CANCEL_REQUESTED (store)
GET  /stations/{station_id}/latest    line.read        StationFeedV1, ETag / If-None-Match 304
GET  /events                          line.read        JSON polling; ``reset`` when the cursor
                                                       is ahead of the store
POST /stream-token                    line.read        ``{station_id}`` -> ``{token, expires_at}``
GET  /events/stream                   stream token     SSE; ``token`` query is the only auth;
                                                       closes when the token expires
GET  /quarantine                      line.read
GET  /quarantine/held                 line.read        station part picker: open held scans of
                                                       ``station_id`` with suggested parts
GET  /released-parts                  line.read        released parts, ``q`` (SKU prefix /
                                                       substring, part name), ``limit``
POST /quarantine/{id}/assign          line.operate     ``{sku}`` or ``{recipe_id}`` (the SKU's
                                                       active recipe only), optional
                                                       ``declared_stage`` / ``stage_index``
                                                       -> inspect job; idempotent
POST /quarantine/{id}/dismiss         line.operate     ``{reason?, note?}``; idempotent
GET  /status                          line.operate
GET  /health/ready                    none             ``{ready, checks}``; 503 when not ready
POST|GET /devices, DELETE /devices/x  devices.manage   token shown once
GET  /metrics                         metrics.read     Prometheus text 0.0.4
====================================  ===============  =========================================

Refusals raised by the service are JSON ``{"detail": {"code", "message"}}``
with the header ``X-Sherman-Error-Code``; they are rendered by this router's
route class, so an app-level exception handler cannot reshape them. Auth
refusals (401/403/503) come from the C04 dependencies.

Text that came from the store (job failure messages, event payloads,
quarantine details) is redacted before any ``line.read`` principal sees it:
credentials and the data-root path are removed (``svc.redact_tree``).

The router is not registered here: C11 includes it before the SPA catch-all.
It never imports the legacy server, ``apps.api.legacy_context`` or
``apps.production_line.recipes`` at import time.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Coroutine
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from apps.api.services import part_picker_service as picker
from apps.api.services import production_line_service as svc
from apps.api.services.production_line_service import (
    LineApiConfig,
    LineApiError,
    ResultLoader,
    StatusProbes,
    get_line_api_config,
    get_line_store,
    get_metrics_registry,
    get_recipe_provider,
    get_result_loader,
    get_status_probes,
    get_stream_secret,
)
from apps.production_line.ingest.ports import RecipeProvider
from apps.production_line.security import stream_tokens
from apps.production_line.security.dependencies import (
    get_device_token_store,
    get_line_settings,
    get_principal,
    require_capability,
    require_line_settings,
)
from apps.production_line.security.device_tokens import DeviceTokenError, DeviceTokenStore
from apps.production_line.security.principals import Principal, has_capability
from apps.production_line.settings import LineSettings
from apps.production_line.store.line_store import JobStateError, LineStore, LineStoreError, QuarantineError
from apps.production_line.telemetry import MetricsRegistry

__all__ = ["router"]


def _error_response(exc: LineApiError) -> JSONResponse:
    headers = {"X-Sherman-Error-Code": exc.code, **exc.headers}
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": {"code": exc.code, "message": exc.message}},
        headers=headers,
    )


class _LineRoute(APIRoute):
    """Renders :class:`LineApiError` (from handlers or dependencies) as JSON."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                return await original(request)
            except LineApiError as exc:
                return _error_response(exc)

        return handler


router = APIRouter(prefix=svc.API_PREFIX, tags=["Production line"], route_class=_LineRoute)


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StreamTokenRequest(_Body):
    station_id: str = Field(min_length=1, max_length=64)


class AssignRequest(_Body):
    """Exactly one of ``sku`` (preferred: the server picks the SKU's active
    recipe) or ``recipe_id`` (must be that active recipe)."""

    sku: str | None = Field(default=None, min_length=1, max_length=64)
    recipe_id: str | None = Field(default=None, min_length=1, max_length=200)
    #: Station part picker: the scan's stage (``DeclaredStageV1``, source OPERATOR).
    declared_stage: Literal["FINAL", "IN_PROCESS"] | None = None
    stage_index: int | None = Field(default=None, ge=0, le=999)


class DismissRequest(_Body):
    reason: Literal["RESCAN", "NOT_A_PART", "DUPLICATE", "WRONG_STATION", "OTHER"] | None = None
    note: str | None = Field(default=None, max_length=500)


class DeviceIssueRequest(_Body):
    station_id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=64)


_NO_CACHE = "private, no-cache"


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@router.get("/jobs")
def list_jobs(
    station_id: str | None = Query(default=None),
    status: list[str] | None = Query(default=None),
    sku: str | None = Query(default=None, max_length=64),
    kind: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    before: str | None = Query(default=None),
    principal: Principal = Depends(require_capability("line.read", "station_id")),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
) -> JSONResponse:
    if station_id is not None:
        svc.check_station_id(station_id)
    statuses = svc.parse_statuses(status)
    rows = store.list_jobs(
        station_id=station_id,
        statuses=statuses,
        sku=sku,
        limit=limit,
        before=svc.parse_before(before),
        kind=svc.parse_kind(kind),
    )
    jobs = [svc.job_summary(settings, store, r).model_dump(mode="json") for r in rows]
    next_before = rows[-1]["created_at"] if len(rows) == limit else None
    return JSONResponse({"jobs": jobs, "next_before": next_before}, headers={"Cache-Control": _NO_CACHE})


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    principal: Principal = Depends(get_principal),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
    loader: ResultLoader = Depends(get_result_loader),
) -> JSONResponse:
    job = svc.load_job_for(principal, store, job_id, settings)
    scan = store.get_scan_file(job["scan_file_id"]) if job.get("scan_file_id") else None
    meta = store.get_job_meta(job_id) or {}
    result_state = None
    if job["execution_status"] == "completed" and job["kind"] == "inspect":
        loaded = loader.load(settings, job)
        result_state = "AVAILABLE" if loaded.result is not None else loaded.code
    summary = svc.without_unverified_verdict(svc.job_summary(settings, store, job, scan=scan), result_state)
    shown = dict(job)
    if shown.get("failure") is not None:
        shown["failure"] = svc.redact_tree(shown["failure"], settings)
    if result_state not in (None, "AVAILABLE"):
        shown["result_summary"] = None  # the recorded verdict did not verify against the result file
    body = {
        "job": shown,
        "summary": summary.model_dump(mode="json"),
        "progress_message": svc.redact_text(meta.get("progress_message") or "", settings, limit=500),
        "scan_file": scan,
        "result_state": result_state,
        "artifacts": svc.available_artifacts(settings, job_id) if job["execution_status"] == "completed" else {},
    }
    if principal.kind == "device" and scan is not None:
        body["scan_file"] = {k: v for k, v in scan.items() if k != "stored_rel_path"}
    body["assignment"] = svc.assignment_view(store, job, principal.kind)
    return JSONResponse(body, headers={"Cache-Control": _NO_CACHE})


@router.get("/jobs/{job_id}/result")
def get_job_result(
    job_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
    loader: ResultLoader = Depends(get_result_loader),
) -> Response:
    job = svc.load_job_for(principal, store, job_id, settings)
    if job["execution_status"] != "completed":
        raise LineApiError(409, "JOB_NOT_COMPLETED", f"job is {job['execution_status']}")
    loaded = loader.load(settings, job)
    if loaded.result is None:
        if loaded.code == "RESULT_MISSING":
            raise LineApiError(404, "RESULT_MISSING", loaded.message)
        raise LineApiError(500, loaded.code or "RESULT_INVALID", loaded.message)
    etag = svc.result_etag(job_id, loaded)
    headers = {"ETag": etag, "Cache-Control": _NO_CACHE, "X-Sherman-Result-Check": loaded.check}
    if svc.if_none_match_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    return JSONResponse(loaded.result.model_dump(mode="json", by_alias=True), headers=headers)


@router.get("/jobs/{job_id}/zeroshot-overlay")
def get_job_zeroshot_overlay(
    job_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
) -> Response:
    """The job's ``sherman_zeroshot_overlay_v1`` (display / REVIEW only; never a measurement)."""
    job = svc.load_job_for(principal, store, job_id, settings)
    if job["execution_status"] != "completed":
        raise LineApiError(409, "JOB_NOT_COMPLETED", f"job is {job['execution_status']}")
    loaded = svc.load_zeroshot_overlay(settings, job)
    if loaded.overlay is None:
        if loaded.code == "OVERLAY_MISSING":
            raise LineApiError(404, "OVERLAY_MISSING", loaded.message)
        raise LineApiError(500, loaded.code or "OVERLAY_INVALID", loaded.message)
    headers = {"ETag": loaded.etag, "Cache-Control": _NO_CACHE}
    if svc.if_none_match_matches(request.headers.get("if-none-match"), loaded.etag):
        return Response(status_code=304, headers=headers)
    return JSONResponse(loaded.overlay, headers=headers)


@router.get("/jobs/{job_id}/artifacts/{name:path}")
def get_artifact(
    job_id: str,
    name: str,
    request: Request,
    principal: Principal = Depends(get_principal),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
) -> Response:
    job = svc.load_job_for(principal, store, job_id, settings)
    handle = svc.open_artifact(settings, store, job, name)
    if svc.if_none_match_matches(request.headers.get("if-none-match"), handle.etag):
        handle.close()
        return Response(status_code=304, headers={k: v for k, v in handle.headers.items() if k != "Content-Disposition"})
    headers = dict(handle.headers)
    headers["Content-Length"] = str(handle.size)
    return StreamingResponse(svc.iter_artifact(handle), media_type=handle.media_type, headers=headers)


@router.post("/jobs/{job_id}/retry")
def retry_job(
    job_id: str,
    principal: Principal = Depends(require_capability("line.operate")),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
) -> JSONResponse:
    svc.load_job_for(principal, store, job_id, settings)
    try:
        new_id = store.request_retry(job_id, principal.audit())
    except KeyError:
        raise LineApiError(404, "JOB_NOT_FOUND", "no such job") from None
    except JobStateError as exc:
        raise LineApiError(409, "JOB_NOT_RETRYABLE", str(exc)) from None
    except LineStoreError as exc:
        raise LineApiError(422, exc.code, str(exc)) from None
    return JSONResponse({"job_id": new_id, "retry_of": job_id})


@router.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str,
    principal: Principal = Depends(require_capability("line.operate")),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
) -> JSONResponse:
    svc.load_job_for(principal, store, job_id, settings)
    try:
        outcome = store.request_cancel(job_id, principal.audit())
    except KeyError:
        raise LineApiError(404, "JOB_NOT_FOUND", "no such job") from None
    if outcome == "not_cancellable":
        raise LineApiError(409, "JOB_NOT_CANCELLABLE", "the job already finished")
    return JSONResponse({"job_id": job_id, "outcome": outcome})


# ---------------------------------------------------------------------------
# Station feed
# ---------------------------------------------------------------------------


@router.get("/stations/{station_id}/latest")
def station_latest(
    station_id: str,
    request: Request,
    principal: Principal = Depends(require_capability("line.read", "station_id")),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
    loader: ResultLoader = Depends(get_result_loader),
) -> Response:
    sid = svc.require_configured_station(settings, station_id)
    state = svc.station_feed_state(settings, store, loader, sid)
    headers = {"ETag": state.etag, "Cache-Control": _NO_CACHE}
    if svc.if_none_match_matches(request.headers.get("if-none-match"), state.etag):
        return Response(status_code=304, headers=headers)
    feed = svc.build_station_feed(settings, store, loader, state)
    return JSONResponse(feed.model_dump(mode="json", by_alias=True), headers=headers)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@router.get("/events")
def list_events(
    after_id: int = Query(default=0, ge=0),
    station_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    include_global: bool = Query(default=False),
    principal: Principal = Depends(require_capability("line.read", "station_id")),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
) -> JSONResponse:
    if station_id is not None:
        svc.check_station_id(station_id)
    glob = include_global and principal.kind != "device"
    newest = store.last_event_id()
    reset = after_id > newest  # restored / rebuilt DB: ids restarted below the client's cursor
    cursor = newest if reset else after_id
    events = store.events_after(cursor, station_id=station_id, limit=limit, include_global=glob)
    body = {
        "events": [svc.event_view(e, principal.kind, settings) for e in events],
        "last_id": events[-1]["id"] if events else cursor,
        "reset": reset,
    }
    return JSONResponse(body, headers={"Cache-Control": "no-store"})


@router.post("/stream-token")
def mint_stream_token(
    payload: StreamTokenRequest,
    principal: Principal = Depends(get_principal),
    settings: LineSettings = Depends(require_line_settings),
    secret: bytes = Depends(get_stream_secret),
    config: LineApiConfig = Depends(get_line_api_config),
) -> JSONResponse:
    sid = svc.check_station_id(payload.station_id)
    if not has_capability(principal, "line.read", sid, release_roles=settings.release_roles):
        raise LineApiError(403, "CAPABILITY_REQUIRED", "capability line.read required for this station")
    svc.require_configured_station(settings, sid)
    try:
        token = stream_tokens.mint(principal, sid, secret, ttl_s=config.stream_token_ttl_s)
    except PermissionError:
        raise LineApiError(403, "CAPABILITY_REQUIRED", "capability line.read required for this station") from None
    claims = stream_tokens.verify(token, secret, station_id=sid)
    if claims is None:  # pragma: no cover - a freshly minted token always verifies
        raise LineApiError(500, "STREAM_TOKEN_ERROR", "could not mint a stream token")
    expires = svc.epoch_to_iso(claims.expires_at)
    return JSONResponse(
        {"token": token, "expires_at": expires, "station_id": sid},
        headers={"Cache-Control": "no-store"},
    )


@router.get("/events/stream")
async def events_stream(
    request: Request,
    station_id: str | None = Query(default=None),
    token: str | None = Query(default=None, max_length=2048),
    after_id: int | None = Query(default=None, ge=0),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
    secret: bytes = Depends(get_stream_secret),
    device_store: DeviceTokenStore = Depends(get_device_token_store),
    config: LineApiConfig = Depends(get_line_api_config),
) -> Response:
    # EventSource cannot send headers: the station-bound stream token is the only auth.
    sid = svc.check_station_id(station_id)
    unauthorized = {"WWW-Authenticate": 'Bearer realm="sherman-line-stream"'}
    if not token:
        raise LineApiError(401, "STREAM_TOKEN_REQUIRED", "a stream token is required", headers=unauthorized)
    claims = stream_tokens.verify(token, secret, station_id=sid)
    if claims is None:
        raise LineApiError(401, "STREAM_TOKEN_INVALID", "invalid or expired stream token", headers=unauthorized)
    still_authorized = None
    if claims.kind == "device":
        subject = claims.subject
        if not await run_in_threadpool(device_store.is_active, subject):
            raise LineApiError(401, "STREAM_TOKEN_INVALID", "the device token was revoked", headers=unauthorized)
        still_authorized = lambda: device_store.is_active(subject)  # noqa: E731
    start = svc.parse_last_event_id(request.headers.get("last-event-id"), after_id)
    if start is None:
        start = await run_in_threadpool(store.last_event_id)
    # A stream never outlives its token: users are not re-checked mid-stream,
    # so the token TTL (<= 300 s) is the bound on a revoked or demoted user.
    remaining = max(0.0, float(claims.expires_at) - time.time())
    stream = svc.sse_event_stream(
        request,
        store,
        sid,
        start,
        config,
        principal_kind=claims.kind,
        still_authorized=still_authorized,
        settings=settings,
        max_lifetime_s=remaining,
    )
    return StreamingResponse(stream, media_type="text/event-stream", headers=dict(svc.SSE_HEADERS))


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------


@router.get("/quarantine")
def list_quarantine(
    station_id: str | None = Query(default=None),
    include_resolved: bool = Query(default=False),
    principal: Principal = Depends(require_capability("line.read", "station_id")),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
) -> JSONResponse:
    if station_id is not None:
        svc.check_station_id(station_id)
    items = store.list_quarantine(station_id=station_id, include_resolved=include_resolved)
    return JSONResponse(
        {"items": [svc.quarantine_view(i, settings, principal.kind) for i in items]},
        headers={"Cache-Control": _NO_CACHE},
    )


@router.get("/quarantine/held")
def held_scans(
    station_id: str = Query(...),
    principal: Principal = Depends(require_capability("line.read", "station_id")),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
    provider: RecipeProvider = Depends(get_recipe_provider),
) -> JSONResponse:
    """Station part picker: the station's open held-back scans with suggested parts."""
    sid = svc.require_configured_station(settings, station_id)
    items = picker.held_scans(store, provider, settings, sid, principal.kind)
    return JSONResponse(
        {"station_id": sid, "items": items, "can_operate": has_capability(principal, "line.operate")},
        headers={"Cache-Control": _NO_CACHE},
    )


@router.get("/released-parts")
def released_parts(
    station_id: str | None = Query(default=None),
    q: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=200),
    principal: Principal = Depends(require_capability("line.read", "station_id")),
    provider: RecipeProvider = Depends(get_recipe_provider),
) -> JSONResponse:
    """Released parts for the picker search (retired recipes never listed)."""
    if station_id is not None:
        svc.check_station_id(station_id)
    try:
        parts = picker.released_parts(provider, query=q, limit=limit)
    except picker.PickerError as exc:
        raise LineApiError(exc.status_code, exc.code, exc.message) from None
    return JSONResponse({"items": parts}, headers={"Cache-Control": _NO_CACHE})


@router.post("/quarantine/{scan_file_id}/assign")
def assign_quarantine(
    scan_file_id: str,
    payload: AssignRequest,
    principal: Principal = Depends(require_capability("line.operate")),
    store: LineStore = Depends(get_line_store),
    provider: RecipeProvider = Depends(get_recipe_provider),
) -> JSONResponse:
    return JSONResponse(
        svc.assign_quarantine(
            store,
            provider,
            scan_file_id,
            principal,
            recipe_id=payload.recipe_id,
            sku=payload.sku,
            declared_stage=payload.declared_stage,
            stage_index=payload.stage_index,
            idempotent=True,
        )
    )


@router.post("/quarantine/{scan_file_id}/dismiss")
def dismiss_quarantine(
    scan_file_id: str,
    payload: DismissRequest | None = None,
    principal: Principal = Depends(require_capability("line.operate")),
    store: LineStore = Depends(get_line_store),
) -> JSONResponse:
    reason = payload.reason if payload else None
    note = payload.note if payload else None
    if reason == "OTHER" and not (note or "").strip():
        raise LineApiError(422, "DISMISS_NOTE_REQUIRED", "reason OTHER needs a note")
    try:
        row = store.dismiss_quarantine(
            scan_file_id, principal.audit(), note=note, reason=reason, idempotent=True, actor_name=principal.username or None
        )
    except KeyError:
        raise LineApiError(404, "SCAN_NOT_FOUND", "no such scan") from None
    except QuarantineError as exc:
        code = "SCAN_NOT_QUARANTINED" if exc.code == "SCAN_STATE" else exc.code
        raise LineApiError(409, code, str(exc)) from None
    return JSONResponse({"scan_file_id": scan_file_id, "ingest_state": row["ingest_state"]})


# ---------------------------------------------------------------------------
# Status, readiness, metrics
# ---------------------------------------------------------------------------


@router.get("/status")
def line_status(
    principal: Principal = Depends(require_capability("line.operate")),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
    config: LineApiConfig = Depends(get_line_api_config),
    probes: StatusProbes = Depends(get_status_probes),
) -> JSONResponse:
    return JSONResponse(svc.status_report(settings, store, config, probes), headers={"Cache-Control": "no-store"})


@router.get("/health/ready")
def health_ready(
    settings: LineSettings | None = Depends(get_line_settings),
    config: LineApiConfig = Depends(get_line_api_config),
) -> JSONResponse:
    ready, checks = svc.cached_readiness(settings, config)
    return JSONResponse(
        {"ready": ready, "checks": checks},
        status_code=200 if ready else 503,
        headers={"Cache-Control": "no-store"},
    )


@router.get("/features")
def line_features(settings: LineSettings | None = Depends(get_line_settings)) -> JSONResponse:
    """Which UI areas apply (no auth, no paths): the main app hides the legacy
    analysis pages ("New Analysis", "Batch Processing", "Bend Inspection") while
    the line is enabled, because the route policy refuses their engine calls
    unless ``SHERMAN_LEGACY_ENGINE=1`` (audit 2026-10-19)."""
    return JSONResponse(
        {
            "line_enabled": settings is not None,
            "legacy_engine_enabled": settings is None or bool(settings.legacy_engine_enabled),
        },
        headers={"Cache-Control": "no-store"},
    )


@router.get("/metrics")
def metrics(
    principal: Principal = Depends(require_capability("metrics.read")),
    store: LineStore = Depends(get_line_store),
    registry: MetricsRegistry = Depends(get_metrics_registry),
) -> PlainTextResponse:
    return PlainTextResponse(
        svc.metrics_text(registry, store),
        media_type="text/plain; version=0.0.4; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Device tokens
# ---------------------------------------------------------------------------


@router.post("/devices", status_code=201)
def issue_device(
    payload: DeviceIssueRequest,
    principal: Principal = Depends(require_capability("devices.manage")),
    settings: LineSettings = Depends(require_line_settings),
    store: LineStore = Depends(get_line_store),
    device_store: DeviceTokenStore = Depends(get_device_token_store),
) -> JSONResponse:
    sid = svc.require_configured_station(settings, payload.station_id)
    label = payload.label.strip()
    try:
        token_id, token = device_store.issue(sid, label, created_by=principal.user_id)
    except DeviceTokenError as exc:
        raise LineApiError(422, "DEVICE_TOKEN_INVALID", str(exc)) from None
    try:
        store.append_event(
            "DEVICE_TOKEN_ISSUED", station_id=sid, principal=principal.audit(), payload={"token_id": token_id, "label": label}
        )
    except Exception:
        # No audit record, no token: revoke before refusing.
        device_store.revoke(token_id, by="system:audit-failed")
        raise LineApiError(503, "AUDIT_UNAVAILABLE", "the token could not be recorded and was revoked") from None
    return JSONResponse(
        {"token_id": token_id, "token": token, "station_id": sid, "label": label, "shown_once": True},
        status_code=201,
        headers={"Cache-Control": "no-store"},
    )


@router.get("/devices")
def list_devices(
    include_revoked: bool = Query(default=True),
    principal: Principal = Depends(require_capability("devices.manage")),
    device_store: DeviceTokenStore = Depends(get_device_token_store),
) -> JSONResponse:
    return JSONResponse({"devices": device_store.list(include_revoked=include_revoked)}, headers={"Cache-Control": "no-store"})


@router.delete("/devices/{token_id}")
def revoke_device(
    token_id: str,
    principal: Principal = Depends(require_capability("devices.manage")),
    store: LineStore = Depends(get_line_store),
    device_store: DeviceTokenStore = Depends(get_device_token_store),
) -> JSONResponse:
    station = next((d.get("station_id") for d in device_store.list() if d.get("token_id") == token_id), None)
    if not device_store.revoke(token_id, by=principal.user_id):
        raise LineApiError(404, "DEVICE_TOKEN_NOT_FOUND", "unknown or already revoked token")
    store.append_event("DEVICE_TOKEN_REVOKED", station_id=station, principal=principal.audit(), payload={"token_id": token_id})
    return JSONResponse({"token_id": token_id, "revoked": True})
