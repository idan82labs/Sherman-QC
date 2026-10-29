"""Line API service (C07): the logic behind ``/api/line``.

The router in :mod:`apps.api.routes.production_line` is thin; everything that
reads the durable line lives here:

- **Dependencies.** :func:`get_line_store`, :func:`get_recipe_provider`,
  :func:`get_line_api_config`, :func:`get_result_loader`,
  :func:`get_status_probes`, :func:`get_metrics_registry` and
  :func:`get_stream_secret`. Settings, the device store and capability checks
  come from :mod:`apps.production_line.security.dependencies` (C04). When the
  line is not configured every one of them answers 503.
- **Jobs and results.** Job rows come from the C02 :class:`LineStore`. A job's
  result is ``<data_root>/jobs/<job_id>/result/shopfloor_result.json``. It is
  served only after validation with the C01 model, an identity check against
  the job row (job id, station, recipe, compiled part, scan, summary) and,
  when the workspace still holds ``recipe.json`` and
  ``compiled/compiled_part.json``, the ENGINE_IO §4.3 cross-document check
  (:func:`contracts.production.consistency.check_result_against_compiled`).
  A result that fails any of these is never shown as a verdict.
- **Artifacts.** Only names in :data:`contracts.production.layout.ARTIFACT_ALLOWLIST`
  are served, and only from inside ``<data_root>/jobs/<job_id>`` (symlinks
  that leave the workspace are refused). Nothing is ever read from the legacy
  ``OUTPUT_DIR``.
- **Station feed.** :func:`station_feed_state` gathers the cheap inputs of
  ``StationFeedV1`` and its ETag; :func:`build_station_feed` adds the validated
  result only when the client's copy is stale.
- **Events.** JSON polling and an SSE generator (:func:`sse_event_stream`)
  that replays from ``Last-Event-ID`` and then polls the store.
- **Quarantine.** :func:`assign_quarantine` assigns a SKU (or that SKU's
  active recipe id) and enqueues; it is never weaker than the hotfolder's
  automatic resolution (retired, superseded and uncompiled recipes refused).
- **Redaction.** Store text shown to ``line.read`` principals (failure
  messages, event payloads, quarantine details) passes :func:`redact_tree`.
- **Retry, cancel, devices, status, readiness (cached) and metrics.**

Vocabulary follows ``contracts/production/ENGINE_IO.md``: summary, validity
and stage states are passed through unchanged; nothing here derives a verdict.
An unavailable or invalid result is reported as unavailable, never as PASS
or FAIL. Execution status (``failed``) is never a part verdict.

Import discipline: this module never imports the legacy server,
``apps.api.legacy_context`` or ``apps.production_line.recipes`` at import time.
The default recipe provider is resolved lazily inside
:func:`get_recipe_provider`.
"""
from __future__ import annotations

import errno
import functools
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import stat as stat_mod
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import anyio
from fastapi import Depends
from pydantic import ValidationError

from apps.production_line import health
from apps.production_line.ingest.ports import ActiveRecipeCatalog, ActiveRecipeRef, RecipeProvider
from apps.production_line.security import stream_tokens
from apps.production_line.security.dependencies import require_line_settings
from apps.production_line.security.principals import Principal, has_capability
from apps.production_line.settings import STATION_ID_RE, LineSettings, get_secret
from apps.production_line.store.line_store import LineStore, LineStoreError, QuarantineError
from apps.production_line.store.vocabulary import EXECUTION_STATUSES, JOB_KINDS
from apps.production_line.telemetry import MetricsRegistry, redact, render_prometheus
from contracts.production.compiled_part import CompiledPartV1
from contracts.production.consistency import check_result_against_compiled
from contracts.production.inspection_result import (
    ShopfloorInspectionResultV1,
    StationActiveV1,
    StationFeedV1,
    StationIssueV1,
    StationLatestCompletedV1,
)
from contracts.production.layout import ARTIFACT_ALLOWLIST, JOB_ID_RE, workspace_paths
from contracts.production.line_job import LineJobSummaryV1
from contracts.production.part_recipe import RECIPE_ID_PATTERN, SKU_PATTERN, PartRecipeV1

__all__ = [
    "API_PREFIX",
    "LineApiError",
    "LineApiConfig",
    "get_line_api_config",
    "get_line_store",
    "line_store_for",
    "get_recipe_provider",
    "get_metrics_registry",
    "get_result_loader",
    "get_status_probes",
    "get_stream_secret",
    "reset_line_api_caches",
    "LINE_REGISTRY",
    "artifact_url",
    "epoch_to_iso",
    "result_etag",
    "check_station_id",
    "require_configured_station",
    "authorize_job_read",
    "load_job_for",
    "LoadedResult",
    "ResultLoader",
    "job_summary",
    "available_artifacts",
    "if_none_match_matches",
    "FeedState",
    "station_feed_state",
    "build_station_feed",
    "ArtifactHandle",
    "open_artifact",
    "iter_artifact",
    "event_view",
    "sse_frame",
    "ClientConnection",
    "sse_event_stream",
    "parse_last_event_id",
    "sse_reset_frame",
    "SSE_RESET_EVENT",
    "redact_text",
    "redact_tree",
    "without_unverified_verdict",
    "cached_readiness",
    "assign_quarantine",
    "quarantine_view",
    "assignment_view",
    "StatusProbes",
    "readiness_checks",
    "status_report",
    "metrics_text",
    "parse_statuses",
    "parse_kind",
    "parse_before",
    "SSE_HEADERS",
    "RESULT_FILE_REL",
]

logger = logging.getLogger("apps.production_line.api")

API_PREFIX = "/api/line"
RESULT_FILE_REL = "result/shopfloor_result.json"
#: Headers every SSE response carries (no proxy buffering, no caching).
SSE_HEADERS: Mapping[str, str] = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_ARTIFACT_CACHE_CONTROL = "private, max-age=3600"
_FEED_RECENT = 10
_ISSUE_MESSAGE_MAX = 1000
_REDACTED_TEXT_MAX = 4000
_RECIPE_ID_RE = re.compile(RECIPE_ID_PATTERN)
_SKU_RE = re.compile(SKU_PATTERN)
_SCAN_FILE_ID_RE = re.compile(r"^scan_[0-9a-f]{24}$")
_FILENAME_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MEDIA_TYPES: Mapping[str, str] = {
    ".ply": "application/octet-stream",
    ".f32": "application/octet-stream",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".json": "application/json",
}
_ARTIFACT_CHUNK = 256 * 1024


class LineApiError(Exception):
    """A refusal with a stable machine-readable ``code``.

    The router turns it into ``{"detail": {"code", "message"}}`` with header
    ``X-Sherman-Error-Code`` (see the route class in the router module).
    """

    def __init__(self, status_code: int, code: str, message: str = "", headers: Mapping[str, str] | None = None):
        super().__init__(f"{status_code} {code}: {message}" if message else f"{status_code} {code}")
        self.status_code = int(status_code)
        self.code = code
        self.message = message
        self.headers = dict(headers or {})


# ---------------------------------------------------------------------------
# Configuration and dependencies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineApiConfig:
    """Tunables of the line API. Override :func:`get_line_api_config` in tests.

    ``heartbeat_s`` is the runner's heartbeat interval (C02 ``RunnerConfig``,
    15 s); a component is stale after ``stale_factor`` intervals. The SSE stream
    polls the store every ``sse_poll_s``, sends ``: heartbeat`` every
    ``sse_heartbeat_s`` and closes after ``sse_max_lifetime_s`` so the client
    reconnects with a fresh stream token (the token itself lives 300 s, and
    a stream never outlives its token). The public readiness probe is cached
    for ``readiness_ttl_s`` so hammering it cannot keep taking the DB write
    lock.
    """

    heartbeat_s: float = 15.0
    stale_factor: float = 3.0
    sse_poll_s: float = 0.5
    sse_heartbeat_s: float = 15.0
    sse_max_lifetime_s: float = 900.0
    sse_retry_ms: int = 3000
    sse_batch: int = 200
    stream_token_ttl_s: int = stream_tokens.MAX_TTL_S
    min_free_bytes: int = health.MIN_FREE_BYTES
    probe_ttl_s: float = 60.0
    python_probe_timeout_s: float = 20.0
    verify_bundle: bool = True
    readiness_ttl_s: float = 2.0

    def __post_init__(self) -> None:
        if self.heartbeat_s <= 0 or self.stale_factor < 1:
            raise ValueError("heartbeat_s must be > 0 and stale_factor >= 1")
        if self.sse_poll_s <= 0 or self.sse_heartbeat_s <= 0 or self.sse_max_lifetime_s <= 0:
            raise ValueError("SSE intervals must be > 0")
        if not 1 <= self.stream_token_ttl_s <= stream_tokens.MAX_TTL_S:
            raise ValueError(f"stream_token_ttl_s must be in 1..{stream_tokens.MAX_TTL_S}")
        if self.min_free_bytes < 0 or self.probe_ttl_s < 0 or not 1 <= self.sse_batch <= 1000:
            raise ValueError("invalid min_free_bytes, probe_ttl_s or sse_batch")
        if self.readiness_ttl_s < 0:
            raise ValueError("readiness_ttl_s must be >= 0")

    @property
    def stale_after_s(self) -> float:
        return self.heartbeat_s * self.stale_factor


_DEFAULT_CONFIG = LineApiConfig()


def get_line_api_config() -> LineApiConfig:
    return _DEFAULT_CONFIG


_stores: dict[Path, LineStore] = {}
_providers: dict[Path, RecipeProvider] = {}
_probes: dict[tuple[Path, LineApiConfig], StatusProbes] = {}
_readiness: dict[tuple[Path, LineApiConfig], tuple[float, bool, dict[str, str]]] = {}
_cache_lock = threading.Lock()
_readiness_compute_lock = threading.Lock()

#: Default metrics registry of the line API process. C11 may override
#: :func:`get_metrics_registry` to share the runner's registry.
LINE_REGISTRY = MetricsRegistry()


def line_store_for(settings: LineSettings) -> LineStore:
    """The (cached) :class:`LineStore` on ``settings.paths.db_line``."""
    path = settings.paths.db_line
    with _cache_lock:
        store = _stores.get(path)
        if store is None:
            store = LineStore(path)
            _stores[path] = store
        return store


def get_line_store(settings: LineSettings = Depends(require_line_settings)) -> LineStore:
    try:
        return line_store_for(settings)
    except (LineStoreError, sqlite3.Error, OSError) as exc:
        logger.error("line store unavailable: %s", type(exc).__name__)
        raise LineApiError(503, "LINE_STORE_UNAVAILABLE", "the line database cannot be opened") from None


def get_recipe_provider(settings: LineSettings = Depends(require_line_settings)) -> RecipeProvider:
    """C05's provider, imported lazily (``apps.production_line.recipes.resolver``)."""
    key = settings.data_root
    with _cache_lock:
        cached = _providers.get(key)
    if cached is not None:
        return cached
    try:
        from apps.production_line.recipes.resolver import build_recipe_provider  # type: ignore[import-not-found]
    except ImportError:
        raise LineApiError(503, "RECIPE_PROVIDER_UNAVAILABLE", "the recipe store is not installed") from None
    provider = build_recipe_provider(settings)
    with _cache_lock:
        _providers.setdefault(key, provider)
        return _providers[key]


def get_metrics_registry() -> MetricsRegistry:
    return LINE_REGISTRY


def get_stream_secret(settings: LineSettings = Depends(require_line_settings)) -> bytes:
    try:
        return get_secret(settings.paths, "stream_token")
    except Exception as exc:  # SettingsError or OSError: fail closed, no path in the reply
        logger.error("stream token secret unavailable: %s", type(exc).__name__)
        raise LineApiError(503, "STREAM_SECRET_UNAVAILABLE", "stream tokens are unavailable") from None


def reset_line_api_caches() -> None:
    """Forget cached stores, providers, probes and results (tests, reconfiguration)."""
    with _cache_lock:
        _stores.clear()
        _providers.clear()
        _probes.clear()
        _readiness.clear()
    _RESULT_LOADER.clear()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _parse_iso(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def epoch_to_iso(seconds: float) -> str:
    """Epoch seconds -> fixed-width UTC ISO-8601 (``...Z``)."""
    return _iso(datetime.fromtimestamp(float(seconds), tz=UTC))


def result_etag(job_id: str, loaded: LoadedResult) -> str:
    """ETag of a served result: changes whenever the result or its context files change."""
    return _etag_of(["result-v1", job_id, loaded.stat_key[1:]])


def artifact_url(job_id: str, name: str) -> str:
    return f"{API_PREFIX}/jobs/{job_id}/artifacts/{name}"


def check_station_id(station_id: str | None) -> str:
    """422 unless ``station_id`` matches ``[a-z0-9-]{1,32}`` (C04 STATION_ID_RE)."""
    if not isinstance(station_id, str) or not STATION_ID_RE.fullmatch(station_id):
        raise LineApiError(422, "STATION_ID_INVALID", "station_id must match [a-z0-9-]{1,32}")
    return station_id


def require_configured_station(settings: LineSettings, station_id: str | None) -> str:
    """The station id when it is configured in ``SHERMAN_LINE_STATIONS``; else 404."""
    sid = check_station_id(station_id)
    if settings.station(sid) is None:
        raise LineApiError(404, "STATION_UNKNOWN", "station is not configured on this line")
    return sid


def _data_root_spellings(settings: LineSettings) -> list[str]:
    """The data root as configured and resolved, longest first (so a resolved
    ``/private/tmp/x`` is replaced whole before its ``/tmp/x`` spelling)."""
    roots = {str(settings.data_root), os.path.realpath(str(settings.data_root))}
    return sorted((r for r in roots if r and r != "/"), key=len, reverse=True)


def redact_text(text: str, settings: LineSettings, limit: int = _REDACTED_TEXT_MAX) -> str:
    """Credentials (C04 :func:`telemetry.redact`) and the data-root path removed."""
    out = redact(str(text))
    for root in _data_root_spellings(settings):
        out = out.replace(root, "<data_root>")
    return out[:limit]


def redact_tree(value: Any, settings: LineSettings, _depth: int = 0) -> Any:
    """:func:`redact_text` applied to every string inside a JSON-like value.

    Used for everything a ``line.read`` principal (kiosk devices included) sees
    of store text: job failure records, event payloads, quarantine details.
    Nothing upstream (runner, hotfolder, store) redacts that text. Keys are
    kept; values nested deeper than 32 levels are dropped.
    """
    if isinstance(value, str):
        return redact_text(value, settings)
    if _depth >= 32:
        return None
    if isinstance(value, Mapping):
        return {k: redact_tree(v, settings, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_tree(v, settings, _depth + 1) for v in value]
    return value


def _clean_message(text: str | None, settings: LineSettings, limit: int = _ISSUE_MESSAGE_MAX) -> str:
    """A one-line display message: :func:`redact_text`, whitespace collapsed."""
    return " ".join(redact_text(str(text or ""), settings, limit=1 << 20).split())[:limit]


def authorize_job_read(principal: Principal, job: Mapping[str, Any], settings: LineSettings) -> None:
    """``line.read`` on the job's station. A device sees only its own station's
    jobs; any other job is reported as missing (404), never as forbidden."""
    if has_capability(principal, "line.read", job.get("station_id"), release_roles=settings.release_roles):
        return
    if principal.kind == "device":
        raise LineApiError(404, "JOB_NOT_FOUND", "no such job")
    raise LineApiError(403, "CAPABILITY_REQUIRED", "capability line.read required")


def load_job_for(principal: Principal, store: LineStore, job_id: str, settings: LineSettings) -> dict[str, Any]:
    """The store row of ``job_id`` if ``principal`` may read it (404 otherwise)."""
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        raise LineApiError(404, "JOB_NOT_FOUND", "no such job")
    job = store.get_job(job_id)
    if job is None:
        raise LineApiError(404, "JOB_NOT_FOUND", "no such job")
    authorize_job_read(principal, job, settings)
    return job


def if_none_match_matches(header: str | None, etag: str) -> bool:
    """RFC 9110 weak comparison of ``If-None-Match`` against ``etag``."""
    if not header:
        return False
    target = etag[2:] if etag.startswith("W/") else etag
    for raw in header.split(","):
        token = raw.strip()
        if token == "*":
            return True
        if token.startswith("W/"):
            token = token[2:]
        if token == target:
            return True
    return False


def _etag_of(parts: Iterable[Any]) -> str:
    digest = hashlib.sha256(json.dumps(list(parts), sort_keys=True, separators=(",", ":"), default=str).encode())
    return '"' + digest.hexdigest()[:32] + '"'


def _stat_key(path: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not stat_mod.S_ISREG(st.st_mode):
        return None
    return (int(st.st_size), int(st.st_mtime_ns))


def _workspace_root(settings: LineSettings, job_id: str) -> Path:
    return Path(workspace_paths(settings.paths.jobs, job_id).root)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedResult:
    """Outcome of loading a job's ``shopfloor_result.json``.

    Exactly one of ``result`` / ``code`` is set. ``code`` is RESULT_MISSING,
    RESULT_INVALID (the C01 model refused it), RESULT_INCONSISTENT (identity or
    §4.3 cross-document check failed) or CONTEXT_INVALID (the workspace recipe
    or compiled part does not validate). ``check`` names what was verified:
    ``model+identity`` or ``model+identity+consistency``.
    """

    result: ShopfloorInspectionResultV1 | None
    code: str | None = None
    message: str = ""
    check: str = ""
    stat_key: tuple[Any, ...] = ()


def _identity_problems(result: ShopfloorInspectionResultV1, job: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    ctx = result.context
    if ctx.job_id is not None and ctx.job_id != job["job_id"]:
        problems.append("CONTEXT_JOB_MISMATCH")
    if ctx.station_id is not None and ctx.station_id != job.get("station_id"):
        problems.append("CONTEXT_STATION_MISMATCH")
    if result.recipe.recipe_id != job["recipe_id"] or result.recipe.recipe_sha256 != job["recipe_sha256"]:
        problems.append("RECIPE_MISMATCH")
    if result.sku != job["sku"]:
        problems.append("SKU_MISMATCH")
    compiled = job.get("compiled_part_sha256")
    if compiled is not None and result.engine.compiled_part_sha256 != compiled:
        problems.append("COMPILED_PART_MISMATCH")
    scan = job.get("scan_sha256")
    if scan is not None and result.scan.sha256 != scan:
        problems.append("SCAN_MISMATCH")
    summary = job.get("result_summary") or {}
    if summary:
        if summary.get("summary_state") != result.summary.state:
            problems.append("SUMMARY_MISMATCH")
        if summary.get("inspection_validity") != result.inspection_validity.state:
            problems.append("VALIDITY_MISMATCH")
    return sorted(set(problems))


class ResultLoader:
    """Validates and caches job results keyed by the files' (size, mtime_ns).

    Thread safe; at most ``capacity`` entries. A rewritten file (PDF re-render,
    repair) has a new stat key and is validated again.
    """

    def __init__(self, capacity: int = 128) -> None:
        self.capacity = int(capacity)
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[Any, ...], LoadedResult] = OrderedDict()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stat_key(self, settings: LineSettings, job: Mapping[str, Any]) -> tuple[Any, ...]:
        ws = workspace_paths(settings.paths.jobs, job["job_id"])
        return (
            job["job_id"],
            _stat_key(Path(ws.root) / RESULT_FILE_REL),
            _stat_key(Path(ws.recipe)),
            _stat_key(Path(ws.compiled_dir) / "compiled_part.json"),
            json.dumps(job.get("result_summary"), sort_keys=True),
        )

    def load(self, settings: LineSettings, job: Mapping[str, Any]) -> LoadedResult:
        key = self.stat_key(settings, job)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        loaded = self._load_uncached(settings, job, key)
        if key[1] is not None:  # never cache a missing file: it may appear any moment
            with self._lock:
                self._cache[key] = loaded
                self._cache.move_to_end(key)
                while len(self._cache) > self.capacity:
                    self._cache.popitem(last=False)
        return loaded

    def _load_uncached(self, settings: LineSettings, job: Mapping[str, Any], key: tuple[Any, ...]) -> LoadedResult:
        ws = workspace_paths(settings.paths.jobs, job["job_id"])
        path = Path(ws.root) / RESULT_FILE_REL
        if key[1] is None:
            return LoadedResult(None, "RESULT_MISSING", "no result has been written for this job", stat_key=key)
        try:
            data = _read_inside(Path(ws.root), path)
            result = ShopfloorInspectionResultV1.model_validate_json(data)
        except (ValidationError, ValueError, OSError) as exc:
            logger.error("job %s: result does not validate: %s", job["job_id"], type(exc).__name__)
            return LoadedResult(None, "RESULT_INVALID", "the result file does not validate", stat_key=key)
        problems = _identity_problems(result, job)
        check = "model+identity"
        recipe_path, compiled_path = Path(ws.recipe), Path(ws.compiled_dir) / "compiled_part.json"
        if not problems and key[2] is not None and key[3] is not None:
            try:
                recipe = PartRecipeV1.model_validate_json(_read_inside(Path(ws.root), recipe_path))
                compiled = CompiledPartV1.model_validate_json(_read_inside(Path(ws.root), compiled_path))
            except (ValidationError, ValueError, OSError) as exc:
                logger.error("job %s: workspace recipe/compiled part invalid: %s", job["job_id"], type(exc).__name__)
                return LoadedResult(
                    None, "CONTEXT_INVALID", "the job's recipe or compiled part does not validate", stat_key=key
                )
            problems = check_result_against_compiled(result, compiled, recipe)
            check = "model+identity+consistency"
        if problems:
            logger.error("job %s: result inconsistent: %s", job["job_id"], ",".join(problems))
            return LoadedResult(None, "RESULT_INCONSISTENT", ", ".join(problems)[:500], check=check, stat_key=key)
        return LoadedResult(result, None, "", check=check, stat_key=key)


_RESULT_LOADER = ResultLoader()


def get_result_loader() -> ResultLoader:
    return _RESULT_LOADER


#: The job-level zero-shot overlay (``sherman_zeroshot_overlay_v1``, runner step 3b).
ZEROSHOT_OVERLAY_REL = "zeroshot/zeroshot_overlay_v1.json"


@dataclass(frozen=True)
class LoadedOverlay:
    """Outcome of loading a job's zero-shot overlay. Exactly one of ``overlay`` / ``code`` is set:
    OVERLAY_MISSING (no overlay: the zs step was off, failed or found nothing to merge), OVERLAY_INVALID
    (the document does not validate) or OVERLAY_INCONSISTENT (it names another scan, another E3 result or
    another summary state than the job)."""

    overlay: dict[str, Any] | None
    code: str | None = None
    message: str = ""
    etag: str = ""


def load_zeroshot_overlay(settings: LineSettings, job: Mapping[str, Any]) -> LoadedOverlay:
    """Validate ``<job>/zeroshot/zeroshot_overlay_v1.json`` against its model and the job (display only).

    The overlay never changes the job's verdict fields; it is served next to the result, not inside it."""
    from domains.production.zeroshot.overlay import ZeroshotOverlayV1, identity_problems

    root = _workspace_root(settings, job["job_id"])
    path = root / ZEROSHOT_OVERLAY_REL
    key = _stat_key(path)
    if key is None:
        return LoadedOverlay(None, "OVERLAY_MISSING", "no zero-shot overlay for this job")
    try:
        data = _read_inside(root, path, max_bytes=8 * 1024 * 1024)
        ov = ZeroshotOverlayV1.model_validate_json(data)
    except (OSError, ValueError) as exc:
        return LoadedOverlay(None, "OVERLAY_INVALID", f"overlay does not validate: {type(exc).__name__}")
    engine_sha: str | None = None
    engine_result = Path(workspace_paths(settings.paths.jobs, job["job_id"]).engine_dir) / "result.json"
    if _stat_key(engine_result) is not None:
        try:
            engine_sha = hashlib.sha256(_read_inside(root, engine_result)).hexdigest()
        except OSError:
            engine_sha = None
    summary = job.get("result_summary") or {}
    problems = identity_problems(ov, scan_sha256=job.get("scan_sha256"),
                                 e3_summary_state=summary.get("summary_state"), engine_result_sha256=engine_sha)
    if problems:
        return LoadedOverlay(None, "OVERLAY_INCONSISTENT", ", ".join(problems))
    return LoadedOverlay(ov.model_dump(mode="json", by_alias=True), etag=_etag_of(["zs-overlay-v1", job["job_id"], key]))


def _read_inside(root: Path, path: Path, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    """Read ``path`` only if it resolves to a regular file inside ``root``."""
    real_root = os.path.realpath(root)
    real = os.path.realpath(path)
    if os.path.commonpath([real_root, real]) != real_root:
        raise OSError(errno.EACCES, "path leaves the job workspace")
    fd = os.open(real, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        st = os.fstat(fd)
        if not stat_mod.S_ISREG(st.st_mode):
            raise OSError(errno.EINVAL, "not a regular file")
        if st.st_size > max_bytes:
            raise OSError(errno.EFBIG, "file too large")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


class _PartNames:
    """``part_name`` of a job, read from its workspace ``recipe.json`` (cached)."""

    def __init__(self, capacity: int = 512) -> None:
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[Any, ...], str | None] = OrderedDict()
        self.capacity = capacity

    def get(self, settings: LineSettings, job_id: str) -> str | None:
        ws = workspace_paths(settings.paths.jobs, job_id)
        key = (job_id, _stat_key(Path(ws.recipe)))
        if key[1] is None:
            return None
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        name: str | None = None
        try:
            doc = json.loads(_read_inside(Path(ws.root), Path(ws.recipe), max_bytes=16 * 1024 * 1024))
            raw = doc.get("part_name") if isinstance(doc, dict) else None
            name = str(raw)[:255] if raw else None
        except (OSError, ValueError):
            name = None
        with self._lock:
            self._cache[key] = name
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
        return name


_PART_NAMES = _PartNames()


def job_summary(
    settings: LineSettings,
    store: LineStore,
    job: Mapping[str, Any],
    *,
    scan: Mapping[str, Any] | None = None,
    part_name: str | None = None,
) -> LineJobSummaryV1:
    """``LineJobSummaryV1`` of a store job row (scan name/serial joined in)."""
    if scan is None and job.get("scan_file_id"):
        scan = store.get_scan_file(job["scan_file_id"])
    parse = (scan or {}).get("parse") or {}
    summary = job.get("result_summary") or {}
    failure = job.get("failure") or {}
    return LineJobSummaryV1(
        job_id=job["job_id"],
        kind=job["kind"],
        station_id=job.get("station_id"),
        sku=job["sku"],
        part_name=part_name if part_name is not None else _PART_NAMES.get(settings, job["job_id"]),
        recipe_id=job["recipe_id"],
        recipe_version=job["recipe_version"],
        scan_file_name=(scan or {}).get("original_name"),
        serial=parse.get("serial"),
        execution_status=job["execution_status"],
        stage=job["stage"],
        progress=job["progress"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        failure_code=failure.get("code"),
        summary_state=summary.get("summary_state"),
        inspection_validity=summary.get("inspection_validity"),
        stage_status=summary.get("stage_status"),
    )


def available_artifacts(settings: LineSettings, job_id: str) -> dict[str, str]:
    """Allow-listed artifacts that exist for the job: ``{name: url}``."""
    root = _workspace_root(settings, job_id)
    out: dict[str, str] = {}
    for name, rel in ARTIFACT_ALLOWLIST.items():
        if _stat_key(root / rel) is not None:
            out[name] = artifact_url(job_id, name)
    return out


# ---------------------------------------------------------------------------
# Station feed
# ---------------------------------------------------------------------------


@dataclass
class FeedState:
    """The cheap part of a station feed: the rows it is built from and its ETag."""

    station_id: str
    etag: str
    latest_completed: dict[str, Any] | None
    active: dict[str, Any] | None
    issue: StationIssueV1 | None
    recent: list[dict[str, Any]]
    scans: dict[str, dict[str, Any] | None] = field(default_factory=dict)


def _scan_for(store: LineStore, cache: dict[str, dict[str, Any] | None], job: Mapping[str, Any]) -> dict[str, Any] | None:
    sid = job.get("scan_file_id")
    if not sid:
        return None
    if sid not in cache:
        cache[sid] = store.get_scan_file(sid)
    return cache[sid]


def _quarantine_message(item: Mapping[str, Any], settings: LineSettings) -> str:
    detail = item.get("quarantine_detail") or {}
    parts: list[str] = []
    if isinstance(detail, Mapping):
        for key in ("resolve_reason", "detail"):
            value = detail.get(key)
            if value:
                parts.append(str(value))
        sku = detail.get("sku")
        if sku:
            parts.append(f"SKU {sku}")
    elif detail:
        parts.append(str(detail))
    return _clean_message("; ".join(parts), settings)


def _latest_issue(
    settings: LineSettings,
    store: LineStore,
    station_id: str,
    latest_completed: Mapping[str, Any] | None,
    scans: dict[str, dict[str, Any] | None],
) -> tuple[StationIssueV1 | None, str | None]:
    """Newer of the newest open quarantine and the newest failed job.

    A failed job counts only if it finished after the latest completed
    inspection; an older failure is history (it stays in ``recent``). An open
    quarantine always counts until it is assigned or dismissed.
    """
    candidates: list[tuple[datetime, str, StationIssueV1]] = []
    open_q = store.list_quarantine(station_id=station_id)
    if open_q:
        item = open_q[0]
        at = _parse_iso(item.get("quarantined_at")) or _utc_now()
        scan = item["scan_file"]
        issue = StationIssueV1(
            kind="QUARANTINE",
            at=item.get("quarantined_at") or _iso(at),
            file_name=str(scan.get("original_name") or "")[:255] or None,
            reason_code=item["quarantine_reason"],
            message=_quarantine_message(item, settings),
        )
        candidates.append((at, f"QUARANTINE:{scan['scan_file_id']}:{item.get('quarantined_at')}", issue))
    failed = store.latest_job_for_station(station_id, ("failed",))
    if failed is not None and failed.get("finished_at"):
        failed_at = _parse_iso(failed["finished_at"])
        done_at = _parse_iso((latest_completed or {}).get("finished_at"))
        if failed_at is not None and (done_at is None or failed_at > done_at):
            scan = _scan_for(store, scans, failed)
            failure = failed.get("failure") or {}
            issue = StationIssueV1(
                kind="JOB_FAILED",
                at=failed["finished_at"],
                file_name=(str(scan.get("original_name"))[:255] if scan and scan.get("original_name") else None),
                reason_code=str(failure.get("code") or "INTERNAL_ERROR")[:64],
                message=_clean_message(failure.get("message"), settings),
            )
            candidates.append((failed_at, f"JOB_FAILED:{failed['job_id']}", issue))
    if not candidates:
        return None, None
    candidates.sort(key=lambda c: c[0])
    _, issue_id, issue = candidates[-1]
    return issue, issue_id


def _active_job(store: LineStore, station_id: str) -> dict[str, Any] | None:
    """The station's running inspect job (latest started), else its next queued one."""
    running = [j for j in store.list_jobs(station_id=station_id, statuses=["running"], limit=20) if j["kind"] == "inspect"]
    if running:
        return max(running, key=lambda j: (j.get("started_at") or "", j["created_at"]))
    queued = [j for j in store.list_jobs(station_id=station_id, statuses=["queued"], limit=200) if j["kind"] == "inspect"]
    if queued:
        return min(queued, key=lambda j: (-int(j.get("priority") or 0), j["created_at"]))
    return None


def station_feed_state(
    settings: LineSettings, store: LineStore, loader: ResultLoader, station_id: str
) -> FeedState:
    """Gather the feed rows and compute its ETag without loading any result.

    ETag = weak ``W/"<sha256 prefix>"`` over (latest completed job id +
    finished_at + result/context file stats + recorded summary, active job id
    + status + stage + progress//5, latest issue id, the recent jobs' ids and
    statuses). It is weak because the body also carries ``server_time``. A
    result-unavailable issue (see :func:`build_station_feed`) is derived from
    the latest completed job's files, which the ETag already covers.
    """
    scans: dict[str, dict[str, Any] | None] = {}
    completed = store.latest_job_for_station(station_id, ("completed",))
    if completed is not None and completed["kind"] != "inspect":
        completed = next(
            iter(store.list_jobs(station_id=station_id, statuses=["completed"], kind="inspect", limit=1)), None
        )
    active = _active_job(store, station_id)
    issue, issue_id = _latest_issue(settings, store, station_id, completed, scans)
    recent = store.list_jobs(station_id=station_id, kind="inspect", limit=_FEED_RECENT)
    parts: list[Any] = [
        "feed-v1",
        station_id,
        None
        if completed is None
        else [completed["job_id"], completed.get("finished_at"), loader.stat_key(settings, completed)[1:]],
        None
        if active is None
        else [active["job_id"], active["execution_status"], active["stage"], int(active["progress"]) // 5],
        issue_id,
        [[j["job_id"], j["execution_status"]] for j in recent],
    ]
    return FeedState(
        station_id=station_id,
        etag="W/" + _etag_of(parts),
        latest_completed=completed,
        active=active,
        issue=issue,
        recent=recent,
        scans=scans,
    )


def build_station_feed(
    settings: LineSettings, store: LineStore, loader: ResultLoader, state: FeedState
) -> StationFeedV1:
    """The full ``StationFeedV1`` for ``state``.

    ``latest_completed`` is set only when the newest completed inspection has
    a validated result. If that result is missing or invalid, the feed shows
    *no* latest result rather than an older part's verdict, and:

    - that job's ``recent`` entry has its verdict fields (``summary_state``,
      ``inspection_validity``, ``stage_status``) cleared, so no unverified
      verdict chip can be drawn from it;
    - unless a newer issue exists, ``latest_issue`` explains the gap:
      ``kind=JOB_FAILED`` (the job did not yield a usable result) with
      ``reason_code`` RESULT_MISSING / RESULT_INVALID / RESULT_INCONSISTENT /
      CONTEXT_INVALID. StationIssueV1 has no dedicated kind for this yet.
    """
    latest: StationLatestCompletedV1 | None = None
    issue = state.issue
    unverified_job_id: str | None = None
    if state.latest_completed is not None:
        job = state.latest_completed
        loaded = loader.load(settings, job)
        if loaded.result is not None:
            latest = StationLatestCompletedV1(
                job=job_summary(
                    settings, store, job, scan=_scan_for(store, state.scans, job), part_name=loaded.result.part_name
                ),
                result=loaded.result,
                artifacts=available_artifacts(settings, job["job_id"]),
            )
        else:
            logger.warning("station %s: latest result of %s unavailable: %s", state.station_id, job["job_id"], loaded.code)
            unverified_job_id = job["job_id"]
            at = job.get("finished_at") or job["created_at"]
            scan = _scan_for(store, state.scans, job)
            gap = StationIssueV1(
                kind="JOB_FAILED",
                at=at,
                file_name=(str(scan.get("original_name"))[:255] if scan and scan.get("original_name") else None),
                reason_code=loaded.code or "RESULT_INVALID",
                message="The inspection finished but its result could not be verified, so no verdict is shown.",
            )
            issue_at = _parse_iso(issue.at) if issue is not None else None
            gap_at = _parse_iso(at)
            if issue is None or (issue_at is not None and gap_at is not None and gap_at >= issue_at):
                issue = gap
    active = None
    if state.active is not None:
        active = StationActiveV1(job=job_summary(settings, store, state.active, scan=_scan_for(store, state.scans, state.active)))
    recent = tuple(
        _without_verdict(summary) if summary.job_id == unverified_job_id else summary
        for summary in (job_summary(settings, store, j, scan=_scan_for(store, state.scans, j)) for j in state.recent)
    )
    return StationFeedV1(
        station_id=state.station_id,
        etag=state.etag,
        server_time=_iso(_utc_now()),
        latest_completed=latest,
        active=active,
        latest_issue=issue,
        recent=recent,
    )


def _without_verdict(summary: LineJobSummaryV1) -> LineJobSummaryV1:
    """``summary`` with the recorded verdict removed (its result did not verify)."""
    return summary.model_copy(update={"summary_state": None, "inspection_validity": None, "stage_status": None})


def without_unverified_verdict(summary: LineJobSummaryV1, result_state: str | None) -> LineJobSummaryV1:
    """For a completed inspection whose result is not AVAILABLE, drop the verdict."""
    if result_state is None or result_state == "AVAILABLE":
        return summary
    return _without_verdict(summary)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


@dataclass
class ArtifactHandle:
    """An opened, verified artifact. The caller streams ``fd`` and closes it."""

    fd: int
    name: str
    size: int
    mtime_ns: int
    media_type: str
    etag: str
    headers: dict[str, str]

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


def _pdf_filename(job: Mapping[str, Any], scan: Mapping[str, Any] | None) -> str:
    serial = ((scan or {}).get("parse") or {}).get("serial")
    token = _FILENAME_UNSAFE_RE.sub("-", str(serial or "")).strip("-.")[:64]
    if not token:
        token = job["job_id"][5:13]
    when = _parse_iso(job.get("finished_at") or job.get("created_at")) or _utc_now()
    sku = _FILENAME_UNSAFE_RE.sub("-", str(job["sku"])).strip("-.") or "part"
    return f"{sku}_{token}_{when:%Y%m%d}.pdf"


def open_artifact(
    settings: LineSettings, store: LineStore, job: Mapping[str, Any], name: str
) -> ArtifactHandle:
    """Open an allow-listed artifact of a completed job.

    404 for a name outside :data:`ARTIFACT_ALLOWLIST`, a file that does not
    exist, or a path that resolves outside ``<data_root>/jobs/<job_id>``.
    409 while the job is not completed (files may be mid-write or stale).
    """
    rel = ARTIFACT_ALLOWLIST.get(name)
    if rel is None:
        raise LineApiError(404, "ARTIFACT_NOT_FOUND", "no such artifact")
    if job["execution_status"] != "completed":
        raise LineApiError(409, "JOB_NOT_COMPLETED", f"job is {job['execution_status']}")
    root = _workspace_root(settings, job["job_id"])
    real_root = os.path.realpath(root)
    real = os.path.realpath(root / rel)
    if os.path.commonpath([real_root, real]) != real_root or not real.startswith(real_root + os.sep):
        logger.error("job %s: artifact %s resolves outside its workspace; refused", job["job_id"], name)
        raise LineApiError(404, "ARTIFACT_NOT_FOUND", "no such artifact")
    try:
        fd = os.open(real, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        raise LineApiError(404, "ARTIFACT_NOT_FOUND", "no such artifact") from None
    try:
        st = os.fstat(fd)
        if not stat_mod.S_ISREG(st.st_mode):
            raise LineApiError(404, "ARTIFACT_NOT_FOUND", "no such artifact")
        size, mtime_ns = int(st.st_size), int(st.st_mtime_ns)
        suffix = Path(rel).suffix.lower()
        media_type = _MEDIA_TYPES.get(suffix, "application/octet-stream")
        etag = _etag_of(["artifact-v1", job["job_id"], name, size, mtime_ns])
        headers = {
            "ETag": etag,
            "Cache-Control": _ARTIFACT_CACHE_CONTROL,
            "X-Content-Type-Options": "nosniff",
        }
        if suffix == ".f32":
            if size % 4 != 0:
                logger.error("job %s: %s has %d bytes, not a multiple of 4", job["job_id"], name, size)
                raise LineApiError(500, "ARTIFACT_INVALID", "deviation file is not a whole number of float32 values")
            headers.update(
                {
                    "X-Deviation-Count": str(size // 4),
                    "X-Deviation-Units": "mm",
                    # ENGINE_IO §4.1 vocabulary (supersedes the survey's SIGNED_ALONG_CAD_NORMAL).
                    "X-Deviation-Sign": "SIGNED_ALONG_STAGE_MODEL_OUTWARD_NORMAL",
                    "X-Deviation-Encoding": "float32_le_nan_missing",
                }
            )
        elif suffix == ".pdf":
            scan = store.get_scan_file(job["scan_file_id"]) if job.get("scan_file_id") else None
            filename = _pdf_filename(job, scan)
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return ArtifactHandle(fd, name, size, mtime_ns, media_type, etag, headers)
    except BaseException:
        os.close(fd)
        raise


def iter_artifact(handle: ArtifactHandle) -> Iterable[bytes]:
    """Stream exactly ``handle.size`` bytes from the opened fd, then close it."""
    try:
        remaining = handle.size
        while remaining > 0:
            chunk = os.read(handle.fd, min(_ARTIFACT_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# Events and SSE
# ---------------------------------------------------------------------------


def event_view(event: Mapping[str, Any], principal_kind: str, settings: LineSettings | None = None) -> dict[str, Any]:
    """An event as served. Devices do not see who performed an action, and
    (with ``settings``) every string in the payload is redacted
    (:func:`redact_tree`): failure messages can carry tokens or paths."""
    out = dict(event)
    if principal_kind == "device":
        out["principal"] = None
    if settings is not None and out.get("payload") is not None:
        out["payload"] = redact_tree(out["payload"], settings)
    return out


#: SSE event name of the cursor-reset frame (see :func:`sse_event_stream`).
SSE_RESET_EVENT = "RESET"


def sse_reset_frame(last_id: int, requested: int) -> str:
    """Tells the client its cursor is ahead of the store (DB restored or
    rebuilt): refetch the feed; the stream continues after ``last_id``. The
    frame carries ``id: last_id`` so EventSource's ``lastEventId`` follows."""
    data = json.dumps(
        {"kind": SSE_RESET_EVENT, "reason": "CURSOR_AHEAD_OF_STORE", "last_id": int(last_id), "requested_after_id": int(requested)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"id: {int(last_id)}\nevent: {SSE_RESET_EVENT}\ndata: {data}\n\n"


def sse_frame(event: Mapping[str, Any]) -> str:
    """One SSE frame: ``id:``, ``event:`` and a single-line ``data:`` JSON."""
    data = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    kind = re.sub(r"[^A-Z0-9_]", "", str(event.get("kind", "message"))) or "message"
    return f"id: {int(event['id'])}\nevent: {kind}\ndata: {data}\n\n"


def parse_last_event_id(header: str | None, after_id: int | None) -> int | None:
    """``Last-Event-ID`` wins over ``after_id``; an unparsable header is ignored."""
    if header is not None:
        text = header.strip()
        if text.isdigit() and len(text) <= 18:
            return int(text)
    if after_id is not None:
        return max(0, int(after_id))
    return None


class ClientConnection(Protocol):
    """What the SSE generator needs from the request (a Starlette ``Request``)."""

    async def is_disconnected(self) -> bool: ...


async def sse_event_stream(
    request: ClientConnection,
    store: LineStore,
    station_id: str,
    start_after: int,
    config: LineApiConfig,
    *,
    principal_kind: str,
    still_authorized: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
    settings: LineSettings | None = None,
    max_lifetime_s: float | None = None,
) -> AsyncIterator[str]:
    """Replay events after ``start_after``, then poll the store.

    Sends ``retry:`` first, every event of ``station_id`` as an SSE frame
    (payload redacted when ``settings`` is given), ``: heartbeat`` every
    ``config.sse_heartbeat_s`` and stops on client disconnect, after
    ``min(config.sse_max_lifetime_s, max_lifetime_s)`` (the router passes the
    stream token's remaining life), or when ``still_authorized`` (checked at
    each heartbeat, e.g. device revocation) returns false.

    A ``start_after`` beyond the store's newest event id (a restored or
    rebuilt DB restarts ids) would silently starve the client, so the stream
    first sends :func:`sse_reset_frame` and continues after the newest id.
    """
    yield f"retry: {int(config.sse_retry_ms)}\n\n"
    lifetime = config.sse_max_lifetime_s if max_lifetime_s is None else min(config.sse_max_lifetime_s, max_lifetime_s)
    cursor = int(start_after)
    newest = int(await anyio.to_thread.run_sync(store.last_event_id))
    if cursor > newest:
        logger.warning("station %s: SSE cursor %d is ahead of the store (%d); reset", station_id, cursor, newest)
        yield sse_reset_frame(newest, cursor)
        cursor = newest
    started = clock()
    last_beat = started
    while True:
        if await request.is_disconnected():
            return
        events = await anyio.to_thread.run_sync(
            functools.partial(store.events_after, cursor, station_id=station_id, limit=config.sse_batch)
        )
        for event in events:
            cursor = int(event["id"])
            yield sse_frame(event_view(event, principal_kind, settings))
        if len(events) >= config.sse_batch:
            continue  # still replaying a backlog
        now = clock()
        if now - started >= lifetime:
            return
        if now - last_beat >= config.sse_heartbeat_s:
            if still_authorized is not None:
                ok = await anyio.to_thread.run_sync(still_authorized)
                if not ok:
                    return
            yield ": heartbeat\n\n"
            last_beat = now
        wait = min(
            config.sse_poll_s,
            max(0.0, config.sse_heartbeat_s - (now - last_beat)),
            max(0.0, lifetime - (now - started)),
        )
        await anyio.sleep(max(wait, 0.005))


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------


def quarantine_view(item: Mapping[str, Any], settings: LineSettings, principal_kind: str) -> dict[str, Any]:
    """A ``list_quarantine`` entry plus a display ``message``; the stored
    ``quarantine_detail`` strings are redacted (:func:`redact_tree`)."""
    out = dict(item)
    out["message"] = _quarantine_message(item, settings)
    if out.get("quarantine_detail") is not None:
        out["quarantine_detail"] = redact_tree(out["quarantine_detail"], settings)
    if principal_kind == "device":
        out["resolved_by"] = None
    return out


def assignment_view(store: LineStore, job: Mapping[str, Any], principal_kind: str) -> dict[str, Any] | None:
    """How the job's scan got its part: ``None`` for the hotfolder's automatic match,
    else ``{by_operator, assigned_at, from_reason, assigned_by}`` (``assigned_by``
    hidden from devices, as in :func:`quarantine_view`)."""
    sid = job.get("scan_file_id")
    if not sid:
        return None
    meta = store.get_scan_meta(sid) or {}
    if meta.get("resolution") != "assigned":
        return None
    return {
        "by_operator": True,
        "assigned_at": meta.get("resolved_at"),
        "from_reason": meta.get("quarantine_reason"),
        "assigned_by": None if principal_kind == "device" else meta.get("resolved_by"),
    }


def _active_refs(catalog: ActiveRecipeCatalog) -> list[ActiveRecipeRef]:
    try:
        return list(catalog.list_active_recipes())
    except Exception as exc:
        logger.error("recipe catalog failed: %s", type(exc).__name__)
        raise LineApiError(503, "RECIPE_PROVIDER_ERROR", "the recipe store could not list active recipes") from None


def assign_quarantine(
    store: LineStore,
    provider: RecipeProvider,
    scan_file_id: str,
    principal: Principal,
    *,
    recipe_id: str | None = None,
    sku: str | None = None,
    declared_stage: str | None = None,
    stage_index: int | None = None,
    idempotent: bool = False,
) -> dict[str, Any]:
    """Assign a quarantined scan to a SKU's active recipe and enqueue its inspect job.

    The target is exactly one of ``sku`` (plan §6 step 7: the operator assigns
    the SKU; the server picks that SKU's active recipe) or ``recipe_id``,
    which must then *be* the SKU's active recipe. The manual path is never
    weaker than the hotfolder's automatic resolution:

    - the provider must also be an ``ActiveRecipeCatalog`` (C05's
      ``LineRecipeResolver`` is); otherwise 503 RECIPE_CATALOG_UNAVAILABLE;
    - ``RecipeProvider.load`` must succeed (404 RECIPE_NOT_FOUND) and return
      the requested recipe;
    - a recipe whose ``extra.status`` is not ``released`` is 409 RECIPE_RETIRED;
    - a superseded version is 409 RECIPE_NOT_ACTIVE (the message names the
      active one); a SKU with no released recipe is 409 PART_NOT_READY when
      it is registered, else 404 SKU_NOT_FOUND;
    - the recipe must be compiled (``compiled_part_sha256`` present,
      ``extra.compile_status`` absent or COMPILED, and the catalog's active
      ref COMPILED with the same compiled-part hash), else 409 PART_NOT_READY.

    The store emits QUARANTINE_ASSIGNED and JOB_QUEUED with the principal.

    Station part picker (``idempotent=True``): ``declared_stage`` FINAL /
    IN_PROCESS (+ ``stage_index``) becomes an OPERATOR ``DeclaredStageV1``
    (:func:`part_picker_service.declared_stage_for`); a repeat of the same
    assignment (double click, a second operator picking the same part) returns
    the existing job with ``assigned_now: false``; a scan already assigned to
    another recipe or dismissed is 409 ALREADY_RESOLVED. The assignment only
    picks the recipe: E3 still runs every identity check (wrong part, mirrored,
    units) and the verdict is the engine's.
    """
    if not _SCAN_FILE_ID_RE.match(scan_file_id or ""):
        raise LineApiError(404, "SCAN_NOT_FOUND", "no such scan")
    if (recipe_id is None) == (sku is None):
        raise LineApiError(422, "ASSIGN_TARGET_INVALID", "give exactly one of sku or recipe_id")
    if recipe_id is not None and (not isinstance(recipe_id, str) or not _RECIPE_ID_RE.match(recipe_id)):
        raise LineApiError(422, "RECIPE_ID_INVALID", "recipe_id must be '<SKU>@v<version>'")
    if sku is not None and (not isinstance(sku, str) or not _SKU_RE.match(sku)):
        raise LineApiError(422, "SKU_INVALID", "sku must be a canonical SKU")
    scan = store.get_scan_file(scan_file_id)
    if scan is None:
        raise LineApiError(404, "SCAN_NOT_FOUND", "no such scan")
    if scan["ingest_state"] != "quarantined" and not idempotent:
        raise LineApiError(409, "SCAN_NOT_QUARANTINED", f"scan is {scan['ingest_state']}")
    if not isinstance(provider, ActiveRecipeCatalog):
        raise LineApiError(503, "RECIPE_CATALOG_UNAVAILABLE", "the recipe store cannot list active recipes")
    refs = _active_refs(provider)
    if sku is not None:
        chosen = next((r for r in refs if r.sku == sku), None)
        if chosen is None:
            try:
                registered = set(provider.list_registered_skus())
            except Exception as exc:
                logger.error("recipe catalog failed: %s", type(exc).__name__)
                raise LineApiError(503, "RECIPE_PROVIDER_ERROR", "the recipe store could not list SKUs") from None
            if sku in registered:
                raise LineApiError(409, "PART_NOT_READY", f"SKU {sku} has no released recipe")
            raise LineApiError(404, "SKU_NOT_FOUND", f"SKU {sku} is not registered")
        recipe_id = chosen.recipe_id
    target: str = str(recipe_id)
    try:
        resolved = provider.load(target)
    except KeyError:
        raise LineApiError(404, "RECIPE_NOT_FOUND", "no such recipe") from None
    except Exception as exc:
        logger.error("recipe provider failed for %s: %s", recipe_id, type(exc).__name__)
        raise LineApiError(503, "RECIPE_PROVIDER_ERROR", "the recipe store could not load the recipe") from None
    if resolved.recipe_id != recipe_id or (sku is not None and resolved.sku != sku):
        raise LineApiError(409, "RECIPE_MISMATCH", "the recipe store returned another recipe")
    extra = resolved.extra or {}
    lifecycle = extra.get("status")
    if lifecycle is not None and lifecycle != "released":
        raise LineApiError(409, "RECIPE_RETIRED", f"{recipe_id} is {lifecycle}; assign the SKU's active recipe")
    compile_status = extra.get("compile_status")
    if resolved.compiled_part_sha256 is None or compile_status not in (None, "COMPILED"):
        raise LineApiError(409, "PART_NOT_READY", "the recipe has no successful compile yet")
    active = next((r for r in refs if r.sku == resolved.sku), None)
    if active is None:
        raise LineApiError(409, "RECIPE_NOT_ACTIVE", f"SKU {resolved.sku} has no active recipe")
    if active.recipe_id != resolved.recipe_id or active.recipe_sha256 != resolved.recipe_sha256:
        raise LineApiError(
            409, "RECIPE_NOT_ACTIVE", f"{recipe_id} is not the active recipe of {resolved.sku} (active: {active.recipe_id})"
        )
    if active.compile_status != "COMPILED" or active.compiled_part_sha256 != resolved.compiled_part_sha256:
        raise LineApiError(409, "PART_NOT_READY", "the recipe has no successful compile yet")
    ref = {
        "recipe_id": resolved.recipe_id,
        "recipe_version": int(resolved.recipe_version),
        "recipe_sha256": resolved.recipe_sha256,
        "sku": resolved.sku,
        "compiled_part_sha256": resolved.compiled_part_sha256,
    }
    declared: dict[str, Any] | None = None
    if declared_stage is not None or stage_index is not None:
        from apps.api.services import part_picker_service as picker

        try:
            declared = picker.declared_stage_for(picker.facts_of(resolved), declared_stage, stage_index)
        except picker.PickerError as exc:
            raise LineApiError(exc.status_code, exc.code, exc.message) from None
    assigned_now = True
    try:
        if idempotent:
            job_id, assigned_now = store.assign_quarantine(
                scan_file_id, ref, principal.audit(), declared_stage=declared, actor_name=principal.username or None
            )
        else:
            job_id = store.resolve_quarantine(scan_file_id, ref, principal.audit(), declared_stage=declared)
    except KeyError:
        raise LineApiError(404, "SCAN_NOT_FOUND", "no such scan") from None
    except QuarantineError as exc:
        code = "SCAN_NOT_QUARANTINED" if exc.code == "SCAN_STATE" else exc.code
        raise LineApiError(409, code, str(exc)) from None
    except LineStoreError as exc:
        raise LineApiError(422, exc.code, str(exc)) from None
    return {
        "scan_file_id": scan_file_id,
        "job_id": job_id,
        "recipe_id": resolved.recipe_id,
        "recipe_version": int(resolved.recipe_version),
        "sku": resolved.sku,
        "assigned_now": assigned_now,
        "declared_stage": declared,
    }


# ---------------------------------------------------------------------------
# Status, readiness and metrics
# ---------------------------------------------------------------------------


class StatusProbes:
    """Expensive probes cached for ``ttl_s`` (python import probe, bundle verify)."""

    def __init__(
        self,
        settings: LineSettings,
        config: LineApiConfig,
        *,
        python_probe: Callable[[], health.ProbeResult] | None = None,
        bundle_probe: Callable[[], health.ProbeResult] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_s = config.probe_ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, health.ProbeResult, str]] = {}
        self._python = python_probe or (
            lambda: health.python_probe(settings.python_exe, timeout_s=config.python_probe_timeout_s)
        )
        self._bundle = bundle_probe or (
            lambda: health.bundle_present(settings.engine_bundle_dir, verify=config.verify_bundle)
        )

    def _cached(self, name: str, fn: Callable[[], health.ProbeResult]) -> tuple[health.ProbeResult, str]:
        now = self._clock()
        with self._lock:
            hit = self._cache.get(name)
            if hit is not None and now - hit[0] < self.ttl_s:
                return hit[1], hit[2]
        try:
            res = fn()
        except Exception as exc:  # a probe never takes the status route down
            res = health.ProbeResult(name, False, f"raised {type(exc).__name__}")
        checked = _iso(_utc_now())
        with self._lock:
            self._cache[name] = (self._clock(), res, checked)
        return res, checked

    def python(self) -> tuple[health.ProbeResult, str]:
        return self._cached("python", self._python)

    def bundle(self) -> tuple[health.ProbeResult, str]:
        return self._cached("bundle", self._bundle)


def get_status_probes(
    settings: LineSettings = Depends(require_line_settings),
    config: LineApiConfig = Depends(get_line_api_config),
) -> StatusProbes:
    key = (settings.data_root, config)
    with _cache_lock:
        probes = _probes.get(key)
        if probes is None:
            probes = StatusProbes(settings, config)
            _probes[key] = probes
        return probes


def _heartbeat_rows(store: LineStore | None) -> list[dict[str, Any]]:
    if store is None:
        return []
    try:
        return store.get_heartbeats()
    except (sqlite3.Error, OSError, LineStoreError):
        return []


def _hb_fresh(rows: list[dict[str, Any]], component: str, max_age_s: float) -> bool:
    for row in rows:
        if row.get("component") == component:
            age = row.get("age_s")
            return isinstance(age, (int, float)) and 0 <= age <= max_age_s
    return False


def readiness_checks(
    settings: LineSettings | None, config: LineApiConfig
) -> tuple[bool, dict[str, str]]:
    """``(ready, {name: 'ok'|'fail'})`` for ``GET /health/ready``.

    Checks: ``db`` (the line DB exists and takes a rolled-back write; never
    created here), ``disk`` (free bytes on the data root >= ``min_free_bytes``),
    ``runner`` and ``hotfolder:<station>`` (heartbeat younger than
    ``stale_factor * heartbeat_s``). Names carry no paths or details.
    """
    if settings is None:
        return False, {"configured": "fail"}
    checks: dict[str, bool] = {}
    db_ok = health.sqlite_writable(settings.paths.db_line, name="db").ok
    checks["db"] = db_ok
    checks["disk"] = health.disk_free(settings.data_root, config.min_free_bytes).ok
    store: LineStore | None = None
    if db_ok:
        try:
            store = line_store_for(settings)
        except (LineStoreError, sqlite3.Error, OSError):
            store = None
            checks["db"] = False
    rows = _heartbeat_rows(store)
    checks["runner"] = _hb_fresh(rows, "runner", config.stale_after_s)
    for sid in settings.station_ids:
        checks[f"hotfolder:{sid}"] = _hb_fresh(rows, f"hotfolder:{sid}", config.stale_after_s)
    ready = all(checks.values())
    return ready, {k: ("ok" if v else "fail") for k, v in checks.items()}


def cached_readiness(
    settings: LineSettings | None, config: LineApiConfig, *, clock: Callable[[], float] = time.monotonic
) -> tuple[bool, dict[str, str]]:
    """:func:`readiness_checks`, cached for ``config.readiness_ttl_s``.

    ``GET /health/ready`` needs no auth, and each uncached check takes a
    rolled-back write transaction on the line DB. The cache (and one shared
    compute lock) keeps a hammered endpoint from competing with the runner and
    hotfolder for the write lock.
    """
    if settings is None or config.readiness_ttl_s <= 0:
        return readiness_checks(settings, config)
    key = (settings.data_root, config)
    with _readiness_compute_lock:
        now = clock()
        with _cache_lock:
            hit = _readiness.get(key)
        if hit is not None and 0 <= now - hit[0] < config.readiness_ttl_s:
            return hit[1], dict(hit[2])
        ready, checks = readiness_checks(settings, config)
        with _cache_lock:
            _readiness[key] = (clock(), ready, dict(checks))
        return ready, checks


def _ro_query(db_path: Path, sql: str, args: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    """One read-only row from the line DB (for status fields the store API lacks)."""
    try:
        conn = sqlite3.connect(health.sqlite_uri(db_path, "ro"), uri=True, timeout=2)
    except sqlite3.Error:
        return None
    try:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, args).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def status_report(
    settings: LineSettings,
    store: LineStore,
    config: LineApiConfig,
    probes: StatusProbes,
) -> dict[str, Any]:
    """The authenticated ``GET /status`` body (operators may see paths here)."""
    rows = _heartbeat_rows(store)
    heartbeats = [
        {
            "component": r["component"],
            "instance_id": r["instance_id"],
            "host": r["host"],
            "pid": r["pid"],
            "ts": r["ts"],
            "age_s": round(float(r["age_s"]), 3),
            "stale": not (0 <= float(r["age_s"]) <= config.stale_after_s),
            "details": r.get("details") or {},
        }
        for r in rows
    ]
    expected = ["runner"] + [f"hotfolder:{sid}" for sid in settings.station_ids]
    present = {h["component"] for h in heartbeats}
    missing_heartbeats = [c for c in expected if c not in present]
    running = [job_summary(settings, store, j).model_dump(mode="json") for j in store.list_jobs(statuses=["running"], limit=20)]
    last_scan = _ro_query(
        settings.paths.db_line,
        "SELECT scan_file_id, original_name, station_id, received_at, ingest_state FROM scan_files"
        " ORDER BY received_at DESC LIMIT 1",
    )
    last_done = _ro_query(
        settings.paths.db_line,
        "SELECT job_id FROM jobs WHERE execution_status='completed' ORDER BY finished_at DESC LIMIT 1",
    )
    last_completion = None
    if last_done is not None:
        job = store.get_job(last_done["job_id"])
        if job is not None:
            last_completion = job_summary(settings, store, job).model_dump(mode="json")
    open_q = store.list_quarantine()
    disk: dict[str, Any]
    try:
        usage = shutil.disk_usage(settings.data_root)
        disk = {
            "free_bytes": int(usage.free),
            "total_bytes": int(usage.total),
            "min_free_bytes": int(config.min_free_bytes),
            "ok": usage.free >= config.min_free_bytes,
        }
    except OSError as exc:
        disk = {"free_bytes": None, "total_bytes": None, "min_free_bytes": int(config.min_free_bytes), "ok": False,
                "detail": type(exc).__name__}
    bundle_res, bundle_checked = probes.bundle()
    bundle_id: str | None = None
    try:
        manifest = json.loads((settings.engine_bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
        bundle_id = str(manifest.get("bundle_id")) if isinstance(manifest, dict) and manifest.get("bundle_id") else None
    except (OSError, ValueError):
        bundle_id = None
    py_res, py_checked = probes.python()
    ready, checks = readiness_checks(settings, config)
    return {
        "server_time": _iso(_utc_now()),
        "ready": ready,
        "checks": checks,
        "stations": list(settings.station_ids),
        "heartbeats": heartbeats,
        "missing_heartbeats": missing_heartbeats,
        "heartbeat_stale_after_s": config.stale_after_s,
        "queue": store.counts(),
        "running": running,
        "last_ingest": None if last_scan is None else dict(last_scan),
        "last_completion": last_completion,
        "open_quarantine": {
            "count": len(open_q),
            "items": [
                {
                    "scan_file_id": q["scan_file"]["scan_file_id"],
                    "station_id": q["scan_file"]["station_id"],
                    "original_name": q["scan_file"]["original_name"],
                    "reason": q["quarantine_reason"],
                    "quarantined_at": q["quarantined_at"],
                    "resolvable": q["resolvable"],
                }
                for q in open_q[:20]
            ],
        },
        "disk": disk,
        "engine_bundle": {
            "bundle_id": bundle_id,
            "ok": bundle_res.ok,
            "verified": bool(config.verify_bundle),
            "detail": bundle_res.detail,
            "checked_at": bundle_checked,
        },
        "python": {"ok": py_res.ok, "detail": py_res.detail, "checked_at": py_checked},
    }


def metrics_text(registry: MetricsRegistry, store: LineStore) -> str:
    """Prometheus text: the registry plus scrape-time gauges from ``store.counts()``."""
    counts = store.counts()
    extra: list[tuple[str, float, str, Mapping[str, str]]] = [
        ("sherman_line_queue_depth", counts["queue_depth"], "Queued jobs", {}),
        ("sherman_line_running_jobs", counts["running"], "Running jobs", {}),
        ("sherman_line_open_quarantine", counts["open_quarantine"], "Open quarantined scans", {}),
        ("sherman_line_failed_jobs_24h", counts["failed_last_24h"], "Jobs failed in the last 24 h", {}),
        ("sherman_line_scans_pending_resolution", counts["scans_pending_resolution"], "Scans received, not yet resolved", {}),
    ]
    for status_name, value in sorted(counts["jobs_by_status"].items()):
        extra.append(("sherman_line_jobs", value, "Jobs by execution status", {"status": status_name}))
    for row in _heartbeat_rows(store):
        extra.append(
            ("sherman_line_heartbeat_age_seconds", float(row["age_s"]), "Age of the last heartbeat", {"component": row["component"]})
        )
    return render_prometheus(registry, extra)


# ---------------------------------------------------------------------------
# Query validation helpers used by the router
# ---------------------------------------------------------------------------


def parse_statuses(values: Iterable[str] | None) -> list[str] | None:
    """``status`` query values (repeated and/or comma separated) -> list, or 422."""
    if values is None:
        return None
    out: list[str] = []
    for value in values:
        out.extend(v.strip() for v in str(value).split(",") if v.strip())
    if not out:
        return None
    bad = sorted(set(out) - EXECUTION_STATUSES)
    if bad:
        raise LineApiError(422, "STATUS_INVALID", f"unknown status {bad[0]!r}")
    return sorted(set(out))


def parse_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    if kind not in JOB_KINDS:
        raise LineApiError(422, "KIND_INVALID", f"unknown job kind {kind!r}")
    return kind


def parse_before(before: str | None) -> str | None:
    """A ``before`` cursor normalised to the store's ``created_at`` format.

    The store compares ``created_at < before`` as text, so the cursor is
    rewritten as fixed-width UTC milliseconds (``YYYY-MM-DDTHH:MM:SS.mmmZ``):
    an offset such as ``+03:00`` is converted, and a sub-millisecond part is
    rounded *up* so no job created before the instant is lost. A date without
    a time is refused (422 CURSOR_INVALID).
    """
    if before is None:
        return None
    parsed = _parse_iso(before) if len(before) <= 40 and "T" in before else None
    if parsed is None:
        raise LineApiError(422, "CURSOR_INVALID", "before must be an ISO-8601 timestamp with a time (a created_at)")
    if parsed.microsecond % 1000:
        parsed = parsed + timedelta(microseconds=1000 - parsed.microsecond % 1000)
    return _iso(parsed)
