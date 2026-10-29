"""Durable SQLite line store: scan files, jobs with leases, events, heartbeats.

This is the single source of truth for the production line's execution state.
Nothing about a job lives only in memory, so the line service can be killed at
any point and resumed (``recover_expired_leases``).

Design rules:

- Each public call opens its own connection and closes it again, so one
  ``LineStore`` instance is safe to share between threads, and several
  processes may open the same file. PRAGMAs: ``journal_mode=WAL`` (set once,
  persistent), ``synchronous=NORMAL``, ``busy_timeout=5000``,
  ``foreign_keys=ON``.
- Every state change and the event that announces it are written in one
  ``BEGIN IMMEDIATE`` transaction, so the event log never disagrees with the
  rows it describes.
- Timestamps are UTC ISO-8601 with millisecond precision and a ``Z`` suffix
  (``2026-10-17T07:15:01.123Z``). The width is fixed, so string comparison is
  time comparison (lease expiry uses it).
- Rows are returned as plain dicts shaped exactly like the C01 contract JSON
  (``ScanFileV1`` / ``LineJobV1`` / ``LineEventV1``, with the ``schema`` tag),
  so ``LineJobV1.model_validate(store.get_job(job_id))`` works. Extra
  bookkeeping is available through ``get_job_meta`` and ``list_quarantine``.
- Execution status never encodes conformance. A failed engine run is a job
  failure code, never a part FAIL.
- No pydantic or contract import here (fast, dependency-free import); the
  literal sets are mirrored in :mod:`apps.production_line.store.vocabulary`.

Public API (C05, C07, C11 and C12 call these names): ``register_scan``,
``set_scan_match``, ``match_and_enqueue``, ``get_scan_meta``, ``quarantine_scan``,
``list_quarantine``, ``get_scan_file``, ``get_scan_by_sha``, ``list_scans``,
``resolve_quarantine``, ``assign_quarantine``, ``dismiss_quarantine``, ``enqueue_job``,
``lease_next``, ``renew_lease``, ``update_progress``, ``complete_job``,
``fail_job``, ``request_cancel``, ``is_cancel_requested``, ``mark_cancelled``,
``recover_expired_leases``, ``expire_leases_of``, ``request_retry``, ``append_event``,
``events_after``, ``get_job``, ``get_job_meta``, ``list_jobs``,
``latest_job_for_station``, ``counts``, ``put_heartbeat``, ``get_heartbeats``,
``record_ingest_source``, ``has_ingest_source``.
"""
from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from apps.production_line.store.vocabulary import (
    EVENT_KINDS,
    EXECUTION_STATUSES,
    FAILURE_CODES,
    INGEST_STATES,
    JOB_KINDS,
    JOB_RESULT_SUMMARY_KEYS,
    LINE_EVENT_SCHEMA,
    LINE_JOB_SCHEMA,
    MAX_ATTEMPTS_BY_KIND,
    PRIORITY_BY_KIND,
    QUARANTINE_REASONS,
    RESOLVABLE_QUARANTINE_REASONS,
    RESOLVE_STATUSES,
    SCAN_FILE_SCHEMA,
    SCAN_FORMATS,
    STAGE_STATUSES,
    SUMMARY_STATES,
    TERMINAL_STATUSES,
    VALIDITY_STATES,
)

__all__ = [
    "SCHEMA_COMPONENT",
    "SCHEMA_VERSION",
    "LineStoreError",
    "JobStateError",
    "QuarantineError",
    "LineStore",
    "utc_now",
    "iso_utc",
    "parse_iso_utc",
    "new_job_id",
    "scan_file_id_for",
    "inspect_dedupe_key",
]

SCHEMA_COMPONENT = "line_store"
SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID_RE = re.compile(r"^line_[0-9a-f]{32}$")
_SCAN_ID_RE = re.compile(r"^scan_[0-9a-f]{24}$")
_RECIPE_ID_RE = re.compile(r"^(?P<sku>.+)@v(?P<version>[1-9][0-9]*)$")
_RUNNING_STAGES = frozenset({"preparing", "engine", "postprocess"})
_PROGRESS_EVENT_MIN_INTERVAL = timedelta(seconds=1)


class LineStoreError(ValueError):
    """Invalid argument or state for a store call. ``code`` is machine readable."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


class JobStateError(LineStoreError):
    """The job is not in a state that allows the requested transition."""


class QuarantineError(LineStoreError):
    """The scan is not quarantined, or its reason cannot be resolved by assignment."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    """Fixed-width UTC ISO-8601 with milliseconds and ``Z``."""
    if dt.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_iso_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def new_job_id() -> str:
    return "line_" + uuid.uuid4().hex


def scan_file_id_for(sha256: str) -> str:
    _require_sha(sha256, "sha256")
    return "scan_" + sha256[:24]


def inspect_dedupe_key(scan_sha256: str, recipe_sha256: str) -> str:
    """One inspect job per (scan bytes, recipe). Retries use ``retry:<job_id>:<n>``."""
    return f"inspect:{scan_sha256}:{recipe_sha256}"


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise LineStoreError("INVALID_ARGUMENT", f"{name} must be 64 lowercase hex characters")
    return value


def _dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


def _principal_dict(principal: Any) -> dict[str, Any] | None:
    """Normalise a principal (mapping, pydantic model or None) to ``{kind, id, role}``."""
    if principal is None:
        return None
    if hasattr(principal, "model_dump"):
        principal = principal.model_dump(mode="json")
    if not isinstance(principal, Mapping):
        raise LineStoreError("INVALID_ARGUMENT", "principal must be a mapping {kind, id, role}")
    kind = principal.get("kind")
    pid = principal.get("id")
    if kind not in ("user", "device", "service") or not isinstance(pid, str) or not pid:
        raise LineStoreError("INVALID_ARGUMENT", "principal needs kind user|device|service and a non-empty id")
    role = principal.get("role")
    return {"kind": kind, "id": pid[:128], "role": None if role is None else str(role)[:64]}


def _check_result_summary(summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    unknown = set(summary) - set(JOB_RESULT_SUMMARY_KEYS)
    if unknown:
        raise LineStoreError("INVALID_ARGUMENT", f"unknown result_summary keys: {sorted(unknown)}")
    out = {key: summary.get(key) for key in JOB_RESULT_SUMMARY_KEYS}
    if out["summary_state"] not in SUMMARY_STATES:
        raise LineStoreError("INVALID_ARGUMENT", "result_summary.summary_state is not a SUMMARY_STATE")
    if out["inspection_validity"] not in VALIDITY_STATES:
        raise LineStoreError("INVALID_ARGUMENT", "result_summary.inspection_validity is not a VALIDITY_STATE")
    if out["stage_status"] is not None and out["stage_status"] not in STAGE_STATUSES:
        raise LineStoreError("INVALID_ARGUMENT", "result_summary.stage_status is not a STAGE_STATUS")
    for key in ("bends_total", "required_fail", "required_review", "required_unknown"):
        value = out[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise LineStoreError("INVALID_ARGUMENT", f"result_summary.{key} must be a non-negative int")
    return out


def _check_declared_stage(declared: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if declared is None:
        return None
    if hasattr(declared, "model_dump"):
        declared = declared.model_dump(mode="json")  # type: ignore[union-attr]
    index = declared.get("stage_index")
    source = declared.get("source")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0 or source not in ("SIDECAR", "OPERATOR", "API"):
        raise LineStoreError("INVALID_ARGUMENT", "declared_stage must be {stage_index >= 0, source SIDECAR|OPERATOR|API}")
    return {"stage_index": index, "source": source}


_SCHEMA_V1 = (
    """
    CREATE TABLE IF NOT EXISTS scan_files (
        scan_file_id TEXT PRIMARY KEY,
        sha256 TEXT NOT NULL UNIQUE,
        original_name TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        station_id TEXT NOT NULL,
        received_at TEXT NOT NULL,
        stored_rel_path TEXT NOT NULL,
        format TEXT,
        vertex_count INTEGER,
        face_count INTEGER,
        bbox_json TEXT,
        parse_json TEXT NOT NULL,
        match_json TEXT NOT NULL,
        declared_stage_json TEXT,
        ingest_state TEXT NOT NULL,
        quarantine_reason TEXT,
        quarantine_detail_json TEXT,
        quarantined_at TEXT,
        resolution TEXT,
        resolved_at TEXT,
        resolved_by_json TEXT,
        job_id TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS scan_files_state ON scan_files(ingest_state, station_id, received_at)",
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        station_id TEXT,
        scan_file_id TEXT REFERENCES scan_files(scan_file_id),
        scan_sha256 TEXT,
        sku TEXT NOT NULL,
        recipe_id TEXT NOT NULL,
        recipe_version INTEGER NOT NULL,
        recipe_sha256 TEXT NOT NULL,
        compiled_part_sha256 TEXT,
        declared_stage_json TEXT,
        priority INTEGER NOT NULL,
        dedupe_key TEXT,
        retry_of TEXT,
        execution_status TEXT NOT NULL,
        stage TEXT NOT NULL,
        progress INTEGER NOT NULL DEFAULT 0,
        progress_message TEXT NOT NULL DEFAULT '',
        attempts INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL,
        lease_owner TEXT,
        lease_expires_at TEXT,
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        failure_code TEXT,
        failure_message TEXT,
        engine_bundle_id TEXT,
        workspace_rel TEXT NOT NULL,
        result_summary_json TEXT,
        peak_rss_bytes INTEGER,
        duration_s REAL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS jobs_dedupe ON jobs(dedupe_key) WHERE dedupe_key IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(execution_status, priority DESC, created_at)",
    "CREATE INDEX IF NOT EXISTS jobs_station ON jobs(station_id, created_at)",
    "CREATE INDEX IF NOT EXISTS jobs_retry_of ON jobs(retry_of)",
    """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        kind TEXT NOT NULL,
        station_id TEXT,
        job_id TEXT,
        scan_file_id TEXT,
        principal_json TEXT,
        payload_json TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS events_station ON events(station_id, id)",
    """
    CREATE TABLE IF NOT EXISTS heartbeats (
        component TEXT PRIMARY KEY,
        instance_id TEXT NOT NULL,
        pid INTEGER NOT NULL,
        host TEXT NOT NULL,
        ts TEXT NOT NULL,
        details_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_sources (
        station_id TEXT NOT NULL,
        source_path TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        mtime_ns INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        PRIMARY KEY (station_id, source_path, size_bytes, mtime_ns)
    )
    """,
)


class LineStore:
    """SQLite-backed line state (see module docstring)."""

    def __init__(self, db_path: str | os.PathLike[str], clock: Callable[[], datetime] = utc_now) -> None:
        self.db_path = Path(db_path)
        self._clock = clock
        self._progress_lock = threading.Lock()
        self._last_progress_event: dict[str, datetime] = {}
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    # ------------------------------------------------------------------ plumbing
    def _now(self) -> datetime:
        return self._clock()

    def _now_iso(self) -> str:
        return iso_utc(self._now())

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _migrate(self) -> None:
        conn = self._connect()
        try:
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":  # pragma: no cover - only on filesystems without WAL support
                raise LineStoreError("STORE_UNAVAILABLE", f"journal_mode WAL not available (got {mode})")
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_meta (component TEXT PRIMARY KEY, version INTEGER NOT NULL,"
                    " updated_at TEXT NOT NULL)"
                )
                row = conn.execute("SELECT version FROM schema_meta WHERE component=?", (SCHEMA_COMPONENT,)).fetchone()
                current = int(row["version"]) if row else 0
                if current > SCHEMA_VERSION:
                    raise LineStoreError(
                        "SCHEMA_TOO_NEW", f"line store schema v{current} is newer than this code (v{SCHEMA_VERSION})"
                    )
                if current < 1:
                    for statement in _SCHEMA_V1:
                        conn.execute(statement)
                    conn.execute(
                        "INSERT OR REPLACE INTO schema_meta(component, version, updated_at) VALUES (?,?,?)",
                        (SCHEMA_COMPONENT, SCHEMA_VERSION, self._now_iso()),
                    )
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    def schema_version(self) -> int:
        with self._read() as conn:
            row = conn.execute("SELECT version FROM schema_meta WHERE component=?", (SCHEMA_COMPONENT,)).fetchone()
        return int(row["version"]) if row else 0

    # ------------------------------------------------------------------ events
    def _event(
        self,
        conn: sqlite3.Connection,
        kind: str,
        *,
        station_id: str | None = None,
        job_id: str | None = None,
        scan_file_id: str | None = None,
        principal: Any = None,
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        if kind not in EVENT_KINDS:
            raise LineStoreError("INVALID_ARGUMENT", f"unknown event kind {kind!r}")
        if job_id is not None and not _JOB_ID_RE.match(job_id):
            raise LineStoreError("INVALID_ARGUMENT", "job_id must match ^line_[0-9a-f]{32}$")
        if scan_file_id is not None and not _SCAN_ID_RE.match(scan_file_id):
            raise LineStoreError("INVALID_ARGUMENT", "scan_file_id must match ^scan_[0-9a-f]{24}$")
        cur = conn.execute(
            "INSERT INTO events(ts, kind, station_id, job_id, scan_file_id, principal_json, payload_json)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                self._now_iso(),
                kind,
                station_id,
                job_id,
                scan_file_id,
                _dumps(_principal_dict(principal)),
                _dumps(dict(payload or {})),
            ),
        )
        if cur.lastrowid is None:  # pragma: no cover - sqlite always sets it after INSERT
            raise LineStoreError("STORE_UNAVAILABLE", "event insert returned no id")
        return int(cur.lastrowid)

    def append_event(
        self,
        kind: str,
        *,
        station_id: str | None = None,
        job_id: str | None = None,
        scan_file_id: str | None = None,
        principal: Any = None,
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        """Append one event (e.g. RECIPE_RELEASED from C05) and return its id."""
        with self._tx() as conn:
            return self._event(
                conn,
                kind,
                station_id=station_id,
                job_id=job_id,
                scan_file_id=scan_file_id,
                principal=principal,
                payload=payload,
            )

    def events_after(
        self,
        after_id: int = 0,
        station_id: str | None = None,
        limit: int = 200,
        *,
        include_global: bool = False,
        kinds: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Events with ``id > after_id`` in id order, shaped like ``LineEventV1``.

        With ``station_id``, only that station's events are returned, plus
        events with no station when ``include_global`` is true.
        """
        limit = max(1, min(int(limit), 1000))
        sql = "SELECT * FROM events WHERE id > ?"
        args: list[Any] = [int(after_id)]
        if station_id is not None:
            sql += " AND (station_id = ?" + (" OR station_id IS NULL)" if include_global else ")")
            args.append(station_id)
        kind_list = sorted(set(kinds)) if kinds is not None else None
        if kind_list:
            sql += " AND kind IN (" + ",".join("?" * len(kind_list)) + ")"
            args.extend(kind_list)
        sql += " ORDER BY id ASC LIMIT ?"
        args.append(limit)
        with self._read() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._event_dict(r) for r in rows]

    def last_event_id(self) -> int:
        with self._read() as conn:
            row = conn.execute("SELECT MAX(id) AS m FROM events").fetchone()
        return int(row["m"] or 0)

    @staticmethod
    def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "schema": LINE_EVENT_SCHEMA,
            "id": int(row["id"]),
            "ts": row["ts"],
            "kind": row["kind"],
            "station_id": row["station_id"],
            "job_id": row["job_id"],
            "scan_file_id": row["scan_file_id"],
            "principal": _loads(row["principal_json"]),
            "payload": _loads(row["payload_json"]) or {},
        }

    # ------------------------------------------------------------------ scans
    @staticmethod
    def _scan_dict(row: sqlite3.Row) -> dict[str, Any]:
        bbox = _loads(row["bbox_json"])
        quarantined = row["ingest_state"] == "quarantined"
        return {
            "schema": SCAN_FILE_SCHEMA,
            "scan_file_id": row["scan_file_id"],
            "sha256": row["sha256"],
            "original_name": row["original_name"],
            "size_bytes": int(row["size_bytes"]),
            "station_id": row["station_id"],
            "received_at": row["received_at"],
            "stored_rel_path": row["stored_rel_path"],
            "format": row["format"],
            "vertex_count": row["vertex_count"],
            "face_count": row["face_count"],
            "bbox_mm": None if bbox is None else [list(bbox[0]), list(bbox[1])],
            "parse": _loads(row["parse_json"]),
            "match": _loads(row["match_json"]),
            "declared_stage": _loads(row["declared_stage_json"]),
            "ingest_state": row["ingest_state"],
            "quarantine_reason": row["quarantine_reason"] if quarantined else None,
        }

    @staticmethod
    def _check_match(match: Mapping[str, Any]) -> dict[str, Any]:
        status = match.get("status")
        if status not in RESOLVE_STATUSES:
            raise LineStoreError("INVALID_ARGUMENT", f"match.status {status!r} is not a RESOLVE_STATUS")
        out = {
            "status": status,
            "recipe_id": match.get("recipe_id"),
            "recipe_version": match.get("recipe_version"),
            "recipe_sha256": match.get("recipe_sha256"),
            "reason": None if match.get("reason") is None else str(match.get("reason"))[:200],
        }
        if status == "MATCHED":
            if out["recipe_id"] is None or out["recipe_version"] is None or out["recipe_sha256"] is None:
                raise LineStoreError("INVALID_ARGUMENT", "a MATCHED scan names its recipe id, version and sha")
            _require_sha(out["recipe_sha256"], "match.recipe_sha256")
        return out

    @staticmethod
    def is_duplicate_drop(existing: Mapping[str, Any], *, original_name: str, station_id: str) -> bool:
        """True when a drop of already-registered bytes is a duplicate, not a resume.

        A row past ``received`` is always a duplicate. A row still in
        ``received`` is resumed only by a drop of the *same* name at the
        *same* station; a drop under another name (e.g. another SKU's file
        name) must never take over that row, or the scan would be inspected
        against the recipe its new name resolves to.
        """
        if existing["ingest_state"] != "received":
            return True
        return existing["original_name"] != original_name or existing["station_id"] != station_id

    def register_scan(
        self,
        *,
        sha256: str,
        original_name: str,
        size_bytes: int,
        station_id: str,
        stored_rel_path: str,
        parse: Mapping[str, Any],
        match: Mapping[str, Any],
        format: str | None = None,
        vertex_count: int | None = None,
        face_count: int | None = None,
        bbox_mm: Sequence[Sequence[float]] | None = None,
        declared_stage: Mapping[str, Any] | None = None,
        ingest_state: str = "received",
        quarantine_reason: str | None = None,
        quarantine_detail: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Insert a scan row, or return the existing row for the same bytes.

        Returns ``(row, created)``. A new row emits SCAN_RECEIVED (and
        SCAN_QUARANTINED when registered quarantined). When the sha256 already
        exists, the existing row is returned with ``created=False``, and a
        SCAN_DUPLICATE event names the new drop when that row has already been
        processed (state other than ``received``) *or* the drop carries another
        file name or station (see :meth:`is_duplicate_drop`). Only a drop of
        the same name at the same station onto a row still in ``received`` is
        the continuation of an interrupted ingest (no event); the caller
        resumes it and resolves the row by its own ``original_name``.
        """
        _require_sha(sha256, "sha256")
        if ingest_state not in ("received", "quarantined"):
            raise LineStoreError("INVALID_ARGUMENT", "a scan is registered as received or quarantined")
        if (ingest_state == "quarantined") != (quarantine_reason is not None):
            raise LineStoreError("INVALID_ARGUMENT", "quarantine_reason is required exactly when quarantined")
        if quarantine_reason is not None and quarantine_reason not in QUARANTINE_REASONS:
            raise LineStoreError("INVALID_ARGUMENT", f"unknown quarantine reason {quarantine_reason!r}")
        if format is not None and format not in SCAN_FORMATS:
            raise LineStoreError("INVALID_ARGUMENT", f"unknown scan format {format!r}")
        if not original_name or len(original_name) > 255:
            raise LineStoreError("INVALID_ARGUMENT", "original_name must be 1-255 characters")
        match_d = self._check_match(match)
        scan_id = scan_file_id_for(sha256)
        now = self._now_iso()
        bbox = None if bbox_mm is None else [[float(v) for v in bbox_mm[0]], [float(v) for v in bbox_mm[1]]]
        with self._tx() as conn:
            existing = conn.execute("SELECT * FROM scan_files WHERE sha256=?", (sha256,)).fetchone()
            if existing is not None:
                if self.is_duplicate_drop(existing, original_name=original_name, station_id=station_id):
                    self._event(
                        conn,
                        "SCAN_DUPLICATE",
                        station_id=station_id,
                        scan_file_id=existing["scan_file_id"],
                        payload={
                            "original_name": original_name,
                            "existing_original_name": existing["original_name"],
                            "existing_state": existing["ingest_state"],
                            "existing_job_id": existing["job_id"],
                        },
                    )
                return self._scan_dict(existing), False
            conn.execute(
                "INSERT INTO scan_files(scan_file_id, sha256, original_name, size_bytes, station_id, received_at,"
                " stored_rel_path, format, vertex_count, face_count, bbox_json, parse_json, match_json,"
                " declared_stage_json, ingest_state, quarantine_reason, quarantine_detail_json, quarantined_at,"
                " updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    scan_id,
                    sha256,
                    original_name,
                    int(size_bytes),
                    station_id,
                    now,
                    stored_rel_path,
                    format,
                    vertex_count,
                    face_count,
                    _dumps(bbox),
                    _dumps(dict(parse)),
                    _dumps(match_d),
                    _dumps(_check_declared_stage(declared_stage)),
                    ingest_state,
                    quarantine_reason,
                    _dumps(dict(quarantine_detail)) if quarantine_detail is not None else None,
                    now if quarantine_reason else None,
                    now,
                ),
            )
            payload = {"original_name": original_name, "size_bytes": int(size_bytes), "format": format}
            self._event(conn, "SCAN_RECEIVED", station_id=station_id, scan_file_id=scan_id, payload=payload)
            if quarantine_reason is not None:
                self._event(
                    conn,
                    "SCAN_QUARANTINED",
                    station_id=station_id,
                    scan_file_id=scan_id,
                    payload={"original_name": original_name, "reason": quarantine_reason},
                )
            row = conn.execute("SELECT * FROM scan_files WHERE scan_file_id=?", (scan_id,)).fetchone()
        return self._scan_dict(row), True

    def get_scan_file(self, scan_file_id: str) -> dict[str, Any] | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM scan_files WHERE scan_file_id=?", (scan_file_id,)).fetchone()
        return None if row is None else self._scan_dict(row)

    def get_scan_meta(self, scan_file_id: str) -> dict[str, Any] | None:
        """Bookkeeping outside the contract: ``{job_id, quarantine_reason (historic),
        quarantine_detail, quarantined_at, resolution, resolved_at, resolved_by, updated_at}``."""
        with self._read() as conn:
            row = conn.execute("SELECT * FROM scan_files WHERE scan_file_id=?", (scan_file_id,)).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "quarantine_reason": row["quarantine_reason"],
            "quarantine_detail": _loads(row["quarantine_detail_json"]),
            "quarantined_at": row["quarantined_at"],
            "resolution": row["resolution"],
            "resolved_at": row["resolved_at"],
            "resolved_by": _loads(row["resolved_by_json"]),
            "updated_at": row["updated_at"],
        }

    def get_scan_by_sha(self, sha256: str) -> dict[str, Any] | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM scan_files WHERE sha256=?", (sha256,)).fetchone()
        return None if row is None else self._scan_dict(row)

    def list_scans(
        self, *, station_id: str | None = None, ingest_state: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM scan_files WHERE 1=1"
        args: list[Any] = []
        if station_id is not None:
            sql += " AND station_id=?"
            args.append(station_id)
        if ingest_state is not None:
            if ingest_state not in INGEST_STATES:
                raise LineStoreError("INVALID_ARGUMENT", f"unknown ingest_state {ingest_state!r}")
            sql += " AND ingest_state=?"
            args.append(ingest_state)
        sql += " ORDER BY received_at ASC LIMIT ?"
        args.append(max(1, min(int(limit), 1000)))
        with self._read() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._scan_dict(r) for r in rows]

    def set_scan_match(
        self,
        scan_file_id: str,
        match: Mapping[str, Any],
        parse: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update the stored parse/match of a scan (no state change)."""
        match_d = self._check_match(match)
        with self._tx() as conn:
            row = self._scan_row(conn, scan_file_id)
            conn.execute(
                "UPDATE scan_files SET match_json=?, parse_json=COALESCE(?, parse_json), updated_at=? WHERE scan_file_id=?",
                (_dumps(match_d), _dumps(dict(parse)) if parse is not None else None, self._now_iso(), scan_file_id),
            )
            row = self._scan_row(conn, scan_file_id)
        return self._scan_dict(row)

    @staticmethod
    def _scan_row(conn: sqlite3.Connection, scan_file_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM scan_files WHERE scan_file_id=?", (scan_file_id,)).fetchone()
        if row is None:
            raise KeyError(scan_file_id)
        return row

    def quarantine_scan(
        self,
        scan_file_id: str,
        reason: str,
        detail: Mapping[str, Any] | None = None,
        *,
        match: Mapping[str, Any] | None = None,
        parse: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Move a ``received`` scan to ``quarantined`` with a reason (SCAN_QUARANTINED)."""
        if reason not in QUARANTINE_REASONS:
            raise LineStoreError("INVALID_ARGUMENT", f"unknown quarantine reason {reason!r}")
        match_d = self._check_match(match) if match is not None else None
        with self._tx() as conn:
            row = self._scan_row(conn, scan_file_id)
            if row["ingest_state"] != "received":
                raise QuarantineError("SCAN_STATE", f"scan is {row['ingest_state']}, not received")
            now = self._now_iso()
            conn.execute(
                "UPDATE scan_files SET ingest_state='quarantined', quarantine_reason=?, quarantine_detail_json=?,"
                " quarantined_at=?, match_json=COALESCE(?, match_json), parse_json=COALESCE(?, parse_json),"
                " updated_at=? WHERE scan_file_id=?",
                (
                    reason,
                    _dumps(dict(detail)) if detail is not None else None,
                    now,
                    _dumps(match_d),
                    _dumps(dict(parse)) if parse is not None else None,
                    now,
                    scan_file_id,
                ),
            )
            self._event(
                conn,
                "SCAN_QUARANTINED",
                station_id=row["station_id"],
                scan_file_id=scan_file_id,
                payload={"original_name": row["original_name"], "reason": reason},
            )
            row = self._scan_row(conn, scan_file_id)
        return self._scan_dict(row)

    def list_quarantine(self, station_id: str | None = None, include_resolved: bool = False) -> list[dict[str, Any]]:
        """Quarantined scans, newest first.

        Each entry: ``{scan_file, quarantine_reason, quarantine_detail,
        quarantined_at, resolvable, resolution, resolved_at, resolved_by,
        job_id}``. ``resolvable`` is false for content reasons (the bytes are
        unusable); only filename/recipe reasons can be assigned.
        """
        sql = "SELECT * FROM scan_files WHERE quarantine_reason IS NOT NULL"
        args: list[Any] = []
        if not include_resolved:
            sql += " AND ingest_state='quarantined'"
        if station_id is not None:
            sql += " AND station_id=?"
            args.append(station_id)
        sql += " ORDER BY quarantined_at DESC, scan_file_id ASC"
        with self._read() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [
            {
                "scan_file": self._scan_dict(r),
                "quarantine_reason": r["quarantine_reason"],
                "quarantine_detail": _loads(r["quarantine_detail_json"]),
                "quarantined_at": r["quarantined_at"],
                "resolvable": r["ingest_state"] == "quarantined"
                and r["quarantine_reason"] in RESOLVABLE_QUARANTINE_REASONS,
                "resolution": r["resolution"],
                "resolved_at": r["resolved_at"],
                "resolved_by": _loads(r["resolved_by_json"]),
                "job_id": r["job_id"],
            }
            for r in rows
        ]

    def match_and_enqueue(
        self,
        scan_file_id: str,
        *,
        match: Mapping[str, Any],
        sku: str,
        compiled_part_sha256: str,
        parse: Mapping[str, Any] | None = None,
        declared_stage: Mapping[str, Any] | None = None,
        priority: int | None = None,
        max_attempts: int | None = None,
    ) -> tuple[str, bool]:
        """Mark a ``received`` scan MATCHED and enqueue its inspect job, atomically.

        Returns ``(job_id, created)``; the dedupe key
        ``inspect:<scan_sha>:<recipe_sha>`` makes a second call a no-op.
        """
        match_d = self._check_match(match)
        if match_d["status"] != "MATCHED":
            raise LineStoreError("INVALID_ARGUMENT", "match_and_enqueue needs a MATCHED outcome")
        with self._tx() as conn:
            row = self._scan_row(conn, scan_file_id)
            if row["ingest_state"] not in ("received", "enqueued"):
                raise QuarantineError("SCAN_STATE", f"scan is {row['ingest_state']}; cannot enqueue")
            declared = _check_declared_stage(declared_stage) if declared_stage is not None else _loads(
                row["declared_stage_json"]
            )
            job_id, created = self._enqueue(
                conn,
                "inspect",
                sku=sku,
                recipe_id=match_d["recipe_id"],
                recipe_version=int(match_d["recipe_version"]),
                recipe_sha256=match_d["recipe_sha256"],
                compiled_part_sha256=compiled_part_sha256,
                station_id=row["station_id"],
                scan_file_id=scan_file_id,
                scan_sha256=row["sha256"],
                declared_stage=declared,
                priority=priority,
                max_attempts=max_attempts,
                dedupe_key=inspect_dedupe_key(row["sha256"], match_d["recipe_sha256"]),
                retry_of=None,
                principal=None,
            )
            conn.execute(
                "UPDATE scan_files SET ingest_state='enqueued', match_json=?, parse_json=COALESCE(?, parse_json),"
                " job_id=?, updated_at=? WHERE scan_file_id=?",
                (_dumps(match_d), _dumps(dict(parse)) if parse is not None else None, job_id, self._now_iso(), scan_file_id),
            )
        return job_id, created

    def resolve_quarantine(
        self,
        scan_file_id: str,
        recipe_ref: Mapping[str, Any],
        principal: Any,
        *,
        declared_stage: Mapping[str, Any] | None = None,
    ) -> str:
        """Assign a quarantined scan to a recipe and enqueue its inspect job.

        ``recipe_ref`` = ``{recipe_id, recipe_version, recipe_sha256, sku,
        compiled_part_sha256}`` (the caller validated it with
        ``RecipeProvider.load``). Content reasons (truncated, unsupported, ...)
        are refused with ``QuarantineError("NOT_RESOLVABLE")``. Emits
        QUARANTINE_ASSIGNED and JOB_QUEUED; returns the job id (the existing
        one when the same scan + recipe already has a job). A scan that is no
        longer quarantined is ``QuarantineError("SCAN_STATE")``; the station
        picker uses the idempotent :meth:`assign_quarantine` instead.
        """
        job_id, _ = self._assign(scan_file_id, recipe_ref, principal, declared_stage=declared_stage, idempotent=False)
        return job_id

    def assign_quarantine(
        self,
        scan_file_id: str,
        recipe_ref: Mapping[str, Any],
        principal: Any,
        *,
        declared_stage: Mapping[str, Any] | None = None,
        actor_name: str | None = None,
    ) -> tuple[str, bool]:
        """Idempotent, race-safe :meth:`resolve_quarantine` for the station part picker.

        Returns ``(job_id, assigned_now)``. A scan already assigned to the *same*
        recipe (a double click, or two operators picking the same part) returns
        the existing job with ``assigned_now=False`` and writes no event. A scan
        already assigned to another recipe, or dismissed, is
        ``QuarantineError("ALREADY_RESOLVED")``. The check and the write share
        one ``BEGIN IMMEDIATE`` transaction, so concurrent callers serialise.
        """
        return self._assign(
            scan_file_id, recipe_ref, principal, declared_stage=declared_stage, idempotent=True, actor_name=actor_name
        )

    def _assign(
        self,
        scan_file_id: str,
        recipe_ref: Mapping[str, Any],
        principal: Any,
        *,
        declared_stage: Mapping[str, Any] | None,
        idempotent: bool,
        actor_name: str | None = None,
    ) -> tuple[str, bool]:
        who = _principal_dict(principal)
        if who is None:
            raise LineStoreError("INVALID_ARGUMENT", "resolving quarantine needs a principal")
        recipe_id = recipe_ref.get("recipe_id")
        version = recipe_ref.get("recipe_version")
        recipe_sha = _require_sha(recipe_ref.get("recipe_sha256"), "recipe_sha256")
        compiled_sha = _require_sha(recipe_ref.get("compiled_part_sha256"), "compiled_part_sha256")
        sku = recipe_ref.get("sku")
        declared_override = _check_declared_stage(declared_stage) if declared_stage is not None else None
        match_d = self._check_match(
            {
                "status": "MATCHED",
                "recipe_id": recipe_id,
                "recipe_version": version,
                "recipe_sha256": recipe_sha,
                "reason": "OPERATOR_ASSIGNED",
            }
        )
        with self._tx() as conn:
            row = self._scan_row(conn, scan_file_id)
            if row["ingest_state"] != "quarantined":
                if idempotent:
                    previous = _loads(row["match_json"]) or {}
                    if (
                        row["resolution"] == "assigned"
                        and row["job_id"]
                        and previous.get("recipe_sha256") == recipe_sha
                        and previous.get("recipe_id") == recipe_id
                    ):
                        return str(row["job_id"]), False
                    if row["resolution"] in ("assigned", "dismissed"):
                        raise QuarantineError(
                            "ALREADY_RESOLVED",
                            f"scan was already {row['resolution']}"
                            + (f" to {previous.get('recipe_id')}" if row["resolution"] == "assigned" else ""),
                        )
                raise QuarantineError("SCAN_STATE", f"scan is {row['ingest_state']}, not quarantined")
            if row["quarantine_reason"] not in RESOLVABLE_QUARANTINE_REASONS:
                raise QuarantineError(
                    "NOT_RESOLVABLE", f"{row['quarantine_reason']} is a content problem; the scan must be re-exported"
                )
            previous_match = _loads(row["match_json"]) or {}
            declared = declared_override if declared_override is not None else _loads(row["declared_stage_json"])
            job_id, created = self._enqueue(
                conn,
                "inspect",
                sku=str(sku),
                recipe_id=str(recipe_id),
                recipe_version=int(version),  # type: ignore[arg-type]
                recipe_sha256=recipe_sha,
                compiled_part_sha256=compiled_sha,
                station_id=row["station_id"],
                scan_file_id=scan_file_id,
                scan_sha256=row["sha256"],
                declared_stage=declared,
                priority=None,
                max_attempts=None,
                dedupe_key=inspect_dedupe_key(row["sha256"], recipe_sha),
                retry_of=None,
                principal=who,
            )
            now = self._now_iso()
            conn.execute(
                "UPDATE scan_files SET ingest_state='enqueued', match_json=?, resolution='assigned', resolved_at=?,"
                " resolved_by_json=?, job_id=?, updated_at=? WHERE scan_file_id=?",
                (_dumps(match_d), now, _dumps(who), job_id, now, scan_file_id),
            )
            self._event(
                conn,
                "QUARANTINE_ASSIGNED",
                station_id=row["station_id"],
                job_id=job_id,
                scan_file_id=scan_file_id,
                principal=who,
                payload={
                    "recipe_id": recipe_id,
                    "recipe_version": int(version),  # type: ignore[arg-type]
                    "recipe_sha256": recipe_sha,
                    "sku": sku,
                    "compiled_part_sha256": compiled_sha,
                    "previous_reason": row["quarantine_reason"],
                    "from_outcome": previous_match.get("status"),
                    "from_detail": previous_match.get("reason"),
                    "original_name": row["original_name"],
                    "declared_stage": declared,
                    "operator_name": None if actor_name is None else str(actor_name)[:128],
                    "assigned_at": now,
                    "job_created": created,
                },
            )
        return job_id, True

    def dismiss_quarantine(
        self,
        scan_file_id: str,
        principal: Any,
        note: str | None = None,
        *,
        reason: str | None = None,
        idempotent: bool = False,
        actor_name: str | None = None,
    ) -> dict[str, Any]:
        """Dismiss a quarantined scan (QUARANTINE_DISMISSED). The archived bytes are kept.

        ``reason`` is the operator's short reason code (station picker), ``note``
        free text. With ``idempotent=True`` a scan that is already dismissed is
        returned unchanged (no second event); one that was assigned is
        ``QuarantineError("ALREADY_RESOLVED")``.
        """
        who = _principal_dict(principal)
        if who is None:
            raise LineStoreError("INVALID_ARGUMENT", "dismissing quarantine needs a principal")
        with self._tx() as conn:
            row = self._scan_row(conn, scan_file_id)
            if row["ingest_state"] != "quarantined":
                if idempotent and row["resolution"] == "dismissed":
                    return self._scan_dict(row)
                if idempotent and row["resolution"] == "assigned":
                    raise QuarantineError("ALREADY_RESOLVED", "scan was already assigned")
                raise QuarantineError("SCAN_STATE", f"scan is {row['ingest_state']}, not quarantined")
            now = self._now_iso()
            previous_match = _loads(row["match_json"]) or {}
            conn.execute(
                "UPDATE scan_files SET ingest_state='dismissed', resolution='dismissed', resolved_at=?,"
                " resolved_by_json=?, updated_at=? WHERE scan_file_id=?",
                (now, _dumps(who), now, scan_file_id),
            )
            self._event(
                conn,
                "QUARANTINE_DISMISSED",
                station_id=row["station_id"],
                scan_file_id=scan_file_id,
                principal=who,
                payload={
                    "reason": row["quarantine_reason"],
                    "note": (note or "")[:500],
                    "dismiss_reason": None if reason is None else str(reason)[:64],
                    "from_outcome": previous_match.get("status"),
                    "original_name": row["original_name"],
                    "operator_name": None if actor_name is None else str(actor_name)[:128],
                    "dismissed_at": now,
                },
            )
            row = self._scan_row(conn, scan_file_id)
        return self._scan_dict(row)

    # ------------------------------------------------------------------ ingest sources (copy_leave)
    def record_ingest_source(
        self, station_id: str, source_path: str, size_bytes: int, mtime_ns: int, sha256: str
    ) -> None:
        """Remember a source file left in place (``copy_leave``) so it is not re-ingested."""
        _require_sha(sha256, "sha256")
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ingest_sources(station_id, source_path, size_bytes, mtime_ns, sha256,"
                " recorded_at) VALUES (?,?,?,?,?,?)",
                (station_id, source_path, int(size_bytes), int(mtime_ns), sha256, self._now_iso()),
            )

    def has_ingest_source(self, station_id: str, source_path: str, size_bytes: int, mtime_ns: int) -> bool:
        with self._read() as conn:
            row = conn.execute(
                "SELECT 1 FROM ingest_sources WHERE station_id=? AND source_path=? AND size_bytes=? AND mtime_ns=?",
                (station_id, source_path, int(size_bytes), int(mtime_ns)),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------ jobs
    def _job_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        failure = None
        if row["failure_code"] is not None:
            failure = {"code": row["failure_code"], "message": row["failure_message"] or ""}
        return {
            "schema": LINE_JOB_SCHEMA,
            "job_id": row["job_id"],
            "kind": row["kind"],
            "station_id": row["station_id"],
            "scan_file_id": row["scan_file_id"],
            "scan_sha256": row["scan_sha256"],
            "sku": row["sku"],
            "recipe_id": row["recipe_id"],
            "recipe_version": int(row["recipe_version"]),
            "recipe_sha256": row["recipe_sha256"],
            "compiled_part_sha256": row["compiled_part_sha256"],
            "declared_stage": _loads(row["declared_stage_json"]),
            "priority": int(row["priority"]),
            "dedupe_key": row["dedupe_key"],
            "retry_of": row["retry_of"],
            "execution_status": row["execution_status"],
            "stage": row["stage"],
            "progress": int(row["progress"]),
            "attempts": int(row["attempts"]),
            "max_attempts": int(row["max_attempts"]),
            "lease_owner": row["lease_owner"],
            "lease_expires_at": row["lease_expires_at"],
            "cancel_requested": bool(row["cancel_requested"]),
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "failure": failure,
            "engine_bundle_id": row["engine_bundle_id"],
            "workspace_rel": row["workspace_rel"],
            "result_summary": _loads(row["result_summary_json"]),
            "peak_rss_bytes": row["peak_rss_bytes"],
            "duration_s": row["duration_s"],
        }

    @staticmethod
    def _job_row(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return row

    def _enqueue(
        self,
        conn: sqlite3.Connection,
        kind: str,
        *,
        sku: str,
        recipe_id: str,
        recipe_version: int,
        recipe_sha256: str,
        compiled_part_sha256: str | None,
        station_id: str | None,
        scan_file_id: str | None,
        scan_sha256: str | None,
        declared_stage: Mapping[str, Any] | None,
        priority: int | None,
        max_attempts: int | None,
        dedupe_key: str | None,
        retry_of: str | None,
        principal: Any,
    ) -> tuple[str, bool]:
        if kind not in JOB_KINDS:
            raise LineStoreError("INVALID_ARGUMENT", f"unknown job kind {kind!r}")
        m = _RECIPE_ID_RE.match(recipe_id or "")
        if not m or m.group("sku") != sku or int(m.group("version")) != int(recipe_version):
            raise LineStoreError("INVALID_ARGUMENT", "recipe_id must be '<sku>@v<recipe_version>'")
        _require_sha(recipe_sha256, "recipe_sha256")
        if compiled_part_sha256 is not None:
            _require_sha(compiled_part_sha256, "compiled_part_sha256")
        if kind == "inspect":
            if scan_file_id is None or scan_sha256 is None:
                raise LineStoreError("INVALID_ARGUMENT", "inspect jobs name their scan")
            if compiled_part_sha256 is None:
                raise LineStoreError("INVALID_ARGUMENT", "inspect jobs pin their compiled_part_sha256")
        if scan_sha256 is not None:
            _require_sha(scan_sha256, "scan_sha256")
            if scan_file_id != scan_file_id_for(scan_sha256):
                raise LineStoreError("INVALID_ARGUMENT", "scan_file_id must match scan_sha256")
        if retry_of is not None and not _JOB_ID_RE.match(retry_of):
            raise LineStoreError("INVALID_ARGUMENT", "retry_of must be a job id")
        if dedupe_key is not None:
            if not dedupe_key or len(dedupe_key) > 200:
                raise LineStoreError("INVALID_ARGUMENT", "dedupe_key must be 1-200 characters")
            hit = conn.execute("SELECT job_id FROM jobs WHERE dedupe_key=?", (dedupe_key,)).fetchone()
            if hit is not None:
                return str(hit["job_id"]), False
        prio = PRIORITY_BY_KIND[kind] if priority is None else int(priority)
        attempts_max = MAX_ATTEMPTS_BY_KIND[kind] if max_attempts is None else int(max_attempts)
        if attempts_max < 1:
            raise LineStoreError("INVALID_ARGUMENT", "max_attempts must be >= 1")
        job_id = new_job_id()
        now = self._now_iso()
        conn.execute(
            "INSERT INTO jobs(job_id, kind, station_id, scan_file_id, scan_sha256, sku, recipe_id, recipe_version,"
            " recipe_sha256, compiled_part_sha256, declared_stage_json, priority, dedupe_key, retry_of,"
            " execution_status, stage, progress, attempts, max_attempts, cancel_requested, created_at,"
            " workspace_rel, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'queued','queued',0,0,?,0,?,?,?)",
            (
                job_id,
                kind,
                station_id,
                scan_file_id,
                scan_sha256,
                sku,
                recipe_id,
                int(recipe_version),
                recipe_sha256,
                compiled_part_sha256,
                _dumps(_check_declared_stage(declared_stage)),
                prio,
                dedupe_key,
                retry_of,
                attempts_max,
                now,
                f"jobs/{job_id}",
                now,
            ),
        )
        self._event(
            conn,
            "JOB_QUEUED",
            station_id=station_id,
            job_id=job_id,
            scan_file_id=scan_file_id,
            principal=principal,
            payload={"kind": kind, "recipe_id": recipe_id, "priority": prio, "retry_of": retry_of},
        )
        return job_id, True

    def enqueue_job(
        self,
        kind: str,
        *,
        sku: str,
        recipe_id: str,
        recipe_version: int,
        recipe_sha256: str,
        compiled_part_sha256: str | None = None,
        station_id: str | None = None,
        scan_file_id: str | None = None,
        scan_sha256: str | None = None,
        declared_stage: Mapping[str, Any] | None = None,
        priority: int | None = None,
        max_attempts: int | None = None,
        dedupe_key: str | None = None,
        retry_of: str | None = None,
        principal: Any = None,
    ) -> tuple[str, bool]:
        """Enqueue a job; returns ``(job_id, created)``. A dedupe hit returns the existing id."""
        with self._tx() as conn:
            return self._enqueue(
                conn,
                kind,
                sku=sku,
                recipe_id=recipe_id,
                recipe_version=recipe_version,
                recipe_sha256=recipe_sha256,
                compiled_part_sha256=compiled_part_sha256,
                station_id=station_id,
                scan_file_id=scan_file_id,
                scan_sha256=scan_sha256,
                declared_stage=declared_stage,
                priority=priority,
                max_attempts=max_attempts,
                dedupe_key=dedupe_key,
                retry_of=retry_of,
                principal=principal,
            )

    def lease_next(self, owner: str, ttl_s: float, engine_bundle_id: str | None = None) -> dict[str, Any] | None:
        """Lease the highest-priority, oldest queued job (JOB_STARTED); None when idle."""
        if not owner:
            raise LineStoreError("INVALID_ARGUMENT", "owner is required")
        now_dt = self._now()
        with self._tx() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE execution_status='queued' AND cancel_requested=0"
                " ORDER BY priority DESC, created_at ASC, rowid ASC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            now = iso_utc(now_dt)
            expires = iso_utc(now_dt + timedelta(seconds=float(ttl_s)))
            attempt = int(row["attempts"]) + 1
            conn.execute(
                "UPDATE jobs SET execution_status='running', stage='preparing', progress=0, progress_message='',"
                " attempts=?, lease_owner=?, lease_expires_at=?, started_at=?, finished_at=NULL, failure_code=NULL,"
                " failure_message=NULL, engine_bundle_id=COALESCE(?, engine_bundle_id), updated_at=? WHERE job_id=?",
                (attempt, owner[:128], expires, now, engine_bundle_id, now, row["job_id"]),
            )
            self._event(
                conn,
                "JOB_STARTED",
                station_id=row["station_id"],
                job_id=row["job_id"],
                scan_file_id=row["scan_file_id"],
                payload={"attempt": attempt, "max_attempts": int(row["max_attempts"]), "owner": owner[:128]},
            )
            row = self._job_row(conn, row["job_id"])
        return self._job_dict(row)

    def renew_lease(self, job_id: str, owner: str, ttl_s: float) -> bool:
        now_dt = self._now()
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE jobs SET lease_expires_at=?, updated_at=? WHERE job_id=? AND lease_owner=?"
                " AND execution_status='running'",
                (iso_utc(now_dt + timedelta(seconds=float(ttl_s))), iso_utc(now_dt), job_id, owner),
            )
            return cur.rowcount == 1

    def update_progress(self, job_id: str, owner: str, stage: str, progress: float, message: str = "") -> bool:
        """Record progress for a leased job; JOB_PROGRESS at most once per second per job."""
        if stage not in _RUNNING_STAGES:
            raise LineStoreError("INVALID_ARGUMENT", f"progress stage must be one of {sorted(_RUNNING_STAGES)}")
        pct = max(0, min(99, int(round(float(progress)))))
        text = " ".join(str(message or "").split())[:500]
        now_dt = self._now()
        with self._tx() as conn:
            row = conn.execute(
                "SELECT station_id FROM jobs WHERE job_id=? AND lease_owner=? AND execution_status='running'",
                (job_id, owner),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "UPDATE jobs SET stage=?, progress=?, progress_message=?, updated_at=? WHERE job_id=?",
                (stage, pct, text, iso_utc(now_dt), job_id),
            )
            with self._progress_lock:
                last = self._last_progress_event.get(job_id)
                emit = last is None or now_dt - last >= _PROGRESS_EVENT_MIN_INTERVAL
                if emit:
                    self._last_progress_event[job_id] = now_dt
            if emit:
                self._event(
                    conn,
                    "JOB_PROGRESS",
                    station_id=row["station_id"],
                    job_id=job_id,
                    payload={"stage": stage, "progress": pct, "message": text},
                )
        return True

    def _finish(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        status: str,
        *,
        failure_code: str | None,
        failure_message: str | None,
        result_summary: Mapping[str, Any] | None,
        peak_rss_bytes: int | None,
        duration_s: float | None,
        payload: Mapping[str, Any],
        principal: Any = None,
    ) -> None:
        now = self._now_iso()
        conn.execute(
            "UPDATE jobs SET execution_status=?, stage=CASE WHEN ?='completed' THEN 'done' ELSE stage END,"
            " progress=CASE WHEN ?='completed' THEN 100 ELSE progress END, lease_owner=NULL,"
            " lease_expires_at=NULL, finished_at=?, failure_code=?, failure_message=?,"
            " result_summary_json=COALESCE(?, result_summary_json), peak_rss_bytes=COALESCE(?, peak_rss_bytes),"
            " duration_s=COALESCE(?, duration_s), updated_at=? WHERE job_id=?",
            (
                status,
                status,
                status,
                now,
                failure_code,
                None if failure_message is None else failure_message[:2000],
                _dumps(dict(result_summary)) if result_summary is not None else None,
                None if peak_rss_bytes is None else int(peak_rss_bytes),
                None if duration_s is None else max(0.0, float(duration_s)),
                now,
                row["job_id"],
            ),
        )
        kind = {"completed": "JOB_COMPLETED", "failed": "JOB_FAILED", "cancelled": "JOB_CANCELLED"}[status]
        self._event(
            conn,
            kind,
            station_id=row["station_id"],
            job_id=row["job_id"],
            scan_file_id=row["scan_file_id"],
            principal=principal,
            payload=dict(payload),
        )
        with self._progress_lock:
            self._last_progress_event.pop(row["job_id"], None)

    def _extra_event(
        self, conn: sqlite3.Connection, row: sqlite3.Row, extra: tuple[str, Mapping[str, Any]] | None
    ) -> None:
        if extra is None:
            return
        kind, payload = extra
        self._event(
            conn,
            kind,
            station_id=row["station_id"],
            job_id=row["job_id"],
            scan_file_id=row["scan_file_id"],
            payload=payload,
        )

    def _leased_row(self, conn: sqlite3.Connection, job_id: str, owner: str) -> sqlite3.Row | None:
        row = self._job_row(conn, job_id)
        if row["execution_status"] != "running" or row["lease_owner"] != owner:
            return None
        return row

    def complete_job(
        self,
        job_id: str,
        owner: str,
        result_summary: Mapping[str, Any] | None,
        peak_rss_bytes: int | None = None,
        duration_s: float | None = None,
        *,
        extra_event: tuple[str, Mapping[str, Any]] | None = None,
    ) -> bool:
        """Mark a leased job completed (JOB_COMPLETED). False when the lease was lost.

        ``extra_event=(kind, payload)`` (e.g. RECIPE_COMPILED) is written in the
        same transaction, just before JOB_COMPLETED, and only if the lease held.
        """
        summary = _check_result_summary(result_summary)
        with self._tx() as conn:
            row = self._leased_row(conn, job_id, owner)
            if row is None:
                return False
            self._extra_event(conn, row, extra_event)
            self._finish(
                conn,
                row,
                "completed",
                failure_code=None,
                failure_message=None,
                result_summary=summary,
                peak_rss_bytes=peak_rss_bytes,
                duration_s=duration_s,
                payload={
                    "kind": row["kind"],
                    "summary_state": None if summary is None else summary["summary_state"],
                    "inspection_validity": None if summary is None else summary["inspection_validity"],
                    "duration_s": duration_s,
                },
            )
        return True

    def fail_job(
        self,
        job_id: str,
        owner: str,
        code: str,
        message: str,
        peak_rss_bytes: int | None = None,
        duration_s: float | None = None,
        *,
        extra_event: tuple[str, Mapping[str, Any]] | None = None,
    ) -> bool:
        """Mark a leased job failed with a FAILURE_CODE (JOB_FAILED). False when the lease was lost.

        ``extra_event`` as in :meth:`complete_job` (e.g. RECIPE_COMPILE_FAILED).
        """
        if code not in FAILURE_CODES:
            raise LineStoreError("INVALID_ARGUMENT", f"unknown failure code {code!r}")
        with self._tx() as conn:
            row = self._leased_row(conn, job_id, owner)
            if row is None:
                return False
            self._extra_event(conn, row, extra_event)
            self._finish(
                conn,
                row,
                "failed",
                failure_code=code,
                failure_message=message or "",
                result_summary=None,
                peak_rss_bytes=peak_rss_bytes,
                duration_s=duration_s,
                payload={"kind": row["kind"], "code": code, "message": (message or "")[:500]},
            )
        return True

    def mark_cancelled(
        self, job_id: str, owner: str, peak_rss_bytes: int | None = None, duration_s: float | None = None
    ) -> bool:
        with self._tx() as conn:
            row = self._leased_row(conn, job_id, owner)
            if row is None:
                return False
            self._finish(
                conn,
                row,
                "cancelled",
                failure_code=None,
                failure_message=None,
                result_summary=None,
                peak_rss_bytes=peak_rss_bytes,
                duration_s=duration_s,
                payload={"kind": row["kind"], "while": "running"},
            )
        return True

    def request_cancel(self, job_id: str, principal: Any) -> str:
        """Request cancellation. Returns ``cancelled`` (a queued job, cancelled now),
        ``cancel_requested`` (a running job; the runner stops it) or
        ``not_cancellable`` (already terminal). ``KeyError`` for an unknown job."""
        who = _principal_dict(principal)
        with self._tx() as conn:
            row = self._job_row(conn, job_id)
            status = row["execution_status"]
            if status in TERMINAL_STATUSES:
                return "not_cancellable"
            if not row["cancel_requested"]:
                conn.execute(
                    "UPDATE jobs SET cancel_requested=1, updated_at=? WHERE job_id=?", (self._now_iso(), job_id)
                )
                self._event(
                    conn,
                    "JOB_CANCEL_REQUESTED",
                    station_id=row["station_id"],
                    job_id=job_id,
                    scan_file_id=row["scan_file_id"],
                    principal=who,
                    payload={"status": status},
                )
            if status == "queued":
                self._finish(
                    conn,
                    row,
                    "cancelled",
                    failure_code=None,
                    failure_message=None,
                    result_summary=None,
                    peak_rss_bytes=None,
                    duration_s=None,
                    payload={"kind": row["kind"], "while": "queued"},
                    principal=who,
                )
                return "cancelled"
            return "cancel_requested"

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._read() as conn:
            row = conn.execute("SELECT cancel_requested FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def recover_expired_leases(self) -> list[tuple[str, str]]:
        """Recover running jobs whose lease has expired (a crashed runner).

        A job with attempts left is requeued; one with none left fails with
        WORKER_INTERRUPTED; one whose cancel was requested is cancelled. Each
        recovery emits JOB_RECOVERED. Returns ``[(job_id, action)]`` with action
        ``requeued`` | ``failed`` | ``cancelled``.
        """
        now = self._now_iso()
        out: list[tuple[str, str]] = []
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE execution_status='running' AND (lease_expires_at IS NULL OR lease_expires_at < ?)"
                " ORDER BY created_at ASC",
                (now,),
            ).fetchall()
            for row in rows:
                attempts, max_attempts = int(row["attempts"]), int(row["max_attempts"])
                if row["cancel_requested"]:
                    action = "cancelled"
                elif attempts < max_attempts:
                    action = "requeued"
                else:
                    action = "failed"
                self._event(
                    conn,
                    "JOB_RECOVERED",
                    station_id=row["station_id"],
                    job_id=row["job_id"],
                    scan_file_id=row["scan_file_id"],
                    payload={
                        "action": action,
                        "attempts": attempts,
                        "max_attempts": max_attempts,
                        "previous_owner": row["lease_owner"],
                        "stage": row["stage"],
                    },
                )
                if action == "requeued":
                    conn.execute(
                        "UPDATE jobs SET execution_status='queued', stage='queued', progress=0, progress_message='',"
                        " lease_owner=NULL, lease_expires_at=NULL, updated_at=? WHERE job_id=?",
                        (now, row["job_id"]),
                    )
                elif action == "failed":
                    self._finish(
                        conn,
                        row,
                        "failed",
                        failure_code="WORKER_INTERRUPTED",
                        failure_message=f"lease expired during stage {row['stage']} after {attempts} attempt(s)",
                        result_summary=None,
                        peak_rss_bytes=None,
                        duration_s=None,
                        payload={"kind": row["kind"], "code": "WORKER_INTERRUPTED", "message": "lease expired"},
                    )
                else:
                    self._finish(
                        conn,
                        row,
                        "cancelled",
                        failure_code=None,
                        failure_message=None,
                        result_summary=None,
                        peak_rss_bytes=None,
                        duration_s=None,
                        payload={"kind": row["kind"], "while": "recovery"},
                    )
                out.append((str(row["job_id"]), action))
        return out

    def expire_leases_of(self, owners: Iterable[str]) -> list[str]:
        """Expire the leases of running jobs held by ``owners`` right now.

        For a runner that knows those owners are dead processes (its own
        previous incarnation on this host); :meth:`recover_expired_leases`
        then recovers the jobs without waiting for the TTL. Returns the job ids.
        """
        owner_list = sorted(set(owners))
        if not owner_list:
            return []
        past = iso_utc(self._now() - timedelta(milliseconds=1))
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT job_id FROM jobs WHERE execution_status='running' AND lease_owner IN ("
                + ",".join("?" * len(owner_list))
                + ")",
                owner_list,
            ).fetchall()
            ids = [str(r["job_id"]) for r in rows]
            for job_id in ids:
                conn.execute("UPDATE jobs SET lease_expires_at=? WHERE job_id=?", (past, job_id))
        return ids

    def request_retry(self, job_id: str, principal: Any) -> str:
        """Enqueue a new attempt of a failed or cancelled job (JOB_RETRY_REQUESTED).

        The new job has ``retry_of=job_id`` and dedupe ``retry:<job_id>:<n>``.
        If a retry of this job is already queued or running, its id is
        returned instead of creating another. ``JobStateError`` otherwise.
        """
        who = _principal_dict(principal)
        with self._tx() as conn:
            row = self._job_row(conn, job_id)
            if row["execution_status"] not in ("failed", "cancelled"):
                raise JobStateError("JOB_STATE", f"only failed or cancelled jobs are retried (job is {row['execution_status']})")
            active = conn.execute(
                "SELECT job_id FROM jobs WHERE retry_of=? AND execution_status IN ('queued','running')"
                " ORDER BY created_at DESC LIMIT 1",
                (job_id,),
            ).fetchone()
            if active is not None:
                return str(active["job_id"])
            n = 1 + int(conn.execute("SELECT COUNT(*) AS c FROM jobs WHERE retry_of=?", (job_id,)).fetchone()["c"])
            new_id, _created = self._enqueue(
                conn,
                row["kind"],
                sku=row["sku"],
                recipe_id=row["recipe_id"],
                recipe_version=int(row["recipe_version"]),
                recipe_sha256=row["recipe_sha256"],
                compiled_part_sha256=row["compiled_part_sha256"],
                station_id=row["station_id"],
                scan_file_id=row["scan_file_id"],
                scan_sha256=row["scan_sha256"],
                declared_stage=_loads(row["declared_stage_json"]),
                priority=int(row["priority"]),
                max_attempts=int(row["max_attempts"]),
                dedupe_key=f"retry:{job_id}:{n}",
                retry_of=job_id,
                principal=who,
            )
            self._event(
                conn,
                "JOB_RETRY_REQUESTED",
                station_id=row["station_id"],
                job_id=new_id,
                scan_file_id=row["scan_file_id"],
                principal=who,
                payload={"retry_of": job_id, "n": n, "previous_status": row["execution_status"]},
            )
            if row["scan_file_id"] is not None:
                conn.execute(
                    "UPDATE scan_files SET job_id=?, updated_at=? WHERE scan_file_id=?",
                    (new_id, self._now_iso(), row["scan_file_id"]),
                )
        return new_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return None if row is None else self._job_dict(row)

    def get_job_meta(self, job_id: str) -> dict[str, Any] | None:
        """Bookkeeping outside the contract: ``{progress_message, updated_at}``."""
        with self._read() as conn:
            row = conn.execute("SELECT progress_message, updated_at FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return None if row is None else {"progress_message": row["progress_message"], "updated_at": row["updated_at"]}

    def list_jobs(
        self,
        station_id: str | None = None,
        statuses: Iterable[str] | None = None,
        sku: str | None = None,
        limit: int = 50,
        before: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """Jobs newest first. ``before`` is a ``created_at`` cursor (exclusive)."""
        sql = "SELECT * FROM jobs WHERE 1=1"
        args: list[Any] = []
        if station_id is not None:
            sql += " AND station_id=?"
            args.append(station_id)
        status_list = sorted(set(statuses)) if statuses is not None else None
        if status_list is not None:
            bad = set(status_list) - EXECUTION_STATUSES
            if bad:
                raise LineStoreError("INVALID_ARGUMENT", f"unknown statuses {sorted(bad)}")
            if not status_list:
                return []
            sql += " AND execution_status IN (" + ",".join("?" * len(status_list)) + ")"
            args.extend(status_list)
        if sku is not None:
            sql += " AND sku=?"
            args.append(sku)
        if kind is not None:
            sql += " AND kind=?"
            args.append(kind)
        if before is not None:
            sql += " AND created_at < ?"
            args.append(before)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        with self._read() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._job_dict(r) for r in rows]

    def latest_job_for_station(
        self, station_id: str, statuses: Iterable[str] = ("completed",)
    ) -> dict[str, Any] | None:
        """Most recent job of a station in ``statuses`` (by finish time, then creation)."""
        status_list = sorted(set(statuses))
        if not status_list:
            return None
        with self._read() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE station_id=? AND execution_status IN ("
                + ",".join("?" * len(status_list))
                + ") ORDER BY COALESCE(finished_at, started_at, created_at) DESC, rowid DESC LIMIT 1",
                (station_id, *status_list),
            ).fetchone()
        return None if row is None else self._job_dict(row)

    def counts(self) -> dict[str, Any]:
        """Queue depth by status, open quarantine and failures in the last 24 h."""
        since = iso_utc(self._now() - timedelta(hours=24))
        with self._read() as conn:
            by_status = {s: 0 for s in sorted(EXECUTION_STATUSES)}
            for row in conn.execute("SELECT execution_status AS s, COUNT(*) AS c FROM jobs GROUP BY execution_status"):
                by_status[row["s"]] = int(row["c"])
            open_q = int(
                conn.execute("SELECT COUNT(*) AS c FROM scan_files WHERE ingest_state='quarantined'").fetchone()["c"]
            )
            failed_24h = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM jobs WHERE execution_status='failed' AND finished_at >= ?", (since,)
                ).fetchone()["c"]
            )
            pending_scans = int(
                conn.execute("SELECT COUNT(*) AS c FROM scan_files WHERE ingest_state='received'").fetchone()["c"]
            )
        return {
            "jobs_by_status": by_status,
            "queue_depth": by_status["queued"],
            "running": by_status["running"],
            "open_quarantine": open_q,
            "failed_last_24h": failed_24h,
            "scans_pending_resolution": pending_scans,
        }

    # ------------------------------------------------------------------ heartbeats
    def put_heartbeat(self, component: str, instance_id: str, details: Mapping[str, Any] | None = None) -> None:
        if not component or not instance_id:
            raise LineStoreError("INVALID_ARGUMENT", "component and instance_id are required")
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO heartbeats(component, instance_id, pid, host, ts, details_json) VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(component) DO UPDATE SET instance_id=excluded.instance_id, pid=excluded.pid,"
                " host=excluded.host, ts=excluded.ts, details_json=excluded.details_json",
                (
                    component[:128],
                    instance_id[:128],
                    os.getpid(),
                    socket.gethostname()[:255],
                    self._now_iso(),
                    _dumps(dict(details or {})),
                ),
            )

    def get_heartbeats(self) -> list[dict[str, Any]]:
        """All heartbeats with ``age_s`` relative to the store clock."""
        now = self._now()
        with self._read() as conn:
            rows = conn.execute("SELECT * FROM heartbeats ORDER BY component").fetchall()
        return [
            {
                "component": r["component"],
                "instance_id": r["instance_id"],
                "pid": int(r["pid"]),
                "host": r["host"],
                "ts": r["ts"],
                "age_s": max(0.0, (now - parse_iso_utc(r["ts"])).total_seconds()),
                "details": _loads(r["details_json"]) or {},
            }
            for r in rows
        ]
