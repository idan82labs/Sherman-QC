"""Durable SQLite recipe store (C05): immutable, versioned PartRecipeV1 releases.

Everything the release gate decides is recorded here, in ``<data_root>/db/recipes.sqlite3``:

- ``sku_registry`` maps a canonical SKU to exactly one catalog part
  (``"abc 1"`` and ``"ABC-1"`` collide: SKU_COLLISION). A part being created
  first *reserves* its SKU (:meth:`RecipeStore.reserve_sku`, placeholder part
  id ``reserved:<token>``), then binds it to the new catalog id; a reservation
  older than :data:`SKU_RESERVATION_TTL_S` (a crashed creator) is ignored and
  reclaimed.
- ``part_recipes`` holds every released recipe as its canonical JSON bytes.
  Rows are never updated except for retirement; ``(sku, version)`` and
  ``recipe_sha256`` are unique, and a new version must be ``max + 1``
  (CONCURRENT_RELEASE otherwise).
- ``compiled_parts`` records the E1 compile installed for a recipe
  (``recipes/compiled/<recipe_sha256>/``): COMPILED, BLOCKED or FAILED, with
  the ``compiled_part_sha256`` jobs pin.
- ``compiled_history`` (schema v2) keeps every compile that is *not* the
  installed one: a COMPILED install superseded by a recompute (moved to
  ``recipes/compiled_history/<recipe_sha256>/<compiled_part_sha256>/``, so a
  job pinned to it can still load it) and a recompute that came back BLOCKED
  (kept for the wizard; the working install stays in place).
- ``recipe_expectations`` holds the nominal expectations derived from that
  compile, keyed by ``(recipe_sha256, algo_version)`` (algo version = E1
  compiler version).
- ``benchmark_references`` holds one row per released recipe with the
  per-bend nominals (plan §10 amendment 10).
- ``release_drafts`` holds the per-part ``ReleaseRequestV1`` the wizard edits,
  with an optimistic revision (RELEASE_DRAFT_REVISION_CHANGED).
- ``compile_attempts`` logs every release compile (including refused ones)
  with the attempted recipe's ``content_sha256`` (schema v2), so the
  onboarding checklist can tell whether the last compile describes the
  release the operator would make now.
- ``part_flats`` holds the flat (DXF / flat STEP) attached to a part during
  onboarding, its parse result, the E1 binding proposal and the operator's
  binding decision, with an optimistic revision (FLAT_REVISION_CHANGED).
  Revisions are monotonic per part across deletes (``part_flat_revisions``,
  schema v2), so a stale tab never matches a re-uploaded flat.

Design rules (the same as the C02 line store): each public call opens and
closes its own connection (thread-safe, multi-process safe), ``journal_mode=WAL``,
``synchronous=NORMAL``, ``busy_timeout=5000``; every multi-row change runs in
one ``BEGIN IMMEDIATE`` transaction. Timestamps are fixed-width UTC ISO-8601
with milliseconds and ``Z``. Recipes are validated as ``PartRecipeV1`` (hashes
recomputed) before they are stored and whenever they are loaded.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_COMPONENT",
    "SCHEMA_VERSION",
    "RECIPE_STORE_ERROR_CODES",
    "COMPILE_STATUSES",
    "EXPECTATION_STATUSES",
    "FLAT_STATES",
    "RecipeStoreError",
    "StoredRecipe",
    "CompiledRecord",
    "ReleaseDraftRecord",
    "PartFlatRecord",
    "CompileAttemptRecord",
    "CompiledHistoryRecord",
    "COMPILED_HISTORY_REASONS",
    "SKU_RESERVATION_PREFIX",
    "SKU_RESERVATION_TTL_S",
    "RecipeStore",
    "KEEP",
    "parse_recipe_id",
    "utc_now",
    "iso_utc",
]

SCHEMA_COMPONENT = "recipe_store"
SCHEMA_VERSION = 2

RECIPE_STORE_ERROR_CODES: frozenset[str] = frozenset(
    {
        "SKU_COLLISION",
        "CONCURRENT_RELEASE",
        "RELEASE_DRAFT_REVISION_CHANGED",
        "FLAT_REVISION_CHANGED",
        "NOT_FOUND",
        "INVALID_ARGUMENT",
        "SCHEMA_TOO_NEW",
    }
)
COMPILE_STATUSES: frozenset[str] = frozenset({"PENDING", "COMPILED", "BLOCKED", "FAILED"})
EXPECTATION_STATUSES: frozenset[str] = frozenset({"PENDING", "READY", "FAILED"})
#: Flat intake states: UPLOADED (parsed, nothing decided), PARSE_FAILED,
#: PROPOSED (E1 proposal stored, not confirmed), CONFIRMED (operator confirmed
#: a bend-line map), DECLINED (operator keeps the flat explicitly unbound).
FLAT_STATES: frozenset[str] = frozenset({"UPLOADED", "PARSE_FAILED", "PROPOSED", "CONFIRMED", "DECLINED"})
#: Why a compile sits in ``compiled_history``: SUPERSEDED (a COMPILED install
#: replaced by a newer COMPILED recompute) or RECOMPUTE_BLOCKED (a recompute
#: that blocked; the working install was kept).
COMPILED_HISTORY_REASONS: frozenset[str] = frozenset({"SUPERSEDED", "RECOMPUTE_BLOCKED"})
#: Placeholder ``part_id`` prefix of a reserved (not yet bound) SKU.
SKU_RESERVATION_PREFIX = "reserved:"
#: A reservation older than this (seconds) is a crashed creator's and is reclaimed.
SKU_RESERVATION_TTL_S = 600.0

#: Sentinel for :meth:`RecipeStore.update_part_flat`: keep the stored value.
KEEP: Any = object()

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_RECIPE_ID_RE = re.compile(r"^([A-Z0-9][A-Z0-9._-]{0,63})@v([1-9][0-9]*)$")

_SCHEMA_V1: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS sku_registry (
        sku TEXT PRIMARY KEY,
        part_id TEXT NOT NULL UNIQUE,
        display TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS part_recipes (
        recipe_id TEXT PRIMARY KEY,
        sku TEXT NOT NULL,
        version INTEGER NOT NULL CHECK (version >= 1),
        part_id TEXT NOT NULL,
        recipe_sha256 TEXT NOT NULL UNIQUE,
        content_sha256 TEXT NOT NULL,
        recipe_json BLOB NOT NULL,
        released_at TEXT NOT NULL,
        released_by_json TEXT NOT NULL,
        supersedes_version INTEGER,
        status TEXT NOT NULL CHECK (status IN ('released', 'retired')),
        retired_at TEXT,
        retired_by_json TEXT,
        UNIQUE (sku, version)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_part_recipes_part ON part_recipes(part_id)",
    """CREATE TABLE IF NOT EXISTS release_drafts (
        part_id TEXT PRIMARY KEY,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        updated_by_json TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS compiled_parts (
        recipe_sha256 TEXT PRIMARY KEY,
        status TEXT NOT NULL CHECK (status IN ('PENDING', 'COMPILED', 'BLOCKED', 'FAILED')),
        compiled_part_sha256 TEXT,
        compiler_version TEXT,
        blocker_codes_json TEXT NOT NULL,
        stored_rel_path TEXT,
        detail TEXT,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS recipe_expectations (
        recipe_sha256 TEXT NOT NULL,
        algo_version TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('PENDING', 'READY', 'FAILED')),
        json TEXT,
        job_id TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (recipe_sha256, algo_version)
    )""",
    """CREATE TABLE IF NOT EXISTS benchmark_references (
        recipe_id TEXT PRIMARY KEY,
        recipe_sha256 TEXT NOT NULL,
        sku TEXT NOT NULL,
        registered_at TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('REGISTERED', 'RETIRED')),
        compiled_part_sha256 TEXT,
        algo_version TEXT,
        nominals_json TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS compile_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        part_id TEXT NOT NULL,
        cad_sha256 TEXT,
        draft_sha256 TEXT,
        flat_sha256 TEXT,
        recipe_sha256 TEXT,
        status TEXT NOT NULL,
        codes_json TEXT NOT NULL,
        report_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_compile_attempts_part ON compile_attempts(part_id, id)",
    """CREATE TABLE IF NOT EXISTS part_flats (
        part_id TEXT PRIMARY KEY,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        flat_sha256 TEXT NOT NULL,
        format TEXT NOT NULL CHECK (format IN ('DXF', 'STEP')),
        original_filename TEXT,
        stored_rel_path TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        uploaded_at TEXT NOT NULL,
        uploaded_by_json TEXT,
        state TEXT NOT NULL CHECK (state IN ('UPLOADED', 'PARSE_FAILED', 'PROPOSED', 'CONFIRMED', 'DECLINED')),
        parse_json TEXT,
        proposal_json TEXT,
        binding_json TEXT,
        updated_at TEXT NOT NULL
    )""",
)
#: Schema v2 (C05 review fixes): attempt content hash, compile history, monotonic flat revisions.
_SCHEMA_V2: tuple[str, ...] = (
    "ALTER TABLE compile_attempts ADD COLUMN content_sha256 TEXT",
    """CREATE TABLE IF NOT EXISTS compiled_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recipe_sha256 TEXT NOT NULL,
        compiled_part_sha256 TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('COMPILED', 'BLOCKED')),
        compiler_version TEXT,
        blocker_codes_json TEXT NOT NULL,
        stored_rel_path TEXT,
        reason TEXT NOT NULL CHECK (reason IN ('SUPERSEDED', 'RECOMPUTE_BLOCKED')),
        report_json TEXT,
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_compiled_history_recipe ON compiled_history(recipe_sha256, id)",
    """CREATE TABLE IF NOT EXISTS part_flat_revisions (
        part_id TEXT PRIMARY KEY,
        last_revision INTEGER NOT NULL
    )""",
    "INSERT OR IGNORE INTO part_flat_revisions(part_id, last_revision) SELECT part_id, revision FROM part_flats",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    """Fixed-width UTC ISO-8601 with milliseconds and ``Z`` (as the line store)."""
    if dt.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class RecipeStoreError(Exception):
    """A refused store operation. ``code`` is one of :data:`RECIPE_STORE_ERROR_CODES`."""

    def __init__(self, code: str, message: str = "") -> None:
        if code not in RECIPE_STORE_ERROR_CODES:  # pragma: no cover - programming error
            raise ValueError(f"unknown recipe store error code {code!r}")
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message


def _dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _loads(text: str | bytes | None) -> Any:
    if text is None:
        return None
    return json.loads(text)


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.match(value):
        raise RecipeStoreError("INVALID_ARGUMENT", f"{name} must be 64 lowercase hex characters")
    return value


def _principal_json(principal: Any) -> str | None:
    """``{kind, id, role}`` of a C04 Principal, a contract Principal or a mapping."""
    if principal is None:
        return None
    if hasattr(principal, "audit") and callable(principal.audit):
        return _dumps(principal.audit())
    if hasattr(principal, "model_dump"):
        principal = principal.model_dump(mode="json")
    if isinstance(principal, Mapping):
        pid = principal.get("id", principal.get("user_id"))
        if pid is None:
            raise RecipeStoreError("INVALID_ARGUMENT", "principal needs an id (or user_id)")
        return _dumps({"kind": principal.get("kind", "user"), "id": str(pid), "role": principal.get("role")})
    raise RecipeStoreError("INVALID_ARGUMENT", "principal must be a Principal or a mapping")


@dataclass(frozen=True)
class StoredRecipe:
    """One ``part_recipes`` row. :meth:`recipe` parses and re-validates the bytes."""

    recipe_id: str
    sku: str
    version: int
    part_id: str
    recipe_sha256: str
    content_sha256: str
    recipe_json: bytes
    released_at: str
    released_by: dict[str, Any]
    supersedes_version: int | None
    status: str
    retired_at: str | None
    retired_by: dict[str, Any] | None

    def recipe(self) -> Any:
        """The stored ``PartRecipeV1`` (hashes recomputed on load)."""
        from contracts.production.part_recipe import PartRecipeV1

        return PartRecipeV1.model_validate_json(self.recipe_json)

    def summary(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "sku": self.sku,
            "version": self.version,
            "part_id": self.part_id,
            "recipe_sha256": self.recipe_sha256,
            "content_sha256": self.content_sha256,
            "released_at": self.released_at,
            "released_by": self.released_by,
            "supersedes_version": self.supersedes_version,
            "status": self.status,
            "retired_at": self.retired_at,
        }


@dataclass(frozen=True)
class CompiledRecord:
    """The E1 compile installed for a recipe (``compiled_parts`` row)."""

    recipe_sha256: str
    status: str
    compiled_part_sha256: str | None = None
    compiler_version: str | None = None
    blocker_codes: tuple[str, ...] = ()
    stored_rel_path: str | None = None
    detail: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        _require_sha(self.recipe_sha256, "recipe_sha256")
        if self.status not in COMPILE_STATUSES:
            raise RecipeStoreError("INVALID_ARGUMENT", f"unknown compile status {self.status!r}")
        if self.status in ("COMPILED", "BLOCKED"):
            _require_sha(self.compiled_part_sha256, "compiled_part_sha256")
        if self.status == "COMPILED" and self.blocker_codes:
            raise RecipeStoreError("INVALID_ARGUMENT", "a COMPILED record has no blocker codes")
        if self.status == "BLOCKED" and not self.blocker_codes:
            raise RecipeStoreError("INVALID_ARGUMENT", "a BLOCKED record names its blocker codes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_sha256": self.recipe_sha256,
            "status": self.status,
            "compiled_part_sha256": self.compiled_part_sha256,
            "compiler_version": self.compiler_version,
            "blocker_codes": list(self.blocker_codes),
            "stored_rel_path": self.stored_rel_path,
            "detail": self.detail,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ReleaseDraftRecord:
    part_id: str
    revision: int
    request: dict[str, Any]
    updated_at: str
    updated_by: dict[str, Any] | None


@dataclass(frozen=True)
class CompileAttemptRecord:
    id: int
    part_id: str
    cad_sha256: str | None
    draft_sha256: str | None
    flat_sha256: str | None
    recipe_sha256: str | None
    status: str
    codes: tuple[str, ...]
    report: dict[str, Any]
    created_at: str
    #: The attempted recipe's ``content_sha256`` (None for schema-v1 rows).
    content_sha256: str | None = None


@dataclass(frozen=True)
class CompiledHistoryRecord:
    """A compile that is not (or no longer) the installed one (``compiled_history`` row)."""

    id: int
    recipe_sha256: str
    compiled_part_sha256: str
    status: str
    compiler_version: str | None
    blocker_codes: tuple[str, ...]
    stored_rel_path: str | None
    reason: str
    report: dict[str, Any] | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_sha256": self.recipe_sha256,
            "compiled_part_sha256": self.compiled_part_sha256,
            "status": self.status,
            "compiler_version": self.compiler_version,
            "blocker_codes": list(self.blocker_codes),
            "stored_rel_path": self.stored_rel_path,
            "reason": self.reason,
            "report": self.report,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class PartFlatRecord:
    """The flat attached to a part during onboarding (``part_flats`` row)."""

    part_id: str
    revision: int
    flat_sha256: str
    format: str
    original_filename: str | None
    stored_rel_path: str
    size_bytes: int
    uploaded_at: str
    uploaded_by: dict[str, Any] | None
    state: str
    parse: dict[str, Any] | None
    proposal: dict[str, Any] | None
    binding: dict[str, Any] | None
    updated_at: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "revision": self.revision,
            "flat_sha256": self.flat_sha256,
            "format": self.format,
            "original_filename": self.original_filename,
            "stored_rel_path": self.stored_rel_path,
            "size_bytes": self.size_bytes,
            "uploaded_at": self.uploaded_at,
            "uploaded_by": self.uploaded_by,
            "state": self.state,
            "parse": self.parse,
            "proposal": self.proposal,
            "binding": self.binding,
            "updated_at": self.updated_at,
        }


class RecipeStore:
    """SQLite-backed recipe state (module docstring)."""

    def __init__(self, db_path: str | Path, clock: Callable[[], datetime] = utc_now) -> None:
        self.db_path = Path(db_path)
        self._clock = clock
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    # ------------------------------------------------------------------ plumbing
    def _now_iso(self) -> str:
        return iso_utc(self._clock())

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

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

    def _migrate(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_meta (component TEXT PRIMARY KEY, version INTEGER NOT NULL,"
                    " updated_at TEXT NOT NULL)"
                )
                row = conn.execute("SELECT version FROM schema_meta WHERE component=?", (SCHEMA_COMPONENT,)).fetchone()
                current = int(row["version"]) if row else 0
                if current > SCHEMA_VERSION:
                    raise RecipeStoreError(
                        "SCHEMA_TOO_NEW", f"recipe store schema v{current} is newer than this code (v{SCHEMA_VERSION})"
                    )
                if current < 1:
                    for statement in _SCHEMA_V1:
                        conn.execute(statement)
                if current < 2:
                    for statement in _SCHEMA_V2:
                        conn.execute(statement)
                if current < SCHEMA_VERSION:
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

    # ------------------------------------------------------------------ SKU registry
    def _reservation_expired(self, row: sqlite3.Row) -> bool:
        """True when ``row`` is a SKU reservation older than :data:`SKU_RESERVATION_TTL_S`."""
        if not str(row["part_id"]).startswith(SKU_RESERVATION_PREFIX):
            return False
        try:
            created = datetime.strptime(str(row["created_at"]), "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return True
        return (self._clock() - created).total_seconds() > SKU_RESERVATION_TTL_S

    def _live_sku_row(self, conn: sqlite3.Connection, sku: str) -> sqlite3.Row | None:
        """The registry row of ``sku``; an expired reservation is deleted and reads as absent."""
        row = conn.execute("SELECT part_id, created_at FROM sku_registry WHERE sku=?", (sku,)).fetchone()
        if row is not None and self._reservation_expired(row):
            conn.execute("DELETE FROM sku_registry WHERE sku=? AND part_id=?", (sku, row["part_id"]))
            return None
        return row

    def _register_sku_in(self, conn: sqlite3.Connection, sku: str, part_id: str, display: str, now: str) -> bool:
        by_sku = self._live_sku_row(conn, sku)
        if by_sku is not None:
            if by_sku["part_id"] != part_id:
                if str(by_sku["part_id"]).startswith(SKU_RESERVATION_PREFIX):
                    raise RecipeStoreError("SKU_COLLISION", f"SKU {sku} is being created as another part")
                raise RecipeStoreError("SKU_COLLISION", f"SKU {sku} is registered to another part")
            return False
        by_part = conn.execute("SELECT sku FROM sku_registry WHERE part_id=?", (part_id,)).fetchone()
        if by_part is not None:
            raise RecipeStoreError(
                "SKU_COLLISION", f"part is already registered as SKU {by_part['sku']}, not {sku}"
            )
        conn.execute(
            "INSERT INTO sku_registry(sku, part_id, display, created_at) VALUES (?,?,?,?)",
            (sku, part_id, display[:128] or sku, now),
        )
        return True

    def register_sku(self, sku: str, part_id: str, display: str | None = None) -> bool:
        """Register ``sku`` (canonical) for ``part_id``; True when newly registered.

        Raises SKU_COLLISION when the SKU belongs to another part, or the part
        is already registered under a different SKU.
        """
        from domains.production.recipe.canonical_sku import is_canonical_sku

        if not is_canonical_sku(sku):
            raise RecipeStoreError("INVALID_ARGUMENT", f"{sku!r} is not a canonical SKU")
        if not isinstance(part_id, str) or not part_id:
            raise RecipeStoreError("INVALID_ARGUMENT", "part_id is required")
        with self._tx() as conn:
            return self._register_sku_in(conn, sku, part_id, display or sku, self._now_iso())

    def reserve_sku(self, sku: str, display: str | None = None) -> str:
        """Reserve ``sku`` for a part that is about to be created; returns the token.

        SKU_COLLISION when the SKU is registered (or live-reserved). Bind the
        reservation with :meth:`bind_sku_reservation` once the catalog part
        exists, or drop it with :meth:`release_sku_reservation`.
        """
        import uuid

        from domains.production.recipe.canonical_sku import is_canonical_sku

        if not is_canonical_sku(sku):
            raise RecipeStoreError("INVALID_ARGUMENT", f"{sku!r} is not a canonical SKU")
        token = uuid.uuid4().hex
        with self._tx() as conn:
            if self._live_sku_row(conn, sku) is not None:
                raise RecipeStoreError("SKU_COLLISION", f"SKU {sku} is already registered or being created")
            conn.execute(
                "INSERT INTO sku_registry(sku, part_id, display, created_at) VALUES (?,?,?,?)",
                (sku, SKU_RESERVATION_PREFIX + token, (display or sku)[:128] or sku, self._now_iso()),
            )
        return token

    def bind_sku_reservation(self, sku: str, token: str, part_id: str) -> None:
        """Bind a reservation to the created catalog part (SKU_COLLISION when it was lost)."""
        if not isinstance(part_id, str) or not part_id or part_id.startswith(SKU_RESERVATION_PREFIX):
            raise RecipeStoreError("INVALID_ARGUMENT", "a real part_id is required")
        with self._tx() as conn:
            row = conn.execute("SELECT part_id FROM sku_registry WHERE sku=?", (sku,)).fetchone()
            if row is None or row["part_id"] != SKU_RESERVATION_PREFIX + token:
                raise RecipeStoreError("SKU_COLLISION", f"the reservation of SKU {sku} was lost")
            other = conn.execute("SELECT sku FROM sku_registry WHERE part_id=?", (part_id,)).fetchone()
            if other is not None:
                raise RecipeStoreError("SKU_COLLISION", f"part is already registered as SKU {other['sku']}")
            conn.execute("UPDATE sku_registry SET part_id=?, created_at=? WHERE sku=?", (part_id, self._now_iso(), sku))

    def release_sku_reservation(self, sku: str, token: str) -> bool:
        """Drop an unbound reservation; False when it is not (or no longer) ours."""
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM sku_registry WHERE sku=? AND part_id=?", (sku, SKU_RESERVATION_PREFIX + token))
            return cur.rowcount > 0

    def sku_owner(self, sku: str) -> str | None:
        """The part id ``sku`` is registered to (a live reservation reads ``reserved:<token>``)."""
        with self._read() as conn:
            row = conn.execute("SELECT part_id, created_at FROM sku_registry WHERE sku=?", (sku,)).fetchone()
        if row is None or self._reservation_expired(row):
            return None
        return str(row["part_id"])

    def sku_for_part(self, part_id: str) -> str | None:
        with self._read() as conn:
            row = conn.execute("SELECT sku FROM sku_registry WHERE part_id=?", (part_id,)).fetchone()
        return None if row is None else str(row["sku"])

    def list_registered_skus(self) -> list[str]:
        with self._read() as conn:
            rows = conn.execute("SELECT sku FROM sku_registry ORDER BY sku").fetchall()
        return [str(r["sku"]) for r in rows]

    # ------------------------------------------------------------------ recipes
    @staticmethod
    def _recipe_row(row: sqlite3.Row) -> StoredRecipe:
        blob = row["recipe_json"]
        return StoredRecipe(
            recipe_id=row["recipe_id"],
            sku=row["sku"],
            version=int(row["version"]),
            part_id=row["part_id"],
            recipe_sha256=row["recipe_sha256"],
            content_sha256=row["content_sha256"],
            recipe_json=bytes(blob) if not isinstance(blob, bytes) else blob,
            released_at=row["released_at"],
            released_by=_loads(row["released_by_json"]) or {},
            supersedes_version=row["supersedes_version"],
            status=row["status"],
            retired_at=row["retired_at"],
            retired_by=_loads(row["retired_by_json"]),
        )

    def insert_recipe(
        self,
        recipe: Any,
        *,
        compiled: CompiledRecord | None = None,
        expectations: Mapping[str, Any] | None = None,
        benchmark_nominals: Mapping[str, Any] | None = None,
    ) -> StoredRecipe:
        """Store a finalized ``PartRecipeV1`` and, in the same transaction, its
        compile record, READY expectations and benchmark reference.

        The SKU is registered for the recipe's part when it is not yet
        (SKU_COLLISION when it belongs to another part). ``recipe_version``
        must be the SKU's current maximum + 1, else CONCURRENT_RELEASE (also
        for a duplicate ``recipe_sha256``).
        """
        from contracts.production.canonical import canonical_json_bytes
        from contracts.production.part_recipe import PartRecipeV1

        if not isinstance(recipe, PartRecipeV1):
            raise RecipeStoreError("INVALID_ARGUMENT", "insert_recipe needs a PartRecipeV1")
        if compiled is not None and compiled.recipe_sha256 != recipe.recipe_sha256:
            raise RecipeStoreError("INVALID_ARGUMENT", "compiled record names another recipe")
        if expectations is not None and (compiled is None or compiled.status != "COMPILED"):
            raise RecipeStoreError("INVALID_ARGUMENT", "expectations come from a COMPILED compile")
        blob = canonical_json_bytes(recipe)
        now = self._now_iso()
        with self._tx() as conn:
            self._register_sku_in(conn, recipe.sku, recipe.part_id, recipe.sku_display, now)
            row = conn.execute("SELECT MAX(version) AS m FROM part_recipes WHERE sku=?", (recipe.sku,)).fetchone()
            current_max = int(row["m"] or 0)
            if recipe.recipe_version != current_max + 1:
                raise RecipeStoreError(
                    "CONCURRENT_RELEASE",
                    f"{recipe.sku} is at v{current_max}; cannot insert v{recipe.recipe_version}",
                )
            dup = conn.execute(
                "SELECT recipe_id FROM part_recipes WHERE recipe_sha256=?", (recipe.recipe_sha256,)
            ).fetchone()
            if dup is not None:
                raise RecipeStoreError("CONCURRENT_RELEASE", f"recipe already stored as {dup['recipe_id']}")
            conn.execute(
                "INSERT INTO part_recipes(recipe_id, sku, version, part_id, recipe_sha256, content_sha256, recipe_json,"
                " released_at, released_by_json, supersedes_version, status) VALUES (?,?,?,?,?,?,?,?,?,?, 'released')",
                (
                    recipe.recipe_id,
                    recipe.sku,
                    recipe.recipe_version,
                    recipe.part_id,
                    recipe.recipe_sha256,
                    recipe.content_sha256,
                    blob,
                    recipe.released_at,
                    _dumps(recipe.released_by.model_dump(mode="json")),
                    recipe.supersedes_version,
                ),
            )
            if compiled is not None:
                self._put_compiled_in(conn, compiled, now)
                if expectations is not None:
                    conn.execute(
                        "INSERT OR REPLACE INTO recipe_expectations(recipe_sha256, algo_version, status, json, job_id,"
                        " updated_at) VALUES (?,?, 'READY', ?, NULL, ?)",
                        (recipe.recipe_sha256, str(compiled.compiler_version), _dumps(dict(expectations)), now),
                    )
            if benchmark_nominals is not None:
                conn.execute(
                    "INSERT INTO benchmark_references(recipe_id, recipe_sha256, sku, registered_at, status,"
                    " compiled_part_sha256, algo_version, nominals_json) VALUES (?,?,?,?, 'REGISTERED', ?,?,?)",
                    (
                        recipe.recipe_id,
                        recipe.recipe_sha256,
                        recipe.sku,
                        now,
                        None if compiled is None else compiled.compiled_part_sha256,
                        None if compiled is None else compiled.compiler_version,
                        _dumps(dict(benchmark_nominals)),
                    ),
                )
            stored = conn.execute("SELECT * FROM part_recipes WHERE recipe_id=?", (recipe.recipe_id,)).fetchone()
        return self._recipe_row(stored)

    def get(self, recipe_id: str) -> StoredRecipe | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM part_recipes WHERE recipe_id=?", (recipe_id,)).fetchone()
        return None if row is None else self._recipe_row(row)

    def get_by_sha(self, recipe_sha256: str) -> StoredRecipe | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM part_recipes WHERE recipe_sha256=?", (recipe_sha256,)).fetchone()
        return None if row is None else self._recipe_row(row)

    def get_active(self, sku: str) -> StoredRecipe | None:
        """The latest non-retired version of ``sku`` (None when none exists)."""
        with self._read() as conn:
            row = conn.execute(
                "SELECT * FROM part_recipes WHERE sku=? AND status='released' ORDER BY version DESC LIMIT 1", (sku,)
            ).fetchone()
        return None if row is None else self._recipe_row(row)

    def get_active_for_part(self, part_id: str) -> StoredRecipe | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT * FROM part_recipes WHERE part_id=? AND status='released' ORDER BY version DESC LIMIT 1",
                (part_id,),
            ).fetchone()
        return None if row is None else self._recipe_row(row)

    def max_version(self, sku: str) -> int:
        with self._read() as conn:
            row = conn.execute("SELECT MAX(version) AS m FROM part_recipes WHERE sku=?", (sku,)).fetchone()
        return int(row["m"] or 0)

    def list_versions(self, sku: str) -> list[StoredRecipe]:
        with self._read() as conn:
            rows = conn.execute("SELECT * FROM part_recipes WHERE sku=? ORDER BY version ASC", (sku,)).fetchall()
        return [self._recipe_row(r) for r in rows]

    def list_active(self) -> list[StoredRecipe]:
        """Every SKU's active recipe, ordered by SKU."""
        with self._read() as conn:
            rows = conn.execute(
                "SELECT p.* FROM part_recipes p JOIN (SELECT sku, MAX(version) AS v FROM part_recipes"
                " WHERE status='released' GROUP BY sku) a ON p.sku = a.sku AND p.version = a.v ORDER BY p.sku"
            ).fetchall()
        return [self._recipe_row(r) for r in rows]

    def list_all(self, sku: str | None = None) -> list[StoredRecipe]:
        sql = "SELECT * FROM part_recipes"
        args: tuple[Any, ...] = ()
        if sku is not None:
            sql += " WHERE sku=?"
            args = (sku,)
        sql += " ORDER BY sku, version"
        with self._read() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._recipe_row(r) for r in rows]

    def retire(self, recipe_id: str, principal: Any = None) -> StoredRecipe:
        """Retire a released recipe (its benchmark reference too). Idempotent."""
        now = self._now_iso()
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM part_recipes WHERE recipe_id=?", (recipe_id,)).fetchone()
            if row is None:
                raise RecipeStoreError("NOT_FOUND", f"recipe {recipe_id} does not exist")
            if row["status"] == "released":
                conn.execute(
                    "UPDATE part_recipes SET status='retired', retired_at=?, retired_by_json=? WHERE recipe_id=?",
                    (now, _principal_json(principal), recipe_id),
                )
                conn.execute("UPDATE benchmark_references SET status='RETIRED' WHERE recipe_id=?", (recipe_id,))
            row = conn.execute("SELECT * FROM part_recipes WHERE recipe_id=?", (recipe_id,)).fetchone()
        return self._recipe_row(row)

    # ------------------------------------------------------------------ compiled parts
    @staticmethod
    def _put_compiled_in(conn: sqlite3.Connection, record: CompiledRecord, now: str) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO compiled_parts(recipe_sha256, status, compiled_part_sha256, compiler_version,"
            " blocker_codes_json, stored_rel_path, detail, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                record.recipe_sha256,
                record.status,
                record.compiled_part_sha256,
                record.compiler_version,
                _dumps(sorted(set(record.blocker_codes))),
                record.stored_rel_path,
                None if record.detail is None else str(record.detail)[:2000],
                now,
            ),
        )

    def set_compiled(
        self,
        record: CompiledRecord,
        *,
        expectations: Mapping[str, Any] | None = None,
        benchmark_nominals: Mapping[str, Any] | None = None,
        superseded: CompiledRecord | None = None,
        superseded_rel_path: str | None = None,
    ) -> CompiledRecord:
        """Record a (re)compile for a stored recipe; READY expectations and the
        benchmark nominals are refreshed in the same transaction when given.

        ``superseded`` (the previous COMPILED install, now archived at
        ``superseded_rel_path``) is written to ``compiled_history`` in the same
        transaction (reason SUPERSEDED).
        """
        now = self._now_iso()
        with self._tx() as conn:
            recipe_row = conn.execute(
                "SELECT recipe_id FROM part_recipes WHERE recipe_sha256=?", (record.recipe_sha256,)
            ).fetchone()
            if recipe_row is None:
                raise RecipeStoreError("NOT_FOUND", "no stored recipe has this recipe_sha256")
            if superseded is not None:
                if superseded.status != "COMPILED" or superseded.recipe_sha256 != record.recipe_sha256:
                    raise RecipeStoreError("INVALID_ARGUMENT", "only this recipe's COMPILED install can be superseded")
                self._put_history_in(conn, superseded, "SUPERSEDED", superseded_rel_path, None, now)
            self._put_compiled_in(conn, record, now)
            if expectations is not None:
                if record.status != "COMPILED":
                    raise RecipeStoreError("INVALID_ARGUMENT", "expectations come from a COMPILED compile")
                conn.execute(
                    "INSERT OR REPLACE INTO recipe_expectations(recipe_sha256, algo_version, status, json, job_id,"
                    " updated_at) VALUES (?,?, 'READY', ?, NULL, ?)",
                    (record.recipe_sha256, str(record.compiler_version), _dumps(dict(expectations)), now),
                )
            if benchmark_nominals is not None:
                conn.execute(
                    "UPDATE benchmark_references SET compiled_part_sha256=?, algo_version=?, nominals_json=?"
                    " WHERE recipe_id=?",
                    (
                        record.compiled_part_sha256,
                        record.compiler_version,
                        _dumps(dict(benchmark_nominals)),
                        recipe_row["recipe_id"],
                    ),
                )
        return self.get_compiled(record.recipe_sha256) or record

    @staticmethod
    def _put_history_in(conn: sqlite3.Connection, record: CompiledRecord, reason: str, rel_path: str | None,
                        report: Mapping[str, Any] | None, now: str) -> None:
        if reason not in COMPILED_HISTORY_REASONS:
            raise RecipeStoreError("INVALID_ARGUMENT", f"unknown compile history reason {reason!r}")
        if record.status not in ("COMPILED", "BLOCKED") or record.compiled_part_sha256 is None:
            raise RecipeStoreError("INVALID_ARGUMENT", "compile history holds COMPILED or BLOCKED compiles")
        conn.execute(
            "INSERT INTO compiled_history(recipe_sha256, compiled_part_sha256, status, compiler_version,"
            " blocker_codes_json, stored_rel_path, reason, report_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                record.recipe_sha256,
                record.compiled_part_sha256,
                record.status,
                record.compiler_version,
                _dumps(sorted(set(record.blocker_codes))),
                rel_path,
                reason,
                _dumps(None if report is None else dict(report)),
                now,
            ),
        )

    def record_compiled_history(self, record: CompiledRecord, *, reason: str, stored_rel_path: str | None,
                                report: Mapping[str, Any] | None = None) -> None:
        """Keep a compile that is not installed (e.g. a recompute that came back BLOCKED)."""
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM part_recipes WHERE recipe_sha256=?", (record.recipe_sha256,)).fetchone() is None:
                raise RecipeStoreError("NOT_FOUND", "no stored recipe has this recipe_sha256")
            self._put_history_in(conn, record, reason, stored_rel_path, report, self._now_iso())

    @staticmethod
    def _history_row(row: sqlite3.Row) -> CompiledHistoryRecord:
        return CompiledHistoryRecord(
            id=int(row["id"]),
            recipe_sha256=row["recipe_sha256"],
            compiled_part_sha256=row["compiled_part_sha256"],
            status=row["status"],
            compiler_version=row["compiler_version"],
            blocker_codes=tuple(_loads(row["blocker_codes_json"]) or ()),
            stored_rel_path=row["stored_rel_path"],
            reason=row["reason"],
            report=_loads(row["report_json"]),
            created_at=row["created_at"],
        )

    def list_compiled_history(self, recipe_sha256: str) -> list[CompiledHistoryRecord]:
        """Every non-installed compile of the recipe, oldest first."""
        with self._read() as conn:
            rows = conn.execute(
                "SELECT * FROM compiled_history WHERE recipe_sha256=? ORDER BY id", (recipe_sha256,)
            ).fetchall()
        return [self._history_row(r) for r in rows]

    def find_compiled(self, recipe_sha256: str, compiled_part_sha256: str) -> tuple[str, str | None] | None:
        """Where the recipe's compile ``compiled_part_sha256`` lives: ``(status, stored_rel_path)``
        of the installed record or the latest history row that holds it; None when unknown."""
        current = self.get_compiled(recipe_sha256)
        if current is not None and current.compiled_part_sha256 == compiled_part_sha256:
            return current.status, current.stored_rel_path
        with self._read() as conn:
            row = conn.execute(
                "SELECT * FROM compiled_history WHERE recipe_sha256=? AND compiled_part_sha256=?"
                " AND stored_rel_path IS NOT NULL ORDER BY id DESC LIMIT 1",
                (recipe_sha256, compiled_part_sha256),
            ).fetchone()
        return None if row is None else (str(row["status"]), row["stored_rel_path"])

    def get_compiled(self, recipe_sha256: str) -> CompiledRecord | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM compiled_parts WHERE recipe_sha256=?", (recipe_sha256,)).fetchone()
        if row is None:
            return None
        return CompiledRecord(
            recipe_sha256=row["recipe_sha256"],
            status=row["status"],
            compiled_part_sha256=row["compiled_part_sha256"],
            compiler_version=row["compiler_version"],
            blocker_codes=tuple(_loads(row["blocker_codes_json"]) or ()),
            stored_rel_path=row["stored_rel_path"],
            detail=row["detail"],
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------ expectations
    def set_expectations(
        self,
        recipe_sha256: str,
        algo_version: str,
        status: str,
        expectations: Mapping[str, Any] | None = None,
        job_id: str | None = None,
    ) -> None:
        _require_sha(recipe_sha256, "recipe_sha256")
        if status not in EXPECTATION_STATUSES:
            raise RecipeStoreError("INVALID_ARGUMENT", f"unknown expectations status {status!r}")
        if status == "READY" and expectations is None:
            raise RecipeStoreError("INVALID_ARGUMENT", "READY expectations carry their document")
        if not algo_version:
            raise RecipeStoreError("INVALID_ARGUMENT", "algo_version is required")
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO recipe_expectations(recipe_sha256, algo_version, status, json, job_id, updated_at)"
                " VALUES (?,?,?,?,?,?)",
                (
                    recipe_sha256,
                    algo_version,
                    status,
                    None if expectations is None else _dumps(dict(expectations)),
                    job_id,
                    self._now_iso(),
                ),
            )

    def get_expectations(self, recipe_sha256: str, algo_version: str | None = None) -> dict[str, Any] | None:
        """``{status, algo_version, expectations?, job_id?, updated_at}``; the
        latest updated row when ``algo_version`` is None."""
        sql = "SELECT * FROM recipe_expectations WHERE recipe_sha256=?"
        args: list[Any] = [recipe_sha256]
        if algo_version is not None:
            sql += " AND algo_version=?"
            args.append(algo_version)
        sql += " ORDER BY updated_at DESC, rowid DESC LIMIT 1"
        with self._read() as conn:
            row = conn.execute(sql, args).fetchone()
        if row is None:
            return None
        return {
            "status": row["status"],
            "algo_version": row["algo_version"],
            "expectations": _loads(row["json"]),
            "job_id": row["job_id"],
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------ benchmark references
    def register_benchmark_reference(self, recipe: Any, nominals: Mapping[str, Any], compiled: CompiledRecord | None) -> None:
        """Insert or refresh the benchmark row of a stored recipe."""
        now = self._now_iso()
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM part_recipes WHERE recipe_id=?", (recipe.recipe_id,)).fetchone() is None:
                raise RecipeStoreError("NOT_FOUND", f"recipe {recipe.recipe_id} is not stored")
            conn.execute(
                "INSERT OR REPLACE INTO benchmark_references(recipe_id, recipe_sha256, sku, registered_at, status,"
                " compiled_part_sha256, algo_version, nominals_json) VALUES (?,?,?,?, 'REGISTERED', ?,?,?)",
                (
                    recipe.recipe_id,
                    recipe.recipe_sha256,
                    recipe.sku,
                    now,
                    None if compiled is None else compiled.compiled_part_sha256,
                    None if compiled is None else compiled.compiler_version,
                    _dumps(dict(nominals)),
                ),
            )

    def get_benchmark_reference(self, recipe_id: str) -> dict[str, Any] | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM benchmark_references WHERE recipe_id=?", (recipe_id,)).fetchone()
        return None if row is None else self._benchmark_dict(row)

    def list_benchmark_references(self, sku: str | None = None, include_retired: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM benchmark_references WHERE 1=1"
        args: list[Any] = []
        if sku is not None:
            sql += " AND sku=?"
            args.append(sku)
        if not include_retired:
            sql += " AND status='REGISTERED'"
        sql += " ORDER BY sku, recipe_id"
        with self._read() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._benchmark_dict(r) for r in rows]

    @staticmethod
    def _benchmark_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "recipe_id": row["recipe_id"],
            "recipe_sha256": row["recipe_sha256"],
            "sku": row["sku"],
            "registered_at": row["registered_at"],
            "status": row["status"],
            "compiled_part_sha256": row["compiled_part_sha256"],
            "algo_version": row["algo_version"],
            "nominals": _loads(row["nominals_json"]),
        }

    # ------------------------------------------------------------------ release drafts
    def get_release_draft(self, part_id: str) -> ReleaseDraftRecord | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM release_drafts WHERE part_id=?", (part_id,)).fetchone()
        if row is None:
            return None
        return ReleaseDraftRecord(
            part_id=row["part_id"],
            revision=int(row["revision"]),
            request=_loads(row["json"]) or {},
            updated_at=row["updated_at"],
            updated_by=_loads(row["updated_by_json"]),
        )

    def put_release_draft(
        self, part_id: str, request: Mapping[str, Any], expected_revision: int, updated_by: Any = None
    ) -> ReleaseDraftRecord:
        """Save the part's release draft when its revision is ``expected_revision``
        (0 = none yet); else RELEASE_DRAFT_REVISION_CHANGED. Returns the new row."""
        if not isinstance(expected_revision, int) or isinstance(expected_revision, bool) or expected_revision < 0:
            raise RecipeStoreError("INVALID_ARGUMENT", "expected_revision must be an integer >= 0")
        now = self._now_iso()
        with self._tx() as conn:
            row = conn.execute("SELECT revision FROM release_drafts WHERE part_id=?", (part_id,)).fetchone()
            current = int(row["revision"]) if row else 0
            if current != expected_revision:
                raise RecipeStoreError(
                    "RELEASE_DRAFT_REVISION_CHANGED", f"release draft is at revision {current}, not {expected_revision}"
                )
            conn.execute(
                "INSERT OR REPLACE INTO release_drafts(part_id, revision, json, updated_at, updated_by_json)"
                " VALUES (?,?,?,?,?)",
                (part_id, current + 1, _dumps(dict(request)), now, _principal_json(updated_by)),
            )
        record = self.get_release_draft(part_id)
        assert record is not None
        return record

    # ------------------------------------------------------------------ compile attempts
    def record_compile_attempt(
        self,
        part_id: str,
        *,
        status: str,
        codes: Sequence[str],
        report: Mapping[str, Any],
        cad_sha256: str | None = None,
        draft_sha256: str | None = None,
        flat_sha256: str | None = None,
        recipe_sha256: str | None = None,
        content_sha256: str | None = None,
    ) -> int:
        """Log a compile attempt; ``content_sha256`` is the attempted recipe's content hash."""
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT INTO compile_attempts(part_id, cad_sha256, draft_sha256, flat_sha256, recipe_sha256,"
                " content_sha256, status, codes_json, report_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    part_id,
                    cad_sha256,
                    draft_sha256,
                    flat_sha256,
                    recipe_sha256,
                    content_sha256,
                    status,
                    _dumps(sorted(set(codes))),
                    _dumps(dict(report)),
                    self._now_iso(),
                ),
            )
            return int(cur.lastrowid or 0)

    def last_compile_attempt(self, part_id: str) -> CompileAttemptRecord | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT * FROM compile_attempts WHERE part_id=? ORDER BY id DESC LIMIT 1", (part_id,)
            ).fetchone()
        if row is None:
            return None
        return CompileAttemptRecord(
            id=int(row["id"]),
            part_id=row["part_id"],
            cad_sha256=row["cad_sha256"],
            draft_sha256=row["draft_sha256"],
            flat_sha256=row["flat_sha256"],
            recipe_sha256=row["recipe_sha256"],
            status=row["status"],
            codes=tuple(_loads(row["codes_json"]) or ()),
            report=_loads(row["report_json"]) or {},
            created_at=row["created_at"],
            content_sha256=row["content_sha256"],
        )

    # ------------------------------------------------------------------ part flats
    @staticmethod
    def _flat_row(row: sqlite3.Row) -> PartFlatRecord:
        return PartFlatRecord(
            part_id=row["part_id"],
            revision=int(row["revision"]),
            flat_sha256=row["flat_sha256"],
            format=row["format"],
            original_filename=row["original_filename"],
            stored_rel_path=row["stored_rel_path"],
            size_bytes=int(row["size_bytes"]),
            uploaded_at=row["uploaded_at"],
            uploaded_by=_loads(row["uploaded_by_json"]),
            state=row["state"],
            parse=_loads(row["parse_json"]),
            proposal=_loads(row["proposal_json"]),
            binding=_loads(row["binding_json"]),
            updated_at=row["updated_at"],
        )

    def get_part_flat(self, part_id: str) -> PartFlatRecord | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM part_flats WHERE part_id=?", (part_id,)).fetchone()
        return None if row is None else self._flat_row(row)

    def put_part_flat(
        self,
        part_id: str,
        *,
        flat_sha256: str,
        format: str,
        original_filename: str | None,
        stored_rel_path: str,
        size_bytes: int,
        state: str,
        parse: Mapping[str, Any] | None,
        uploaded_by: Any = None,
    ) -> PartFlatRecord:
        """Attach (or replace) the part's flat; proposal and binding are reset."""
        _require_sha(flat_sha256, "flat_sha256")
        if format not in ("DXF", "STEP"):
            raise RecipeStoreError("INVALID_ARGUMENT", "flat format must be DXF or STEP")
        if state not in ("UPLOADED", "PARSE_FAILED"):
            raise RecipeStoreError("INVALID_ARGUMENT", "a new flat is UPLOADED or PARSE_FAILED")
        now = self._now_iso()
        with self._tx() as conn:
            revision = self._last_flat_revision(conn, part_id) + 1
            conn.execute(
                "INSERT OR REPLACE INTO part_flats(part_id, revision, flat_sha256, format, original_filename,"
                " stored_rel_path, size_bytes, uploaded_at, uploaded_by_json, state, parse_json, proposal_json,"
                " binding_json, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,?)",
                (
                    part_id,
                    revision,
                    flat_sha256,
                    format,
                    None if original_filename is None else original_filename[:255],
                    stored_rel_path,
                    int(size_bytes),
                    now,
                    _principal_json(uploaded_by),
                    state,
                    _dumps(None if parse is None else dict(parse)),
                    now,
                ),
            )
            self._note_flat_revision(conn, part_id, revision)
        record = self.get_part_flat(part_id)
        assert record is not None
        return record

    def update_part_flat(
        self,
        part_id: str,
        expected_revision: int,
        *,
        state: str,
        proposal: Any = KEEP,
        binding: Any = KEEP,
    ) -> PartFlatRecord:
        """Move the flat to ``state`` (optimistic: FLAT_REVISION_CHANGED). :data:`KEEP`
        keeps the stored proposal / binding; ``None`` clears it; a mapping replaces it."""
        if state not in FLAT_STATES:
            raise RecipeStoreError("INVALID_ARGUMENT", f"unknown flat state {state!r}")
        now = self._now_iso()
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM part_flats WHERE part_id=?", (part_id,)).fetchone()
            if row is None:
                raise RecipeStoreError("NOT_FOUND", "the part has no flat")
            if int(row["revision"]) != expected_revision:
                raise RecipeStoreError(
                    "FLAT_REVISION_CHANGED", f"flat is at revision {row['revision']}, not {expected_revision}"
                )
            proposal_json = row["proposal_json"] if proposal is KEEP else _dumps(None if proposal is None else dict(proposal))
            binding_json = row["binding_json"] if binding is KEEP else _dumps(None if binding is None else dict(binding))
            conn.execute(
                "UPDATE part_flats SET revision=?, state=?, proposal_json=?, binding_json=?, updated_at=?"
                " WHERE part_id=?",
                (expected_revision + 1, state, proposal_json, binding_json, now, part_id),
            )
            self._note_flat_revision(conn, part_id, expected_revision + 1)
        record = self.get_part_flat(part_id)
        assert record is not None
        return record

    def delete_part_flat(self, part_id: str, expected_revision: int | None = None) -> bool:
        """Detach the part's flat (FLAT_REVISION_CHANGED when ``expected_revision`` is stale).
        The revision counter survives, so the next upload continues from it."""
        with self._tx() as conn:
            row = conn.execute("SELECT revision FROM part_flats WHERE part_id=?", (part_id,)).fetchone()
            if row is None:
                return False
            if expected_revision is not None and int(row["revision"]) != expected_revision:
                raise RecipeStoreError(
                    "FLAT_REVISION_CHANGED", f"flat is at revision {row['revision']}, not {expected_revision}"
                )
            self._note_flat_revision(conn, part_id, int(row["revision"]))
            conn.execute("DELETE FROM part_flats WHERE part_id=?", (part_id,))
        return True

    @staticmethod
    def _last_flat_revision(conn: sqlite3.Connection, part_id: str) -> int:
        row = conn.execute("SELECT revision FROM part_flats WHERE part_id=?", (part_id,)).fetchone()
        counter = conn.execute("SELECT last_revision FROM part_flat_revisions WHERE part_id=?", (part_id,)).fetchone()
        return max(int(row["revision"]) if row else 0, int(counter["last_revision"]) if counter else 0)

    @staticmethod
    def _note_flat_revision(conn: sqlite3.Connection, part_id: str, revision: int) -> None:
        conn.execute(
            "INSERT INTO part_flat_revisions(part_id, last_revision) VALUES (?,?) ON CONFLICT(part_id)"
            " DO UPDATE SET last_revision=MAX(last_revision, excluded.last_revision)",
            (part_id, int(revision)),
        )


def parse_recipe_id(recipe_id: str) -> tuple[str, int] | None:
    """``("SKU", version)`` of a recipe id, or None when malformed."""
    match = _RECIPE_ID_RE.match(recipe_id or "")
    if match is None:
        return None
    return match.group(1), int(match.group(2))
