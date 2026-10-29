"""Revocable station device tokens for kiosk screens (``db/security.sqlite3``).

A token is ``"sdt_" + secrets.token_urlsafe(32)`` and is shown exactly once, by
:meth:`DeviceTokenStore.issue`.  Only its SHA-256 is stored.  A verified token yields
a device :class:`~apps.production_line.security.principals.Principal` scoped to one
station with ``line.read`` only.  ``last_used_at`` is written at most once per
minute per token so that a kiosk polling every 2 s does not hammer the database.
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

from apps.production_line.security.principals import Principal

__all__ = ["DEVICE_TOKEN_PREFIX", "DeviceTokenError", "DeviceTokenStore", "hash_token"]

DEVICE_TOKEN_PREFIX = "sdt_"
_TOKEN_RE = re.compile(r"^sdt_[A-Za-z0-9_-]{32,128}$")
_STATION_RE = re.compile(r"^[a-z0-9-]{1,32}$")
_TOKEN_ID_RE = re.compile(r"^dt_[0-9a-f]{16}$")
LAST_USED_MIN_INTERVAL_S = 60.0
MAX_LABEL_LEN = 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS device_tokens (
    token_id      TEXT PRIMARY KEY,
    token_sha256  TEXT NOT NULL UNIQUE,
    station_id    TEXT NOT NULL,
    label         TEXT NOT NULL,
    created_by    TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    revoked_at    TEXT,
    revoked_by    TEXT,
    last_used_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_device_tokens_station ON device_tokens(station_id);
"""


class DeviceTokenError(ValueError):
    """Invalid issue/revoke request (bad station id or label)."""


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_iso(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


class DeviceTokenStore:
    """SQLite-backed device token registry.  Safe for use from several threads."""

    def __init__(self, db_path: Path, *, clock: Callable[[], datetime] = _utcnow) -> None:
        self.db_path = Path(db_path)
        self._clock = clock
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.db_path.exists():
            fd = os.open(self.db_path, os.O_WRONLY | os.O_CREAT, 0o600)
            os.close(fd)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout = 10000")
            yield conn
        finally:
            conn.close()

    # -- issue / revoke ------------------------------------------------------

    def issue(self, station_id: str, label: str, created_by: str) -> tuple[str, str]:
        """Create a token for ``station_id``.  Returns ``(token_id, token)``; the
        plaintext token is never stored or retrievable again."""
        if not isinstance(station_id, str) or not _STATION_RE.fullmatch(station_id):
            raise DeviceTokenError("station_id must match [a-z0-9-]{1,32}")
        label = (label or "").strip()
        if not label or len(label) > MAX_LABEL_LEN:
            raise DeviceTokenError(f"label must be 1..{MAX_LABEL_LEN} characters")
        if not created_by:
            raise DeviceTokenError("created_by is required")
        token_id = "dt_" + secrets.token_hex(8)
        token = DEVICE_TOKEN_PREFIX + secrets.token_urlsafe(32)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO device_tokens (token_id, token_sha256, station_id, label, created_by, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (token_id, hash_token(token), station_id, label, str(created_by), _iso(self._clock())),
            )
        return token_id, token

    def revoke(self, token_id: str, by: str) -> bool:
        """Revoke ``token_id``.  Returns False when it is unknown or already revoked."""
        if not isinstance(token_id, str) or not _TOKEN_ID_RE.fullmatch(token_id):
            return False
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE device_tokens SET revoked_at = ?, revoked_by = ? WHERE token_id = ? AND revoked_at IS NULL",
                (_iso(self._clock()), str(by), token_id),
            )
            return cur.rowcount == 1

    # -- verify --------------------------------------------------------------

    def verify(self, token: str) -> Optional[Principal]:
        """The device principal for ``token``, or None when unknown, malformed or revoked."""
        if not isinstance(token, str) or not _TOKEN_RE.fullmatch(token):
            return None
        digest = hash_token(token)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT token_id, token_sha256, station_id, label, revoked_at, last_used_at"
                " FROM device_tokens WHERE token_sha256 = ?",
                (digest,),
            ).fetchone()
            if row is None or row["revoked_at"] is not None:
                return None
            if not secrets.compare_digest(row["token_sha256"], digest):  # defence in depth
                return None
            now = self._clock()
            last = row["last_used_at"]
            if last is None or (now - _parse_iso(last)).total_seconds() >= LAST_USED_MIN_INTERVAL_S:
                conn.execute(
                    "UPDATE device_tokens SET last_used_at = ? WHERE token_id = ? AND revoked_at IS NULL",
                    (_iso(now), row["token_id"]),
                )
            return Principal.for_device(row["token_id"], row["label"], row["station_id"])

    def is_active(self, token_id: str) -> bool:
        """True when ``token_id`` exists and is not revoked (used to re-check stream tokens)."""
        with self._connect() as conn:
            row = conn.execute("SELECT revoked_at FROM device_tokens WHERE token_id = ?", (token_id,)).fetchone()
        return row is not None and row["revoked_at"] is None

    def list(self, *, include_revoked: bool = True) -> list[dict[str, Optional[str]]]:
        """Token metadata, newest first.  Never includes the token or its hash."""
        sql = (
            "SELECT token_id, station_id, label, created_by, created_at, revoked_at, revoked_by, last_used_at"
            " FROM device_tokens"
        )
        if not include_revoked:
            sql += " WHERE revoked_at IS NULL"
        sql += " ORDER BY created_at DESC, token_id"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql).fetchall()]
