"""Startup hardening: refuse to serve the production line with insecure defaults.

:func:`startup_checks` returns every problem at once as :class:`HardeningFailure`
records (an empty list means the host is acceptable); :func:`enforce` prints them
and raises ``SystemExit(2)`` when any has severity ``error``.  C11's server calls
``enforce`` when ``SHERMAN_PRODUCTION_HARDENING=1``.

Codes:

- ``SECRET_KEY_MISSING_OR_DEFAULT`` – ``SECRET_KEY`` empty, a known default, or
  shorter than 32 characters (``backend/auth.py:23-30`` otherwise signs JWTs with a
  per-process random key or a public default).
- ``PRODUCTION_FLAG_MISSING`` – ``PRODUCTION`` is not truthy, so ``auth.py`` does not
  enforce its own secret check.
- ``DEFAULT_ADMIN_PASSWORD_ACTIVE`` – ``user_probe("admin", "admin123")`` is true
  (the seed at ``backend/auth.py:246-256``), unless ``SHERMAN_ALLOW_DEFAULT_ADMIN=1``.
- ``USER_PROBE_FAILED`` – the probe raised; fail closed.
- ``SELF_REGISTER_ENABLED`` – warning only.
- ``MULTIPLE_WORKERS`` – ``WEB_CONCURRENCY`` or ``UVICORN_WORKERS`` > 1 (the line
  keeps SQLite leases and SSE fan-out in one process).
- ``DATA_ROOT_INVALID`` – no valid settings, or the data root is inside the repo,
  not a directory, or writable by group/other.
"""
from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Optional, Sequence, TextIO

from apps.production_line.settings import LineSettings, forbidden_location_reason

__all__ = [
    "HARDENING_CODES",
    "DEFAULT_SECRET_VALUES",
    "MIN_SECRET_KEY_LEN",
    "HardeningFailure",
    "startup_checks",
    "enforce",
    "legacy_user_probe",
]

HARDENING_CODES: frozenset[str] = frozenset(
    {
        "SECRET_KEY_MISSING_OR_DEFAULT",
        "PRODUCTION_FLAG_MISSING",
        "DEFAULT_ADMIN_PASSWORD_ACTIVE",
        "USER_PROBE_FAILED",
        "SELF_REGISTER_ENABLED",
        "MULTIPLE_WORKERS",
        "DATA_ROOT_INVALID",
    }
)

DEFAULT_SECRET_VALUES: frozenset[str] = frozenset(
    {
        "change-this-to-a-secure-random-string",
        "change-this-in-production",
        "changeme",
        "change-me",
        "secret",
        "secret-key",
        "dev",
        "development",
        "test",
    }
)
MIN_SECRET_KEY_LEN = 32
_TRUE = frozenset({"1", "true", "yes", "on"})

UserProbe = Callable[[str, str], bool]


@dataclass(frozen=True)
class HardeningFailure:
    code: str
    message: str
    severity: Literal["error", "warning"] = "error"


def _truthy(env: Mapping[str, str], name: str) -> bool:
    return str(env.get(name, "") or "").strip().lower() in _TRUE


def _worker_count(env: Mapping[str, str], name: str) -> Optional[int]:
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return -1


def startup_checks(
    settings: Optional[LineSettings],
    env: Optional[Mapping[str, str]] = None,
    user_probe: Optional[UserProbe] = None,
) -> list[HardeningFailure]:
    """Every hardening failure for this host (empty list = acceptable)."""
    env = os.environ if env is None else env
    out: list[HardeningFailure] = []

    secret = str(env.get("SECRET_KEY", "") or "")
    if not secret.strip() or secret.strip().lower() in DEFAULT_SECRET_VALUES or len(secret) < MIN_SECRET_KEY_LEN:
        out.append(
            HardeningFailure(
                "SECRET_KEY_MISSING_OR_DEFAULT",
                f"SECRET_KEY must be set to a non-default value of at least {MIN_SECRET_KEY_LEN} characters",
            )
        )

    if not _truthy(env, "PRODUCTION"):
        out.append(HardeningFailure("PRODUCTION_FLAG_MISSING", "PRODUCTION=1 must be set for the production line"))

    if not _truthy(env, "SHERMAN_ALLOW_DEFAULT_ADMIN"):
        if user_probe is None:
            out.append(HardeningFailure("USER_PROBE_FAILED", "no user probe was supplied; cannot rule out admin/admin123"))
        else:
            try:
                if bool(user_probe("admin", "admin123")):
                    out.append(
                        HardeningFailure(
                            "DEFAULT_ADMIN_PASSWORD_ACTIVE",
                            "the default admin/admin123 account is active; rotate it with "
                            "python -m apps.production_line.security.bootstrap_admin --rotate",
                        )
                    )
            except Exception as exc:
                out.append(HardeningFailure("USER_PROBE_FAILED", f"user probe raised {type(exc).__name__}"))

    if settings is not None and settings.allow_self_register:
        out.append(
            HardeningFailure(
                "SELF_REGISTER_ENABLED",
                "SHERMAN_ALLOW_SELF_REGISTER=1: anyone can create an active operator account",
                "warning",
            )
        )

    for name in ("WEB_CONCURRENCY", "UVICORN_WORKERS"):
        n = _worker_count(env, name)
        if n is not None and n != 1:
            out.append(HardeningFailure("MULTIPLE_WORKERS", f"{name}={env.get(name)!s}: the line API runs exactly 1 worker"))

    out.extend(_data_root_checks(settings))
    return out


def _data_root_checks(settings: Optional[LineSettings]) -> list[HardeningFailure]:
    if settings is None:
        return [HardeningFailure("DATA_ROOT_INVALID", "production line settings are missing or invalid")]
    root = settings.paths.data_root
    reason = forbidden_location_reason(root, settings.repo_root)
    if reason:
        return [HardeningFailure("DATA_ROOT_INVALID", f"data root {reason}")]
    if root.exists() or root.is_symlink():
        if root.is_symlink() or not root.is_dir():
            return [HardeningFailure("DATA_ROOT_INVALID", "data root is not a real directory")]
        mode = root.stat().st_mode
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            return [HardeningFailure("DATA_ROOT_INVALID", "data root is writable by group or others")]
    return []


def enforce(
    settings: Optional[LineSettings],
    env: Optional[Mapping[str, str]] = None,
    user_probe: Optional[UserProbe] = None,
    *,
    stream: TextIO = sys.stderr,
) -> list[HardeningFailure]:
    """Run :func:`startup_checks`; print all findings; exit 2 on any error.

    Returns the (warning-only) findings when startup may proceed.
    """
    failures = startup_checks(settings, env, user_probe)
    for f in failures:
        print(f"[hardening] {f.severity.upper()} {f.code}: {f.message}", file=stream)
    errors: Sequence[HardeningFailure] = [f for f in failures if f.severity == "error"]
    if errors:
        print(
            f"[hardening] refusing to start: {len(errors)} error(s): " + ", ".join(sorted({f.code for f in errors})),
            file=stream,
        )
        raise SystemExit(2)
    return failures


def legacy_user_probe(users_db: Optional[Path] = None) -> UserProbe:
    """A probe that checks credentials against the legacy users DB *without* side
    effects (no last_login update, no default-admin seeding of a missing DB)."""

    def probe(username: str, password: str) -> bool:
        import sqlite3

        from apps.production_line.health import sqlite_uri

        import auth  # type: ignore  # lazy: backend/ on sys.path

        db = Path(users_db) if users_db is not None else Path(auth.DATABASE_PATH)
        if not db.exists():
            # A missing DB is seeded with admin/admin123 on first open
            # (backend/auth.py:240-244), so it counts as the default admin being active.
            return True
        conn = sqlite3.connect(sqlite_uri(db, "ro"), uri=True, timeout=10)
        try:
            row = conn.execute(
                "SELECT password_hash, is_active FROM users WHERE username = ?", (username,)
            ).fetchone()
        finally:
            conn.close()
        if row is None or not bool(row[1]):
            return False
        return bool(auth.PasswordHasher.verify_password(password, row[0]))

    return probe
