"""Create or rotate the first admin account of a deployed app copy.

Usage::

    printf '%s\\n' "$PASSWORD" | python -m apps.production_line.security.bootstrap_admin \\
        --users-db /srv/sherman/app/data/users.db --username alice --password-stdin [--rotate]

- Uses ``backend/auth.py`` ``SQLiteUserManager(db_path=PATH)``, imported lazily.
- Opening a new users DB seeds ``admin``/``admin123`` (``backend/auth.py:240-256``).
  Bootstrap therefore always neutralises that seed: it rotates it when the requested
  username is ``admin``, and otherwise deactivates the seeded admin while it still
  has the default password.
- An existing user is changed only with ``--rotate`` (the password is reset, the
  role set to admin and the account re-activated).
- Refuses a ``PATH`` inside the *dev repo* unless ``--allow-repo-path``: the dev
  repo's ``data/users.db`` is research custody, and E2E runs use an app copy.  When
  this module runs from an app copy (its root has ``bundle_manifest.json``), the dev
  repo is the ``source_repo_root`` that manifest records, so the copy's own
  ``<copy>/data/users.db`` (where the copy's legacy ``auth.DATABASE_PATH`` lives) is
  accepted.  A copy whose manifest does not name its source repo is refused.
- Never echoes the password.  Exit codes: 0 done, 2 invalid input, 3 user exists
  without ``--rotate``, 4 auth backend failure.

The output is one JSON line ``{"status", "username", "users_db",
"deactivated_default_admin"}``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, TextIO

__all__ = ["MIN_PASSWORD_LEN", "BootstrapError", "bootstrap_admin", "dev_repo_root", "main"]

MIN_PASSWORD_LEN = 12
_DEFAULT_PASSWORD = "admin123"


class BootstrapError(Exception):
    def __init__(self, exit_code: int, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _repo_root() -> Path:
    """The source root this module runs from (a dev checkout or an app copy)."""
    return Path(__file__).resolve().parents[3]


def dev_repo_root(running_root: Optional[Path] = None) -> Path:
    """The dev repo whose ``data/`` is custody: the recorded source of an app copy,
    or the running root itself when it is a dev checkout.  Raises
    :class:`BootstrapError` (exit 2) when an app copy does not record its source."""
    from apps.production_line.settings import SettingsError, source_repo_root_of

    root = Path(os.path.realpath(running_root or _repo_root()))
    try:
        src = source_repo_root_of(root)
    except SettingsError as exc:
        raise BootstrapError(2, f"cannot locate the dev repo: {exc.message}") from None
    return root if src is None else src


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_auth():  # pragma: no cover - trivial
    backend = str(_repo_root() / "backend")
    if backend not in sys.path:
        sys.path.append(backend)
    import auth  # type: ignore

    return auth


def bootstrap_admin(
    users_db: Path,
    username: str,
    password: str,
    *,
    rotate: bool = False,
    allow_repo_path: bool = False,
    email: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> dict[str, object]:
    """Create or rotate ``username`` as an active admin.  Raises :class:`BootstrapError`.

    ``repo_root`` is the running source root (default: the root this module is
    loaded from); see :func:`dev_repo_root` for which paths are refused.
    """
    users_db = Path(users_db)
    if not users_db.is_absolute():
        raise BootstrapError(2, "--users-db must be an absolute path")
    real_db = Path(os.path.realpath(users_db))
    repo = dev_repo_root(repo_root)
    if _inside(real_db, repo) and not allow_repo_path:
        raise BootstrapError(2, f"refusing a users DB inside the dev repo ({repo}); pass --allow-repo-path")
    username = (username or "").strip()
    if not username or len(username) > 64 or any(c.isspace() for c in username):
        raise BootstrapError(2, "username must be 1..64 non-space characters")
    if len(password) < MIN_PASSWORD_LEN:
        raise BootstrapError(2, f"password must be at least {MIN_PASSWORD_LEN} characters")
    if password == _DEFAULT_PASSWORD or password.lower() == username.lower():
        raise BootstrapError(2, "password is too weak")

    try:
        auth = _load_auth()
        manager = auth.SQLiteUserManager(db_path=real_db)
    except Exception as exc:
        raise BootstrapError(4, f"could not open the users DB ({type(exc).__name__})") from None

    hasher = auth.PasswordHasher
    deactivated_default = False
    existing = manager.get_user_by_username(username)
    if existing is not None:
        seeded_default = username == "admin" and hasher.verify_password(_DEFAULT_PASSWORD, existing.password_hash)
        if not rotate and not seeded_default:
            raise BootstrapError(3, f"user {username!r} exists; pass --rotate to reset it")
        if not manager.update_password(existing.id, password):
            raise BootstrapError(4, "password update failed")
        manager.update_user(existing.id, role="admin", is_active=True)
        status = "rotated"
    else:
        created = manager.create_user(
            username=username,
            email=email or f"{username}@sherman-qc.local",
            password=password,
            role="admin",
        )
        if created is None:
            raise BootstrapError(4, "user creation failed (email may already exist)")
        status = "created"

    if username != "admin":
        seed = manager.get_user_by_username("admin")
        if seed is not None and seed.is_active and hasher.verify_password(_DEFAULT_PASSWORD, seed.password_hash):
            manager.update_user(seed.id, is_active=False)
            deactivated_default = True

    return {
        "status": status,
        "username": username,
        "users_db": str(real_db),
        "deactivated_default_admin": deactivated_default,
    }


def _read_password(stream: TextIO) -> str:
    line = stream.readline()
    return line[:-1] if line.endswith("\n") else line


def main(argv: Optional[Sequence[str]] = None, *, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    parser = argparse.ArgumentParser(prog="python -m apps.production_line.security.bootstrap_admin")
    parser.add_argument("--users-db", required=True, type=Path)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password-stdin", action="store_true", required=True, help="read the password from stdin")
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--allow-repo-path", action="store_true")
    parser.add_argument("--email")
    args = parser.parse_args(argv)
    password = _read_password(stdin).rstrip("\r")
    try:
        result = bootstrap_admin(
            args.users_db,
            args.username,
            password,
            rotate=args.rotate,
            allow_repo_path=args.allow_repo_path,
            email=args.email,
        )
    except BootstrapError as exc:
        print(f"bootstrap_admin: {exc}", file=sys.stderr)
        return exc.exit_code
    print(json.dumps(result, sort_keys=True), file=stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
