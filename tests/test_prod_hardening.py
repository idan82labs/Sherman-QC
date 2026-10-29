"""C04 startup hardening and the admin bootstrap CLI.

``auth`` (backend/auth.py) is imported lazily inside the bootstrap tests only and
always against a users DB under tmp_path.
"""
from __future__ import annotations

import io
import os
import sqlite3
from pathlib import Path

import pytest

from apps.production_line.security.hardening import HARDENING_CODES, enforce, legacy_user_probe, startup_checks
from apps.production_line.settings import ensure_layout, load_settings

GOOD_SECRET = "s" * 48


@pytest.fixture()
def settings(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    s = load_settings({"SHERMAN_LINE_ENABLED": "1", "SHERMAN_LINE_DATA_ROOT": str(tmp_path / "root")}, repo_root=repo)
    ensure_layout(s.paths)
    return s


def _good_env(**extra: str) -> dict[str, str]:
    env = {"SECRET_KEY": GOOD_SECRET, "PRODUCTION": "1", "WEB_CONCURRENCY": "1"}
    env.update(extra)
    return env


def _no_default_admin(_u: str, _p: str) -> bool:
    return False


def _codes(failures) -> set[str]:
    return {f.code for f in failures}


def test_all_good_env_returns_empty(settings) -> None:
    assert startup_checks(settings, _good_env(), _no_default_admin) == []
    assert enforce(settings, _good_env(), _no_default_admin, stream=io.StringIO()) == []


@pytest.mark.parametrize(
    "secret",
    ["", "   ", "change-this-in-production", "change-this-to-a-secure-random-string", "short-but-random-1234"],
)
def test_secret_key_missing_or_default(settings, secret: str) -> None:
    env = _good_env(SECRET_KEY=secret)
    assert _codes(startup_checks(settings, env, _no_default_admin)) == {"SECRET_KEY_MISSING_OR_DEFAULT"}
    env.pop("SECRET_KEY")
    assert _codes(startup_checks(settings, env, _no_default_admin)) == {"SECRET_KEY_MISSING_OR_DEFAULT"}


def test_production_flag_missing(settings) -> None:
    env = _good_env()
    env.pop("PRODUCTION")
    assert _codes(startup_checks(settings, env, _no_default_admin)) == {"PRODUCTION_FLAG_MISSING"}
    assert _codes(startup_checks(settings, _good_env(PRODUCTION="0"), _no_default_admin)) == {"PRODUCTION_FLAG_MISSING"}


def test_default_admin_probe(settings) -> None:
    calls = []

    def probe(u: str, p: str) -> bool:
        calls.append((u, p))
        return True

    assert _codes(startup_checks(settings, _good_env(), probe)) == {"DEFAULT_ADMIN_PASSWORD_ACTIVE"}
    assert calls == [("admin", "admin123")]
    assert startup_checks(settings, _good_env(SHERMAN_ALLOW_DEFAULT_ADMIN="1"), probe) == []


def test_probe_missing_or_raising_fails_closed(settings) -> None:
    def boom(_u: str, _p: str) -> bool:
        raise sqlite3.OperationalError("locked")

    assert _codes(startup_checks(settings, _good_env(), boom)) == {"USER_PROBE_FAILED"}
    assert _codes(startup_checks(settings, _good_env(), None)) == {"USER_PROBE_FAILED"}


@pytest.mark.parametrize("name,value", [("WEB_CONCURRENCY", "2"), ("UVICORN_WORKERS", "4"), ("WEB_CONCURRENCY", "many")])
def test_multiple_workers(settings, name: str, value: str) -> None:
    assert _codes(startup_checks(settings, _good_env(**{name: value}), _no_default_admin)) == {"MULTIPLE_WORKERS"}


def test_self_register_is_warning_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    s = load_settings(
        {"SHERMAN_LINE_ENABLED": "1", "SHERMAN_LINE_DATA_ROOT": str(tmp_path / "r"), "SHERMAN_ALLOW_SELF_REGISTER": "1"},
        repo_root=repo,
    )
    failures = startup_checks(s, _good_env(), _no_default_admin)
    assert [(f.code, f.severity) for f in failures] == [("SELF_REGISTER_ENABLED", "warning")]
    out = io.StringIO()
    assert enforce(s, _good_env(), _no_default_admin, stream=out)  # does not exit
    assert "SELF_REGISTER_ENABLED" in out.getvalue()


def test_data_root_invalid(settings, tmp_path: Path) -> None:
    assert _codes(startup_checks(None, _good_env(), _no_default_admin)) == {"DATA_ROOT_INVALID"}
    os.chmod(settings.paths.data_root, 0o777)
    try:
        assert _codes(startup_checks(settings, _good_env(), _no_default_admin)) == {"DATA_ROOT_INVALID"}
    finally:
        os.chmod(settings.paths.data_root, 0o700)


def test_enforce_exits_2_and_lists_every_failure(settings) -> None:
    env = {"WEB_CONCURRENCY": "2"}
    out = io.StringIO()
    with pytest.raises(SystemExit) as exc:
        enforce(settings, env, lambda u, p: True, stream=out)
    assert exc.value.code == 2
    text = out.getvalue()
    for code in ("SECRET_KEY_MISSING_OR_DEFAULT", "PRODUCTION_FLAG_MISSING", "DEFAULT_ADMIN_PASSWORD_ACTIVE", "MULTIPLE_WORKERS"):
        assert code in text
    assert {f.code for f in startup_checks(settings, env, lambda u, p: True)} <= HARDENING_CODES


# --------------------------------------------------------------------------- bootstrap + probe


@pytest.fixture()
def clean_auth_env(monkeypatch):
    monkeypatch.delenv("PRODUCTION", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)


def _run_bootstrap(args: list[str], password: str):
    from apps.production_line.security import bootstrap_admin

    out = io.StringIO()
    code = bootstrap_admin.main(args, stdin=io.StringIO(password + "\n"), stdout=out)
    return code, out.getvalue()


def test_legacy_probe_missing_db_counts_as_default_admin(tmp_path: Path, clean_auth_env) -> None:
    assert legacy_user_probe(tmp_path / "absent.db")("admin", "admin123") is True
    assert not (tmp_path / "absent.db").exists()


def test_bootstrap_creates_admin_and_neutralises_seed(tmp_path: Path, clean_auth_env, capsys) -> None:
    db = tmp_path / "app" / "data" / "users.db"
    password = "correct horse battery staple"
    code, out = _run_bootstrap(["--users-db", str(db), "--username", "alice", "--password-stdin"], password)
    assert code == 0, capsys.readouterr().err
    assert password not in out and password not in capsys.readouterr().err
    assert '"status": "created"' in out and '"deactivated_default_admin": true' in out
    probe = legacy_user_probe(db)
    assert probe("admin", "admin123") is False  # the seed is deactivated
    assert probe("alice", password) is True
    assert probe("alice", "wrong-password-123") is False
    # Hardening now passes the default-admin check against this DB.
    assert "DEFAULT_ADMIN_PASSWORD_ACTIVE" not in {f.code for f in startup_checks(None, _good_env(), probe)}


def test_bootstrap_existing_user_needs_rotate(tmp_path: Path, clean_auth_env) -> None:
    db = tmp_path / "users.db"
    first = "first password long enough"
    assert _run_bootstrap(["--users-db", str(db), "--username", "bob", "--password-stdin"], first)[0] == 0
    second = "second password long enough"
    assert _run_bootstrap(["--users-db", str(db), "--username", "bob", "--password-stdin"], second)[0] == 3
    assert legacy_user_probe(db)("bob", first) is True
    code, out = _run_bootstrap(["--users-db", str(db), "--username", "bob", "--password-stdin", "--rotate"], second)
    assert code == 0 and '"status": "rotated"' in out
    assert legacy_user_probe(db)("bob", second) is True and legacy_user_probe(db)("bob", first) is False


def test_bootstrap_rotates_seeded_admin_without_flag(tmp_path: Path, clean_auth_env) -> None:
    db = tmp_path / "users.db"
    pw = "a much better admin password"
    code, out = _run_bootstrap(["--users-db", str(db), "--username", "admin", "--password-stdin"], pw)
    assert code == 0 and '"status": "rotated"' in out
    probe = legacy_user_probe(db)
    assert probe("admin", "admin123") is False and probe("admin", pw) is True


def test_bootstrap_refusals(tmp_path: Path, clean_auth_env) -> None:
    from apps.production_line.security import bootstrap_admin

    fake_repo = tmp_path / "repo"
    (fake_repo / "data").mkdir(parents=True)
    with pytest.raises(bootstrap_admin.BootstrapError) as exc:
        bootstrap_admin.bootstrap_admin(fake_repo / "data" / "users.db", "alice", "x" * 20, repo_root=fake_repo)
    assert exc.value.exit_code == 2
    assert not (fake_repo / "data" / "users.db").exists()
    # The CLI defaults to the real source root; a path there is refused before any open.
    real_repo_db = Path(bootstrap_admin.__file__).resolve().parents[3] / "never-created-users.db"
    assert _run_bootstrap(["--users-db", str(real_repo_db), "--username", "alice", "--password-stdin"], "x" * 20)[0] == 2
    db = tmp_path / "users.db"
    assert _run_bootstrap(["--users-db", str(db), "--username", "alice", "--password-stdin"], "short")[0] == 2
    assert _run_bootstrap(["--users-db", str(db), "--username", "alice", "--password-stdin"], "admin123")[0] == 2
    assert _run_bootstrap(["--users-db", "rel/users.db", "--username", "alice", "--password-stdin"], "y" * 20)[0] == 2
    assert not db.exists()


def test_legacy_user_token_verifier_requires_active_user(tmp_path: Path, clean_auth_env, monkeypatch) -> None:
    import auth  # type: ignore  # lazy; pointed at a tmp users DB below, never data/users.db

    from apps.production_line.security.dependencies import LegacyUserTokenVerifier

    manager = auth.SQLiteUserManager(db_path=tmp_path / "users.db")
    monkeypatch.setattr(auth, "_user_manager", manager)
    user = manager.create_user("olga", "olga@x.local", "operator password 1", role="operator")
    token = auth.JWTManager.create_token(user)
    verifier = LegacyUserTokenVerifier()
    p = verifier.verify(token)
    assert p is not None and p.kind == "user" and p.role == "operator" and p.username == "olga"
    assert p.user_id == str(user.id)
    # The role comes from the user row, not the token: a demotion applies immediately.
    manager.update_user(user.id, role="viewer")
    assert verifier.verify(token).role == "viewer"
    manager.update_user(user.id, is_active=False)
    assert verifier.verify(token) is None
    manager.update_user(user.id, is_active=True)
    head, body, sig = token.split(".")
    assert verifier.verify(f"{head}.{body}.{sig[:-2]}xx") is None
    assert verifier.verify("") is None and verifier.verify("not-a-jwt") is None


def test_legacy_probe_read_only_with_reserved_characters(tmp_path: Path, clean_auth_env) -> None:
    d = tmp_path / "Shop Data #1"
    d.mkdir()
    db = d / "users.db"
    # A missing DB counts as the default admin and is never created, even with '#'.
    assert legacy_user_probe(db)("admin", "admin123") is True
    assert list(d.iterdir()) == [] and sorted(p.name for p in tmp_path.iterdir()) == [d.name]
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE users (username TEXT, password_hash TEXT, is_active INTEGER)")
    conn.commit()
    conn.close()
    before = (db.stat().st_size, db.stat().st_mtime_ns)
    assert legacy_user_probe(db)("admin", "admin123") is False  # no such user
    assert (db.stat().st_size, db.stat().st_mtime_ns) == before
    assert sorted(p.name for p in d.iterdir()) == ["users.db"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [d.name]


def test_data_root_check_guards_the_dev_repo_of_an_app_copy(tmp_path: Path) -> None:
    import dataclasses
    import json as _json

    from apps.production_line.settings import APP_MANIFEST_NAME, APP_MANIFEST_SCHEMA, LinePaths

    dev = tmp_path / "dev-repo"
    (dev / "output").mkdir(parents=True)
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / APP_MANIFEST_NAME).write_text(
        _json.dumps({"schema": APP_MANIFEST_SCHEMA, "source_repo_root": str(dev)})
    )
    s = load_settings({"SHERMAN_LINE_ENABLED": "1", "SHERMAN_LINE_DATA_ROOT": str(tmp_path / "root")}, repo_root=copy)
    ensure_layout(s.paths)
    assert startup_checks(s, _good_env(), _no_default_admin) == []
    for bad in (dev / "tmp" / "root", dev / "output" / "root"):
        moved = dataclasses.replace(s, paths=LinePaths.for_root(bad))
        assert "DATA_ROOT_INVALID" in _codes(startup_checks(moved, _good_env(), _no_default_admin)), bad
