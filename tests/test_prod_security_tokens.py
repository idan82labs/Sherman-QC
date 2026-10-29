"""C04 principals, device tokens and stream tokens."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from apps.production_line.security import stream_tokens
from apps.production_line.security.device_tokens import DeviceTokenError, DeviceTokenStore, hash_token
from apps.production_line.security.principals import CAPABILITIES, Principal, has_capability

SECRET = b"k" * 32


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 17, 3, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


# --------------------------------------------------------------------------- principals


@pytest.mark.parametrize(
    "role,cap,expected",
    [
        ("viewer", "line.read", True),
        ("viewer", "line.operate", False),
        ("viewer", "parts.write", False),
        ("operator", "line.operate", True),
        ("operator", "parts.write", True),
        ("operator", "parts.delete", False),
        ("operator", "parts.import", False),
        ("operator", "recipe.release", False),
        ("operator", "devices.manage", False),
        ("admin", "parts.delete", True),
        ("admin", "parts.import", True),
        ("admin", "recipe.release", True),
        ("admin", "devices.manage", True),
        ("admin", "metrics.read", True),
        ("admin", "no.such.capability", False),
    ],
)
def test_user_capabilities(role: str, cap: str, expected: bool) -> None:
    assert has_capability(Principal.for_user(1, "u", role), cap) is expected


def test_release_roles_widen_recipe_release_but_never_to_viewer() -> None:
    op = Principal.for_user(2, "op", "operator", release_roles=("admin", "operator"))
    assert has_capability(op, "recipe.release", release_roles=("admin", "operator"))
    assert not has_capability(op, "recipe.release")
    viewer = Principal.for_user(3, "v", "viewer")
    assert not has_capability(viewer, "recipe.release", release_roles=("viewer", "admin"))


def test_unknown_role_and_none_principal() -> None:
    with pytest.raises(ValueError):
        Principal.for_user(1, "x", "superuser")
    assert not has_capability(None, "line.read")
    forged = Principal(kind="user", user_id="1", username="x", role="root")
    assert not has_capability(forged, "line.read")


def test_device_principal_gets_only_line_read_for_its_station() -> None:
    dev = Principal.for_device("dt_0123456789abcdef", "kiosk", "cell-1")
    assert has_capability(dev, "line.read", "cell-1")
    assert not has_capability(dev, "line.read", "cell-2")
    assert not has_capability(dev, "line.read", None)
    for cap in CAPABILITIES:
        if cap != "line.read":
            assert not has_capability(dev, cap, "cell-1"), cap
    # Even a hand-built device principal with inflated scopes gets nothing more.
    inflated = Principal(
        kind="device", user_id="dt_x", username="k", role="admin", station_scope="cell-1", scopes=frozenset(CAPABILITIES)
    )
    assert not has_capability(inflated, "parts.delete", "cell-1")


# --------------------------------------------------------------------------- device tokens


def test_device_token_issue_verify_scope(tmp_path: Path) -> None:
    store = DeviceTokenStore(tmp_path / "db" / "security.sqlite3")
    token_id, token = store.issue("cell-1", "Press 1 kiosk", created_by="admin")
    assert token.startswith("sdt_") and token_id.startswith("dt_")
    p = store.verify(token)
    assert p is not None and p.kind == "device" and p.station_scope == "cell-1" and p.user_id == token_id
    assert has_capability(p, "line.read", "cell-1")
    assert not has_capability(p, "line.read", "cell-2")
    assert not has_capability(p, "line.operate", "cell-1")


def test_device_token_store_keeps_only_hash(tmp_path: Path) -> None:
    db = tmp_path / "security.sqlite3"
    store = DeviceTokenStore(db)
    _, token = store.issue("cell-1", "kiosk", created_by="admin")
    raw = db.read_bytes()
    for side in (db.with_name(db.name + "-wal"), db.with_name(db.name + "-journal")):
        if side.exists():
            raw += side.read_bytes()
    assert token.encode() not in raw
    with sqlite3.connect(db) as conn:
        stored = conn.execute("SELECT token_sha256 FROM device_tokens").fetchone()[0]
    assert stored == hash_token(token)
    listed = store.list()
    assert len(listed) == 1 and "token_sha256" not in listed[0] and token not in str(listed)
    assert (db.stat().st_mode & 0o077) == 0


def test_revoked_and_unknown_tokens_rejected(tmp_path: Path) -> None:
    store = DeviceTokenStore(tmp_path / "s.sqlite3")
    tid, token = store.issue("cell-1", "kiosk", created_by="admin")
    other_tid, other = store.issue("cell-2", "kiosk 2", created_by="admin")
    assert store.revoke(tid, by="admin") is True
    assert store.verify(token) is None
    assert store.revoke(tid, by="admin") is False
    assert store.revoke("dt_ffffffffffffffff", by="admin") is False
    assert store.revoke("'; DROP TABLE device_tokens; --", by="admin") is False
    assert store.verify(other) is not None and store.is_active(other_tid) and not store.is_active(tid)
    assert store.verify(token[:-1] + ("A" if token[-1] != "A" else "B")) is None
    for junk in ("", "sdt_", "sdt_short", "Bearer " + other, other + " ", "xyz_" + other[4:]):
        assert store.verify(junk) is None
    assert len(store.list(include_revoked=False)) == 1 and len(store.list()) == 2


def test_last_used_written_at_most_once_per_minute(tmp_path: Path) -> None:
    clock = FakeClock()
    store = DeviceTokenStore(tmp_path / "s.sqlite3", clock=clock)
    _, token = store.issue("cell-1", "kiosk", created_by="admin")
    assert store.list()[0]["last_used_at"] is None
    store.verify(token)
    first = store.list()[0]["last_used_at"]
    assert first is not None
    clock.now += timedelta(seconds=30)
    store.verify(token)
    assert store.list()[0]["last_used_at"] == first
    clock.now += timedelta(seconds=31)
    store.verify(token)
    assert store.list()[0]["last_used_at"] != first


@pytest.mark.parametrize("station,label", [("Cell_1", "k"), ("", "k"), ("cell-1", ""), ("cell-1", "x" * 65)])
def test_issue_validates_input(tmp_path: Path, station: str, label: str) -> None:
    store = DeviceTokenStore(tmp_path / "s.sqlite3")
    with pytest.raises(DeviceTokenError):
        store.issue(station, label, created_by="admin")


# --------------------------------------------------------------------------- stream tokens


def _viewer() -> Principal:
    return Principal.for_user(7, "viewer1", "viewer")


def test_stream_token_roundtrip() -> None:
    tok = stream_tokens.mint(_viewer(), "cell-1", SECRET, ttl_s=300, now=1000)
    claims = stream_tokens.verify(tok, SECRET, now=1100, station_id="cell-1")
    assert claims is not None
    assert (claims.kind, claims.subject, claims.role, claims.station_id) == ("user", "7", "viewer", "cell-1")
    assert claims.expires_at == 1300


def test_stream_token_rejected_after_ttl() -> None:
    tok = stream_tokens.mint(_viewer(), "cell-1", SECRET, ttl_s=60, now=1000)
    assert stream_tokens.verify(tok, SECRET, now=1059) is not None
    assert stream_tokens.verify(tok, SECRET, now=1060) is None
    assert stream_tokens.verify(tok, SECRET, now=5000) is None


def test_stream_token_rejected_when_tampered() -> None:
    tok = stream_tokens.mint(_viewer(), "cell-1", SECRET, now=1000)
    prefix, body, mac = tok.split(".")
    # Re-encode a body that claims another station; the MAC no longer matches.
    import base64
    import json

    claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    claims["station_id"] = "cell-9"
    forged_body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    assert stream_tokens.verify(f"{prefix}.{forged_body}.{mac}", SECRET, now=1001) is None
    flipped = mac[:-1] + ("A" if mac[-1] != "A" else "B")
    assert stream_tokens.verify(f"{prefix}.{body}.{flipped}", SECRET, now=1001) is None
    assert stream_tokens.verify(tok, b"z" * 32, now=1001) is None
    assert stream_tokens.verify("sst2." + body + "." + mac, SECRET, now=1001) is None
    assert stream_tokens.verify(tok + ".x", SECRET, now=1001) is None
    assert stream_tokens.verify("garbage", SECRET, now=1001) is None


def test_stream_token_station_binding_and_future_iat() -> None:
    tok = stream_tokens.mint(_viewer(), "cell-1", SECRET, now=1000)
    assert stream_tokens.verify(tok, SECRET, now=1001, station_id="cell-2") is None
    assert stream_tokens.verify(tok, SECRET, now=900) is None  # issued 100 s in the future


def test_stream_token_rejects_overlong_lifetime_even_if_signed() -> None:
    import base64
    import hashlib
    import hmac
    import json

    claims = {"v": 1, "kind": "user", "sub": "1", "role": "admin", "station_id": "cell-1", "iat": 1000, "exp": 99999, "nonce": "x"}
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    mac = base64.urlsafe_b64encode(hmac.new(SECRET, body.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    assert stream_tokens.verify(f"sst1.{body}.{mac}", SECRET, now=1001) is None


def test_stream_token_mint_policy() -> None:
    with pytest.raises(ValueError):
        stream_tokens.mint(_viewer(), "cell-1", SECRET, ttl_s=301)
    with pytest.raises(ValueError):
        stream_tokens.mint(_viewer(), "cell-1", SECRET, ttl_s=0)
    with pytest.raises(ValueError):
        stream_tokens.mint(_viewer(), "cell-1", b"short")
    dev = Principal.for_device("dt_0123456789abcdef", "kiosk", "cell-1")
    assert stream_tokens.verify(stream_tokens.mint(dev, "cell-1", SECRET, now=10), SECRET, now=11).kind == "device"
    with pytest.raises(PermissionError):
        stream_tokens.mint(dev, "cell-2", SECRET)
