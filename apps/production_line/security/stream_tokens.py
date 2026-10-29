"""Short-lived, station-bound tokens for the SSE event stream.

``EventSource`` cannot send an ``Authorization`` header, so an authenticated client
first calls ``POST /api/line/stream-token`` and then opens
``/api/line/events/stream?station_id=..&token=..``.  The token is::

    "sst1." + base64url(canonical JSON claims) + "." + base64url(HMAC-SHA256(secret, claims_b64))

Claims: ``{v, kind, sub, role, station_id, iat, exp, nonce}``.  The TTL is at most
300 s.  Verification is constant-time on the MAC, rejects expired, future-dated,
tampered and wrong-station tokens, and returns :class:`StreamClaims`.  Callers that
need instant revocation of a device should re-check ``sub`` against the device store.
The secret comes from ``settings.get_secret(paths, "stream_token")``.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Optional

from apps.production_line.security.principals import Principal, has_capability

__all__ = ["MAX_TTL_S", "PREFIX", "StreamClaims", "mint", "verify"]

MAX_TTL_S = 300
PREFIX = "sst1"
CLOCK_SKEW_S = 30
_MIN_SECRET_LEN = 16


@dataclass(frozen=True)
class StreamClaims:
    kind: str
    subject: str
    role: str
    station_id: str
    issued_at: int
    expires_at: int


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _mac(secret: bytes, body_b64: str) -> str:
    return _b64e(hmac.new(secret, body_b64.encode("ascii"), hashlib.sha256).digest())


def _check_secret(secret: bytes) -> None:
    if not isinstance(secret, (bytes, bytearray)) or len(secret) < _MIN_SECRET_LEN:
        raise ValueError("stream token secret must be at least 16 bytes")


def mint(
    principal: Principal,
    station_id: str,
    secret: bytes,
    ttl_s: int = MAX_TTL_S,
    *,
    now: Optional[float] = None,
) -> str:
    """Mint a stream token for ``station_id``.

    Raises ``PermissionError`` when ``principal`` may not read that station and
    ``ValueError`` for a TTL outside 1..300 s or a weak secret.
    """
    _check_secret(secret)
    if not isinstance(ttl_s, int) or not 1 <= ttl_s <= MAX_TTL_S:
        raise ValueError(f"ttl_s must be an integer in 1..{MAX_TTL_S}")
    if not has_capability(principal, "line.read", station_id):
        raise PermissionError("principal may not read this station")
    issued = int(now if now is not None else time.time())
    claims = {
        "v": 1,
        "kind": principal.kind,
        "sub": principal.user_id,
        "role": principal.role,
        "station_id": station_id,
        "iat": issued,
        "exp": issued + ttl_s,
        "nonce": secrets.token_hex(8),
    }
    body = _b64e(json.dumps(claims, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return f"{PREFIX}.{body}.{_mac(bytes(secret), body)}"


def verify(
    token: str,
    secret: bytes,
    now: Optional[float] = None,
    *,
    station_id: Optional[str] = None,
) -> Optional[StreamClaims]:
    """The claims of a valid token, else None.  When ``station_id`` is given the
    token must be bound to that station."""
    try:
        _check_secret(secret)
    except ValueError:
        return None
    if not isinstance(token, str) or len(token) > 2048:
        return None
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != PREFIX:
        return None
    _, body, mac = parts
    try:
        expected = _mac(bytes(secret), body)
    except (UnicodeEncodeError, ValueError):
        return None
    if not hmac.compare_digest(expected.encode("ascii"), mac.encode("ascii", "replace")):
        return None
    try:
        claims = json.loads(_b64d(body))
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return None
    if not isinstance(claims, dict) or claims.get("v") != 1:
        return None
    try:
        iat = int(claims["iat"])
        exp = int(claims["exp"])
        kind = str(claims["kind"])
        sub = str(claims["sub"])
        role = str(claims["role"])
        sid = str(claims["station_id"])
    except (KeyError, TypeError, ValueError):
        return None
    current = float(now if now is not None else time.time())
    if exp - iat > MAX_TTL_S or exp <= iat:
        return None
    if current >= exp or iat > current + CLOCK_SKEW_S:
        return None
    if kind not in ("user", "device"):
        return None
    if station_id is not None and sid != station_id:
        return None
    return StreamClaims(kind=kind, subject=sub, role=role, station_id=sid, issued_at=iat, expires_at=exp)
