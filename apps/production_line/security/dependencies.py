"""FastAPI dependencies for the production line routers (C05, C07).

- :func:`get_line_settings` loads :class:`~apps.production_line.settings.LineSettings`
  once (``lru_cache``) and returns ``None`` when the line is not configured; every
  line route then answers ``503 {"detail": "production line not configured"}``
  through :func:`require_line_settings`.  Override it with
  ``app.dependency_overrides[get_line_settings] = lambda: settings`` in tests.
- :func:`get_principal` authenticates a ``Bearer`` token: ``sdt_…`` tokens go to the
  device store, anything else to the user verifier (the legacy JWT).  No token or an
  invalid one is a 401 with ``WWW-Authenticate: Bearer``.
- :func:`require_capability` returns a dependency that yields the principal or
  raises 403.  ``station_param`` names a path or query parameter that carries the
  station id, which device tokens must match.

The legacy user lookup is imported lazily, inside :class:`LegacyUserTokenVerifier`,
so importing this module never loads ``auth`` or the legacy server.  It imports the
``auth`` module by the same top-level name the legacy server uses
(``backend/_server_legacy_impl.py:34``), which matters because ``auth.SECRET_KEY`` is
random per process when ``SECRET_KEY`` is not configured.
"""
from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Protocol, runtime_checkable

from fastapi import Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from apps.production_line.security.device_tokens import DEVICE_TOKEN_PREFIX, DeviceTokenStore
from apps.production_line.security.principals import Principal, has_capability
from apps.production_line.settings import LineSettings, SettingsError, load_settings

__all__ = [
    "NOT_CONFIGURED_DETAIL",
    "UserTokenVerifier",
    "LegacyUserTokenVerifier",
    "get_line_settings",
    "line_settings_error",
    "reset_line_settings_cache",
    "require_line_settings",
    "get_device_token_store",
    "get_user_token_verifier",
    "set_user_token_verifier",
    "parse_bearer",
    "authenticate_token",
    "get_principal",
    "require_capability",
]

logger = logging.getLogger(__name__)

NOT_CONFIGURED_DETAIL = "production line not configured"


@runtime_checkable
class UserTokenVerifier(Protocol):
    """Maps a user bearer token to an active user's Principal, or None."""

    def verify(self, token: str) -> Optional[Principal]: ...


class LegacyUserTokenVerifier:
    """The same lookup as legacy ``require_auth`` (``_server_legacy_impl.py:556-566``):
    ``auth.JWTManager.verify_token`` then ``get_user_manager().get_user_by_id``.

    Unlike legacy it also requires ``is_active`` and takes the role from the user
    row (not the token), so a demoted or deactivated user loses access immediately.
    """

    def __init__(self, release_roles: frozenset[str] = frozenset({"admin"})) -> None:
        self.release_roles = release_roles

    def verify(self, token: str) -> Optional[Principal]:
        if not token or len(token) > 4096:
            return None
        import auth  # type: ignore  # lazy: backend/ is on sys.path; same module legacy uses

        data = auth.JWTManager.verify_token(token)
        if data is None:
            return None
        user = auth.get_user_manager().get_user_by_id(data.user_id)
        if user is None or not bool(getattr(user, "is_active", False)):
            return None
        try:
            return Principal.for_user(user.id, user.username, user.role, release_roles=self.release_roles)
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

_settings_error: dict[str, Optional[str]] = {"code": None}


@lru_cache(maxsize=1)
def get_line_settings() -> Optional[LineSettings]:
    """Line settings from the environment, or None when not configured/invalid."""
    try:
        settings = load_settings()
    except SettingsError as exc:
        _settings_error["code"] = exc.code
        if exc.code != "LINE_DISABLED":
            logger.error("production line settings refused: %s", exc.code)
        return None
    _settings_error["code"] = None
    return settings


def line_settings_error() -> Optional[str]:
    """The SettingsError code from the last :func:`get_line_settings` load, if any."""
    return _settings_error["code"]


def reset_line_settings_cache() -> None:
    get_line_settings.cache_clear()
    _device_stores.clear()
    _settings_error["code"] = None


def require_line_settings(settings: Optional[LineSettings] = Depends(get_line_settings)) -> LineSettings:
    if settings is None:
        raise HTTPException(status_code=503, detail=NOT_CONFIGURED_DETAIL)
    return settings


# ---------------------------------------------------------------------------
# Stores and verifiers
# ---------------------------------------------------------------------------

_device_stores: dict[Path, DeviceTokenStore] = {}
_device_lock = threading.Lock()


def device_token_store_for(settings: LineSettings) -> DeviceTokenStore:
    path = settings.paths.db_security
    with _device_lock:
        store = _device_stores.get(path)
        if store is None:
            store = DeviceTokenStore(path)
            _device_stores[path] = store
        return store


def get_device_token_store(settings: LineSettings = Depends(require_line_settings)) -> DeviceTokenStore:
    return device_token_store_for(settings)


_verifier_override: dict[str, Optional[UserTokenVerifier]] = {"v": None}


def set_user_token_verifier(verifier: Optional[UserTokenVerifier]) -> None:
    """Install a process-wide user verifier (None restores the legacy one)."""
    _verifier_override["v"] = verifier


def get_user_token_verifier(settings: LineSettings = Depends(require_line_settings)) -> UserTokenVerifier:
    override = _verifier_override["v"]
    if override is not None:
        return override
    return LegacyUserTokenVerifier(release_roles=settings.release_roles)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def parse_bearer(authorization: Optional[str]) -> Optional[str]:
    """The token of an ``Authorization: Bearer <token>`` header, else None."""
    if not authorization:
        return None
    scheme, _, token = authorization.strip().partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token or " " in token:
        return None
    return token


def authenticate_token(
    token: str,
    verifier: Optional[UserTokenVerifier],
    device_store: Optional[DeviceTokenStore],
) -> Optional[Principal]:
    """Synchronous token dispatch shared with the route-policy middleware."""
    if token.startswith(DEVICE_TOKEN_PREFIX):
        return device_store.verify(token) if device_store is not None else None
    if verifier is None:
        return None
    try:
        return verifier.verify(token)
    except Exception:  # a broken verifier must fail closed, never 500 with details
        logger.exception("user token verification failed")
        return None


def _unauthorized(detail: str, code: str) -> HTTPException:
    # A plain-string detail survives the legacy ``http_exception_handler`` (which
    # stringifies it); the stable code travels in a header.
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer", "X-Sherman-Auth-Code": code},
    )


async def get_principal(
    request: Request,
    settings: LineSettings = Depends(require_line_settings),
    verifier: UserTokenVerifier = Depends(get_user_token_verifier),
    device_store: DeviceTokenStore = Depends(get_device_token_store),
) -> Principal:
    cached = getattr(request.state, "sherman_principal", None)
    if isinstance(cached, Principal):
        return cached
    token = parse_bearer(request.headers.get("authorization"))
    if token is None:
        raise _unauthorized("Authentication required", "AUTH_REQUIRED")
    principal = await run_in_threadpool(authenticate_token, token, verifier, device_store)
    if principal is None:
        raise _unauthorized("Invalid or expired token", "TOKEN_INVALID")
    request.state.sherman_principal = principal
    return principal


def require_capability(
    capability: str, station_param: Optional[str] = None
) -> Callable[..., Awaitable[Principal]]:
    """Dependency factory: ``Depends(require_capability("line.read", "station_id"))``.

    The station id is taken from the path parameter ``station_param`` or, failing
    that, the query parameter of the same name.
    """

    async def _dependency(
        request: Request,
        principal: Principal = Depends(get_principal),
        settings: LineSettings = Depends(require_line_settings),
    ) -> Principal:
        station_id: Optional[str] = None
        if station_param:
            raw: Any = request.path_params.get(station_param)
            if raw is None:
                raw = request.query_params.get(station_param)
            station_id = str(raw) if raw is not None else None
        if not has_capability(principal, capability, station_id, release_roles=settings.release_roles):
            raise HTTPException(
                status_code=403,
                detail=f"capability {capability} required",
                headers={"X-Sherman-Auth-Code": "CAPABILITY_REQUIRED"},
            )
        return principal

    _dependency.__name__ = f"require_{capability.replace('.', '_')}"
    return _dependency
