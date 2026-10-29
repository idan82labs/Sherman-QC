"""Pure-ASGI middleware that closes unauthenticated legacy mutation routes.

The legacy app exposes several mutating routes with no authentication
(``apps/api/routes/part_catalog.py:10-26``, ``apps/api/routes/live_scan.py:9-22``),
lets anyone self-register an active operator (``_server_legacy_impl.py:601-620``) and
lets anyone delete jobs.  :class:`RoutePolicyMiddleware` is evaluated *before*
routing for every ``http`` scope and applies the first matching rule:

====================  ==============================  ===========================================
Methods               Path (regex, full match)        Action
====================  ==============================  ===========================================
POST                  ``^/api/auth/register$``        deny unless ``allow_self_register``
POST                  ``^/api/parts/import-csv$``     ``parts.import`` (admin)
POST, PUT             ``^/api/parts(/.*)?$``          ``parts.write`` (operator, admin)
DELETE                ``^/api/parts(/.*)?$``          ``parts.delete`` (admin)
DELETE                ``^/api/jobs/[^/]+$``           ``admin``
DELETE                ``^/api/batch/[^/]+$``          ``admin``
POST                  ``^/api/recognize/reindex$``    ``admin``
POST                  ``^/api/(gdt|spc)/.*$``         authenticated user (devices refused)
POST                  ``^/api/recognize``             authenticated user (devices refused)
                      ``(/compare)?$``
any                   ``^/api/admin(/.*)?$``          ``admin``
POST                  ``^/api/(batch/)?analyze$``     ``line.operate`` (operator, admin)
POST                  ``^/api/live-scan/.*$``         deny unless legacy live scan is enabled,
                                                      then ``admin``
GET, HEAD             ``^/api/parts(/.*)?$``          authenticated user (devices refused)
GET, HEAD             ``^/api/live-scan/.*$``         authenticated user (devices refused)
====================  ==============================  ===========================================

Rules match the *route path*: ``scope["path"]`` with ``scope["root_path"]`` removed,
exactly as Starlette routes it, so a ``--root-path``/proxy prefix cannot make a
request miss every rule (fail open).  A path that does not start with a non-empty
``root_path`` is matched as given.

Unmatched requests pass through untouched, and existing route guards still apply
after a rule allows a request.  Refusals are JSON ``{"detail", "code"}`` with status
401 (``AUTH_REQUIRED``/``TOKEN_INVALID``) or 403 (``ROUTE_DISABLED``,
``CAPABILITY_REQUIRED``, ``USER_REQUIRED``).  HEAD is matched as GET because Starlette
serves HEAD through GET routes.  The authenticated principal is stored in
``scope["state"]["sherman_principal"]`` for downstream dependencies.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

from apps.production_line.security.principals import CAPABILITIES, Principal, has_capability

__all__ = ["Rule", "DEFAULT_RULES", "RoutePolicyMiddleware", "match_rule", "route_path"]

logger = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

DENY = "deny"
AUTHENTICATED = "authenticated"


@dataclass(frozen=True)
class Rule:
    """``methods`` × ``pattern`` → ``action`` (a capability, ``deny`` or ``authenticated``).

    ``unless`` names a middleware flag (``allow_self_register`` or
    ``legacy_live_scan_enabled``); when that flag is true the rule is skipped and the
    next matching rule applies.
    """

    methods: frozenset[str]
    pattern: "re.Pattern[str]"
    action: str
    unless: Optional[str] = None

    def __post_init__(self) -> None:
        if self.action not in (DENY, AUTHENTICATED) and self.action not in CAPABILITIES:
            raise ValueError(f"unknown rule action {self.action!r}")
        if self.unless not in (None, "allow_self_register", "legacy_live_scan_enabled", "legacy_engine_enabled"):
            raise ValueError(f"unknown rule flag {self.unless!r}")


def _rule(methods: Iterable[str], pattern: str, action: str, unless: Optional[str] = None) -> Rule:
    return Rule(frozenset(m.upper() for m in methods), re.compile(pattern), action, unless)


DEFAULT_RULES: tuple[Rule, ...] = (
    _rule(["POST"], r"^/api/auth/register$", DENY, unless="allow_self_register"),
    _rule(["POST"], r"^/api/parts/import-csv$", "parts.import"),
    _rule(["POST", "PUT", "PATCH"], r"^/api/parts(/.*)?$", "parts.write"),
    _rule(["DELETE"], r"^/api/parts(/.*)?$", "parts.delete"),
    _rule(["DELETE"], r"^/api/jobs/[^/]+$", "admin"),
    _rule(["DELETE"], r"^/api/batch/[^/]+$", "admin"),
    _rule(["POST"], r"^/api/recognize/reindex$", "admin"),
    # Stateless legacy compute (GD&T, SPC, part recognition): no state changes, but an
    # anonymous LAN client must not be able to burn CPU on them.
    _rule(["POST"], r"^/api/(gdt|spc)/.*$", AUTHENTICATED),
    _rule(["POST"], r"^/api/recognize(/compare)?$", AUTHENTICATED),
    _rule(["GET", "POST", "PUT", "PATCH", "DELETE"], r"^/api/admin(/.*)?$", "admin"),
    # The discarded legacy inspection runtime (backend/bend_analysis_worker.py and the old
    # bend-inspection pipeline) must not be reachable on a production line: these uploads run
    # the old engine, not E3. Opt back in only with SHERMAN_LEGACY_ENGINE=1 (audit 2026-10-19).
    _rule(["POST"], r"^/api/(batch/)?analyze$", DENY, unless="legacy_engine_enabled"),
    _rule(["POST"], r"^/api/bend-inspection/analyze(-from-catalog)?$", DENY, unless="legacy_engine_enabled"),
    _rule(["POST"], r"^/api/(batch/)?analyze$", "line.operate"),
    _rule(["POST"], r"^/api/bend-inspection/analyze(-from-catalog)?$", "line.operate"),
    _rule(["POST", "PUT", "PATCH", "DELETE"], r"^/api/live-scan/.*$", DENY, unless="legacy_live_scan_enabled"),
    _rule(["POST", "PUT", "PATCH", "DELETE"], r"^/api/live-scan/.*$", "admin"),
    _rule(["GET"], r"^/api/parts(/.*)?$", AUTHENTICATED),
    _rule(["GET"], r"^/api/live-scan/.*$", AUTHENTICATED),
)


def match_rule(rules: Sequence[Rule], method: str, path: str, flags: Mapping[str, bool]) -> Optional[Rule]:
    """The first rule that applies to ``method path`` under ``flags``, else None."""
    m = method.upper()
    if m == "HEAD":
        m = "GET"
    for rule in rules:
        if m not in rule.methods or not rule.pattern.fullmatch(path):
            continue
        if rule.unless is not None and flags.get(rule.unless, False):
            continue
        return rule
    return None


def route_path(scope: Scope) -> str:
    """The path Starlette routes on (mirrors ``starlette._utils.get_route_path``):
    ``path`` minus a leading ``root_path`` that ends at a segment boundary."""
    path = str(scope.get("path", "") or "")
    root_path = str(scope.get("root_path", "") or "")
    if not root_path or not path.startswith(root_path):
        return path
    if path == root_path:
        return ""
    if path[len(root_path)] == "/":
        return path[len(root_path):]
    return path


def _bearer(scope: Scope) -> Optional[str]:
    for key, value in scope.get("headers") or ():
        if key.lower() == b"authorization":
            try:
                text = value.decode("latin-1").strip()
            except Exception:  # pragma: no cover - latin-1 never fails
                return None
            scheme, _, token = text.partition(" ")
            token = token.strip()
            if scheme.lower() == "bearer" and token and " " not in token:
                return token
            return None
    return None


class RoutePolicyMiddleware:
    """See the module docstring.  ``verifier`` maps user tokens to principals
    (:class:`~apps.production_line.security.dependencies.UserTokenVerifier`);
    ``device_store`` (optional) verifies ``sdt_`` device tokens."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        rules: Sequence[Rule] = DEFAULT_RULES,
        verifier: Any = None,
        device_store: Any = None,
        allow_self_register: bool = False,
        legacy_live_scan_enabled: bool = False,
        legacy_engine_enabled: bool = False,
        enabled: bool = True,
        release_roles: Iterable[str] = ("admin",),
    ) -> None:
        self.app = app
        self.rules = tuple(rules)
        self.verifier = verifier
        self.device_store = device_store
        self.flags = {
            "allow_self_register": bool(allow_self_register),
            "legacy_live_scan_enabled": bool(legacy_live_scan_enabled),
            "legacy_engine_enabled": bool(legacy_engine_enabled),
        }
        self.enabled = bool(enabled)
        self.release_roles = tuple(release_roles)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self.enabled or scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        rule = match_rule(self.rules, str(scope.get("method", "GET")), route_path(scope), self.flags)
        if rule is None:
            await self.app(scope, receive, send)
            return
        if rule.action == DENY:
            await self._respond(send, 403, "This route is disabled on the production line", "ROUTE_DISABLED")
            return
        token = _bearer(scope)
        if token is None:
            await self._respond(send, 401, "Authentication required", "AUTH_REQUIRED")
            return
        principal = await self._authenticate(token)
        if principal is None:
            await self._respond(send, 401, "Invalid or expired token", "TOKEN_INVALID")
            return
        if rule.action == AUTHENTICATED:
            if principal.kind != "user":
                await self._respond(send, 403, "A user login is required", "USER_REQUIRED")
                return
        elif not has_capability(principal, rule.action, release_roles=self.release_roles):
            await self._respond(send, 403, f"capability {rule.action} required", "CAPABILITY_REQUIRED")
            return
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["sherman_principal"] = principal
        await self.app(scope, receive, send)

    def _dispatch(self, token: str) -> Optional[Principal]:
        """Device tokens to the device store, everything else to the user verifier.
        Any verifier exception fails closed."""
        try:
            if token.startswith("sdt_"):
                return self.device_store.verify(token) if self.device_store is not None else None
            return self.verifier.verify(token) if self.verifier is not None else None
        except Exception:
            logger.exception("route policy token verification failed")
            return None

    async def _authenticate(self, token: str) -> Optional[Principal]:
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(self._dispatch, token)

    @staticmethod
    async def _respond(send: Send, status: int, detail: str, code: str) -> None:
        body = json.dumps({"detail": detail, "code": code}).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"cache-control", b"no-store"),
        ]
        if status == 401:
            headers.append((b"www-authenticate", b"Bearer"))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": False})
