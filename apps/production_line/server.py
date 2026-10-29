"""Production API server of the Sherman QC line (C11).

Usage (from an app copy built with ``apps.production_line.engine_bundle --profile app``,
``cwd=<copy>``, ``PYTHONPATH=<copy>:<copy>/backend``; plan §10 amendment 10)::

    python -m apps.production_line.server [--host 127.0.0.1] [--port 9000]

Startup, in order (each refusal prints its code and exits 2, before the legacy
server is imported, so a refused start writes nothing under the app root):

1. ``WEB_CONCURRENCY`` / ``UVICORN_WORKERS`` other than 1: ``MULTIPLE_WORKERS``.
   The line API runs exactly one worker (SSE, caches and the SQLite stores
   are per process).
2. :func:`~apps.production_line.settings.load_settings`. ``LINE_DISABLED`` is
   not an error: the legacy app is served and every line route answers 503.
   Any other :class:`~apps.production_line.settings.SettingsError` (an unsafe
   data root, a bad station list, an unverifiable app copy) exits 2 with its
   code: a misconfigured line never half-starts.
3. With ``SHERMAN_PRODUCTION_HARDENING=1``: C04
   :func:`~apps.production_line.security.hardening.enforce` (``SECRET_KEY``,
   ``PRODUCTION=1``, the default admin password, self-registration, worker
   count, data-root permissions), probing ``<app root>/data/users.db``
   without side effects.
4. :func:`~apps.production_line.settings.ensure_layout` and C04 JSON logging
   (``<data_root>/logs/line.jsonl``).
5. ``apps.api.main.app`` (the legacy routes plus the C05/C07 routers, which
   ``apps/api/routes/__init__.py`` registers before the SPA catch-all), with
   :func:`~apps.production_line.wiring.install_api_overrides`.
6. The ASGI stack, outermost first: :class:`AccessLogMiddleware` (request id
   plus one JSON access-log line per request, with ``token`` and other
   credential query values redacted), then C04
   :class:`~apps.production_line.security.route_policy.RoutePolicyMiddleware`
   (closes the unauthenticated legacy mutation routes), then the app.
   Uvicorn's own access log is disabled because it would record the SSE
   ``?token=`` query in clear.

:func:`build_asgi` assembles steps 5-6 for an already-loaded app (tests).
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode

__all__ = [
    "ACCESS_LOGGER",
    "REQUEST_ID_HEADER",
    "AccessLogMiddleware",
    "redact_query",
    "build_asgi",
    "worker_count_refusal",
    "main",
]

ACCESS_LOGGER = "sherman_line.access"
REQUEST_ID_HEADER = "x-request-id"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
#: Query keys whose values are never logged (case-insensitive substring match).
_SECRET_QUERY_KEYS = ("token", "password", "passwd", "pwd", "secret", "api_key", "apikey", "authorization")
_TRUE = frozenset({"1", "true", "yes", "on"})

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def redact_query(query: str) -> str:
    """``query`` with the value of every credential-like key replaced by ``[REDACTED]``.

    ``token`` (the SSE stream token), ``access_token``, ``password`` and the like
    are blanked by key; the result is then passed through C04
    :func:`~apps.production_line.telemetry.redact` as a second net (bearer
    values, ``sdt_`` device tokens, JWTs in any position).
    """
    from apps.production_line.telemetry import redact

    if not query:
        return ""
    pairs = parse_qsl(query, keep_blank_values=True)
    out = [(k, "[REDACTED]" if any(s in k.lower() for s in _SECRET_QUERY_KEYS) else v) for k, v in pairs]
    return redact(urlencode(out, safe="[]/:"))


class AccessLogMiddleware:
    """Pure-ASGI request id + access log (module docstring, step 6).

    The request id is the client's ``X-Request-ID`` when it is 8-64 characters of
    ``[A-Za-z0-9._-]``, otherwise a fresh uuid4 hex. It is echoed in the response
    header and bound into the C04 log context for the duration of the request.
    """

    def __init__(self, app: ASGIApp, *, logger: logging.Logger | None = None) -> None:
        self.app = app
        self.log = logger or logging.getLogger(ACCESS_LOGGER)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        from apps.production_line.telemetry import log_context

        given = ""
        for key, value in scope.get("headers") or ():
            if key.lower() == REQUEST_ID_HEADER.encode("ascii"):
                given = value.decode("latin-1").strip()
                break
        request_id = given if _REQUEST_ID_RE.match(given) else uuid.uuid4().hex
        status_holder: dict[str, int] = {"status": 0}
        started = time.monotonic()

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                status_holder["status"] = int(message.get("status", 0))
                headers = list(message.get("headers") or [])
                headers.append((REQUEST_ID_HEADER.encode("ascii"), request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        with log_context(request_id=request_id):
            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                query = scope.get("query_string") or b""
                qtext = redact_query(query.decode("latin-1")) if query else ""
                client = scope.get("client") or ("-", 0)
                self.log.info(
                    "http method=%s path=%s%s status=%s duration_ms=%.1f client=%s",
                    scope.get("method", "-"),
                    scope.get("path", "-"),
                    f"?{qtext}" if qtext else "",
                    status_holder["status"] or "-",
                    (time.monotonic() - started) * 1000.0,
                    client[0] if client else "-",
                )


def worker_count_refusal(env: MutableMapping[str, str] | dict[str, str]) -> Optional[str]:
    """``MULTIPLE_WORKERS: ...`` when the environment asks for more than one worker."""
    for name in ("WEB_CONCURRENCY", "UVICORN_WORKERS"):
        raw = str(env.get(name, "") or "").strip()
        if raw and raw != "1":
            return f"MULTIPLE_WORKERS: {name}={raw}: the line API runs exactly 1 worker"
    return None


def build_asgi(app: Any, settings: Any | None, runtime: Any | None = None) -> ASGIApp:
    """Wire ``app`` (overrides when ``settings`` is given) and wrap it in the
    route policy and the access log (module docstring, steps 5-6)."""
    from apps.production_line.security.dependencies import LegacyUserTokenVerifier
    from apps.production_line.security.route_policy import RoutePolicyMiddleware

    if settings is not None:
        if runtime is None:
            from apps.production_line.wiring import build_runtime

            runtime = build_runtime(settings)
        from apps.production_line.wiring import install_api_overrides

        install_api_overrides(app, runtime)
        # Keep the runtime (and the weakly held release listeners it owns) alive with the app.
        app.state.sherman_line_runtime = runtime
        policy = RoutePolicyMiddleware(
            app,
            verifier=LegacyUserTokenVerifier(release_roles=settings.release_roles),
            device_store=runtime.device_store,
            allow_self_register=settings.allow_self_register,
            legacy_live_scan_enabled=settings.legacy_live_scan_enabled,
            legacy_engine_enabled=settings.legacy_engine_enabled,
            release_roles=tuple(sorted(settings.release_roles)),
        )
    else:
        # Line disabled: legacy app with the route policy; self-registration stays
        # closed unless explicitly allowed, legacy live scan keeps working.
        policy = RoutePolicyMiddleware(
            app,
            verifier=LegacyUserTokenVerifier(),
            device_store=None,
            allow_self_register=str(os.environ.get("SHERMAN_ALLOW_SELF_REGISTER", "")).strip().lower() in _TRUE,
            legacy_live_scan_enabled=True,
            legacy_engine_enabled=True,
        )
    return AccessLogMiddleware(policy)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m apps.production_line.server", description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--log-level", default="INFO")
    return p


def _err(text: str) -> None:
    print(f"[server] {text}", file=sys.stderr, flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    refusal = worker_count_refusal(os.environ)
    if refusal:
        _err(refusal)
        return 2

    from apps.production_line.settings import SettingsError, default_repo_root, ensure_layout, load_settings

    settings = None
    try:
        settings = load_settings()
    except SettingsError as exc:
        if exc.code != "LINE_DISABLED":
            _err(f"{exc.code}: {exc.message}")
            return 2
        _err("LINE_DISABLED: serving the legacy app; /api/line and /api/production answer 503")

    hardening = (
        settings.production_hardening
        if settings is not None
        else str(os.environ.get("SHERMAN_PRODUCTION_HARDENING", "")).strip().lower() in _TRUE
    )
    if hardening:
        from apps.production_line.security.hardening import enforce, legacy_user_probe

        root = settings.repo_root if settings is not None else default_repo_root()
        try:
            enforce(settings, os.environ, legacy_user_probe(Path(root) / "data" / "users.db"))
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else 2

    if settings is not None:
        from apps.production_line.telemetry import DEFAULT_LOGGERS, configure_logging

        ensure_layout(settings.paths)
        configure_logging(settings.paths, args.log_level, settings.log_retention_days,
                          logger_names=tuple(DEFAULT_LOGGERS))
    else:
        logging.basicConfig(level=args.log_level)

    from apps.api.main import app  # imports the legacy server (creates data/, uploads/ under the app root)

    asgi = build_asgi(app, settings)
    import uvicorn

    uvicorn.run(asgi, host=args.host, port=args.port, workers=1, access_log=False, proxy_headers=False,
                log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
