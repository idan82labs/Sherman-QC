"""C04 legacy route-policy middleware and FastAPI capability dependencies.

Everything runs against stub ASGI/FastAPI apps; the legacy server is never imported.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from apps.production_line.security import dependencies as deps
from apps.production_line.security.device_tokens import DeviceTokenStore
from apps.production_line.security.principals import Principal
from apps.production_line.security.route_policy import DEFAULT_RULES, RoutePolicyMiddleware, match_rule, route_path
from apps.production_line.settings import load_settings

USERS = {
    "tok-viewer": Principal.for_user(1, "vera", "viewer"),
    "tok-operator": Principal.for_user(2, "omer", "operator"),
    "tok-admin": Principal.for_user(3, "ada", "admin"),
}


class StubVerifier:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def verify(self, token: str) -> Optional[Principal]:
        self.calls.append(token)
        if token == "tok-explode":
            raise RuntimeError("backend down")
        return USERS.get(token)


class RecordingApp:
    """Downstream ASGI app: records what reached it and answers 200 JSON."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str, Optional[Principal]]] = []

    async def __call__(self, scope, receive, send):  # noqa: ANN001
        if scope["type"] != "http":
            return
        principal = (scope.get("state") or {}).get("sherman_principal")
        self.seen.append((scope["method"], scope["path"], principal))
        body = b'{"ok":true}'
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _client(tmp_path: Path, **kw) -> tuple[TestClient, RecordingApp, DeviceTokenStore]:
    inner = RecordingApp()
    store = DeviceTokenStore(tmp_path / "security.sqlite3")
    app = RoutePolicyMiddleware(inner, verifier=StubVerifier(), device_store=store, **kw)
    return TestClient(app), inner, store


def _h(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_register_denied_by_default_and_allowed_when_enabled(tmp_path: Path) -> None:
    client, inner, _ = _client(tmp_path)
    r = client.post("/api/auth/register", json={"username": "x"})
    assert r.status_code == 403 and r.json()["code"] == "ROUTE_DISABLED"
    assert inner.seen == []
    client2, inner2, _ = _client(tmp_path, allow_self_register=True)
    assert client2.post("/api/auth/register", json={}).status_code == 200
    assert inner2.seen[0][:2] == ("POST", "/api/auth/register")


def test_parts_write_requires_operator(tmp_path: Path) -> None:
    client, inner, _ = _client(tmp_path)
    r = client.post("/api/parts", json={})
    assert r.status_code == 401 and r.json()["code"] == "AUTH_REQUIRED"
    assert r.headers["www-authenticate"] == "Bearer"
    r = client.post("/api/parts", json={}, headers=_h("tok-viewer"))
    assert r.status_code == 403 and r.json()["code"] == "CAPABILITY_REQUIRED"
    r = client.post("/api/parts", json={}, headers=_h("tok-operator"))
    assert r.status_code == 200
    assert inner.seen[-1][2] == USERS["tok-operator"]  # principal handed downstream
    assert client.put("/api/parts/p1", json={}, headers=_h("tok-viewer")).status_code == 403
    assert client.put("/api/parts/p1", json={}, headers=_h("tok-operator")).status_code == 200
    assert client.post("/api/parts/p1/bend-specs", json={}, headers=_h("tok-viewer")).status_code == 403


def test_parts_delete_and_import_require_admin(tmp_path: Path) -> None:
    client, _, _ = _client(tmp_path)
    assert client.delete("/api/parts/x", headers=_h("tok-operator")).status_code == 403
    assert client.delete("/api/parts/x", headers=_h("tok-admin")).status_code == 200
    assert client.post("/api/parts/import-csv", headers=_h("tok-operator")).status_code == 403
    assert client.post("/api/parts/import-csv", headers=_h("tok-admin")).status_code == 200


def test_jobs_delete_admin_only(tmp_path: Path) -> None:
    client, _, _ = _client(tmp_path)
    assert client.delete("/api/jobs/abc").status_code == 401
    assert client.delete("/api/jobs/abc", headers=_h("tok-operator")).status_code == 403
    assert client.delete("/api/jobs/abc", headers=_h("tok-admin")).status_code == 200
    # Not a policy route: only DELETE of a single job id is covered.
    assert client.get("/api/jobs/abc").status_code == 200


def test_live_scan_mutations_disabled_unless_legacy_enabled(tmp_path: Path) -> None:
    client, inner, _ = _client(tmp_path)
    r = client.post("/api/live-scan/config", json={"watch_path": "/"}, headers=_h("tok-admin"))
    assert r.status_code == 403 and r.json()["code"] == "ROUTE_DISABLED"
    assert client.post("/api/live-scan/start").status_code == 403
    assert inner.seen == []
    enabled, _, _ = _client(tmp_path, legacy_live_scan_enabled=True)
    assert enabled.post("/api/live-scan/config", json={}).status_code == 401
    assert enabled.post("/api/live-scan/config", json={}, headers=_h("tok-operator")).status_code == 403
    assert enabled.post("/api/live-scan/config", json={}, headers=_h("tok-admin")).status_code == 200


def test_reads_require_an_authenticated_user(tmp_path: Path) -> None:
    client, _, store = _client(tmp_path)
    assert client.get("/api/parts").status_code == 401
    assert client.get("/api/parts", headers=_h("tok-viewer")).status_code == 200
    assert client.head("/api/parts/p1").status_code == 401
    assert client.get("/api/live-scan/status").status_code == 401
    assert client.get("/api/live-scan/status", headers=_h("tok-viewer")).status_code == 200
    _, device = store.issue("cell-1", "kiosk", created_by="admin")
    r = client.get("/api/parts", headers=_h(device))
    assert r.status_code == 403 and r.json()["code"] == "USER_REQUIRED"
    assert client.post("/api/parts", json={}, headers=_h(device)).status_code == 403


def test_unmatched_routes_untouched(tmp_path: Path) -> None:
    client, inner, _ = _client(tmp_path)
    assert client.get("/api/health").status_code == 200
    assert client.post("/api/auth/login", json={}).status_code == 200
    assert client.get("/station/cell-1").status_code == 200
    assert client.post("/api/production/parts", json={}).status_code == 200
    assert [p for _, p, _ in inner.seen] == ["/api/health", "/api/auth/login", "/station/cell-1", "/api/production/parts"]
    assert all(principal is None for *_, principal in inner.seen)


def test_bad_tokens_fail_closed(tmp_path: Path) -> None:
    client, inner, _ = _client(tmp_path)
    for header in ({"Authorization": "Bearer nope"}, {"Authorization": "Bearer tok-explode"}):
        r = client.post("/api/parts", json={}, headers=header)
        assert r.status_code == 401 and r.json()["code"] == "TOKEN_INVALID"
    for header in ({"Authorization": "Basic dG9rLWFkbWlu"}, {"Authorization": "Bearer"}, {"Authorization": "tok-admin"}):
        assert client.post("/api/parts", json={}, headers=header).status_code == 401
    assert client.post("/api/parts", json={}, headers={"Authorization": "bearer tok-admin"}).status_code == 200
    assert len(inner.seen) == 1


def test_root_path_prefix_cannot_bypass_rules(tmp_path: Path) -> None:
    inner = RecordingApp()
    app = RoutePolicyMiddleware(inner, verifier=StubVerifier(), device_store=DeviceTokenStore(tmp_path / "s.sqlite3"))
    client = TestClient(app, root_path="/sherman")
    r = client.post("/sherman/api/auth/register", json={})
    assert r.status_code == 403 and r.json()["code"] == "ROUTE_DISABLED"
    assert client.delete("/sherman/api/jobs/abc").status_code == 401
    assert client.delete("/sherman/api/jobs/abc", headers=_h("tok-operator")).status_code == 403
    assert client.delete("/sherman/api/jobs/abc", headers=_h("tok-admin")).status_code == 200
    assert client.get("/sherman/api/health").status_code == 200
    assert [p for _, p, _ in inner.seen] == ["/sherman/api/jobs/abc", "/sherman/api/health"]


def test_route_path_mirrors_starlette() -> None:
    from starlette._utils import get_route_path

    cases = [
        {"path": "/api/parts", "root_path": ""},
        {"path": "/sherman/api/parts", "root_path": "/sherman"},
        {"path": "/sherman", "root_path": "/sherman"},
        {"path": "/shermanx/api/parts", "root_path": "/sherman"},
        {"path": "/api/parts", "root_path": "/sherman"},
        {"path": "/api/parts"},
    ]
    for scope in cases:
        assert route_path(scope) == get_route_path(scope), scope


def test_legacy_compute_routes_require_an_authenticated_user(tmp_path: Path) -> None:
    client, inner, store = _client(tmp_path)
    _, device = store.issue("cell-1", "kiosk", created_by="admin")
    for path in ("/api/gdt/flatness", "/api/spc/capability", "/api/recognize", "/api/recognize/compare"):
        r = client.post(path, json={})
        assert r.status_code == 401 and r.json()["code"] == "AUTH_REQUIRED", path
        r = client.post(path, json={}, headers=_h(device))
        assert r.status_code == 403 and r.json()["code"] == "USER_REQUIRED", path
        assert client.post(path, json={}, headers=_h("tok-viewer")).status_code == 200, path
    assert [(m, p) for m, p, _ in inner.seen] == [
        ("POST", p) for p in ("/api/gdt/flatness", "/api/spc/capability", "/api/recognize", "/api/recognize/compare")
    ]
    # Reads stay open, and reindex keeps its stricter admin rule.
    assert client.get("/api/recognize/status").status_code == 200
    assert client.get("/api/gdt/types").status_code == 200
    assert client.post("/api/recognize/reindex", headers=_h("tok-viewer")).status_code == 403


def test_additional_legacy_mutations_closed(tmp_path: Path) -> None:
    client, inner, store = _client(tmp_path)
    # DELETE /api/batch/{id}: admin only.
    assert client.delete("/api/batch/b1").status_code == 401
    assert client.delete("/api/batch/b1", headers=_h("tok-operator")).status_code == 403
    assert client.delete("/api/batch/b1", headers=_h("tok-admin")).status_code == 200
    # FAISS reindex and the /api/admin namespace: admin only.
    assert client.post("/api/recognize/reindex").status_code == 401
    assert client.post("/api/recognize/reindex", headers=_h("tok-operator")).status_code == 403
    assert client.post("/api/recognize/reindex", headers=_h("tok-admin")).status_code == 200
    assert client.post("/api/admin/manuals/ingest").status_code == 401
    assert client.get("/api/admin/manuals", headers=_h("tok-viewer")).status_code == 403
    assert client.post("/api/admin/manuals/ingest", headers=_h("tok-admin")).status_code == 200
    # Legacy analysis uploads (opt-in legacy engine only): operators and admins (line.operate),
    # never anonymous or viewers.
    legacy, legacy_inner, legacy_store = _client(tmp_path, legacy_engine_enabled=True)
    for path in ("/api/analyze", "/api/batch/analyze", "/api/bend-inspection/analyze", "/api/bend-inspection/analyze-from-catalog"):
        assert legacy.post(path).status_code == 401, path
        r = legacy.post(path, headers=_h("tok-viewer"))
        assert r.status_code == 403 and r.json()["code"] == "CAPABILITY_REQUIRED", path
        assert legacy.post(path, headers=_h("tok-operator")).status_code == 200, path
    _, device = legacy_store.issue("cell-1", "kiosk", created_by="admin")
    assert legacy.post("/api/analyze", headers=_h(device)).status_code == 403
    # Reads of batches stay untouched (only DELETE is covered).
    assert client.get("/api/batch/b1").status_code == 200
    seen = [(m, p) for m, p, _ in inner.seen]
    assert ("DELETE", "/api/batch/b1") in seen and ("GET", "/api/batch/b1") in seen
    assert [(m, p) for m, p, _ in legacy_inner.seen].count(("POST", "/api/analyze")) == 1


def test_legacy_engine_routes_refused_on_the_line(tmp_path: Path) -> None:
    """Audit 2026-10-19: the discarded legacy runtime (bend_analysis_worker / old bend-inspection
    pipeline) is unreachable while the line is enabled, even for an admin."""
    client, inner, _ = _client(tmp_path)
    for path in ("/api/analyze", "/api/batch/analyze", "/api/bend-inspection/analyze", "/api/bend-inspection/analyze-from-catalog"):
        for tok in ("tok-admin", "tok-operator"):
            r = client.post(path, headers=_h(tok))
            assert r.status_code == 403 and r.json()["code"] == "ROUTE_DISABLED", (path, tok)
    # Reads of old jobs and the CAD extractor are not engine runs and stay as they were.
    assert client.get("/api/bend-inspection/jobs", headers=_h("tok-viewer")).status_code == 200
    assert inner.seen == [("GET", "/api/bend-inspection/jobs", inner.seen[0][2])]
    flags = {"allow_self_register": False, "legacy_live_scan_enabled": False, "legacy_engine_enabled": False}
    assert match_rule(DEFAULT_RULES, "POST", "/api/analyze", flags).action == "deny"
    assert match_rule(DEFAULT_RULES, "POST", "/api/analyze", dict(flags, legacy_engine_enabled=True)).action == "line.operate"


def test_disabled_middleware_is_transparent(tmp_path: Path) -> None:
    client, inner, _ = _client(tmp_path, enabled=False)
    assert client.post("/api/auth/register").status_code == 200
    assert client.delete("/api/parts/x").status_code == 200
    assert len(inner.seen) == 2


def test_match_rule_order_and_flags() -> None:
    flags = {"allow_self_register": False, "legacy_live_scan_enabled": False}
    assert match_rule(DEFAULT_RULES, "POST", "/api/parts/import-csv", flags).action == "parts.import"
    assert match_rule(DEFAULT_RULES, "POST", "/api/parts/", flags).action == "parts.write"
    assert match_rule(DEFAULT_RULES, "HEAD", "/api/parts", flags).action == "authenticated"
    assert match_rule(DEFAULT_RULES, "POST", "/api/partsX", flags) is None
    assert match_rule(DEFAULT_RULES, "DELETE", "/api/jobs/a/b", flags) is None
    on = dict(flags, legacy_live_scan_enabled=True)
    assert match_rule(DEFAULT_RULES, "POST", "/api/live-scan/start", flags).action == "deny"
    assert match_rule(DEFAULT_RULES, "POST", "/api/live-scan/start", on).action == "admin"


# --------------------------------------------------------------------------- dependencies


@pytest.fixture()
def line_app(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    settings = load_settings(
        {"SHERMAN_LINE_ENABLED": "1", "SHERMAN_LINE_DATA_ROOT": str(tmp_path / "root")}, repo_root=repo
    )
    deps.reset_line_settings_cache()
    app = FastAPI()

    @app.get("/api/line/stations/{station_id}/latest")
    def latest(station_id: str, p: Principal = Depends(deps.require_capability("line.read", "station_id"))):
        return {"station": station_id, "who": p.username, "kind": p.kind}

    @app.get("/api/line/events")
    def events(p: Principal = Depends(deps.require_capability("line.read", "station_id"))):
        return {"who": p.username}

    @app.post("/api/line/jobs/{job_id}/retry")
    def retry(job_id: str, p: Principal = Depends(deps.require_capability("line.operate"))):
        return {"job": job_id}

    @app.post("/api/production/parts/{part_id}/recipe/release")
    def release(part_id: str, p: Principal = Depends(deps.require_capability("recipe.release"))):
        return {"released": part_id}

    app.dependency_overrides[deps.get_line_settings] = lambda: settings
    app.dependency_overrides[deps.get_user_token_verifier] = lambda: StubVerifier()
    yield app, settings
    deps.reset_line_settings_cache()


def test_dependencies_not_configured_returns_503() -> None:
    app = FastAPI()

    @app.get("/x")
    def x(p: Principal = Depends(deps.require_capability("line.read"))):
        return {}

    app.dependency_overrides[deps.get_line_settings] = lambda: None
    r = TestClient(app).get("/x", headers=_h("tok-admin"))
    assert r.status_code == 503 and r.json() == {"detail": "production line not configured"}


def test_dependencies_capabilities(line_app) -> None:
    app, settings = line_app
    client = TestClient(app)
    assert client.get("/api/line/stations/cell-1/latest").status_code == 401
    assert client.get("/api/line/stations/cell-1/latest", headers=_h("bogus")).status_code == 401
    r = client.get("/api/line/stations/cell-1/latest", headers=_h("tok-viewer"))
    assert r.status_code == 200 and r.json()["who"] == "vera"
    assert client.post("/api/line/jobs/j/retry", headers=_h("tok-viewer")).status_code == 403
    assert client.post("/api/line/jobs/j/retry", headers=_h("tok-operator")).status_code == 200
    assert client.post("/api/production/parts/p/recipe/release", headers=_h("tok-operator")).status_code == 403
    assert client.post("/api/production/parts/p/recipe/release", headers=_h("tok-admin")).status_code == 200


def test_dependencies_device_token_station_scope(line_app) -> None:
    app, settings = line_app
    store = deps.device_token_store_for(settings)
    _, token = store.issue("cell-1", "kiosk", created_by="admin")
    client = TestClient(app)
    r = client.get("/api/line/stations/cell-1/latest", headers=_h(token))
    assert r.status_code == 200 and r.json()["kind"] == "device"
    assert client.get("/api/line/stations/cell-2/latest", headers=_h(token)).status_code == 403
    assert client.get("/api/line/events?station_id=cell-1", headers=_h(token)).status_code == 200
    assert client.get("/api/line/events?station_id=cell-2", headers=_h(token)).status_code == 403
    assert client.get("/api/line/events", headers=_h(token)).status_code == 403  # no station named
    assert client.post("/api/line/jobs/j/retry", headers=_h(token)).status_code == 403


def test_parse_bearer() -> None:
    assert deps.parse_bearer("Bearer abc") == "abc"
    assert deps.parse_bearer("bearer  abc ") == "abc"
    for bad in (None, "", "Basic abc", "Bearer", "Bearer a b", "abc"):
        assert deps.parse_bearer(bad) is None


def test_no_legacy_modules_imported() -> None:
    """Importing the security modules must not load legacy/auth/contracts (fresh process,
    so the result does not depend on test order)."""
    import subprocess

    code = (
        "import sys, apps.production_line.settings, apps.production_line.security.route_policy, "
        "apps.production_line.security.dependencies, apps.production_line.security.hardening, "
        "apps.production_line.engine_bundle; "
        "bad=[m for m in sys.modules if m.startswith(('_server_legacy_impl','apps.api','auth','contracts.production'))]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    root = Path(__file__).resolve().parents[1]
    env = {"PYTHONPATH": f"{root}:{root / 'backend'}", "PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"}
    proc = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
