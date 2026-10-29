"""C05 ``/api/production`` routes on a minimal FastAPI app with only this router.

A tmp ``PartCatalogManager`` holds an E2 synthetic STEP hat with its drawing
and bend specs, and a confirmed draft saved with ``save_vendor_setup_draft``;
the Vendor Setup validator runs with a stub legacy context. Capability checks
are the real C04 dependencies with a stub user-token verifier. The release
runs the real E1 compile (supervised subprocess from the repo root).
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from part_catalog import PartCatalogManager

import apps.api.routes.production_recipes as routes
from apps.production_line.recipes.catalog_port import CatalogManagerPort
from apps.production_line.recipes.flat_intake import FlatTool
from apps.production_line.recipes.synthetic_onboarding import (
    onboard_case,
    release_request,
    vendor_setup_loader,
)
from apps.production_line.security import dependencies as deps
from apps.production_line.security.principals import Principal
from apps.production_line.settings import load_settings
from tests.test_prod_release_service import (  # noqa: F401 (fixture)
    LAUNCHER,
    REPO,
    hat_case,
    resave_draft,
)

assert hat_case


class StubVerifier:
    USERS = {
        "tok-admin": ("1", "ada", "admin"),
        "tok-operator": ("2", "otto", "operator"),
        "tok-viewer": ("3", "vera", "viewer"),
    }

    def verify(self, token: str):
        row = self.USERS.get(token)
        return None if row is None else Principal.for_user(*row)


def h(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def api(tmp_path: Path, hat_case: Path):  # noqa: F811
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    settings = load_settings({"SHERMAN_LINE_ENABLED": "1", "SHERMAN_LINE_DATA_ROOT": str(tmp_path / "root")},
                             repo_root=fake_repo)
    deps.reset_line_settings_cache()
    routes.reset_caches()
    catalog = PartCatalogManager(str(tmp_path / "catalog.sqlite"))
    ob = onboard_case(catalog, hat_case, part_number="SYN-HAT-1", upload_dir=tmp_path / "uploads")
    port = CatalogManagerPort(catalog, vendor_setup_loader=vendor_setup_loader(catalog))
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[deps.get_line_settings] = lambda: settings
    app.dependency_overrides[deps.get_user_token_verifier] = lambda: StubVerifier()
    app.dependency_overrides[routes.get_catalog_port] = lambda: port
    app.dependency_overrides[routes.get_compiler] = lambda: LAUNCHER
    app.dependency_overrides[routes.get_flat_tool] = lambda: FlatTool(source_root=REPO, python_exe=Path(sys.executable))
    client = TestClient(app)
    yield SimpleNamespace(client=client, app=app, settings=settings, catalog=catalog, ob=ob, port=port,
                          root=Path(settings.data_root))
    deps.reset_line_settings_cache()
    routes.reset_caches()


def release(api, body=None, token="tok-admin"):
    return api.client.post(f"/api/production/parts/{api.ob.part_id}/recipe/release",
                           json=body if body is not None else release_request(api.ob), headers=h(token))


def codes(resp) -> list[str]:
    return resp.json()["detail"]["codes"]


def line_store(api):
    return routes.get_line_store(api.settings)


def test_release_flow_versions_and_side_effects(api) -> None:
    r = release(api)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["outcome"] == "released" and body["recipe_id"] == "SYN-HAT-1@v1" and body["supersedes_version"] is None
    assert [b["source_feature_id"] for b in body["recipe"]["bends"]] == api.ob.sequence
    part = api.catalog.get_part(api.ob.part_id)
    cad = api.root / "recipes" / "cad" / f"{part.cad_file_hash}.step"
    drawing = api.root / "recipes" / "drawings" / f"{part.drawing_file_hash}.pdf"
    assert cad.is_file() and hashlib.sha256(cad.read_bytes()).hexdigest() == part.cad_file_hash
    assert drawing.is_file() and hashlib.sha256(drawing.read_bytes()).hexdigest() == part.drawing_file_hash
    assert body["compiled"]["status"] == "COMPILED"
    store = line_store(api)
    assert [e["kind"] for e in store.events_after(0)].count("RECIPE_RELEASED") == 1
    assert store.list_jobs() == []  # the compile ran at release; no expectations job

    again = release(api)
    assert again.status_code == 200 and again.json()["outcome"] == "unchanged"
    assert [e["kind"] for e in store.events_after(0)].count("RECIPE_RELEASED") == 1 and store.list_jobs() == []

    req = release_request(api.ob)
    req["default_tolerances"] = {**req["default_tolerances"], "flange_mm": {"plus": 0.3, "minus": 0.3}}
    v2 = release(api, req)
    assert v2.status_code == 201 and v2.json()["recipe_id"] == "SYN-HAT-1@v2" and v2.json()["supersedes_version"] == 1
    for sku in ("SYN-HAT-1", "syn hat 1"):
        got = api.client.get(f"/api/production/recipes/by-sku/{sku}", headers=h("tok-viewer"))
        assert got.status_code == 200 and got.json()["recipe_id"] == "SYN-HAT-1@v2"
        assert [v["version"] for v in got.json()["versions"]] == [1, 2]
    assert api.client.get("/api/production/recipes/by-sku/NOPE-1", headers=h("tok-viewer")).status_code == 404
    v1 = api.client.get("/api/production/recipes/SYN-HAT-1@v1", headers=h("tok-viewer")).json()
    assert v1["status"] == "released" and v1["benchmark_reference"]["status"] == "REGISTERED"

    exp = api.client.get("/api/production/recipes/SYN-HAT-1@v2/expectations", headers=h("tok-viewer")).json()
    assert exp["status"] == "READY" and exp["expectations"]["bends"] and exp["recompute_job"] is None
    rc = api.client.post("/api/production/recipes/SYN-HAT-1@v2/expectations/recompute", headers=h("tok-admin"))
    assert rc.status_code == 202 and rc.json()["created"] is True
    rc2 = api.client.post("/api/production/recipes/SYN-HAT-1@v2/expectations/recompute", headers=h("tok-admin"))
    assert rc2.json() == {"job_id": rc.json()["job_id"], "created": False}
    jobs = store.list_jobs()
    assert len(jobs) == 1 and jobs[0]["kind"] == "recipe_compile" and jobs[0]["recipe_id"] == "SYN-HAT-1@v2"
    exp = api.client.get("/api/production/recipes/SYN-HAT-1@v2/expectations", headers=h("tok-viewer")).json()
    assert exp["recompute_job"]["job_id"] == rc.json()["job_id"]

    assert api.client.post("/api/production/recipes/SYN-HAT-1@v2/retire", headers=h("tok-operator")).status_code == 403
    ret = api.client.post("/api/production/recipes/SYN-HAT-1@v2/retire", headers=h("tok-admin"))
    assert ret.status_code == 200 and ret.json()["status"] == "retired" and ret.json()["active_recipe_id"] == "SYN-HAT-1@v1"
    assert api.client.get("/api/production/recipes/by-sku/SYN-HAT-1", headers=h("tok-viewer")).json()["recipe_id"] == (
        "SYN-HAT-1@v1"
    )
    assert "RECIPE_RETIRED" in [e["kind"] for e in store.events_after(0)]
    onboarding = api.client.get(f"/api/production/parts/{api.ob.part_id}/onboarding", headers=h("tok-viewer")).json()
    assert onboarding["onboarding_state"] == "RELEASED" and onboarding["active_recipe"]["recipe_id"] == "SYN-HAT-1@v1"
    parts = api.client.get("/api/production/parts", headers=h("tok-viewer")).json()["parts"]
    assert [(p["sku"], p["onboarding_state"]) for p in parts] == [("SYN-HAT-1", "RELEASED")]


def test_gate_codes_over_http(api) -> None:
    req = release_request(api.ob)
    r = release(api, {**req, "default_tolerances": None})
    assert r.status_code == 422 and codes(r) == ["TOLERANCES_MISSING"]
    assert r.json()["detail"]["messages"]["TOLERANCES_MISSING"]
    r = release(api, {**req, "expected_draft_revision": 99})
    assert r.status_code == 409 and codes(r) == ["DRAFT_REVISION_CHANGED"]
    saved = resave_draft(api, lambda d: d.update(operator_sequence_confirmed=False))
    r = release(api, {**req, "expected_draft_revision": saved["draft_revision"]})
    assert r.status_code == 422 and "SEQUENCE_NOT_CONFIRMED" in codes(r)

    def one_pending(d):
        d["operator_sequence_confirmed"] = True
        d["marks"][2]["review_status"] = "pending"
    saved = resave_draft(api, one_pending)
    r = release(api, {**req, "expected_draft_revision": saved["draft_revision"]})
    assert r.status_code == 422 and codes(r) == ["UNCONFIRMED_ACTIVE_MARKS"]
    saved = resave_draft(api, lambda d: d["marks"][2].update(review_status="confirmed"))
    req["expected_draft_revision"] = saved["draft_revision"]
    # The catalog CAD is republished after the draft was saved.
    part = api.catalog.get_part(api.ob.part_id)
    new_cad = Path(part.cad_file_path).with_name("republished.step")
    new_cad.write_bytes(Path(part.cad_file_path).read_bytes() + b"\n")
    api.catalog.update_part(api.ob.part_id, cad_file_path=str(new_cad),
                            cad_file_hash=hashlib.sha256(new_cad.read_bytes()).hexdigest(),
                            cad_uploaded_at="2026-10-17T08:00:00")
    r = release(api, req)
    assert r.status_code == 422 and "DRAFT_STALE" in codes(r)
    # No drawing.
    api.catalog.update_part(api.ob.part_id, cad_file_path=part.cad_file_path, cad_file_hash=part.cad_file_hash,
                            cad_uploaded_at=part.cad_uploaded_at)
    nodraw = api.catalog.create_part("NODRAW-1", revision="A")
    api.catalog.update_part(nodraw.id, cad_file_path=part.cad_file_path, cad_file_hash=part.cad_file_hash)
    r = api.client.post(f"/api/production/parts/{nodraw.id}/recipe/release",
                        json={**req, "expected_draft_revision": 0}, headers=h("tok-admin"))
    assert r.status_code == 422 and "DRAWING_MISSING" in codes(r) and "DRAFT_MISSING" in codes(r)
    r = api.client.post("/api/production/parts/missing/recipe/release", json=req, headers=h("tok-admin"))
    assert r.status_code == 404 and codes(r) == ["PART_NOT_FOUND"]
    # Nothing was released by any refusal.
    assert api.client.get("/api/production/recipes/by-sku/SYN-HAT-1", headers=h("tok-viewer")).status_code == 404


def test_create_parts_and_sku_collision(api) -> None:
    r = api.client.post("/api/production/parts", json={"part_number": "abc 1", "part_name": "Bracket"},
                        headers=h("tok-operator"))
    assert r.status_code == 201 and r.json()["sku"] == "ABC-1" and r.json()["part"]["part_number"] == "abc 1"
    r = api.client.post("/api/production/parts", json={"part_number": "ABC-1"}, headers=h("tok-admin"))
    assert r.status_code == 409 and codes(r) == ["SKU_COLLISION"]
    r = api.client.post("/api/production/parts", json={"part_number": "bad/sku"}, headers=h("tok-admin"))
    assert r.status_code == 422 and codes(r) == ["SKU_INVALID"]
    # A catalog part created elsewhere (legacy UI) still collides on the canonical form.
    api.catalog.create_part("zz 9")
    r = api.client.post("/api/production/parts", json={"part_number": "ZZ-9"}, headers=h("tok-admin"))
    assert r.status_code == 409 and codes(r) == ["SKU_COLLISION"]


def test_release_draft_round_trip(api) -> None:
    url = f"/api/production/parts/{api.ob.part_id}/release-draft"
    empty = api.client.get(url, headers=h("tok-viewer")).json()
    assert empty["revision"] == 0 and empty["request"] is None
    body = {k: v for k, v in release_request(api.ob).items()}
    r = api.client.put(url, json={**body, "expected_release_draft_revision": 0}, headers=h("tok-operator"))
    assert r.status_code == 200 and r.json()["revision"] == 1
    assert r.json()["request"]["default_tolerances"]["angle_deg"] == body["default_tolerances"]["angle_deg"]
    assert "expected_release_draft_revision" not in r.json()["request"]
    r = api.client.put(url, json={**body, "expected_release_draft_revision": 0}, headers=h("tok-operator"))
    assert r.status_code == 409 and codes(r) == ["RELEASE_DRAFT_REVISION_CHANGED"]
    bad = {**body, "expected_release_draft_revision": 1,
           "default_tolerances": {**body["default_tolerances"], "flange_mm": {"plus": 0, "minus": 0}}}
    r = api.client.put(url, json=bad, headers=h("tok-operator"))
    assert r.status_code == 422 and codes(r) == ["TOLERANCE_INVALID"]
    # Releasing against a release draft revision someone else replaced is a conflict.
    r = release(api, {**body, "expected_release_draft_revision": 7})
    assert r.status_code == 409 and codes(r) == ["RELEASE_DRAFT_REVISION_CHANGED"]
    doc = api.client.get(f"/api/production/parts/{api.ob.part_id}/onboarding", headers=h("tok-viewer")).json()
    assert doc["onboarding_state"] == "READY_TO_RELEASE" and doc["release_draft_revision"] == 1


def test_capabilities(api) -> None:
    pid = api.ob.part_id
    assert release(api, token="tok-operator").status_code == 403  # SHERMAN_RELEASE_ROLES defaults to admin
    assert api.client.post(f"/api/production/parts/{pid}/recipe/release", json={}).status_code == 401
    assert api.client.post("/api/production/parts", json={"part_number": "V-1"}, headers=h("tok-viewer")).status_code == 403
    _, device = deps.device_token_store_for(api.settings).issue("cell-1", "kiosk", created_by="admin")
    writes = [
        ("post", "/api/production/parts", {"json": {"part_number": "DEV-1"}}),
        ("put", f"/api/production/parts/{pid}/release-draft", {"json": {"expected_draft_revision": 0}}),
        ("post", f"/api/production/parts/{pid}/recipe/release", {"json": release_request(api.ob)}),
        ("post", f"/api/production/parts/{pid}/flat", {"files": {"file": ("f.dxf", b"0\nSECTION\n")}}),
        ("post", f"/api/production/parts/{pid}/flat/proposal", {}),
        ("put", f"/api/production/parts/{pid}/flat/binding", {"json": {"expected_revision": 1, "state": "UNBOUND"}}),
        ("delete", f"/api/production/parts/{pid}/flat", {}),
        ("post", "/api/production/recipes/SYN-HAT-1@v1/expectations/recompute", {}),
        ("post", "/api/production/recipes/SYN-HAT-1@v1/retire", {}),
    ]
    for method, url, kw in writes:
        resp = getattr(api.client, method)(url, headers=h(device), **kw)
        assert resp.status_code == 403, (method, url, resp.status_code)
    # Devices hold line.read only for their station; these routes name none.
    assert api.client.get("/api/production/parts", headers=h(device)).status_code == 403
    assert api.client.get("/api/production/parts", headers=h("tok-viewer")).status_code == 200
    # Nothing was written by any refused call.
    assert line_store(api).events_after(0) == [] and api.catalog.get_part_by_number("DEV-1") is None


def test_not_configured_is_503(api) -> None:
    api.app.dependency_overrides[deps.get_line_settings] = lambda: None
    for method, url in (("get", "/api/production/parts"), ("post", f"/api/production/parts/{api.ob.part_id}/recipe/release")):
        resp = getattr(api.client, method)(url, headers=h("tok-admin"), **({"json": {}} if method == "post" else {}))
        assert resp.status_code == 503


def test_router_import_does_not_load_legacy() -> None:
    code = (
        "import sys, apps.api.routes.production_recipes, apps.api.services.production_recipe_service, "
        "apps.production_line.recipes.release_service, apps.production_line.recipes.resolver, "
        "apps.production_line.recipes.onboarding, apps.production_line.recipes.flat_intake; "
        "bad = [m for m in ('_server_legacy_impl', 'apps.api.legacy_context', 'apps.api.services.vendor_setup_service', "
        "'OCP', 'ezdxf') if m in sys.modules]; assert not bad, bad"
    )
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{REPO}:{REPO / 'backend'}", "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# ---------------------------------------------------------------------------- review fixes


def test_dev_auto_confirmed_is_refused_over_http(api) -> None:
    """Reviewer repro: a PROPOSED (unreviewed) flat must not ride a DEV_AUTO_CONFIRMED release past the gate."""
    pid = api.ob.part_id
    data = (Path(api.ob.case_dir) / "flat.dxf").read_bytes()
    assert api.client.post(f"/api/production/parts/{pid}/flat", files={"file": ("flat.dxf", data)},
                           headers=h("tok-operator")).status_code == 201
    r = api.client.post(f"/api/production/parts/{pid}/flat/proposal?expected_revision=1", headers=h("tok-operator"))
    assert r.status_code == 200 and r.json()["state"] == "PROPOSED"
    r = release(api)
    assert r.status_code == 422 and codes(r) == ["FLAT_BINDING_UNCONFIRMED"]
    r = release(api, {**release_request(api.ob), "confirmation": "DEV_AUTO_CONFIRMED"})
    assert r.status_code == 422 and codes(r) == ["CONFIRMATION_NOT_ALLOWED"]
    assert r.json()["detail"]["messages"]["CONFIRMATION_NOT_ALLOWED"]
    for bad in ("SOMETHING_ELSE", None):
        assert codes(release(api, {**release_request(api.ob), "confirmation": bad})) == ["CONFIRMATION_NOT_ALLOWED"]
    url = f"/api/production/parts/{pid}/release-draft"
    r = api.client.put(url, json={**release_request(api.ob), "confirmation": "DEV_AUTO_CONFIRMED",
                                  "expected_release_draft_revision": 0}, headers=h("tok-operator"))
    assert r.status_code == 422 and codes(r) == ["CONFIRMATION_NOT_ALLOWED"]
    # Nothing was stored, compiled or announced.
    assert api.client.get(url, headers=h("tok-viewer")).json()["revision"] == 0
    assert api.client.get("/api/production/recipes/by-sku/SYN-HAT-1", headers=h("tok-viewer")).status_code == 404
    assert line_store(api).events_after(0) == []
    assert not (api.root / "recipes" / "compiled").exists() or not any(
        p.name[:1] != "." for p in (api.root / "recipes" / "compiled").iterdir()
    )
    # The explicit dev-only wiring still allows it (E4/dev tooling; never set by the shipped router).
    from apps.api.services.production_recipe_service import (
        RecipeApiError,
        RecipeContext,
        RecipeService,
    )
    from apps.production_line.recipes.store import RecipeStore

    ctx = RecipeContext(catalog=api.port, store=RecipeStore(api.settings.paths.db_recipes), data_root=api.root,
                        compiler=None, allow_dev_auto_confirm=True)
    svc = RecipeService(ctx)
    svc._require_operator_confirmation({"confirmation": "DEV_AUTO_CONFIRMED"})  # no raise
    with pytest.raises(RecipeApiError) as err:
        svc._require_operator_confirmation({"confirmation": "SOMETHING_ELSE"})
    assert err.value.codes == ("CONFIRMATION_NOT_ALLOWED",) and err.value.status == 422


def test_create_part_reserves_the_sku_before_the_catalog_row(api) -> None:
    from apps.api.services.production_recipe_service import (
        RecipeApiError,
        RecipeContext,
        RecipeService,
    )
    from apps.production_line.recipes.store import RecipeStore

    store = RecipeStore(api.settings.paths.db_recipes)
    svc = RecipeService(RecipeContext(catalog=api.port, store=store, data_root=api.root))
    real_create = api.port.create_part
    inner: list[str] = []

    def racing_create(**fields):
        # A second creator arrives while the first is inside the catalog insert.
        try:
            svc.create_part({"part_number": "race 7"})
        except RecipeApiError as exc:
            inner.extend(exc.codes)
        return real_create(**fields)

    api.port.create_part = racing_create  # type: ignore[method-assign]
    try:
        out = svc.create_part({"part_number": "RACE-7"})
    finally:
        api.port.create_part = real_create  # type: ignore[method-assign]
    assert inner == ["SKU_COLLISION"] and out["sku"] == "RACE-7"
    assert [p.part_number for p in api.catalog.list_parts(search="RACE")] == ["RACE-7"]  # no orphan row
    assert store.sku_owner("RACE-7") == out["part"]["id"]

    # A failing catalog insert releases the reservation: the SKU is free again.
    def failing_create(**fields):
        raise RuntimeError("catalog down")

    api.port.create_part = failing_create  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError):
            svc.create_part({"part_number": "DOWN-1"})
    finally:
        api.port.create_part = real_create  # type: ignore[method-assign]
    assert store.sku_owner("DOWN-1") is None
    assert svc.create_part({"part_number": "DOWN-1"})["sku"] == "DOWN-1"
    # A catalog collision found after reserving also releases the reservation.
    api.catalog.create_part("leg 5")
    with pytest.raises(RecipeApiError) as err:
        svc.create_part({"part_number": "LEG-5"})
    assert err.value.codes == ("SKU_COLLISION",) and store.sku_owner("LEG-5") is None


def test_release_listeners_are_weak_for_bound_methods(api) -> None:
    import gc

    from apps.production_line.recipes import release_service as rs
    from apps.production_line.recipes.resolver import build_scan_resolver

    before = rs.release_listener_count()
    resolvers = [build_scan_resolver(api.settings) for _ in range(5)]
    assert rs.release_listener_count() == before + 5
    del resolvers
    gc.collect()
    assert rs.release_listener_count() == before
    calls: list[str] = []
    fn = lambda r: calls.append(r)  # noqa: E731 - a plain function is held strongly
    rs.add_release_listener(fn)
    rs.add_release_listener(fn)
    rs.notify_release_listeners("x")
    assert calls == ["x"]
    rs.remove_release_listener(fn)
    rs.notify_release_listeners("y")
    assert calls == ["x"] and rs.release_listener_count() == before


def test_flat_tool_and_compile_use_the_same_e1_root(tmp_path: Path) -> None:
    from apps.production_line.recipes.flat_intake import FlatTool
    from apps.production_line.recipes.release_service import CompileLauncher

    repo = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    repo.mkdir()
    bundle.mkdir()
    settings = SimpleNamespace(repo_root=repo, engine_bundle_dir=bundle, python_exe=Path(sys.executable),
                               rss_limit_bytes=4 * 1024**3, program_status_path=tmp_path / "status.json")
    dev_tool, dev_compile = FlatTool.from_settings(settings), CompileLauncher.from_settings(settings)
    assert dev_tool.engine_root is None and dev_tool.run_root == repo == dev_compile.engine_root
    (bundle / "bundle_manifest.json").write_text("{}")
    tool, launcher = FlatTool.from_settings(settings), CompileLauncher.from_settings(settings)
    assert launcher.engine_root == bundle == tool.engine_root == tool.run_root
    path = tool.env()["PYTHONPATH"].split(":")
    assert path[:2] == [str(bundle), str(repo)]  # E1 from the bundle, this CLI (apps/) from the source root
