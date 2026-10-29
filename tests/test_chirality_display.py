"""Display-only chirality (left/right) map: apps-side service, worker adapter and /api/production routes.

Synthetic only: a fake compile directory (a tiny ``compiled_part.json`` with stage ids), fake recipe
store / catalog objects, and a fake runner that writes a summary instead of running the research map.
No legacy server import, no E1 compile, no real CAD or scan.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import apps.api.routes.production_recipes as routes
from apps.production_line.recipes import chirality as ch
from apps.production_line.security import dependencies as deps
from apps.production_line.security.principals import Principal
from apps.production_line.settings import load_settings

RECIPE_SHA = "a" * 64
COMPILED_SHA = "b" * 64

SUMMARY = {
    "schema": "chirality_summary_v0", "part_class": "CHIRAL_BY_FEATURES", "left_right_sensitive": True,
    "stage_classes": {"S3": "CHIRAL_BY_FEATURES"},
    "features": [{"feature": 0, "type": "CUTOUT", "opening_in": "proper", "size_mm": [15.0, 28.0], "panel_id": "P2",
                  "centre_final_mm": [10.0, 20.0, 30.0], "max_deviation_mm": 10.0, "verifiable": True}],
    "verifiable": True, "map_sha256": "c" * 64, "thresholds_sha256": "d" * 64,
}


def _compile_dir(root: Path) -> Path:
    d = root / "recipes" / "compiled" / RECIPE_SHA
    d.mkdir(parents=True)
    stages = [{"stage_model_id": f"S{k}", "stage_index": k, "kind": "SEQUENCE_PREFIX"} for k in range(4)]
    stages.append({"stage_model_id": "X1", "stage_index": 1, "kind": "OUT_OF_ORDER"})
    (d / "compiled_part.json").write_text(json.dumps({"compiled_part_sha256": COMPILED_SHA, "stages": stages}))
    return d


class FakeStore:
    def __init__(self, compiled_status: str = "COMPILED") -> None:
        self.recipe = SimpleNamespace(recipe_id="SYN-HAT-1@v1", recipe_sha256=RECIPE_SHA, sku="SYN-HAT-1")
        self.compiled = SimpleNamespace(status=compiled_status, compiled_part_sha256=COMPILED_SHA,
                                        stored_rel_path=f"recipes/compiled/{RECIPE_SHA}/compiled_part.json")
        self.released = True

    def get_active(self, sku):
        return self.recipe if self.released and sku == "SYN-HAT-1" else None

    def get_active_for_part(self, part_id):
        return None

    def get(self, recipe_id):
        return self.recipe if recipe_id == self.recipe.recipe_id else None

    def get_compiled(self, recipe_sha256):
        return self.compiled if recipe_sha256 == RECIPE_SHA else None


class FakeRunner:
    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.calls: list[tuple[Path, Path, list[str] | None]] = []
        self.gate = threading.Event()
        self.gate.set()

    def __call__(self, compiled_dir: Path, out_dir: Path, stages):
        self.calls.append((compiled_dir, out_dir, stages))
        self.gate.wait(10)
        if self.mode == "fail":
            raise ch.ChiralityError("CHIRALITY_TIMEOUT", "too slow")
        body = {"summary": SUMMARY if self.mode == "ok" else {"schema": "nope"}, "stages_computed": stages,
                "computed_at": "2026-10-28T00:00:00Z"}
        (out_dir / ch.SUMMARY_FILE).write_text(json.dumps(body))


def test_final_stage_is_the_last_sequence_prefix(tmp_path: Path) -> None:
    d = _compile_dir(tmp_path)
    assert ch.final_stage_id(d / "compiled_part.json") == "S3"


def test_service_lifecycle_pending_ready_and_storage(tmp_path: Path) -> None:
    d = _compile_dir(tmp_path)
    before = (d / "compiled_part.json").read_bytes()
    runner = FakeRunner()
    runner.gate.clear()
    svc = ch.ChiralityService(data_root=tmp_path, runner=runner)
    store = FakeStore()
    assert svc.read(store, None)["reason"] == "NO_RELEASED_RECIPE"
    idle = svc.read(store, store.recipe)
    assert (idle["status"], idle["reason"]) == ("NOT_COMPUTED", "NOT_SCHEDULED") and not runner.calls
    first = svc.ensure(store, store.recipe)
    assert first["status"] == "PENDING" and first["display_only"] is True and first["compiled_part_sha256"] == COMPILED_SHA
    assert svc.ensure(store, store.recipe)["status"] == "PENDING"  # no second run while one is in flight
    runner.gate.set()
    assert svc.wait_idle()
    ready = svc.ensure(store, store.recipe)
    assert ready["status"] == "READY" and ready["summary"]["part_class"] == "CHIRAL_BY_FEATURES"
    assert ready["stages_computed"] == ["S3"] and ready["recipe_id"] == "SYN-HAT-1@v1"
    assert len(runner.calls) == 1
    compiled_dir, out_dir, stages = runner.calls[0]
    assert compiled_dir == d and stages == ["S3"]
    assert out_dir == tmp_path / "recipes" / "chirality" / COMPILED_SHA  # beside, never inside, the compile
    assert (d / "compiled_part.json").read_bytes() == before and sorted(p.name for p in d.iterdir()) == ["compiled_part.json"]


def test_service_failure_is_recorded_not_raised_and_retry(tmp_path: Path) -> None:
    _compile_dir(tmp_path)
    runner = FakeRunner("fail")
    svc = ch.ChiralityService(data_root=tmp_path, runner=runner)
    store = FakeStore()
    svc.ensure(store, store.recipe)
    assert svc.wait_idle()
    failed = svc.ensure(store, store.recipe)
    assert failed["status"] == "FAILED" and failed["reason"] == "CHIRALITY_TIMEOUT" and failed["summary"] is None
    assert len(runner.calls) == 1  # a failure is not retried by a read
    runner.mode = "ok"
    assert svc.ensure(store, store.recipe, retry_failed=True)["status"] == "PENDING"
    assert svc.wait_idle()
    assert svc.read(store, store.recipe)["status"] == "READY"
    assert not (tmp_path / "recipes" / "chirality" / COMPILED_SHA / ch.FAILED_FILE).exists()


def test_service_invalid_output_and_missing_compile(tmp_path: Path) -> None:
    _compile_dir(tmp_path)
    svc = ch.ChiralityService(data_root=tmp_path, runner=FakeRunner("bad"))
    store = FakeStore()
    svc.ensure(store, store.recipe)
    assert svc.wait_idle()
    assert svc.read(store, store.recipe)["reason"] == "CHIRALITY_OUTPUT_INVALID"
    blocked = FakeStore(compiled_status="BLOCKED")
    doc = svc.ensure(blocked, blocked.recipe)
    assert (doc["status"], doc["reason"]) == ("NOT_COMPUTED", "COMPILE_NOT_INSTALLED")


def test_worker_writes_summary_last(tmp_path: Path, monkeypatch) -> None:
    from apps.production_line.recipes import chirality_worker
    from research.chirality import chirality_map as cm

    def fake_write(cdir, out_dir=None, stages=None):
        p = Path(out_dir) / "chirality_map_v0.json"
        p.write_text(json.dumps({"stages": [{"stage_model_id": s} for s in stages]}))
        return p

    monkeypatch.setattr(cm, "write", fake_write)
    monkeypatch.setattr(cm, "load", lambda p: (json.loads(Path(p).read_text()), {}))
    monkeypatch.setattr(cm, "summary", lambda doc: SUMMARY)
    assert chirality_worker.main([str(tmp_path), "--out", str(tmp_path), "--stages", "S3"]) == 0
    body = json.loads((tmp_path / ch.SUMMARY_FILE).read_text())
    assert body["summary"] == SUMMARY and body["stages_computed"] == ["S3"]


class StubVerifier:
    USERS = {"tok-admin": ("1", "ada", "admin"), "tok-viewer": ("3", "vera", "viewer")}

    def verify(self, token: str):
        row = self.USERS.get(token)
        return None if row is None else Principal.for_user(*row)


@pytest.fixture()
def api(tmp_path: Path):
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    settings = load_settings({"SHERMAN_LINE_ENABLED": "1", "SHERMAN_LINE_DATA_ROOT": str(tmp_path / "root")},
                             repo_root=fake_repo)
    deps.reset_line_settings_cache()
    routes.reset_caches()
    root = Path(settings.data_root)
    _compile_dir(root)
    store = FakeStore()
    catalog = SimpleNamespace(get_part=lambda pid: {"id": pid, "part_number": "SYN-HAT-1"} if pid == "P1" else None)
    runner = FakeRunner()
    svc = ch.ChiralityService(data_root=root, runner=runner)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[deps.get_line_settings] = lambda: settings
    app.dependency_overrides[deps.get_user_token_verifier] = lambda: StubVerifier()
    app.dependency_overrides[routes.get_recipe_store] = lambda: store
    app.dependency_overrides[routes.get_catalog_port] = lambda: catalog
    app.dependency_overrides[routes.get_chirality_service] = lambda: svc
    yield SimpleNamespace(client=TestClient(app), settings=settings, store=store, svc=svc, runner=runner)
    deps.reset_line_settings_cache()
    routes.reset_caches()


def h(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_routes_part_schedules_station_reads_only(api) -> None:
    station = api.client.get("/api/production/recipes/SYN-HAT-1@v1/chirality", headers=h("tok-viewer"))
    assert station.status_code == 200 and station.json()["status"] == "NOT_COMPUTED" and not api.runner.calls
    r = api.client.get("/api/production/parts/P1/chirality", headers=h("tok-viewer"))
    assert r.status_code == 200 and r.json()["status"] in ("PENDING", "READY") and r.json()["part_id"] == "P1"
    assert api.svc.wait_idle()
    ready = api.client.get("/api/production/parts/P1/chirality", headers=h("tok-viewer")).json()
    assert ready["status"] == "READY" and ready["summary"]["left_right_sensitive"] is True
    got = api.client.get("/api/production/recipes/SYN-HAT-1@v1/chirality", headers=h("tok-viewer")).json()
    assert got["summary"] == SUMMARY and got["compiled_part_sha256"] == COMPILED_SHA
    assert api.client.get("/api/production/parts/NOPE/chirality", headers=h("tok-viewer")).status_code == 404
    assert api.client.get("/api/production/recipes/X@v9/chirality", headers=h("tok-viewer")).status_code == 404
    api.store.released = False
    assert api.client.get("/api/production/parts/P1/chirality", headers=h("tok-viewer")).json()["reason"] == "NO_RELEASED_RECIPE"


def test_routes_capabilities(api) -> None:
    assert api.client.get("/api/production/parts/P1/chirality").status_code == 401
    assert api.client.post("/api/production/parts/P1/chirality/retry", headers=h("tok-viewer")).status_code == 403
    assert api.client.post("/api/production/parts/P1/chirality/retry", headers=h("tok-admin")).status_code == 200
    _, device = deps.device_token_store_for(api.settings).issue("cell-1", "kiosk", created_by="admin")
    url = "/api/production/recipes/SYN-HAT-1@v1/chirality"
    assert api.client.get(url, headers=h(device)).status_code == 403  # a device must name its station
    assert api.client.get(f"{url}?station_id=cell-1", headers=h(device)).status_code == 200
    assert api.client.get("/api/production/parts/P1/chirality", headers=h(device)).status_code == 403
    assert api.svc.wait_idle()


def test_not_configured_is_503(api) -> None:
    api.client.app.dependency_overrides[deps.get_line_settings] = lambda: None
    assert api.client.get("/api/production/parts/P1/chirality", headers=h("tok-viewer")).status_code == 503

