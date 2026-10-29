"""C07 line API routes: scoping, jobs, feed/ETag, quarantine, retry/cancel,
devices, status, readiness and metrics.

Every test runs a minimal FastAPI app that includes only
``apps.api.routes.production_line.router``, a tmp data root, a real C02
``LineStore``, a fake ``RecipeProvider`` and the C01 example JSON copied into
tmp job workspaces. No real scans or CAD are read.

This module also holds the shared harness (``LineHarness``, ``harness``)
imported by ``test_prod_line_events_stream.py`` and
``test_prod_line_artifacts.py``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes.production_line import router
from apps.api.services import production_line_service as svc
from apps.production_line import health
from apps.production_line.ingest.ports import ActiveRecipeRef, ResolvedRecipe
from apps.production_line.security import dependencies as deps
from apps.production_line.security.principals import Principal
from apps.production_line.settings import LineSettings, load_settings
from apps.production_line.store.line_store import LineStore, iso_utc
from contracts.production.canonical import canonical_json_bytes
from contracts.production.inspection_result import ShopfloorInspectionResultV1, StationFeedV1

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "contracts" / "production" / "examples"
RECIPE_DOC = json.loads((EXAMPLES / "part_recipe_v1.example.json").read_text())
COMPILED_DOC = json.loads((EXAMPLES / "compiled_part_v1.example.json").read_text())
RESULT_EXAMPLE = "shopfloor_result_v1.example.json"
SKU = RECIPE_DOC["sku"]
RECIPE_ID = RECIPE_DOC["recipe_id"]
RECIPE_SHA = RECIPE_DOC["recipe_sha256"]
COMPILED_SHA = json.loads((EXAMPLES / RESULT_EXAMPLE).read_text())["engine"]["compiled_part_sha256"]
STATION, OTHER = "cell-1", "cell-2"

USERS = {
    "tok-admin": Principal.for_user(1, "alice", "admin"),
    "tok-operator": Principal.for_user(2, "oscar", "operator"),
    "tok-viewer": Principal.for_user(3, "vera", "viewer"),
}


class StubVerifier:
    def verify(self, token: str) -> Principal | None:
        return USERS.get(token)


class FakeProvider:
    """RecipeProvider + ActiveRecipeCatalog over in-memory ResolvedRecipe values.

    The active recipe of a SKU is its highest version whose ``extra.status``
    is absent or ``released`` (C05's ``list_active`` rule).
    """

    def __init__(self) -> None:
        self.recipes: dict[str, ResolvedRecipe] = {}
        self.calls: list[str] = []
        self.registered: set[str] = set()

    def list_active_recipes(self) -> list[ActiveRecipeRef]:
        best: dict[str, ResolvedRecipe] = {}
        for r in self.recipes.values():
            if (r.extra or {}).get("status", "released") != "released":
                continue
            if r.sku not in best or r.recipe_version > best[r.sku].recipe_version:
                best[r.sku] = r
        out = []
        for r in best.values():
            status = (r.extra or {}).get("compile_status") or ("COMPILED" if r.compiled_part_sha256 else "PENDING")
            out.append(
                ActiveRecipeRef(
                    sku=r.sku,
                    recipe_id=r.recipe_id,
                    recipe_version=r.recipe_version,
                    recipe_sha256=r.recipe_sha256,
                    compiled_part_sha256=r.compiled_part_sha256 if status == "COMPILED" else None,
                    compile_status=status,
                )
            )
        return out

    def list_registered_skus(self) -> list[str]:
        return sorted(self.registered | {r.sku for r in self.recipes.values()})

    def add(self, recipe_id: str, *, compiled: str | None = COMPILED_SHA, extra: dict | None = None) -> None:
        sku, version = recipe_id.split("@v")
        self.recipes[recipe_id] = ResolvedRecipe(
            recipe_id=recipe_id,
            recipe_version=int(version),
            recipe_sha256=RECIPE_SHA,
            sku=sku,
            recipe_json_bytes=b"{}",
            cad_path=Path("/nonexistent/cad.step"),
            cad_sha256="e" * 64,
            compiled_part_sha256=compiled,
            extra=extra or {},
        )

    def load(self, recipe_id: str) -> ResolvedRecipe:
        self.calls.append(recipe_id)
        return self.recipes[recipe_id]


class StepClock:
    """Strictly increasing store clock: +10 ms per call (stable ordering)."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime.now(UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(milliseconds=10)
        return self.now


def h(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def result_doc(job: dict[str, Any], example: str = RESULT_EXAMPLE) -> dict[str, Any]:
    """The C01 example result rebound to ``job`` (context, scan sha)."""
    doc = json.loads((EXAMPLES / example).read_text())
    doc["context"]["job_id"] = job["job_id"]
    doc["context"]["station_id"] = job["station_id"]
    doc["scan"]["sha256"] = job["scan_sha256"]
    return doc


def summary_of(doc: dict[str, Any]) -> dict[str, Any]:
    res = ShopfloorInspectionResultV1.model_validate(doc)
    return {
        "inspection_validity": res.inspection_validity.state,
        "summary_state": res.summary.state,
        "bends_total": res.summary.bends_total,
        "required_fail": res.summary.required_fail,
        "required_review": res.summary.required_review,
        "required_unknown": res.summary.required_unknown,
        "stage_status": res.stage_decision.status,
    }


@dataclass
class LineHarness:
    tmp: Path
    settings: LineSettings
    store: LineStore
    provider: FakeProvider
    app: FastAPI
    client: TestClient
    config: svc.LineApiConfig
    clock: StepClock
    _n: int = 0
    probe_calls: dict[str, int] = field(default_factory=lambda: {"python": 0, "bundle": 0})

    # ------------------------------------------------------------- paths
    def ws(self, job_id: str) -> Path:
        return self.settings.paths.jobs / job_id

    # ------------------------------------------------------------- tokens
    def device_token(self, station: str = STATION, label: str = "kiosk") -> tuple[str, str]:
        return deps.device_token_store_for(self.settings).issue(station, label, created_by="test")

    # ------------------------------------------------------------- scans & jobs
    def _sha(self) -> str:
        self._n += 1
        return f"{self._n:024x}" + "ab" * 20

    def scan(
        self,
        station: str = STATION,
        name: str | None = None,
        serial: str | None = "SN0042",
        quarantine: str | None = None,
        detail: dict | None = None,
    ) -> dict[str, Any]:
        sha = self._sha()
        parse: dict[str, Any] = {"status": "OK", "sku": SKU, "serial": serial}
        match: dict[str, Any]
        if quarantine:
            match = {"status": "UNMATCHED", "reason": "no recipe"}
            parse = {"status": "UNMATCHED", "sku": None, "serial": serial}
        else:
            match = {"status": "MATCHED", "recipe_id": RECIPE_ID, "recipe_version": 1, "recipe_sha256": RECIPE_SHA}
        row, created = self.store.register_scan(
            sha256=sha,
            original_name=name or f"{SKU}_A_{serial or 'X'}_{self._n}.ply",
            size_bytes=1234,
            station_id=station,
            stored_rel_path=f"archive/scans/2026/10/{sha}.ply",
            parse=parse,
            match=match,
            format="PLY_BINARY_LE",
            ingest_state="quarantined" if quarantine else "received",
            quarantine_reason=quarantine,
            quarantine_detail=detail,
        )
        assert created
        return row

    def queue_job(self, station: str = STATION, serial: str | None = "SN0042") -> dict[str, Any]:
        scan = self.scan(station, serial=serial)
        job_id, created = self.store.match_and_enqueue(
            scan["scan_file_id"],
            match={"status": "MATCHED", "recipe_id": RECIPE_ID, "recipe_version": 1, "recipe_sha256": RECIPE_SHA},
            sku=SKU,
            compiled_part_sha256=COMPILED_SHA,
        )
        assert created
        job = self.store.get_job(job_id)
        assert job is not None
        return job

    def start(self, job: dict[str, Any]) -> dict[str, Any]:
        leased = self.store.lease_next("runner@test", 600)
        assert leased is not None and leased["job_id"] == job["job_id"], "test queues one job at a time"
        return leased

    def write_workspace(
        self,
        job: dict[str, Any],
        doc: dict[str, Any] | None = None,
        *,
        artifacts: bool = True,
        context: bool = True,
    ) -> Path:
        ws = self.ws(job["job_id"])
        (ws / "result").mkdir(parents=True, exist_ok=True)
        (ws / "engine").mkdir(parents=True, exist_ok=True)
        (ws / "job.json").write_text(json.dumps({"job_id": job["job_id"]}))
        (ws / "engine" / "bend_inspection_report.json").write_text("{}")
        if context:
            (ws / "compiled").mkdir(exist_ok=True)
            (ws / "recipe.json").write_bytes(canonical_json_bytes(RECIPE_DOC))
            (ws / "compiled" / "compiled_part.json").write_bytes(canonical_json_bytes(COMPILED_DOC))
        if doc is not None:
            (ws / "result" / "shopfloor_result.json").write_bytes(canonical_json_bytes(doc))
        if artifacts:
            (ws / "result" / "viewer_scan.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n" + b"\0" * 24)
            (ws / "result" / "viewer_deviations.f32").write_bytes(b"\0\0\0\0" * 10)
            (ws / "result" / "report.pdf").write_bytes(b"%PDF-1.4\n%test\n")
            (ws / "result" / "bend_geometry.json").write_text('{"schema":"sherman_bend_geometry_v1"}')
        return ws

    def completed_job(
        self,
        station: str = STATION,
        *,
        serial: str | None = "SN0042",
        example: str = RESULT_EXAMPLE,
        mutate: Callable[[dict[str, Any]], None] | None = None,
        write_result: bool = True,
        artifacts: bool = True,
        context: bool = True,
    ) -> dict[str, Any]:
        job = self.start(self.queue_job(station, serial=serial))
        doc = result_doc(job, example)
        summary = summary_of(doc)
        if mutate is not None:
            mutate(doc)
        self.write_workspace(job, doc if write_result else None, artifacts=artifacts, context=context)
        assert self.store.complete_job(job["job_id"], "runner@test", summary, duration_s=12.0)
        out = self.store.get_job(job["job_id"])
        assert out is not None
        return out

    def failed_job(self, station: str = STATION, code: str = "WORKER_TIMEOUT", message: str = "engine exceeded 180 s") -> dict[str, Any]:
        job = self.start(self.queue_job(station))
        assert self.store.fail_job(job["job_id"], "runner@test", code, message)
        out = self.store.get_job(job["job_id"])
        assert out is not None
        return out

    def feed(self, token: str = "tok-viewer", station: str = STATION, **headers: str) -> Any:
        return self.client.get(f"/api/line/stations/{station}/latest", headers={**h(token), **headers})


def build_harness(tmp_path: Path, config: svc.LineApiConfig | None = None, stations: tuple[str, ...] = (STATION, OTHER)) -> LineHarness:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    station_cfg = []
    for sid in stations:
        inbox = tmp_path / f"inbox-{sid}"
        inbox.mkdir(exist_ok=True)
        station_cfg.append({"station_id": sid, "inbox": str(inbox)})
    settings = load_settings(
        {
            "SHERMAN_LINE_ENABLED": "1",
            "SHERMAN_LINE_DATA_ROOT": str(tmp_path / "root"),
            "SHERMAN_LINE_STATIONS": json.dumps(station_cfg),
            "SHERMAN_LINE_ENGINE_BUNDLE_DIR": str(tmp_path / "bundle"),
        },
        repo_root=repo,
    )
    deps.reset_line_settings_cache()
    svc.reset_line_api_caches()
    clock = StepClock()
    store = LineStore(settings.paths.db_line, clock=clock)
    provider = FakeProvider()
    provider.add(RECIPE_ID)
    cfg = config or svc.LineApiConfig(
        min_free_bytes=0, sse_heartbeat_s=0.1, sse_poll_s=0.02, sse_max_lifetime_s=0.4, readiness_ttl_s=0
    )
    probe_calls = {"python": 0, "bundle": 0}

    def fake_python() -> health.ProbeResult:
        probe_calls["python"] += 1
        return health.ProbeResult("engine_python", True, "imports ok")

    def fake_bundle() -> health.ProbeResult:
        probe_calls["bundle"] += 1
        return health.ProbeResult("engine_bundle", False, "bundle manifest missing")

    probes = svc.StatusProbes(settings, cfg, python_probe=fake_python, bundle_probe=fake_bundle)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[deps.get_line_settings] = lambda: settings
    app.dependency_overrides[deps.get_user_token_verifier] = lambda: StubVerifier()
    app.dependency_overrides[svc.get_line_store] = lambda: store
    app.dependency_overrides[svc.get_recipe_provider] = lambda: provider
    app.dependency_overrides[svc.get_line_api_config] = lambda: cfg
    app.dependency_overrides[svc.get_status_probes] = lambda: probes
    harness = LineHarness(
        tmp=tmp_path,
        settings=settings,
        store=store,
        provider=provider,
        app=app,
        client=TestClient(app),
        config=cfg,
        clock=clock,
        probe_calls=probe_calls,
    )
    return harness


@pytest.fixture()
def harness(tmp_path: Path) -> Iterator[LineHarness]:
    hs = build_harness(tmp_path)
    yield hs
    hs.client.close()
    deps.reset_line_settings_cache()
    svc.reset_line_api_caches()


def beat(hs: LineHarness, component: str, age_s: float = 0.0) -> None:
    """Write a heartbeat whose timestamp is ``age_s`` seconds old (real clock)."""
    when = datetime.now(UTC) - timedelta(seconds=age_s)
    LineStore(hs.settings.paths.db_line, clock=lambda: when).put_heartbeat(component, f"{component}@test", {})


# =========================================================================== import discipline


def test_router_import_does_not_load_legacy_or_recipes() -> None:
    code = (
        "import sys, apps.api.routes.production_line; "
        "bad=[m for m in sys.modules if m.startswith(('_server_legacy_impl','apps.api.legacy_context',"
        "'apps.production_line.recipes'))]; assert not bad, bad"
    )
    env = dict(os.environ, PYTHONPATH=f"{REPO}:{REPO / 'backend'}", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


def test_router_paths_are_under_api_line() -> None:
    paths = {r.path for r in router.routes}  # type: ignore[attr-defined]
    assert paths and all(p.startswith("/api/line/") for p in paths)
    for expected in (
        "/api/line/jobs",
        "/api/line/jobs/{job_id}",
        "/api/line/jobs/{job_id}/result",
        "/api/line/jobs/{job_id}/artifacts/{name:path}",
        "/api/line/jobs/{job_id}/retry",
        "/api/line/jobs/{job_id}/cancel",
        "/api/line/stations/{station_id}/latest",
        "/api/line/events",
        "/api/line/stream-token",
        "/api/line/events/stream",
        "/api/line/quarantine",
        "/api/line/quarantine/{scan_file_id}/assign",
        "/api/line/quarantine/{scan_file_id}/dismiss",
        "/api/line/status",
        "/api/line/health/ready",
        "/api/line/features",
        "/api/line/devices",
        "/api/line/devices/{token_id}",
        "/api/line/metrics",
    ):
        assert expected in paths, expected


# =========================================================================== not configured


def test_not_configured_returns_503(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[deps.get_line_settings] = lambda: None
    with TestClient(app) as client:
        for method, url in (
            ("get", "/api/line/stations/cell-1/latest"),
            ("get", "/api/line/jobs"),
            ("post", "/api/line/jobs/line_" + "0" * 32 + "/retry"),
            ("get", "/api/line/events/stream?station_id=cell-1&token=x"),
            ("get", "/api/line/metrics"),
        ):
            r = getattr(client, method)(url, headers=h("tok-admin"))
            assert r.status_code == 503, (url, r.status_code)
        r = client.get("/api/line/health/ready")
        assert r.status_code == 503
        assert r.json() == {"ready": False, "checks": {"configured": "fail"}}
        r = client.get("/api/line/features")  # line disabled: the legacy pages stay
        assert r.status_code == 200 and r.json() == {"line_enabled": False, "legacy_engine_enabled": True}


def test_features_hide_the_legacy_engine_on_the_line(harness: LineHarness) -> None:
    r = harness.client.get("/api/line/features")  # no auth needed; flags only, no paths
    assert r.status_code == 200
    assert r.json() == {"line_enabled": True, "legacy_engine_enabled": bool(harness.settings.legacy_engine_enabled)}
    assert r.json()["legacy_engine_enabled"] is False  # SHERMAN_LEGACY_ENGINE defaults to 0


def test_unauthenticated_is_401(harness: LineHarness) -> None:
    r = harness.client.get("/api/line/stations/cell-1/latest")
    assert r.status_code == 401
    r = harness.client.get("/api/line/stations/cell-1/latest", headers=h("tok-nobody"))
    assert r.status_code == 401


# =========================================================================== station scoping


def test_device_token_station_scoping(harness: LineHarness) -> None:
    own = harness.completed_job(STATION)
    other = harness.completed_job(OTHER)
    _, device = harness.device_token(STATION)

    assert harness.feed(device, STATION).status_code == 200
    assert harness.feed(device, OTHER).status_code == 403

    # Another station's job is invisible (404), not forbidden.
    r = harness.client.get(f"/api/line/jobs/{other['job_id']}/artifacts/report.pdf", headers=h(device))
    assert r.status_code == 404
    assert harness.client.get(f"/api/line/jobs/{other['job_id']}", headers=h(device)).status_code == 404
    assert harness.client.get(f"/api/line/jobs/{other['job_id']}/result", headers=h(device)).status_code == 404
    # Its own station's job is readable.
    assert harness.client.get(f"/api/line/jobs/{own['job_id']}/artifacts/report.pdf", headers=h(device)).status_code == 200
    assert harness.client.get(f"/api/line/jobs/{own['job_id']}/result", headers=h(device)).status_code == 200

    # A device can never operate.
    for action in ("retry", "cancel"):
        r = harness.client.post(f"/api/line/jobs/{own['job_id']}/{action}", headers=h(device))
        assert r.status_code == 403, action
        r = harness.client.post(f"/api/line/jobs/{other['job_id']}/{action}", headers=h(device))
        assert r.status_code == 403, action

    # Listing: a device must name its own station; it then sees only that station.
    assert harness.client.get("/api/line/jobs", headers=h(device)).status_code == 403
    assert harness.client.get(f"/api/line/jobs?station_id={OTHER}", headers=h(device)).status_code == 403
    r = harness.client.get(f"/api/line/jobs?station_id={STATION}", headers=h(device))
    assert r.status_code == 200
    assert [j["job_id"] for j in r.json()["jobs"]] == [own["job_id"]]
    # Devices cannot manage devices or read metrics/status.
    assert harness.client.get("/api/line/devices", headers=h(device)).status_code == 403
    assert harness.client.get("/api/line/metrics", headers=h(device)).status_code == 403
    assert harness.client.get("/api/line/status", headers=h(device)).status_code == 403


def test_unknown_station_feed_is_404(harness: LineHarness) -> None:
    assert harness.feed("tok-viewer", "cell-9").status_code == 404
    r = harness.feed("tok-viewer", "Cell_1")
    assert r.status_code == 422
    assert r.headers["X-Sherman-Error-Code"] == "STATION_ID_INVALID"


# =========================================================================== feed and ETag


def test_feed_completed_result_etag_and_issue(harness: LineHarness) -> None:
    first = harness.completed_job(STATION)
    r = harness.feed()
    assert r.status_code == 200
    feed = StationFeedV1.model_validate(r.json())
    assert feed.latest_completed is not None
    assert feed.latest_completed.job.job_id == first["job_id"]
    assert feed.latest_completed.result.summary.state == "OUT_OF_TOLERANCE_FOUND"
    assert feed.latest_completed.job.part_name == "Synthetic hat section"
    assert feed.latest_completed.job.serial == "SN0042"
    urls = feed.latest_completed.artifacts
    assert urls["report.pdf"] == f"/api/line/jobs/{first['job_id']}/artifacts/report.pdf"
    assert set(urls) == {"viewer_scan.ply", "viewer_deviations.f32", "report.pdf", "bend_geometry.json"}
    assert feed.active is None and feed.latest_issue is None
    etag = r.headers["ETag"]
    # Weak: the body also carries server_time, so it is not byte-identical per ETag.
    assert feed.etag == etag and etag.startswith('W/"')

    again = harness.feed(**{"If-None-Match": etag})
    assert again.status_code == 304 and again.headers["ETag"] == etag and not again.content
    assert harness.feed(**{"If-None-Match": f"{etag[2:]}, \"other\""}).status_code == 304
    assert harness.feed(**{"If-None-Match": '"stale"'}).status_code == 200

    newer = harness.completed_job(STATION, serial="SN0043")
    r2 = harness.feed(**{"If-None-Match": etag})
    assert r2.status_code == 200 and r2.headers["ETag"] != etag
    feed2 = StationFeedV1.model_validate(r2.json())
    assert feed2.latest_completed is not None and feed2.latest_completed.job.job_id == newer["job_id"]
    assert [j.job_id for j in feed2.recent] == [newer["job_id"], first["job_id"]]

    failed = harness.failed_job(STATION, message=f"bad output under {harness.settings.data_root}/jobs/x")
    r3 = harness.feed(**{"If-None-Match": r2.headers["ETag"]})
    assert r3.status_code == 200 and r3.headers["ETag"] != r2.headers["ETag"]
    feed3 = StationFeedV1.model_validate(r3.json())
    assert feed3.latest_completed is not None and feed3.latest_completed.job.job_id == newer["job_id"]
    assert feed3.latest_issue is not None
    assert feed3.latest_issue.kind == "JOB_FAILED" and feed3.latest_issue.reason_code == "WORKER_TIMEOUT"
    assert feed3.latest_issue.at == failed["finished_at"]
    assert str(harness.settings.data_root) not in feed3.latest_issue.message
    # Execution failure is not a part verdict: the failed job carries no summary state.
    failed_row = next(j for j in feed3.recent if j.job_id == failed["job_id"])
    assert failed_row.execution_status == "failed" and failed_row.summary_state is None
    assert failed_row.failure_code == "WORKER_TIMEOUT"

    # A completion after the failure supersedes it: the old failure is history.
    harness.completed_job(STATION, serial="SN0044")
    feed4 = StationFeedV1.model_validate(harness.feed().json())
    assert feed4.latest_issue is None


def test_feed_other_station_is_isolated(harness: LineHarness) -> None:
    harness.completed_job(OTHER)
    harness.failed_job(OTHER)
    feed = StationFeedV1.model_validate(harness.feed().json())
    assert feed.latest_completed is None and feed.latest_issue is None and feed.recent == ()


def test_feed_active_job_progress_and_etag(harness: LineHarness) -> None:
    queued = harness.queue_job(STATION)
    r = harness.feed()
    feed = StationFeedV1.model_validate(r.json())
    assert feed.active is not None and feed.active.job.job_id == queued["job_id"]
    assert feed.active.job.execution_status == "queued"
    e0 = r.headers["ETag"]

    harness.start(queued)
    harness.store.update_progress(queued["job_id"], "runner@test", "engine", 41)
    r1 = harness.feed()
    feed1 = StationFeedV1.model_validate(r1.json())
    assert feed1.active is not None
    assert (feed1.active.job.execution_status, feed1.active.job.stage, feed1.active.job.progress) == ("running", "engine", 41)
    e1 = r1.headers["ETag"]
    assert e1 != e0
    # Same 5 % bucket: unchanged ETag (cheap polling); next bucket: changed.
    harness.store.update_progress(queued["job_id"], "runner@test", "engine", 44)
    assert harness.feed(**{"If-None-Match": e1}).status_code == 304
    harness.store.update_progress(queued["job_id"], "runner@test", "engine", 45)
    assert harness.feed(**{"If-None-Match": e1}).status_code == 200


def _tamper_expected(doc: dict[str, Any]) -> None:
    # Shift one expected value and its deviation together: internally valid,
    # but no longer the compiled nominal (EXPECTED_MISMATCH under ENGINE_IO §4.3).
    for row in doc["bends"]:
        for char in row["characteristics"]:
            if char.get("state") == "MEASURED" and char.get("expected") is not None:
                char["expected"] = round(char["expected"] + 0.25, 6)
                char["deviation"] = round(char["actual"] - char["expected"], 6)
                return
    raise AssertionError("example has no MEASURED characteristic")


def test_feed_never_shows_an_older_verdict_when_latest_result_is_bad(harness: LineHarness) -> None:
    harness.completed_job(STATION)
    bad = harness.completed_job(STATION, serial="SN0099", mutate=_tamper_expected)
    try:
        ShopfloorInspectionResultV1.model_validate_json((harness.ws(bad["job_id"]) / svc.RESULT_FILE_REL).read_bytes())
        model_valid = True
    except Exception:
        model_valid = False
    feed = StationFeedV1.model_validate(harness.feed().json())
    assert feed.latest_completed is None, "an unverifiable latest result must not fall back to an older part"
    assert feed.recent[0].job_id == bad["job_id"]
    code = "RESULT_INCONSISTENT" if model_valid else "RESULT_INVALID"
    # The recorded (unverified) verdict is not drawn from recent[0]; the older verified one stays.
    top = feed.recent[0]
    assert (top.summary_state, top.inspection_validity, top.stage_status) == (None, None, None)
    assert top.execution_status == "completed"
    assert feed.recent[1].summary_state == "OUT_OF_TOLERANCE_FOUND"
    # The station is told why there is no result.
    assert feed.latest_issue is not None and feed.latest_issue.kind == "JOB_FAILED"
    assert feed.latest_issue.reason_code == code and feed.latest_issue.at == bad["finished_at"]
    r = harness.client.get(f"/api/line/jobs/{bad['job_id']}/result", headers=h("tok-viewer"))
    assert r.status_code == 500
    assert r.json()["detail"]["code"] == code
    detail = harness.client.get(f"/api/line/jobs/{bad['job_id']}", headers=h("tok-viewer")).json()
    assert detail["result_state"] == code
    assert detail["summary"]["summary_state"] is None and detail["job"]["result_summary"] is None
    # A newer open quarantine is the newer issue.
    harness.scan(STATION, quarantine="UNMATCHED_SKU")
    issue = StationFeedV1.model_validate(harness.feed().json()).latest_issue
    assert issue is not None and issue.kind == "QUARANTINE"


def test_feed_missing_latest_result_is_an_issue(harness: LineHarness) -> None:
    job = harness.completed_job(STATION, write_result=False)
    feed = StationFeedV1.model_validate(harness.feed().json())
    assert feed.latest_completed is None
    assert feed.latest_issue is not None and feed.latest_issue.reason_code == "RESULT_MISSING"
    assert feed.recent[0].job_id == job["job_id"] and feed.recent[0].summary_state is None


# =========================================================================== jobs and results


def test_job_result_states(harness: LineHarness) -> None:
    queued = harness.queue_job(STATION)
    r = harness.client.get(f"/api/line/jobs/{queued['job_id']}/result", headers=h("tok-viewer"))
    assert r.status_code == 409 and r.json()["detail"]["code"] == "JOB_NOT_COMPLETED"
    harness.start(queued)
    harness.write_workspace(queued, result_doc(queued))  # written mid-run: still not served
    assert harness.client.get(f"/api/line/jobs/{queued['job_id']}/result", headers=h("tok-viewer")).status_code == 409
    assert harness.store.complete_job(queued["job_id"], "runner@test", summary_of(result_doc(queued)))

    r = harness.client.get(f"/api/line/jobs/{queued['job_id']}/result", headers=h("tok-viewer"))
    assert r.status_code == 200
    assert r.headers["X-Sherman-Result-Check"] == "model+identity+consistency"
    res = ShopfloorInspectionResultV1.model_validate(r.json())
    assert res.context.job_id == queued["job_id"]
    assert harness.client.get(
        f"/api/line/jobs/{queued['job_id']}/result", headers={**h("tok-viewer"), "If-None-Match": r.headers["ETag"]}
    ).status_code == 304

    missing = harness.completed_job(STATION, serial="SN1", write_result=False)
    r = harness.client.get(f"/api/line/jobs/{missing['job_id']}/result", headers=h("tok-viewer"))
    assert r.status_code == 404 and r.json()["detail"]["code"] == "RESULT_MISSING"

    assert harness.client.get("/api/line/jobs/line_" + "f" * 32 + "/result", headers=h("tok-viewer")).status_code == 404
    assert harness.client.get("/api/line/jobs/not-a-job/result", headers=h("tok-viewer")).status_code == 404


@pytest.mark.parametrize(
    "field_path,value,code",
    [
        (("context", "job_id"), "line_" + "1" * 32, "CONTEXT_JOB_MISMATCH"),
        (("context", "station_id"), "cell-2", "CONTEXT_STATION_MISMATCH"),
        (("scan", "sha256"), "9" * 64, "SCAN_MISMATCH"),
    ],
)
def test_result_identity_mismatch_is_refused(harness: LineHarness, field_path: tuple[str, str], value: str, code: str) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc[field_path[0]][field_path[1]] = value

    job = harness.completed_job(STATION, mutate=mutate)
    r = harness.client.get(f"/api/line/jobs/{job['job_id']}/result", headers=h("tok-viewer"))
    assert r.status_code == 500
    assert r.json()["detail"]["code"] == "RESULT_INCONSISTENT"
    assert code in r.json()["detail"]["message"]


def test_result_summary_must_match_job_row(harness: LineHarness) -> None:
    job = harness.start(harness.queue_job(STATION))
    doc = result_doc(job)
    wrong = dict(summary_of(doc), summary_state="ALL_MEASURED_IN_TOLERANCE")
    harness.write_workspace(job, doc)
    assert harness.store.complete_job(job["job_id"], "runner@test", wrong)
    r = harness.client.get(f"/api/line/jobs/{job['job_id']}/result", headers=h("tok-viewer"))
    assert r.status_code == 500 and "SUMMARY_MISMATCH" in r.json()["detail"]["message"]
    assert StationFeedV1.model_validate(harness.feed().json()).latest_completed is None


def test_result_without_workspace_context_is_identity_checked_only(harness: LineHarness) -> None:
    job = harness.completed_job(STATION, context=False)
    r = harness.client.get(f"/api/line/jobs/{job['job_id']}/result", headers=h("tok-viewer"))
    assert r.status_code == 200 and r.headers["X-Sherman-Result-Check"] == "model+identity"


def test_corrupt_workspace_recipe_blocks_result(harness: LineHarness) -> None:
    job = harness.completed_job(STATION)
    (harness.ws(job["job_id"]) / "recipe.json").write_text("{not json")
    r = harness.client.get(f"/api/line/jobs/{job['job_id']}/result", headers=h("tok-viewer"))
    assert r.status_code == 500 and r.json()["detail"]["code"] == "CONTEXT_INVALID"


def test_job_list_filters_and_cursor(harness: LineHarness) -> None:
    a = harness.completed_job(STATION, serial="A1")
    b = harness.failed_job(STATION)
    c = harness.queue_job(OTHER)
    r = harness.client.get("/api/line/jobs?limit=2", headers=h("tok-viewer"))
    body = r.json()
    assert [j["job_id"] for j in body["jobs"]] == [c["job_id"], b["job_id"]]
    assert body["next_before"] == b["created_at"]
    r = harness.client.get(f"/api/line/jobs?limit=2&before={body['next_before']}", headers=h("tok-viewer"))
    assert [j["job_id"] for j in r.json()["jobs"]] == [a["job_id"]]
    assert r.json()["next_before"] is None
    r = harness.client.get("/api/line/jobs?status=failed,queued", headers=h("tok-viewer"))
    assert {j["job_id"] for j in r.json()["jobs"]} == {b["job_id"], c["job_id"]}
    r = harness.client.get(f"/api/line/jobs?station_id={STATION}&status=completed", headers=h("tok-viewer"))
    assert [j["job_id"] for j in r.json()["jobs"]] == [a["job_id"]]
    assert r.json()["jobs"][0]["summary_state"] == "OUT_OF_TOLERANCE_FOUND"
    assert harness.client.get("/api/line/jobs?status=done", headers=h("tok-viewer")).status_code == 422
    assert harness.client.get("/api/line/jobs?kind=nope", headers=h("tok-viewer")).status_code == 422
    assert harness.client.get("/api/line/jobs?before=yesterday", headers=h("tok-viewer")).status_code == 422


def test_before_cursor_is_normalised_to_store_format(harness: LineHarness) -> None:
    a = harness.queue_job(STATION, serial="A1")
    b = harness.queue_job(STATION, serial="B1")
    # b's created_at spelled in +03:00: as raw text it sorts after every 'Z' value of the same day.
    b_at = datetime.fromisoformat(b["created_at"].replace("Z", "+00:00"))
    offset = b_at.astimezone(timezone(timedelta(hours=3))).isoformat(timespec="milliseconds")
    r = harness.client.get("/api/line/jobs", params={"before": offset}, headers=h("tok-viewer"))
    assert r.status_code == 200 and [j["job_id"] for j in r.json()["jobs"]] == [a["job_id"]]
    assert svc.parse_before("2026-10-17T12:00:00+03:00") == "2026-10-17T09:00:00.000Z"
    assert svc.parse_before("2026-10-17T09:00:00.123456Z") == "2026-10-17T09:00:00.124Z", "sub-ms rounds up"
    assert svc.parse_before("2026-10-17T09:00:00.123Z") == "2026-10-17T09:00:00.123Z"
    for bad in ("2026-10-17", "yesterday", "2026-10-17T25:00:00Z"):
        with pytest.raises(svc.LineApiError):
            svc.parse_before(bad)
    assert harness.client.get("/api/line/jobs?before=2026-10-17", headers=h("tok-viewer")).status_code == 422


def test_job_detail(harness: LineHarness) -> None:
    job = harness.completed_job(STATION)
    r = harness.client.get(f"/api/line/jobs/{job['job_id']}", headers=h("tok-viewer"))
    assert r.status_code == 200
    body = r.json()
    assert body["job"]["job_id"] == job["job_id"] and body["result_state"] == "AVAILABLE"
    assert body["summary"]["scan_file_name"].startswith(SKU)
    assert "report.pdf" in body["artifacts"]


# =========================================================================== retry / cancel


def test_cancel_and_retry_by_operator(harness: LineHarness) -> None:
    queued = harness.queue_job(STATION)
    assert harness.client.post(f"/api/line/jobs/{queued['job_id']}/cancel", headers=h("tok-viewer")).status_code == 403
    r = harness.client.post(f"/api/line/jobs/{queued['job_id']}/cancel", headers=h("tok-operator"))
    assert r.status_code == 200 and r.json() == {"job_id": queued["job_id"], "outcome": "cancelled"}
    assert harness.client.post(f"/api/line/jobs/{queued['job_id']}/cancel", headers=h("tok-operator")).status_code == 409

    r = harness.client.post(f"/api/line/jobs/{queued['job_id']}/retry", headers=h("tok-operator"))
    assert r.status_code == 200
    new_id = r.json()["job_id"]
    assert new_id != queued["job_id"] and r.json()["retry_of"] == queued["job_id"]
    retry_row = harness.store.get_job(new_id)
    assert retry_row is not None and retry_row["retry_of"] == queued["job_id"] and retry_row["execution_status"] == "queued"
    events = harness.store.events_after(0, kinds=["JOB_RETRY_REQUESTED", "JOB_CANCEL_REQUESTED"])
    kinds = {e["kind"]: e for e in events}
    assert kinds["JOB_RETRY_REQUESTED"]["principal"] == {"kind": "user", "id": "2", "role": "operator"}
    assert kinds["JOB_CANCEL_REQUESTED"]["principal"]["id"] == "2"
    # Retrying again while the retry is queued returns the same retry.
    again = harness.client.post(f"/api/line/jobs/{queued['job_id']}/retry", headers=h("tok-operator"))
    assert again.json()["job_id"] == new_id

    # A running job: cancel is a request; a completed job is neither cancellable nor retryable.
    harness.start(retry_row)
    r = harness.client.post(f"/api/line/jobs/{new_id}/cancel", headers=h("tok-admin"))
    assert r.json()["outcome"] == "cancel_requested"
    done = harness.completed_job(STATION, serial="SN7")
    assert harness.client.post(f"/api/line/jobs/{done['job_id']}/retry", headers=h("tok-operator")).status_code == 409
    assert harness.client.post(f"/api/line/jobs/{done['job_id']}/cancel", headers=h("tok-operator")).status_code == 409
    assert harness.client.post("/api/line/jobs/line_" + "a" * 32 + "/retry", headers=h("tok-operator")).status_code == 404


# =========================================================================== quarantine


def test_quarantine_list_assign_and_events(harness: LineHarness) -> None:
    q = harness.scan(STATION, name="9999_WRONG_SKU.ply", quarantine="UNMATCHED_SKU", detail={"resolve_reason": "no recipe for 9999"})
    feed = StationFeedV1.model_validate(harness.feed().json())
    assert feed.latest_issue is not None and feed.latest_issue.kind == "QUARANTINE"
    assert feed.latest_issue.reason_code == "UNMATCHED_SKU" and feed.latest_issue.file_name == "9999_WRONG_SKU.ply"
    assert "no recipe for 9999" in feed.latest_issue.message

    r = harness.client.get("/api/line/quarantine", headers=h("tok-viewer"))
    items = r.json()["items"]
    assert [i["scan_file"]["scan_file_id"] for i in items] == [q["scan_file_id"]]
    assert items[0]["resolvable"] is True

    url = f"/api/line/quarantine/{q['scan_file_id']}/assign"
    assert harness.client.post(url, json={"recipe_id": RECIPE_ID}, headers=h("tok-viewer")).status_code == 403
    r = harness.client.post(url, json={"recipe_id": "NOPE@v1"}, headers=h("tok-operator"))
    assert r.status_code == 404 and r.json()["detail"]["code"] == "RECIPE_NOT_FOUND"
    r = harness.client.post(url, json={"recipe_id": "not a recipe"}, headers=h("tok-operator"))
    assert r.status_code == 422
    harness.provider.add(f"{SKU}@v2", compiled=None)
    r = harness.client.post(url, json={"recipe_id": f"{SKU}@v2"}, headers=h("tok-operator"))
    assert r.status_code == 409 and r.json()["detail"]["code"] == "PART_NOT_READY"
    # v2 is the SKU's active recipe now (released, not compiled): neither v1 nor the SKU is assignable.
    r = harness.client.post(url, json={"recipe_id": RECIPE_ID}, headers=h("tok-operator"))
    assert r.status_code == 409 and r.json()["detail"]["code"] == "RECIPE_NOT_ACTIVE"
    r = harness.client.post(url, json={"sku": SKU}, headers=h("tok-operator"))
    assert r.status_code == 409 and r.json()["detail"]["code"] == "PART_NOT_READY"
    harness.provider.add(f"{SKU}@v3", extra={"compile_status": "BLOCKED"})
    r = harness.client.post(url, json={"recipe_id": f"{SKU}@v3"}, headers=h("tok-operator"))
    assert r.status_code == 409 and r.json()["detail"]["code"] == "PART_NOT_READY"
    del harness.provider.recipes[f"{SKU}@v2"], harness.provider.recipes[f"{SKU}@v3"]

    r = harness.client.post(url, json={"recipe_id": RECIPE_ID}, headers=h("tok-operator"))
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    job = harness.store.get_job(job_id)
    assert job is not None and job["execution_status"] == "queued" and job["compiled_part_sha256"] == COMPILED_SHA
    assert job["scan_file_id"] == q["scan_file_id"] and job["station_id"] == STATION
    ev = harness.store.events_after(0, kinds=["QUARANTINE_ASSIGNED"])
    assert len(ev) == 1 and ev[0]["principal"]["id"] == "2" and ev[0]["job_id"] == job_id
    # Resolved: no longer an open issue; the job is active.
    feed = StationFeedV1.model_validate(harness.feed().json())
    assert feed.latest_issue is None and feed.active is not None and feed.active.job.job_id == job_id
    assert harness.client.get("/api/line/quarantine", headers=h("tok-viewer")).json()["items"] == []
    # Idempotent (station part picker): the same assignment again returns the same job, no new event.
    r = harness.client.post(url, json={"recipe_id": RECIPE_ID}, headers=h("tok-operator"))
    assert r.status_code == 200 and r.json()["job_id"] == job_id and r.json()["assigned_now"] is False
    assert len(harness.store.events_after(0, kinds=["QUARANTINE_ASSIGNED"])) == 1


def test_quarantine_content_reason_is_not_assignable_and_dismiss(harness: LineHarness) -> None:
    q = harness.scan(STATION, quarantine="PLY_TRUNCATED", detail={"detail": "final vertex record is incomplete"})
    url = f"/api/line/quarantine/{q['scan_file_id']}"
    r = harness.client.post(url + "/assign", json={"recipe_id": RECIPE_ID}, headers=h("tok-operator"))
    assert r.status_code == 409 and r.json()["detail"]["code"] == "NOT_RESOLVABLE"
    assert harness.client.post(url + "/dismiss", headers=h("tok-viewer")).status_code == 403
    r = harness.client.post(url + "/dismiss", json={"note": "re-exported"}, headers=h("tok-operator"))
    assert r.status_code == 200 and r.json()["ingest_state"] == "dismissed"
    assert harness.client.post(url + "/dismiss", headers=h("tok-operator")).status_code == 200  # idempotent
    assert harness.client.post("/api/line/quarantine/scan_" + "0" * 24 + "/dismiss", headers=h("tok-operator")).status_code == 404
    ev = harness.store.events_after(0, kinds=["QUARANTINE_DISMISSED"])
    assert ev[0]["payload"]["note"] == "re-exported"


def test_quarantine_device_scope(harness: LineHarness) -> None:
    harness.scan(OTHER, quarantine="UNMATCHED_SKU")
    mine = harness.scan(STATION, quarantine="AMBIGUOUS_SKU")
    _, device = harness.device_token(STATION)
    assert harness.client.get("/api/line/quarantine", headers=h(device)).status_code == 403
    r = harness.client.get(f"/api/line/quarantine?station_id={STATION}", headers=h(device))
    assert [i["scan_file"]["scan_file_id"] for i in r.json()["items"]] == [mine["scan_file_id"]]


def test_quarantine_newer_than_failure_wins_issue(harness: LineHarness) -> None:
    harness.failed_job(STATION)
    harness.scan(STATION, quarantine="UNPARSEABLE_FILENAME")
    issue = StationFeedV1.model_validate(harness.feed().json()).latest_issue
    assert issue is not None and issue.kind == "QUARANTINE" and issue.reason_code == "UNPARSEABLE_FILENAME"
    harness.failed_job(STATION, code="POSTPROCESS_ERROR")
    issue = StationFeedV1.model_validate(harness.feed().json()).latest_issue
    assert issue is not None and issue.kind == "JOB_FAILED" and issue.reason_code == "POSTPROCESS_ERROR"


def test_assign_refuses_retired_and_superseded_and_assigns_by_sku(harness: LineHarness) -> None:
    q = harness.scan(STATION, name="weird name.ply", quarantine="UNMATCHED_SKU")
    url = f"/api/line/quarantine/{q['scan_file_id']}/assign"
    op = h("tok-operator")
    # A retired version, compiled: refused (the hotfolder treats retired-only SKUs as PART_NOT_READY).
    harness.provider.add(f"{SKU}@v7", extra={"compile_status": "COMPILED", "status": "retired"})
    r = harness.client.post(url, json={"recipe_id": f"{SKU}@v7"}, headers=op)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "RECIPE_RETIRED"
    # A superseded but still released version: refused, naming the active one.
    harness.provider.add(f"{SKU}@v2", extra={"compile_status": "COMPILED", "status": "released"})
    r = harness.client.post(url, json={"recipe_id": RECIPE_ID}, headers=op)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "RECIPE_NOT_ACTIVE"
    assert f"{SKU}@v2" in r.json()["detail"]["message"]
    # Body shape: exactly one target, canonical SKU.
    for body in ({}, {"sku": SKU, "recipe_id": RECIPE_ID}):
        r = harness.client.post(url, json=body, headers=op)
        assert r.status_code == 422 and r.json()["detail"]["code"] == "ASSIGN_TARGET_INVALID"
    r = harness.client.post(url, json={"sku": "bad sku"}, headers=op)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "SKU_INVALID"
    r = harness.client.post(url, json={"sku": "UNKNOWN-1"}, headers=op)
    assert r.status_code == 404 and r.json()["detail"]["code"] == "SKU_NOT_FOUND"
    harness.provider.registered.add("REGISTERED-ONLY")
    r = harness.client.post(url, json={"sku": "REGISTERED-ONLY"}, headers=op)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "PART_NOT_READY"
    # Assign by SKU: the server picks the active version.
    r = harness.client.post(url, json={"sku": SKU}, headers=op)
    assert r.status_code == 200, r.text
    assert r.json()["recipe_id"] == f"{SKU}@v2" and r.json()["sku"] == SKU and r.json()["recipe_version"] == 2
    job = harness.store.get_job(r.json()["job_id"])
    assert job is not None and job["recipe_id"] == f"{SKU}@v2" and job["execution_status"] == "queued"


def test_assign_requires_a_catalog_and_a_compiled_active_ref(harness: LineHarness) -> None:
    q = harness.scan(STATION, quarantine="UNMATCHED_SKU")
    url = f"/api/line/quarantine/{q['scan_file_id']}/assign"

    class LoaderOnly:
        def load(self, recipe_id: str) -> ResolvedRecipe:
            return harness.provider.load(recipe_id)

    harness.app.dependency_overrides[svc.get_recipe_provider] = lambda: LoaderOnly()
    r = harness.client.post(url, json={"recipe_id": RECIPE_ID}, headers=h("tok-operator"))
    assert r.status_code == 503 and r.json()["detail"]["code"] == "RECIPE_CATALOG_UNAVAILABLE"

    class StaleCatalog(FakeProvider):
        def list_active_recipes(self) -> list[ActiveRecipeRef]:
            # The catalog says the active recipe is not compiled although load() returns a hash.
            return [ActiveRecipeRef(SKU, RECIPE_ID, 1, RECIPE_SHA, None, None, "PENDING")]

    stale = StaleCatalog()
    stale.add(RECIPE_ID)
    harness.app.dependency_overrides[svc.get_recipe_provider] = lambda: stale
    r = harness.client.post(url, json={"recipe_id": RECIPE_ID}, headers=h("tok-operator"))
    assert r.status_code == 409 and r.json()["detail"]["code"] == "PART_NOT_READY"
    row = harness.store.get_scan_file(q["scan_file_id"])
    assert row is not None and row["ingest_state"] == "quarantined"


def test_store_text_is_redacted_for_line_read(harness: LineHarness) -> None:
    root = str(harness.settings.data_root)
    secret = "Bearer abcdefSECRET123"
    failed = harness.failed_job(STATION, message=f"crash in {root}/jobs/x/engine {secret}")
    harness.store.append_event(
        "SCAN_QUARANTINED", station_id=STATION, payload={"detail": f"probe failed at {root}/inbox/a.ply", "nested": [secret]}
    )
    harness.scan(STATION, quarantine="UNMATCHED_SKU", detail={"resolve_reason": f"{root}/x {secret}"})
    _, device = harness.device_token(STATION)
    for auth in (device, "tok-viewer"):
        bodies = [
            harness.client.get(f"/api/line/jobs/{failed['job_id']}", headers=h(auth)).text,
            harness.client.get(f"/api/line/events?station_id={STATION}", headers=h(auth)).text,
            harness.client.get(f"/api/line/quarantine?station_id={STATION}", headers=h(auth)).text,
        ]
        for text in bodies:
            assert root not in text and "SECRET123" not in text, text[:300]
            assert "<data_root>" in text, "the message is still there, only redacted"
    detail = harness.client.get(f"/api/line/jobs/{failed['job_id']}", headers=h(device)).json()
    assert detail["job"]["failure"]["code"] == "WORKER_TIMEOUT"
    assert detail["job"]["failure"]["message"].startswith("crash in <data_root>/jobs/x/engine")
    stored = harness.store.get_job(failed["job_id"])
    assert stored is not None and stored["failure"]["message"].startswith(f"crash in {root}"), "store unchanged"


def test_redact_tree_handles_both_data_root_spellings(harness: LineHarness) -> None:
    import os as _os

    real = _os.path.realpath(str(harness.settings.data_root))
    out = svc.redact_tree({"a": [f"{real}/jobs", 3, None], "b": {"c": "token=abc"}}, harness.settings)
    assert out == {"a": ["<data_root>/jobs", 3, None], "b": {"c": out["b"]["c"]}}
    assert "abc" not in out["b"]["c"]


# =========================================================================== events JSON


def test_events_json_station_filter_and_device_view(harness: LineHarness) -> None:
    job = harness.queue_job(STATION)
    harness.client.post(f"/api/line/jobs/{job['job_id']}/cancel", headers=h("tok-operator"))
    harness.queue_job(OTHER)
    r = harness.client.get(f"/api/line/events?station_id={STATION}", headers=h("tok-viewer"))
    events = r.json()["events"]
    assert events and all(e["station_id"] == STATION for e in events)
    assert any(e["principal"] for e in events), "users see who acted"
    assert r.json()["last_id"] == events[-1]["id"]
    _, device = harness.device_token(STATION)
    r = harness.client.get(f"/api/line/events?station_id={STATION}&include_global=true", headers=h(device))
    devents = r.json()["events"]
    assert devents and all(e["principal"] is None for e in devents)
    assert all(e["station_id"] == STATION for e in devents)
    r = harness.client.get(f"/api/line/events?after_id={events[-1]['id']}&station_id={STATION}", headers=h("tok-viewer"))
    assert r.json() == {"events": [], "last_id": events[-1]["id"], "reset": False}


def test_events_json_cursor_ahead_of_store_is_reset(harness: LineHarness) -> None:
    harness.queue_job(STATION)
    newest = harness.store.last_event_id()
    r = harness.client.get(f"/api/line/events?after_id={newest + 5000}&station_id={STATION}", headers=h("tok-viewer"))
    assert r.json() == {"events": [], "last_id": newest, "reset": True}
    nxt = harness.store.append_event("JOB_QUEUED", station_id=STATION, payload={})
    r = harness.client.get(f"/api/line/events?after_id={r.json()['last_id']}&station_id={STATION}", headers=h("tok-viewer"))
    assert [e["id"] for e in r.json()["events"]] == [nxt] and r.json()["reset"] is False


# =========================================================================== devices


def test_device_issue_list_revoke(harness: LineHarness) -> None:
    body = {"station_id": STATION, "label": "press 7 kiosk"}
    assert harness.client.post("/api/line/devices", json=body, headers=h("tok-operator")).status_code == 403
    r = harness.client.post("/api/line/devices", json=body, headers=h("tok-admin"))
    assert r.status_code == 201 and r.headers["Cache-Control"] == "no-store"
    token_id, token = r.json()["token_id"], r.json()["token"]
    assert token.startswith("sdt_")
    assert harness.feed(token).status_code == 200

    listing = harness.client.get("/api/line/devices", headers=h("tok-admin"))
    assert token not in listing.text and "token_sha256" not in listing.text
    assert [d["token_id"] for d in listing.json()["devices"]] == [token_id]

    issued = harness.store.events_after(0, kinds=["DEVICE_TOKEN_ISSUED"])
    assert issued[0]["payload"] == {"token_id": token_id, "label": "press 7 kiosk"}
    assert token not in json.dumps(issued)

    assert harness.client.delete(f"/api/line/devices/{token_id}", headers=h("tok-operator")).status_code == 403
    assert harness.client.delete(f"/api/line/devices/{token_id}", headers=h("tok-admin")).status_code == 200
    assert harness.client.delete(f"/api/line/devices/{token_id}", headers=h("tok-admin")).status_code == 404
    assert harness.feed(token).status_code == 401
    revoked = harness.store.events_after(0, kinds=["DEVICE_TOKEN_REVOKED"])
    assert revoked[0]["station_id"] == STATION and revoked[0]["principal"]["id"] == "1"

    assert harness.client.post("/api/line/devices", json={"station_id": "cell-9", "label": "x"}, headers=h("tok-admin")).status_code == 404
    assert harness.client.post("/api/line/devices", json={"station_id": STATION, "label": "   "}, headers=h("tok-admin")).status_code == 422


# =========================================================================== readiness and status


def _no_paths(body: Any, settings: LineSettings) -> None:
    text = json.dumps(body)
    assert str(settings.data_root) not in text and "/" not in text.replace("hotfolder:", "")


def test_readiness_stale_and_fresh(harness: LineHarness) -> None:
    stale_after = harness.config.stale_after_s
    beat(harness, "runner", age_s=stale_after + 30)
    for sid in (STATION, OTHER):
        beat(harness, f"hotfolder:{sid}")
    r = harness.client.get("/api/line/health/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False and body["checks"]["runner"] == "fail"
    assert body["checks"]["db"] == "ok" and body["checks"][f"hotfolder:{STATION}"] == "ok"
    assert set(body) == {"ready", "checks"}
    _no_paths(body, harness.settings)

    beat(harness, "runner")
    r = harness.client.get("/api/line/health/ready")
    assert r.status_code == 200, r.json()
    assert r.json() == {
        "ready": True,
        "checks": {"db": "ok", "disk": "ok", "runner": "ok", f"hotfolder:{STATION}": "ok", f"hotfolder:{OTHER}": "ok"},
    }
    _no_paths(r.json(), harness.settings)


def test_readiness_is_cached_for_its_ttl(harness: LineHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = svc.LineApiConfig(min_free_bytes=0, readiness_ttl_s=2.0)
    calls = {"n": 0}
    real = svc.health.sqlite_writable

    def counting(*a: Any, **k: Any) -> health.ProbeResult:
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(svc.health, "sqlite_writable", counting)
    now = [100.0]
    beat(harness, "runner", age_s=999)
    for sid in (STATION, OTHER):
        beat(harness, f"hotfolder:{sid}")
    assert svc.cached_readiness(harness.settings, cfg, clock=lambda: now[0])[0] is False
    beat(harness, "runner")
    now[0] = 101.9
    for _ in range(20):
        assert svc.cached_readiness(harness.settings, cfg, clock=lambda: now[0])[0] is False
    assert calls["n"] == 1, "one DB write probe per TTL, however often the public route is hit"
    now[0] = 102.1
    ready, checks = svc.cached_readiness(harness.settings, cfg, clock=lambda: now[0])
    assert ready is True and checks["runner"] == "ok" and calls["n"] == 2
    # A caller mutating the returned dict cannot poison the cache.
    checks["db"] = "fail"
    assert svc.cached_readiness(harness.settings, cfg, clock=lambda: now[0])[1]["db"] == "ok"
    # The route uses the cache too.
    harness.app.dependency_overrides[svc.get_line_api_config] = lambda: cfg
    before = calls["n"]
    for _ in range(5):
        assert harness.client.get("/api/line/health/ready").status_code == 200
    assert calls["n"] - before == 1


def test_readiness_missing_hotfolder_and_disk_threshold(tmp_path: Path) -> None:
    hs = build_harness(tmp_path, svc.LineApiConfig(min_free_bytes=1 << 62))
    try:
        beat(hs, "runner")
        beat(hs, f"hotfolder:{STATION}")
        body = hs.client.get("/api/line/health/ready").json()
        assert body["checks"]["disk"] == "fail"
        assert body["checks"][f"hotfolder:{OTHER}"] == "fail"
        assert body["checks"][f"hotfolder:{STATION}"] == "ok"
    finally:
        hs.client.close()


def test_readiness_db_missing_is_not_created(tmp_path: Path) -> None:
    hs = build_harness(tmp_path)
    try:
        hs.client.close()
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(hs.settings.paths.db_line) + suffix)
            if p.exists():
                p.unlink()
        svc.reset_line_api_caches()
        ready, checks = svc.readiness_checks(hs.settings, hs.config)
        assert ready is False and checks["db"] == "fail" and checks["runner"] == "fail"
        assert not hs.settings.paths.db_line.exists()
    finally:
        svc.reset_line_api_caches()


def test_status_report_and_probe_cache(harness: LineHarness) -> None:
    beat(harness, "runner")
    beat(harness, f"hotfolder:{STATION}", age_s=999)
    done = harness.completed_job(STATION)
    running = harness.start(harness.queue_job(STATION))
    harness.scan(STATION, quarantine="UNMATCHED_SKU")
    assert harness.client.get("/api/line/status", headers=h("tok-viewer")).status_code == 403
    r = harness.client.get("/api/line/status", headers=h("tok-operator"))
    assert r.status_code == 200
    body = r.json()
    comps = {b["component"]: b for b in body["heartbeats"]}
    assert comps["runner"]["stale"] is False and comps[f"hotfolder:{STATION}"]["stale"] is True
    assert body["missing_heartbeats"] == [f"hotfolder:{OTHER}"]
    assert body["queue"]["running"] == 1 and body["queue"]["open_quarantine"] == 1
    assert [j["job_id"] for j in body["running"]] == [running["job_id"]]
    assert body["last_completion"]["job_id"] == done["job_id"]
    assert body["last_ingest"]["ingest_state"] == "quarantined"
    assert body["open_quarantine"]["count"] == 1
    assert body["engine_bundle"]["ok"] is False and body["python"]["ok"] is True
    assert body["ready"] is False and body["checks"]["runner"] == "ok"
    harness.client.get("/api/line/status", headers=h("tok-admin"))
    assert harness.probe_calls == {"python": 1, "bundle": 1}, "probes are cached for probe_ttl_s"


def test_status_probes_expire(tmp_path: Path) -> None:
    hs = build_harness(tmp_path)
    try:
        now = [0.0]
        calls = {"n": 0}

        def probe() -> health.ProbeResult:
            calls["n"] += 1
            return health.ProbeResult("p", True)

        probes = svc.StatusProbes(hs.settings, svc.LineApiConfig(probe_ttl_s=60), python_probe=probe, bundle_probe=probe, clock=lambda: now[0])
        probes.python()
        now[0] = 59.0
        probes.python()
        assert calls["n"] == 1
        now[0] = 61.0
        probes.python()
        assert calls["n"] == 2

        def boom() -> health.ProbeResult:
            raise RuntimeError("x")

        res, _ = svc.StatusProbes(hs.settings, hs.config, python_probe=boom, bundle_probe=boom).python()
        assert res.ok is False
    finally:
        hs.client.close()


# =========================================================================== metrics


def test_metrics_text(harness: LineHarness) -> None:
    harness.queue_job(STATION)
    beat(harness, "runner")
    assert harness.client.get("/api/line/metrics", headers=h("tok-operator")).status_code == 403
    r = harness.client.get("/api/line/metrics", headers=h("tok-admin"))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain; version=0.0.4")
    text = r.text
    assert "sherman_line_queue_depth 1" in text
    assert 'sherman_line_jobs{status="queued"} 1' in text
    assert 'sherman_line_heartbeat_age_seconds{component="runner"}' in text


# =========================================================================== small helpers


def test_if_none_match_parsing() -> None:
    assert svc.if_none_match_matches('"a"', '"a"')
    assert svc.if_none_match_matches('W/"a"', '"a"')
    assert svc.if_none_match_matches('"b", "a"', '"a"')
    assert svc.if_none_match_matches("*", '"a"')
    assert not svc.if_none_match_matches('"b"', '"a"')
    assert not svc.if_none_match_matches(None, '"a"')
    assert not svc.if_none_match_matches("", '"a"')


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        svc.LineApiConfig(stream_token_ttl_s=301)
    with pytest.raises(ValueError):
        svc.LineApiConfig(sse_poll_s=0)
    with pytest.raises(ValueError):
        svc.LineApiConfig(stale_factor=0.5)
    with pytest.raises(ValueError):
        svc.LineApiConfig(readiness_ttl_s=-1)


def test_store_timestamps_are_contract_timestamps(harness: LineHarness) -> None:
    job = harness.completed_job(STATION)
    assert job["finished_at"].endswith("Z") and iso_utc(datetime.now(UTC)).endswith("Z")
