"""Engine end-to-end test of the production line (C11): the REAL E1 compile and the REAL
E3 engine, run from a pinned engine bundle. Gated: ``SHERMAN_E2E_ENGINE=1``.

One heavy run, serial (one job at a time, ``OMP_NUM_THREADS=1``)::

    SHERMAN_E2E_ENGINE=1 PYT tests/test_prod_e2e_engine.py -q -s

1. An engine bundle is built from the repo into ``tmp_path`` (C04 ``--profile
   engine``) and verified, so no step can read the repo's ``output/`` or ``data/``.
2. An E2 part set (HAT, DEV seeds, a real STEP BRep) is onboarded into a tmp
   catalog and released through the C05 router; the release compile (E1) runs
   from the bundle.
3. Three scans of that part go through the real hotfolder and the real runner:
   ``nominal``; ``errors`` (+1.5 mm on a free flange of B2, +2 deg on B3); and
   ``stage_k``, an incomplete part under the FINAL recipe. Each runs the real
   E3 (``python -m domains.production.engine.inspect``, 180 s / 4 GiB hard
   stop) and the real C06 render.
4. Asserted for every job: completion (or a documented execution failure
   code, which fails the test with that code), the result validates against
   the contract and against the compiled part, rows are in sequence order,
   validity and summary agree with ``derive_summary``, and no NOT_YET_FORMED
   bend is FAIL. Engine-specific expectations (plan §10 amendment 10): both
   injected errors are MEASURED and FAIL or REVIEW, with actuals within 0.2 mm /
   0.3 deg of truth (an UNKNOWN there fails the test: it would let an engine
   that measures nothing pass); the nominal scan is ALL_MEASURED_IN_TOLERANCE
   with every required characteristic measured; the incomplete scan has
   NOT_YET_FORMED bends matching truth and is never green.

Printed: per job ``duration_s`` and ``engine_peak_rss_bytes`` (the engine step's
process-group peak RSS from the C02 supervisor), the whole run's wall time, and
``test_process_max_rss_bytes`` (this pytest process's own peak, from getrusage,
which includes E2 generation and the in-process API/runner; it is the larger
number). Truth (``truth.json``, ``part_set.json``) is read by this test only,
never by the engine.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
STATION = "cell-1"

pytestmark = pytest.mark.skipif(os.environ.get("SHERMAN_E2E_ENGINE") != "1",
                                reason="engine E2E is gated: set SHERMAN_E2E_ENGINE=1 (one heavy run)")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.test_prod_e2e_fast import StubVerifier, drop_scan, h  # noqa: E402

#: Documented execution failure codes (C02 runner, ENGINE_IO §5). Any of them fails the test,
#: naming the code, instead of passing silently.
EXECUTION_FAILURE_CODES = frozenset({"WORKER_TIMEOUT", "WORKER_RSS_LIMIT", "WORKER_ERROR", "WORKER_CRASH",
                                     "WORKER_NO_RESULT", "ENGINE_OUTPUT_INVALID", "POSTPROCESS_ERROR",
                                     "POSTPROCESS_TIMEOUT"})


def _scan(part_set: dict[str, Any], name: str) -> dict[str, Any]:
    return next(s for s in part_set["scans"] if s["name"] == name)


def test_real_engine_end_to_end(tmp_path: Path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from part_catalog import PartCatalogManager

    import apps.api.routes.production_line as line_routes
    import apps.api.routes.production_recipes as recipe_routes
    from apps.production_line.engine_bundle import build_bundle, verify
    from apps.production_line.recipes.catalog_port import CatalogManagerPort
    from apps.production_line.recipes.synthetic_onboarding import onboard_case, release_request, vendor_setup_loader
    from apps.production_line.security import dependencies as deps
    from apps.production_line.server import build_asgi
    from apps.production_line.settings import ensure_layout, load_settings
    from apps.production_line.wiring import build_runtime
    from contracts.production.compiled_part import CompiledPartV1
    from contracts.production.consistency import check_result_against_compiled
    from contracts.production.inspection_result import ShopfloorInspectionResultV1, derive_summary
    from domains.production.engine.synthetic.cases import generate_part_set

    run_t0 = time.monotonic()
    bundle = tmp_path / "engine_bundle"
    built = build_bundle(REPO, bundle, profile="engine")
    assert verify(bundle) == [], "freshly built engine bundle does not verify"
    # Native scanner spacing (about 270k points for this part). Note: a 60k-point override makes E3
    # call the incomplete scan WRONG_PART_SUSPECTED (reported to E3 / A1).
    meta = generate_part_set("HAT", "DEV", tmp_path / "e2", ray_threads=1)
    set_dir = tmp_path / "e2" / meta["set_id"]
    part_set = json.loads((set_dir / "part_set.json").read_text(encoding="utf-8"))
    sku = part_set["sku"]
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    settings = load_settings({
        "SHERMAN_LINE_ENABLED": "1",
        "SHERMAN_LINE_DATA_ROOT": str(tmp_path / "line"),
        "SHERMAN_LINE_STATION_ID": STATION,
        "SHERMAN_LINE_INBOX": str(inbox),
        "SHERMAN_LINE_ENGINE_BUNDLE_DIR": str(bundle),
    }, repo_root=REPO)
    ensure_layout(settings.paths)
    catalog = PartCatalogManager(str(tmp_path / "catalog.sqlite"))
    ob = onboard_case(catalog, set_dir, part_number=sku, upload_dir=tmp_path / "uploads",
                      revision=part_set["revision"], part_name=part_set["part_name"])
    port = CatalogManagerPort(catalog, vendor_setup_loader=vendor_setup_loader(catalog))
    runtime = build_runtime(
        settings, workers=True, catalog=port, instance_id="runner@e2e-engine",
        hotfolder_overrides={"poll_interval_s": 0.05, "stable_polls": 1, "min_age_s": 0.0, "min_size_bytes": 0},
    )
    assert runtime.runner.config.engine_root == bundle
    assert runtime.runner.config.inspect_module == "domains.production.engine.inspect"
    assert runtime.runner.config.engine_bundle_id == built.bundle_id
    app = FastAPI()
    app.include_router(recipe_routes.router)
    app.include_router(line_routes.router)
    asgi = build_asgi(app, settings, runtime)
    app.dependency_overrides[deps.get_user_token_verifier] = lambda: StubVerifier()
    client = TestClient(asgi)
    try:
        t = time.monotonic()
        r = client.post(f"/api/production/parts/{ob.part_id}/recipe/release", json=release_request(ob), headers=h())
        assert r.status_code == 201, r.text
        release_s = time.monotonic() - t
        stored = runtime.recipe_store.get(r.json()["recipe_id"])
        recipe = stored.recipe()
        resolved = runtime.provider.load(r.json()["recipe_id"])
        compiled = CompiledPartV1.model_validate_json((Path(resolved.compiled_dir) / "compiled_part.json").read_bytes())
        print(f"\n[e2e-engine] bundle {built.bundle_id}; release (E1 compile) {release_s:.1f} s")

        report: list[dict[str, Any]] = []
        for name in ("nominal", "errors", "stage_k"):
            entry = _scan(part_set, name)
            drop_scan(inbox, set_dir / entry["dir"] / "scan.ply", entry["zeiss_filename"])
            job_id = None
            for _ in range(40):
                for outcome in runtime.pollers[0].poll_once():
                    if outcome.action == "enqueued":
                        job_id = outcome.job_id
                if job_id:
                    break
                time.sleep(0.02)
            assert job_id, f"{name}: the scan was not enqueued"
            assert runtime.runner.run_once() == job_id
            job = runtime.line_store.get_job(job_id)
            print(f"[e2e-engine] {name}: status={job['execution_status']} duration_s={job.get('duration_s')} "
                  f"peak_rss_bytes={job.get('peak_rss_bytes')}")
            failure = job.get("failure") or {}
            code = failure.get("code")
            kind = "documented execution failure" if code in EXECUTION_FAILURE_CODES else "UNDOCUMENTED failure code"
            assert job["execution_status"] == "completed", f"{name}: {kind} {code}: {failure.get('message')}"
            res = client.get(f"/api/line/jobs/{job_id}/result", headers=h("tok-viewer"))
            assert res.status_code == 200, res.text
            result = ShopfloorInspectionResultV1.model_validate(res.json())
            assert check_result_against_compiled(result, compiled, recipe) == []
            assert [b.sequence_index for b in result.bends] == list(range(1, len(result.bends) + 1))
            assert result.summary == derive_summary(
                validity_state=result.inspection_validity.state, bends=result.bends,
                general_measurements=result.general_measurements, stage_decision=result.stage_decision.decision,
                stage_status=result.stage_decision.status, unplanned_creases=len(result.unplanned_creases))
            if result.inspection_validity.state != "VALID":
                assert result.summary.state == "INVALID_INSPECTION"
            for row in result.bends:
                if row.state == "NOT_YET_FORMED":
                    assert all(c.decision != "FAIL" for c in row.characteristics), (name, row.label)
                for c in row.characteristics:
                    if c.state != "MEASURED":
                        assert c.actual is None and c.decision in ("UNKNOWN", "NOT_APPLICABLE"), (name, c.characteristic_id)
            report.append({"name": name, "job": job, "result": result, "entry": entry})

        by = {r["name"]: r for r in report}
        nominal = by["nominal"]["result"]
        ns = nominal.summary
        print(f"[e2e-engine] nominal: summary={ns.state} required={ns.characteristics_required} "
              f"pass={ns.required_pass} unknown={ns.required_unknown} review={ns.required_review}")
        assert all(c.decision != "FAIL" for b in nominal.bends for c in b.characteristics), "FAIL on the nominal scan"
        # An all-UNKNOWN engine must not pass: the nominal part is measured and in tolerance.
        assert ns.required_pass > 0 and ns.required_unknown == 0, ns
        assert ns.state == "ALL_MEASURED_IN_TOLERANCE", ns

        errors = by["errors"]["result"]
        injected = by["errors"]["entry"]["expected"]["injected"]
        never_pass = set(by["errors"]["entry"]["expected"]["characteristics_never_pass"])
        assert errors.summary.state in ("OUT_OF_TOLERANCE_FOUND", "REVIEW_REQUIRED"), errors.summary.state
        seen: set[str] = set()
        for b in errors.bends:
            for c in b.characteristics:
                key = f"{c.bend_uid}/{c.kind}" + (f"/{c.side}" if c.side else "")
                if key in never_pass:
                    seen.add(key)
                    assert c.decision != "PASS", f"false PASS on injected {key}"
                    # Amendment 10: the injected error must be detected, not merely not passed.
                    assert c.state == "MEASURED", f"injected {key} not measured: {c.state} {c.reason}"
                    assert c.decision in ("FAIL", "REVIEW"), (key, c.decision)
                if key in injected and c.state == "MEASURED":
                    err = abs(float(c.actual) - injected[key]["truth"])
                    print(f"[e2e-engine] errors {key}: actual={c.actual:.3f} truth={injected[key]['truth']:.3f} "
                          f"|err|={err:.3f} decision={c.decision}")
                    assert err <= (0.2 if c.unit == "mm" else 0.3), key
        assert seen == never_pass and len(seen) >= 2, f"injected characteristics missing from the result: {never_pass - seen}"

        staged = by["stage_k"]["result"]
        truth_states = by["stage_k"]["entry"]["expected"]["bend_states_contract"]
        got_states = {b.bend_uid: b.state for b in staged.bends}
        print(f"[e2e-engine] stage_k: validity={staged.inspection_validity.state} "
              f"stage={staged.stage_decision.status} summary={staged.summary.state}")
        for b in staged.bends:
            print(f"[e2e-engine] stage_k {b.label} {b.bend_uid}: engine={b.state} truth={truth_states.get(b.bend_uid)}")
        assert "NOT_YET_FORMED" in truth_states.values()
        assert staged.inspection_validity.state == "VALID", staged.inspection_validity.reasons
        for uid, truth_state in truth_states.items():
            # An engine UNKNOWN is honest; a confident wrong state is not.
            assert got_states[uid] in (truth_state, "UNKNOWN"), (uid, got_states[uid], truth_state)
        assert staged.stage_decision.status == "FINAL_INCOMPLETE"
        assert staged.summary.state not in ("ALL_MEASURED_IN_TOLERANCE", "STAGE_IN_TOLERANCE")
        peak = max(int(r["job"].get("peak_rss_bytes") or 0) for r in report)
        # ru_maxrss is bytes on macOS, KiB on Linux.
        own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        print(f"[e2e-engine] duration_s={time.monotonic() - run_t0:.1f} engine_peak_rss_bytes={peak} "
              f"test_process_max_rss_bytes={own}")
        assert 0 < peak < 4 * 1024**3
        assert own < 4 * 1024**3
    finally:
        client.close()
        runtime.close()

