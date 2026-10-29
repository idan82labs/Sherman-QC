"""Station part picker (held-back scans): list, suggestions, released-part search,
idempotent and race-safe assign / dismiss, audit events, and that an
operator-assigned job still goes through every E3 identity check.

Backend only. The API tests reuse the C07 harness of
``test_prod_line_routes.py`` (minimal FastAPI app with only the line router,
tmp data root, real ``LineStore``, fake recipe provider, C01 example JSON).
The runner test reuses the C02 fake-engine bundle of ``test_prod_runner.py``
(fake ``fake_inspect`` / ``fake_render`` modules speaking the step protocol,
the SYN-HAT-1 examples re-finalised around synthetic CAD), so the runner's real
contract validation and ``check_result_against_compiled`` run. No legacy server
import, no real scans or CAD.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import textwrap
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from apps.api.services import part_picker_service as picker
from apps.api.services import production_line_service as svc
from apps.production_line.ingest.ports import ActiveRecipeRef, ResolvedRecipe
from apps.production_line.security import dependencies as deps
from apps.production_line.security.principals import Principal
from apps.production_line.store.line_store import LineStore
from contracts.production.canonical import canonical_json_bytes
from tests.test_prod_line_routes import (
    COMPILED_SHA,
    OTHER,
    RECIPE_DOC,
    RECIPE_ID,
    SKU,
    STATION,
    LineHarness,
    build_harness,
    h,
)

OPERATOR = Principal.for_user(2, "oscar", "operator")
OPERATOR_2 = Principal.for_user(4, "olga", "operator")


@pytest.fixture()
def hs(tmp_path: Path) -> Iterator[LineHarness]:
    harness = build_harness(tmp_path)
    # The example recipe bytes (4 bends, IN_PROCESS, revision A) so stage / facts are real.
    r = harness.provider.recipes[RECIPE_ID]
    harness.provider.recipes[RECIPE_ID] = dataclasses.replace(
        r, recipe_json_bytes=canonical_json_bytes(RECIPE_DOC), part_name="Synthetic hat section", part_revision="A"
    )
    picker.reset_picker_caches()
    yield harness
    harness.client.close()
    deps.reset_line_settings_cache()
    svc.reset_line_api_caches()
    picker.reset_picker_caches()


def held(hs: LineHarness, token: str = "tok-operator", station: str = STATION) -> Any:
    return hs.client.get(f"/api/line/quarantine/held?station_id={station}", headers=h(token))


def quarantined(hs: LineHarness, name: str, reason: str, parse: dict[str, Any], match: dict[str, Any]) -> dict[str, Any]:
    sha = hs._sha()
    row, created = hs.store.register_scan(
        sha256=sha,
        original_name=name,
        size_bytes=4321,
        station_id=STATION,
        stored_rel_path=f"archive/scans/2026/10/{sha}.ply",
        parse=parse,
        match=match,
        format="PLY_BINARY_LE",
        vertex_count=120000,
        face_count=0,
        bbox_mm=((0.0, 0.0, 0.0), (120.0, 80.0, 40.0)),
        ingest_state="quarantined",
        quarantine_reason=reason,
        quarantine_detail={"resolve_reason": match.get("reason")},
    )
    assert created
    return row


def _parse(status: str, sku: str | None = None, **kw: Any) -> dict[str, Any]:
    out = {"status": status, "sku": sku, "serial": None, "candidates": [], "revision": None, "expected_revision": None,
           "revision_status": None, "scan_label": None}  # fmt: skip
    out.update(kw)
    return out


# =========================================================================== list + suggestions


def test_held_scans_list_carries_outcome_tokens_stats_and_suggestions(hs: LineHarness) -> None:
    hs.provider.add("SYN-HAT-2@v1")
    unmatched = quarantined(hs, "SYN-HAT1_A_FULL_SCAN 1.ply", "UNMATCHED_SKU", _parse("UNMATCHED"),
                            {"status": "UNMATCHED", "reason": "NO_KNOWN_SKU_TOKEN"})  # fmt: skip
    ambiguous = quarantined(hs, "SYN-HAT-1 SYN-HAT-2 scan.ply", "AMBIGUOUS_SKU",
                            _parse("AMBIGUOUS", candidates=["SYN-HAT-1", "SYN-HAT-2"]),
                            {"status": "AMBIGUOUS", "reason": "MULTIPLE_KNOWN_SKUS: SYN-HAT-1, SYN-HAT-2"})  # fmt: skip
    revision = quarantined(hs, "SYN-HAT-1_C_FULL_SCAN 1.ply", "REVISION_MISMATCH",
                           _parse("REVISION_MISMATCH", SKU, revision="C", expected_revision="A", revision_status="MISMATCH",
                                  scan_label="FULL_SCAN"),
                           {"status": "REVISION_MISMATCH", "reason": "name revision 'C' != recipe revision 'A'"})  # fmt: skip
    content = quarantined(hs, "SYN-HAT-1_A_cut.ply", "PLY_TRUNCATED", _parse("UNMATCHED"), {"status": "UNMATCHED"})
    hs.scan(OTHER, quarantine="UNMATCHED_SKU")  # another station: never listed here

    r = held(hs)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["can_operate"] is True
    items = {i["scan_file_id"]: i for i in body["items"]}
    assert set(items) == {unmatched["scan_file_id"], ambiguous["scan_file_id"], revision["scan_file_id"], content["scan_file_id"]}

    amb = items[ambiguous["scan_file_id"]]
    assert amb["reason_code"] == "AMBIGUOUS_SKU" and amb["outcome"] == "AMBIGUOUS" and "MULTIPLE_KNOWN_SKUS" in amb["outcome_reason"]
    assert [s["sku"] for s in amb["suggestions"]][:2] == ["SYN-HAT-1", "SYN-HAT-2"]
    assert {s["source"] for s in amb["suggestions"][:2]} == {"RESOLVER_CANDIDATE"}
    assert amb["parsed"]["candidates"] == ["SYN-HAT-1", "SYN-HAT-2"]
    assert amb["stats"] == {"size_bytes": 4321, "format": "PLY_BINARY_LE", "vertex_count": 120000, "face_count": 0,
                            "bbox_mm": [[0.0, 0.0, 0.0], [120.0, 80.0, 40.0]]}  # fmt: skip
    assert amb["file_name"] == "SYN-HAT-1 SYN-HAT-2 scan.ply" and amb["quarantined_at"] and amb["received_at"]

    rev = items[revision["scan_file_id"]]
    assert rev["parsed"]["revision"] == "C" and rev["parsed"]["expected_revision"] == "A"
    assert rev["parsed"]["scan_label"] == "FULL_SCAN"
    first = rev["suggestions"][0]
    assert first["sku"] == SKU and first["source"] == "FILENAME_SKU"
    assert first["inspection_stage"] == "IN_PROCESS" and first["revision"] == "A" and first["assignable"] is True

    unm = items[unmatched["scan_file_id"]]
    fuzzy = [s for s in unm["suggestions"] if s["source"] == "FILENAME_FUZZY"]
    assert fuzzy and fuzzy[0]["sku"] == SKU and fuzzy[0]["score"] >= 0.75  # 'SYN-HAT1' ~ 'SYN-HAT-1'

    cut = items[content["scan_file_id"]]
    assert cut["resolvable"] is False and cut["suggestions"] == []  # bytes unusable: re-export, never assign


def test_held_scans_device_scope_and_station_check(hs: LineHarness) -> None:
    quarantined(hs, "nothing.ply", "UNPARSEABLE_FILENAME", _parse("UNPARSEABLE"), {"status": "UNPARSEABLE"})
    _, device = hs.device_token(STATION)
    r = held(hs, device)
    assert r.status_code == 200 and len(r.json()["items"]) == 1
    assert r.json()["can_operate"] is False  # a kiosk device token can look, not assign
    assert held(hs, device, OTHER).status_code == 403
    assert hs.client.get("/api/line/quarantine/held", headers=h("tok-operator")).status_code == 422
    assert held(hs, "tok-operator", "cell-9").status_code == 404


def test_suggest_parts_fuzzy_ranks_and_released_only() -> None:
    refs = [
        ActiveRecipeRef("10839000", "10839000@v2", 2, "a" * 64, "01", "b" * 64, "COMPILED"),
        ActiveRecipeRef("10839001", "10839001@v1", 1, "c" * 64, None, "d" * 64, "COMPILED"),
        ActiveRecipeRef("23553000", "23553000@v1", 1, "e" * 64, "B", "f" * 64, "COMPILED"),
    ]
    out = picker.suggest_parts({"original_name": "1083900_01_FULL_SCAN 1.ply", "parse": {"status": "UNMATCHED"}}, refs)
    assert [s["sku"] for s in out][:2] in (["10839000", "10839001"], ["10839001", "10839000"])
    assert "23553000" not in [s["sku"] for s in out]
    # Split tokens are joined ("10839 000" -> "10839000"); a retired / unknown candidate is never suggested.
    out = picker.suggest_parts({"original_name": "10839 000 scan.ply", "parse": {"candidates": ["99999999"]}}, refs)
    assert out[0]["sku"] == "10839000" and out[0]["score"] == 1.0
    assert all(s["sku"] != "99999999" for s in out)


# =========================================================================== released parts search


def test_released_parts_search_lists_released_only(hs: LineHarness) -> None:
    hs.provider.add("SYN-HAT-9@v1", extra={"status": "retired", "compile_status": "COMPILED"})  # retired only
    hs.provider.add("SYN-BOX-3@v1", compiled=None)  # released, not compiled
    hs.provider.add("ZED-1@v1")
    r = hs.client.get(f"/api/line/released-parts?station_id={STATION}", headers=h("tok-operator"))
    assert r.status_code == 200, r.text
    skus = [p["sku"] for p in r.json()["items"]]
    assert "SYN-HAT-9" not in skus and set(skus) == {SKU, "SYN-BOX-3", "ZED-1"}
    by = {p["sku"]: p for p in r.json()["items"]}
    assert by["SYN-BOX-3"]["assignable"] is False and by["SYN-BOX-3"]["not_ready_reason"] == "RECIPE_COMPILE_PENDING"
    hat = by[SKU]
    assert hat["assignable"] and hat["recipe_id"] == RECIPE_ID and hat["inspection_stage"] == "IN_PROCESS"
    assert hat["revision"] == "A" and hat["part_name"] == "Synthetic hat section" and hat["bends_total"] == 4
    assert hat["preview_url"] is None
    assert [v["recipe_id"] for v in hat["versions"] if v["selectable"]] == [RECIPE_ID]

    def search(q: str) -> list[str]:
        resp = hs.client.get("/api/line/released-parts", params={"q": q, "station_id": STATION}, headers=h("tok-viewer"))
        assert resp.status_code == 200
        return [p["sku"] for p in resp.json()["items"]]

    assert search("syn") == [SKU, "SYN-BOX-3"]  # prefix, assignable first
    assert search("HAT") == [SKU]  # substring
    assert search("hat-9") == []  # retired never found
    assert search("hat section") == [SKU]  # part name
    assert search("zz") == []
    _, device = hs.device_token(STATION)
    assert hs.client.get("/api/line/released-parts", params={"station_id": STATION}, headers=h(device)).status_code == 200
    assert hs.client.get("/api/line/released-parts", headers=h(device)).status_code == 403


def test_recipe_versions_from_the_real_resolver_store(tmp_path: Path) -> None:
    from apps.production_line.recipes.resolver import LineRecipeResolver
    from apps.production_line.recipes.store import RecipeStore

    store = RecipeStore(tmp_path / "recipes.sqlite3")
    resolver = LineRecipeResolver(store, data_root=tmp_path)
    assert resolver.list_recipe_versions("NOPE-1") == []


# =========================================================================== assign


def test_assign_happy_path_records_operator_stage_and_audit(hs: LineHarness) -> None:
    q = quarantined(hs, "SYN-HAT-1_C_FULL_SCAN 1.ply", "REVISION_MISMATCH", _parse("REVISION_MISMATCH", SKU, revision="C"),
                    {"status": "REVISION_MISMATCH", "reason": "name revision 'C' != recipe revision 'A'"})  # fmt: skip
    url = f"/api/line/quarantine/{q['scan_file_id']}/assign"
    _, device = hs.device_token(STATION)
    assert hs.client.post(url, json={"sku": SKU}, headers=h(device)).status_code == 403
    r = hs.client.post(url, json={"sku": SKU, "declared_stage": "FINAL"}, headers=h("tok-operator"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["assigned_now"] is True and body["recipe_id"] == RECIPE_ID
    assert body["declared_stage"] == {"stage_index": 4, "source": "OPERATOR"}  # FINAL = every one of the 4 bends
    job = hs.store.get_job(body["job_id"])
    assert job is not None and job["execution_status"] == "queued"
    assert job["declared_stage"] == {"stage_index": 4, "source": "OPERATOR"}
    assert job["compiled_part_sha256"] == COMPILED_SHA and job["scan_file_id"] == q["scan_file_id"]

    ev = hs.store.events_after(0, kinds=["QUARANTINE_ASSIGNED"])
    assert len(ev) == 1
    e = ev[0]
    assert e["principal"] == {"kind": "user", "id": "2", "role": "operator"} and e["ts"]
    p = e["payload"]
    assert p["from_outcome"] == "REVISION_MISMATCH" and p["previous_reason"] == "REVISION_MISMATCH"
    assert p["recipe_id"] == RECIPE_ID and p["recipe_version"] == 1 and p["sku"] == SKU
    assert p["compiled_part_sha256"] == COMPILED_SHA and p["operator_name"] == "oscar" and p["assigned_at"]
    assert p["declared_stage"] == {"stage_index": 4, "source": "OPERATOR"}
    queued = hs.store.events_after(0, kinds=["JOB_QUEUED"])
    assert queued[-1]["job_id"] == body["job_id"] and queued[-1]["principal"]["id"] == "2"

    # The job record says the part was assigned by an operator.
    detail = hs.client.get(f"/api/line/jobs/{body['job_id']}", headers=h("tok-viewer")).json()
    assert detail["assignment"]["by_operator"] is True and detail["assignment"]["from_reason"] == "REVISION_MISMATCH"
    assert detail["assignment"]["assigned_by"]["id"] == "2"
    assert detail["scan_file"]["match"]["reason"] == "OPERATOR_ASSIGNED"
    dev = hs.client.get(f"/api/line/jobs/{body['job_id']}", headers=h(device)).json()
    assert dev["assignment"]["by_operator"] is True and dev["assignment"]["assigned_by"] is None
    assert held(hs).json()["items"] == []
    # An automatically matched job carries no assignment.
    auto = hs.queue_job()
    assert hs.client.get(f"/api/line/jobs/{auto['job_id']}", headers=h("tok-viewer")).json()["assignment"] is None


def test_assign_rejects_retired_unreleased_and_uncompiled(hs: LineHarness) -> None:
    q = quarantined(hs, "mystery.ply", "UNMATCHED_SKU", _parse("UNMATCHED"), {"status": "UNMATCHED"})
    url = f"/api/line/quarantine/{q['scan_file_id']}/assign"
    op = h("tok-operator")
    hs.provider.add("OLD-1@v1", extra={"status": "retired", "compile_status": "COMPILED"})
    r = hs.client.post(url, json={"recipe_id": "OLD-1@v1"}, headers=op)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "RECIPE_RETIRED"
    r = hs.client.post(url, json={"sku": "OLD-1"}, headers=op)  # only retired versions: not released
    assert r.status_code == 409 and r.json()["detail"]["code"] == "PART_NOT_READY"
    hs.provider.registered.add("DRAFT-1")  # registered, never released
    r = hs.client.post(url, json={"sku": "DRAFT-1"}, headers=op)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "PART_NOT_READY"
    hs.provider.add("NOCOMP-1@v1", compiled=None)  # released, no compiled part
    r = hs.client.post(url, json={"sku": "NOCOMP-1"}, headers=op)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "PART_NOT_READY"
    r = hs.client.post(url, json={"sku": SKU, "declared_stage": "SOMETIMES"}, headers=op)
    assert r.status_code == 422
    r = hs.client.post(url, json={"sku": SKU, "stage_index": 9}, headers=op)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "STAGE_INDEX_INVALID"
    assert hs.store.events_after(0, kinds=["QUARANTINE_ASSIGNED", "JOB_QUEUED"]) == []
    row = hs.store.get_scan_file(q["scan_file_id"])
    assert row is not None and row["ingest_state"] == "quarantined"


def test_declared_stage_rules() -> None:
    final = picker.RecipeFacts("X@v1", "X", 1, None, None, "FINAL", 5, "released", "COMPILED")
    staged = dataclasses.replace(final, inspection_stage="IN_PROCESS")
    assert picker.declared_stage_for(final, None, None) is None
    assert picker.declared_stage_for(final, "FINAL", None) is None  # the recipe's own policy: nothing to declare
    assert picker.declared_stage_for(staged, "FINAL", None) == {"stage_index": 5, "source": "OPERATOR"}
    assert picker.declared_stage_for(staged, "IN_PROCESS", None) is None  # E3 infers the stage
    assert picker.declared_stage_for(final, "IN_PROCESS", 2) == {"stage_index": 2, "source": "OPERATOR"}
    for facts, stage, index, code in (
        (final, "IN_PROCESS", None, "STAGE_INDEX_REQUIRED"),
        (final, "FINAL", 3, "STAGE_INDEX_INVALID"),
        (final, "IN_PROCESS", 6, "STAGE_INDEX_INVALID"),
        (final, "LATER", None, "DECLARED_STAGE_INVALID"),
        (dataclasses.replace(staged, bends_total=None), "FINAL", None, "STAGE_UNAVAILABLE"),
    ):
        with pytest.raises(picker.PickerError) as exc:
            picker.declared_stage_for(facts, stage, index)
        assert exc.value.code == code


def test_double_assign_and_two_operators_race(hs: LineHarness) -> None:
    hs.provider.add("SYN-HAT-2@v1")
    q = quarantined(hs, "SYN-HAT-1 SYN-HAT-2.ply", "AMBIGUOUS_SKU", _parse("AMBIGUOUS", candidates=["SYN-HAT-1", "SYN-HAT-2"]),
                    {"status": "AMBIGUOUS"})  # fmt: skip
    sid = q["scan_file_id"]
    barrier = threading.Barrier(6)
    results: list[Any] = []
    lock = threading.Lock()

    def click(principal: Principal) -> None:
        store = LineStore(hs.settings.paths.db_line)  # its own connection, like another request worker
        barrier.wait()
        try:
            out: Any = svc.assign_quarantine(store, hs.provider, sid, principal, sku=SKU, idempotent=True)
        except svc.LineApiError as exc:
            out = exc
        with lock:
            results.append(out)

    threads = [threading.Thread(target=click, args=(p,)) for p in (OPERATOR, OPERATOR, OPERATOR, OPERATOR_2, OPERATOR_2, OPERATOR)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(results) == 6 and all(isinstance(r, dict) for r in results), results
    assert len({r["job_id"] for r in results}) == 1
    assert sum(r["assigned_now"] for r in results) == 1
    assert len(hs.store.events_after(0, kinds=["QUARANTINE_ASSIGNED"])) == 1
    assert len(hs.store.list_jobs(station_id=STATION)) == 1

    # A late second operator picking a *different* part is refused, naming the first pick.
    r = hs.client.post(f"/api/line/quarantine/{sid}/assign", json={"sku": "SYN-HAT-2"}, headers=h("tok-operator"))
    assert r.status_code == 409 and r.json()["detail"]["code"] == "ALREADY_RESOLVED"
    assert RECIPE_ID in r.json()["detail"]["message"]
    r = hs.client.post(f"/api/line/quarantine/{sid}/dismiss", json={"reason": "RESCAN"}, headers=h("tok-operator"))
    assert r.status_code == 409 and r.json()["detail"]["code"] == "ALREADY_RESOLVED"


# =========================================================================== dismiss


def test_dismiss_with_reason_is_idempotent_and_audited(hs: LineHarness) -> None:
    q = quarantined(hs, "junk.ply", "UNMATCHED_SKU", _parse("UNMATCHED"), {"status": "UNMATCHED", "reason": "NO_KNOWN_SKU_TOKEN"})
    url = f"/api/line/quarantine/{q['scan_file_id']}/dismiss"
    op = h("tok-operator")
    assert hs.client.post(url, json={"reason": "RESCAN"}, headers=h("tok-viewer")).status_code == 403
    r = hs.client.post(url, json={"reason": "OTHER"}, headers=op)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "DISMISS_NOTE_REQUIRED"
    assert hs.client.post(url, json={"reason": "BECAUSE"}, headers=op).status_code == 422
    r = hs.client.post(url, json={"reason": "NOT_A_PART", "note": "test coupon"}, headers=op)
    assert r.status_code == 200 and r.json()["ingest_state"] == "dismissed"
    again = hs.client.post(url, json={"reason": "NOT_A_PART"}, headers=op)
    assert again.status_code == 200 and again.json()["ingest_state"] == "dismissed"
    ev = hs.store.events_after(0, kinds=["QUARANTINE_DISMISSED"])
    assert len(ev) == 1
    p = ev[0]["payload"]
    assert p["dismiss_reason"] == "NOT_A_PART" and p["note"] == "test coupon" and p["reason"] == "UNMATCHED_SKU"
    assert p["from_outcome"] == "UNMATCHED" and p["operator_name"] == "oscar" and p["dismissed_at"]
    assert ev[0]["principal"]["id"] == "2"
    assert held(hs).json()["items"] == []
    r = hs.client.post(f"/api/line/quarantine/{q['scan_file_id']}/assign", json={"sku": SKU}, headers=op)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "ALREADY_RESOLVED"


# =========================================================================== E3 still decides


WRONG_PART_INSPECT = r'''
import argparse, json, os, pathlib, sys
ap = argparse.ArgumentParser()
for a in ["--compiled", "--recipe", "--scan", "--out", "--scan-sha256", "--compiled-sha256", "--recipe-sha256",
          "--time-budget-s", "--declared-stage", "--declared-stage-source", "--engine-bundle-id", "--config"]:
    ap.add_argument(a)
args = ap.parse_args()
d = pathlib.Path(os.environ["FAKE_DIR"])
(d / "engine_marker.json").write_text(json.dumps({"argv": sys.argv[1:]}))
result = json.loads((d / "template_result.json").read_text())
result["context"] = {}
result["scan"]["sha256"] = args.scan_sha256
out = pathlib.Path(args.out)
out.mkdir(parents=True)
(out / "result.json").write_text(json.dumps(result))
print("RESULT\t" + json.dumps({"summary_state": result["summary"]["state"],
                                "inspection_validity": result["inspection_validity"]["state"], "result": "result.json"}), flush=True)
'''


class CatalogProvider:
    """RecipeProvider + ActiveRecipeCatalog over one runner-fixture recipe."""

    def __init__(self, resolved: ResolvedRecipe) -> None:
        self.resolved = resolved

    def load(self, recipe_id: str, **_: Any) -> ResolvedRecipe:
        if recipe_id != self.resolved.recipe_id:
            raise KeyError(recipe_id)
        return self.resolved

    def list_active_recipes(self) -> list[ActiveRecipeRef]:
        r = self.resolved
        return [ActiveRecipeRef(r.sku, r.recipe_id, r.recipe_version, r.recipe_sha256, None, r.compiled_part_sha256, "COMPILED")]

    def list_registered_skus(self) -> list[str]:
        return [self.resolved.sku]


def test_operator_assigned_job_still_runs_e3_identity_checks(tmp_path: Path) -> None:
    from apps.production_line.runner.runner import JobRunner, RunnerConfig
    from tests.test_prod_runner import EXAMPLES, FAKE_RENDER, build_fixture_docs, ply_bytes, sha, write_compiled

    root, engine_root, fake_dir = tmp_path / "root", tmp_path / "bundle", tmp_path / "fake"
    for d in (root, engine_root, fake_dir):
        d.mkdir()
    (engine_root / "fake_inspect.py").write_text(textwrap.dedent(WRONG_PART_INSPECT))
    (engine_root / "fake_render.py").write_text(textwrap.dedent(FAKE_RENDER))
    cad = b"ISO-10303-21;\nHEADER; synthetic SYN-HAT-1 test CAD\n"
    recipe, compiled, stage_files, _ = build_fixture_docs(cad)
    cad_path = root / recipe.cad.stored_rel_path
    cad_path.parent.mkdir(parents=True)
    cad_path.write_bytes(cad)
    compiled_dir = root / "recipes" / "compiled" / recipe.recipe_sha256
    write_compiled(compiled_dir, compiled, stage_files)
    # E3's answer for this scan: WRONG_PART_SUSPECTED (the scan is another part under a right name).
    wrong = json.loads((EXAMPLES / "shopfloor_result_v1.invalid.example.json").read_text())
    wrong["recipe"].update(recipe_sha256=recipe.recipe_sha256, content_sha256=recipe.content_sha256)
    wrong["engine"]["compiled_part_sha256"] = compiled.compiled_part_sha256
    wrong["inspection_validity"]["state"] = "WRONG_PART_SUSPECTED"
    wrong["summary"]["reasons"] = ["VALIDITY:WRONG_PART_SUSPECTED"]
    (fake_dir / "template_result.json").write_text(json.dumps(wrong))
    provider = CatalogProvider(
        ResolvedRecipe(
            recipe_id=recipe.recipe_id, recipe_version=recipe.recipe_version, recipe_sha256=recipe.recipe_sha256,
            sku=recipe.sku, recipe_json_bytes=canonical_json_bytes(recipe), cad_path=cad_path, cad_sha256=recipe.cad.sha256,
            part_name=recipe.part_name, compiled_dir=compiled_dir, compiled_part_sha256=compiled.compiled_part_sha256,
            extra={"status": "released", "compile_status": "COMPILED"},
        )
    )  # fmt: skip

    store = LineStore(root / "db" / "line.sqlite3")
    scan = ply_bytes(7)
    scan_sha = sha(scan)
    rel = f"archive/scans/2026/10/{scan_sha}.ply"
    (root / rel).parent.mkdir(parents=True)
    (root / rel).write_bytes(scan)
    row, _ = store.register_scan(
        sha256=scan_sha, original_name="mystery part.ply", size_bytes=len(scan), station_id="cell-1",
        stored_rel_path=rel, parse=_parse("UNMATCHED"), match={"status": "UNMATCHED", "reason": "NO_KNOWN_SKU_TOKEN"},
        format="PLY_BINARY_LE", vertex_count=3, face_count=0, ingest_state="quarantined", quarantine_reason="UNMATCHED_SKU",
    )  # fmt: skip
    out = svc.assign_quarantine(store, provider, row["scan_file_id"], OPERATOR, sku=recipe.sku, declared_stage="FINAL", idempotent=True)
    assert out["declared_stage"] == {"stage_index": len(compiled.bends), "source": "OPERATOR"}

    runner = JobRunner(
        RunnerConfig(
            engine_root=engine_root, python_exe=Path(sys.executable), data_root=root, inspect_module="fake_inspect",
            render_module="fake_render", compile_module="fake_compile", supervisor_poll_s=0.1, lease_ttl_s=30,
            heartbeat_s=0.5, kill_grace_s=2, engine_bundle_id="bundle-test", extra_env={"FAKE_DIR": str(fake_dir)},
        ),
        store, provider,
    )  # fmt: skip
    assert runner.run_once() == out["job_id"]
    job = store.get_job(out["job_id"])
    assert job is not None and job["execution_status"] == "completed", job and job["failure"]
    # The operator picked the recipe; the verdict is E3's: refused, never a PASS.
    assert job["result_summary"]["inspection_validity"] == "WRONG_PART_SUSPECTED"
    assert job["result_summary"]["summary_state"] == "INVALID_INSPECTION"
    argv = json.loads((fake_dir / "engine_marker.json").read_text())["argv"]

    def arg(name: str) -> str:
        return argv[argv.index(name) + 1]

    # Every identity pin reaches E3 exactly as on the automatic path, plus the operator's declaration.
    assert arg("--scan-sha256") == scan_sha
    assert arg("--compiled-sha256") == compiled.compiled_part_sha256 and arg("--recipe-sha256") == recipe.recipe_sha256
    assert arg("--declared-stage") == str(len(compiled.bends)) and arg("--declared-stage-source") == "OPERATOR"
    assert not any("mystery" in a for a in argv)  # filename tokens never reach the engine
    final = json.loads((root / "jobs" / out["job_id"] / "result" / "shopfloor_result.json").read_text())
    assert final["inspection_validity"]["state"] == "WRONG_PART_SUSPECTED" and final["summary"]["state"] == "INVALID_INSPECTION"
