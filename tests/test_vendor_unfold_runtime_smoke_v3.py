"""Pure mocked V3 evaluation/provenance tests; no generation or native work.

Private temporary inputs are freshly authored text/JSON, never CAD. Dependency
pin discovery and wrapper calls are mocked. Pure numeric charts exercise the
PUBLIC composed validator without importing OCP or executing file jobs.
"""
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from backend import vendor_unfold_preview_v4 as engine
from backend.vendor_unfold_contract_v3 import PreviewContractError
from scripts import run_vendor_unfold_runtime_smoke_v3 as smoke
from tests.test_vendor_unfold_preview_v4 import quarter


RUNTIME_PINS = {"/authored/v4-runtime-source": "b"*64}
GENERATION_PINS = {"/authored/v2-generator-source": "a"*64}


@pytest.fixture(autouse=True)
def forbid_live_dependencies_and_generation(monkeypatch):
    monkeypatch.setattr(smoke.worker, "_source_pins", lambda: RUNTIME_PINS.copy())
    monkeypatch.setattr(smoke.generation, "_pins", lambda: GENERATION_PINS.copy())
    monkeypatch.setattr(smoke.worker, "_cache", {})
    monkeypatch.setattr(smoke.worker, "_cache_bytes", 0)
    monkeypatch.setattr(smoke.generation, "generate", lambda *args: pytest.fail("Generation forbidden in V3 evaluation tests"))
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", lambda *args: pytest.fail("Wrapper must be mocked"))


@pytest.fixture
def prepared(tmp_path):
    folder = tmp_path/"fresh-authored-mocks"
    folder.mkdir()
    cases = []
    for case_id in smoke.generation.CASE_IDS:
        case = smoke.generation.case_definition(case_id)
        raw = ("fresh authored non-CAD bytes: "+case_id).encode()
        (folder/case["step_filename"]).write_bytes(raw)
        cases.append({**case, "source_sha256": smoke.generation._hash(raw)})
    smoke.generation.save_new(folder/"generated-cases.json", cases)
    record = tmp_path/"authored-generation.jsonl"
    frozen = smoke.generation.freeze_generation(folder, record)
    with record.open("xb") as stream:
        smoke.generation._append(stream, {"event": "START", "freeze": frozen})
        smoke.generation._append(stream, {"event": "TERMINAL", "freeze": frozen,
            "status": "COMPLETE_SYNTHETIC_GENERATION", "cases": cases, "returncode": 0,
            "physical_inputs_read": False, "manufacturing_ready": False,
            "elapsed_seconds": .5, "peak_process_group_rss_kib": 100})
    return folder, record, cases


def mocked_pre(monkeypatch):
    monkeypatch.setattr(smoke, "_review", lambda *args: {"authored_mock_pre": True})


def receipt(case, bundle):
    """Real pure-chart V4 receipt with the fixed L90 inner-R2 dimensions."""
    faces, axes, _ = quarter(parent_depth=20., child_depth=15.)
    for face in faces:
        old_radius = 4. if face.index < 3 else 5.
        new_radius = 2. if face.index < 3 else 3.
        if face.kind == "cylinder":
            face.vertices[:, :2] *= new_radius/old_radius
            face.radius, face.v1 = new_radius, 40.
        elif face.index in (0, 3):
            face.vertices[:, 0] = new_radius
        else:
            face.vertices[:, 1] = new_radius
        face.vertices[:, 2] *= 40./12.
        face.origin = face.vertices[0].copy() if face.kind == "plane" else np.zeros(3)
        for points in face.edges.values():
            points[:, :2] *= new_radius/old_radius
            points[:, 2] *= 40./12.
    candidate = smoke.generation.case_definition("L90")["candidates"][0]
    axes[0].update(radii_mm=[2.,3.], canonical_line_start=candidate["line_start"],
                   canonical_line_end=candidate["line_end"])
    result = engine._develop_faces(faces, axes, [candidate], case["source_sha256"])
    result["diagnostics"].update(transport_version=3, cache_profile=smoke.worker.PROFILE,
        v4_engine_source_sha256=smoke.worker.ENGINE_SHA256,
        candidate_binding=copy.deepcopy(case["candidates"]),
        candidate_input_sha256=smoke.generation._hash(smoke.generation._json(case["candidates"])),
        source_bundle_sha256=bundle, limits=dict(smoke.worker.LIMITS),
        elapsed_seconds=.2, peak_process_group_rss_kib=100, cache_hit=False)
    return result


def mock_wrapper(monkeypatch, cases, bundle, *, aliased=False, advance=None):
    by_hash, cached, calls = {case["source_sha256"]: case for case in cases}, {}, []
    def bounded(path, expected, candidates):
        calls.append((Path(path).name, expected, copy.deepcopy(candidates)))
        if smoke.generation._hash(Path(path).read_bytes()) != expected:
            result = smoke.worker.unavailable("CAD_HASH_MISMATCH")
        elif expected in cached:
            result = cached[expected] if aliased else copy.deepcopy(cached[expected])
            result["diagnostics"]["cache_hit"] = True
        else:
            result = receipt(by_hash[expected], bundle)
            cached[expected] = result if aliased else copy.deepcopy(result)
        if advance is not None:
            advance(len(calls))
        return result
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", bounded)
    return calls


def test_module_has_no_native_import_or_generation_entrypoint():
    tree = ast.parse(Path(smoke.__file__).read_text(encoding="utf-8"))
    imported = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
    imported += [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert not any(name.split(".")[0] in {"OCP", "FreeCAD", "numpy"} for name in imported)
    assert not hasattr(smoke, "generate") and not hasattr(smoke, "_generate_native")


def test_freeze_binds_full_generator_cohort_but_actual_worker_subset_only(prepared, tmp_path):
    folder, generation_record, cases = prepared
    frozen = smoke.freeze_evaluation(folder, generation_record, tmp_path/"evaluation.jsonl")
    assert frozen["cases"] == cases
    assert frozen["checks"] == list(smoke.CHECKS)
    assert frozen["selection"] == "ALL_GENERATED_CASES_IN_RETAINED_ORDER"
    assert frozen["wrapper_source_pins"] == RUNTIME_PINS
    assert frozen["wrapper_source_bundle_sha256"] == smoke.generation._hash(smoke.generation._json(RUNTIME_PINS))
    assert frozen["wrapper_source_bundle_sha256"] != smoke.generation._hash(smoke.generation._json(frozen["source_pins"]))
    assert set(RUNTIME_PINS) | set(GENERATION_PINS) <= set(frozen["source_pins"])
    assert set(GENERATION_PINS).isdisjoint(frozen["wrapper_source_pins"])
    assert cases[0]["oracle"] == cases[1]["oracle"]
    assert cases[0]["oracle"]["flat_area_mm2"] == 40*(35+smoke.generation.math.pi)
    assert frozen["limits"]["total_wall_seconds"] == 420
    frozen["limits"]["total_wall_seconds"] = 999
    assert smoke.LIMITS["total_wall_seconds"] == 420


def test_conflicting_pin_union_is_refused(monkeypatch):
    monkeypatch.setattr(smoke.generation, "_pins", lambda: {**GENERATION_PINS, **{key: "c"*64 for key in RUNTIME_PINS}})
    with pytest.raises(ValueError, match="PIN_UNION_CONFLICT"):
        smoke._pin_sets()


@pytest.mark.parametrize("change", ["subset", "oracle", "source", "generation_record", "extra"])
def test_generator_provenance_is_reused_without_reducing_or_adapting_cases(prepared, tmp_path, change):
    folder, record, cases = prepared
    if change in {"subset", "oracle"}:
        modified = copy.deepcopy(cases[:1] if change == "subset" else cases)
        if change == "oracle": modified[1]["oracle"]["flat_area_mm2"] += 1
        (folder/"generated-cases.json").write_bytes(smoke.generation._json(modified))
    elif change == "source":
        (folder/cases[0]["step_filename"]).write_bytes(b"different authored mock bytes")
    elif change == "generation_record":
        events = [json.loads(line) for line in record.read_bytes().splitlines()]
        events[-1]["returncode"] = 1
        record.write_bytes(b"\n".join(smoke.generation._json(event) for event in events)+b"\n")
    else:
        (folder/"unselected.step").write_bytes(b"authored mock")
    with pytest.raises(ValueError):
        smoke.freeze_evaluation(folder, record, tmp_path/"refused.jsonl")


def test_complete_mocked_evaluation_retains_all_six_receipts_before_assertions(prepared, tmp_path, monkeypatch):
    folder, generation_record, cases = prepared
    record = tmp_path/"evaluation.jsonl"
    frozen = smoke.freeze_evaluation(folder, generation_record, record)
    mocked_pre(monkeypatch)
    calls = mock_wrapper(monkeypatch, cases, frozen["wrapper_source_bundle_sha256"])
    ledger = smoke.evaluate(frozen, folder, record, generation_record, "mocked", "mocked")
    assert ledger["status"] == "COMPLETE_SYNTHETIC_V3_RUNTIME_SMOKE" and len(calls) == 6
    checks = [check for row in ledger["cases"] for check in row["checks"]]
    assert len(checks) == 6 and all(check["status"] == "PASSED" for check in checks)
    assert ledger["total_receipt_bytes"] == sum(check["receipt_bytes"] for check in checks)
    for check in checks:
        raw = Path(check["receipt_path"]).read_bytes()
        assert smoke.generation._hash(raw) == check["receipt_sha256"]
        if check["check"] == "cold":
            assert json.loads(raw)["diagnostics"]["limits"]["wall_seconds"] == 60
    assert [json.loads(line)["event"] for line in record.read_bytes().splitlines()] == ["START", "CASE", "CASE", "TERMINAL"]
    with pytest.raises(FileExistsError):
        smoke.evaluate(frozen, folder, record, generation_record, "mocked", "mocked")
    assert len(calls) == 6


def test_cache_alias_failure_retains_pristine_cold_and_actual_corrupted_cache(prepared, tmp_path, monkeypatch):
    folder, generation_record, cases = prepared
    record = tmp_path/"alias-failure.jsonl"
    frozen = smoke.freeze_evaluation(folder, generation_record, record)
    mocked_pre(monkeypatch)
    calls = mock_wrapper(monkeypatch, cases, frozen["wrapper_source_bundle_sha256"], aliased=True)
    ledger = smoke.evaluate(frozen, folder, record, generation_record, "mocked", "mocked")
    assert ledger["status"] == "FAILED" and len(calls) == 4
    for row in ledger["cases"]:
        assert [check["status"] for check in row["checks"]] == ["PASSED", "FAILED", "NOT_EXECUTED_AFTER_FAILURE"]
        cold, cache, _ = row["checks"]
        cold_payload = json.loads(Path(cold["receipt_path"]).read_bytes())
        cache_payload = json.loads(Path(cache["receipt_path"]).read_bytes())
        assert cold_payload["diagnostics"]["limits"]["wall_seconds"] == 60
        assert cache_payload["diagnostics"]["limits"]["wall_seconds"] == -1


def test_cold_failure_keeps_both_cases_and_all_checks(prepared, tmp_path, monkeypatch):
    folder, generation_record, _ = prepared
    record = tmp_path/"cold-failure.jsonl"
    frozen = smoke.freeze_evaluation(folder, generation_record, record)
    mocked_pre(monkeypatch)
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", lambda *args: smoke.worker.unavailable("MOCK_FAILURE"))
    ledger = smoke.evaluate(frozen, folder, record, generation_record, "mocked", "mocked")
    assert ledger["status"] == "FAILED" and len(ledger["cases"]) == 2
    for row in ledger["cases"]:
        assert [check["status"] for check in row["checks"]] == ["FAILED", "NOT_EXECUTED_AFTER_FAILURE", "NOT_EXECUTED_AFTER_FAILURE"]
        assert json.loads(Path(row["checks"][0]["receipt_path"]).read_bytes())["reason"] == "MOCK_FAILURE"


def test_campaign_interrupt_leaves_no_running_or_unaccounted_checks(prepared, tmp_path, monkeypatch):
    folder, generation_record, _ = prepared
    record = tmp_path/"interrupted.jsonl"
    frozen = smoke.freeze_evaluation(folder, generation_record, record)
    mocked_pre(monkeypatch)
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt("authored interruption")))
    ledger = smoke.evaluate(frozen, folder, record, generation_record, "mocked", "mocked")
    assert ledger["status"] == "FAILED" and ledger["exception_type"] == "KeyboardInterrupt"
    assert [check["status"] for check in ledger["cases"][0]["checks"]] == ["FAILED", "NOT_EXECUTED_AFTER_FAILURE", "NOT_EXECUTED_AFTER_FAILURE"]
    assert all(check["status"] == "NOT_EXECUTED_AFTER_FAILURE" for check in ledger["cases"][1]["checks"])
    assert json.loads(record.read_bytes().splitlines()[-1]) == ledger


def test_slow_pins_before_cold_call_cannot_launch_after_budget(prepared, tmp_path, monkeypatch):
    folder, generation_record, _ = prepared
    record = tmp_path/"slow-pins.jsonl"
    frozen = smoke.freeze_evaluation(folder, generation_record, record)
    mocked_pre(monkeypatch)
    clock, calls = {"now": 0., "pins": 0}, []
    original = smoke._pin_sets
    def pin_sets():
        clock["pins"] += 1
        if clock["pins"] > 1: clock["now"] = 500.
        return original()
    monkeypatch.setattr(smoke, "_pin_sets", pin_sets)
    monkeypatch.setattr(smoke.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", lambda *args: calls.append(args))
    ledger = smoke.evaluate(frozen, folder, record, generation_record, "mocked", "mocked")
    assert not calls and ledger["status"] == "FAILED"
    assert all(check["status"] == "NOT_EXECUTED_AFTER_FAILURE" for row in ledger["cases"] for check in row["checks"])


@pytest.mark.parametrize("after_call", [1, 2])
def test_between_call_exhaustion_preserves_receipts_and_blocks_next_call(prepared, tmp_path, monkeypatch, after_call):
    folder, generation_record, cases = prepared
    record = tmp_path/f"between-call-{after_call}.jsonl"
    frozen = smoke.freeze_evaluation(folder, generation_record, record)
    mocked_pre(monkeypatch)
    clock = {"now": 0.}
    monkeypatch.setattr(smoke.time, "monotonic", lambda: clock["now"])
    def advance(count):
        if count == after_call: clock["now"] = 361.
    calls = mock_wrapper(monkeypatch, cases, frozen["wrapper_source_bundle_sha256"], advance=advance)
    ledger = smoke.evaluate(frozen, folder, record, generation_record, "mocked", "mocked")
    assert len(calls) == after_call and ledger["status"] == "FAILED"
    assert [check["status"] for check in ledger["cases"][0]["checks"]] == ["PASSED"]*after_call+["NOT_EXECUTED_AFTER_FAILURE"]*(3-after_call)
    assert all(check["status"] == "NOT_EXECUTED_AFTER_FAILURE" for check in ledger["cases"][1]["checks"])


@pytest.mark.parametrize("final_elapsed", [420.,421.])
def test_final_campaign_clock_cannot_publish_complete_after_budget(prepared,tmp_path,monkeypatch,final_elapsed):
    folder,generation_record,cases=prepared
    record=tmp_path/"final-clock-expiry.jsonl"
    frozen=smoke.freeze_evaluation(folder,generation_record,record)
    mocked_pre(monkeypatch)
    mock_wrapper(monkeypatch,cases,frozen["wrapper_source_bundle_sha256"])
    ticks=iter([0.]*7+[419.,final_elapsed])
    monkeypatch.setattr(smoke.time,"monotonic",lambda:next(ticks))
    ledger=smoke.evaluate(frozen,folder,record,generation_record,"mocked","mocked")
    assert ledger["status"]=="FAILED" and ledger["elapsed_seconds"]==final_elapsed
    assert ledger["reason"]=="V3_EVALUATION_CAMPAIGN_WALL_BUDGET_EXCEEDED"
    assert all(check["status"]=="PASSED" for row in ledger["cases"] for check in row["checks"])
    assert all(Path(check["receipt_path"]).exists() for row in ledger["cases"] for check in row["checks"])


@pytest.mark.parametrize("now,admitted", [(360., True), (360.000001, False)])
def test_call_admission_requires_one_full_sixty_second_budget(monkeypatch, now, admitted):
    calls, check = [], {"status": "NOT_EXECUTED"}
    case = {"source_sha256": "a"*64, "candidates": []}
    monkeypatch.setattr(smoke.time, "monotonic", lambda: now)
    monkeypatch.setattr(smoke.worker, "build_unfold_preview_bounded", lambda *args: calls.append(args))
    if admitted:
        smoke._call_admitted(0., check, "authored", case)
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError, match="ADMISSION_BUDGET"):
            smoke._call_admitted(0., check, "authored", case)
        assert not calls and check["status"] == "NOT_EXECUTED"


@pytest.mark.parametrize("change", ["case", "oracle", "runtime_subset", "all_pins", "limits", "checks"])
def test_frozen_evaluation_cannot_be_relaxed(prepared, tmp_path, monkeypatch, change):
    folder, generation_record, _ = prepared
    record = tmp_path/"refused-freeze.jsonl"
    frozen = smoke.freeze_evaluation(folder, generation_record, record)
    mocked_pre(monkeypatch)
    if change == "case": frozen["cases"].pop()
    elif change == "oracle": frozen["cases"][0]["oracle"]["flat_area_mm2"] += 1
    elif change == "runtime_subset": frozen["wrapper_source_pins"].update(GENERATION_PINS)
    elif change == "all_pins": frozen["source_pins"].pop(next(iter(frozen["source_pins"])))
    elif change == "limits": frozen["limits"]["total_wall_seconds"] = 999
    else: frozen["checks"].pop()
    with pytest.raises(ValueError, match="FREEZE_MISMATCH"):
        smoke.evaluate(frozen, folder, record, generation_record, "mocked", "mocked")
    assert not record.exists()


def test_fresh_process_cache_is_required(prepared, tmp_path, monkeypatch):
    folder, generation_record, _ = prepared
    record = tmp_path/"nonempty-cache.jsonl"
    frozen = smoke.freeze_evaluation(folder, generation_record, record)
    mocked_pre(monkeypatch)
    monkeypatch.setattr(smoke.worker, "_cache_bytes", 1)
    with pytest.raises(ValueError, match="FRESH_PROCESS_CACHE"):
        smoke.evaluate(frozen, folder, record, generation_record, "mocked", "mocked")
    assert not record.exists()


def test_receipt_budget_retains_observed_hash_and_skipped_checks(prepared, tmp_path, monkeypatch):
    folder, generation_record, cases = prepared
    record = tmp_path/"receipt-budget.jsonl"
    monkeypatch.setattr(smoke, "LIMITS", {**smoke.LIMITS, "total_receipt_bytes": 1})
    frozen = smoke.freeze_evaluation(folder, generation_record, record)
    mocked_pre(monkeypatch)
    mock_wrapper(monkeypatch, cases, frozen["wrapper_source_bundle_sha256"])
    ledger = smoke.evaluate(frozen, folder, record, generation_record, "mocked", "mocked")
    assert ledger["status"] == "FAILED" and ledger["total_receipt_bytes"] == 0
    for row in ledger["cases"]:
        cold = row["checks"][0]
        assert cold["result_observed"] is True and len(cold["observed_payload_sha256"]) == 64
        assert "receipt_path" not in cold
        assert [check["status"] for check in row["checks"]] == ["FAILED", "NOT_EXECUTED_AFTER_FAILURE", "NOT_EXECUTED_AFTER_FAILURE"]


def test_derived_receipt_collision_never_overwrites_existing_mock(prepared, tmp_path, monkeypatch):
    folder, generation_record, cases = prepared
    record = tmp_path/"collision.jsonl"
    frozen = smoke.freeze_evaluation(folder, generation_record, record)
    mocked_pre(monkeypatch)
    mock_wrapper(monkeypatch, cases, frozen["wrapper_source_bundle_sha256"])
    prior = record.with_name(record.stem+"-L90-cold.json")
    prior.write_bytes(b"authored prior receipt")
    ledger = smoke.evaluate(frozen, folder, record, generation_record, "mocked", "mocked")
    assert ledger["status"] == "FAILED" and prior.read_bytes() == b"authored prior receipt"


def test_new_pre_is_exact_eight_key_schema_bound_to_full_union(prepared, tmp_path, monkeypatch):
    folder, generation_record, _ = prepared
    frozen = smoke.freeze_evaluation(folder, generation_record, tmp_path/"evaluation.jsonl")
    review = {"schema": smoke.PRE_SCHEMA, "disposition": "GREEN", "author": "author",
        "reviewer": "independent", "blocking_findings": [],
        "freeze_sha256": smoke.generation._hash(smoke.generation._json(frozen)),
        "source_bundle_sha256": smoke.generation._hash(smoke.generation._json(frozen["source_pins"])),
        "source_archive_sha256": "c"*64}
    path = tmp_path/"authored-pre.json"
    path.write_bytes(smoke.generation._json(review))
    seen = []
    monkeypatch.setattr(smoke.generation, "_verify_archive", lambda path,pins: seen.append(pins.copy()) or "c"*64)
    assert smoke._review(frozen, path, "mocked-not-read-archive")["source_archive_sha256"] == "c"*64
    assert seen == [frozen["source_pins"]]
    for changes in ({"schema": "SHERMAN_VENDOR_UNFOLD_RUNTIME_PRE_V2"}, {"reviewer": "author"},
                    {"extra": True}, {"blocking_findings": ["finding"]},
                    {"freeze_sha256": "0"*64}, {"source_bundle_sha256": "0"*64},
                    {"source_archive_sha256": "0"*64}):
        path.write_bytes(smoke.generation._json({**review, **changes}))
        with pytest.raises(ValueError):
            smoke._review(frozen, path, "mocked-not-read-archive")


@pytest.mark.parametrize("change", ["transport", "mechanism", "core", "binding", "digest", "bundle", "profile", "limits", "elapsed", "rss", "oracle", "source", "width_proof"])
def test_public_composed_contract_and_fixed_oracles_reject_adversarial_receipts(prepared, change):
    case = prepared[2][0]
    bundle = smoke.generation._hash(smoke.generation._json(RUNTIME_PINS))
    result = receipt(case, bundle)
    diag = result["diagnostics"]
    if change == "transport": diag["transport_version"] = 2
    elif change == "mechanism": diag["mechanism_version"] = 3
    elif change == "core": diag["v4_engine_source_sha256"] = "c"*64
    elif change == "binding": diag["candidate_binding"][0]["line_start"][0] = True
    elif change == "digest": diag["candidate_input_sha256"] = "c"*64
    elif change == "bundle": diag["source_bundle_sha256"] = "c"*64
    elif change == "profile": diag["cache_profile"] = "OTHER"
    elif change == "limits": diag["limits"]["wall_seconds"] = 999
    elif change == "elapsed": diag["elapsed_seconds"] = float("inf")
    elif change == "rss": diag["peak_process_group_rss_kib"] = 0
    elif change == "oracle": case = copy.deepcopy(case); case["oracle"]["flat_area_mm2"] += 1
    elif change == "source": result["source_sha256"] = "c"*64
    else: diag["finite_width_proofs"].clear()
    with pytest.raises((ValueError, PreviewContractError)):
        smoke.assert_preview(case, result, bundle)
