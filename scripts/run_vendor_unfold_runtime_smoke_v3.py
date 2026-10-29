"""Deferred, separately frozen V3-worker/V4-geometry synthetic evaluation.

Generation belongs to the existing, separately audited V2 generator. Evaluate
ALL of its two fresh L90 cases against the fixed inner-R2 selected-skin oracle,
including the independent rigid pose. Never adapt that oracle after execution.
This is mechanism/runtime evidence only, not broader physical CAD support,
neutral-axis development, calibration, manufacturing or release evidence.
Importing this module does not hash dependencies, read CAD, or run native work.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time

from backend import vendor_unfold_worker_v3 as worker
from backend.vendor_unfold_contract_v3 import canonical_candidates, strict_json, validate_preview
from scripts import run_vendor_unfold_runtime_smoke_v2 as generation


ROOT = Path(__file__).resolve().parents[1]
OWN_SOURCES = ("scripts/run_vendor_unfold_runtime_smoke_v3.py", "tests/test_vendor_unfold_runtime_smoke_v3.py")
CHECKS = generation.CHECKS
LIMITS = {"per_wrapper_call": dict(worker.LIMITS), "case_limit": 2, "total_wall_seconds": 420.0,
          "receipt_bytes_per_check": 5_000_000, "total_receipt_bytes": 5_000_000,
          "wrapper_admission": "IMMEDIATE_REMAINING_TIME_AT_LEAST_ONE_FULL_CALL"}
PRE_SCHEMA = "SHERMAN_VENDOR_UNFOLD_RUNTIME_PRE_V3"


def _pin_sets():
    """Retain the worker's actual subset separately from the runner union."""
    runtime = worker._source_pins()
    groups = (runtime, generation._pins(),
              {str((ROOT / name).resolve()): generation._hash((ROOT / name).read_bytes())
               for name in OWN_SOURCES})
    union = {}
    for group in groups:
        for name, digest in group.items():
            if name in union and union[name] != digest:
                raise ValueError("SMOKE_SOURCE_PIN_UNION_CONFLICT")
            union[name] = digest
    return union, runtime


def freeze_evaluation(directory, generation_record, record_path):
    cases = generation.load_cases(directory)
    provenance = generation._generation_provenance(generation_record, directory, cases)
    pins, runtime = _pin_sets()
    return {"schema": "SHERMAN_VENDOR_UNFOLD_EVALUATION_FREEZE_V3", "source_pins": pins,
            "wrapper_source_pins": runtime,
            "wrapper_source_bundle_sha256": generation._hash(generation._json(runtime)),
            "runner_path": str((ROOT / OWN_SOURCES[0]).resolve()), "limits": copy.deepcopy(LIMITS),
            "directory": str(Path(directory).resolve()), "record_path": str(Path(record_path).resolve()),
            "generation_provenance": provenance, "cases": cases, "checks": list(CHECKS),
            "selection": "ALL_GENERATED_CASES_IN_RETAINED_ORDER",
            "physical_inputs_read": False, "manufacturing_ready": False,
            "qualification_scope": "SYNTHETIC_SELECTED_SKIN_RUNTIME_ONLY",
            "native_face_order_tie_risk": "FIXED_INNER_R2_ORACLE_MAY_FAIL;NO_POSTRUN_ADAPTATION"}


def _review(frozen, review_path, archive_path):
    raw = generation._read(review_path, 100_000)
    review = strict_json(raw)
    keys = {"schema", "disposition", "author", "reviewer", "blocking_findings",
            "freeze_sha256", "source_bundle_sha256", "source_archive_sha256"}
    if (type(review) is not dict or set(review) != keys or review["schema"] != PRE_SCHEMA
            or review["disposition"] != "GREEN" or review["blocking_findings"] != []
            or type(review["author"]) is not str or not review["author"].strip()
            or type(review["reviewer"]) is not str or not review["reviewer"].strip()
            or review["author"] == review["reviewer"]
            or review["freeze_sha256"] != generation._hash(generation._json(frozen))
            or review["source_bundle_sha256"] != generation._hash(generation._json(frozen["source_pins"]))):
        raise ValueError("EXACT_INDEPENDENT_V3_PRE_REQUIRED")
    archive_hash = generation._verify_archive(archive_path, frozen["source_pins"])
    if review["source_archive_sha256"] != archive_hash:
        raise ValueError("V3_PRE_ARCHIVE_BINDING_MISMATCH")
    return {"review_sha256": generation._hash(raw), "source_archive_sha256": archive_hash,
            "independent_review_identity_declared_not_cryptographically_verified": True}


def assert_preview(case, result, wrapper_source_bundle):
    validate_preview(result, case["source_sha256"], case["candidates"])
    if result["status"] != case["oracle"]["status"]:
        raise ValueError("EXPECTED_AVAILABLE_V4_SELECTED_SKIN")
    diag = result["diagnostics"]
    bound = canonical_candidates(diag.get("candidate_binding"))
    if (diag.get("cache_profile") != worker.PROFILE
            or diag.get("v4_engine_source_sha256") != worker.ENGINE_SHA256
            or diag.get("source_bundle_sha256") != wrapper_source_bundle
            or generation._json(bound) != generation._json(case["candidates"])
            or generation._json(diag["candidate_binding"]) != generation._json(bound)
            or diag.get("candidate_input_sha256") != generation._hash(generation._json(bound))
            or generation._json(diag.get("limits")) != generation._json(worker.LIMITS)
            or type(diag.get("elapsed_seconds")) not in (int, float)
            or not 0 <= diag["elapsed_seconds"] < worker.MAX_SECONDS
            or type(diag.get("peak_process_group_rss_kib")) is not int
            or not 0 < diag["peak_process_group_rss_kib"] <= worker.MAX_RSS_KIB
            or len(result["bend_lines"]) != case["oracle"]["bend_count"]):
        raise ValueError("V3_WRAPPER_RECEIPT_BINDING_MISMATCH")
    points = [result["vertices"][i:i+3] for i in range(0, len(result["vertices"]), 3)]
    bbox = sorted(max(point[axis] for point in points)-min(point[axis] for point in points)
                  for axis in (0, 1))
    area = 0.
    for index in range(0, len(result["triangles"]), 3):
        a, b, c = [points[value] for value in result["triangles"][index:index+3]]
        area += abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2
    oracle = case["oracle"]
    if (any(abs(a-b) > generation.ORACLE_ERROR for a,b in zip(bbox, oracle["sorted_flat_bbox_mm"]))
            or abs(area-oracle["flat_area_mm2"]) > generation.ORACLE_ERROR):
        raise ValueError("FIXED_INNER_R2_ANALYTICAL_ORACLE_FAILED")
    return {"flat_bbox_mm": bbox, "flat_area_mm2": area, "bend_count": len(result["bend_lines"])}


def _call_admitted(started, row, path, case):
    if LIMITS["total_wall_seconds"]-(time.monotonic()-started) < worker.MAX_SECONDS:
        raise ValueError("V3_EVALUATION_WRAPPER_ADMISSION_BUDGET_EXHAUSTED")
    row.update(status="RUNNING", reason="CALL_STARTED")
    return worker.build_unfold_preview_bounded(path, case["source_sha256"], case["candidates"])


def _capture(record_path, case_id, row, payload, used):
    encoded = generation._json(payload)+b"\n"
    row.update(result_observed=True, actual_status=payload.get("status") if type(payload) is dict else None,
               observed_payload_sha256=generation._hash(encoded), observed_payload_bytes=len(encoded))
    if len(encoded) > LIMITS["receipt_bytes_per_check"] or used+len(encoded) > LIMITS["total_receipt_bytes"]:
        raise ValueError("V3_EVALUATION_RECEIPT_BYTE_BUDGET_EXCEEDED")
    target = Path(record_path).with_name(Path(record_path).stem+"-"+case_id+"-"+row["check"]+".json")
    generation.save_new(target, payload)
    row.update(receipt_path=str(target.resolve()), receipt_sha256=generation._hash(encoded),
               receipt_bytes=len(encoded), status="OBSERVED", reason="ASSERTIONS_PENDING")
    return used+len(encoded)


def _fail_checks(row, error):
    row.update(reason=str(error)[:1000], exception_type=type(error).__name__)
    for check in row["checks"]:
        if check["status"] in {"RUNNING", "OBSERVED"}:
            check.update(status="FAILED", reason=str(error)[:1000], exception_type=type(error).__name__)
        elif check["status"] == "NOT_EXECUTED":
            check.update(status="NOT_EXECUTED_AFTER_FAILURE", reason="PRIOR_CASE_CHECK_OR_PRECHECK_FAILED")


def evaluate(frozen, directory, record_path, generation_record, review_path, archive_path):
    if generation._json(frozen) != generation._json(freeze_evaluation(directory, generation_record, record_path)):
        raise ValueError("V3_EVALUATION_FREEZE_MISMATCH")
    review = _review(frozen, review_path, archive_path)
    if worker._cache or worker._cache_bytes:
        raise ValueError("V3_EVALUATION_REQUIRES_FRESH_PROCESS_CACHE")
    started, used = time.monotonic(), 0
    rows = [{"case_id": case["case_id"], "passed": False, "reason": "NOT_EXECUTED",
             "checks": [{"check": check, "status": "NOT_EXECUTED", "reason": "NOT_EXECUTED"}
                        for check in CHECKS]} for case in frozen["cases"]]
    ledger = {"event": "TERMINAL", "status": "FAILED", "cases": rows,
              "physical_inputs_read": False, "manufacturing_ready": False,
              "qualification_scope": "SYNTHETIC_SELECTED_SKIN_RUNTIME_ONLY"}
    with Path(record_path).open("xb") as stream:
        generation._append(stream, {"event": "START", "freeze": frozen, "review": review,
                                    "started_at": datetime.now(timezone.utc).isoformat()})
        try:
            with tempfile.TemporaryDirectory(prefix="vendor-smoke-evaluation-v3-") as temp:
                folder = Path(temp)
                for case,row in zip(frozen["cases"], rows):
                    try:
                        if _pin_sets()[0] != frozen["source_pins"]:
                            raise ValueError("SOURCE_CHANGED_DURING_V3_EVALUATION")
                        raw = generation._read(Path(directory)/case["step_filename"], generation.MAX_GENERATED_BYTES)
                        if generation._hash(raw) != case["source_sha256"]:
                            raise ValueError("GENERATED_CASE_HASH_MISMATCH")
                        snapshot = folder/case["step_filename"]
                        with snapshot.open("xb") as source:
                            source.write(raw)
                        cold_row, cache_row, refusal_row = row["checks"]
                        caller_cold = _call_admitted(started, cold_row, snapshot, case)
                        retained_cold = copy.deepcopy(caller_cold)
                        used = _capture(record_path, case["case_id"], cold_row, retained_cold, used)
                        observed = assert_preview(case, retained_cold, frozen["wrapper_source_bundle_sha256"])
                        if retained_cold["diagnostics"].get("cache_hit") is not False:
                            raise ValueError("V3_COLD_CALL_DID_NOT_EXECUTE")
                        cold_row.update(status="PASSED", reason="FIXED_V4_COLD_ORACLES_PASSED")
                        pristine = copy.deepcopy(retained_cold)
                        caller_cold["vertices"][0] += 99
                        caller_cold["diagnostics"]["limits"]["wall_seconds"] = -1
                        cached = _call_admitted(started, cache_row, snapshot, case)
                        used = _capture(record_path, case["case_id"], cache_row, cached, used)
                        assert_preview(case, cached, frozen["wrapper_source_bundle_sha256"])
                        pristine["diagnostics"]["cache_hit"] = True
                        if generation._json(cached) != generation._json(pristine):
                            raise ValueError("V3_CACHE_ISOLATION_OR_HIT_FAILED")
                        cache_row.update(status="PASSED", reason="CACHE_COPY_AND_FIXED_V4_ORACLES_PASSED")
                        changed = folder/(case["case_id"]+"-changed.step")
                        with changed.open("xb") as source:
                            source.write(raw+b"\n/* authored smoke byte change */\n")
                        refusal = _call_admitted(started, refusal_row, changed, case)
                        used = _capture(record_path, case["case_id"], refusal_row, refusal, used)
                        validate_preview(refusal, case["source_sha256"], case["candidates"])
                        if refusal["status"] != "UNAVAILABLE" or refusal["reason"] != "CAD_HASH_MISMATCH":
                            raise ValueError("V3_CHANGED_BYTE_HASH_REFUSAL_FAILED")
                        refusal_row.update(status="PASSED", reason="CHANGED_BYTE_HASH_REFUSAL_PASSED")
                        row.update(passed=True, reason="ALL_FIXED_V4_ORACLES_PASSED", observed=observed)
                    except Exception as error:
                        _fail_checks(row, error)
                    generation._append(stream, {"event": "CASE", "result": row})
            if _pin_sets()[0] != frozen["source_pins"]:
                raise ValueError("SOURCE_CHANGED_DURING_V3_EVALUATION")
            if time.monotonic()-started > LIMITS["total_wall_seconds"]:
                raise ValueError("V3_EVALUATION_CAMPAIGN_WALL_BUDGET_EXCEEDED")
            if all(row["passed"] for row in rows):
                ledger["status"] = "COMPLETE_SYNTHETIC_V3_RUNTIME_SMOKE"
        except BaseException as error:
            ledger.update(reason=str(error)[:1000], exception_type=type(error).__name__)
            for row in rows:
                if any(check["status"] in {"RUNNING", "OBSERVED"} for check in row["checks"]):
                    _fail_checks(row, error)
        for row in rows:
            if row["reason"] == "NOT_EXECUTED":
                row["reason"] = "NOT_EXECUTED_AFTER_CAMPAIGN_FAILURE"
            for check in row["checks"]:
                if check["status"] == "NOT_EXECUTED":
                    check.update(status="NOT_EXECUTED_AFTER_FAILURE", reason="CAMPAIGN_FAILED_BEFORE_CASE")
        elapsed=time.monotonic()-started
        if elapsed >= LIMITS["total_wall_seconds"]:
            ledger.update(status="FAILED",reason="V3_EVALUATION_CAMPAIGN_WALL_BUDGET_EXCEEDED")
        ledger.update(elapsed_seconds=elapsed, total_receipt_bytes=used,
                      finished_at=datetime.now(timezone.utc).isoformat())
        generation._append(stream, ledger)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze-evaluation", "evaluate"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--generation-record", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze-evaluation":
        generation.save_new(args.protocol, freeze_evaluation(args.directory, args.generation_record, args.record))
        return 0
    if args.review is None or args.source_archive is None:
        parser.error("new exact independent V3 PRE and source archive are required before evaluation")
    frozen = strict_json(generation._read(args.protocol, generation.MAX_RECORD_BYTES))
    ledger = evaluate(frozen, args.directory, args.record, args.generation_record, args.review, args.source_archive)
    print(generation._json({"status": ledger["status"]}).decode())
    return 0 if ledger["status"] == "COMPLETE_SYNTHETIC_V3_RUNTIME_SMOKE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
