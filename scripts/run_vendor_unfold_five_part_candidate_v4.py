"""Deferred five-approved-part V4 exploratory successor; never activate a Site.

Only freeze/execute may read the five explicitly released derived JSON files.
Raw CAD is NEVER inspected by this module: after exact current qualification,
PRE/archive/protocol verification and per-call admission, the bounded worker
owns source-byte verification/private snapshots. Original CAD, Site fixtures
and terminal generations are read-only; all proposals/receipts/input snapshots
are NEW. The five exact released derived inputs are retained before any call.
An AVAILABLE selected skin is not accuracy, neutral-axis, physical calibration,
manufacturing or generalization evidence. An empty UNAVAILABLE is not a defect.
Module import performs no source discovery, raw/derived I/O or native work.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import time

from backend import vendor_unfold_worker_v3 as worker
from backend.vendor_unfold_contract_v3 import canonical_candidates, strict_json, validate_preview
from scripts import run_vendor_unfold_runtime_smoke_v2 as helpers


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT.parent / "source-assets/02_CAD_QC_3D"  # No resolve/stat/open here.
FIXTURE_ROOT = ROOT / "sites/vendor-tagging-demo/public/vendor-corpus"
APPROVED = (
    ("37361005", "37361005_01.stp", "81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e"),
    ("49001000", "49001000_A-PRELIM.stp", "d8c1042f1673258976fa2b9ef812930c1be81ea3a60ab3f6357f966f7f452fd5"),
    ("42288005", "42288005_01.stp", "7afb000cdb1f18eddb4d270c5b8c2c5854d822d67adfd3cfebc5e7ad9e126837"),
    ("38172000", "38172000_01.stp", "4d512b5c565ab0e9e4d02e73fcaafd164ae82bcd977eba69198c8b0356f5644f"),
    ("10839000", "10839000_01.stp", "ea3e4820dff568916164c79417925ead79f307fba8aff11c7b3aeda7375e0b37"),
)
OWN_SOURCES = ("scripts/run_vendor_unfold_five_part_candidate_v4.py", "tests/test_vendor_unfold_five_part_candidate_v4.py")
HELPER_SOURCE = "scripts/run_vendor_unfold_runtime_smoke_v2.py"
APPROVAL_SOURCE = "scripts/build_vendor_unfold_demo.py"
FE_SOURCES = ("frontend/react/src/features/part-catalog/pages/vendorUnfoldPreview.ts",
              "frontend/react/src/features/part-catalog/pages/vendorUnfoldFiniteWidth.ts")
LIMITS = {"per_wrapper_call": dict(worker.LIMITS), "case_limit": 5, "total_wall_seconds": 420.,
          "fixture_bytes_per_case": 40_000_000, "total_fixture_bytes": 200_000_000,
          "retained_input_bytes_per_case": 40_000_000, "total_retained_input_bytes": 200_000_000,
          "receipt_bytes_per_case": 40_000_000, "total_receipt_bytes": 200_000_000,
          "proposal_bytes_per_case": 80_000_000, "total_proposal_bytes": 400_000_000,
          "protocol_bytes": 1_900_000, "ledger_bytes_per_event": 2_000_000,
          "total_ledger_bytes": 14_000_000, "source_archive_bytes": helpers.MAX_ARCHIVE_BYTES,
          "wrapper_admission": "IMMEDIATE_REMAINING_TIME_AT_LEAST_ONE_FULL_CALL"}
POST_SCHEMA = "SHERMAN_VENDOR_UNFOLD_V3_RUNTIME_POST_V1"
POST_DISPOSITION = "ACCEPTED_CONTROLLED_SYNTHETIC_V4_RUNTIME_WITH_SCOPE_LIMITATIONS"
POST_SCOPE = "FRESH_TWO_CASE_SELECTED_SKIN_RUNTIME_ONLY"
PRE_SCHEMA = "SHERMAN_VENDOR_UNFOLD_FIVE_PART_V4_PRE_V1"
QUALIFICATION_SCOPE = "CONSUMED_FIVE_PART_EXPLORATORY_SELECTED_SKIN_SUPPORT_ONLY"


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _pin_sets():
    tree = ast.parse((ROOT/APPROVAL_SOURCE).read_text(encoding="utf-8"))
    rows = next((ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                 and target.id == "SOURCES" for target in node.targets)), None)
    if rows != APPROVED:
        raise ValueError("EXACT_FIVE_APPROVED_SOURCE_CONSTANTS_REQUIRED")
    runtime = worker._source_pins()
    pins = dict(runtime)
    for name in (*OWN_SOURCES, HELPER_SOURCE, APPROVAL_SOURCE, *FE_SOURCES):
        path = (ROOT/name).resolve()
        digest = helpers._hash(path.read_bytes())
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError("PHYSICAL_SOURCE_PIN_UNION_CONFLICT")
        pins[str(path)] = digest
    return pins, runtime


def _qualification_post(path, expected, runtime_digest):
    if not _sha(expected):
        raise ValueError("EXACT_NEW_QUALIFICATION_POST_HASH_REQUIRED")
    path = _auxiliary_path(path)
    raw = helpers._read(path, 100_000)
    if helpers._hash(raw) != expected:
        raise ValueError("QUALIFICATION_POST_HASH_MISMATCH")
    post = strict_json(raw)
    keys = {"schema", "disposition", "author", "reviewer", "blocking_findings",
            "evaluation_freeze_sha256", "evaluation_record_sha256", "wrapper_source_bundle_sha256",
            "source_archive_sha256", "case_count", "checks_passed", "qualification_scope"}
    if (type(post) is not dict or set(post) != keys or post["schema"] != POST_SCHEMA
            or post["disposition"] != POST_DISPOSITION or post["qualification_scope"] != POST_SCOPE
            or post["blocking_findings"] != [] or type(post["author"]) is not str
            or not post["author"].strip() or type(post["reviewer"]) is not str
            or not post["reviewer"].strip() or post["author"] == post["reviewer"]
            or type(post["case_count"]) is not int or post["case_count"] != 2
            or type(post["checks_passed"]) is not int or post["checks_passed"] != 6
            or not all(_sha(post[key]) for key in ("evaluation_freeze_sha256", "evaluation_record_sha256",
                       "wrapper_source_bundle_sha256", "source_archive_sha256"))
            or post["wrapper_source_bundle_sha256"] != runtime_digest):
        raise ValueError("CURRENT_V4_NATIVE_RUNTIME_POST_REQUIRED")
    # The independently reviewed POST binds its evaluation receipts. This
    # runner does not reopen them or imply transitive/software sandbox closure.
    return {"path": str(path), "sha256": expected, "receipt": post,
            "identity_declared_not_cryptographically_verified": True}


def _fixture(number, expected_hash, *, include_raw=False):
    if (number, expected_hash) not in {(part, digest) for part, _, digest in APPROVED}:
        raise ValueError("ONLY_EXACT_APPROVED_RELEASED_FIXTURE_READS_ALLOWED")
    path = FIXTURE_ROOT/(number+".json")
    raw = helpers._read(path, LIMITS["fixture_bytes_per_case"])
    fixture = strict_json(raw)
    if (type(fixture) is not dict or fixture.get("part_number") != number
            or fixture.get("cad_sha256") != expected_hash):
        raise ValueError("RELEASED_FIXTURE_IDENTITY_MISMATCH")
    candidates = canonical_candidates(fixture.get("candidates"))
    baseline = fixture.get("unfold_preview")
    if (type(baseline) is not dict or baseline.get("status") not in {"AVAILABLE", "UNAVAILABLE"}
            or baseline.get("model") != "SELECTED_CAD_SKIN_DEVELOPMENT"
            or baseline.get("units") != "assumed_mm" or baseline.get("units_verified") is not False
            or baseline.get("manufacturing_ready") is not False
            or baseline.get("source_sha256") not in (None, expected_hash)
            or type(baseline.get("bend_lines")) is not list or type(baseline.get("triangles")) is not list):
        raise ValueError("RELEASED_BASELINE_DISPOSITION_REQUIRED")
    if baseline["status"] == "AVAILABLE":
        if baseline.get("reason") is not None or baseline.get("source_sha256") != expected_hash:
            raise ValueError("INVALID_RELEASED_AVAILABLE_BASELINE")
    elif (type(baseline.get("reason")) is not str or not 0 < len(baseline["reason"]) <= 200
          or any(baseline.get(key) != [] for key in ("vertices", "triangles", "bend_lines", "flange_surfaces"))):
        raise ValueError("INVALID_RELEASED_UNAVAILABLE_BASELINE")
    metadata = {"fixture_path": str(path), "fixture_sha256": helpers._hash(raw),
        "fixture_bytes": len(raw), "candidates": candidates,
        "candidate_sha256": helpers._hash(helpers._json(candidates)),
        "baseline": {"preview_sha256": helpers._hash(helpers._json(baseline)),
            "status": baseline["status"], "reason": baseline.get("reason"),
            "bend_count": len(baseline["bend_lines"]) if baseline["status"] == "AVAILABLE" else 0,
            "triangle_count": len(baseline["triangles"])//3 if baseline["status"] == "AVAILABLE" else 0}}
    return (fixture, metadata, raw) if include_raw else (fixture, metadata)


def _auxiliary_path(path):
    # No symlink traversal: a declared output alias must never make a freeze
    # inspect raw CAD or an execution write into the released fixture tree.
    attempt = Path(os.path.abspath(path))
    if attempt.is_relative_to(FIXTURE_ROOT) or attempt.is_relative_to(RAW_ROOT):
        raise ValueError("SEPARATE_NEW_ATTEMPT_DIRECTORY_REQUIRED")
    if any(parent.is_symlink() for parent in (attempt, *attempt.parents)):
        raise ValueError("SYMLINK_ATTEMPT_DIRECTORY_REFUSED")
    return attempt


def _attempt_directory(path):
    return _auxiliary_path(path)


def freeze(attempt_directory, qualification_post, qualification_post_sha256):
    attempt = _attempt_directory(attempt_directory)
    pins, runtime = _pin_sets()
    digest = helpers._hash(helpers._json(runtime))
    post = _qualification_post(qualification_post, qualification_post_sha256, digest)
    cases, total = [], 0
    for number, filename, expected_hash in APPROVED:
        _, metadata = _fixture(number, expected_hash)
        total += metadata["fixture_bytes"]
        cases.append({"part_number": number, "raw_filename": filename, "raw_sha256": expected_hash,
                      **metadata})
    if total > LIMITS["total_fixture_bytes"]:
        raise ValueError("RELEASED_FIXTURE_TOTAL_BYTE_BUDGET_EXCEEDED")
    frozen = {"schema": "SHERMAN_VENDOR_UNFOLD_FIVE_PART_V4_FREEZE_V1", "source_pins": pins,
        "wrapper_source_pins": runtime, "wrapper_source_bundle_sha256": digest,
        "frontend_source_identities": {name: pins[str((ROOT/name).resolve())] for name in FE_SOURCES},
        "frontend_validator_executed": False, "qualification_post": post,
        "attempt_directory": str(attempt), "results_directory": str(attempt/"candidate-results"),
        "raw_root": str(RAW_ROOT), "released_fixture_root": str(FIXTURE_ROOT),
        "limits": copy.deepcopy(LIMITS), "cases": cases, "selection": "EXACT_ALL_FIVE_APPROVED_PARTS",
        "qualification_scope": QUALIFICATION_SCOPE, "physical_inputs_read_during_freeze": False,
        "derived_input_retention": "EXACT_ALL_FIVE_BYTES_EXCLUSIVE_AFTER_PRE_BEFORE_ANY_WRAPPER_CALL",
        "manufacturing_ready": False, "site_or_default_activation_authorized": False,
        "accuracy_claim": "NONE;AVAILABLE_COUNTS_ARE_NOT_ACCURACY", "expected_gold": None}
    if len(helpers._json(frozen))+1 > LIMITS["protocol_bytes"]:
        raise ValueError("PHYSICAL_FREEZE_PROTOCOL_BYTE_BUDGET_EXCEEDED")
    return frozen


def _review(frozen, review_path, archive_path):
    raw = helpers._read(_auxiliary_path(review_path), 100_000)
    review = strict_json(raw)
    keys = {"schema", "disposition", "author", "reviewer", "blocking_findings",
            "freeze_sha256", "source_bundle_sha256", "source_archive_sha256"}
    if (type(review) is not dict or set(review) != keys or review["schema"] != PRE_SCHEMA
            or review["disposition"] != "GREEN" or review["blocking_findings"] != []
            or type(review["author"]) is not str or not review["author"].strip()
            or type(review["reviewer"]) is not str or not review["reviewer"].strip()
            or review["author"] == review["reviewer"]
            or review["freeze_sha256"] != helpers._hash(helpers._json(frozen))
            or review["source_bundle_sha256"] != helpers._hash(helpers._json(frozen["source_pins"]))):
        raise ValueError("EXACT_INDEPENDENT_FIVE_PART_PRE_REQUIRED")
    archive_hash = helpers._verify_archive(_auxiliary_path(archive_path), frozen["source_pins"])
    if review["source_archive_sha256"] != archive_hash:
        raise ValueError("FIVE_PART_PRE_ARCHIVE_BINDING_MISMATCH")
    return {"review_sha256": helpers._hash(raw), "source_archive_sha256": archive_hash,
            "identity_declared_not_cryptographically_verified": True}


def _receipt_disposition(case, payload, runtime_digest):
    validate_preview(payload, case["raw_sha256"], case["candidates"])
    diag = payload["diagnostics"]
    native_fields = {"v4_engine_source_sha256", "candidate_binding", "candidate_input_sha256",
                     "source_bundle_sha256", "limits", "peak_process_group_rss_kib", "elapsed_seconds"}
    native = native_fields.issubset(diag)
    if native_fields.intersection(diag) and not native:
        raise ValueError("INCOMPLETE_PHYSICAL_NATIVE_RECEIPT_BINDINGS")
    if diag.get("cache_profile") != worker.PROFILE:
        raise ValueError("PHYSICAL_V4_TRANSPORT_PROFILE_MISMATCH")
    if payload["status"] == "AVAILABLE" or native:
        bound = canonical_candidates(diag.get("candidate_binding"))
        if (not native or payload["source_sha256"] != case["raw_sha256"]
                or diag.get("cache_profile") != worker.PROFILE
                or diag.get("v4_engine_source_sha256") != worker.ENGINE_SHA256
                or helpers._json(bound) != helpers._json(case["candidates"])
                or helpers._json(diag.get("candidate_binding")) != helpers._json(bound)
                or diag.get("candidate_input_sha256") != case["candidate_sha256"]
                or diag.get("source_bundle_sha256") != runtime_digest
                or helpers._json(diag.get("limits")) != helpers._json(worker.LIMITS)
                or type(diag.get("peak_process_group_rss_kib")) is not int
                or not 0 < diag["peak_process_group_rss_kib"] <= worker.MAX_RSS_KIB
                or type(diag.get("elapsed_seconds")) not in (int, float)
                or not 0 <= diag["elapsed_seconds"] < worker.MAX_SECONDS
                or diag.get("cache_hit") is not False):
            raise ValueError("PHYSICAL_V4_TRANSPORT_BINDING_MISMATCH")
    return {"status": payload["status"], "reason": payload["reason"],
        "native_receipt_verified": native,
        "bend_count": len(payload["bend_lines"]) if payload["status"] == "AVAILABLE" else 0,
        "triangle_count": len(payload["triangles"])//3 if payload["status"] == "AVAILABLE" else 0,
        "candidate_count": len(case["candidates"]), "accuracy": None,
        "support_status": "SELECTED_SKIN_AVAILABLE" if payload["status"] == "AVAILABLE" else "UNKNOWN_OR_UNAVAILABLE_NOT_A_DEFECT"}


def _capture(folder, number, payload, used, row):
    encoded = helpers._json(payload)+b"\n"
    observed = {"payload_sha256": helpers._hash(encoded), "payload_bytes": len(encoded)}
    row["observed_receipt"] = observed
    if len(encoded) > LIMITS["receipt_bytes_per_case"] or used+len(encoded) > LIMITS["total_receipt_bytes"]:
        raise ValueError("PHYSICAL_RECEIPT_BYTE_BUDGET_EXCEEDED")
    path = folder/(number+".raw-receipt.json")
    row["receipt_write_attempt"] = {"path": str(path), "status": "WRITE_STARTED"}
    helpers.save_new(path, payload)
    row["receipt_write_attempt"]["status"] = "RETAINED"
    return {**observed, "path": str(path)}, used+len(encoded)


def _append_event(stream, event):
    size = len(helpers._json(event))+1
    if size > LIMITS["ledger_bytes_per_event"] or stream.tell()+size > LIMITS["total_ledger_bytes"]:
        raise ValueError("PHYSICAL_LEDGER_BYTE_BUDGET_EXCEEDED")
    helpers._append(stream, event)


def _retain_inputs(folder, frozen, rows, ledger):
    """Persist the exact released JSON bytes, including whitespace, for failures.

    No source archive substitution: these are separate retained derived inputs.
    All five must be retained and match the freeze before any CAD worker call.
    The ledger records each successfully retained input immediately in memory;
    campaign failure/interrupt still emits every row and its snapshot truth.
    """
    for case, row in zip(frozen["cases"], rows):
        _, metadata, raw = _fixture(case["part_number"], case["raw_sha256"], include_raw=True)
        if helpers._json(metadata) != helpers._json({key: case[key] for key in metadata}):
            raise ValueError("RELEASED_FIXTURE_CHANGED_BEFORE_INPUT_RETENTION")
        if (len(raw) > LIMITS["retained_input_bytes_per_case"]
                or ledger["total_retained_input_bytes"]+len(raw) > LIMITS["total_retained_input_bytes"]):
            raise ValueError("PHYSICAL_RETAINED_INPUT_BYTE_BUDGET_EXCEEDED")
        target = folder/(case["part_number"]+".released-input.json")
        row["input_snapshot_write_attempt"] = {"path": str(target), "status": "WRITE_STARTED",
                                               "expected_sha256": case["fixture_sha256"], "expected_bytes": len(raw)}
        with target.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        row["input_snapshot"] = {"path": str(target), "sha256": helpers._hash(raw), "bytes": len(raw),
                                 "input_type": "EXACT_RELEASED_DERIVED_JSON_NOT_RAW_CAD"}
        row["input_snapshot_write_attempt"]["status"] = "RETAINED"
        ledger["total_retained_input_bytes"] += len(raw)


def execute(frozen, attempt_directory, qualification_post, qualification_post_sha256, review_path, archive_path):
    # Aggregate time includes fresh custody preparation, directory setup,
    # every call, receipt retention, assertions, and final source checks.
    started, receipt_bytes, proposal_bytes = time.monotonic(), 0, 0
    if helpers._json(frozen) != helpers._json(freeze(attempt_directory, qualification_post, qualification_post_sha256)):
        raise ValueError("FIVE_PART_PHYSICAL_FREEZE_MISMATCH")
    review = _review(frozen, review_path, archive_path)
    if worker._cache or worker._cache_bytes:
        raise ValueError("PHYSICAL_EXECUTION_REQUIRES_FRESH_PROCESS_CACHE")
    folder = Path(frozen["results_directory"])
    folder.mkdir(mode=0o700, parents=False, exist_ok=False)
    rows = [{"part_number": case["part_number"], "execution_status": "NOT_EXECUTED",
             "reason": "NOT_EXECUTED", "baseline": case["baseline"], "proposal": None,
             "input_snapshot": None, "accuracy": None} for case in frozen["cases"]]
    ledger = {"event": "TERMINAL", "status": "FAILED", "cases": rows,
              "qualification_scope": QUALIFICATION_SCOPE, "manufacturing_ready": False,
              "site_or_default_activation_authorized": False, "accuracy": None,
              "physical_inputs_read": None, "worker_source_access_attempted": False,
              "total_retained_input_bytes": 0}
    emitted = set()
    with (folder/"execution.jsonl").open("xb") as stream:
        _append_event(stream, {"event": "START", "freeze": frozen, "review": review,
                                 "started_at": datetime.now(timezone.utc).isoformat()})
        try:
            _retain_inputs(folder, frozen, rows, ledger)
            for case,row in zip(frozen["cases"], rows):
                try:
                    if _pin_sets()[0] != frozen["source_pins"]:
                        raise ValueError("SOURCE_CHANGED_DURING_PHYSICAL_CANDIDATE")
                    fixture, current = _fixture(case["part_number"], case["raw_sha256"])
                    if helpers._json(current) != helpers._json({key: case[key] for key in current}):
                        raise ValueError("RELEASED_FIXTURE_CHANGED_AFTER_FREEZE")
                    # NO raw stat/open/hash here. Check the final remaining
                    # budget after preparation, then let the bounded worker
                    # verify/copy the approved bytes under its own clock.
                    if LIMITS["total_wall_seconds"]-(time.monotonic()-started) < worker.MAX_SECONDS:
                        raise ValueError("PHYSICAL_CALL_ADMISSION_BUDGET_EXHAUSTED")
                    row.update(execution_status="RUNNING", reason="CALL_STARTED")
                    ledger["worker_source_access_attempted"] = True
                    payload = worker.build_unfold_preview_bounded(RAW_ROOT/case["raw_filename"],
                                                                  case["raw_sha256"], case["candidates"])
                    # Retain the actual receipt BEFORE any assertion; no source
                    # fixture or terminal result is ever rewritten.
                    row["receipt"], receipt_bytes = _capture(folder, case["part_number"], payload, receipt_bytes, row)
                    row["execution_status"] = "OBSERVED"
                    row["disposition"] = _receipt_disposition(case, payload, frozen["wrapper_source_bundle_sha256"])
                    proposed = copy.deepcopy(fixture)
                    proposed["unfold_preview"] = copy.deepcopy(payload)
                    encoded = helpers._json(proposed)+b"\n"
                    if (len(encoded) > LIMITS["proposal_bytes_per_case"]
                            or proposal_bytes+len(encoded) > LIMITS["total_proposal_bytes"]):
                        raise ValueError("PHYSICAL_PROPOSAL_BYTE_BUDGET_EXCEEDED")
                    target = folder/(case["part_number"]+".candidate-v4.json")
                    helpers.save_new(target, proposed)
                    proposal_bytes += len(encoded)
                    row.update(execution_status="COMPLETED", reason="BOUNDED_DISPOSITION_RETAINED_NO_ACCURACY_CLAIM",
                        proposal={"path": str(target), "sha256": helpers._hash(encoded), "bytes": len(encoded)})
                except Exception as error:
                    previous = row["execution_status"]
                    row.update(execution_status="FAILED" if previous in {"RUNNING", "OBSERVED"} else "NOT_EXECUTED_AFTER_FAILURE",
                               reason=str(error)[:1000], exception_type=type(error).__name__)
                _append_event(stream, {"event": "CASE", "result": row})
                emitted.add(row["part_number"])
            if _pin_sets()[0] != frozen["source_pins"]:
                raise ValueError("SOURCE_CHANGED_DURING_PHYSICAL_CANDIDATE")
            if all(row["execution_status"] == "COMPLETED" for row in rows):
                ledger["status"] = "COMPLETE_EXPLORATORY_EXECUTION"
        except BaseException as error:
            ledger.update(reason=str(error)[:1000], exception_type=type(error).__name__)
        for row in rows:
            if row["execution_status"] in {"RUNNING", "OBSERVED"}:
                row.update(execution_status="FAILED", reason="CAMPAIGN_INTERRUPTED_AFTER_CALL_STARTED")
            elif row["execution_status"] == "NOT_EXECUTED":
                row.update(execution_status="NOT_EXECUTED_AFTER_FAILURE", reason="CAMPAIGN_FAILED_BEFORE_CASE")
            if row["part_number"] not in emitted:
                _append_event(stream, {"event": "CASE", "result": row})
                emitted.add(row["part_number"])
        elapsed = time.monotonic()-started
        if elapsed >= LIMITS["total_wall_seconds"]:
            ledger.update(status="FAILED", reason="PHYSICAL_CAMPAIGN_WALL_BUDGET_EXCEEDED")
        ledger.update(elapsed_seconds=elapsed, total_receipt_bytes=receipt_bytes, total_proposal_bytes=proposal_bytes,
            wrapper_cases_completed=sum(row["execution_status"] == "COMPLETED" for row in rows),
            geometry_available_cases=sum(row.get("disposition", {}).get("status") == "AVAILABLE" for row in rows),
            native_receipt_verified_cases=sum(row.get("disposition", {}).get("native_receipt_verified") is True for row in rows),
            finished_at=datetime.now(timezone.utc).isoformat())
        if not ledger["worker_source_access_attempted"]:
            ledger["physical_inputs_read"] = False
        _append_event(stream, ledger)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "execute"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--qualification-post", type=Path, required=True)
    parser.add_argument("--qualification-post-sha256", required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        helpers.save_new(_auxiliary_path(args.protocol), freeze(args.attempt, args.qualification_post, args.qualification_post_sha256))
        return 0
    if args.review is None or args.source_archive is None:
        parser.error("new exact independent physical PRE and source archive are required")
    frozen = strict_json(helpers._read(_auxiliary_path(args.protocol), helpers.MAX_RECORD_BYTES))
    ledger = execute(frozen, args.attempt, args.qualification_post, args.qualification_post_sha256,
                     args.review, args.source_archive)
    print(helpers._json({"status": ledger["status"], "geometry_available_cases": ledger["geometry_available_cases"],
                       "accuracy": None}).decode())
    return 0 if ledger["status"] == "COMPLETE_EXPLORATORY_EXECUTION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
