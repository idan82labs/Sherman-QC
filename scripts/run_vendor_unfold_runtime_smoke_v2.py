"""Deferred V2 wrapper smoke on two freshly authored L90 solids.

Source preparation only until an independent PRE and exact source archive are
supplied. Generation and evaluation are separate freezes. This qualifies only
the visual selected-skin mechanism/runtime, never physical CAD support,
neutral-axis blanks, calibration, or manufacturing readiness. The inner-R2
oracle is fixed before execution: tied planar root areas/native STEP face
ordering may select the outer-R3 skin, which must remain an honest failure.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import tarfile
import tempfile
import time

from backend import vendor_unfold_worker_v2 as worker
from backend.vendor_unfold_contract_v2 import canonical_candidates, strict_json, validate_preview


ROOT = Path(__file__).resolve().parents[1]
OWN_SOURCES = ("scripts/run_vendor_unfold_runtime_smoke_v2.py", "tests/test_vendor_unfold_runtime_smoke_v2.py")
CASE_IDS = ("L90", "L90_independent_pose")
MAX_GENERATED_BYTES = 2_000_000
MAX_RECORD_BYTES = 2_000_000
MAX_ARCHIVE_BYTES = 1_000_000_000
MAX_ARCHIVE_MEMBER_BYTES = 512_000_000
ORACLE_ERROR = 1e-5  # Numerical synthetic oracle, not a shop tolerance.
GEN_LIMITS = {"wall_seconds": 60.0, "process_group_rss_kib": 4 * 1024 * 1024,
              "combined_log_bytes": 1_000_000, "generated_case_limit": 2,
              "generated_total_bytes": MAX_GENERATED_BYTES, "omp_threads": 1}
EVAL_LIMITS = {"per_wrapper_call": dict(worker.LIMITS), "case_limit": 2, "total_wall_seconds": 420.0,
               "receipt_bytes_per_check": 5_000_000, "total_receipt_bytes": 5_000_000,
               "wrapper_admission": "IMMEDIATE_REMAINING_TIME_AT_LEAST_ONE_FULL_CALL"}
CHECKS = ("cold", "cached_after_caller_mutation", "changed_bytes_same_expected_hash")
POSE = {"axis": [1.0, 2.0, 3.0], "angle_degrees": 71.0, "translation": [43.0, -21.0, 15.0]}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, maximum):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("SYMLINK_REFUSED")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if not 0 < len(raw) <= maximum:
        raise ValueError("FILE_BYTE_BUDGET_EXCEEDED")
    return raw


def _pins():
    return {**worker._source_pins(), **{str((ROOT / name).resolve()): _hash((ROOT / name).read_bytes())
                                      for name in OWN_SOURCES}}


def save_new(path, value):
    """Exclusive immutable protocol publication; never replace an existing file."""
    with Path(path).open("xb") as stream:
        stream.write(_json(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append(stream, event):
    stream.write(_json(event) + b"\n")
    stream.flush()
    os.fsync(stream.fileno())


def _transform_point(point):
    axis = POSE["axis"]
    norm = math.sqrt(sum(value * value for value in axis))
    x, y, z = [value / norm for value in axis]
    angle = math.radians(POSE["angle_degrees"])
    c, s, d = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    matrix = ((c+x*x*d, x*y*d-z*s, x*z*d+y*s),
              (y*x*d+z*s, c+y*y*d, y*z*d-x*s),
              (z*x*d-y*s, z*y*d+x*s, c+z*z*d))
    return [sum(row[j] * point[j] for j in range(3)) + POSE["translation"][i]
            for i, row in enumerate(matrix)]


def case_definition(case_id):
    if case_id not in CASE_IDS:
        raise ValueError("UNKNOWN_SYNTHETIC_CASE")
    radial = 2.5 / math.sqrt(2.0)
    endpoints = [[radial, radial, 0.0], [radial, radial, 40.0]]
    if case_id == CASE_IDS[1]:
        endpoints = [_transform_point(point) for point in endpoints]
    return {"case_id": case_id, "step_filename": case_id + ".step",
            "candidates": canonical_candidates([{"feature_id": "synthetic:B1",
                "line_start": endpoints[0], "line_end": endpoints[1]}]),
            "oracle": {"status": "AVAILABLE", "bend_count": 1,
                "sorted_flat_bbox_mm": sorted([40.0, 35.0 + math.pi]),
                "flat_area_mm2": 40.0 * (35.0 + math.pi), "numerical_error_limit": ORACLE_ERROR,
                "selected_skin": "FIXED_INNER_R2_NOT_NEUTRAL_AXIS",
                "units": "assumed_mm", "units_verified": False, "manufacturing_ready": False},
            "construction": {"inner_radius": 2.0, "outer_radius": 3.0,
                "parent_length": 20.0, "child_length": 15.0, "extrusion": 40.0,
                "rigid_pose": copy.deepcopy(POSE) if case_id == CASE_IDS[1] else None}}


def freeze_generation(directory, record_path):
    return {"schema": "SHERMAN_VENDOR_UNFOLD_GENERATION_FREEZE_V2", "source_pins": _pins(),
            "runner_path": str((ROOT / OWN_SOURCES[0]).resolve()), "limits": copy.deepcopy(GEN_LIMITS),
            "directory": str(Path(directory).resolve()), "record_path": str(Path(record_path).resolve()),
            "cases": [case_definition(case_id) for case_id in CASE_IDS],
            "physical_inputs_read": False, "manufacturing_ready": False,
            "qualification_scope": "SYNTHETIC_SELECTED_SKIN_RUNTIME_ONLY",
            "native_face_order_tie_risk": "FIXED_INNER_R2_ORACLE_MAY_FAIL;NO_POSTRUN_ADAPTATION"}


def _verify_archive(path, source_pins):
    """Archive members use sources/<SHA256-of-absolute-source-path> names."""
    path = Path(path)
    if path.is_symlink() or not 0 < path.stat().st_size <= MAX_ARCHIVE_BYTES:
        raise ValueError("SOURCE_ARCHIVE_BUDGET_OR_PATH_INVALID")
    started, total = time.monotonic(), 0
    expected = {"sources/" + _hash(name.encode()): digest for name, digest in source_pins.items()}
    seen = set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if (not member.isfile() or member.name not in expected or member.name in seen
                    or not 0 <= member.size <= MAX_ARCHIVE_MEMBER_BYTES):
                raise ValueError("SOURCE_ARCHIVE_MEMBER_INVALID")
            seen.add(member.name)
            total += member.size
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("SOURCE_ARCHIVE_BYTE_BUDGET_EXCEEDED")
            digest = hashlib.sha256()
            with archive.extractfile(member) as source:
                while chunk := source.read(1_000_000):
                    digest.update(chunk)
                    if time.monotonic() - started > 60:
                        raise ValueError("SOURCE_ARCHIVE_TIME_BUDGET_EXCEEDED")
            if digest.hexdigest() != expected[member.name]:
                raise ValueError("SOURCE_ARCHIVE_PIN_MISMATCH")
    if seen != set(expected):
        raise ValueError("SOURCE_ARCHIVE_INCOMPLETE")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1_000_000):
            digest.update(chunk)
            if time.monotonic() - started > 60:
                raise ValueError("SOURCE_ARCHIVE_TIME_BUDGET_EXCEEDED")
    return digest.hexdigest()


def _review(frozen, review_path, archive_path):
    raw = _read(review_path, 100_000)
    review = strict_json(raw)
    keys = {"schema", "disposition", "author", "reviewer", "blocking_findings",
            "freeze_sha256", "source_bundle_sha256", "source_archive_sha256"}
    if (type(review) is not dict or set(review) != keys
            or review["schema"] != "SHERMAN_VENDOR_UNFOLD_RUNTIME_PRE_V2"
            or review["disposition"] != "GREEN" or review["blocking_findings"] != []
            or type(review["author"]) is not str or not review["author"].strip()
            or type(review["reviewer"]) is not str or not review["reviewer"].strip()
            or review["author"] == review["reviewer"]
            or review["freeze_sha256"] != _hash(_json(frozen))
            or review["source_bundle_sha256"] != _hash(_json(frozen["source_pins"]))):
        raise ValueError("EXACT_INDEPENDENT_PRE_REQUIRED")
    archive_hash = _verify_archive(archive_path, frozen["source_pins"])
    if review["source_archive_sha256"] != archive_hash:
        raise ValueError("PRE_ARCHIVE_BINDING_MISMATCH")
    return {"review_sha256": _hash(raw), "source_archive_sha256": archive_hash,
            "independent_review_identity_declared_not_cryptographically_verified": True}


def _environment(folder, nonce):
    return {"PATH": "/usr/bin:/bin", "HOME": str(folder), "TMPDIR": str(folder),
            "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(folder / "pycache"), "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
            "SHERMAN_VENDOR_SMOKE_GENERATION_V2": nonce}


def _check_generation_output(directory):
    allowed = {"generated-cases.json", *(name + ".step" for name in CASE_IDS)}
    total = 0
    for path in Path(directory).iterdir():
        if path.name not in allowed or path.is_symlink() or not path.is_file():
            raise ValueError("UNEXPECTED_GENERATION_DIRECTORY_ENTRY")
        size = path.stat().st_size
        if path.name == "generated-cases.json":
            if size > 100_000:
                raise ValueError("GENERATED_METADATA_BYTE_BUDGET_EXCEEDED")
        else:
            total += size
    if total > MAX_GENERATED_BYTES:
        raise ValueError("GENERATED_TOTAL_BYTE_BUDGET_EXCEEDED")


def _supervise_generation(folder, request, lock, started=None):
    started = time.monotonic() if started is None else started
    process, peak = None, 0
    worker._record(lock, {"state": "ADMITTING", "pgid": None})
    try:
        with (folder / "stdout.log").open("xb") as out, (folder / "stderr.log").open("xb") as err:
            command = [sys.executable, "-X", "faulthandler", str(ROOT / OWN_SOURCES[0]),
                       "--native-generate", str(folder / "request.json")]
            environment = _environment(folder, request["nonce"])
            # Admission/preparation is charged to the same aggregate clock;
            # never create a native child after that budget has expired.
            if time.monotonic() - started >= GEN_LIMITS["wall_seconds"]:
                raise ValueError("GENERATION_AGGREGATE_TIME_BUDGET_EXCEEDED")
            process = subprocess.Popen(command, cwd=ROOT,
                env=environment, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, start_new_session=True)
            worker._record(lock, {"state": "RUNNING", "pgid": process.pid})
            while process.poll() is None:
                if time.monotonic() - started >= GEN_LIMITS["wall_seconds"]:
                    raise ValueError("GENERATION_TIME_BUDGET_EXCEEDED")
                rows = worker._group_rows(process.pid)
                if not rows and process.poll() is None:
                    raise ValueError("GENERATION_RSS_MONITOR_UNAVAILABLE")
                peak = max(peak, sum(rss for _, rss in rows))
                if peak > GEN_LIMITS["process_group_rss_kib"]:
                    raise ValueError("GENERATION_RSS_BUDGET_EXCEEDED")
                if out.tell() + err.tell() > GEN_LIMITS["combined_log_bytes"]:
                    raise ValueError("GENERATION_LOG_BUDGET_EXCEEDED")
                _check_generation_output(request["freeze"]["directory"])
                time.sleep(.1)
            if any(pid != process.pid for pid, _ in worker._group_rows(process.pid)):
                raise ValueError("GENERATION_LIVE_DESCENDANTS_AFTER_EXIT")
            if out.tell() + err.tell() > GEN_LIMITS["combined_log_bytes"]:
                raise ValueError("GENERATION_LOG_BUDGET_EXCEEDED")
            _check_generation_output(request["freeze"]["directory"])
            if process.returncode != 0:
                raise ValueError("GENERATION_CHILD_FAILED")
            if time.monotonic() - started >= GEN_LIMITS["wall_seconds"]:
                raise ValueError("GENERATION_TIME_BUDGET_EXCEEDED")
            return {"returncode": process.returncode, "peak_process_group_rss_kib": peak,
                    "elapsed_seconds": time.monotonic() - started}
    finally:
        if process is not None:
            try:
                worker._stop_group(process)
            except BaseException:
                worker._record(lock, {"state": "QUARANTINED", "pgid": process.pid})
                raise
        worker._record(lock, {"state": "QUIESCENT", "pgid": None})


def load_cases(directory):
    directory = Path(directory)
    cases = strict_json(_read(directory / "generated-cases.json", 100_000))
    if type(cases) is not list or len(cases) != len(CASE_IDS):
        raise ValueError("COMPLETE_TWO_CASE_COHORT_REQUIRED")
    total = 0
    for case, case_id in zip(cases, CASE_IDS):
        expected = case_definition(case_id)
        if (type(case) is not dict or set(case) != set(expected) | {"source_sha256"}
                or _json({key: case[key] for key in expected}) != _json(expected)):
            raise ValueError("FROZEN_CASE_OR_ORACLE_MISMATCH")
        raw = _read(directory / expected["step_filename"], MAX_GENERATED_BYTES)
        total += len(raw)
        if _hash(raw) != case["source_sha256"]:
            raise ValueError("GENERATED_CASE_HASH_MISMATCH")
    if total > MAX_GENERATED_BYTES:
        raise ValueError("GENERATED_TOTAL_BYTE_BUDGET_EXCEEDED")
    if {path.name for path in directory.iterdir()} != {"generated-cases.json", *(name + ".step" for name in CASE_IDS)}:
        raise ValueError("UNEXPECTED_GENERATION_DIRECTORY_ENTRY")
    return cases


def generate(frozen, directory, record_path, review_path, archive_path):
    if _json(frozen) != _json(freeze_generation(directory, record_path)):
        raise ValueError("GENERATION_FREEZE_MISMATCH")
    review = _review(frozen, review_path, archive_path)
    started = time.monotonic()
    with Path(record_path).open("xb") as stream:
        _append(stream, {"event": "START", "freeze": frozen, "review": review,
                         "started_at": datetime.now(timezone.utc).isoformat()})
        ledger = {"event": "TERMINAL", "status": "FAILED", "freeze": frozen,
                  "cases": [], "physical_inputs_read": False, "manufacturing_ready": False}
        try:
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(worker.ADMISSION_LOCK_PATH, flags, 0o600)
            with os.fdopen(fd, "r+b") as lock:
                info = os.fstat(lock.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                    raise ValueError("GENERATION_LOCK_UNAVAILABLE")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                worker._check_admission_record(lock)
                directory = Path(directory)
                directory.mkdir(mode=0o700, parents=False, exist_ok=False)
                with tempfile.TemporaryDirectory(prefix="vendor-smoke-generation-v2-") as temp:
                    folder = Path(temp)
                    request = {"nonce": secrets.token_hex(24), "freeze": frozen}
                    save_new(folder / "request.json", request)
                    ledger.update(_supervise_generation(folder, request, lock, started))
                if time.monotonic() - started >= GEN_LIMITS["wall_seconds"]:
                    raise ValueError("GENERATION_AGGREGATE_TIME_BUDGET_EXCEEDED")
                ledger["cases"] = load_cases(directory)
                if _pins() != frozen["source_pins"]:
                    raise ValueError("SOURCE_CHANGED_DURING_GENERATION")
                if time.monotonic() - started >= GEN_LIMITS["wall_seconds"]:
                    raise ValueError("GENERATION_AGGREGATE_TIME_BUDGET_EXCEEDED")
                ledger["status"] = "COMPLETE_SYNTHETIC_GENERATION"
        except BaseException as error:
            ledger.update(reason=str(error)[:1000], exception_type=type(error).__name__)
        ledger["elapsed_seconds"] = time.monotonic() - started
        if ledger["elapsed_seconds"] >= GEN_LIMITS["wall_seconds"]:
            ledger.update(status="FAILED", reason="GENERATION_AGGREGATE_TIME_BUDGET_EXCEEDED")
        ledger["finished_at"] = datetime.now(timezone.utc).isoformat()
        _append(stream, ledger)
    return ledger


def _generation_provenance(record_path, directory, cases):
    raw = _read(record_path, MAX_RECORD_BYTES)
    events = [strict_json(line) for line in raw.splitlines()]
    if (len(events) != 2 or not all(type(event) is dict for event in events)
            or events[0].get("event") != "START"
            or events[1].get("event") != "TERMINAL"
            or events[1].get("status") != "COMPLETE_SYNTHETIC_GENERATION"
            or _json(events[1].get("cases")) != _json(cases)
            or type(events[1].get("returncode")) is not int or events[1]["returncode"] != 0
            or events[1].get("physical_inputs_read") is not False
            or events[1].get("manufacturing_ready") is not False
            or type(events[1].get("freeze")) is not dict
            or _json(events[0].get("freeze")) != _json(events[1]["freeze"])
            or _json(events[1]["freeze"]) != _json(freeze_generation(directory, record_path))
            or type(events[1].get("elapsed_seconds")) not in (int, float)
            or not 0 <= events[1]["elapsed_seconds"] < GEN_LIMITS["wall_seconds"]
            or type(events[1].get("peak_process_group_rss_kib")) is not int
            or not 0 < events[1]["peak_process_group_rss_kib"] <= GEN_LIMITS["process_group_rss_kib"]):
        raise ValueError("COMPLETE_BOUND_GENERATION_PROVENANCE_REQUIRED")
    return {"record_path": str(Path(record_path).resolve()), "record_sha256": _hash(raw)}


def freeze_evaluation(directory, generation_record, record_path):
    cases = load_cases(directory)
    provenance = _generation_provenance(generation_record, directory, cases)
    pins = _pins()
    runtime_pins = {key: value for key, value in pins.items()
                    if key not in {str((ROOT / name).resolve()) for name in OWN_SOURCES}}
    return {"schema": "SHERMAN_VENDOR_UNFOLD_EVALUATION_FREEZE_V2", "source_pins": pins,
            "runner_path": str((ROOT / OWN_SOURCES[0]).resolve()), "limits": copy.deepcopy(EVAL_LIMITS),
            "directory": str(Path(directory).resolve()), "record_path": str(Path(record_path).resolve()),
            "generation_provenance": provenance, "cases": cases, "checks": list(CHECKS),
            "wrapper_source_bundle_sha256": _hash(_json(runtime_pins)),
            "selection": "ALL_GENERATED_CASES_IN_RETAINED_ORDER",
            "physical_inputs_read": False, "manufacturing_ready": False,
            "qualification_scope": "SYNTHETIC_SELECTED_SKIN_RUNTIME_ONLY"}


def assert_preview(case, result, wrapper_source_bundle):
    validate_preview(result, case["source_sha256"], case["candidates"])
    oracle = case["oracle"]
    if result["status"] != oracle["status"]:
        raise ValueError("EXPECTED_AVAILABLE_SELECTED_SKIN")
    diag = result["diagnostics"]
    if (diag.get("transport_version") != 2 or diag.get("source_bundle_sha256") != wrapper_source_bundle
            or diag.get("candidate_binding") != case["candidates"]
            or diag.get("candidate_input_sha256") != _hash(_json(case["candidates"]))
            or diag.get("limits") != worker.LIMITS
            or type(diag.get("elapsed_seconds")) not in (int, float)
            or not 0 <= diag["elapsed_seconds"] < worker.MAX_SECONDS
            or type(diag.get("peak_process_group_rss_kib")) is not int
            or not 0 < diag["peak_process_group_rss_kib"] <= worker.MAX_RSS_KIB
            or len(result["bend_lines"]) != oracle["bend_count"]):
        raise ValueError("WRAPPER_RECEIPT_BINDING_MISMATCH")
    points = [result["vertices"][i:i+3] for i in range(0, len(result["vertices"]), 3)]
    bbox = sorted(max(point[axis] for point in points) - min(point[axis] for point in points)
                  for axis in (0, 1))
    area = 0.0
    for index in range(0, len(result["triangles"]), 3):
        a, b, c = [points[value] for value in result["triangles"][index:index+3]]
        area += abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) / 2
    if (any(abs(a-b) > ORACLE_ERROR for a, b in zip(bbox, oracle["sorted_flat_bbox_mm"]))
            or abs(area - oracle["flat_area_mm2"]) > ORACLE_ERROR):
        raise ValueError("FIXED_INNER_R2_ANALYTICAL_ORACLE_FAILED")
    return {"flat_bbox_mm": bbox, "flat_area_mm2": area, "bend_count": len(result["bend_lines"])}


def _call_wrapper_admitted(started, check_row, source, expected_hash, candidates):
    # Recheck after pinning, snapshot I/O, serialization, assertions and other
    # preparation. A case-level reservation is not authority for a later call.
    if EVAL_LIMITS["total_wall_seconds"] - (time.monotonic() - started) < worker.MAX_SECONDS:
        raise ValueError("EVALUATION_WRAPPER_ADMISSION_BUDGET_EXHAUSTED")
    check_row.update(status="RUNNING", reason="CALL_STARTED")
    return worker.build_unfold_preview_bounded(source, expected_hash, candidates)


def evaluate(frozen, directory, record_path, generation_record, review_path, archive_path):
    if _json(frozen) != _json(freeze_evaluation(directory, generation_record, record_path)):
        raise ValueError("EVALUATION_FREEZE_MISMATCH")
    review = _review(frozen, review_path, archive_path)
    if worker._cache:
        raise ValueError("EVALUATION_REQUIRES_FRESH_PROCESS_CACHE")
    started = time.monotonic()
    retained = [{"case_id": case["case_id"], "passed": False, "reason": "NOT_EXECUTED",
                 "checks": [{"check": check, "status": "NOT_EXECUTED", "reason": "NOT_EXECUTED"}
                            for check in CHECKS]}
                for case in frozen["cases"]]
    receipt_bytes = 0
    ledger = {"event": "TERMINAL", "status": "FAILED", "cases": retained,
              "physical_inputs_read": False, "manufacturing_ready": False}
    with Path(record_path).open("xb") as stream:
        _append(stream, {"event": "START", "freeze": frozen, "review": review,
                         "started_at": datetime.now(timezone.utc).isoformat()})
        try:
            with tempfile.TemporaryDirectory(prefix="vendor-smoke-evaluation-v2-") as temp:
                folder = Path(temp)
                for case, row in zip(frozen["cases"], retained):
                    def capture(check_row, payload):
                        nonlocal receipt_bytes
                        encoded = _json(payload) + b"\n"
                        check_row.update(result_observed=True, actual_status=payload.get("status") if type(payload) is dict else None,
                                         observed_payload_sha256=_hash(encoded), observed_payload_bytes=len(encoded))
                        if (len(encoded) > EVAL_LIMITS["receipt_bytes_per_check"]
                                or receipt_bytes + len(encoded) > EVAL_LIMITS["total_receipt_bytes"]):
                            raise ValueError("EVALUATION_RECEIPT_BYTE_BUDGET_EXCEEDED")
                        target = Path(record_path).with_name(Path(record_path).stem + "-" + case["case_id"]
                                                             + "-" + check_row["check"] + ".json")
                        save_new(target, payload)
                        receipt_bytes += len(encoded)
                        check_row.update(receipt_path=str(target.resolve()), receipt_sha256=_hash(encoded),
                                         receipt_bytes=len(encoded), status="OBSERVED", reason="ASSERTIONS_PENDING")

                    try:
                        if time.monotonic() - started > EVAL_LIMITS["total_wall_seconds"] - 3*worker.MAX_SECONDS:
                            raise ValueError("EVALUATION_CAMPAIGN_WALL_BUDGET_EXCEEDED")
                        if _pins() != frozen["source_pins"]:
                            raise ValueError("SOURCE_CHANGED_DURING_EVALUATION")
                        raw = _read(Path(directory) / case["step_filename"], MAX_GENERATED_BYTES)
                        if _hash(raw) != case["source_sha256"]:
                            raise ValueError("GENERATED_CASE_HASH_MISMATCH")
                        snapshot = folder / case["step_filename"]
                        with snapshot.open("xb") as source:
                            source.write(raw)
                        cold_check, cache_check, refusal_check = row["checks"]
                        caller_cold = _call_wrapper_admitted(started, cold_check, snapshot,
                                                             case["source_sha256"], case["candidates"])
                        # Freeze a separate copy before assertions or caller-mutation
                        # probes. Mutate the actual returned caller object to test
                        # alias isolation, never the retained cold receipt/copy.
                        retained_cold = copy.deepcopy(caller_cold)
                        capture(cold_check, retained_cold)
                        observed = assert_preview(case, retained_cold, frozen["wrapper_source_bundle_sha256"])
                        if retained_cold["diagnostics"].get("cache_hit") is not False:
                            raise ValueError("COLD_CALL_DID_NOT_EXECUTE")
                        cold_check.update(status="PASSED", reason="FIXED_COLD_ORACLES_PASSED")
                        pristine = copy.deepcopy(retained_cold)
                        caller_cold["vertices"][0] += 99
                        caller_cold["diagnostics"]["limits"]["wall_seconds"] = -1
                        cached = _call_wrapper_admitted(started, cache_check, snapshot,
                                                        case["source_sha256"], case["candidates"])
                        capture(cache_check, cached)
                        assert_preview(case, cached, frozen["wrapper_source_bundle_sha256"])
                        pristine["diagnostics"]["cache_hit"] = True
                        if cached != pristine:
                            raise ValueError("CACHE_ISOLATION_OR_HIT_FAILED")
                        cache_check.update(status="PASSED", reason="CACHE_COPY_AND_FIXED_ORACLES_PASSED")
                        changed = folder / (case["case_id"] + "-changed.step")
                        with changed.open("xb") as source:
                            source.write(raw + b"\n/* authored smoke byte change */\n")
                        refusal = _call_wrapper_admitted(started, refusal_check, changed,
                                                         case["source_sha256"], case["candidates"])
                        capture(refusal_check, refusal)
                        validate_preview(refusal, case["source_sha256"], case["candidates"])
                        if refusal["status"] != "UNAVAILABLE" or refusal["reason"] != "CAD_HASH_MISMATCH":
                            raise ValueError("CHANGED_BYTE_HASH_REFUSAL_FAILED")
                        refusal_check.update(status="PASSED", reason="CHANGED_BYTE_HASH_REFUSAL_PASSED")
                        row.update(passed=True, reason="ALL_FIXED_ORACLES_PASSED", observed=observed)
                    except Exception as error:
                        row.update(reason=str(error)[:1000], exception_type=type(error).__name__)
                        for check_row in row["checks"]:
                            if check_row["status"] in {"RUNNING", "OBSERVED"}:
                                check_row.update(status="FAILED", reason=str(error)[:1000],
                                                 exception_type=type(error).__name__)
                            elif check_row["status"] == "NOT_EXECUTED":
                                check_row.update(status="NOT_EXECUTED_AFTER_FAILURE", reason="PRIOR_CASE_CHECK_OR_PRECHECK_FAILED")
                    _append(stream, {"event": "CASE", "result": row})
            if _pins() != frozen["source_pins"]:
                raise ValueError("SOURCE_CHANGED_DURING_EVALUATION")
            if time.monotonic() - started > EVAL_LIMITS["total_wall_seconds"]:
                raise ValueError("EVALUATION_CAMPAIGN_WALL_BUDGET_EXCEEDED")
            if all(row["passed"] for row in retained):
                ledger["status"] = "COMPLETE_SYNTHETIC_RUNTIME_SMOKE"
        except BaseException as error:
            ledger.update(reason=str(error)[:1000], exception_type=type(error).__name__)
        # A campaign-level failure must still retain all selected checks.
        for row in retained:
            for check_row in row["checks"]:
                if check_row["status"] == "NOT_EXECUTED":
                    check_row.update(status="NOT_EXECUTED_AFTER_FAILURE", reason="CAMPAIGN_FAILED_BEFORE_CASE")
        ledger.update(elapsed_seconds=time.monotonic() - started, total_receipt_bytes=receipt_bytes,
                      finished_at=datetime.now(timezone.utc).isoformat())
        _append(stream, ledger)
    return ledger


def _generate_native(request_path):
    """Only this private supervised child imports OCP or constructs geometry."""
    request = strict_json(_read(request_path, MAX_RECORD_BYTES))
    if (not request.get("nonce") or request["nonce"] != os.environ.get("SHERMAN_VENDOR_SMOKE_GENERATION_V2")):
        raise ValueError("SUPERVISED_GENERATION_ENTRYPOINT_REQUIRED")
    frozen = request["freeze"]
    if _json(frozen) != _json(freeze_generation(frozen["directory"], frozen["record_path"])):
        raise ValueError("NATIVE_GENERATION_FREEZE_MISMATCH")
    folder = Path(frozen["directory"])
    if folder.is_symlink() or not folder.is_dir() or any(folder.iterdir()):
        raise ValueError("FRESH_EMPTY_GENERATION_DIRECTORY_REQUIRED")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Ax1, gp_Dir
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopAbs import TopAbs_SOLID

    def fresh_shape():
        # Inner parent edge is authored first. Native roundtrip ordering is not
        # guaranteed: the frozen oracle must fail, not switch to outer R3.
        points = [(-20.,2.), (0.,2.), (2.,0.), (2.,-15.), (3.,-15.), (3.,0.), (0.,3.), (-20.,3.)]
        wire = BRepBuilderAPI_MakeWire()
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            a, b = gp_Pnt(*start, 0.), gp_Pnt(*end, 0.)
            if index in (1, 5):
                radius = 2. if index == 1 else 3.
                curve = GC_MakeArcOfCircle(a, gp_Pnt(radius/math.sqrt(2.), radius/math.sqrt(2.), 0.), b)
                if not curve.IsDone():
                    raise ValueError("SYNTHETIC_ARC_CONSTRUCTION_FAILED")
                edge = BRepBuilderAPI_MakeEdge(curve.Value())
            else:
                edge = BRepBuilderAPI_MakeEdge(a, b)
            if not edge.IsDone():
                raise ValueError("SYNTHETIC_EDGE_CONSTRUCTION_FAILED")
            wire.Add(edge.Edge())
        if not wire.IsDone():
            raise ValueError("SYNTHETIC_WIRE_CONSTRUCTION_FAILED")
        face = BRepBuilderAPI_MakeFace(wire.Wire())
        if not face.IsDone():
            raise ValueError("SYNTHETIC_FACE_CONSTRUCTION_FAILED")
        prism = BRepPrimAPI_MakePrism(face.Face(), gp_Vec(0., 0., 40.))
        if not prism.IsDone():
            raise ValueError("SYNTHETIC_EXTRUSION_FAILED")
        return prism.Shape()

    cases, total = [], 0
    for case_id in CASE_IDS:
        shape = fresh_shape()  # Independent construction, never copy the first solid.
        if case_id == CASE_IDS[1]:
            transform = gp_Trsf()
            transform.SetRotation(gp_Ax1(gp_Pnt(0.,0.,0.), gp_Dir(*POSE["axis"])), math.radians(POSE["angle_degrees"]))
            transform.SetTranslationPart(gp_Vec(*POSE["translation"]))
            shape = BRepBuilderAPI_Transform(shape, transform, True).Shape()
        if shape.ShapeType() != TopAbs_SOLID or not BRepCheck_Analyzer(shape).IsValid():
            raise ValueError("INVALID_FRESH_SYNTHETIC_SOLID")
        case = case_definition(case_id)
        target = folder / case["step_filename"]
        if target.exists():
            raise ValueError("SYNTHETIC_STEP_ALREADY_EXISTS")
        writer = STEPControl_Writer()
        if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone or writer.Write(str(target)) != IFSelect_RetDone:
            raise ValueError("SYNTHETIC_STEP_EXPORT_FAILED")
        raw = _read(target, MAX_GENERATED_BYTES)
        total += len(raw)
        if total > MAX_GENERATED_BYTES:
            raise ValueError("GENERATED_TOTAL_BYTE_BUDGET_EXCEEDED")
        cases.append({**case, "source_sha256": _hash(raw)})
    if _pins() != frozen["source_pins"]:
        raise ValueError("SOURCE_CHANGED_DURING_NATIVE_GENERATION")
    save_new(folder / "generated-cases.json", cases)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--native-generate":
        _generate_native(Path(sys.argv[2])); return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze-generation", "generate", "freeze-evaluation", "evaluate"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--generation-record", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze-generation":
        save_new(args.protocol, freeze_generation(args.directory, args.record)); return 0
    if args.mode in {"freeze-evaluation", "evaluate"} and args.generation_record is None:
        parser.error("generation-record is required for evaluation")
    if args.mode == "freeze-evaluation":
        save_new(args.protocol, freeze_evaluation(args.directory, args.generation_record, args.record)); return 0
    if args.review is None or args.source_archive is None:
        parser.error("a new exact independent PRE and source archive are required before launch")
    frozen = strict_json(_read(args.protocol, MAX_RECORD_BYTES))
    ledger = (generate(frozen, args.directory, args.record, args.review, args.source_archive)
              if args.mode == "generate" else evaluate(frozen, args.directory, args.record,
                args.generation_record, args.review, args.source_archive))
    print(json.dumps({"status": ledger["status"]}))
    return 0 if ledger["status"].startswith("COMPLETE_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
