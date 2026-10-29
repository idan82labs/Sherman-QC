"""Deferred two-case native finite-width benefit test, source-only until PRE.

Generation/evaluation have distinct exclusive freezes and terminal records.
The fixed inner-R2 oracle never switches to the outer skin after execution.
The intrusion negative can refuse at a native seam/coverage gate before the
finite guard: that is conservative refusal, NOT evidence that clipping ran.
No imports of OCP/NumPy/FreeCAD, data access or source hashing occur on import.
The 60/420 second aggregate clocks cover execution and complete resource
cleanup, after exact freeze/PRE/archive verification; they are not a bound
on the whole CLI entrypoint or the separately bounded archive preparation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time

from backend import vendor_unfold_worker_v2 as legacy_worker
from backend import vendor_unfold_worker_v3 as finite_worker
from backend.vendor_unfold_contract_v2 import canonical_candidates, validate_preview as validate_legacy
from backend.vendor_unfold_contract_v3 import validate_preview as validate_finite
from scripts import run_vendor_unfold_runtime_smoke_v2 as protocol

ROOT = protocol.ROOT
OWN_SOURCES = ("scripts/run_vendor_unfold_finite_width_native_v1.py",
               "tests/test_vendor_unfold_finite_width_native_v1.py")
CASE_IDS = ("outside_span_tabs", "in_span_intrusion")
NONCE_ENV = "SHERMAN_FINITE_WIDTH_NATIVE_V1"
GEN_SCHEMA = "SHERMAN_VENDOR_UNFOLD_FINITE_WIDTH_GENERATION_FREEZE_V1"
EVAL_SCHEMA = "SHERMAN_VENDOR_UNFOLD_FINITE_WIDTH_EVALUATION_FREEZE_V1"
LEGACY_PROFILE = "V3_CORE_TRANSPORT_2"
FINITE_PROFILE = finite_worker.PROFILE
NUMERICAL_ERROR = 1e-5  # Synthetic numerical oracle, NEVER a shop tolerance.
MAX_STEP_BYTES = 2_000_000
MAX_METADATA_BYTES = 100_000
MAX_RECORD_BYTES = 2_000_000
GEN_LIMITS = {"wall_seconds": 60.0, "process_group_rss_kib": 4*1024*1024,
    "combined_log_bytes": 1_000_000, "generated_case_limit": 2,
    "generated_total_step_bytes": MAX_STEP_BYTES, "generated_metadata_bytes": MAX_METADATA_BYTES,
    "omp_threads": 1}
EVAL_LIMITS = {"total_wall_seconds": 420.0, "fixed_call_limit": 3,
    "per_legacy_call": dict(legacy_worker.LIMITS), "per_finite_call": dict(finite_worker.LIMITS),
    "receipt_bytes_per_call": 5_000_000, "total_receipt_bytes": 5_000_000,
    "admission": "IMMEDIATE_REMAINING_TIME_AT_LEAST_ONE_FULL_60_SECOND_CALL"}
# Entire whitelist frozen in source. No candidate/source/hash, dependency,
# monitor/transport, time/RSS/work budgets, generic exception or retry success.
NATIVE_REFUSALS = (
    "UNSUPPORTED_NATIVE_TRIM_CURVE", "UNSUPPORTED_BEND_CYLINDER_GROUP",
    "SPLIT_OR_AMBIGUOUS_TANGENT_SEAM", "NON_STRAIGHT_TANGENT_SEAM",
    "NON_AXIAL_TANGENT_SEAM", "TANGENT_SEAM_NOT_NATIVE_TRIM_BOUNDARY",
    "UNEQUAL_BEND_SKIN_SWEEPS", "UNEQUAL_BEND_SKIN_ANGULAR_TRIMS",
    "UNEQUAL_BEND_SKIN_AXIAL_TRIMS", "BEND_REQUIRES_TWO_TANGENT_PLANAR_FLANGES",
    "DIRECT_CYLINDER_TO_CYLINDER_FORMING_EDGE", "NO_CONNECTED_SKIN_COVERS_ALL_CAD_BENDS",
    "INCOMPLETE_OPPOSITE_SKIN_GRAPH", "TWO_COMPLETE_SKIN_LAYERS_REQUIRED",
    "COPLANAR_SKIN_FACE_SPLITS_UNSUPPORTED", "UNKNOWN_PLANAR_SKIN_BOUNDARY",
    "UNKNOWN_CYLINDRICAL_SKIN_BOUNDARY", "UNKNOWN_SELECTED_SKIN_BOUNDARY",
    "UNEXPLAINED_NATIVE_MATERIAL_SURFACE", "DEVELOPED_SKIN_SELF_OVERLAP",
    "FINITE_WIDTH_PARENT:FLANGE_EXTENDS_INTO_FINITE_DEVELOPED_BEND",
    "FINITE_WIDTH_CHILD:FLANGE_EXTENDS_INTO_FINITE_DEVELOPED_BEND",
)
CHECKS = (
    {"check_id": "legacy_outside_span_control", "case_id": CASE_IDS[0], "profile": LEGACY_PROFILE,
     "expected_status": "UNAVAILABLE", "expected_reason": "PARENT_FLANGE_EXTENDS_INTO_DEVELOPED_BEND"},
    {"check_id": "v4_outside_span_positive", "case_id": CASE_IDS[0], "profile": FINITE_PROFILE,
     "expected_status": "AVAILABLE"},
    {"check_id": "v4_in_span_refusal", "case_id": CASE_IDS[1], "profile": FINITE_PROFILE,
     "expected_status": "UNAVAILABLE", "native_reason_whitelist": list(NATIVE_REFUSALS)},
)
_json, _hash, _read = protocol._json, protocol._hash, protocol._read
save_new = protocol.save_new


def _profile_sources():
    profiles = {LEGACY_PROFILE: legacy_worker._source_pins(), FINITE_PROFILE: finite_worker._source_pins()}
    union = {}
    for pins in profiles.values():
        for name, digest in pins.items():
            if name in union and union[name] != digest:
                raise ValueError("SOURCE_CHANGED_BETWEEN_PROFILES")
            union[name] = digest
    for name in (*OWN_SOURCES, *protocol.OWN_SOURCES):
        path = (ROOT/name).resolve()
        digest = _hash(path.read_bytes())
        if str(path) in union and union[str(path)] != digest:
            raise ValueError("SOURCE_CHANGED_BETWEEN_PROFILES")
        union[str(path)] = digest
    if len(union) > 2000:
        raise ValueError("DECLARED_SOURCE_PIN_BUDGET_EXCEEDED")
    return union, {name: _hash(_json(pins)) for name, pins in profiles.items()}


def _append(stream, value):
    raw = _json(value)+b"\n"
    if stream.tell()+len(raw) > MAX_RECORD_BYTES:
        raise ValueError("TERMINAL_RECORD_BYTE_BUDGET_EXCEEDED")
    stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _deadline(started, limit, reason):
    if time.monotonic()-started >= limit:
        raise ValueError(reason)


def case_definition(case_id):
    if case_id not in CASE_IDS:
        raise ValueError("UNKNOWN_SYNTHETIC_CASE")
    # (x,z) footprint; exact 1mm relief from BOTH native bend endpoints.
    footprint = [[-20.,-8.],[8.,-8.],[8.,-1.],[0.,-1.]]
    if case_id == CASE_IDS[1]:
        footprint += [[0.,10.],[8.,10.],[8.,30.],[0.,30.]]
    footprint += [[0.,41.],[8.,41.],[8.,48.],[-20.,48.]]
    radial = 2.5/math.sqrt(2.)
    oracle = ({"status": "AVAILABLE", "selected_skin": "FIXED_INNER_R2_NOT_NEUTRAL_AXIS",
        "flat_area_mm2": 1832.+40.*math.pi, "sorted_flat_bbox_mm": [35.+math.pi,56.],
        "bend_count": 1, "sweep_deg": 90., "bend_direction": "unknown", "marker_length_mm": 40.,
        "numerical_error_limit": NUMERICAL_ERROR} if case_id == CASE_IDS[0] else
        {"status": "UNAVAILABLE", "native_reason_whitelist": list(NATIVE_REFUSALS),
         "finite_guard_execution_claimed": False})
    oracle.update(units="assumed_mm", units_verified=False, manufacturing_ready=False)
    return {"case_id": case_id, "step_filename": case_id+".step",
        "candidates": canonical_candidates([{"feature_id": "synthetic:B1",
            "line_start": [radial,radial,0.], "line_end": [radial,radial,40.]}]),
        "construction": {"inner_radius":2.,"outer_radius":3.,"parent_length":20.,
            "child_length":15.,"bend_axial_span":[0.,40.],"parent_normal_span":[2.,3.],
            "parent_footprint_xz":footprint,"minimum_outside_tab_relief_mm":1.,
            "independent_fresh_base":True,"holes":0,"fuse_and_unify_analytic_only":True},
        "oracle":oracle}


def freeze_generation(directory, record_path):
    pins, bundles = _profile_sources()
    return {"schema":GEN_SCHEMA,"source_pins":pins,"runtime_source_bundles":bundles,
        "runner_path":str((ROOT/OWN_SOURCES[0]).resolve()),"directory":str(Path(directory).resolve()),
        "record_path":str(Path(record_path).resolve()),"limits":copy.deepcopy(GEN_LIMITS),
        "cases":[case_definition(name) for name in CASE_IDS],"physical_inputs_read":False,
        "manufacturing_ready":False,"qualification_scope":"SYNTHETIC_FINITE_WIDTH_BENEFIT_ONLY",
        "native_face_order_tie_risk":"FIXED_INNER_R2_ORACLE_MAY_FAIL;NO_POSTRUN_ADAPTATION"}


def _environment(folder, nonce):
    env = {"PATH":"/usr/bin:/bin","HOME":str(folder),"TMPDIR":str(folder),"PYTHONPATH":str(ROOT),
        "PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1","PYTHONPYCACHEPREFIX":str(folder/"pycache"),
        "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "VECLIB_MAXIMUM_THREADS":"1",NONCE_ENV:nonce}
    return env


def _preflight_monitor():
    try:
        rows = legacy_worker._group_rows(os.getpgrp())
        if (not any(pid == os.getpid() for pid, _ in rows)
                or any(type(pid) is not int or type(rss) is not int or pid <= 1 or rss <= 0 for pid,rss in rows)):
            raise ValueError("INVALID_PROCESS_MONITOR_ROWS")
    except Exception as exc:
        raise ValueError("PROCESS_MONITOR_PREFLIGHT_UNAVAILABLE") from exc


def _check_generation_output(directory):
    total = 0
    for path in Path(directory).iterdir():
        if path.name not in {"generated-cases.json",*(name+".step" for name in CASE_IDS)} or path.is_symlink() or not path.is_file():
            raise ValueError("UNEXPECTED_GENERATION_DIRECTORY_ENTRY")
        size = path.stat().st_size
        if path.name == "generated-cases.json":
            if size > MAX_METADATA_BYTES: raise ValueError("GENERATED_METADATA_BYTE_BUDGET_EXCEEDED")
        else: total += size
    if total > MAX_STEP_BYTES: raise ValueError("GENERATED_TOTAL_STEP_BYTE_BUDGET_EXCEEDED")


def _retain_generation_logs(folder, record_path, evidence):
    # Retain a bounded prefix of BOTH streams, not an unbounded trace. A log
    # overflow is still FAILED; truncation is explicit and never a full hash.
    retained = []
    for name in ("stdout","stderr"):
        source = folder/(name+".log")
        size = source.stat().st_size if source.exists() else 0
        if source.exists():
            with source.open("rb") as incoming:
                raw = incoming.read(GEN_LIMITS["combined_log_bytes"]//2)
        else: raw = b""
        target = Path(record_path).with_name(Path(record_path).stem+"-"+name+".log")
        with target.open("xb") as out:
            out.write(raw); out.flush(); os.fsync(out.fileno())
        retained.append({"stream":name,"path":str(target.resolve()),"retained_prefix_bytes":len(raw),
            "retained_prefix_sha256":_hash(raw),"observed_total_bytes":size,"truncated":len(raw)<size})
    evidence["retained_logs"] = retained


def _supervise_generation(folder, request, lock, started, evidence):
    process, peak = None, 0
    evidence.update(pid=None,pgid=None,process_started=False,cleanup_quiescent=False)
    try:
        with (folder/"stdout.log").open("xb") as out, (folder/"stderr.log").open("xb") as err:
            _deadline(started,60.,"GENERATION_TIME_BUDGET_EXCEEDED")
            pins,bundles = _profile_sources()
            if pins != request["freeze"]["source_pins"] or bundles != request["freeze"]["runtime_source_bundles"]:
                raise ValueError("SOURCE_CHANGED_BEFORE_GENERATION_SPAWN")
            _preflight_monitor()  # Mandatory; NEVER spawn merely to discover ps is blocked.
            evidence["process_monitor_preflight"] = True
            _deadline(started,60.,"GENERATION_TIME_BUDGET_EXCEEDED")
            legacy_worker._record(lock,{"state":"ADMITTING","pgid":None})
            _deadline(started,60.,"GENERATION_TIME_BUDGET_EXCEEDED")
            process = subprocess.Popen([sys.executable,"-X","faulthandler","-m",OWN_SOURCES[0][:-3].replace("/","."),
                "--native-generate",str(folder/"request.json")],cwd=ROOT,env=_environment(folder,request["nonce"]),
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            evidence.update(pid=process.pid,pgid=process.pid,process_started=True)
            legacy_worker._record(lock,{"state":"RUNNING","pgid":process.pid})
            while process.poll() is None:
                _deadline(started,60.,"GENERATION_TIME_BUDGET_EXCEEDED")
                rows = legacy_worker._group_rows(process.pid)
                if not rows and process.poll() is None: raise ValueError("GENERATION_RSS_MONITOR_UNAVAILABLE")
                peak = max(peak,sum(rss for _,rss in rows))
                evidence["peak_process_group_rss_kib"] = peak
                if peak > GEN_LIMITS["process_group_rss_kib"]: raise ValueError("GENERATION_RSS_BUDGET_EXCEEDED")
                if out.tell()+err.tell() > GEN_LIMITS["combined_log_bytes"]: raise ValueError("GENERATION_LOG_BUDGET_EXCEEDED")
                _check_generation_output(request["freeze"]["directory"])
                time.sleep(.1)
            evidence["returncode"] = process.returncode
            if any(pid != process.pid for pid,_ in legacy_worker._group_rows(process.pid)):
                raise ValueError("GENERATION_LIVE_DESCENDANTS_AFTER_EXIT")
            if out.tell()+err.tell() > GEN_LIMITS["combined_log_bytes"]: raise ValueError("GENERATION_LOG_BUDGET_EXCEEDED")
            _check_generation_output(request["freeze"]["directory"])
            if process.returncode != 0: raise ValueError("GENERATION_CHILD_FAILED")
            if peak <= 0: raise ValueError("GENERATION_RSS_NEVER_OBSERVED")
    finally:
        if process is not None:
            evidence["returncode_before_cleanup"] = process.returncode
            try: legacy_worker._stop_group(process)
            except BaseException:
                evidence["returncode"] = process.returncode
                legacy_worker._record(lock,{"state":"QUARANTINED","pgid":process.pid})
                raise
            evidence["returncode"] = process.returncode
        legacy_worker._record(lock,{"state":"QUIESCENT","pgid":None})
        evidence["cleanup_quiescent"] = True


def load_cases(directory):
    directory = Path(directory)
    _check_generation_output(directory)
    cases = protocol.strict_json(_read(directory/"generated-cases.json",MAX_METADATA_BYTES))
    if type(cases) is not list or len(cases) != 2: raise ValueError("COMPLETE_TWO_CASE_COHORT_REQUIRED")
    total = 0
    for row,name in zip(cases,CASE_IDS):
        expected = case_definition(name)
        if type(row) is not dict or set(row) != set(expected)|{"source_sha256"} or _json({k:row[k] for k in expected}) != _json(expected):
            raise ValueError("FROZEN_CASE_OR_ORACLE_MISMATCH")
        raw = _read(directory/expected["step_filename"],MAX_STEP_BYTES); total += len(raw)
        if _hash(raw) != row["source_sha256"]: raise ValueError("GENERATED_CASE_HASH_MISMATCH")
    if total > MAX_STEP_BYTES: raise ValueError("GENERATED_TOTAL_STEP_BYTE_BUDGET_EXCEEDED")
    if {p.name for p in directory.iterdir()} != {"generated-cases.json",*(name+".step" for name in CASE_IDS)}:
        raise ValueError("UNEXPECTED_GENERATION_DIRECTORY_ENTRY")
    return cases


def generate(frozen,directory,record_path,review_path,archive_path):
    if _json(frozen) != _json(freeze_generation(directory,record_path)): raise ValueError("GENERATION_FREEZE_MISMATCH")
    review = protocol._review(frozen,review_path,archive_path)
    started = time.monotonic()
    ledger = {"event":"TERMINAL","status":"FAILED","cases":[],"selected_case_ids":list(CASE_IDS),
        "child":{},"physical_inputs_read":False,"manufacturing_ready":False}
    with Path(record_path).open("xb") as stream:
        _append(stream,{"event":"START","freeze":frozen,"review":review,"started_at":datetime.now(timezone.utc).isoformat()})
        try:
            if not legacy_worker._slot.acquire(blocking=False): raise ValueError("GENERATION_SHARED_SLOT_BUSY")
            try:
                fd = os.open(legacy_worker.ADMISSION_LOCK_PATH,os.O_CREAT|os.O_RDWR|getattr(os,"O_NOFOLLOW",0),0o600)
                with os.fdopen(fd,"r+b") as lock:
                    info = os.fstat(lock.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid(): raise ValueError("GENERATION_LOCK_UNAVAILABLE")
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    legacy_worker._check_admission_record(lock)
                    Path(directory).mkdir(mode=0o700,parents=False,exist_ok=False)
                    with tempfile.TemporaryDirectory(prefix="finite-width-native-v1-") as temp:
                        folder = Path(temp)
                        request = {"nonce":secrets.token_hex(24),"freeze":frozen}
                        save_new(folder/"request.json",request)
                        try:
                            _supervise_generation(folder,request,lock,started,ledger["child"])
                            ledger["cases"] = load_cases(directory)
                            if _profile_sources() != (frozen["source_pins"],frozen["runtime_source_bundles"]):
                                raise ValueError("SOURCE_CHANGED_DURING_GENERATION")
                        finally:
                            # Preserve evidence even if output/source validation,
                            # temporary-directory cleanup, or later slot release
                            # fails. Publishing a log never makes the run succeed.
                            _retain_generation_logs(folder,record_path,ledger["child"])
            finally: legacy_worker._slot.release()
            # Temporary directory, flock and BOTH admission resources have now
            # completed cleanup; a late exit can never publish success.
            _deadline(started,60.,"GENERATION_TIME_BUDGET_EXCEEDED")
            ledger["status"] = "COMPLETE_SYNTHETIC_GENERATION"
        except BaseException as exc:
            ledger.update(reason=str(exc)[:1000],exception_type=type(exc).__name__)
        ledger.update(elapsed_seconds=time.monotonic()-started,finished_at=datetime.now(timezone.utc).isoformat())
        if ledger["elapsed_seconds"] >= 60.: ledger.update(status="FAILED",reason="GENERATION_TIME_BUDGET_EXCEEDED")
        _append(stream,ledger)
    return ledger


def _generation_provenance(record,directory,cases):
    raw = _read(record,MAX_RECORD_BYTES)
    events = [protocol.strict_json(line) for line in raw.splitlines()]
    if len(events) != 2 or any(type(e) is not dict for e in events): raise ValueError("COMPLETE_GENERATION_PROVENANCE_REQUIRED")
    first,last = events
    child = last.get("child",{})
    review = first.get("review")
    if (type(review) is not dict or set(review) != {"review_sha256","source_archive_sha256",
            "independent_review_identity_declared_not_cryptographically_verified"}
            or any(type(review.get(key)) is not str or re.fullmatch(r"[0-9a-f]{64}",review[key]) is None
                   for key in ("review_sha256","source_archive_sha256"))
            or review.get("independent_review_identity_declared_not_cryptographically_verified") is not True):
        raise ValueError("COMPLETE_GENERATION_PROVENANCE_REQUIRED")
    if (first.get("event") != "START" or last.get("event") != "TERMINAL" or last.get("status") != "COMPLETE_SYNTHETIC_GENERATION"
            or _json(first.get("freeze")) != _json(freeze_generation(directory,record))
            or _json(last.get("cases")) != _json(cases) or last.get("selected_case_ids") != list(CASE_IDS)
            or last.get("physical_inputs_read") is not False or last.get("manufacturing_ready") is not False
            or type(last.get("elapsed_seconds")) not in (int,float) or not 0 <= last["elapsed_seconds"] < 60.
            or type(child) is not dict or child.get("process_started") is not True or child.get("process_monitor_preflight") is not True
            or child.get("cleanup_quiescent") is not True or type(child.get("returncode")) is not int or child["returncode"] != 0
            or type(child.get("pid")) is not int or child["pid"] <= 1 or type(child.get("pgid")) is not int or child["pgid"] != child["pid"]
            or type(child.get("peak_process_group_rss_kib")) is not int or not 0 < child["peak_process_group_rss_kib"] <= GEN_LIMITS["process_group_rss_kib"]):
        raise ValueError("COMPLETE_GENERATION_PROVENANCE_REQUIRED")
    return {"record_path":str(Path(record).resolve()),"record_sha256":_hash(raw)}


def freeze_evaluation(directory,generation_record,record_path):
    cases = load_cases(directory)
    provenance = _generation_provenance(generation_record,directory,cases)
    pins,bundles = _profile_sources()
    return {"schema":EVAL_SCHEMA,"source_pins":pins,"runtime_source_bundles":bundles,
        "runner_path":str((ROOT/OWN_SOURCES[0]).resolve()),"directory":str(Path(directory).resolve()),
        "record_path":str(Path(record_path).resolve()),"generation_provenance":provenance,"cases":cases,
        "checks":copy.deepcopy(list(CHECKS)),"limits":copy.deepcopy(EVAL_LIMITS),
        "physical_inputs_read":False,"manufacturing_ready":False,"selection":"ALL_THREE_CHECKS_IN_FROZEN_ORDER",
        "qualification_scope":"SYNTHETIC_FINITE_WIDTH_BENEFIT_ONLY"}


def assert_result(check,case,result,source_bundle):
    validator = validate_legacy if check["profile"] == LEGACY_PROFILE else validate_finite
    validator(result,case["source_sha256"],case["candidates"])
    diag = result["diagnostics"]
    version = 3 if check["profile"] == LEGACY_PROFILE else 4
    transport = 2 if version == 3 else 3
    limits = legacy_worker.LIMITS if version == 3 else finite_worker.LIMITS
    if (type(diag.get("mechanism_version")) is not int or diag.get("mechanism_version") != version
            or type(diag.get("transport_version")) is not int or diag.get("transport_version") != transport
            or diag.get("source_bundle_sha256") != source_bundle or _json(diag.get("candidate_binding")) != _json(case["candidates"])
            or canonical_candidates(diag.get("candidate_binding")) != case["candidates"]
            or diag.get("candidate_input_sha256") != _hash(_json(case["candidates"])) or _json(diag.get("limits")) != _json(limits)
            or diag.get("cache_hit") is not False or type(diag.get("elapsed_seconds")) not in (int,float)
            or not 0 <= diag["elapsed_seconds"] < 60. or type(diag.get("peak_process_group_rss_kib")) is not int
            or not 0 < diag["peak_process_group_rss_kib"] <= GEN_LIMITS["process_group_rss_kib"]
            or diag.get("transitive_runtime_closure_verified") is not False or diag.get("hostile_upload_sandbox_verified") is not False
            or result.get("source_sha256") != case["source_sha256"] or result.get("status") != check["expected_status"]):
        raise ValueError("COLD_WRAPPER_RECEIPT_BINDING_MISMATCH")
    # The legacy transport binds its engine through the source bundle, not a
    # separate mandatory field. If such a field is present it cannot disagree.
    if version == 3 and "engine_source_sha256" in diag and diag["engine_source_sha256"] != legacy_worker.ENGINE_SHA256:
        raise ValueError("LEGACY_ENGINE_DECLARATION_MISMATCH")
    if version == 4 and (diag.get("cache_profile") != FINITE_PROFILE or diag.get("v4_engine_source_sha256") != finite_worker.ENGINE_SHA256):
        raise ValueError("V4_CONTROLLED_PROFILE_CLAIM_MISMATCH")
    if result["status"] == "UNAVAILABLE":
        reason = result["reason"]
        if reason != check.get("expected_reason") and reason not in check.get("native_reason_whitelist",[]):
            raise ValueError("EXPECTED_PREDECLARED_NATIVE_GEOMETRIC_REFUSAL")
        return {"native_reason":reason,"finite_guard_execution_claimed":reason.startswith("FINITE_WIDTH_")}
    points = [result["vertices"][i:i+3] for i in range(0,len(result["vertices"]),3)]
    bbox = sorted(max(p[axis] for p in points)-min(p[axis] for p in points) for axis in (0,1))
    area = 0.
    for i in range(0,len(result["triangles"]),3):
        a,b,c = [points[j] for j in result["triangles"][i:i+3]]
        area += abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2
    oracle = case["oracle"]
    if (any(abs(a-b)>NUMERICAL_ERROR for a,b in zip(bbox,oracle["sorted_flat_bbox_mm"]))
            or abs(area-oracle["flat_area_mm2"]) > NUMERICAL_ERROR or len(result["bend_lines"]) != 1):
        raise ValueError("FIXED_INNER_R2_BENEFIT_ORACLE_FAILED")
    line = result["bend_lines"][0]
    if (line["feature_id"] != "synthetic:B1" or line["bend_direction"] != "unknown"
            or abs(line["sweep_deg"]-90.)>NUMERICAL_ERROR or abs(math.dist(line["line_start"],line["line_end"])-40.)>NUMERICAL_ERROR):
        raise ValueError("FIXED_BEND_MARKER_ORACLE_FAILED")
    return {"flat_bbox_mm":bbox,"flat_area_mm2":area,"bend_count":1,"marker_length_mm":math.dist(line["line_start"],line["line_end"])}


def _call_admitted(started,check,case,path,row):
    if 420.-(time.monotonic()-started) < 60.: raise ValueError("EVALUATION_CALL_ADMISSION_BUDGET_EXHAUSTED")
    # The held legacy worker has no pre-spawn ps probe. Admission must prove
    # this monitor is available before entering EITHER wrapper, then reserve
    # the whole 60s again because the probe itself may have been slow.
    _preflight_monitor()
    if 420.-(time.monotonic()-started) < 60.: raise ValueError("EVALUATION_CALL_ADMISSION_BUDGET_EXHAUSTED")
    row["process_monitor_preflight"] = True
    worker = legacy_worker if check["profile"] == LEGACY_PROFILE else finite_worker
    row.update(status="RUNNING",reason="CALL_STARTED")
    call_started = time.monotonic()
    result = worker.build_unfold_preview_bounded(path,case["source_sha256"],case["candidates"])
    row["actual_call_elapsed_seconds"] = time.monotonic()-call_started
    return result


def evaluate(frozen,directory,record_path,generation_record,review_path,archive_path):
    if _json(frozen) != _json(freeze_evaluation(directory,generation_record,record_path)): raise ValueError("EVALUATION_FREEZE_MISMATCH")
    review = protocol._review(frozen,review_path,archive_path)
    if legacy_worker._cache or finite_worker._cache: raise ValueError("FRESH_BOTH_PROFILE_CACHES_REQUIRED")
    started,receipt_bytes = time.monotonic(),0
    rows = [{**copy.deepcopy(check),"status":"NOT_EXECUTED","passed":False,"reason":"NOT_EXECUTED"} for check in CHECKS]
    ledger = {"event":"TERMINAL","status":"FAILED","checks":rows,"physical_inputs_read":False,"manufacturing_ready":False}
    cases = {row["case_id"]:row for row in frozen["cases"]}
    with Path(record_path).open("xb") as stream:
        _append(stream,{"event":"START","freeze":frozen,"review":review,"started_at":datetime.now(timezone.utc).isoformat()})
        try:
            with tempfile.TemporaryDirectory(prefix="finite-width-evaluation-v1-") as temp:
                folder = Path(temp)
                for check,row in zip(CHECKS,rows):
                    try:
                        if _profile_sources() != (frozen["source_pins"],frozen["runtime_source_bundles"]): raise ValueError("SOURCE_CHANGED_DURING_EVALUATION")
                        case = cases[check["case_id"]]
                        raw = _read(Path(directory)/case["step_filename"],MAX_STEP_BYTES)
                        if _hash(raw) != case["source_sha256"]: raise ValueError("GENERATED_CASE_HASH_MISMATCH")
                        snapshot = folder/(check["check_id"]+".step")
                        with snapshot.open("xb") as out: out.write(raw)
                        result = _call_admitted(started,check,case,snapshot,row)
                        encoded = _json(result)+b"\n"
                        row.update(result_observed=True,observed_payload_sha256=_hash(encoded),observed_payload_bytes=len(encoded),
                                   actual_status=result.get("status") if type(result) is dict else None)
                        if len(encoded)>EVAL_LIMITS["receipt_bytes_per_call"] or receipt_bytes+len(encoded)>EVAL_LIMITS["total_receipt_bytes"]:
                            raise ValueError("EVALUATION_RECEIPT_BYTE_BUDGET_EXCEEDED")
                        target = Path(record_path).with_name(Path(record_path).stem+"-"+check["check_id"]+".json")
                        save_new(target,result); receipt_bytes += len(encoded)
                        row.update(receipt_path=str(target.resolve()),receipt_sha256=_hash(encoded),receipt_bytes=len(encoded))
                        if row["actual_call_elapsed_seconds"] >= 60.: raise ValueError("EVALUATION_WRAPPER_WALL_BUDGET_EXCEEDED")
                        _deadline(started,420.,"EVALUATION_AGGREGATE_TIME_BUDGET_EXCEEDED")
                        observed = assert_result(check,case,result,frozen["runtime_source_bundles"][check["profile"]])
                        row.update(status="PASSED",passed=True,reason="FIXED_ORACLES_PASSED",observed=observed)
                    except Exception as exc:
                        row.update(status="FAILED",reason=str(exc)[:1000],exception_type=type(exc).__name__)
                    _append(stream,{"event":"CHECK","result":row})
            if _profile_sources() != (frozen["source_pins"],frozen["runtime_source_bundles"]): raise ValueError("SOURCE_CHANGED_DURING_EVALUATION")
            _deadline(started,420.,"EVALUATION_AGGREGATE_TIME_BUDGET_EXCEEDED")
            if all(r["passed"] for r in rows): ledger["status"] = "COMPLETE_SYNTHETIC_FINITE_WIDTH_BENEFIT"
        except BaseException as exc: ledger.update(reason=str(exc)[:1000],exception_type=type(exc).__name__)
        for row in rows:
            if row["status"] == "NOT_EXECUTED": row.update(status="NOT_EXECUTED_AFTER_FAILURE",reason="CAMPAIGN_FAILED_BEFORE_CHECK")
            elif row["status"] == "RUNNING": row.update(status="FAILED",reason="CAMPAIGN_ABORTED_DURING_CALL")
        ledger.update(elapsed_seconds=time.monotonic()-started,total_receipt_bytes=receipt_bytes,finished_at=datetime.now(timezone.utc).isoformat())
        if ledger["elapsed_seconds"] >= 420.: ledger.update(status="FAILED",reason="EVALUATION_AGGREGATE_TIME_BUDGET_EXCEEDED")
        _append(stream,ledger)
    return ledger


def _generate_native(request_path):
    """ONLY this private child imports OCP; no kernel healing/approximation."""
    request = protocol.strict_json(_read(request_path,MAX_RECORD_BYTES))
    if (type(request) is not dict or set(request)!={"nonce","freeze"} or type(request["nonce"]) is not str
            or re.fullmatch(r"[0-9a-f]{48}",request["nonce"]) is None or request["nonce"]!=os.environ.get(NONCE_ENV)):
        raise ValueError("SUPERVISED_GENERATION_ENTRYPOINT_REQUIRED")
    frozen = request["freeze"]
    if type(frozen) is not dict or _json(frozen)!=_json(freeze_generation(frozen["directory"],frozen["record_path"])):
        raise ValueError("NATIVE_GENERATION_FREEZE_MISMATCH")
    folder = Path(frozen["directory"])
    if folder.is_symlink() or not folder.is_dir() or any(folder.iterdir()): raise ValueError("FRESH_EMPTY_GENERATION_DIRECTORY_REQUIRED")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_COMPOUND, TopAbs_FORWARD
    from OCP.TopoDS import TopoDS, TopoDS_Iterator
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone

    def one_solid(shape):
        # Boolean fixture construction can yield a sole direct-Solid container.
        # No extra/nested child is discarded, no face/root order is rearranged.
        if shape.ShapeType()==TopAbs_COMPOUND:
            children=[]; it=TopoDS_Iterator(shape)
            while it.More(): children.append(it.Value()); it.Next()
            if len(children)!=1 or children[0].ShapeType()!=TopAbs_SOLID or shape.Orientation()!=TopAbs_FORWARD:
                raise ValueError("SYNTHETIC_SINGLE_DIRECT_SOLID_REQUIRED")
            shape=TopoDS.Solid_s(children[0])
        if shape.ShapeType()!=TopAbs_SOLID or not BRepCheck_Analyzer(shape).IsValid(): raise ValueError("INVALID_FRESH_SYNTHETIC_SOLID")
        return shape

    def fresh_base():
        points=[(-20.,2.),(0.,2.),(2.,0.),(2.,-15.),(3.,-15.),(3.,0.),(0.,3.),(-20.,3.)]
        wire=BRepBuilderAPI_MakeWire()
        for i,start in enumerate(points):
            end=points[(i+1)%len(points)]; a,b=gp_Pnt(*start,0.),gp_Pnt(*end,0.)
            if i in (1,5):
                r=2. if i==1 else 3.; arc=GC_MakeArcOfCircle(a,gp_Pnt(r/math.sqrt(2.),r/math.sqrt(2.),0.),b)
                if not arc.IsDone(): raise ValueError("SYNTHETIC_ARC_CONSTRUCTION_FAILED")
                edge=BRepBuilderAPI_MakeEdge(arc.Value())
            else: edge=BRepBuilderAPI_MakeEdge(a,b)
            if not edge.IsDone(): raise ValueError("SYNTHETIC_EDGE_CONSTRUCTION_FAILED")
            wire.Add(edge.Edge())
        if not wire.IsDone(): raise ValueError("SYNTHETIC_WIRE_CONSTRUCTION_FAILED")
        face=BRepBuilderAPI_MakeFace(wire.Wire())
        if not face.IsDone(): raise ValueError("SYNTHETIC_FACE_CONSTRUCTION_FAILED")
        prism=BRepPrimAPI_MakePrism(face.Face(),gp_Vec(0.,0.,40.))
        if not prism.IsDone(): raise ValueError("SYNTHETIC_EXTRUSION_FAILED")
        return one_solid(prism.Shape())

    cases,total=[],0
    for case_id in CASE_IDS:
        base=fresh_base()  # Independently construct EACH base; never copy case1.
        case=case_definition(case_id); points=case["construction"]["parent_footprint_xz"]
        wire=BRepBuilderAPI_MakeWire()
        for i,(x,z) in enumerate(points):
            qx,qz=points[(i+1)%len(points)]; edge=BRepBuilderAPI_MakeEdge(gp_Pnt(x,2.,z),gp_Pnt(qx,2.,qz))
            if not edge.IsDone(): raise ValueError("SYNTHETIC_TAB_EDGE_FAILED")
            wire.Add(edge.Edge())
        if not wire.IsDone(): raise ValueError("SYNTHETIC_TAB_WIRE_FAILED")
        face=BRepBuilderAPI_MakeFace(wire.Wire())
        if not face.IsDone(): raise ValueError("SYNTHETIC_TAB_FACE_FAILED")
        prism=BRepPrimAPI_MakePrism(face.Face(),gp_Vec(0.,1.,0.))
        if not prism.IsDone(): raise ValueError("SYNTHETIC_TAB_EXTRUSION_FAILED")
        fusion=BRepAlgoAPI_Fuse(base,prism.Shape())
        if not fusion.IsDone(): raise ValueError("SYNTHETIC_ANALYTIC_FUSION_FAILED")
        fused=one_solid(fusion.Shape())
        unify=ShapeUpgrade_UnifySameDomain(fused,True,True,False); unify.Build()
        shape=one_solid(unify.Shape())
        target=folder/case["step_filename"]
        if target.exists(): raise ValueError("SYNTHETIC_STEP_ALREADY_EXISTS")
        writer=STEPControl_Writer()
        if writer.Transfer(shape,STEPControl_AsIs)!=IFSelect_RetDone or writer.Write(str(target))!=IFSelect_RetDone:
            raise ValueError("SYNTHETIC_STEP_EXPORT_FAILED")
        raw=_read(target,MAX_STEP_BYTES); total+=len(raw)
        if total>MAX_STEP_BYTES: raise ValueError("GENERATED_TOTAL_STEP_BYTE_BUDGET_EXCEEDED")
        cases.append({**case,"source_sha256":_hash(raw)})
    if _profile_sources()!=(frozen["source_pins"],frozen["runtime_source_bundles"]): raise ValueError("SOURCE_CHANGED_DURING_NATIVE_GENERATION")
    save_new(folder/"generated-cases.json",cases)


def main():
    if len(sys.argv)==3 and sys.argv[1]=="--native-generate": _generate_native(Path(sys.argv[2])); return 0
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("freeze-generation","generate","freeze-evaluation","evaluate"),required=True)
    for name in ("protocol","directory","record"): parser.add_argument("--"+name,type=Path,required=True)
    for name in ("generation-record","review","source-archive"): parser.add_argument("--"+name,type=Path)
    args=parser.parse_args()
    if args.mode=="freeze-generation": save_new(args.protocol,freeze_generation(args.directory,args.record)); return 0
    if args.mode in {"freeze-evaluation","evaluate"} and args.generation_record is None: parser.error("generation-record required")
    if args.mode=="freeze-evaluation": save_new(args.protocol,freeze_evaluation(args.directory,args.generation_record,args.record)); return 0
    if args.review is None or args.source_archive is None: parser.error("new exact independent PRE and source archive required")
    frozen=protocol.strict_json(_read(args.protocol,MAX_RECORD_BYTES))
    ledger=(generate(frozen,args.directory,args.record,args.review,args.source_archive) if args.mode=="generate" else
        evaluate(frozen,args.directory,args.record,args.generation_record,args.review,args.source_archive))
    print(_json({"status":ledger["status"]}).decode())
    return 0 if ledger["status"].startswith("COMPLETE_") else 1


if __name__=="__main__": raise SystemExit(main())
