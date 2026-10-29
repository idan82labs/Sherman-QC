"""Separate controlled transport-3 candidate for the V4 visual-skin engine.

V2 remains untouched. Share its process-group cleanup, persistent admission
journal, OS lock AND in-process slot; retain an independent byte-only cache.
There is no native import or source/CAD access at module import. Actual source
pins and the composed V3 contract are checked before supervised native work.
Private HOME/env and PGID caps are not an OS hostile-upload/network sandbox.
"""
from __future__ import annotations

from collections import OrderedDict
import fcntl
import importlib
import importlib.util
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import tempfile
import threading
import time

from backend import vendor_unfold_worker_v2 as shared
from backend.vendor_unfold_contract_v2 import canonical_candidates, strict_json, PreviewContractError


ROOT = shared.ROOT
MAX_SECONDS = 60.0
MAX_RSS_KIB = 4*1024*1024
MAX_RESULT_BYTES = 40_000_000
MAX_REQUEST_BYTES = 100_000
MAX_CAD_BYTES = 5_000_000
MAX_LOG_BYTES = 1_000_000
MAX_PIN_BYTES = 2_000_000
MAX_CACHE_BYTES = 32_000_000
MAX_CACHE_ENTRY_BYTES = 8_000_000
MAX_CACHE_ENTRIES = 16
PROFILE = "V4_CONTROLLED_TRANSPORT_3"
NONCE_ENV = "SHERMAN_UNFOLD_SUPERVISED_V3"
ENGINE_SHA256 = "01ff5236499273d28e36a75ebeb6e34c8172de2a0ee1d6b99cfb416de27d59dc"
LEGACY_ENGINE_SHA256 = "8412778cee16d97eb70454cdc76904be939125c9fcacf4159d18c49e150f8f07"
FINITE_WIDTH_SHA256 = "11d8cc1b912797a9a14b15b0c64d900566f07eaa536bacd28428142ca55615a2"
ADDON_SHA256 = "55d88a99b3a421786b8b07c93267bf501659ca9a74652fd603a183df8e385e48"
BASE_WORKER_SHA256 = "713b84499a1049c166000eea8e23e84385c1b1840234157ea578abc7874c9886"
COMMON_CONTRACT_SHA256 = "cbcd792106750a866b6fe960e6ff8046cd4c68a842b40c8acc01f551615be76f"
EXTRACTOR_SHA256 = "f089f71466e0f9bba47139c06efdf8bf651e7707fb05c78ca2195f06146b1314"
# Held source identities, not permission to execute or a runtime qualification.
CONTRACT_SHA256 = "6f6a1c869c382c5bdea70dae0d24312aa54e1a6e2cdb8407d5f2a6f30f2934d9"
LIMITS = dict(shared.LIMITS)
_slot = shared._slot
ADMISSION_LOCK_PATH = shared.ADMISSION_LOCK_PATH
_group_rows = shared._group_rows
_stop_group = shared._stop_group
_record = shared._record
_check_admission_record = shared._check_admission_record
_hash, _read, _json = shared._hash, shared._read, shared._json
_cache = OrderedDict()
_cache_lock = threading.Lock()
_cache_bytes = 0


def _deadline(started):
    if time.monotonic()-started >= MAX_SECONDS:
        raise PreviewContractError("UNFOLD_TIME_BUDGET_EXCEEDED")


def unavailable(reason, source_hash=None):
    """Typed V4 operational abstention, never a legacy V3 fallback."""
    return {"status": "UNAVAILABLE", "reason": reason, "model": "SELECTED_CAD_SKIN_DEVELOPMENT",
        "units": "assumed_mm", "units_verified": False, "manufacturing_ready": False,
        "source_sha256": source_hash, "vertices": [], "triangles": [], "bend_lines": [],
        "flange_surfaces": [], "diagnostics": {"mechanism_version": 4, "transport_version": 3,
            "cache_profile": PROFILE, "visual_only": True}}


def validate_preview(payload, expected_hash, candidates):
    """Only the PUBLIC composed contract; no mechanism-3 shadow or patch."""
    return importlib.import_module("backend.vendor_unfold_contract_v3").validate_preview(
        payload, expected_hash, candidates)


def _source_pins():
    pins = shared._source_pins()  # Same declared interpreter/OCP/NumPy runtime.
    required = {
        "backend/vendor_unfold_worker_v2.py": BASE_WORKER_SHA256,
        "backend/vendor_unfold_contract_v2.py": COMMON_CONTRACT_SHA256,
        "backend/vendor_unfold_preview.py": LEGACY_ENGINE_SHA256,
        "backend/vendor_unfold_preview_v4.py": ENGINE_SHA256,
        "backend/vendor_unfold_finite_width.py": FINITE_WIDTH_SHA256,
        "backend/vendor_unfold_finite_width_contract_v4.py": ADDON_SHA256,
        "backend/vendor_unfold_contract_v3.py": CONTRACT_SHA256,
        "scripts/extract_step_topology_bend_axes.py": EXTRACTOR_SHA256,
    }
    if type(CONTRACT_SHA256) is not str or len(CONTRACT_SHA256) != 64:
        raise PreviewContractError("UNFOLD_V3_CONTRACT_NOT_AUDITED")
    files = {ROOT/name for name in required} | {ROOT/"backend/vendor_unfold_worker_v3.py"}
    # Fraction arithmetic depends on these stdlib implementations. Decimal's
    # extension is pinned whenever inspectable; closure claims remain false.
    for name in ("fractions", "numbers", "decimal", "_pydecimal", "_decimal", "math"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise PreviewContractError("UNFOLD_FRACTION_RUNTIME_UNAVAILABLE")
        if spec.origin not in {"built-in", "frozen"}:
            files.add(Path(spec.origin))
        elif name in {"fractions", "numbers", "decimal", "_pydecimal"}:
            raise PreviewContractError("UNFOLD_FRACTION_RUNTIME_UNAVAILABLE")
    if len(set(pins) | {str(path.resolve()) for path in files}) > 2000:
        raise PreviewContractError("UNFOLD_SOURCE_BUDGET_EXCEEDED")
    for path in sorted(files):
        pins[str(path.resolve())] = _hash(path.read_bytes())
    for name, expected in required.items():
        if pins[str((ROOT/name).resolve())] != expected:
            raise PreviewContractError("UNFOLD_V4_SOURCE_NOT_AUDITED")
    return pins


def _binding(candidates):
    rows = canonical_candidates(candidates)
    return rows, _hash(_json(rows))


def _validate_transport(result, expected, candidates):
    validate_preview(result, expected, candidates)
    diag = result["diagnostics"]
    rows, digest = _binding(candidates)
    # The common unavailable path does not inspect optional binding rows. Do
    # not let Python's True == 1 / False == 0 equality admit mistyped custody.
    bound_rows = canonical_candidates(diag.get("candidate_binding"))
    if (diag.get("transport_version") != 3 or type(diag.get("transport_version")) is not int
            or diag.get("cache_profile") != PROFILE or diag.get("v4_engine_source_sha256") != ENGINE_SHA256
            or bound_rows != rows or diag.get("candidate_binding") != rows
            or diag.get("candidate_input_sha256") != digest):
        raise PreviewContractError("UNFOLD_V4_TRANSPORT_BINDING_MISMATCH")
    return result


def _cache_get(key, expected, candidates):
    with _cache_lock:
        raw = _cache.get(key)
        if raw is not None:
            _cache.move_to_end(key)
    if raw is None:
        return None
    result = _validate_transport(strict_json(raw), expected, candidates)
    result["diagnostics"]["cache_hit"] = True
    return result


def _cache_put(key, result, expected, candidates, started):
    global _cache_bytes
    if result["status"] != "AVAILABLE":
        _deadline(started)
        return
    _validate_transport(result, expected, candidates)
    raw = _json(result)
    _deadline(started)
    if len(raw) > MAX_CACHE_ENTRY_BYTES:
        return
    with _cache_lock:
        _deadline(started)
        old = _cache.pop(key, None)
        _cache_bytes -= len(old) if old is not None else 0
        _cache[key] = raw
        _cache_bytes += len(raw)
        while len(_cache) > MAX_CACHE_ENTRIES or _cache_bytes > MAX_CACHE_BYTES:
            _, value = _cache.popitem(last=False)
            _cache_bytes -= len(value)
        try:
            _deadline(started)
        except PreviewContractError:
            # A late serialization/LRU commit must not retain a refused entry.
            late = _cache.pop(key, None)
            _cache_bytes -= len(late) if late is not None else 0
            raise


def _run(folder, request, started, on_spawn=None, on_quiescent=None):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(folder), "TMPDIR": str(folder),
        "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(folder/"pycache"), "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        NONCE_ENV: request["nonce"]}
    process, peak = None, 0
    try:
        with (folder/"stdout.log").open("wb") as out, (folder/"stderr.log").open("wb") as err:
            _deadline(started)
            if _source_pins() != request["source_pins"]:
                raise PreviewContractError("UNFOLD_SOURCE_CHANGED_BEFORE_SPAWN")
            _deadline(started)
            # Prove the mandatory monitor is usable before creating a native
            # process. A monitor permission failure must not strand an
            # unobserved child and force persistent admission quarantine.
            try:
                monitor_rows = _group_rows(os.getpgrp())
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                raise PreviewContractError("UNFOLD_RSS_MONITOR_UNAVAILABLE") from exc
            if not monitor_rows:
                raise PreviewContractError("UNFOLD_RSS_MONITOR_UNAVAILABLE")
            _deadline(started)
            process = subprocess.Popen([sys.executable, "-X", "faulthandler", "-m",
                "backend.vendor_unfold_worker_v3", "--worker", str(folder/"request.json")],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
            if on_spawn is not None:
                on_spawn(process.pid)
            while process.poll() is None:
                _deadline(started)
                rows = _group_rows(process.pid)
                if not rows and process.poll() is None:
                    raise PreviewContractError("UNFOLD_RSS_MONITOR_UNAVAILABLE")
                peak = max(peak, sum(rss for _, rss in rows))
                if peak > MAX_RSS_KIB:
                    raise PreviewContractError("UNFOLD_RSS_BUDGET_EXCEEDED")
                if out.tell()+err.tell() > MAX_LOG_BYTES:
                    raise PreviewContractError("UNFOLD_LOG_BUDGET_EXCEEDED")
                time.sleep(.1)
            if any(pid != process.pid for pid, _ in _group_rows(process.pid)):
                raise PreviewContractError("UNFOLD_LIVE_DESCENDANTS_AFTER_EXIT")
            if out.tell()+err.tell() > MAX_LOG_BYTES:
                raise PreviewContractError("UNFOLD_LOG_BUDGET_EXCEEDED")
            if process.returncode != 0:
                raise PreviewContractError("UNFOLD_WORKER_FAILED")
    finally:
        # The journal becomes QUIESCENT only after successful whole-group
        # cleanup. If cleanup or journaling raises, its prior record quarantines
        # later V2/V4 work even after flock is released or the parent crashes.
        if process is not None:
            _stop_group(process)
        if on_quiescent is not None:
            on_quiescent()
    _deadline(started)
    result = _validate_transport(strict_json(_read(folder/"result.json", MAX_RESULT_BYTES)),
                                 request["expected_hash"], request["candidates"])
    if _source_pins() != request["source_pins"]:
        raise PreviewContractError("UNFOLD_SOURCE_CHANGED_DURING_JOB")
    _deadline(started)
    result["diagnostics"].update({"limits": dict(LIMITS), "peak_process_group_rss_kib": peak,
        "elapsed_seconds": time.monotonic()-started, "cache_hit": False,
        "source_bundle_sha256": _hash(_json(request["source_pins"])),
        "dependency_pin_scope": "DECLARED_V4_V3_HELPER_CONTRACT_INTERPRETER_OCP_NUMPY_FRACTION_FILES",
        "transitive_runtime_closure_verified": False, "hostile_upload_sandbox_verified": False})
    _validate_transport(result, request["expected_hash"], request["candidates"])
    _deadline(started)
    return result


def build_unfold_preview_bounded(cad_path, expected_hash, candidates):
    started = time.monotonic()
    source_hash = None
    try:
        candidates = canonical_candidates(candidates)
        if type(expected_hash) is not str or len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
            raise PreviewContractError("INVALID_EXPECTED_CAD_HASH")
        if Path(cad_path).suffix.lower() not in {".stp", ".step"}:
            raise PreviewContractError("STEP_SOURCE_REQUIRED")
        _deadline(started)
        pins = _source_pins()
        _deadline(started)
        raw = _read(cad_path, MAX_CAD_BYTES)
        source_hash = _hash(raw)
        if source_hash != expected_hash:
            raise PreviewContractError("CAD_HASH_MISMATCH")
        _deadline(started)
        key = _hash(_json({"cache_profile": PROFILE, "mechanism_version": 4, "transport_version": 3,
                          "source_sha256": source_hash, "candidates": candidates,
                          "source_pins": pins, "limits": LIMITS}))
        cached = _cache_get(key, expected_hash, candidates)
        if cached is not None:
            _deadline(started)
            return cached
        if not _slot.acquire(blocking=False):
            return unavailable("UNFOLD_WORKER_BUSY", source_hash)
        try:
            fd = os.open(ADMISSION_LOCK_PATH, os.O_CREAT|os.O_RDWR|getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "r+b") as lock:
                info = os.fstat(lock.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                    raise PreviewContractError("UNFOLD_LOCK_UNAVAILABLE")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    return unavailable("UNFOLD_WORKER_BUSY", source_hash)
                _check_admission_record(lock)
                _deadline(started)
                with tempfile.TemporaryDirectory(prefix="sherman-unfold-v3-") as name:
                    folder = Path(name)
                    (folder/"source.step").write_bytes(raw)
                    request = {"cache_profile": PROFILE, "cad_path": str(folder/"source.step"),
                        "expected_hash": expected_hash, "candidates": candidates, "nonce": secrets.token_hex(24)}
                    encoded, pin_bytes = _json(request), _json(pins)
                    if len(encoded) > MAX_REQUEST_BYTES or len(pin_bytes) > MAX_PIN_BYTES:
                        raise PreviewContractError("UNFOLD_REQUEST_BUDGET_EXCEEDED")
                    (folder/"pins.json").write_bytes(pin_bytes)
                    (folder/"request.json").write_bytes(encoded)
                    request["source_pins"] = pins
                    _deadline(started)
                    _record(lock, {"state": "ADMITTING", "pgid": None})
                    result = _run(folder, request, started,
                        lambda pgid: _record(lock, {"state": "RUNNING", "pgid": pgid}),
                        lambda: _record(lock, {"state": "QUIESCENT", "pgid": None}))
                    _validate_transport(result, expected_hash, candidates)
        finally:
            _slot.release()
        # Cleanup of the private snapshot, OS lock and admission slot is part
        # of the same job budget. Never publish/cache success from inside a
        # context whose exit can still fail or consume the remaining time.
        _deadline(started)
        result["diagnostics"]["elapsed_seconds"] = time.monotonic()-started
        _deadline(started)
        _cache_put(key, result, expected_hash, candidates, started)
        return result
    except (PreviewContractError, OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError, ImportError) as exc:
        reason = str(exc) if isinstance(exc, PreviewContractError) else "UNFOLD_WORKER_UNAVAILABLE"
        return unavailable(reason, source_hash if source_hash == expected_hash else None)


def _main(path):
    folder = Path(path).parent
    request = strict_json(_read(path, MAX_REQUEST_BYTES))
    if (type(request) is not dict or set(request) != {"cache_profile", "cad_path", "expected_hash", "candidates", "nonce"}
            or request.get("cache_profile") != PROFILE or type(request.get("nonce")) is not str
            or not request["nonce"] or request["nonce"] != os.environ.get(NONCE_ENV)):
        raise PreviewContractError("SUPERVISED_ENTRYPOINT_REQUIRED")
    if type(request["cad_path"]) is not str or request["cad_path"] != str(folder/"source.step"):
        raise PreviewContractError("PRIVATE_CAD_SNAPSHOT_REQUIRED")
    expected = strict_json(_read(folder/"pins.json", MAX_PIN_BYTES))
    pins = _source_pins()
    if pins != expected:
        raise PreviewContractError("UNFOLD_SOURCE_BUNDLE_MISMATCH")
    candidates, digest = _binding(request["candidates"])
    from backend.vendor_unfold_preview_v4 import build_unfold_preview
    result = build_unfold_preview(request["cad_path"], request["expected_hash"], candidates)
    if type(result) is not dict or type(result.get("diagnostics")) is not dict:
        raise PreviewContractError("INVALID_PREVIEW_DIAGNOSTICS")
    fields = {"transport_version": 3, "cache_profile": PROFILE,
        "v4_engine_source_sha256": pins[str((ROOT/"backend/vendor_unfold_preview_v4.py").resolve())],
        "candidate_binding": candidates, "candidate_input_sha256": digest}
    for key, value in fields.items():
        if key in result["diagnostics"] and _json(result["diagnostics"][key]) != _json(value):
            raise PreviewContractError("UNFOLD_NATIVE_TRANSPORT_CLAIM_MISMATCH")
    result["diagnostics"].update(fields)
    _validate_transport(result, request["expected_hash"], candidates)
    encoded = _json(result)
    if len(encoded) > MAX_RESULT_BYTES:
        raise PreviewContractError("UNFOLD_RESULT_BUDGET_EXCEEDED")
    (folder/"result.json").write_bytes(encoded)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Private supervised entrypoint only")
    _main(sys.argv[2])
