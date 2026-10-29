"""Controlled production-app candidate for V3 visual skin development.

Private byte snapshots, serial cross-process admission, whole-process-group
limits, strict receipts and bounded content-addressed in-process cache. No CAD,
catalog, credential or native-module access happens at import. Cutting remains
disabled; cache entries are not durable qualification records.
"""
from __future__ import annotations

from collections import OrderedDict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time

from backend.vendor_unfold_contract_v2 import (canonical_candidates, strict_json,
    unavailable, validate_preview, PreviewContractError)

ROOT = Path(__file__).resolve().parents[1]
MAX_SECONDS = 60.0
MAX_RSS_KIB = 4 * 1024 * 1024
MAX_RESULT_BYTES = 40_000_000
MAX_REQUEST_BYTES = 100_000
MAX_CAD_BYTES = 5_000_000
MAX_LOG_BYTES = 1_000_000
MAX_CACHE_BYTES = 32_000_000
MAX_CACHE_ENTRY_BYTES = 8_000_000
MAX_CACHE_ENTRIES = 16
ENGINE_SHA256 = "8412778cee16d97eb70454cdc76904be939125c9fcacf4159d18c49e150f8f07"
LIMITS = {"wall_seconds": MAX_SECONDS, "process_group_rss_kib": MAX_RSS_KIB,
    "result_bytes": MAX_RESULT_BYTES, "request_bytes": MAX_REQUEST_BYTES,
    "source_bytes": MAX_CAD_BYTES, "combined_log_bytes": MAX_LOG_BYTES, "omp_threads": 1}
_cache = OrderedDict()
_cache_lock = threading.Lock()
_cache_bytes = 0
_slot = threading.Lock()
ADMISSION_LOCK_PATH = "/tmp/sherman-vendor-unfold-v2.lock"


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, maximum):
    with Path(path).open("rb") as stream:
        raw = stream.read(maximum + 1)
    if not 0 < len(raw) <= maximum:
        raise PreviewContractError("UNFOLD_FILE_BUDGET_EXCEEDED")
    return raw


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _source_pins():
    files = {ROOT / name for name in ("backend/vendor_unfold_worker_v2.py",
        "backend/vendor_unfold_contract_v2.py", "backend/vendor_unfold_preview.py",
        "scripts/extract_step_topology_bend_axes.py")}
    files.add(Path(sys.executable).resolve())
    for package in ("OCP", "numpy"):
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise PreviewContractError("UNFOLD_DEPENDENCY_UNAVAILABLE")
        for location in spec.submodule_search_locations:
            files.update(path for path in Path(location).rglob("*")
                         if path.is_file() and path.suffix in {".py", ".so", ".dylib"})
    if not 4 < len(files) <= 2000:
        raise PreviewContractError("UNFOLD_SOURCE_BUDGET_EXCEEDED")
    pins = {str(path.resolve()): _hash(path.read_bytes()) for path in sorted(files)}
    if pins[str((ROOT / "backend/vendor_unfold_preview.py").resolve())] != ENGINE_SHA256:
        raise PreviewContractError("UNFOLD_ENGINE_NOT_AUDITED")
    return pins


def _cache_get(key, expected, candidates):
    with _cache_lock:
        raw = _cache.get(key)
        if raw is not None:
            _cache.move_to_end(key)
    if raw is None:
        return None
    # Never return an object retained by a previous caller.
    result = strict_json(raw)
    validate_preview(result, expected, candidates)
    result["diagnostics"]["cache_hit"] = True
    return result


def _cache_put(key, result, started=None):
    global _cache_bytes
    # Operational failures, abstentions and busy requests are not frozen as
    # deterministic part properties. Start with successful previews only.
    if result["status"] != "AVAILABLE":
        return
    raw = _json(result)
    if started is not None and time.monotonic() - started >= MAX_SECONDS:
        raise PreviewContractError("UNFOLD_TIME_BUDGET_EXCEEDED")
    if len(raw) > MAX_CACHE_ENTRY_BYTES:
        return
    with _cache_lock:
        if started is not None and time.monotonic() - started >= MAX_SECONDS:
            raise PreviewContractError("UNFOLD_TIME_BUDGET_EXCEEDED")
        old = _cache.pop(key, None)
        _cache_bytes -= len(old) if old is not None else 0
        _cache[key] = raw
        _cache_bytes += len(raw)
        while len(_cache) > MAX_CACHE_ENTRIES or _cache_bytes > MAX_CACHE_BYTES:
            _, value = _cache.popitem(last=False)
            _cache_bytes -= len(value)


def _group_rows(pgid):
    result = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,rss=,stat="],
        capture_output=True, text=True, timeout=2, check=True)
    rows = []
    for line in result.stdout.splitlines():
        values = line.split()
        if len(values) == 4 and int(values[1]) == pgid and not values[3].startswith("Z"):
            rows.append((int(values[0]), int(values[2])))
    return rows


def _stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)
    deadline = time.monotonic() + 3
    while _group_rows(process.pid):
        if time.monotonic() >= deadline:
            raise PreviewContractError("UNFOLD_PROCESS_GROUP_NOT_QUIESCENT")
        time.sleep(.05)


def _run(folder, request, started, on_spawn=None, on_quiescent=None):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(folder), "TMPDIR": str(folder),
        "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(folder / "pycache"), "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "SHERMAN_UNFOLD_SUPERVISED_V2": request["nonce"]}
    process, peak = None, 0
    try:
        with (folder / "stdout.log").open("wb") as out, (folder / "stderr.log").open("wb") as err:
            # Source/candidate/snapshot preparation shares the same clock.
            # Expired work must never admit a native process merely to kill it
            # at the first monitoring iteration.
            if time.monotonic() - started >= MAX_SECONDS:
                raise PreviewContractError("UNFOLD_TIME_BUDGET_EXCEEDED")
            process = subprocess.Popen([sys.executable, "-X", "faulthandler", "-m",
                "backend.vendor_unfold_worker_v2", "--worker", str(folder / "request.json")],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
            if on_spawn is not None:
                on_spawn(process.pid)
            while process.poll() is None:
                if time.monotonic() - started >= MAX_SECONDS:
                    raise PreviewContractError("UNFOLD_TIME_BUDGET_EXCEEDED")
                rows = _group_rows(process.pid)
                if not rows and process.poll() is None:
                    raise PreviewContractError("UNFOLD_RSS_MONITOR_UNAVAILABLE")
                peak = max(peak, sum(rss for _, rss in rows))
                if peak > MAX_RSS_KIB:
                    raise PreviewContractError("UNFOLD_RSS_BUDGET_EXCEEDED")
                if out.tell() + err.tell() > MAX_LOG_BYTES:
                    raise PreviewContractError("UNFOLD_LOG_BUDGET_EXCEEDED")
                time.sleep(.1)
            if any(pid != process.pid for pid, _ in _group_rows(process.pid)):
                raise PreviewContractError("UNFOLD_LIVE_DESCENDANTS_AFTER_EXIT")
            if out.tell() + err.tell() > MAX_LOG_BYTES:
                raise PreviewContractError("UNFOLD_LOG_BUDGET_EXCEEDED")
            if process.returncode != 0:
                raise PreviewContractError("UNFOLD_WORKER_FAILED")
    finally:
        if process is not None:
            _stop_group(process)
        if on_quiescent is not None:
            on_quiescent()
    if time.monotonic() - started >= MAX_SECONDS:
        raise PreviewContractError("UNFOLD_TIME_BUDGET_EXCEEDED")
    result = strict_json(_read(folder / "result.json", MAX_RESULT_BYTES))
    validate_preview(result, request["expected_hash"], request["candidates"])
    if _source_pins() != request["source_pins"]:
        raise PreviewContractError("UNFOLD_SOURCE_CHANGED_DURING_JOB")
    if time.monotonic() - started >= MAX_SECONDS:
        raise PreviewContractError("UNFOLD_TIME_BUDGET_EXCEEDED")
    result["diagnostics"].update({"transport_version": 2, "limits": dict(LIMITS),
        "peak_process_group_rss_kib": peak, "elapsed_seconds": time.monotonic() - started,
        "cache_hit": False, "source_bundle_sha256": _hash(_json(request["source_pins"])),
        "candidate_binding": canonical_candidates(request["candidates"]),
        "candidate_input_sha256": _hash(_json(canonical_candidates(request["candidates"]))),
        "dependency_pin_scope": "DECLARED_ENGINE_INTERPRETER_OCP_NUMPY_FILES",
        "transitive_runtime_closure_verified": False,
        "hostile_upload_sandbox_verified": False})
    return result


def _record(lock, state):
    """A parent crash/cleanup failure must survive release of the OS flock."""
    raw = _json(state)
    lock.seek(0)
    lock.write(raw)
    lock.truncate()
    lock.flush()
    os.fsync(lock.fileno())


def _check_admission_record(lock):
    lock.seek(0)
    raw = lock.read(1025)
    if not raw:
        return  # New lock file, no predecessor process.
    if len(raw) > 1024:
        raise PreviewContractError("UNFOLD_ADMISSION_RECORD_INVALID")
    state = strict_json(raw)
    if type(state) is not dict or set(state) != {"state", "pgid"}:
        raise PreviewContractError("UNFOLD_ADMISSION_RECORD_INVALID")
    if state == {"state": "QUIESCENT", "pgid": None}:
        return
    pgid = state.get("pgid")
    if state.get("state") != "RUNNING" or type(pgid) is not int or pgid <= 1:
        raise PreviewContractError("UNFOLD_ADMISSION_QUARANTINED")
    if _group_rows(pgid):
        raise PreviewContractError("UNFOLD_ADMISSION_QUARANTINED")
    _record(lock, {"state": "QUIESCENT", "pgid": None})


def build_unfold_preview_bounded(cad_path, expected_hash, candidates):
    started = time.monotonic()
    source_hash = None
    try:
        candidates = canonical_candidates(candidates)
        if type(expected_hash) is not str or len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
            raise PreviewContractError("INVALID_EXPECTED_CAD_HASH")
        if Path(cad_path).suffix.lower() not in {".stp", ".step"}:
            raise PreviewContractError("STEP_SOURCE_REQUIRED")
        pins = _source_pins()
        raw = _read(cad_path, MAX_CAD_BYTES)
        source_hash = _hash(raw)
        if source_hash != expected_hash:
            raise PreviewContractError("CAD_HASH_MISMATCH")
        key = _hash(_json({"source_sha256": source_hash, "candidates": candidates,
                          "source_pins": pins, "limits": LIMITS}))
        cached = _cache_get(key, expected_hash, candidates)
        if cached is not None:
            if time.monotonic() - started >= MAX_SECONDS:
                raise PreviewContractError("UNFOLD_TIME_BUDGET_EXCEEDED")
            return cached
        if not _slot.acquire(blocking=False):
            return unavailable("UNFOLD_WORKER_BUSY", source_hash)
        try:
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(ADMISSION_LOCK_PATH, flags, 0o600)
            with os.fdopen(fd, "r+b") as lock:
                info = os.fstat(lock.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                    raise PreviewContractError("UNFOLD_LOCK_UNAVAILABLE")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return unavailable("UNFOLD_WORKER_BUSY", source_hash)
                _check_admission_record(lock)
                with tempfile.TemporaryDirectory(prefix="sherman-unfold-v2-") as name:
                    folder = Path(name)
                    (folder / "source.step").write_bytes(raw)
                    request = {"cad_path": str(folder / "source.step"), "expected_hash": expected_hash,
                        "candidates": candidates, "source_pins": pins, "nonce": secrets.token_hex(24)}
                    # Source-pin maps are carried separately so request candidate
                    # growth cannot silently enlarge the original 100-kB budget.
                    pin_bytes = _json(request.pop("source_pins"))
                    encoded = _json(request)
                    if len(encoded) > MAX_REQUEST_BYTES or len(pin_bytes) > 2_000_000:
                        raise PreviewContractError("UNFOLD_REQUEST_BUDGET_EXCEEDED")
                    (folder / "pins.json").write_bytes(pin_bytes)
                    (folder / "request.json").write_bytes(encoded)
                    request["source_pins"] = pins
                    _record(lock, {"state": "ADMITTING", "pgid": None})
                    result = _run(folder, request, started,
                        lambda pgid: _record(lock, {"state": "RUNNING", "pgid": pgid}),
                        lambda: _record(lock, {"state": "QUIESCENT", "pgid": None}))
                    if time.monotonic() - started >= MAX_SECONDS:
                        raise PreviewContractError("UNFOLD_TIME_BUDGET_EXCEEDED")
                    # The last bounded publication decision is made inside the
                    # cache lock, before any insertion; no refused result sticks.
                    _cache_put(key, result, started)
                    return result
        finally:
            _slot.release()
    except (PreviewContractError, OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        reason = str(exc) if isinstance(exc, PreviewContractError) else "UNFOLD_WORKER_UNAVAILABLE"
        return unavailable(reason, source_hash if source_hash == expected_hash else None)


def _main(path):
    folder = Path(path).parent
    request = strict_json(_read(path, MAX_REQUEST_BYTES))
    if request.get("nonce") != os.environ.get("SHERMAN_UNFOLD_SUPERVISED_V2") or not request.get("nonce"):
        raise PreviewContractError("SUPERVISED_ENTRYPOINT_REQUIRED")
    expected = strict_json(_read(folder / "pins.json", 2_000_000))
    if _source_pins() != expected:
        raise PreviewContractError("UNFOLD_SOURCE_BUNDLE_MISMATCH")
    candidates = canonical_candidates(request["candidates"])
    from backend.vendor_unfold_preview import build_unfold_preview
    result = build_unfold_preview(request["cad_path"], request["expected_hash"], candidates)
    validate_preview(result, request["expected_hash"], candidates)
    encoded = _json(result)
    if len(encoded) > MAX_RESULT_BYTES:
        raise PreviewContractError("UNFOLD_RESULT_BUDGET_EXCEEDED")
    (folder / "result.json").write_bytes(encoded)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Private supervised entrypoint only")
    _main(sys.argv[2])
