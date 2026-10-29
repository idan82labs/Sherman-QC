"""Pure mocked transport/cache/process tests, never a native subprocess.

Temporary inputs contain authored text/JSON, not CAD. The composed validator
is mocked here to isolate supervision; its separate held suite owns geometry
admission. All tests replace dependency hashing and native process invocation.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from backend import vendor_unfold_worker_v2 as shared
from backend import vendor_unfold_worker_v4 as worker
from backend.vendor_unfold_contract_v2 import PreviewContractError


CANDIDATES = [{"feature_id": "B1", "line_start": [0, 0, 0], "line_end": [12, 0, 0]}]
HASH = "a"*64


def receipt(expected=HASH, candidates=None):
    candidates = CANDIDATES if candidates is None else candidates
    rows, digest = worker._binding(candidates)
    return {"status": "AVAILABLE", "source_sha256": expected,
        "vertices": [0, 0, 0], "diagnostics": {"mechanism_version": 5, "transport_version": 4,
        "cache_profile": worker.PROFILE, "v5_engine_source_sha256": worker.ENGINE_SHA256,
        "candidate_binding": rows, "candidate_input_sha256": digest}}


@pytest.fixture(autouse=True)
def isolated_worker(monkeypatch):
    with worker._cache_lock:
        worker._cache.clear(); worker._cache_bytes = 0
    def fake_contract(value, expected, candidates):
        assert value["diagnostics"]["mechanism_version"] == 5
        if value["source_sha256"] != expected:
            raise PreviewContractError("MOCK_SOURCE_MISMATCH")
        return value
    monkeypatch.setattr(worker, "validate_preview", fake_contract)
    yield


def prepared(monkeypatch, tmp_path):
    source = tmp_path/"authored.step"
    raw = b"authored text, NOT STEP/CAD"
    source.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    pins, calls = {"authored-runtime": "1"}, []
    monkeypatch.setattr(worker, "_source_pins", lambda: pins.copy())
    monkeypatch.setattr(worker, "ADMISSION_LOCK_PATH", str(tmp_path/"shared-admission.lock"))
    def run(folder, request, started, on_spawn=None, on_quiescent=None):
        assert Path(request["cad_path"]).read_bytes() == raw
        assert Path(request["cad_path"]) != source
        assert request["cache_profile"] == worker.PROFILE
        assert json.loads((folder/"request.json").read_bytes())["candidates"] == request["candidates"]
        calls.append(request)
        if on_quiescent: on_quiescent()
        return receipt(expected, request["candidates"])
    monkeypatch.setattr(worker, "_run", run)
    return source, expected, pins, calls


def test_shared_cross_profile_slot_and_os_admission_are_not_independent():
    assert worker._slot is shared._slot
    assert worker.ADMISSION_LOCK_PATH == shared.ADMISSION_LOCK_PATH
    assert worker._check_admission_record is shared._check_admission_record
    assert worker._record is shared._record and worker._stop_group is shared._stop_group
    assert worker._cache is not shared._cache


def test_cache_immutable_bounded_and_profile_bound(monkeypatch):
    started = worker.time.monotonic()
    monkeypatch.setattr(worker, "MAX_CACHE_ENTRIES", 2)
    worker._cache_put("one", receipt(), HASH, CANDIDATES, started)
    first = worker._cache_get("one", HASH, CANDIDATES)
    first["vertices"][0] = 900
    assert worker._cache_get("one", HASH, CANDIDATES)["vertices"][0] == 0
    worker._cache_put("two", receipt(), HASH, CANDIDATES, started)
    worker._cache_put("three", receipt(), HASH, CANDIDATES, started)
    assert worker._cache_get("one", HASH, CANDIDATES) is None
    assert worker._cache_bytes == sum(map(len, worker._cache.values()))
    monkeypatch.setattr(worker, "MAX_CACHE_BYTES", 1)
    worker._cache_put("four", receipt(), HASH, CANDIDATES, started)
    assert not worker._cache


def test_operational_unavailable_and_oversized_results_are_not_cached(monkeypatch):
    worker._cache_put("busy", worker.unavailable("BUSY"), HASH, CANDIDATES, worker.time.monotonic())
    monkeypatch.setattr(worker, "MAX_CACHE_ENTRY_BYTES", 1)
    worker._cache_put("big", receipt(), HASH, CANDIDATES, worker.time.monotonic())
    assert not worker._cache


@pytest.mark.parametrize("field,value", [("transport_version", 2), ("transport_version", True),
    ("cache_profile", "V3"), ("v5_engine_source_sha256", "b"*64),
    ("candidate_input_sha256", "b"*64), ("candidate_binding", [])])
def test_cached_transport_cannot_alias_another_profile_or_candidate(field, value):
    result = receipt(); result["diagnostics"][field] = value
    worker._cache["bad"] = worker._json(result)
    with pytest.raises(PreviewContractError):
        worker._cache_get("bad", HASH, CANDIDATES)


def test_unavailable_boolean_candidate_coordinate_is_not_numeric_custody():
    result = receipt(); result["status"] = "UNAVAILABLE"
    result["diagnostics"]["candidate_binding"][0]["line_start"][0] = False
    with pytest.raises(PreviewContractError, match="INVALID_CANDIDATE_LINE"):
        worker._validate_transport(result, HASH, CANDIDATES)


def test_exact_source_candidates_and_runtime_changes_invalidate_cache(monkeypatch, tmp_path):
    source, expected, pins, calls = prepared(monkeypatch, tmp_path)
    assert worker.build_unfold_preview_bounded(source, expected, CANDIDATES)["status"] == "AVAILABLE"
    assert worker.build_unfold_preview_bounded(source, expected, CANDIDATES)["diagnostics"]["cache_hit"] is True
    assert len(calls) == 1
    pins["authored-runtime"] = "2"
    worker.build_unfold_preview_bounded(source, expected, CANDIDATES)
    changed = [{**CANDIDATES[0], "line_end": [13, 0, 0]}]
    worker.build_unfold_preview_bounded(source, expected, changed)
    assert len(calls) == 3
    source.write_bytes(b"other authored bytes")
    refusal = worker.build_unfold_preview_bounded(source, expected, CANDIDATES)
    assert refusal["reason"] == "CAD_HASH_MISMATCH" and refusal["source_sha256"] is None
    assert len(calls) == 3


def test_v2_slot_busy_blocks_v4_without_native_job(monkeypatch, tmp_path):
    source, expected, _, calls = prepared(monkeypatch, tmp_path)
    shared._slot.acquire()
    try:
        refusal = worker.build_unfold_preview_bounded(source, expected, CANDIDATES)
    finally:
        shared._slot.release()
    assert refusal["reason"] == "UNFOLD_WORKER_BUSY" and not calls
    assert refusal["diagnostics"]["mechanism_version"] == 5


@pytest.mark.parametrize("state", [{"state": "ADMITTING", "pgid": None},
    {"state": "RUNNING", "pgid": 12345}, {"state": "RUNNING", "pgid": True},
    {"state": "INVALID", "pgid": None}])
def test_shared_persistent_quarantine_blocks_v4(monkeypatch, tmp_path, state):
    source, expected, _, calls = prepared(monkeypatch, tmp_path)
    Path(worker.ADMISSION_LOCK_PATH).write_text(json.dumps(state))
    monkeypatch.setattr(shared, "_group_rows", lambda _: [(12345, 100)])
    refusal = worker.build_unfold_preview_bounded(source, expected, CANDIDATES)
    assert "QUARANTINED" in refusal["reason"] and not calls


def test_failed_cleanup_preserves_running_journal_for_next_profile(monkeypatch, tmp_path):
    source, expected, _, calls = prepared(monkeypatch, tmp_path)
    def failed(folder, request, started, on_spawn, on_quiescent):
        on_spawn(12345)
        raise PreviewContractError("UNFOLD_PROCESS_GROUP_NOT_QUIESCENT")
    monkeypatch.setattr(worker, "_run", failed)
    assert worker.build_unfold_preview_bounded(source, expected, CANDIDATES)["reason"] == "UNFOLD_PROCESS_GROUP_NOT_QUIESCENT"
    monkeypatch.setattr(shared, "_group_rows", lambda _: [(12345, 100)])
    assert worker.build_unfold_preview_bounded(source, expected, CANDIDATES)["reason"] == "UNFOLD_ADMISSION_QUARANTINED"
    assert not calls


@pytest.mark.parametrize("mode,reason", [("rss", "RSS_BUDGET"), ("monitor", "RSS_MONITOR"),
    ("descendant", "LIVE_DESCENDANTS"), ("crash", "WORKER_FAILED"), ("expired", "TIME_BUDGET"),
    ("source", "SOURCE_CHANGED_BEFORE")])
def test_pure_process_failures_env_and_cleanup(monkeypatch, tmp_path, mode, reason):
    captured, stopped, quiescent = {}, [], []
    process = SimpleNamespace(pid=12345, returncode=1 if mode == "crash" else 0,
        poll=lambda: None if mode in {"rss", "monitor"} else 0)
    def popen(args, **kwargs):
        captured["args"] = args; captured.update(kwargs)
        return process
    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    monkeypatch.setattr(worker, "_source_pins", lambda: {"runtime": "new" if mode == "source" else "old"})
    rows = [] if mode in {"monitor", "crash"} else [(12345, 1), (12346, worker.MAX_RSS_KIB+1)] if mode == "rss" else [(12346, 1)]
    monkeypatch.setattr(worker, "_group_rows", lambda pgid: rows if pgid==12345 else [(1,1)])
    monkeypatch.setattr(worker, "_stop_group", lambda p: stopped.append(p.pid))
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "must-not-inherit")
    monkeypatch.setenv("PYTHONHOME", "must-not-inherit")
    request = {"nonce": "authored", "source_pins": {"runtime": "old"}}
    started = worker.time.monotonic()-(61 if mode == "expired" else 0)
    with pytest.raises(PreviewContractError, match=reason):
        worker._run(tmp_path, request, started, on_quiescent=lambda: quiescent.append(True))
    if mode in {"expired", "source"}:
        assert not captured and not stopped
    else:
        assert stopped == [12345]
        assert "backend.vendor_unfold_worker_v4" in captured["args"]
        env = captured["env"]
        assert worker.NONCE_ENV in env and "SHERMAN_UNFOLD_SUPERVISED_V2" not in env
        assert "DYLD_LIBRARY_PATH" not in env and "PYTHONHOME" not in env
        assert env["HOME"] == str(tmp_path) and env["OMP_NUM_THREADS"] == "1"
        assert captured["start_new_session"] is True
    assert quiescent == [True]


def test_cleanup_failure_does_not_mark_quiescent(monkeypatch, tmp_path):
    process = SimpleNamespace(pid=12345, returncode=1, poll=lambda: 0)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(worker, "_source_pins", lambda: {})
    monkeypatch.setattr(worker, "_group_rows", lambda pgid: [] if pgid==12345 else [(1,1)])
    monkeypatch.setattr(worker, "_stop_group", lambda _: (_ for _ in ()).throw(PreviewContractError("CLEANUP_FAILED")))
    quiescent = []
    with pytest.raises(PreviewContractError, match="CLEANUP_FAILED"):
        worker._run(tmp_path, {"nonce": "test", "source_pins": {}}, worker.time.monotonic(),
                    on_quiescent=lambda: quiescent.append(True))
    assert not quiescent


@pytest.mark.parametrize("kind", ["duplicate", "nonfinite", "missing", "oversized", "source", "late"])
def test_post_exit_receipt_and_publication_refusals(monkeypatch, tmp_path, kind):
    process = SimpleNamespace(pid=12345, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(worker, "_group_rows", lambda pgid: [] if pgid==12345 else [(1,1)])
    monkeypatch.setattr(worker, "_stop_group", lambda _: None)
    calls = []
    def pins():
        calls.append(True)
        return {"runtime": "new" if kind == "source" and len(calls)>1 else "old"}
    monkeypatch.setattr(worker, "_source_pins", pins)
    request = {"nonce": "test", "source_pins": {"runtime": "old"}, "expected_hash": HASH, "candidates": CANDIDATES}
    if kind != "missing":
        raw = b'{"a":1,"a":2}' if kind == "duplicate" else b'{"a":1e999}' if kind == "nonfinite" else worker._json(receipt())
        (tmp_path/"result.json").write_bytes(raw)
    if kind == "oversized": monkeypatch.setattr(worker, "MAX_RESULT_BYTES", 1)
    if kind == "late":
        ticks = iter([0, 0, 61])
        monkeypatch.setattr(worker.time, "monotonic", lambda: next(ticks))
    with pytest.raises((PreviewContractError, OSError)):
        worker._run(tmp_path, request, 0 if kind == "late" else worker.time.monotonic())


def test_late_cache_commit_is_rolled_back(monkeypatch):
    ticks = iter([0, 0, 61])
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(ticks))
    with pytest.raises(PreviewContractError, match="TIME_BUDGET"):
        worker._cache_put("late", receipt(), HASH, CANDIDATES, 0)
    assert not worker._cache and worker._cache_bytes == 0


@pytest.mark.parametrize("mode", ["denied","empty","slow"])
def test_monitor_preflight_failure_never_spawns_native_child(monkeypatch,tmp_path,mode):
    now=[0.0]
    monkeypatch.setattr(worker.time,"monotonic",lambda:now[0])
    monkeypatch.setattr(worker,"_source_pins",lambda:{})
    spawned,quiescent=[],[]
    monkeypatch.setattr(worker.subprocess,"Popen",lambda *a,**kw:spawned.append(True))
    def monitor(pgid):
        assert pgid==worker.os.getpgrp()
        if mode=="denied": raise PermissionError("authored monitor denial")
        if mode=="slow": now[0]=61.0
        return [] if mode=="empty" else [(1,1)]
    monkeypatch.setattr(worker,"_group_rows",monitor)
    reason="TIME_BUDGET" if mode=="slow" else "RSS_MONITOR"
    with pytest.raises(PreviewContractError,match=reason):
        worker._run(tmp_path,{"nonce":"authored","source_pins":{}},0,on_quiescent=lambda:quiescent.append(True))
    assert not spawned and quiescent==[True]


def test_unavailable_publication_still_obeys_deadline(monkeypatch):
    monkeypatch.setattr(worker.time, "monotonic", lambda: 61)
    with pytest.raises(PreviewContractError, match="TIME_BUDGET"):
        worker._cache_put("refused", worker.unavailable("REFUSED"), HASH, CANDIDATES, 0)


@pytest.mark.parametrize("cleanup", ["temporary_directory", "admission_slot"])
def test_cleanup_expiry_refuses_success_without_caching(monkeypatch, tmp_path, cleanup):
    source,expected,_,calls=prepared(monkeypatch,tmp_path)
    now=[0.0]
    monkeypatch.setattr(worker.time,"monotonic",lambda:now[0])
    if cleanup=="temporary_directory":
        original=worker.tempfile.TemporaryDirectory
        class SlowCleanup:
            def __init__(self,**kwargs): self.inner=original(**kwargs)
            def __enter__(self): return self.inner.__enter__()
            def __exit__(self,*args):
                result=self.inner.__exit__(*args)
                now[0]=61.0
                return result
        monkeypatch.setattr(worker.tempfile,"TemporaryDirectory",SlowCleanup)
    else:
        class SlowSlot:
            def acquire(self,blocking=False): return True
            def release(self): now[0]=61.0
        monkeypatch.setattr(worker,"_slot",SlowSlot())
    result=worker.build_unfold_preview_bounded(source,expected,CANDIDATES)
    assert result["status"]=="UNAVAILABLE" and result["reason"]=="UNFOLD_TIME_BUDGET_EXCEEDED"
    assert len(calls)==1 and not worker._cache and worker._cache_bytes==0


def test_elapsed_receipt_includes_successful_snapshot_cleanup(monkeypatch,tmp_path):
    source,expected,_,_=prepared(monkeypatch,tmp_path)
    now=[0.0]
    monkeypatch.setattr(worker.time,"monotonic",lambda:now[0])
    original=worker.tempfile.TemporaryDirectory
    class TimedCleanup:
        def __init__(self,**kwargs): self.inner=original(**kwargs)
        def __enter__(self): return self.inner.__enter__()
        def __exit__(self,*args):
            result=self.inner.__exit__(*args)
            now[0]=3.0
            return result
    monkeypatch.setattr(worker.tempfile,"TemporaryDirectory",TimedCleanup)
    result=worker.build_unfold_preview_bounded(source,expected,CANDIDATES)
    assert result["status"]=="AVAILABLE" and result["diagnostics"]["elapsed_seconds"]==3.0
    assert len(worker._cache)==1


@pytest.mark.parametrize("expected,candidates", [("invalid", CANDIDATES), (HASH, []), (HASH, [None])])
def test_invalid_admission_never_hashes_runtime_or_reads_source(monkeypatch, expected, candidates):
    monkeypatch.setattr(worker, "_source_pins", lambda: (_ for _ in ()).throw(AssertionError("runtime must not read")))
    refusal = worker.build_unfold_preview_bounded("not-accessed.step", expected, candidates)
    assert refusal["status"] == "UNAVAILABLE" and refusal["vertices"] == []
    assert refusal["diagnostics"]["transport_version"] == 4


def test_wrong_private_nonce_profile_or_snapshot_refuses_without_native(monkeypatch):
    monkeypatch.setattr(worker, "_read", lambda *args: worker._json({"nonce": "test"}))
    monkeypatch.delenv(worker.NONCE_ENV, raising=False)
    with pytest.raises(PreviewContractError, match="SUPERVISED"):
        worker._main("authored/request.json")
    request = {"nonce": "test", "cache_profile": worker.PROFILE, "cad_path": "external.step",
               "expected_hash": HASH, "candidates": CANDIDATES}
    monkeypatch.setenv(worker.NONCE_ENV, "test")
    monkeypatch.setattr(worker, "_read", lambda *args: worker._json(request))
    with pytest.raises(PreviewContractError, match="PRIVATE_CAD_SNAPSHOT"):
        worker._main("authored/request.json")


def authored_pin_files(monkeypatch, tmp_path):
    """Only authored fake source bytes; no runtime package/native reads."""
    required = {
        "backend/vendor_unfold_worker_v2.py": worker.BASE_WORKER_SHA256,
        "backend/vendor_unfold_contract_v2.py": worker.COMMON_CONTRACT_SHA256,
        "backend/vendor_unfold_preview.py": worker.LEGACY_ENGINE_SHA256,
        "backend/vendor_unfold_preview_v4.py": worker.INCUMBENT_ENGINE_SHA256,
        "backend/vendor_unfold_preview_v5.py": worker.ENGINE_SHA256,
        "backend/vendor_unfold_finite_width.py": worker.FINITE_WIDTH_SHA256,
        "backend/vendor_unfold_finite_width_contract_v4.py": worker.ADDON_SHA256,
        "backend/vendor_unfold_contract_v3.py": worker.INCUMBENT_CONTRACT_SHA256,
        "backend/vendor_unfold_contract_v4.py": worker.CONTRACT_SHA256,
        **worker.HELPER_PINS,
        "scripts/extract_step_topology_bend_axes.py": worker.EXTRACTOR_SHA256,
    }
    hashes = {}
    for name, digest in {**required, "backend/vendor_unfold_worker_v4.py": "c"*64}.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = ("AUTHORED PIN PLACEHOLDER "+name).encode()
        path.write_bytes(raw); hashes[raw] = digest
    specs = {}
    for name in ("fractions", "numbers", "decimal", "_pydecimal", "_decimal", "math"):
        path = tmp_path/(name+(".so" if name in {"_decimal", "math"} else ".py"))
        raw = ("AUTHORED NONEXECUTABLE PLACEHOLDER "+name).encode()
        path.write_bytes(raw); hashes[raw] = "d"*64
        specs[name] = SimpleNamespace(origin=str(path))
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(shared, "_source_pins", lambda: {"authored-interpreter": "e"*64})
    monkeypatch.setattr(worker.importlib.util, "find_spec", specs.get)
    monkeypatch.setattr(worker, "_hash", hashes.__getitem__)
    return required, hashes, specs


def test_all_critical_repo_and_fraction_sources_are_in_profile(monkeypatch, tmp_path):
    required, _, specs = authored_pin_files(monkeypatch, tmp_path)
    pins = worker._source_pins()
    for name, digest in required.items():
        assert pins[str((tmp_path/name).resolve())] == digest
    assert pins[str((tmp_path/"backend/vendor_unfold_worker_v4.py").resolve())] == "c"*64
    assert all(str(Path(spec.origin).resolve()) in pins for spec in specs.values())
    assert pins["authored-interpreter"] == "e"*64


@pytest.mark.parametrize("name", ["backend/vendor_unfold_worker_v2.py",
    "backend/vendor_unfold_contract_v2.py", "backend/vendor_unfold_preview.py",
    "backend/vendor_unfold_preview_v4.py", "backend/vendor_unfold_finite_width.py",
    "backend/vendor_unfold_finite_width_contract_v4.py", "backend/vendor_unfold_contract_v3.py",
    "scripts/extract_step_topology_bend_axes.py"])
def test_unaudited_critical_source_refuses_before_native(monkeypatch, tmp_path, name):
    _, hashes, _ = authored_pin_files(monkeypatch, tmp_path)
    hashes[("AUTHORED PIN PLACEHOLDER "+name).encode()] = "0"*64
    with pytest.raises(PreviewContractError, match="SOURCE_NOT_AUDITED"):
        worker._source_pins()


@pytest.mark.parametrize("mode", ["missing", "builtin", "budget", "contract"])
def test_fraction_source_or_profile_budget_is_fail_closed(monkeypatch, tmp_path, mode):
    _, _, specs = authored_pin_files(monkeypatch, tmp_path)
    if mode == "missing": specs["fractions"] = None
    elif mode == "builtin": specs["fractions"] = SimpleNamespace(origin="built-in")
    elif mode == "budget":
        monkeypatch.setattr(shared, "_source_pins", lambda: {str(i): "a"*64 for i in range(2001)})
    else: monkeypatch.setattr(worker, "CONTRACT_SHA256", None)
    with pytest.raises(PreviewContractError):
        worker._source_pins()


@pytest.mark.parametrize("status", ["AVAILABLE", "UNAVAILABLE"])
def test_private_producer_binds_verified_v4_before_composed_validation(monkeypatch, tmp_path, status):
    request = {"cache_profile": worker.PROFILE, "cad_path": str(tmp_path/"source.step"),
        "expected_hash": HASH, "candidates": CANDIDATES, "nonce": "authored-nonce"}
    pins = {str((worker.ROOT/"backend/vendor_unfold_preview_v5.py").resolve()): worker.ENGINE_SHA256}
    (tmp_path/"request.json").write_bytes(worker._json(request))
    (tmp_path/"pins.json").write_bytes(worker._json(pins))
    monkeypatch.setenv(worker.NONCE_ENV, request["nonce"])
    monkeypatch.setattr(worker, "_source_pins", lambda: pins.copy())
    core_result = {"status": status, "source_sha256": HASH, "diagnostics": {"mechanism_version": 5}}
    built, validations = [], []
    def build(path, expected, candidates):
        built.append((path, expected, candidates))
        assert "transport_version" not in core_result["diagnostics"]
        return core_result
    monkeypatch.setitem(sys.modules, "backend.vendor_unfold_preview_v5", SimpleNamespace(build_unfold_preview=build))
    def composed(value, expected, candidates):
        assert value["diagnostics"]["transport_version"] == 4
        assert value["diagnostics"]["v5_engine_source_sha256"] == pins[next(iter(pins))]
        assert value["diagnostics"]["mechanism_version"] == 5
        validations.append(value["status"])
        return value
    monkeypatch.setattr(worker, "validate_preview", composed)
    worker._main(tmp_path/"request.json")
    encoded = worker.strict_json((tmp_path/"result.json").read_bytes())
    assert encoded["diagnostics"]["candidate_binding"] == worker.canonical_candidates(CANDIDATES)
    assert encoded["diagnostics"]["candidate_input_sha256"] == worker._binding(CANDIDATES)[1]
    assert encoded["status"] == status and validations == [status]
    assert len(built) == 1


@pytest.mark.parametrize("field,value", [("transport_version", 2), ("cache_profile", "legacy"),
    ("v5_engine_source_sha256", "b"*64), ("candidate_binding", []), ("candidate_input_sha256", "b"*64)])
def test_private_producer_never_overwrites_contradictory_native_claim(monkeypatch, tmp_path, field, value):
    request = {"cache_profile": worker.PROFILE, "cad_path": str(tmp_path/"source.step"),
        "expected_hash": HASH, "candidates": CANDIDATES, "nonce": "authored-nonce"}
    pins = {str((worker.ROOT/"backend/vendor_unfold_preview_v5.py").resolve()): worker.ENGINE_SHA256}
    (tmp_path/"request.json").write_bytes(worker._json(request))
    (tmp_path/"pins.json").write_bytes(worker._json(pins))
    monkeypatch.setenv(worker.NONCE_ENV, request["nonce"])
    monkeypatch.setattr(worker, "_source_pins", lambda: pins.copy())
    core_result = {"status": "UNAVAILABLE", "source_sha256": HASH,
                   "diagnostics": {"mechanism_version": 5, field: value}}
    monkeypatch.setitem(sys.modules, "backend.vendor_unfold_preview_v5",
                        SimpleNamespace(build_unfold_preview=lambda *args: core_result))
    with pytest.raises(PreviewContractError, match="NATIVE_TRANSPORT_CLAIM"):
        worker._main(tmp_path/"request.json")
    assert not (tmp_path/"result.json").exists()
