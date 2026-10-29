"""Pure mocked supervisor/cache tests. No native subprocess or real CAD input."""
import hashlib
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import vendor_unfold_worker_v2 as worker
from backend.vendor_unfold_contract_v2 import PreviewContractError, unavailable
from tests.test_vendor_unfold_contract_v2 import CANDIDATES, HASH, receipt


@pytest.fixture(autouse=True)
def empty_cache():
    with worker._cache_lock:
        worker._cache.clear()
        worker._cache_bytes = 0
    yield


def test_cached_objects_are_immutable_and_lru_is_bounded(monkeypatch):
    monkeypatch.setattr(worker, "MAX_CACHE_ENTRIES", 2)
    worker._cache_put("one",receipt())
    first = worker._cache_get("one",HASH,CANDIDATES)
    first["vertices"][0] = 999
    assert worker._cache_get("one",HASH,CANDIDATES)["vertices"][0] == 0
    worker._cache_put("two",receipt())
    worker._cache_put("three",receipt())
    assert worker._cache_get("one",HASH,CANDIDATES) is None
    assert worker._cache_bytes == sum(len(raw) for raw in worker._cache.values())
    monkeypatch.setattr(worker, "MAX_CACHE_BYTES", 1)
    worker._cache_put("four",receipt())
    assert not worker._cache and worker._cache_bytes == 0


def test_oversized_and_transient_results_are_not_cached(monkeypatch):
    worker._cache_put("busy",unavailable("UNFOLD_WORKER_BUSY",HASH))
    worker._cache_put("failed",unavailable("UNFOLD_WORKER_FAILED",HASH))
    monkeypatch.setattr(worker, "MAX_CACHE_ENTRY_BYTES", 1)
    worker._cache_put("oversized",receipt())
    assert not worker._cache


def prepared(monkeypatch, tmp_path):
    source = tmp_path / "authored.step"
    source.write_bytes(b"synthetic mock bytes, not CAD")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    pins = {"authored-runtime": "1"}
    monkeypatch.setattr(worker,"_source_pins",lambda:dict(pins))
    monkeypatch.setattr(worker,"ADMISSION_LOCK_PATH",str(tmp_path/"authored-runtime.lock"))
    calls = []
    def run(folder, request, started, on_spawn=None, on_quiescent=None):
        assert Path(request["cad_path"]).read_bytes() == source.read_bytes()
        assert Path(request["cad_path"]) != source
        assert json.loads((folder/"request.json").read_bytes())["candidates"] == CANDIDATES
        result = receipt()
        result["source_sha256"] = expected
        calls.append(request)
        if on_quiescent is not None: on_quiescent()
        return result
    monkeypatch.setattr(worker,"_run",run)
    return source, expected, pins, calls


def test_exact_bytes_candidates_and_runtime_invalidate_cache(monkeypatch,tmp_path):
    source,expected,pins,calls = prepared(monkeypatch,tmp_path)
    assert worker.build_unfold_preview_bounded(source,expected,CANDIDATES)["status"] == "AVAILABLE"
    assert worker.build_unfold_preview_bounded(source,expected,CANDIDATES)["diagnostics"]["cache_hit"] is True
    assert len(calls) == 1
    pins["authored-runtime"] = "2"
    assert worker.build_unfold_preview_bounded(source,expected,CANDIDATES)["status"] == "AVAILABLE"
    assert len(calls) == 2
    changed = [{**CANDIDATES[0],"line_end":[11,0,0]}]
    # Mock producer has to receive the exact changed geometry.
    def changed_run(folder,request,started,on_spawn=None,on_quiescent=None):
        calls.append(request)
        if on_quiescent: on_quiescent()
        return {**receipt(),"source_sha256":expected}
    monkeypatch.setattr(worker,"_run",changed_run)
    worker.build_unfold_preview_bounded(source,expected,changed)
    assert len(calls) == 3
    source.write_bytes(b"different mock bytes")
    assert worker.build_unfold_preview_bounded(source,expected,CANDIDATES)["reason"] == "CAD_HASH_MISMATCH"
    assert len(calls) == 3


def test_busy_and_failed_job_preserve_empty_abstention(monkeypatch,tmp_path):
    source,expected,_,calls = prepared(monkeypatch,tmp_path)
    worker._slot.acquire()
    try:
        assert worker.build_unfold_preview_bounded(source,expected,CANDIDATES)["reason"] == "UNFOLD_WORKER_BUSY"
    finally:
        worker._slot.release()
    monkeypatch.setattr(worker,"_run",lambda *a: (_ for _ in ()).throw(PreviewContractError("UNFOLD_WORKER_FAILED")))
    result = worker.build_unfold_preview_bounded(source,expected,CANDIDATES)
    assert result["reason"] == "UNFOLD_WORKER_FAILED" and result["vertices"] == []
    assert not worker._cache and not calls


def test_private_entrypoint_requires_supervision(tmp_path,monkeypatch):
    path = tmp_path/"request.json"
    path.write_text('{"nonce":"authored-test"}')
    monkeypatch.delenv("SHERMAN_UNFOLD_SUPERVISED_V2",raising=False)
    with pytest.raises(PreviewContractError,match="SUPERVISED"):
        worker._main(path)


def test_whole_group_cleanup_even_after_leader_exit(monkeypatch):
    killed, waited = [], []
    process = SimpleNamespace(pid=12345,wait=lambda timeout:waited.append(timeout))
    monkeypatch.setattr(worker.os,"killpg",lambda pgid,sig:killed.append((pgid,sig)))
    monkeypatch.setattr(worker,"_group_rows",lambda _:[])
    worker._stop_group(process)
    assert killed == [(12345,signal.SIGKILL)] and waited == [5]


@pytest.mark.parametrize("mode,reason", [("rss","UNFOLD_RSS_BUDGET_EXCEEDED"),
    ("monitor","UNFOLD_RSS_MONITOR_UNAVAILABLE"),("descendant","UNFOLD_LIVE_DESCENDANTS_AFTER_EXIT"),
    ("crash","UNFOLD_WORKER_FAILED"),("time","UNFOLD_TIME_BUDGET_EXCEEDED")])
def test_supervisor_refuses_group_failure_and_cleans(monkeypatch,tmp_path,mode,reason):
    captured, stopped = {}, []
    process = SimpleNamespace(pid=12345,returncode=1 if mode=="crash" else 0,
        poll=lambda: None if mode in {"rss","monitor","time"} else 0)
    def popen(args,**kwargs):
        captured.update(kwargs)
        return process
    monkeypatch.setattr(worker.subprocess,"Popen",popen)
    rows = [] if mode in {"monitor","crash"} else [(12345,1),(12346,worker.MAX_RSS_KIB+1)] if mode=="rss" else [(12346,1)]
    monkeypatch.setattr(worker,"_group_rows",lambda _:rows)
    monkeypatch.setattr(worker,"_stop_group",lambda p:stopped.append(p.pid))
    monkeypatch.setenv("DYLD_LIBRARY_PATH","not-inherited")
    monkeypatch.setenv("PYTHONHOME","not-inherited")
    started = worker.time.monotonic() - (61 if mode=="time" else 0)
    with pytest.raises(PreviewContractError,match=reason):
        worker._run(tmp_path,{"nonce":"authored-test"},started)
    if mode == "time":
        assert not captured and not stopped
        return
    assert stopped == [12345]
    assert captured["start_new_session"] is True
    assert "DYLD_LIBRARY_PATH" not in captured["env"] and "PYTHONHOME" not in captured["env"]
    assert captured["env"]["HOME"] == str(tmp_path) and captured["env"]["OMP_NUM_THREADS"] == "1"


def test_pre_spawn_deadline_refusal_proves_no_native_admission(monkeypatch,tmp_path):
    calls, quiescent = [], []
    monkeypatch.setattr(worker.time,"monotonic",lambda:worker.MAX_SECONDS)
    monkeypatch.setattr(worker.subprocess,"Popen",lambda *args,**kwargs:calls.append("spawn"))
    monkeypatch.setattr(worker,"_stop_group",lambda process:calls.append("stop"))
    with pytest.raises(PreviewContractError,match="TIME_BUDGET"):
        worker._run(tmp_path,{"nonce":"authored"},0,
            on_spawn=lambda pgid:calls.append("journal-running"),
            on_quiescent=lambda:quiescent.append(True))
    assert calls == [] and quiescent == [True]


@pytest.mark.parametrize("kind",["duplicate","source-change","late","missing","oversized"])
def test_post_exit_receipt_checks(monkeypatch,tmp_path,kind):
    process = SimpleNamespace(pid=12345,returncode=0,poll=lambda:0)
    monkeypatch.setattr(worker.subprocess,"Popen",lambda *a,**kw:process)
    monkeypatch.setattr(worker,"_group_rows",lambda _:[])
    monkeypatch.setattr(worker,"_stop_group",lambda _:None)
    monkeypatch.setattr(worker,"_source_pins",lambda:{"runtime":"new" if kind=="source-change" else "old"})
    request = {"nonce":"test","source_pins":{"runtime":"old"},"expected_hash":HASH,"candidates":CANDIDATES}
    if kind != "missing":
        raw = b'{"a":1,"a":2}' if kind=="duplicate" else json.dumps(receipt()).encode()
        (tmp_path/"result.json").write_bytes(raw)
    if kind=="oversized": monkeypatch.setattr(worker,"MAX_RESULT_BYTES",1)
    if kind=="late":
        ticks = iter([0,61])
        monkeypatch.setattr(worker.time,"monotonic",lambda:next(ticks))
    with pytest.raises((PreviewContractError,OSError)):
        worker._run(tmp_path,request,0 if kind=="late" else worker.time.monotonic())


def test_expired_cache_commit_never_retains_result(monkeypatch):
    monkeypatch.setattr(worker.time,"monotonic",lambda:61)
    with pytest.raises(PreviewContractError,match="TIME_BUDGET"):
        worker._cache_put("late",receipt(),0)
    assert not worker._cache


@pytest.mark.parametrize("state",[{"state":"ADMITTING","pgid":None},{"state":"RUNNING","pgid":12345},
    {"state":"RUNNING","pgid":True},{"state":"INVALID","pgid":None}])
def test_persistent_admission_quarantine_refuses_new_jobs(monkeypatch,tmp_path,state):
    source,expected,_,calls=prepared(monkeypatch,tmp_path)
    Path(worker.ADMISSION_LOCK_PATH).write_text(json.dumps(state))
    monkeypatch.setattr(worker,"_group_rows",lambda _:[(12345,100)])
    result=worker.build_unfold_preview_bounded(source,expected,CANDIDATES)
    assert result["status"]=="UNAVAILABLE" and "QUARANTINED" in result["reason"]
    assert not calls


def test_quiescent_predecessor_may_admit_new_job(monkeypatch,tmp_path):
    source,expected,_,calls=prepared(monkeypatch,tmp_path)
    Path(worker.ADMISSION_LOCK_PATH).write_text('{"state":"RUNNING","pgid":12345}')
    monkeypatch.setattr(worker,"_group_rows",lambda _:[])
    assert worker.build_unfold_preview_bounded(source,expected,CANDIDATES)["status"]=="AVAILABLE"
    assert len(calls)==1


def test_failed_cleanup_keeps_persistent_running_record(monkeypatch,tmp_path):
    source,expected,_,calls=prepared(monkeypatch,tmp_path)
    def run(folder,request,started,on_spawn,on_quiescent):
        on_spawn(12345)
        raise PreviewContractError("UNFOLD_PROCESS_GROUP_NOT_QUIESCENT")
    monkeypatch.setattr(worker,"_run",run)
    assert worker.build_unfold_preview_bounded(source,expected,CANDIDATES)["reason"]=="UNFOLD_PROCESS_GROUP_NOT_QUIESCENT"
    monkeypatch.setattr(worker,"_group_rows",lambda _:[(12345,100)])
    assert worker.build_unfold_preview_bounded(source,expected,CANDIDATES)["reason"]=="UNFOLD_ADMISSION_QUARANTINED"
    assert not calls
