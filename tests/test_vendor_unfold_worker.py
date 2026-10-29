"""Supervisor contract checks without physical CAD or product initialization."""
from types import SimpleNamespace

from backend import vendor_unfold_worker as worker


def test_busy_does_not_launch_or_access_cad():
    assert worker._slot.acquire(blocking=False)
    try:
        result = worker.build_unfold_preview_bounded("/not-opened.step", "bad", [])
    finally:
        worker._slot.release()
    assert result["reason"] == "UNFOLD_WORKER_BUSY"
    assert result["source_sha256"] is None
    assert result["vertices"] == []


def test_invalid_expected_hash_refused_in_fresh_worker():
    result = worker.build_unfold_preview_bounded("/not-opened.step", "bad", [{}])
    assert result["reason"] == "INVALID_EXPECTED_CAD_HASH"
    assert result["source_sha256"] is None
    assert result["bend_lines"] == []


def test_large_request_refused_before_launch(monkeypatch):
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch")))
    result = worker.build_unfold_preview_bounded("/not-opened.step", "bad", [{"note": "x" * 100_001}])
    assert result["reason"] == "UNFOLD_REQUEST_BUDGET_EXCEEDED"


class FakeInput:
    closed = False
    def write(self, value):
        self.value = value
    def close(self):
        self.closed = True


class FakeProcess:
    pid = 99
    returncode = None
    def __init__(self):
        self.stdin = FakeInput()
        self.killed = False
    def poll(self):
        return self.returncode
    def kill(self):
        self.killed = True
        self.returncode = -9
    def wait(self):
        return self.returncode


def test_rss_cap_kills_child_and_releases_slot(monkeypatch):
    child = FakeProcess()
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=str(worker.MAX_RSS_KIB + 1)))
    result = worker.build_unfold_preview_bounded("/not-opened.step", "bad", [{}])
    assert child.killed
    assert result["reason"] == "UNFOLD_RSS_BUDGET_EXCEEDED"
    assert worker._slot.acquire(blocking=False)
    worker._slot.release()


def test_time_cap_kills_child_without_changing_budget(monkeypatch):
    child = FakeProcess()
    ticks = iter([0, worker.MAX_SECONDS + 1])
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(ticks))
    result = worker.build_unfold_preview_bounded("/not-opened.step", "bad", [{}])
    assert child.killed
    assert result["reason"] == "UNFOLD_TIME_BUDGET_EXCEEDED"
    assert worker.MAX_SECONDS == 60
