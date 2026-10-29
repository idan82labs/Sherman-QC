"""Synthetic runner bounds; no physical CAD or fixture access."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("diagnostic_runner", Path(__file__).resolve().parents[1] / "scripts/diagnose_vendor_unfold_case.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Child:
    pid = 99
    returncode = None
    killed = False
    def poll(self): return self.returncode
    def kill(self): self.killed = True; self.returncode = -9
    def wait(self): return self.returncode


def test_request_bound_prevents_launch(monkeypatch):
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("No launch")))
    assert runner.bounded_diagnostic("/not-read.step", "bad", [{"s": "x" * 100001}])["reason"] == "REQUEST_BUDGET_EXCEEDED"


def test_time_bound_kills_child(monkeypatch):
    child = Child()
    ticks = iter([0, 61])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: child)
    assert runner.bounded_diagnostic("/not-read.step", "bad", [{}])["reason"] == "TIME_BUDGET_EXCEEDED"
    assert child.killed


def test_rss_bound_kills_child(monkeypatch):
    child = Child()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=str(runner.MAX_RSS_KIB + 1)))
    assert runner.bounded_diagnostic("/not-read.step", "bad", [{}])["reason"] == "RSS_BUDGET_EXCEEDED"
    assert child.killed


def test_missing_monitor_refuses(monkeypatch):
    child = Child()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    assert runner.bounded_diagnostic("/not-read.step", "bad", [{}])["reason"] == "RSS_MONITOR_UNAVAILABLE"
    assert child.killed


def test_budgets_and_pinned_kernel_unchanged():
    assert runner.MAX_SECONDS == 60
    assert runner.MAX_RSS_KIB == 4 * 1024 * 1024
    assert runner.MAX_RESULT_BYTES == 40_000_000
    assert runner.PART == "37361005"
    assert runner.PINNED_SOURCE["backend/vendor_unfold_preview.py"] == "f316f56530e2fd6e44f21aa28a9871f43c795714c4b15b652139ee2cfdca461a"


def _result_child(monkeypatch, result):
    def launch(command, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        Path(request["result_path"]).write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, poll=lambda: 0)
    monkeypatch.setattr(runner.subprocess, "Popen", launch)


def test_available_or_partial_preview_is_rejected(monkeypatch):
    for result in ({"status": "AVAILABLE"}, {"status": "DIAGNOSTIC_ONLY", "vertices": [1, 2, 3]},
                   {"status": "DIAGNOSTIC_ONLY", "bend_lines": [{"feature_id": "fake"}]}):
        _result_child(monkeypatch, result)
        assert runner.bounded_diagnostic("/not-read.step", "bad", [{}])["reason"] == "INVALID_DIAGNOSTIC_ENVELOPE"


def test_empty_diagnostic_result_is_retained(monkeypatch):
    expected = {"status": "DIAGNOSTIC_ONLY", "vertices": [], "triangles": [], "bend_lines": [],
                "flange_surfaces": [], "engine_boundary_reason": "SYNTHETIC_REFUSAL"}
    _result_child(monkeypatch, expected)
    assert runner.bounded_diagnostic("/not-read.step", "bad", [{}]) == expected
