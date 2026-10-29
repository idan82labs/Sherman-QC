"""C04 structured logging, metrics rendering and readiness probes."""
from __future__ import annotations

import io
import json
import logging
import logging.handlers
import os
import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from apps.production_line import health, telemetry
from apps.production_line.settings import LinePaths


# --------------------------------------------------------------------------- logging


@pytest.fixture()
def logger_name(request) -> str:
    name = f"sherman_line.test.{request.node.name}"
    yield name
    lg = logging.getLogger(name)
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()


def _lines(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_json_lines_with_context(tmp_path: Path, logger_name: str) -> None:
    err = io.StringIO()
    log_file = telemetry.configure_logging(
        LinePaths.for_root(tmp_path), "INFO", 7, logger_names=[logger_name], stream=err
    )
    assert log_file == tmp_path / "logs" / "line.jsonl"
    lg = logging.getLogger(logger_name)
    with telemetry.log_context(job_id="line_" + "a" * 32, sku="SKU-1", station_id="cell-1"):
        with telemetry.log_context(scan_sha256="f" * 64):
            lg.info("engine started %s", "now")
        lg.warning("outer", extra={"request_id": "req-1", "sku": "OVERRIDE"})
    lg.info("no context")
    lg.debug("filtered out")
    rows = _lines(log_file)
    assert [r["msg"] for r in rows] == ["engine started now", "outer", "no context"]
    first, second, third = rows
    assert set(first) == {"ts", "level", "logger", "msg", "job_id", "sku", "scan_sha256", "station_id", "request_id"}
    assert first["job_id"] == "line_" + "a" * 32 and first["scan_sha256"] == "f" * 64 and first["level"] == "INFO"
    assert second["scan_sha256"] is None and second["request_id"] == "req-1" and second["sku"] == "OVERRIDE"
    assert third["job_id"] is None and third["station_id"] is None
    assert first["ts"].endswith("Z") and first["logger"] == logger_name
    assert len(err.getvalue().splitlines()) == 3  # stderr mirror
    assert (log_file.stat().st_mode & 0o077) == 0


def test_log_file_stays_private_across_rollover(tmp_path: Path, logger_name: str) -> None:
    """The fresh line.jsonl created at midnight rollover must be 0600, not umask 0644."""
    old_umask = os.umask(0o022)
    try:
        log_file = telemetry.configure_logging(tmp_path, "INFO", 3, logger_names=[logger_name], stream=io.StringIO())
        lg = logging.getLogger(logger_name)
        lg.info("before rollover")
        (handler,) = [h for h in lg.handlers if isinstance(h, logging.handlers.TimedRotatingFileHandler)]
        handler.doRollover()
        lg.info("after rollover")
    finally:
        os.umask(old_umask)
    assert (log_file.stat().st_mode & 0o777) == 0o600
    backups = [p for p in log_file.parent.iterdir() if p.name.startswith("line.jsonl.")]
    assert backups and all((p.stat().st_mode & 0o077) == 0 for p in backups)
    assert [r["msg"] for r in _lines(log_file)] == ["after rollover"]
    # A pre-existing world-readable log file is tightened when the handler opens it.
    other = tmp_path / "pre"
    other.mkdir()
    (other / "line.jsonl").write_text("", encoding="utf-8")
    (other / "line.jsonl").chmod(0o644)
    telemetry.configure_logging(other, "INFO", 3, logger_names=[logger_name], stream=io.StringIO())
    assert ((other / "line.jsonl").stat().st_mode & 0o777) == 0o600


def test_configure_logging_is_idempotent_and_rotates_daily(tmp_path: Path, logger_name: str) -> None:
    for _ in range(3):
        telemetry.configure_logging(tmp_path, "INFO", 30, logger_names=[logger_name], stream=io.StringIO())
    handlers = logging.getLogger(logger_name).handlers
    assert len(handlers) == 2
    rot = [h for h in handlers if isinstance(h, logging.handlers.TimedRotatingFileHandler)]
    assert len(rot) == 1 and rot[0].backupCount == 30 and rot[0].when == "MIDNIGHT"
    assert logging.getLogger(logger_name).propagate is False


def test_secrets_redacted(tmp_path: Path, logger_name: str) -> None:
    log_file = telemetry.configure_logging(tmp_path, "INFO", 1, logger_names=[logger_name], stream=io.StringIO())
    lg = logging.getLogger(logger_name)
    lg.info("GET /api/line/events/stream?station_id=c1&token=sst1.abc.def HTTP/1.1")
    lg.info("auth header Bearer eyJhbGciOi.payload.sig and device sdt_SECRETSECRETSECRET")
    lg.info("login password=hunter2 ok")
    lg.info("oauth access_token=ACCESSLEAK&refresh_token=REFRESHLEAK client_secret=CLIENTLEAK max_tokens=7")
    lg.info('body {"password": "JSONLEAK", "username": "u"} dict {\'api_key\': \'DICTLEAK\'}')
    lg.info("Authorization: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.RAWJWTLEAK")
    lg.info('form password="QUOTEDLEAK" done')
    lg.info("Authorization: Bearer OPAQUELEAK123")  # opaque token: only the Bearer rule catches it
    try:
        raise ValueError("token=leaky-value")
    except ValueError:
        lg.exception("boom")
    text = log_file.read_text()
    for leaked in (
        "sst1.abc.def",
        "eyJhbGciOi",
        "SECRETSECRETSECRET",
        "hunter2",
        "leaky-value",
        "ACCESSLEAK",
        "REFRESHLEAK",
        "CLIENTLEAK",
        "JSONLEAK",
        "DICTLEAK",
        "RAWJWTLEAK",
        "QUOTEDLEAK",
        "OPAQUELEAK123",
    ):
        assert leaked not in text, leaked
    # Non-secret look-alikes survive.
    assert "max_tokens=7" in text and '\\"username\\": \\"u\\"' in text
    assert "[REDACTED]" in text and '"exc"' in text


def test_log_context_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        with telemetry.log_context(user="x"):
            pass
    assert telemetry.current_log_context() == {}


def test_log_context_is_task_local() -> None:
    seen = {}

    def worker() -> None:
        seen["thread"] = dict(telemetry.current_log_context())

    with telemetry.log_context(job_id="j1"):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert seen["thread"] == {}


# --------------------------------------------------------------------------- metrics


def test_render_prometheus_counters_gauges_histograms() -> None:
    reg = telemetry.MetricsRegistry()
    c = reg.counter("sherman_line_jobs_finished_total", "Jobs finished", ("kind", "status"))
    c.inc(kind="inspect", status="completed")
    c.inc(2, kind="inspect", status="failed")
    assert reg.counter("sherman_line_jobs_finished_total", "again", ("kind", "status")) is c
    g = reg.gauge("sherman_line_queue_depth", "Queued jobs")
    g.set(3)
    g.dec()
    h = reg.histogram("sherman_line_job_duration_seconds", "Duration", ("kind",), buckets=(1, 10, 60))
    for v in (0.5, 5, 5, 100):
        h.observe(v, kind="inspect")
    text = telemetry.render_prometheus(reg, {"sherman_line_disk_free_bytes": 1.5e10})
    lines = text.splitlines()
    assert "# TYPE sherman_line_jobs_finished_total counter" in lines
    assert 'sherman_line_jobs_finished_total{kind="inspect",status="completed"} 1' in lines
    assert 'sherman_line_jobs_finished_total{kind="inspect",status="failed"} 2' in lines
    assert "sherman_line_queue_depth 2" in lines
    assert 'sherman_line_job_duration_seconds_bucket{kind="inspect",le="1"} 1' in lines
    assert 'sherman_line_job_duration_seconds_bucket{kind="inspect",le="10"} 3' in lines
    assert 'sherman_line_job_duration_seconds_bucket{kind="inspect",le="60"} 3' in lines
    assert 'sherman_line_job_duration_seconds_bucket{kind="inspect",le="+Inf"} 4' in lines
    assert 'sherman_line_job_duration_seconds_count{kind="inspect"} 4' in lines
    assert 'sherman_line_job_duration_seconds_sum{kind="inspect"} 110.5' in lines
    assert "# TYPE sherman_line_disk_free_bytes gauge" in lines
    assert "sherman_line_disk_free_bytes 15000000000" in lines
    assert text.endswith("\n")


def test_metric_validation_and_escaping() -> None:
    reg = telemetry.MetricsRegistry()
    with pytest.raises(ValueError):
        reg.counter("bad-name", "x")
    with pytest.raises(ValueError):
        reg.counter("ok_total", "x", ("le",))
    c = reg.counter("ok_total", "help with\nnewline", ("reason",))
    with pytest.raises(ValueError):
        c.inc(-1, reason="x")
    with pytest.raises(ValueError):
        c.inc(reason="x", extra="y")
    with pytest.raises(ValueError):
        reg.gauge("ok_total", "type clash")
    c.inc(reason='quote " and \\ backslash\n')
    text = telemetry.render_prometheus(reg, [("sherman_x", 1, "h", {"station_id": "cell-1"})])
    assert 'ok_total{reason="quote \\" and \\\\ backslash\\n"} 1' in text
    assert "# HELP ok_total help with\\nnewline" in text
    assert 'sherman_x{station_id="cell-1"} 1' in text
    with pytest.raises(ValueError):
        telemetry.render_prometheus(reg, {"ok_total": 1})  # duplicates a registered metric


# --------------------------------------------------------------------------- health


def test_sqlite_writable(tmp_path: Path) -> None:
    db = tmp_path / "line.sqlite3"
    assert not health.sqlite_writable(db).ok
    assert not db.exists()  # the probe never creates a database
    sqlite3.connect(db).close()
    assert health.sqlite_writable(db, "line_db") == health.ProbeResult("line_db", True, "writable")
    holder = sqlite3.connect(db, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        res = health.sqlite_writable(db)
        assert not res.ok and "not writable" in res.detail
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    db.chmod(0o444)
    try:
        assert not health.sqlite_writable(db).ok
    finally:
        db.chmod(0o644)


@pytest.mark.parametrize("dirname", ["Shop Data #1", "pct%20dir", "q?mark=x", "100% sure"])
def test_sqlite_probes_quote_reserved_characters(tmp_path: Path, dirname: str) -> None:
    d = tmp_path / dirname
    d.mkdir()
    db = d / "line.sqlite3"
    assert not health.sqlite_writable(db).ok
    assert sorted(p.name for p in tmp_path.iterdir()) == [dirname] and list(d.iterdir()) == []
    sqlite3.connect(db).close()
    res = health.sqlite_writable(db)
    assert res.ok, res.detail
    assert sorted(p.name for p in tmp_path.iterdir()) == [dirname]
    assert sorted(p.name for p in d.iterdir()) == ["line.sqlite3"]


def test_sqlite_uri_rejects_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        health.sqlite_uri(tmp_path / "x.db", "rwx")
    assert health.sqlite_uri(tmp_path / "a #b.db", "ro").endswith("a%20%23b.db?mode=ro")


def test_disk_free(tmp_path: Path) -> None:
    assert health.disk_free(tmp_path, min_bytes=0).ok
    assert not health.disk_free(tmp_path, min_bytes=1 << 62).ok
    assert not health.disk_free(tmp_path / "missing" / "dir").ok


def test_heartbeat_fresh() -> None:
    now = datetime(2026, 10, 17, 3, 0, 0, tzinfo=timezone.utc)
    beats = {"runner": (now - timedelta(seconds=10)).isoformat(), "hotfolder:cell-1": now - timedelta(seconds=120)}
    assert health.heartbeat_fresh(beats, "runner", 45, now=now).ok
    assert not health.heartbeat_fresh(beats, "hotfolder:cell-1", 45, now=now).ok
    assert not health.heartbeat_fresh(beats, "absent", 45, now=now).ok
    rows = [{"component": "runner", "ts": (now - timedelta(seconds=5)).isoformat().replace("+00:00", "Z")}]
    assert health.heartbeat_fresh(rows, "runner", 45, now=now).ok
    future = {"runner": (now + timedelta(minutes=5)).isoformat()}
    assert not health.heartbeat_fresh(future, "runner", 45, now=now).ok
    assert not health.heartbeat_fresh({"runner": "not a time"}, "runner", 45, now=now).ok


def test_python_probe() -> None:
    ok = health.python_probe(Path(sys.executable), timeout_s=60)
    assert ok.ok, ok.detail
    bad = health.python_probe(Path(sys.executable), code="import module_that_does_not_exist_xyz")
    assert not bad.ok and "ModuleNotFoundError" in bad.detail
    assert not health.python_probe(Path("/nonexistent/python3")).ok
    slow = health.python_probe(Path(sys.executable), timeout_s=0.5, code="import time; time.sleep(5)")
    assert not slow.ok and "timed out" in slow.detail


def test_bundle_present(tmp_path: Path) -> None:
    assert not health.bundle_present(tmp_path).ok
    (tmp_path / "bundle_manifest.json").write_text("{not json")
    assert not health.bundle_present(tmp_path).ok
    (tmp_path / "bundle_manifest.json").write_text(json.dumps({"bundle_id": "abc123", "files": {}}))
    res = health.bundle_present(tmp_path)
    assert res.ok and "abc123" in res.detail
    # verify=True re-checks against the manifest (bundle id no longer matches the content).
    assert not health.bundle_present(tmp_path, verify=True).ok


def test_readiness_fail_closed() -> None:
    def raising() -> health.ProbeResult:
        raise OSError("disk gone")

    ok, results = health.readiness([health.ProbeResult("a", True), lambda: health.ProbeResult("b", True)])
    assert ok and [r.name for r in results] == ["a", "b"]
    ok, results = health.readiness([health.ProbeResult("a", True), raising])
    assert not ok and results[1].name == "raising" and not results[1].ok and "OSError" in results[1].detail
    assert health.readiness([]) == (False, [])
    ok, results = health.readiness([lambda: None])  # type: ignore[list-item]
    assert not ok
    body = health.readiness_body(*health.readiness([health.ProbeResult("disk_free", False, "free=1 min=2")]))
    assert body == {"ready": False, "checks": {"disk_free": False}}
