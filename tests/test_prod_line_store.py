"""C02 line store: SQLite durability, leases, dedupe, quarantine, events, parity.

tmp_path databases only; no real scans. The contract imports are used to check
that store rows validate as the C01 models and that the mirrored vocabularies
equal the contract sets.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from apps.production_line.store import vocabulary as vocab
from apps.production_line.store.line_store import (
    JobStateError,
    LineStore,
    LineStoreError,
    QuarantineError,
    inspect_dedupe_key,
    iso_utc,
    scan_file_id_for,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
RECIPE_SHA = "c" * 64
COMPILED_SHA = "d" * 64
ADMIN = {"kind": "user", "id": "admin-1", "role": "admin"}


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 17, 7, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def store(tmp_path, clock) -> LineStore:
    return LineStore(tmp_path / "db" / "line.sqlite3", clock=clock)


def _parse(status: str = "OK", sku: str | None = "23553000") -> dict:
    return {"status": status, "sku": sku, "candidates": []}


def _register(store: LineStore, sha: str = SHA_A, name: str = "23553000_B_FULL_SCAN 1.ply", **kw):
    defaults = dict(
        sha256=sha,
        original_name=name,
        size_bytes=2048,
        station_id="cell-1",
        stored_rel_path=f"archive/scans/2026/10/{sha}.ply",
        parse=_parse(),
        match={"status": "MATCHED", "recipe_id": "23553000@v1", "recipe_version": 1, "recipe_sha256": RECIPE_SHA},
        format="PLY_BINARY_LE",
        vertex_count=10,
        face_count=0,
    )
    defaults.update(kw)
    return store.register_scan(**defaults)


def _enqueue_inspect(store: LineStore, sha: str = SHA_A, **kw):
    return store.enqueue_job(
        "inspect",
        sku="23553000",
        recipe_id="23553000@v1",
        recipe_version=1,
        recipe_sha256=RECIPE_SHA,
        compiled_part_sha256=COMPILED_SHA,
        station_id="cell-1",
        scan_file_id=scan_file_id_for(sha),
        scan_sha256=sha,
        dedupe_key=inspect_dedupe_key(sha, RECIPE_SHA),
        **kw,
    )


def _kinds(store: LineStore) -> list[str]:
    return [e["kind"] for e in store.events_after(0, limit=1000)]


# ---------------------------------------------------------------- schema / pragmas
def test_schema_is_idempotent_and_wal(tmp_path, clock):
    db = tmp_path / "line.sqlite3"
    LineStore(db, clock=clock)
    second = LineStore(db, clock=clock)  # re-running migrations is a no-op
    assert second.schema_version() == 1
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    finally:
        conn.close()
    assert {"schema_meta", "scan_files", "jobs", "events", "heartbeats", "ingest_sources"} <= tables
    assert {"jobs_dedupe", "jobs_queue"} <= indexes


def test_refuses_newer_schema(tmp_path, clock):
    db = tmp_path / "line.sqlite3"
    LineStore(db, clock=clock)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE schema_meta SET version=99 WHERE component='line_store'")
    conn.commit()
    conn.close()
    with pytest.raises(LineStoreError) as exc:
        LineStore(db, clock=clock)
    assert exc.value.code == "SCHEMA_TOO_NEW"


def test_timestamps_are_fixed_width_utc():
    ts = iso_utc(datetime(2026, 1, 2, 3, 4, 5, 6000, tzinfo=timezone.utc))
    assert ts == "2026-01-02T03:04:05.006Z"
    with pytest.raises(ValueError):
        iso_utc(datetime(2026, 1, 2))


# ---------------------------------------------------------------- scans
def test_register_scan_and_duplicate(store):
    row, created = _register(store)
    assert created and row["scan_file_id"] == "scan_" + SHA_A[:24]
    assert row["ingest_state"] == "received"
    assert _kinds(store) == ["SCAN_RECEIVED"]
    # Still 'received' (ingest not finished): a re-drop of the SAME name resumes, no event.
    again, created = _register(store)
    assert not created and again["original_name"] == "23553000_B_FULL_SCAN 1.ply"
    assert _kinds(store) == ["SCAN_RECEIVED"]
    # ... but the same bytes under another name are a duplicate even while 'received'.
    other, created = _register(store, name="copy (1).ply")
    assert not created and other["original_name"] == "23553000_B_FULL_SCAN 1.ply"
    assert _kinds(store) == ["SCAN_RECEIVED", "SCAN_DUPLICATE"]
    assert LineStore.is_duplicate_drop(other, original_name="copy (1).ply", station_id=other["station_id"])
    assert LineStore.is_duplicate_drop(other, original_name=other["original_name"], station_id="cell-other")
    assert not LineStore.is_duplicate_drop(other, original_name=other["original_name"], station_id=other["station_id"])
    store.match_and_enqueue(
        row["scan_file_id"],
        match=row["match"],
        sku="23553000",
        compiled_part_sha256=COMPILED_SHA,
    )
    _again, created = _register(store, name="copy (2).ply")
    assert not created
    dup = [e for e in store.events_after(0) if e["kind"] == "SCAN_DUPLICATE"]
    assert len(dup) == 2 and dup[-1]["payload"]["original_name"] == "copy (2).ply"
    assert dup[0]["payload"]["existing_state"] == "received" and dup[-1]["payload"]["existing_state"] == "enqueued"


def test_register_scan_validates(store):
    with pytest.raises(LineStoreError):
        _register(store, sha="XYZ")
    with pytest.raises(LineStoreError):
        _register(store, format="OBJ")
    with pytest.raises(LineStoreError):  # quarantine reason without the state
        _register(store, quarantine_reason="PLY_TRUNCATED")
    with pytest.raises(LineStoreError):
        _register(store, ingest_state="quarantined", quarantine_reason="NOT_A_REASON")
    with pytest.raises(LineStoreError):  # MATCHED without its recipe
        _register(store, match={"status": "MATCHED"})


def test_scan_rows_validate_as_contract(store):
    from contracts.production.line_job import ScanFileV1

    row, _ = _register(store, bbox_mm=((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)))
    model = ScanFileV1.model_validate(row)
    assert model.bbox_mm == ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    q, _ = _register(
        store,
        sha=SHA_B,
        ingest_state="quarantined",
        quarantine_reason="PLY_TRUNCATED",
        format=None,
        match={"status": "UNMATCHED", "reason": "x"},
        parse=_parse("UNMATCHED", None),
    )
    assert ScanFileV1.model_validate(q).quarantine_reason == "PLY_TRUNCATED"


def test_quarantine_assign_and_content_reasons_not_resolvable(store):
    row, _ = _register(store, match={"status": "UNMATCHED", "reason": "NO_KNOWN_SKU_TOKEN"}, parse=_parse("UNMATCHED", None))
    store.quarantine_scan(row["scan_file_id"], "UNMATCHED_SKU", {"why": "test"})
    bad, _ = _register(
        store,
        sha=SHA_B,
        ingest_state="quarantined",
        quarantine_reason="PLY_TRUNCATED",
        match={"status": "UNMATCHED"},
        parse=_parse("UNMATCHED", None),
        format=None,
    )
    entries = {e["scan_file"]["scan_file_id"]: e for e in store.list_quarantine("cell-1")}
    assert entries[row["scan_file_id"]]["resolvable"] is True
    assert entries[bad["scan_file_id"]]["resolvable"] is False
    ref = {
        "recipe_id": "23553000@v1",
        "recipe_version": 1,
        "recipe_sha256": RECIPE_SHA,
        "sku": "23553000",
        "compiled_part_sha256": COMPILED_SHA,
    }
    with pytest.raises(QuarantineError) as exc:
        store.resolve_quarantine(bad["scan_file_id"], ref, ADMIN)
    assert exc.value.code == "NOT_RESOLVABLE"
    with pytest.raises(LineStoreError):  # a principal is mandatory
        store.resolve_quarantine(row["scan_file_id"], ref, None)
    job_id = store.resolve_quarantine(row["scan_file_id"], ref, ADMIN)
    job = store.get_job(job_id)
    assert job["kind"] == "inspect" and job["scan_sha256"] == SHA_A and job["compiled_part_sha256"] == COMPILED_SHA
    scan = store.get_scan_file(row["scan_file_id"])
    assert scan["ingest_state"] == "enqueued" and scan["match"]["status"] == "MATCHED"
    assert scan["quarantine_reason"] is None  # contract: only while quarantined
    assert store.get_scan_meta(row["scan_file_id"])["resolution"] == "assigned"
    assert [e["scan_file"]["scan_file_id"] for e in store.list_quarantine()] == [bad["scan_file_id"]]
    resolved = store.list_quarantine(include_resolved=True)
    assert {e["resolution"] for e in resolved} == {"assigned", None}
    kinds = _kinds(store)
    assert "QUARANTINE_ASSIGNED" in kinds and kinds.count("SCAN_QUARANTINED") == 2
    with pytest.raises(QuarantineError):  # not quarantined any more
        store.resolve_quarantine(row["scan_file_id"], ref, ADMIN)
    dismissed = store.dismiss_quarantine(bad["scan_file_id"], ADMIN, note="re-export requested")
    assert dismissed["ingest_state"] == "dismissed"
    assert store.counts()["open_quarantine"] == 0


# ---------------------------------------------------------------- jobs
def test_enqueue_validation_and_dedupe(store):
    _register(store)
    job_id, created = _enqueue_inspect(store)
    assert created
    again, created = _enqueue_inspect(store)
    assert (again, created) == (job_id, False)
    assert _kinds(store).count("JOB_QUEUED") == 1
    with pytest.raises(LineStoreError):  # inspect jobs pin the compiled part
        store.enqueue_job(
            "inspect", sku="X", recipe_id="X@v1", recipe_version=1, recipe_sha256=RECIPE_SHA,
            scan_file_id=scan_file_id_for(SHA_A), scan_sha256=SHA_A,
        )  # fmt: skip
    with pytest.raises(LineStoreError):  # recipe id must be <sku>@v<version>
        store.enqueue_job("recipe_compile", sku="X", recipe_id="Y@v1", recipe_version=1, recipe_sha256=RECIPE_SHA)
    with pytest.raises(LineStoreError):
        store.enqueue_job("recipe_expectations", sku="X", recipe_id="X@v1", recipe_version=1, recipe_sha256=RECIPE_SHA)
    with pytest.raises(LineStoreError):  # scan id must match the sha
        store.enqueue_job(
            "inspect", sku="X", recipe_id="X@v1", recipe_version=1, recipe_sha256=RECIPE_SHA,
            compiled_part_sha256=COMPILED_SHA, scan_file_id=scan_file_id_for(SHA_B), scan_sha256=SHA_A,
        )  # fmt: skip


def test_job_rows_validate_as_contract(store):
    from contracts.production.line_job import LineJobV1

    _register(store)
    job_id, _ = _enqueue_inspect(store)
    assert LineJobV1.model_validate(store.get_job(job_id)).execution_status == "queued"
    job = store.lease_next("r1", 30)
    assert LineJobV1.model_validate(job).attempts == 1
    summary = {
        "inspection_validity": "VALID",
        "summary_state": "STAGE_IN_TOLERANCE",
        "bends_total": 4,
        "required_fail": 0,
        "required_review": 0,
        "required_unknown": 0,
        "stage_status": "CONSISTENT",
    }
    assert store.complete_job(job_id, "r1", summary, peak_rss_bytes=123, duration_s=1.5)
    done = LineJobV1.model_validate(store.get_job(job_id))
    assert done.execution_status == "completed" and done.stage == "done" and done.progress == 100
    assert done.result_summary.summary_state == "STAGE_IN_TOLERANCE"


def test_lease_order_priority_then_age(store, clock):
    a, _ = store.enqueue_job("recipe_compile", sku="P1", recipe_id="P1@v1", recipe_version=1, recipe_sha256=SHA_A)
    clock.advance(1)
    _register(store)
    b, _ = _enqueue_inspect(store)
    clock.advance(1)
    c, _ = store.enqueue_job("recipe_compile", sku="P2", recipe_id="P2@v1", recipe_version=1, recipe_sha256=SHA_B)
    assert store.get_job(b)["priority"] == 100 and store.get_job(a)["priority"] == 10
    order = [store.lease_next("r", 30)["job_id"] for _ in range(3)]
    assert order == [b, a, c]
    assert store.lease_next("r", 30) is None


def test_concurrent_leases_never_double_lease(tmp_path):
    db = tmp_path / "line.sqlite3"
    seed = LineStore(db)
    ids = {
        seed.enqueue_job("recipe_compile", sku=f"P{i}", recipe_id=f"P{i}@v1", recipe_version=1, recipe_sha256=SHA_A)[0]
        for i in range(20)
    }
    leased: list[str] = []
    lock = threading.Lock()

    def worker(n: int) -> None:
        local = LineStore(db)
        while True:
            job = local.lease_next(f"w{n}", 30)
            if job is None:
                return
            with lock:
                leased.append(job["job_id"])

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(leased) == sorted(ids)


def test_progress_rate_limit_and_owner_check(store, clock):
    _register(store)
    job_id, _ = _enqueue_inspect(store)
    store.lease_next("r1", 30)
    assert store.update_progress(job_id, "r1", "engine", 10, "a")
    assert store.update_progress(job_id, "r1", "engine", 20, "b")  # row updated, event suppressed
    clock.advance(1.0)
    assert store.update_progress(job_id, "r1", "engine", 30, "c")
    assert not store.update_progress(job_id, "intruder", "engine", 40, "x")
    progress = [e for e in store.events_after(0) if e["kind"] == "JOB_PROGRESS"]
    assert [e["payload"]["progress"] for e in progress] == [10, 30]
    assert store.get_job(job_id)["progress"] == 30
    assert store.get_job_meta(job_id)["progress_message"] == "c"
    with pytest.raises(LineStoreError):
        store.update_progress(job_id, "r1", "done", 50)


def test_lease_lost_owner_cannot_finish(store):
    _register(store)
    job_id, _ = _enqueue_inspect(store)
    store.lease_next("r1", 30)
    assert not store.complete_job(job_id, "someone-else", None)
    assert not store.fail_job(job_id, "someone-else", "WORKER_CRASH", "x")
    assert not store.renew_lease(job_id, "someone-else", 30)
    assert store.renew_lease(job_id, "r1", 30)
    with pytest.raises(LineStoreError):
        store.fail_job(job_id, "r1", "NOT_A_CODE", "x")
    assert store.fail_job(job_id, "r1", "WORKER_TIMEOUT", "too slow", peak_rss_bytes=5, duration_s=2)
    job = store.get_job(job_id)
    assert job["failure"] == {"code": "WORKER_TIMEOUT", "message": "too slow"}
    assert job["result_summary"] is None  # a failure never carries a conformance summary
    with pytest.raises(LineStoreError):
        store.complete_job(job_id, "r1", {"summary_state": "PASS"})


def test_cancel_queued_and_running(store):
    _register(store)
    queued, _ = _enqueue_inspect(store)
    assert store.request_cancel(queued, ADMIN) == "cancelled"
    assert store.get_job(queued)["execution_status"] == "cancelled"
    assert store.lease_next("r1", 30) is None  # a cancelled job is never leased
    run, _ = store.enqueue_job("recipe_compile", sku="P", recipe_id="P@v1", recipe_version=1, recipe_sha256=SHA_B)
    store.lease_next("r1", 30)
    assert store.request_cancel(run, ADMIN) == "cancel_requested"
    assert store.is_cancel_requested(run)
    assert store.mark_cancelled(run, "r1", duration_s=1.0)
    assert store.request_cancel(run, ADMIN) == "not_cancellable"
    with pytest.raises(KeyError):
        store.request_cancel("line_" + "0" * 32, ADMIN)


def test_recover_expired_leases_durability(tmp_path, clock):
    """Acceptance: lease, drop every object (crash), expire, recover twice."""
    db = tmp_path / "line.sqlite3"
    s1 = LineStore(db, clock=clock)
    _register(s1)
    job_id, _ = _enqueue_inspect(s1)
    assert s1.lease_next("dead-runner", 60)["attempts"] == 1
    del s1  # simulated crash: nothing survives but the file
    clock.advance(30)
    assert LineStore(db, clock=clock).recover_expired_leases() == []  # lease still valid
    clock.advance(31)
    s2 = LineStore(db, clock=clock)
    assert s2.recover_expired_leases() == [(job_id, "requeued")]
    job = s2.get_job(job_id)
    assert job["execution_status"] == "queued" and job["attempts"] == 1 and job["lease_owner"] is None
    assert s2.lease_next("runner-2", 60)["attempts"] == 2
    clock.advance(61)
    s3 = LineStore(db, clock=clock)
    assert s3.recover_expired_leases() == [(job_id, "failed")]
    job = s3.get_job(job_id)
    assert job["execution_status"] == "failed" and job["failure"]["code"] == "WORKER_INTERRUPTED"
    recovered = [e for e in s3.events_after(0) if e["kind"] == "JOB_RECOVERED"]
    assert [e["payload"]["action"] for e in recovered] == ["requeued", "failed"]
    # dedupe still blocks a second job for the same scan + recipe
    assert _enqueue_inspect(s3) == (job_id, False)


def test_recover_cancel_requested_job_is_cancelled(store, clock):
    job_id, _ = store.enqueue_job("recipe_compile", sku="P", recipe_id="P@v1", recipe_version=1, recipe_sha256=SHA_B)
    store.lease_next("r1", 10)
    store.request_cancel(job_id, ADMIN)
    clock.advance(11)
    assert store.recover_expired_leases() == [(job_id, "cancelled")]
    assert store.get_job(job_id)["execution_status"] == "cancelled"


def test_retry_creates_linked_job_once(store):
    _register(store)
    job_id, _ = _enqueue_inspect(store)
    with pytest.raises(JobStateError):
        store.request_retry(job_id, ADMIN)  # queued: nothing to retry
    store.lease_next("r1", 30)
    store.fail_job(job_id, "r1", "WORKER_CRASH", "boom")
    retry = store.request_retry(job_id, ADMIN)
    assert store.request_retry(job_id, ADMIN) == retry  # idempotent while the retry is pending
    job = store.get_job(retry)
    assert job["retry_of"] == job_id and job["dedupe_key"] == f"retry:{job_id}:1" and job["attempts"] == 0
    assert store.get_scan_meta(scan_file_id_for(SHA_A))["job_id"] == retry
    store.lease_next("r1", 30)
    store.fail_job(retry, "r1", "WORKER_CRASH", "boom again")
    second = store.request_retry(job_id, ADMIN)
    assert store.get_job(second)["dedupe_key"] == f"retry:{job_id}:2"
    assert "JOB_RETRY_REQUESTED" in _kinds(store)


# ---------------------------------------------------------------- events / queries
def test_events_after_filters(store):
    store.append_event("RECIPE_RELEASED", principal=ADMIN, payload={"recipe_id": "X@v1"})
    _register(store)
    with pytest.raises(LineStoreError):
        store.append_event("NOT_A_KIND")
    all_events = store.events_after(0)
    assert [e["kind"] for e in all_events] == ["RECIPE_RELEASED", "SCAN_RECEIVED"]
    assert [e["kind"] for e in store.events_after(0, station_id="cell-1")] == ["SCAN_RECEIVED"]
    assert len(store.events_after(0, station_id="cell-1", include_global=True)) == 2
    assert store.events_after(all_events[0]["id"])[0]["kind"] == "SCAN_RECEIVED"
    assert store.events_after(0, kinds=["RECIPE_RELEASED"])[0]["principal"] == ADMIN
    assert store.last_event_id() == all_events[-1]["id"]
    from contracts.production.events import LineEventV1

    for event in all_events:
        LineEventV1.model_validate(event)


def test_list_jobs_latest_counts_heartbeats(store, clock):
    _register(store)
    j1, _ = _enqueue_inspect(store)
    clock.advance(1)
    j2, _ = store.enqueue_job(
        "recipe_compile", sku="P", recipe_id="P@v1", recipe_version=1, recipe_sha256=SHA_B, station_id="cell-1"
    )
    assert [j["job_id"] for j in store.list_jobs(station_id="cell-1")] == [j2, j1]
    assert [j["job_id"] for j in store.list_jobs(statuses=["queued"], sku="P")] == [j2]
    assert store.list_jobs(statuses=[]) == []
    before = store.get_job(j2)["created_at"]
    assert [j["job_id"] for j in store.list_jobs(before=before)] == [j1]
    store.lease_next("r1", 30)  # j1 (priority 100)
    store.fail_job(j1, "r1", "WORKER_CRASH", "x")
    assert store.latest_job_for_station("cell-1", ("failed",))["job_id"] == j1
    assert store.latest_job_for_station("cell-1") is None
    counts = store.counts()
    assert counts["queue_depth"] == 1 and counts["failed_last_24h"] == 1 and counts["jobs_by_status"]["failed"] == 1
    clock.advance(25 * 3600)
    assert store.counts()["failed_last_24h"] == 0
    store.put_heartbeat("runner", "i-1", {"job_id": None})
    clock.advance(5)
    beats = store.get_heartbeats()
    assert beats[0]["component"] == "runner" and beats[0]["age_s"] == pytest.approx(5.0)


def test_ingest_sources(store):
    assert not store.has_ingest_source("cell-1", "/in/a.ply", 10, 5)
    store.record_ingest_source("cell-1", "/in/a.ply", 10, 5, SHA_A)
    assert store.has_ingest_source("cell-1", "/in/a.ply", 10, 5)
    assert not store.has_ingest_source("cell-1", "/in/a.ply", 11, 5)


# ---------------------------------------------------------------- parity with C01
@pytest.mark.parametrize(
    ("mirror", "module", "name"),
    [
        ("EXECUTION_STATUSES", "contracts.production.line_job", "EXECUTION_STATUSES"),
        ("JOB_KINDS", "contracts.production.line_job", "JOB_KINDS"),
        ("STAGES", "contracts.production.line_job", "STAGES"),
        ("FAILURE_CODES", "contracts.production.line_job", "FAILURE_CODES"),
        ("PARSE_STATUSES", "contracts.production.line_job", "PARSE_STATUSES"),
        ("RESOLVE_STATUSES", "contracts.production.line_job", "RESOLVE_STATUSES"),
        ("INGEST_STATES", "contracts.production.line_job", "INGEST_STATES"),
        ("QUARANTINE_REASONS", "contracts.production.line_job", "QUARANTINE_REASONS"),
        ("SCAN_FORMATS", "contracts.production.line_job", "SCAN_FORMATS"),
        ("PRIORITY_BY_KIND", "contracts.production.line_job", "PRIORITY_BY_KIND"),
        ("MAX_ATTEMPTS_BY_KIND", "contracts.production.line_job", "MAX_ATTEMPTS_BY_KIND"),
        ("ENGINE_HARD_TIMEOUT_S", "contracts.production.line_job", "ENGINE_HARD_TIMEOUT_S"),
        ("ENGINE_RSS_LIMIT_BYTES", "contracts.production.line_job", "ENGINE_RSS_LIMIT_BYTES"),
        ("SCAN_FILE_SCHEMA", "contracts.production.line_job", "SCAN_FILE_SCHEMA"),
        ("LINE_JOB_SCHEMA", "contracts.production.line_job", "LINE_JOB_SCHEMA"),
        ("EVENT_KINDS", "contracts.production.events", "EVENT_KINDS"),
        ("LINE_EVENT_SCHEMA", "contracts.production.events", "LINE_EVENT_SCHEMA"),
        ("SUMMARY_STATES", "contracts.production.states", "SUMMARY_STATES"),
        ("VALIDITY_STATES", "contracts.production.states", "VALIDITY_STATES"),
        ("STAGE_STATUSES", "contracts.production.states", "STAGE_STATUSES"),
    ],
)
def test_vocabulary_mirrors_contracts(mirror, module, name):
    import importlib

    assert getattr(vocab, mirror) == getattr(importlib.import_module(module), name)


def test_derived_vocabularies_are_consistent():
    from contracts.production.line_job import JobResultSummaryV1

    assert tuple(JobResultSummaryV1.model_fields) == vocab.JOB_RESULT_SUMMARY_KEYS
    assert vocab.CONTENT_QUARANTINE_REASONS | vocab.RESOLVABLE_QUARANTINE_REASONS == vocab.QUARANTINE_REASONS
    assert set(vocab.RESOLVE_STATUS_TO_QUARANTINE_REASON.values()) == vocab.RESOLVABLE_QUARANTINE_REASONS
    assert set(vocab.RESOLVE_STATUS_TO_QUARANTINE_REASON) == vocab.RESOLVE_STATUSES - {"MATCHED"}
    assert vocab.ENGINE_INPUT_ERROR_CODES <= vocab.FAILURE_CODES
