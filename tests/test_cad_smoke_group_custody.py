"""Pure invented/mocked smoke-registry controls: no subprocess/native/raw use."""
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def harness():
    return _module("_invented_registry_harness", ROOT/"scripts/check_cad_guided_measurement.py")


@pytest.fixture
def custody(harness, tmp_path):
    directory = (tmp_path/"private-registry").resolve()
    directory.mkdir(mode=0o700)
    nonce = "a"*64
    registry = harness.SmokeGroupRegistry(directory, nonce, 10)
    request = dict(schema=harness.SMOKE_SCHEMA, nonce=nonce, slot=0, pytest_pid=10, pid=20, pgid=20)
    table = {10: dict(ppid=1, pgid=10, rss=10, state="S"),
             20: dict(ppid=10, pgid=20, rss=20, state="Ss"),
             21: dict(ppid=20, pgid=20, rss=30, state="S"),
             30: dict(ppid=10, pgid=10, rss=40, state="R"),
             99: dict(ppid=1, pgid=99, rss=999, state="S")}
    return SimpleNamespace(directory=directory, nonce=nonce, registry=registry, request=request, table=table)


def _publish(harness, custody, request=None):
    harness.smoke_publish(custody.directory, "0.request.json", request or custody.request)


def _update(custody, table, **kwargs):
    return custody.registry.update(custody.registry.capture_requests(), table, **kwargs)


def test_ack_requires_live_direct_session_child_and_retains_bound_request(harness, custody):
    _publish(harness, custody)
    _update(custody, custody.table)
    ack = harness.smoke_read(custody.directory/"0.ack.json")
    assert ack["decision"] == "ACK" and ack["pid"] == ack["pgid"] == 20
    assert ack["ack_phase"] == "MAIN_ADMISSION" and ack["ack_monotonic"] >= 0
    assert ack["pytest_pid"] == 10 and ack["nonce"] == custody.nonce
    assert ack["registration_sha256"] == harness.hashlib.sha256(harness.smoke_encoded(custody.request)).hexdigest()
    before = (custody.directory/"0.ack.json").read_bytes()
    _update(custody, custody.table)
    assert (custody.directory/"0.ack.json").read_bytes() == before
    assert custody.registry.active_groups() == {20} and custody.registry.rss(custody.table) == 100
    record = custody.registry.receipt()["registrations"][0]
    assert record["ack_phase"] == record["first_observed_phase"] == "MAIN_ADMISSION"
    assert record["ack_publication_completed_monotonic"] >= record["ack_monotonic"]


@pytest.mark.parametrize("fault", ("nonce", "pytest", "pgid", "slot", "extra", "parent", "session", "zombie", "missing"))
def test_invalid_registration_never_gets_ack(harness, custody, fault):
    request = dict(custody.request)
    if fault == "nonce":
        request["nonce"] = "b"*64
    elif fault == "pytest":
        request["pytest_pid"] = 999
    elif fault == "pgid":
        request["pgid"] = 10
    elif fault == "slot":
        request["slot"] = False
    elif fault == "extra":
        request["path"] = "/not-a-capability"
    elif fault == "parent":
        custody.table[20]["ppid"] = 1
    elif fault == "session":
        custody.table[20]["pgid"] = 10
    elif fault == "zombie":
        custody.table[20]["state"] = "Z"
    else:
        del custody.table[20]
    _publish(harness, custody, request)
    with pytest.raises(RuntimeError):
        _update(custody, custody.table)
    assert not custody.registry.active_groups()
    assert not (custody.directory/"0.ack.json").exists()


def test_acknowledged_orphans_remain_counted_and_owned_until_group_empty(harness, custody, monkeypatch):
    _publish(harness, custody)
    _update(custody, custody.table)
    orphans = {pid: dict(row, ppid=1) for pid, row in custody.table.items() if pid in (20, 21, 99)}
    _update(custody, orphans, pytest_exited=True)
    assert custody.registry.rss(orphans) == 50 and custody.registry.active_groups() == {20}
    killed = []
    monkeypatch.setattr(harness.os, "killpg", lambda pid, sig: killed.append(pid))
    custody.registry.kill_owned()
    assert killed == [20]
    _update(custody, {99: orphans[99]}, pytest_exited=True)
    assert not custody.registry.active_groups() and custody.registry.rss({99: orphans[99]}) == 0
    assert custody.registry.receipt()["all_owned_groups_empty_confirmed"]
    assert custody.registry.receipt()["registrations"][0]["empty_confirmed"]
    custody.registry.kill_owned()
    assert killed == [20]
    # A sampled-empty number is NOT ownership proof if it later reappears.
    with pytest.raises(RuntimeError, match="RETIRED_GROUP_REAPPEARED"):
        _update(custody, orphans, pytest_exited=True)
    custody.registry.kill_owned()
    assert killed == [20]


def test_duplicate_group_cannot_consume_second_registration(harness, custody):
    _publish(harness, custody)
    _update(custody, custody.table)
    harness.smoke_publish(custody.directory, "1.request.json", {**custody.request, "slot": 1})
    with pytest.raises(RuntimeError, match="REGISTRATION_BINDING"):
        _update(custody, custody.table)
    assert not (custody.directory/"1.ack.json").exists()


@pytest.mark.parametrize("name", ("2.request.json", "unbounded-extra"))
def test_registry_has_two_fixed_slots_no_extra_entries(harness, custody, name):
    (custody.directory/name).write_bytes(b"invented")
    with pytest.raises(RuntimeError, match="ENTRY_CAP"):
        _update(custody, custody.table)


def test_private_atomic_publication_is_exclusive_and_retains_bounded_stage(harness, custody):
    _publish(harness, custody)
    original = (custody.directory/"0.request.json").read_bytes()
    with pytest.raises(FileExistsError):
        _publish(harness, custody, {**custody.request, "nonce": "b"*64})
    assert (custody.directory/"0.request.json").read_bytes() == original
    assert sorted(p.name for p in custody.directory.iterdir()) == [".0.request.json.tmp", "0.request.json"]
    assert (custody.directory/"0.request.json").stat().st_nlink == 2
    assert (custody.directory/"0.request.json").stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("fault", ("permission", "symlink", "oversize", "duplicate_json"))
def test_registry_reads_fail_closed_on_untrusted_descriptor(harness, custody, fault):
    path = custody.directory/"invented.json"
    if fault == "symlink":
        path.symlink_to(custody.directory/"does-not-exist")
    else:
        path.write_bytes(b'{"x":1,"x":2}' if fault == "duplicate_json" else b"x"*(2049 if fault == "oversize" else 1))
        path.chmod(0o644 if fault == "permission" else 0o600)
    with pytest.raises((RuntimeError, OSError)):
        harness.smoke_read(path)


def test_ack_publication_failure_keeps_verified_group_owned(harness, custody, monkeypatch):
    _publish(harness, custody)
    monkeypatch.setattr(harness, "smoke_publish", lambda *a: (_ for _ in ()).throw(OSError("invented ACK failure")))
    with pytest.raises(OSError):
        _update(custody, custody.table)
    assert custody.registry.active_groups() == {20}
    assert not custody.registry.receipt()["registrations"][0]["acknowledged"]


def test_mutated_request_or_removed_ack_cannot_stay_successful(harness, custody):
    _publish(harness, custody)
    _update(custody, custody.table)
    (custody.directory/"0.request.json").write_bytes(harness.smoke_encoded({**custody.request, "nonce": "b"*64}))
    with pytest.raises(RuntimeError, match="REGISTRATION_CHANGED"):
        _update(custody, custody.table)
    (custody.directory/"0.request.json").write_bytes(harness.smoke_encoded(custody.request))
    (custody.directory/"0.ack.json").unlink()
    with pytest.raises(FileNotFoundError):
        _update(custody, custody.table)


@pytest.mark.parametrize("fault", ("valid", "wrong_ack", "timeout", "child_exit", "full", "cleanup_ack", "future_ack"))
def test_client_barrier_requires_exact_ack_and_two_slot_cap(harness, custody, monkeypatch, fault):
    monkeypatch.setattr(harness.os, "getpid", lambda: 10)
    monkeypatch.setattr(harness.os, "getppid", lambda: 9)
    if fault == "full":
        for slot in range(2):
            harness.smoke_publish(custody.directory, f"{slot}.request.json", {**custody.request, "slot": slot})
    def poll():
        if fault == "child_exit":
            return 1
        record = harness.smoke_read(custody.directory/"0.request.json")
        ack = dict(**record, decision="ACK", harness_pid=9,
                   registration_sha256=harness.hashlib.sha256(harness.smoke_encoded(record)).hexdigest(),
                   ack_phase="MAIN_ADMISSION", ack_monotonic=harness.time.monotonic())
        if fault == "wrong_ack":
            ack["pgid"] = 999
        elif fault == "cleanup_ack":
            ack["ack_phase"] = "CLEANUP_CUSTODY"
        elif fault == "future_ack":
            ack["ack_monotonic"] += 1000
        harness.smoke_publish(custody.directory, "0.ack.json", ack)
        return None
    child = SimpleNamespace(pid=20, poll=poll)
    if fault == "timeout":
        ticks = iter((0., 3.))
        monkeypatch.setattr(harness.time, "monotonic", lambda: next(ticks))
    if fault == "valid":
        assert harness.register_smoke_child(child, custody.directory, custody.nonce) == 0
    else:
        with pytest.raises(RuntimeError):
            harness.register_smoke_child(child, custody.directory, custody.nonce)


@pytest.mark.parametrize("failure", (False, True))
def test_popen_proxy_cannot_return_or_release_stdin_before_ack(harness, custody, monkeypatch, failure):
    tests = _module("_invented_supervisor_test_helpers", ROOT/"tests/test_v2_native_validity_supervisor.py")
    events = []
    child = SimpleNamespace(pid=20, poll=lambda: None, wait=lambda **kw: events.append(("reap", kw["timeout"])))
    fake_driver = SimpleNamespace(subprocess=SimpleNamespace(Popen=lambda *a, **kw: events.append("created") or child),
        os=SimpleNamespace(killpg=lambda pid, sig: events.append(("kill", pid))), signal=harness.signal)
    def acknowledge(*args):
        events.append("registration")
        if failure:
            raise RuntimeError("invented ACK refusal")
        events.append("ACK")
    monkeypatch.setattr(harness, "register_smoke_child", acknowledge)
    tests._install_smoke_group_proxy(fake_driver, monkeypatch, harness, custody.directory, custody.nonce)
    if failure:
        with pytest.raises(RuntimeError, match="ACK refusal"):
            fake_driver.subprocess.Popen(["invented"], start_new_session=True)
        assert events == ["created", "registration", ("kill", 20), ("reap", 2.)]
    else:
        assert fake_driver.subprocess.Popen(["invented"], start_new_session=True) is child
        events.append("caller may now write stdin")
        assert events == ["created", "registration", "ACK", "caller may now write stdin"]


def test_proxy_does_not_signal_an_already_reaped_pre_ack_child(harness, custody, monkeypatch):
    tests = _module("_invented_reaped_proxy", ROOT/"tests/test_v2_native_validity_supervisor.py")
    child = SimpleNamespace(pid=20, poll=lambda: 1, wait=lambda **kw: 1)
    fake_driver = SimpleNamespace(subprocess=SimpleNamespace(Popen=lambda *a, **kw: child),
        os=SimpleNamespace(killpg=lambda *args: pytest.fail("cannot signal a reaped numeric PGID")), signal=harness.signal)
    monkeypatch.setattr(harness, "register_smoke_child", lambda *a: (_ for _ in ()).throw(RuntimeError("exited")))
    tests._install_smoke_group_proxy(fake_driver, monkeypatch, harness, custody.directory, custody.nonce)
    with pytest.raises(RuntimeError, match="exited"):
        fake_driver.subprocess.Popen(["invented"], start_new_session=True)


@pytest.mark.parametrize("payload", (b"", b"garbage", b"{}", b'{"token":"wrong"}', b"x"*4097))
def test_both_invented_coordinators_refuse_eof_or_malformed_release(payload):
    tests = _module("_invented_stdin_gate_helpers", ROOT/"tests/test_v2_native_validity_supervisor.py")
    with pytest.raises(SystemExit, match="INVENTED_SMOKE_RELEASE_REQUIRED"):
        exec(tests._SMOKE_STDIN_GATE, {"sys": SimpleNamespace(stdin=SimpleNamespace(buffer=io.BytesIO(payload)))})


def test_invented_coordinator_accepts_only_exact_fixed_release():
    tests = _module("_invented_valid_stdin_gate", ROOT/"tests/test_v2_native_validity_supervisor.py")
    payload = b'{"token":"' + b"a"*64 + b'"}'
    exec(tests._SMOKE_STDIN_GATE, {"sys": SimpleNamespace(stdin=SimpleNamespace(buffer=io.BytesIO(payload)))})


@pytest.mark.parametrize("output", ("", "1 2 3 4", "1 2 3 nope S", "1 2 3 4 S\n1 2 3 4 S"))
def test_smoke_process_table_is_fail_closed_without_running_ps(harness, monkeypatch, output):
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=output))
    with pytest.raises(RuntimeError):
        harness.smoke_process_table()


def test_stop_scans_kills_and_confirms_reparented_groups_without_wait(harness, custody, monkeypatch):
    _publish(harness, custody)
    tables = iter((custody.table, {20: dict(ppid=1, pgid=20, rss=20, state="S")}, {}))
    monkeypatch.setattr(harness, "smoke_process_table", lambda: next(tables))
    monkeypatch.setattr(harness.time, "sleep", lambda _: None)
    monkeypatch.setattr(harness.time, "monotonic", lambda: 0.)
    killed = []
    monkeypatch.setattr(harness.os, "killpg", lambda pid, sig: killed.append(pid))
    receipt = harness.stop_smoke_groups(custody.registry, SimpleNamespace(pid=10, poll=lambda: 0), 1.)
    assert killed.count(10) == 1 and killed.count(20) == 2
    assert receipt["all_owned_groups_empty_confirmed"] and not receipt["requires_group_attention"]
    assert not receipt["cleanup_errors"] and receipt["registrations"][0]["empty_confirmed"]


def test_stop_monitor_failure_still_kills_previously_owned_groups(harness, custody, monkeypatch):
    _publish(harness, custody)
    _update(custody, custody.table)
    monkeypatch.setattr(harness, "smoke_process_table", lambda: (_ for _ in ()).throw(RuntimeError("invented ps failure")))
    monkeypatch.setattr(harness.time, "monotonic", lambda: 2.)
    killed = []
    monkeypatch.setattr(harness.os, "killpg", lambda pid, sig: killed.append(pid))
    direct_kills = []
    receipt = harness.stop_smoke_groups(custody.registry,
        SimpleNamespace(pid=10, poll=lambda: None, kill=lambda: direct_kills.append(10)), 1.)
    assert set(killed) == {10, 20} and receipt["requires_group_attention"] and receipt["cleanup_errors"]
    assert direct_kills == [10]


def test_missing_live_pytest_row_never_retires_owned_group(harness, custody, monkeypatch):
    _publish(harness, custody)
    _update(custody, custody.table)
    missing = {pid: row for pid, row in custody.table.items() if row["pgid"] != 10}
    with pytest.raises(RuntimeError, match="ACTIVE_PYTEST_ROW_MISSING"):
        _update(custody, missing, pytest_exited=False)
    assert not custody.registry.pytest_empty
    # A later complete sample recovers naturally; there was no false retirement.
    _update(custody, custody.table, pytest_exited=False)
    monkeypatch.setattr(harness, "smoke_process_table", lambda: missing)
    monkeypatch.setattr(harness.time, "monotonic", lambda: 2.)
    killed, direct = [], []
    monkeypatch.setattr(harness.os, "killpg", lambda pid, sig: killed.append(pid))
    receipt = harness.stop_smoke_groups(custody.registry,
        SimpleNamespace(pid=10, poll=lambda: None, kill=lambda: direct.append(10)), 1.)
    assert set(killed) == {10, 20} and direct == [10]
    assert not receipt["pytest_group_empty_confirmed"]
    assert receipt["requires_group_attention"] and receipt["cleanup_errors"]


@pytest.mark.parametrize("keep", (False, True))
def test_registry_cache_is_preserved_when_custody_is_unconfirmed(harness, tmp_path, monkeypatch, keep):
    directory = tmp_path/"invented-cache"
    directory.mkdir(mode=0o700)
    monkeypatch.setattr(harness.tempfile, "mkdtemp", lambda **kwargs: str(directory))
    manager = harness.SmokeCache()
    with manager as path:
        assert Path(path) == directory
        manager.keep = keep
    assert directory.exists() == keep


def test_named_registry_and_v4_suites_bind_exact_sources(harness):
    assert harness.TESTS["cad_smoke_group_custody"] == "tests/test_cad_smoke_group_custody.py"
    assert "tests/test_v2_native_validity_supervisor.py" in harness.V1_SOURCE_BINDINGS["cad_smoke_group_custody"]
    assert harness.TESTS["mesh_topology_v4"] == "tests/test_cad_guided_mesh_topology_v4.py"
    assert harness.TESTS["plane_hypotheses_v4"] == "tests/test_cad_guided_plane_hypotheses_v4.py"
    for suite in ("mesh_topology_v4", "plane_hypotheses_v4"):
        assert set(harness.V4_PLANE_PROTOCOLS) <= set(harness.V1_SOURCE_BINDINGS[suite])


def test_publication_during_ps_is_deferred_until_next_captured_sample(harness, custody, monkeypatch):
    calls = []
    original_capture = custody.registry.capture_requests
    def capture():
        calls.append("capture")
        return original_capture()
    def sample():
        calls.append("ps")
        if calls.count("ps") == 1:
            # Request becomes visible AFTER capture. An older/missing coordinator
            # row cannot falsely reject it, because it is not in this batch.
            _publish(harness, custody)
            return {pid: row for pid, row in custody.table.items() if row["pgid"] != 20}
        return custody.table
    monkeypatch.setattr(custody.registry, "capture_requests", capture)
    monkeypatch.setattr(harness, "smoke_process_table", sample)
    child = SimpleNamespace(pid=10, poll=lambda: None)
    custody.registry.sample(child, phase="MAIN_ADMISSION")
    assert not custody.registry.records and not (custody.directory/"0.ack.json").exists()
    custody.registry.sample(child, phase="MAIN_ADMISSION")
    assert calls == ["capture", "ps", "capture", "ps"]
    assert custody.registry.records[0]["acknowledged"]


def test_captured_request_gets_a_fresh_process_sample_before_ack(harness, custody, monkeypatch):
    _publish(harness, custody)
    events = []
    original_read, original_publish = harness.smoke_read, harness.smoke_publish
    def read(path):
        if path.name == "0.request.json":
            events.append("request")
        return original_read(path)
    def ps():
        events.append("ps")
        assert events == ["request", "ps"]
        return custody.table
    def publish(directory, filename, value):
        events.append("ACK")
        assert events == ["request", "ps", "ACK"]
        return original_publish(directory, filename, value)
    monkeypatch.setattr(harness, "smoke_read", read)
    monkeypatch.setattr(harness, "smoke_process_table", ps)
    monkeypatch.setattr(harness, "smoke_publish", publish)
    custody.registry.sample(SimpleNamespace(pid=10, poll=lambda: None), phase="MAIN_ADMISSION")
    assert events == ["request", "ps", "ACK"]


def test_cleanup_retains_and_kills_pending_group_without_issuing_ack(harness, custody, monkeypatch):
    _publish(harness, custody)
    monkeypatch.setattr(harness, "smoke_process_table", lambda: custody.table)
    monkeypatch.setattr(harness, "smoke_publish", lambda *a: pytest.fail("cleanup must never ACK"))
    custody.registry.sample(SimpleNamespace(pid=10, poll=lambda: None), phase="CLEANUP_CUSTODY")
    record = custody.registry.receipt()["registrations"][0]
    assert record["first_observed_phase"] == "CLEANUP_CUSTODY"
    assert not record["acknowledged"] and record["ack_phase"] is None and record["ack_monotonic"] is None
    assert record["ack_publication_completed_monotonic"] is None
    assert not (custody.directory/"0.ack.json").exists()
    killed = []
    monkeypatch.setattr(harness.os, "killpg", lambda pid, sig: killed.append(pid))
    custody.registry.kill_owned()
    assert set(killed) == {10, 20}


def test_descriptor_open_at_publication_is_stable_when_publisher_returns(harness, custody, monkeypatch):
    opened = []
    original_link = harness.os.link
    def link(stage, target):
        original_link(stage, target)
        fd = harness.os.open(target, harness.os.O_RDONLY | harness.os.O_NOFOLLOW)
        opened.append((fd, harness.os.fstat(fd)))
    monkeypatch.setattr(harness.os, "link", link)
    try:
        _publish(harness, custody)
        assert len(opened) == 1
        fd, before = opened[0]
        after = harness.os.fstat(fd)
        assert before.st_nlink == after.st_nlink == 2
        assert before.st_ctime_ns == after.st_ctime_ns
        assert harness.smoke_read(custody.directory/"0.request.json") == custody.request
    finally:
        for fd, _ in opened:
            harness.os.close(fd)


def test_prior_stage_unlink_interleaving_still_fails_closed_in_reader(harness, custody, monkeypatch):
    _publish(harness, custody)
    original_fstat, calls = harness.os.fstat, []
    def fstat(fd):
        calls.append(fd)
        if len(calls) == 2:
            (custody.directory/".0.request.json.tmp").unlink()
        return original_fstat(fd)
    monkeypatch.setattr(harness.os, "fstat", fstat)
    with pytest.raises(RuntimeError, match="REGISTRY_FILE_CHANGED"):
        harness.smoke_read(custody.directory/"0.request.json")


def test_first_exception_reason_and_stage_survive_cleanup_permission_error(harness, monkeypatch):
    monkeypatch.setattr(harness.time, "monotonic", lambda: 10.)
    with pytest.raises(RuntimeError) as caught:
        harness.smoke_require(False, "SMOKE_COORDINATOR_NOT_LIVE_DIRECT_SESSION_CHILD")
    primary = harness.first_smoke_failure(None, caught.value, "MAIN_ADMISSION_VALIDATE_CAPTURED")
    cleanup = harness.smoke_failure(PermissionError("untrusted/path/must/not/be/retained"), "CLEANUP_GROUP_KILL", pgid=20)
    assert harness.first_smoke_failure(primary, PermissionError(), "CLEANUP_GROUP_KILL") is primary
    assert primary == dict(exception_type="RuntimeError", reason="SMOKE_COORDINATOR_NOT_LIVE_DIRECT_SESSION_CHILD",
        stage="MAIN_ADMISSION_VALIDATE_CAPTURED", monotonic=10.)
    assert cleanup == dict(exception_type="PermissionError", reason="SMOKE_OS_PERMISSION_DENIED",
        stage="CLEANUP_GROUP_KILL", monotonic=10., pgid=20)


def test_proxy_reap_error_does_not_replace_original_registration_error(harness, custody, monkeypatch, capsys):
    tests = _module("_invented_proxy_primary_error", ROOT/"tests/test_v2_native_validity_supervisor.py")
    original = RuntimeError("original registration failure")
    def reap(**kwargs):
        raise PermissionError("cleanup failure")
    child = SimpleNamespace(pid=20, poll=lambda: None, wait=reap)
    fake_driver = SimpleNamespace(subprocess=SimpleNamespace(Popen=lambda *a, **kw: child),
        os=SimpleNamespace(killpg=lambda *args: None), signal=harness.signal)
    monkeypatch.setattr(harness, "register_smoke_child", lambda *a: (_ for _ in ()).throw(original))
    tests._install_smoke_group_proxy(fake_driver, monkeypatch, harness, custody.directory, custody.nonce)
    with pytest.raises(RuntimeError) as caught:
        fake_driver.subprocess.Popen(["invented"], start_new_session=True)
    assert caught.value is original
    report = harness.json.loads(capsys.readouterr().out)
    assert report["primary_error"]["stage"] == "SMOKE_CLIENT_REGISTRATION"
    assert report["primary_error"]["exception_type"] == "RuntimeError"
    assert report["cleanup_errors"][0]["stage"] == "SMOKE_CLIENT_REAP"
    assert report["cleanup_errors"][0]["exception_type"] == "PermissionError"
