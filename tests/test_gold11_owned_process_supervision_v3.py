"""SOURCE CANDIDATE: deterministic mock-only generation/ABI qualification.

No real subprocess, libproc call, signal, geometry, application or data access.
Only the named supervisor source is loaded after a separately frozen test gate.
These mocks are not host/API qualification and cannot establish containment.
"""
import ast
import ctypes
import errno
from pathlib import Path
import signal
import types

import pytest


SOURCE = Path(__file__).resolve().parents[1]/"scripts/gold11_owned_process_supervision_v3.py"


@pytest.fixture
def helper():
    module = types.ModuleType("_invented_owned_process_supervision_v3")
    exec(compile(SOURCE.read_bytes(), str(SOURCE), "exec"), module.__dict__)
    return module


def row(pid, *, unique=None, parent=100, parent_unique=10000, group=200,
        version=1, status=2, rss=1000, flags=0):
    return dict(pid=pid, uniqueid=pid*100 if unique is None else unique,
        ppid=parent, parent_uniqueid=parent_unique, pgid=group, idversion=version,
        status=status, flags=flags, rss=rss, uid=501, gid=20, ruid=501, rgid=20,
        start_sec=1000, start_usec=42)


def receipt():
    return dict(observed_foreign_group_escape_detected=False,
        observed_process_identity_change_detected=False,
        observed_foreign_pid_count_peak=0,
        observed_foreign_group_escape_unconfirmed_after_cleanup=False,
        observed_identity_unknown_still_present=False,
        observed_identity_unknown_pid_count_peak=0,
        observed_foreign_identity_changed_or_unknown=False)


class Census:
    """Current-generation records and scoped listings, no OS access."""
    def __init__(self):
        self.rows = {100: row(100, parent=1, parent_unique=1, group=50), 200: row(200)}
        self.errors, self.lists, self.signals = {}, [], []
        self.list_override, self.signal_override = None, None
    def begin_sample(self, end):
        assert end == 10
    def check_time(self):
        pass
    def inspect(self, pid, *, rss=False):
        if pid in self.errors:
            raise self.errors[pid]("INVENTED_IDENTITY_UNAVAILABLE")
        answer = self.rows.get(pid)
        return None if answer is None else dict(answer)
    def list_pids(self, kind, identifier):
        self.lists.append((kind, identifier))
        if self.list_override is not None:
            return self.list_override(kind, identifier)
        key = "pgid" if kind == 2 else "ppid"
        return tuple(pid for pid, value in sorted(self.rows.items()) if value[key] == identifier)
    def signal_owned(self, expected):
        self.signals.append(dict(expected))
        if self.signal_override is not None:
            return self.signal_override(expected)
        current = self.inspect(expected["pid"])
        return dict(status="ATOMIC_SIGNAL_ACCEPTED", current=current)


def custody(helper, outer=True):
    api, output, stops = Census(), receipt(), []
    if not outer:
        api.rows[200]["pgid"] = 50
    state = helper._Custody(api, 200, api.rows[100], outer, output, stops.append)
    return api, state, output, stops


def test_no_numeric_signal_poll_or_import_time_library_load():
    tree = ast.parse(SOURCE.read_text())
    forbidden = {"kill", "killpg", "terminate", "poll"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden for node in ast.walk(tree))
    imports = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    assert imports == {"ctypes", "errno", "json", "math", "os", "selectors", "signal", "subprocess", "sys", "time"}
    assert not any(isinstance(node, ast.ImportFrom) for node in ast.walk(tree))
    assert all(not isinstance(node, ast.Expr) or isinstance(node.value, ast.Constant) for node in tree.body)


def test_abi_sizes_are_literal_qualified_layout_not_a_kernel_probe(helper):
    assert [ctypes.sizeof(value) for value in (helper._BSDInfo, helper._UniqueInfo,
        helper._BSDWithUnique, helper._TaskInfo, helper._AuditToken)] == [136, 56, 192, 96, 32]


@pytest.mark.parametrize("new_group", (50, 200, 999))
def test_recycled_pid_retires_old_generation_without_escape_or_wrong_signal(helper, new_group):
    api, state, output, stops = custody(helper)
    api.rows[201] = row(201, parent=200, parent_unique=20000)
    state.snapshot(10)
    api.rows[201] = row(201, unique=999999, parent=1, parent_unique=1, group=new_group)
    state.snapshot(10)
    assert state.counters["replacement_retirements"] == 1
    assert not stops and not output["observed_process_identity_change_detected"]
    if new_group == 200:
        assert state.known[201]["uniqueid"] == 999999  # Fresh group-scope admission.
    else:
        assert 201 not in state.known
    state.signal_current_owned()
    assert all(value["pid"] != 201 or value["uniqueid"] == 999999 for value in api.signals)


def test_thousands_of_verified_retirements_do_not_grow_active_pid_history(helper):
    api, state, output, stops = custody(helper)
    for generation in range(1, 4097):
        api.rows[201] = row(201, unique=30000+generation, parent=200, parent_unique=20000)
        state.snapshot(10)
        api.rows[201] = row(201, unique=90000+generation, parent=1, parent_unique=1, group=999)
        state.snapshot(10)
    state.publish()
    assert state.counters["replacement_retirements"] == 4096
    assert state.counters["tracked_peak"] == 2 and len(state.known) == 1
    assert len(state.events) <= helper.MAX_IDENTITY_EVENTS
    assert not stops and not output["observed_foreign_group_escape_detected"]


def test_same_generation_exec_refreshes_version_without_lifetime_failure(helper):
    api, state, output, stops = custody(helper)
    api.rows[200]["idversion"] = 2
    state.snapshot(10)
    assert state.counters["exec_version_updates"] == 1
    assert state.known[200]["uniqueid"] == 20000
    assert not stops and not output["observed_process_identity_change_detected"]


def test_same_generation_group_escape_remains_failure_after_exit_and_reuse(helper):
    api, state, output, stops = custody(helper)
    api.rows[201] = row(201, parent=200, parent_unique=20000)
    state.snapshot(10)
    api.rows[201]["pgid"] = 999
    state.snapshot(10)
    assert output["observed_foreign_group_escape_detected"] and stops
    api.rows[201] = row(201, unique=90000, parent=1, parent_unique=1, group=50)
    state.snapshot(10)
    state.publish()
    assert 201 not in state.known
    assert output["observed_foreign_group_escape_detected"]
    assert not output["observed_foreign_group_escape_unconfirmed_after_cleanup"]
    assert any(event["kind"] == "SAME_GENERATION_GROUP_CHANGE" for event in state.events)


def test_authenticated_descendant_is_retained_after_reparenting(helper):
    api, state, _, _ = custody(helper, outer=False)
    api.rows[201] = row(201, parent=200, parent_unique=20000, group=50)
    state.snapshot(10)
    api.rows[201]["ppid"] = 1
    # Original parent unique ID remains as in XNU ordinary reparenting.
    observed = state.snapshot(10)
    assert 201 in observed["retained"] and state.counters["reparent_updates"] == 1


def test_inner_scope_does_not_admit_unrelated_shared_group_member(helper):
    api, state, _, _ = custody(helper, outer=False)
    api.rows[201] = row(201, parent=100, parent_unique=10000, group=50)
    state.snapshot(10)
    state.signal_current_owned()
    assert set(state.known) == {200} and all(value["pid"] == 200 for value in api.signals)
    assert all(kind == 6 for kind, _ in api.lists)


def test_new_foreign_descendant_requires_positive_parent_generation(helper):
    api, state, output, stops = custody(helper)
    api.rows[201] = row(201, parent=200, parent_unique=99999, group=999)
    state.snapshot(10)
    assert 201 not in state.known and not stops
    api.rows[201]["parent_uniqueid"] = 20000
    state.snapshot(10)
    assert 201 in state.known and output["observed_foreign_group_escape_detected"]


@pytest.mark.parametrize("error", (PermissionError, TimeoutError, RuntimeError))
def test_failed_owned_identity_query_is_not_retirement_or_zero_rss(helper, error):
    api, state, _, _ = custody(helper)
    api.rows[201] = row(201, parent=200, parent_unique=20000)
    state.snapshot(10)
    api.errors[201] = error
    with pytest.raises(error):
        state.snapshot(10)
    assert 201 in state.known and 201 in state.unknown
    assert state.counters["verified_absence_retirements"] == 0


def test_explicit_absence_retires_but_atomic_esrch_does_not(helper):
    api, state, _, _ = custody(helper)
    api.rows[201] = row(201, parent=200, parent_unique=20000)
    state.snapshot(10)
    api.signal_override = lambda expected: dict(status="TOKEN_STALE_OR_EXIT_NOT_RETIREMENT_PROOF", current=dict(expected))
    state.signal_current_owned()
    assert 201 in state.known
    del api.rows[201]
    state.snapshot(10)
    assert 201 not in state.known and state.counters["verified_absence_retirements"] == 1


@pytest.mark.parametrize("change", ("replaced", "missing", "reaped"))
def test_direct_anchor_loss_never_signals_any_numeric_pid(helper, change):
    api, state, _, _ = custody(helper)
    if change == "replaced":
        api.rows[200]["uniqueid"] += 1
    elif change == "missing":
        del api.rows[200]
    else:
        state.reaped = True
    with pytest.raises(RuntimeError):
        state.signal_current_owned()
    assert not api.signals


def test_zombie_root_remains_anchor_while_live_descendant_is_outstanding(helper):
    api, state, _, _ = custody(helper)
    api.rows[200].update(status=5, rss=None)
    api.rows[201] = row(201, parent=1, parent_unique=20000)
    observed = state.snapshot(10)
    assert observed["done"] and observed["group"] == {201}
    assert not state.reaped and 200 in state.known


@pytest.mark.parametrize("cap_name", ("MAX_TRACKED_IDENTITIES", "MAX_ADMITTED_IDENTITIES"))
def test_tracking_cap_is_precharged_before_new_member_insertion(helper, cap_name):
    api, state, _, _ = custody(helper)
    setattr(helper, cap_name, 1)
    api.rows[201] = row(201, parent=200, parent_unique=20000)
    with pytest.raises(RuntimeError, match="TRACKING_CAP"):
        state.snapshot(10)
    assert 201 not in state.known and state.counters["admitted"] == 2


def test_transient_unknown_peak_survives_later_successful_census(helper):
    api, state, output, _ = custody(helper)
    api.errors[200] = PermissionError
    with pytest.raises(PermissionError):
        state.snapshot(10)
    del api.errors[200]
    state.snapshot(10)
    state.publish()
    assert output["observed_identity_unknown_pid_count_peak"] == 1
    assert not output["observed_identity_unknown_still_present"]


def test_late_escape_identity_evidence_survives_full_normal_event_log(helper):
    api, state, output, _ = custody(helper)
    for index in range(helper.MAX_IDENTITY_EVENTS+1):
        api.rows[201] = row(201, unique=30000+index, parent=200, parent_unique=20000)
        state.snapshot(10)
        del api.rows[201]
        state.snapshot(10)
    api.rows[202] = row(202, parent=200, parent_unique=20000)
    state.snapshot(10)
    api.rows[202]["pgid"] = 999
    state.snapshot(10)
    state.publish()
    # Either reserve first-error evidence or a distinct bounded error ledger;
    # a discarded count alone cannot diagnose the late actual failing PID.
    records = output["identity_events"] + output.get("identity_error_events", [])
    assert any(value["kind"] == "SAME_GENERATION_GROUP_CHANGE"
        and value["before"]["pid"] == 202 and value["after"]["pgid"] == 999 for value in records)


def native_without_library(helper):
    api = helper._MacOSProcessAPI.__new__(helper._MacOSProcessAPI)
    api.work = dict(coherence_retries=0, explicit_esrch_observations=0, token_stale_or_exit=0)

    api.work.update(task_rss_esrch_observations=0, exit_transition_entries=0,
        exit_transition_waits=0, exit_transition_wait_seconds=0.,
        exit_transition_resolved_rss=0, exit_transition_resolved_zombie=0,
        exit_transition_resolved_absence=0, exit_transition_unresolved=0,
        exit_transition_diagnostics_discarded=0)
    api.exit_transition_diagnostics, api.exit_transition_first_failure = [], None
    api.lib = types.SimpleNamespace(proc_pidinfo=object(), proc_signal_with_audittoken=object())
    return api


@pytest.mark.parametrize("size,error,absent", ((0, errno.ESRCH, True),
    (0, errno.EPERM, False), (0, 0, False), (191, 0, False), (193, 0, False)))
def test_only_exact_kernel_esrch_identity_response_is_absence(helper, size, error, absent):
    api = native_without_library(helper)
    api._call = lambda *args: (size, error)
    if absent:
        assert api._bsd(200) is None
    else:
        with pytest.raises(RuntimeError):
            api._bsd(200)


def test_identity_flavor18_includes_zombies_and_returns_all_bound_fields(helper):
    api = native_without_library(helper)
    def call(kind, function, pid, flavor, arg, pointer, size):
        assert (kind, pid, flavor, arg, size) == ("identity", 200, 18, 1, 192)
        record = pointer._obj
        record.bsd.pid, record.bsd.ppid, record.bsd.pgid = 200, 100, 200
        record.bsd.status, record.bsd.start_usec = 5, 999999
        record.unique.uniqueid, record.unique.idversion = 20000, 12
        return 192, 0
    api._call = call
    answer = api._bsd(200)
    assert answer["status"] == 5 and answer["uniqueid"] == 20000 and answer["idversion"] == 12


@pytest.mark.parametrize("change", ("uniqueid", "idversion", "pgid", "ppid"))
def test_rss_is_bracketed_by_coherent_identity_and_group(helper, change):
    api = native_without_library(helper)
    first, second = row(200), row(200)
    second[change] += 1
    sequence = iter((first, second, first, second))
    api._bsd = lambda pid: dict(next(sequence))
    def call(kind, function, pid, flavor, arg, pointer, size):
        assert kind == "rss" and flavor == 4 and size == 96
        pointer._obj.resident_size = 12345
        return 96, 0
    api._call = call
    with pytest.raises(RuntimeError, match="DID_NOT_STABILIZE"):
        api.inspect(200, rss=True)
    assert api.work["coherence_retries"] == 2


@pytest.mark.parametrize("case,expected", (("absent", "VERIFIED_ABSENT"),
    ("replacement", "REPLACEMENT_NOT_SIGNALED"), ("group", "GROUP_CHANGE_NOT_SIGNALED"),
    ("zombie", "ZOMBIE_NOT_SIGNALED")))
def test_pre_signal_identity_mismatch_never_reaches_atomic_signal(helper, case, expected):
    api = native_without_library(helper)
    original, current = row(200), row(200)
    if case == "absent":
        current = None
    elif case == "replacement":
        current["uniqueid"] += 1
    elif case == "group":
        current["pgid"] += 1
    else:
        current["status"] = 5
    api.inspect = lambda pid: current
    api._call = lambda *args: pytest.fail("No signal operation may occur")
    assert api.signal_owned(original)["status"] == expected


@pytest.mark.parametrize("kernel_result", (0, errno.ESRCH, errno.EPERM))
def test_atomic_signal_binds_fresh_exec_token_and_has_no_numeric_fallback(helper, kernel_result):
    api = native_without_library(helper)
    original, current = row(200), row(200, version=7)
    api.inspect = lambda pid: dict(current)
    calls = []
    def call(kind, function, pointer, signum):
        calls.append((kind, tuple(pointer._obj.val), signum))
        return kernel_result, 0
    api._call = call
    if kernel_result == errno.EPERM:
        with pytest.raises(RuntimeError, match="DENIED"):
            api.signal_owned(original)
    else:
        answer = api.signal_owned(original)
        assert answer["status"] == ("ATOMIC_SIGNAL_ACCEPTED" if kernel_result == 0
            else "TOKEN_STALE_OR_EXIT_NOT_RETIREMENT_PROOF")
    assert len(calls) == 1 and calls[0][0] == "signal" and calls[0][2] == signal.SIGKILL
    assert calls[0][1][5] == 200 and calls[0][1][7] == 7


# Public-loop controls below use the actual _Custody implementation with only
# invented process rows. No fixture delegates to subprocess, selectors, file
# descriptors, wall clocks, signals, libproc, or a prior process fixture.
class FakeClock:
    def __init__(self): self.now = 0.
    def monotonic(self): return self.now
    def sleep(self, seconds):
        assert 0 <= seconds <= .02
        self.now += seconds


class FakePipe:
    def __init__(self, machine, name):
        self.machine, self.name, self.closed, self.writes = machine, name, False, []
    def fileno(self): return 17
    def write(self, blob):
        self.writes.append(blob)
        self.machine.fail("stdin_write")
        return len(blob)
    def close(self):
        self.closed = True
        self.machine.events.append(self.name+"_close")
        self.machine.fail(self.name+"_close")


class FakeChild:
    pid = 200
    def __init__(self, machine):
        self.machine, self.returncode, self.reaped = machine, None, False
        self.stdin, self.stdout = FakePipe(machine, "stdin"), FakePipe(machine, "stdout")
    def wait(self, timeout):
        assert 0 < timeout <= 5
        assert self.machine.zombie and not self.reaped
        assert not any(value["pgid"] == self.machine.group and value["parent_uniqueid"] == 20000
            for value in self.machine.extra.values())
        self.machine.events.append("wait")
        self.machine.waits.append(timeout)
        self.machine.fail("wait")
        self.reaped = True
        self.returncode = -signal.SIGKILL if self.machine.killed_root else self.machine.eof_code
        return self.returncode
    def poll(self): pytest.fail("The direct child must remain unreaped until explicit wait")
    def kill(self): pytest.fail("Numeric child signaling is forbidden")
    def terminate(self): pytest.fail("Numeric child signaling is forbidden")


class FakeSelector:
    def __init__(self, machine): self.machine, self.mapping = machine, {}
    def register(self, pipe, event):
        assert event == 1
        self.machine.fail("register")
        self.mapping[17] = types.SimpleNamespace(fileobj=pipe)
    def unregister(self, pipe):
        self.machine.fail("unregister")
        self.mapping.pop(17)
    def get_map(self): return self.mapping
    def select(self, timeout):
        assert 0 < timeout <= .05
        self.machine.clock.now += min(timeout, .01)
        self.machine.fail("select")
        return [(value, 1) for value in self.mapping.values()]
    def close(self):
        self.machine.events.append("selector_close")
        self.machine.fail("selector_close")


class FakeLog:
    def __init__(self, machine): self.machine, self.blob = machine, bytearray()
    def __enter__(self): return self
    def __exit__(self, *args):
        self.machine.events.append("log_close")
        self.machine.fail("log_close")
        return False
    def write(self, blob):
        self.machine.fail("log_write")
        self.blob.extend(blob)
        return len(blob)
    def flush(self): self.machine.fail("log_flush")


class PublicCensus:
    """Fake native API; the actual public loop and _Custody remain connected."""
    def __init__(self, machine):
        self.machine, self.phase = machine, "work"
        self.work = dict(work_samples=0, cleanup_samples=0)
        self.exit_transition_diagnostics, self.exit_transition_first_failure = [], None
    def begin_sample(self, end):
        self.work[self.phase+"_samples"] += 1
        self.machine.clock.now += .001
        self.end = end
        self.check_time()
    def check_time(self):
        if self.machine.clock.now >= self.end:
            raise TimeoutError("INVENTED_NATIVE_DEADLINE")
    def inspect(self, pid, *, rss=False):
        self.machine.inspections.append((pid, rss, self.phase))
        self.machine.fail("inspect_"+str(pid))
        if pid == 100:
            return row(100, parent=1, parent_unique=1, group=50, rss=1000)
        if pid == 200 and self.machine.started and not self.machine.child.reaped:
            answer = row(200, group=self.machine.group,
                status=5 if self.machine.zombie else 2,
                rss=None if self.machine.zombie else 2000)
            if self.machine.bad_initial_auth:
                answer["parent_uniqueid"] = 99999
            return answer
        answer = self.machine.extra.get(pid)
        return None if answer is None else dict(answer)
    def list_pids(self, kind, identifier):
        self.machine.events.append("list_after_reap" if self.machine.child.reaped else "list_before_reap")
        self.machine.fail("list")
        if self.machine.child.reaped and kind == 2 and self.machine.nonempty_after_reap:
            return (777,)
        rows = {pid: self.inspect(pid) for pid in (100, 200)}
        rows.update(self.machine.extra)
        key = "pgid" if kind == 2 else "ppid"
        return tuple(pid for pid, value in sorted(rows.items())
            if value is not None and value[key] == identifier)
    def signal_owned(self, expected):
        assert not self.machine.child.reaped, "No signal after releasing the root anchor"
        current = self.inspect(expected["pid"])
        assert current is not None and current["uniqueid"] == expected["uniqueid"]
        assert current["pgid"] == expected["pgid"] == self.machine.group
        self.machine.signals.append(dict(expected))
        self.machine.events.append("signal_"+str(expected["pid"]))
        self.machine.fail("signal")
        if not self.machine.ignore_signals:
            if expected["pid"] == 200:
                self.machine.zombie, self.machine.killed_root = True, True
            else:
                del self.machine.extra[expected["pid"]]
        return dict(status="ATOMIC_SIGNAL_ACCEPTED", current=current)


class PublicMachine:
    def __init__(self, helper):
        self.helper, self.clock = helper, FakeClock()
        self.child, self.selector, self.log = FakeChild(self), FakeSelector(self), FakeLog(self)
        self.api = PublicCensus(self)
        self.blocks, self.failures, self.extra = [b"ok\n", b""], {}, {}
        self.events, self.waits, self.signals, self.starts, self.inspections = [], [], [], [], []
        self.started, self.zombie, self.killed_root = False, False, False
        self.bad_initial_auth = self.nonempty_after_reap = self.ignore_signals = False
        self.eof_code, self.group, self.factory_calls = 0, 200, 0
        self.exit_at_eof, self.after_read = True, None
        self.cache = Path("/invented/no-access/cache")
        self.log_path = types.SimpleNamespace(open=self.open_log)
        helper.time = self.clock
        helper.os = types.SimpleNamespace(getpid=lambda: 100, read=self.read)
        helper.signal = types.SimpleNamespace(SIGCHLD=20, SIG_DFL=0, SIGKILL=9, getsignal=lambda _: 0)
        helper.selectors = types.SimpleNamespace(DefaultSelector=lambda: self.selector, EVENT_READ=1)
        helper.subprocess = types.SimpleNamespace(Popen=self.popen, PIPE=-1, STDOUT=-2)
        helper._MacOSProcessAPI = self.factory
    def fail(self, stage):
        if stage in self.failures:
            raise self.failures[stage]("INVENTED_"+stage)
    def factory(self):
        self.factory_calls += 1
        self.fail("factory")
        return self.api
    def popen(self, command, **kwargs):
        self.starts.append((command, kwargs))
        self.fail("popen")
        self.started = True
        self.group = 200 if kwargs["start_new_session"] else 50
        return self.child
    def open_log(self, mode):
        assert mode == "xb"
        self.fail("log_open")
        return self.log
    def read(self, fd, limit):
        assert (fd, limit) == (17, 65536)
        self.fail("read")
        block = self.blocks.pop(0) if self.blocks else b""
        if not block and self.exit_at_eof:
            self.zombie = True  # Critically, Popen.returncode is still None.
        if self.after_read is not None:
            self.after_read(self)
        return block
    def run(self, request=None, **kwargs):
        options = dict(cache=self.cache, log_path=self.log_path, wall_seconds=2,
            rss_limit=10000, log_limit=100, new_session=True)
        options.update(kwargs)
        return self.helper.supervise_process(["invented-python", "not-a-real-command"],
            {"answer": 42} if request is None else request, **options)


@pytest.mark.parametrize("outer", (False, True))
def test_public_eof_keeps_zombie_until_reap_then_closes_owned_controls(helper, outer):
    machine = PublicMachine(helper)
    result = machine.run(new_session=outer)
    assert result["exit_code"] == 0 and result["budget_stop"] is None
    assert result["direct_child_reaped"] and not result["requires_process_attention"]
    assert result["owned_group_empty_confirmed"] is (True if outer else None)
    assert result["sampled_tree_peak_rss_bytes"] == 2000
    assert result["sampled_budget_peak_rss_bytes"] == (3000 if outer else 2000)
    assert result["forwarded_log_bytes"] == result["stored_log_bytes"] == 3
    assert machine.child.stdin.writes == [b'{"answer":42}']
    assert not machine.signals and len(machine.waits) == 1
    assert machine.events.index("wait") < machine.events.index("selector_close")
    assert result["cleanup_errors"] == result["control_close_errors"] == []
    if outer:
        assert machine.events.index("wait") < machine.events.index("list_after_reap")
    assert machine.starts[0][1] == dict(cwd=machine.cache,
        env=helper.clean_environment(machine.cache), stdin=-1, stdout=-1,
        stderr=-2, start_new_session=outer)


def test_public_fast_zombie_does_not_fabricate_zero_rss_or_zero_exit(helper):
    machine = PublicMachine(helper)
    machine.blocks, machine.eof_code = [b""], 7
    result = machine.run()
    assert result["exit_code"] == 7 and result["budget_stop"] is None
    assert result["sampled_tree_peak_rss_bytes"] is None
    assert result["child_tree_rss_sample_count"] == 0 and result["direct_child_reaped"]
    assert "UNOBSERVED" in result["child_rss_observation_status"]


@pytest.mark.parametrize("outer", (False, True))
def test_public_live_descendant_is_cleaned_before_zombie_anchor_reap(helper, outer):
    machine = PublicMachine(helper)
    machine.extra[201] = row(201, parent=200, parent_unique=20000,
        group=200 if outer else 50, rss=4000)
    machine.extra[999] = row(999, parent=100, parent_unique=10000, group=50, rss=999999)
    result = machine.run(new_session=outer)
    assert result["budget_stop"] == "OWNED_PROCESSES_REMAIN_AFTER_DIRECT_CHILD_EXIT"
    assert [value["pid"] for value in machine.signals] == [201]
    assert machine.events.index("signal_201") < machine.events.index("wait")
    assert result["direct_child_reaped"] and not result["requires_process_attention"]
    assert 999 in machine.extra and result["sampled_tree_peak_rss_bytes"] == 6000
    assert result["native_process_work"]["cleanup_samples"] > 0


def test_public_failed_initial_authentication_after_spawn_cannot_signal_or_reap(helper):
    machine = PublicMachine(helper)
    machine.bad_initial_auth = True
    result = machine.run()
    assert len(machine.starts) == 1 and machine.signals == machine.waits == []
    assert result["budget_stop"] == "SUPERVISOR_FAILURE:RuntimeError"
    assert result["supervisor_failure_reason"] == "DIRECT_CHILD_IDENTITY_NOT_AUTHENTICATED"
    assert result["requires_process_attention"] and not result["direct_child_reaped"]
    assert not result["owned_group_empty_confirmed"] and result["cleanup_errors"]
    assert len(result["cleanup_errors"]) <= helper.MAX_IDENTITY_EVENTS


def test_public_unknown_fresh_identity_keeps_attention_and_never_blind_signals(helper):
    machine = PublicMachine(helper)
    machine.after_read = lambda value: value.failures.update(inspect_200=PermissionError)
    result = machine.run()
    assert result["budget_stop"] == "SUPERVISOR_FAILURE:PermissionError"
    assert result["requires_process_attention"] and result["observed_identity_unknown_still_present"]
    assert result["observed_identity_unknown_pid_count_peak"] == 1
    assert not machine.signals and not machine.waits
    assert result["identity_error_events"][0]["before"]["uniqueid"] == 20000


def test_public_nonempty_post_reap_group_cannot_restore_signal_authority(helper):
    machine = PublicMachine(helper)
    machine.nonempty_after_reap = True
    result = machine.run()
    assert result["direct_child_reaped"] and result["exit_code"] == 0
    assert result["budget_stop"] == "SUPERVISOR_FAILURE:RuntimeError"
    assert result["supervisor_failure_reason"] == "OWNED_GROUP_NOT_EMPTY_AFTER_REAP"
    assert result["requires_process_attention"] and not result["owned_group_empty_confirmed"]
    assert machine.signals == [] and len(machine.waits) == 1
    assert result["native_process_work"]["cleanup_samples"] == 0


@pytest.mark.parametrize("fault", ("stdin_write", "stdin_close", "register", "unregister",
    "log_open", "log_write", "log_flush", "log_close", "read", "select"))
def test_public_post_spawn_io_errors_preserve_failure_and_cleanup(helper, fault):
    machine = PublicMachine(helper)
    machine.failures[fault] = OSError
    result = machine.run()
    assert result["budget_stop"] == "SUPERVISOR_FAILURE:OSError"
    assert result["direct_child_reaped"] and result["owned_group_empty_confirmed"]
    assert not result["requires_process_attention"] and machine.child.stdout.closed
    assert all(value["pid"] == 200 for value in machine.signals)
    assert len(machine.starts) == len(machine.waits) == 1


@pytest.mark.parametrize("fault,accepted,retained", (("partial_exception", 0, 1),
    ("short", 1, 1), ("invalid", 0, 3), ("flush", 3, 3)))
def test_public_log_failure_reports_unknown_storage_without_losing_observed_bytes(helper, fault, accepted, retained):
    machine = PublicMachine(helper)
    if fault == "flush":
        machine.failures["log_flush"] = OSError
    else:
        def write(blob):
            machine.log.blob.extend(blob[:retained])
            if fault == "partial_exception":
                raise OSError("INVENTED_PARTIAL_LOG_WRITE")
            return 1 if fault == "short" else None
        machine.log.write = write
    result = machine.run()
    assert result["forwarded_log_bytes"] == result["log_write_attempted_bytes"] == 3
    assert result["log_write_accepted_bytes"] == accepted and len(machine.log.blob) == retained
    assert result["stored_log_bytes"] is None and result["log_storage_bytes_uncertain"]
    assert result["log_storage_status"] == "WRITE_OR_FLUSH_FAILED_STORED_BYTE_COUNT_UNKNOWN"
    assert result["direct_child_reaped"] and result["budget_stop"] == "SUPERVISOR_FAILURE:OSError"


@pytest.mark.parametrize("fault", ("selector_close", "stdout_close"))
@pytest.mark.parametrize("earlier", (False, True))
def test_public_control_close_cannot_hide_earlier_error_or_fake_success(helper, fault, earlier):
    machine = PublicMachine(helper)
    machine.failures[fault] = OSError
    if earlier:
        machine.failures["read"] = OSError
    result = machine.run()
    assert result["direct_child_reaped"] and result["owned_group_empty_confirmed"]
    assert result["budget_stop"] == ("SUPERVISOR_FAILURE:OSError" if earlier
        else "SUPERVISOR_CONTROL_CLOSE_FAILURE")
    assert result["control_close_errors"] == [("SELECTOR:" if fault == "selector_close" else "STDOUT:")+"OSError"]
    assert machine.events.index("wait") < machine.events.index(fault)


@pytest.mark.parametrize("fault", ("wait", "signal", "ignored_signal"))
def test_public_cleanup_failure_preserves_first_limit_and_unconfirmed_attention(helper, fault):
    machine = PublicMachine(helper)
    machine.blocks = [b"overflow"]
    if fault == "ignored_signal":
        machine.ignore_signals = True
    else:
        machine.failures[fault] = OSError
    result = machine.run(log_limit=1)
    assert result["budget_stop"] == "LOG_BYTE_LIMIT"
    assert result["requires_process_attention"] and not result["owned_group_empty_confirmed"]
    assert not result["direct_child_reaped"] and result["cleanup_errors"]
    assert len(result["cleanup_errors"]) <= helper.MAX_IDENTITY_EVENTS
    assert result["wall_seconds"] <= 7.01


@pytest.mark.parametrize("size", (4, 5))
def test_public_exact_log_boundary_preserves_observed_versus_stored_counts(helper, size):
    machine = PublicMachine(helper)
    machine.blocks = [b"x"*size, b""]
    result = machine.run(log_limit=4)
    assert result["forwarded_log_bytes"] == size
    assert result["stored_log_bytes"] == len(machine.log.blob) == min(size, 4)
    assert result["budget_stop"] == (None if size == 4 else "LOG_BYTE_LIMIT")
    assert result["direct_child_reaped"]


@pytest.mark.parametrize("case", ("nan", "over_cap", "sigchld", "factory", "popen"))
def test_public_prelaunch_refusals_create_no_cleanup_signal_authority(helper, case):
    machine = PublicMachine(helper)
    payload = {"answer": 42}
    if case == "nan": payload = {"bad": float("nan")}
    elif case == "over_cap": payload = {"bad": "x"*16385}
    elif case == "sigchld": helper.signal.getsignal = lambda _: 1
    else: machine.failures[case] = OSError
    result = machine.run(request=payload)
    assert result["budget_stop"] is not None
    assert result["child_pid"] is None and not result["direct_child_reaped"]
    assert not result["requires_process_attention"] and machine.signals == machine.waits == []
    assert len(machine.starts) == int(case == "popen")
    if case in ("nan", "over_cap", "sigchld"):
        assert machine.factory_calls == 0


@pytest.mark.parametrize("outer,limit,parent_limit,reason", (
    (True, 2999, None, "GLOBAL_RSS_INCLUDING_SUPERVISOR_LIMIT"),
    (False, 1999, None, "TREE_RSS_LIMIT"),
    (False, 10000, 999, "COORDINATOR_RSS_LIMIT")))
def test_public_memory_limits_keep_existing_scopes_and_first_failure(helper, outer, limit, parent_limit, reason):
    machine = PublicMachine(helper)
    result = machine.run(new_session=outer, rss_limit=limit, parent_rss_limit=parent_limit)
    assert result["budget_stop"] == reason and result["direct_child_reaped"]
    assert [value["pid"] for value in machine.signals] == [200]


def test_public_expired_work_deadline_uses_only_separate_cleanup_time(helper):
    machine = PublicMachine(helper)
    machine.exit_at_eof = False
    machine.blocks = [b""]
    result = machine.run(wall_seconds=.005)
    assert result["budget_stop"] == "SUPERVISOR_FAILURE:TimeoutError"
    assert result["direct_child_reaped"] and result["owned_group_empty_confirmed"]
    assert result["wall_seconds"] > .005
    assert result["native_process_work"]["cleanup_samples"] > 0


def counter_api(helper):
    """Real counter/deadline methods, fake time and no library construction."""
    api = helper._MacOSProcessAPI.__new__(helper._MacOSProcessAPI)
    api.phase, api.deadline, api.in_sample = "work", 1., 0
    api.work = dict(work_samples=0, cleanup_samples=0, work_native_calls=0,
        cleanup_native_calls=0, native_calls_attempted=0, list_calls=0,
        identity_calls=0, rss_calls=0, signal_calls=0, coherence_retries=0,
        explicit_esrch_observations=0, token_stale_or_exit=0)

    api.work.update(task_rss_esrch_observations=0, exit_transition_entries=0,
        exit_transition_waits=0, exit_transition_wait_seconds=0.,
        exit_transition_resolved_rss=0, exit_transition_resolved_zombie=0,
        exit_transition_resolved_absence=0, exit_transition_unresolved=0,
        exit_transition_diagnostics_discarded=0)
    api.exit_transition_diagnostics, api.exit_transition_first_failure = [], None
    clock = FakeClock()
    helper.time = clock
    # These counter tests use no native errno accessor, even incidentally.
    saved_errno = [0]
    helper.ctypes = types.SimpleNamespace(set_errno=lambda value: saved_errno.__setitem__(0, value),
        get_errno=lambda: saved_errno[0])
    return api, clock


@pytest.mark.parametrize("phase", ("work", "cleanup"))
def test_native_sample_exact_limit_and_one_over_are_precharged_without_queries(helper, phase):
    api, _ = counter_api(helper)
    api.phase = phase
    maximum = helper.MAX_WORK_SAMPLES if phase == "work" else helper.MAX_CLEANUP_SAMPLES
    api.work[phase+"_samples"] = maximum-1
    api.begin_sample(10)
    assert api.work[phase+"_samples"] == maximum
    with pytest.raises(RuntimeError, match="SAMPLE_COUNT_LIMIT"):
        api.begin_sample(10)
    assert api.work[phase+"_samples"] == maximum+1
    assert api.work["native_calls_attempted"] == 0


@pytest.mark.parametrize("limit", ("work", "cleanup", "per_sample"))
def test_native_call_exact_limit_and_one_over_never_invoke_over_budget_function(helper, limit):
    api, _ = counter_api(helper)
    api.phase = "cleanup" if limit == "cleanup" else "work"
    if limit == "per_sample":
        api.in_sample = helper.MAX_NATIVE_CALLS_PER_SAMPLE-1
    else:
        maximum = helper.MAX_WORK_NATIVE_CALLS if limit == "work" else helper.MAX_CLEANUP_NATIVE_CALLS
        api.work[limit+"_native_calls"] = maximum-1
    calls = []
    function = lambda: calls.append("invented") or 192
    assert api._call("identity", function) == (192, 0)
    with pytest.raises(RuntimeError, match="NATIVE_PROCESS_CALL_LIMIT"):
        api._call("identity", function)
    assert calls == ["invented"]
    assert api.work["native_calls_attempted"] == api.work["identity_calls"] == 2


def test_native_cleanup_reserve_is_distinct_after_work_counters_are_exhausted(helper):
    api, _ = counter_api(helper)
    api.work["work_samples"] = helper.MAX_WORK_SAMPLES+1
    api.work["work_native_calls"] = helper.MAX_WORK_NATIVE_CALLS+1
    api.phase = "cleanup"
    api.begin_sample(10)
    assert api._call("identity", lambda: 192) == (192, 0)
    assert api.work["cleanup_samples"] == api.work["cleanup_native_calls"] == 1
    assert api.work["work_samples"] == helper.MAX_WORK_SAMPLES+1
    assert api.work["work_native_calls"] == helper.MAX_WORK_NATIVE_CALLS+1


@pytest.mark.parametrize("late", (False, True))
def test_native_call_deadline_distinguishes_no_call_from_uncredited_late_effect(helper, late):
    api, clock = counter_api(helper)
    calls = []
    if not late:
        clock.now = 1.
    def function():
        calls.append("invented_effect")
        clock.now = 1.
        return 0
    with pytest.raises(TimeoutError, match="NATIVE_PROCESS_SAMPLE_DEADLINE"):
        api._call("signal", function)
    assert calls == (["invented_effect"] if late else [])
    assert api.work["native_calls_attempted"] == int(late)
    assert api.work["signal_calls"] == int(late)


@pytest.mark.parametrize("end,expected", ((.25, .25), (10, 1.)))
def test_native_sample_deadline_is_minimum_of_phase_and_one_second(helper, end, expected):
    api, _ = counter_api(helper)
    api.begin_sample(end)
    assert api.deadline == expected


@pytest.mark.parametrize("end", (True, None, "1", float("nan"), float("inf")))
def test_native_invalid_deadline_refuses_before_sample_or_call_precharge(helper, end):
    api, _ = counter_api(helper)
    with pytest.raises(RuntimeError, match="INVALID_NATIVE_SAMPLE_DEADLINE"):
        api.begin_sample(end)
    assert api.work["work_samples"] == api.work["native_calls_attempted"] == 0


def list_api(helper, returned_size, error, values=()):
    api = native_without_library(helper)
    api.lib.proc_listpids = object()
    calls = []
    def call(kind, function, query, identifier, pointer, buffer_bytes):
        calls.append((kind, query, identifier, buffer_bytes))
        assert buffer_bytes == 4*(helper.MAX_PIDS_PER_LIST+1)
        for index, value in enumerate(values):
            pointer._obj[index] = value
        return returned_size, error
    api._call = call
    return api, calls


@pytest.mark.parametrize("kind", (2, 6))
def test_native_exact_4096_pid_census_is_complete_and_sorted_without_truncation(helper, kind):
    # 4096 integers / 16 KiB plus the one overflow slot; no process enumeration.
    assert helper.MAX_PIDS_PER_LIST == 4096
    values = tuple(range(4096, 0, -1))
    api, calls = list_api(helper, 4096*4, 0, values)
    assert api.list_pids(kind, 200) == tuple(range(1, 4097))
    assert calls == [("list", kind, 200, 4097*4)]


def test_native_one_over_pid_cap_refuses_before_interpreting_buffer_contents(helper):
    api, _ = list_api(helper, (helper.MAX_PIDS_PER_LIST+1)*4, 0)
    with pytest.raises(RuntimeError, match="CAP_OR_TRUNCATION"):
        api.list_pids(2, 200)


@pytest.mark.parametrize("size,error,values", ((-1, 0, ()), (2, 0, ()),
    (20, 0, ()), (0, errno.EPERM, ()), (8, 0, (201, 201)),
    (4, 0, (0,)), (4, 0, (-1,))))
def test_native_denied_malformed_duplicate_and_short_pid_lists_are_not_empty_success(helper, size, error, values):
    helper.MAX_PIDS_PER_LIST = 3  # Fixed four-slot invented buffer.
    api, calls = list_api(helper, size, error, values)
    with pytest.raises(RuntimeError):
        api.list_pids(2, 200)
    assert len(calls) == 1


def test_native_exact_zero_pid_list_requires_success_not_an_errno(helper):
    api, calls = list_api(helper, 0, 0)
    assert api.list_pids(2, 200) == () and len(calls) == 1


@pytest.mark.parametrize("kind,pid", ((True, 200), (1, 200), (2, True),
    (2, 0), (2, -1), (6, 2**31)))
def test_native_invalid_scoped_census_never_calls_adapter(helper, kind, pid):
    api, calls = list_api(helper, 0, 0)
    with pytest.raises(RuntimeError, match="INVALID_SCOPED_PID_ENUMERATION"):
        api.list_pids(kind, pid)
    assert calls == []


@pytest.mark.parametrize("field,value", (("pid", 201), ("pgid", 0),
    ("ppid", 2**31), ("status", 0), ("status", 6), ("start_usec", 1000000),
    ("uniqueid", 0), ("idversion", 0)))
def test_native_full_size_but_malformed_identity_is_never_admitted(helper, field, value):
    api = native_without_library(helper)
    def call(kind, function, pid, flavor, arg, pointer, size):
        record = pointer._obj
        record.bsd.pid, record.bsd.ppid, record.bsd.pgid = 200, 100, 200
        record.bsd.status, record.bsd.start_usec = 2, 42
        record.unique.uniqueid, record.unique.idversion = 20000, 1
        setattr(record.unique if field in ("uniqueid", "idversion") else record.bsd, field, value)
        return 192, 0
    api._call = call
    with pytest.raises(RuntimeError, match="MALFORMED_PROCESS_IDENTITY_FIELDS"):
        api._bsd(200)


@pytest.mark.parametrize("size,error", ((0, errno.EPERM), (0, 0), (95, 0), (97, 0)))
def test_native_rss_permission_or_short_read_cannot_become_zero_memory(helper, size, error):
    api = native_without_library(helper)
    api._bsd = lambda pid: row(pid)
    api._call = lambda *args: (size, error)
    with pytest.raises(RuntimeError, match="TASK_RSS_UNAVAILABLE_OR_SHORT"):
        api.inspect(200, rss=True)


def test_native_live_to_zombie_rss_transition_retries_and_publishes_unobserved_not_zero(helper):
    api = native_without_library(helper)
    live, zombie = row(200), row(200, status=5, rss=None)
    sequence = iter((live, zombie, zombie, zombie))
    api._bsd = lambda pid: dict(next(sequence))
    calls = []
    api._call = lambda *args: calls.append(args[0]) or (0, errno.ESRCH)
    answer = api.inspect(200, rss=True)
    assert answer["status"] == 5 and answer["rss"] is None
    assert calls == ["rss"] and api.work["coherence_retries"] == 1


def test_native_rss_missing_without_stable_exit_refuses_after_two_attempts(helper):
    api = native_without_library(helper)
    api._bsd = lambda pid: row(pid)
    calls = []
    api._call = lambda *args: calls.append(args[0]) or (0, errno.ESRCH)
    with pytest.raises(RuntimeError, match="DID_NOT_STABILIZE"):
        api.inspect(200, rss=True)
    assert calls == ["rss", "rss"] and api.work["coherence_retries"] == 2


def test_native_stale_exec_token_requires_new_snapshot_before_retry_and_never_implies_exit(helper):
    api = native_without_library(helper)
    current, tokens = row(200, version=7), []
    api.inspect = lambda pid: dict(current)
    def call(kind, function, pointer, signum):
        tokens.append(pointer._obj.val[7])
        current["idversion"] = 8
        return (errno.ESRCH if len(tokens) == 1 else 0), 0
    api._call = call
    first = api.signal_owned(row(200))
    assert first["status"] == "TOKEN_STALE_OR_EXIT_NOT_RETIREMENT_PROOF"
    assert first["current"]["idversion"] == 7 and api.work["token_stale_or_exit"] == 1
    second = api.signal_owned(row(200))
    assert second["status"] == "ATOMIC_SIGNAL_ACCEPTED" and tokens == [7, 8]
    assert current["status"] == 2  # Accepted signal does not attest kernel exit.


# New independently scripted native boundary. The real _bsd/_call/inspect
# methods run against in-memory ctypes records and a virtual clock; no library,
# process, OS errno accessor, signal or wall-clock wait is used.
def transition_api(helper, steps, *, deadline=1., phase="work"):
    api = native_without_library(helper)
    api.phase, api.deadline, api.in_sample = phase, deadline, 0
    api.work.update(work_samples=0, cleanup_samples=0, work_native_calls=0,
        cleanup_native_calls=0, native_calls_attempted=0, list_calls=0,
        identity_calls=0, rss_calls=0, signal_calls=0)
    clock, pending, calls, saved_errno = FakeClock(), list(steps), [], [0]
    helper.time = clock
    helper.ctypes = types.SimpleNamespace(byref=ctypes.byref, sizeof=ctypes.sizeof,
        set_errno=lambda value: saved_errno.__setitem__(0, value),
        get_errno=lambda: saved_errno[0])
    def native(pid, flavor, arg, pointer, size):
        assert pending, "Unexpected native call past scripted evidence"
        kind, value = pending.pop(0)
        calls.append((pid, flavor, arg))
        assert kind == ("identity" if flavor == 18 else "rss")
        if kind == "identity":
            assert arg == 1 and size == 192
            if value is None:
                saved_errno[0] = errno.ESRCH
                return 0
            record = pointer._obj
            for field in ("pid", "ppid", "pgid", "status", "flags", "uid", "gid",
                          "ruid", "rgid", "start_sec", "start_usec"):
                setattr(record.bsd, field, value[field])
            for field in ("uniqueid", "parent_uniqueid", "idversion"):
                setattr(record.unique, field, value[field])
            return 192
        assert flavor == 4 and arg == 0 and size == 96
        returned, error, resident = value
        pointer._obj.resident_size = resident
        saved_errno[0] = error
        return returned
    api.lib.proc_pidinfo = native
    return api, clock, pending, calls


def bracket(first, second=None, task=(0, errno.ESRCH, 999999)):
    result = [("identity", first)]
    if first is not None:
        if first["status"] != 5:
            result.append(("rss", task))
        result.append(("identity", dict(first) if second is None else second))
    return result


@pytest.mark.parametrize("resolution", ("rss_zero", "rss_positive", "rss_max", "zombie",
    "absence_first", "absence_second"))
def test_exit_transition_exceeds_old_retry_count_then_requires_fresh_evidence(helper, resolution):
    live = row(200, flags=4)
    steps = bracket(live)*3
    if resolution.startswith("rss"):
        resident = {"rss_zero": 0, "rss_positive": 123456, "rss_max": 2**64-1}[resolution]
        steps += bracket(row(200, flags=0), task=(96, 0, resident))
    elif resolution == "zombie":
        steps += bracket(row(200, status=5, flags=4, rss=None))
    elif resolution == "absence_first":
        steps += [("identity", None)]
    else:
        steps += [("identity", live), ("rss", (0, errno.ESRCH, 999999)), ("identity", None)]
    api, clock, pending, calls = transition_api(helper, steps)
    answer = api.inspect(200, rss=True)
    assert not pending and api.deadline == 1. and clock.now == .003
    assert api.work["coherence_retries"] == 0
    assert api.work["exit_transition_entries"] == 1 and api.work["exit_transition_waits"] == 3
    assert api.work["exit_transition_unresolved"] == 0
    if resolution.startswith("rss"):
        assert answer["rss"] == resident and answer["status"] != 5
        key = "rss"
    elif resolution == "zombie":
        assert answer["rss"] is None and answer["status"] == 5
        key = "zombie"
    else:
        assert answer is None
        key = "absence"
    assert api.work["exit_transition_resolved_"+key] == 1
    assert api.work["native_calls_attempted"] == len(calls)
    assert api.work["work_samples"] == 0 and api.work["signal_calls"] == 0
    entered, final = api.exit_transition_diagnostics
    assert entered["before"]["pid"] == 200 and entered["after"]["flags"] & 4
    assert (entered["task_size"], entered["task_errno"], entered["entry"]) == (0, errno.ESRCH, None)
    assert final["entry"] == entered and final["entry"]["entry"] is None


@pytest.mark.parametrize("field", ("uniqueid", "idversion", "ppid", "parent_uniqueid",
    "pgid", "uid", "gid", "ruid", "rgid", "start_sec", "start_usec"))
def test_real_identity_mismatch_allowance_is_not_reset_by_transition_wait(helper, field):
    original, changed = row(200, flags=4), row(200, flags=4)
    changed[field] += 1
    steps = bracket(original, changed, (96, 0, 12)) + bracket(original)*3
    steps += bracket(original, changed, (96, 0, 99))
    api, _, pending, _ = transition_api(helper, steps)
    with pytest.raises(RuntimeError, match="DID_NOT_STABILIZE"):
        api.inspect(200, rss=True)
    assert not pending and api.work["coherence_retries"] == 2
    assert api.work["exit_transition_waits"] == 3
    assert api.work["exit_transition_unresolved"] == 1
    assert api.exit_transition_first_failure["after"][field] == changed[field]
    assert api.exit_transition_first_failure["before"][field] == original[field]


@pytest.mark.parametrize("first_flags,second_flags,eligible", ((0, 4, True), (4, 0, False),
    (0x80000000, 0x80000004, True), (0x80000000, 0x80000000, False)))
def test_only_second_bracket_inexit_bit_enables_wait_and_flags_are_not_identity(helper,
        first_flags, second_flags, eligible):
    first, second = row(200, flags=first_flags), row(200, flags=second_flags, status=3)
    steps = bracket(first, second)
    steps += bracket(row(200, status=5, flags=4)) if eligible else bracket(first, second)
    api, _, pending, _ = transition_api(helper, steps)
    if eligible:
        assert api.inspect(200, rss=True)["status"] == 5
    else:
        with pytest.raises(RuntimeError, match="DID_NOT_STABILIZE"):
            api.inspect(200, rss=True)
    assert not pending and api.work["exit_transition_waits"] == int(eligible)


@pytest.mark.parametrize("size,error", ((0, errno.EPERM), (0, 0), (95, 0), (97, 0),
    (0, errno.EIO)))
def test_inexit_never_excuses_denied_or_nonexact_task_record(helper, size, error):
    api, _, pending, _ = transition_api(helper,
        [("identity", row(200, flags=4)), ("rss", (size, error, 123))])
    with pytest.raises(RuntimeError, match="TASK_RSS_UNAVAILABLE_OR_SHORT"):
        api.inspect(200, rss=True)
    assert not pending and api.work["exit_transition_entries"] == 0


@pytest.mark.parametrize("phase", ("work", "cleanup"))
def test_exit_transition_short_remaining_deadline_is_not_renewed(helper, phase):
    api, clock, pending, calls = transition_api(helper, bracket(row(200, flags=4)),
        deadline=.0005, phase=phase)
    with pytest.raises(helper._ExitTransitionDeadline, match="UNRESOLVED_DEADLINE") as caught:
        api.inspect(200, rss=True)
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert not pending and len(calls) == 3 and clock.now == .0005 and api.deadline == .0005
    assert api.work[phase+"_native_calls"] == 3 and api.work[phase+"_samples"] == 0
    assert api.work["exit_transition_waits"] == api.work["exit_transition_unresolved"] == 1
    failure = api.exit_transition_first_failure
    assert failure["phase"] == phase and failure["task_errno"] == errno.ESRCH
    assert failure["entry"]["before"]["uniqueid"] == 20000


def test_transition_oversleep_is_failure_with_actual_wait_duration_not_fresh_grace(helper):
    api, clock, _, calls = transition_api(helper, bracket(row(200, flags=4)), deadline=.002)
    def oversleep(seconds):
        assert seconds == .001
        clock.now += .003
    clock.sleep = oversleep
    with pytest.raises(helper._ExitTransitionDeadline):
        api.inspect(200, rss=True)
    assert len(calls) == 3 and api.work["exit_transition_wait_seconds"] == .003
    assert api.deadline == .002


@pytest.mark.parametrize("late", (False, True))
def test_deadline_before_or_after_next_native_read_preserves_entry_cause(helper, late):
    api, clock, pending, calls = transition_api(helper,
        bracket(row(200, flags=4))+[("identity", row(200, flags=4))], deadline=.002)
    original = api.lib.proc_pidinfo
    def native(*args):
        answer = original(*args)
        if len(calls) == 4:
            clock.now = .002
        return answer
    api.lib.proc_pidinfo = native
    original_sleep = clock.sleep
    def sleep(seconds):
        original_sleep(seconds)
        if not late:
            clock.now = .002
    clock.sleep = sleep
    with pytest.raises(helper._ExitTransitionDeadline):
        api.inspect(200, rss=True)
    assert len(calls) == (4 if late else 3)
    assert len(pending) == (0 if late else 1)
    assert api.exit_transition_first_failure["entry"]["task_errno"] == errno.ESRCH
    assert api.exit_transition_first_failure["entry"]["after"]["flags"] == 4


@pytest.mark.parametrize("cap", ("work", "cleanup", "per_sample"))
def test_transition_native_cap_is_precharged_without_executing_excess_call(helper, cap):
    phase = "cleanup" if cap == "cleanup" else "work"
    api, _, pending, calls = transition_api(helper,
        bracket(row(200, flags=4))+[("identity", row(200, status=5, flags=4))], phase=phase)
    if cap == "per_sample":
        api.in_sample = helper.MAX_NATIVE_CALLS_PER_SAMPLE-3
    else:
        maximum = helper.MAX_WORK_NATIVE_CALLS if cap == "work" else helper.MAX_CLEANUP_NATIVE_CALLS
        api.work[cap+"_native_calls"] = maximum-3
    with pytest.raises(RuntimeError, match="NATIVE_PROCESS_CALL_LIMIT"):
        api.inspect(200, rss=True)
    assert len(calls) == 3 and len(pending) == 1
    assert api.work["native_calls_attempted"] == 4
    assert api.work["exit_transition_unresolved"] == 1
    assert api.exit_transition_first_failure["cause"] == "NATIVE_PROCESS_CALL_LIMIT"


def test_two_processes_share_original_sample_deadline(helper):
    steps = bracket(row(200, flags=4)) + bracket(row(200, status=5, flags=4))
    steps += bracket(row(201, flags=4))
    api, clock, pending, _ = transition_api(helper, steps, deadline=.0015)
    assert api.inspect(200, rss=True)["status"] == 5
    with pytest.raises(helper._ExitTransitionDeadline):
        api.inspect(201, rss=True)
    assert not pending and clock.now == api.deadline == .0015
    assert api.work["exit_transition_entries"] == 2 and api.work["exit_transition_unresolved"] == 1
    assert api.work["exit_transition_resolved_zombie"] == 1 and api.work["work_samples"] == 0


def test_fresh_replacement_after_wait_cannot_inherit_prior_rss(helper):
    old = row(200, flags=4)
    fresh = row(200, unique=999999, version=9, group=999, flags=0)
    api, _, pending, _ = transition_api(helper, bracket(old)*3+bracket(fresh, task=(96, 0, 777)))
    answer = api.inspect(200, rss=True)
    assert not pending and answer["uniqueid"] == 999999 and answer["idversion"] == 9
    assert answer["pgid"] == 999 and answer["rss"] == 777
    assert api.work["signal_calls"] == 0


def test_rss_false_does_not_wait_or_read_task_even_with_inexit(helper):
    first, second = row(200, flags=0), row(200, flags=4)
    api, clock, pending, calls = transition_api(helper,
        [("identity", first), ("identity", second)])
    answer = api.inspect(200, rss=False)
    assert not pending and answer["rss"] is None and answer["flags"] == 4
    assert clock.now == 0 and len(calls) == 2 and api.work["rss_calls"] == 0
    assert api.exit_transition_diagnostics == []


def test_diagnostic_saturation_preserves_first_failure_and_fixed_depth(helper):
    steps = []
    for pid in range(200, 209):
        steps += bracket(row(pid, flags=4)) + bracket(row(pid, status=5, flags=4))
    steps += bracket(row(900, flags=4))
    api, clock, pending, _ = transition_api(helper, steps, deadline=.0095)
    for pid in range(200, 209):
        assert api.inspect(pid, rss=True)["status"] == 5
    with pytest.raises(helper._ExitTransitionDeadline):
        api.inspect(900, rss=True)
    assert not pending and len(api.exit_transition_diagnostics) == 16
    assert api.work["exit_transition_diagnostics_discarded"] == 4
    failure = api.exit_transition_first_failure
    assert failure["before"]["pid"] == failure["entry"]["before"]["pid"] == 900
    assert failure["entry"]["entry"] is None and failure["cause"] == "NATIVE_PROCESS_SAMPLE_DEADLINE"
    assert all(type(value) in (int, float) for value in api.work.values())
    assert api.work["exit_transition_entries"] == 10 and api.work["exit_transition_unresolved"] == 1


def test_public_receipt_retains_transition_diagnostics_without_changing_custody(helper):
    machine = PublicMachine(helper)
    event = dict(outcome="UNRESOLVED", before=dict(pid=777), cause="invented")
    machine.api.exit_transition_diagnostics = [event]
    machine.api.exit_transition_first_failure = event
    result = machine.run()
    assert result["exit_transition_diagnostics"] == [event]
    assert result["exit_transition_first_failure"] == event
    assert result["direct_child_reaped"] and result["owned_group_empty_confirmed"]
    assert result["supervisor_schema"] == "GOLD11_OWNED_PROCESS_SUPERVISION_V3"
