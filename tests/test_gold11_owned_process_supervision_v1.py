"""Independent supervision qualification; SOURCE is mocked, SMOKE is opt-in.

Read only this named stdlib source. Never import any runner/native module, open
physical/Gold/cached inputs, or create a real process group. One optional tiny
same-group child is contained by the independently released outer supervisor.
All normal tests replace Popen, ps, process signals, clocks and pipe/log I/O.
"""
import ast
import io
import json
import os
from pathlib import Path
import signal
import sys
import types

import pytest


SOURCE = Path(__file__).resolve().parents[1]/"scripts/gold11_owned_process_supervision_v1.py"
RESOURCE_LIMITS = dict(wall_seconds=60, worker_cpu_seconds=45,
    process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)
SMOKE_OPT_IN = "SHERMAN_RUN_OWNED_PROCESS_SUPERVISION_V1_SMOKE"
CHILD_LIMITS = dict(wall_seconds=5, cpu_seconds=2, alarm_seconds=3,
    output_bytes=65536, process_tree_rss_bytes=1024**3)


@pytest.fixture
def helper():
    # Full named source, with inspected stdlib imports and definitions only.
    # No repository paths, runner constants or import-time process/file actions.
    module = types.ModuleType("_invented_owned_process_supervision")
    exec(compile(SOURCE.read_bytes(), str(SOURCE), "exec"), module.__dict__)
    return module


class _Clock:
    def __init__(self): self.now = 0.
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds


class _Pipe:
    def __init__(self, machine, kind):
        self.machine, self.kind, self.closed, self.writes = machine, kind, False, []
    def fileno(self): return 17
    def write(self, blob):
        self.writes.append(blob)
        self.machine.fail("stdin_write")
        return len(blob)
    def close(self):
        self.closed = True
        self.machine.events.append(self.kind+"_close")
        self.machine.fail(self.kind+"_close")


class _Child:
    pid = 200
    def __init__(self, machine):
        self.machine, self.returncode = machine, None
        self.stdin, self.stdout = _Pipe(machine, "stdin"), _Pipe(machine, "stdout")
    def poll(self):
        self.machine.fail("poll")
        return self.returncode
    def wait(self, timeout):
        self.machine.waits.append(timeout)
        assert 0 < timeout <= 5
        self.machine.events.append("wait")
        self.machine.fail("wait")
        if self.returncode is None: self.returncode = -signal.SIGKILL
        return self.returncode


class _Selector:
    def __init__(self, machine): self.machine, self.mapping = machine, {}
    def register(self, pipe, event):
        assert event == 1
        self.machine.fail("register")
        self.mapping[17] = types.SimpleNamespace(fileobj=pipe)
    def unregister(self, pipe): self.mapping.pop(17)
    def get_map(self): return self.mapping
    def select(self, timeout):
        assert 0 < timeout <= .05
        self.machine.select_timeouts.append(timeout)
        self.machine.clock.now += self.machine.select_advance
        self.machine.fail("select")
        return [(key, 1) for key in self.mapping.values()]
    def close(self):
        self.machine.events.append("selector_close")
        self.machine.fail("selector_close")


class _Log:
    def __init__(self, machine): self.machine, self.blob = machine, bytearray()
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def write(self, blob):
        self.machine.fail("log_write")
        self.blob.extend(blob)
        return len(blob)
    def flush(self): self.machine.fail("log_flush")


class _Machine:
    """Deterministic process/OS stand-in, never an actual subprocess fixture."""
    parent, parent_group = 100, 50
    def __init__(self, helper):
        self.helper, self.clock = helper, _Clock()
        self.child, self.selector, self.log = _Child(self), _Selector(self), _Log(self)
        self.blocks, self.exit_at_eof, self.eof_code = [b"ok\n", b""], True, 0
        self.failures, self.events, self.waits, self.signals = {}, [], [], []
        self.starts, self.samples, self.select_timeouts, self.read_limits = [], [], [], []
        self.select_advance, self.sample_advance = .01, .01
        self.outer, self.extra, self.killed = True, {}, set()
        self.sticky_group, self.table_override = False, None
        self.parent_rss, self.child_rss = 1000, 2000
        self.cache = Path("/invented/owned_cache")
        self.log_path = types.SimpleNamespace(open=self.open_log)
        helper.time = self.clock
        helper.selectors = types.SimpleNamespace(DefaultSelector=lambda: self.selector, EVENT_READ=1)
        helper.subprocess = types.SimpleNamespace(Popen=self.popen, PIPE=-1, STDOUT=-2)
        helper.os = types.SimpleNamespace(getpid=lambda: self.parent, getpgrp=lambda: self.parent_group,
            getpgid=self.getpgid, killpg=self.killpg, kill=self.kill, read=self.read)
        helper.process_table = self.table
    def fail(self, stage):
        error = self.failures.get(stage)
        if error is not None: raise error("INVENTED_"+stage)
    def popen(self, command, **kwargs):
        self.starts.append((command, kwargs))
        self.outer = kwargs["start_new_session"]
        self.fail("popen")
        return self.child
    def open_log(self, mode):
        assert mode == "xb"
        self.fail("log_open")
        return self.log
    def read(self, fd, count):
        assert fd == 17 and count == 65536
        self.read_limits.append(count)
        self.fail("read")
        block = self.blocks.pop(0) if self.blocks else b""
        if not block and self.exit_at_eof: self.child.returncode = self.eof_code
        return block
    def rows(self):
        table = {self.parent: dict(ppid=1, pgid=self.parent_group, rss=self.parent_rss)}
        if self.child.returncode is None:
            table[self.child.pid] = dict(ppid=self.parent,
                pgid=self.child.pid if self.outer else self.parent_group, rss=self.child_rss)
        table.update({pid: dict(row) for pid, row in self.extra.items() if pid not in self.killed})
        return table
    def table(self, timeout=1):
        assert 0 < timeout <= 1
        self.samples.append(timeout)
        self.clock.now += self.sample_advance
        self.fail("table")
        return self.table_override(self) if self.table_override else self.rows()
    def getpgid(self, pid):
        if pid == self.child.pid: return self.child.pid if self.outer else self.parent_group
        if pid in self.killed or pid not in self.extra: raise ProcessLookupError()
        return self.extra[pid]["pgid"]
    def killpg(self, pgid, sig):
        self.signals.append(("group", pgid, sig))
        self.events.append("killpg")
        assert pgid == self.child.pid, "Never signal an unrelated process group"
        self.fail("killpg")
        if self.child.returncode is None: self.child.returncode = -sig
        if not self.sticky_group:
            self.killed.update(pid for pid, row in self.extra.items() if row["pgid"] == pgid)
    def kill(self, pid, sig):
        self.signals.append(("pid", pid, sig))
        self.events.append("kill")
        self.fail("kill")
        assert pid != self.parent
        if pid == self.child.pid and self.child.returncode is None: self.child.returncode = -sig
        self.killed.add(pid)
    def run(self, **kwargs):
        options = dict(cache=self.cache, log_path=self.log_path, wall_seconds=2,
            rss_limit=10000, log_limit=100, new_session=True)
        options.update(kwargs)
        return self.helper.supervise_process(["invented-python", "invented-child"], {"answer": 42}, **options)


def test_source_closure_has_only_stdlib_imports_and_definitions_no_import_time_io():
    tree = ast.parse(SOURCE.read_text())
    imports = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    assert imports == {"json", "math", "os", "selectors", "signal", "subprocess", "time"}
    assert all(isinstance(node, (ast.Import, ast.FunctionDef)) or
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for node in tree.body)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert functions == {"process_table", "descendants", "clean_environment", "memory_budget", "supervise_process"}
    assert not any(isinstance(node, ast.ImportFrom) for node in ast.walk(tree))


def test_process_table_parses_all_four_columns_in_bytes_and_fixes_command(helper):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=0, stdout=" 100 1 50 7\n 200 100 200 11\n")
    helper.subprocess = types.SimpleNamespace(run=run)
    assert helper.process_table(.25) == {
        100: dict(ppid=1, pgid=50, rss=7*1024), 200: dict(ppid=100, pgid=200, rss=11*1024)}
    assert calls == [(["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss="],
        dict(capture_output=True, text=True, timeout=.25))]


@pytest.mark.parametrize("stdout,code", (("", 0), (" \n", 0), ("100 1 50 7", 1),
    ("100 1 7", 0), ("100 1 50 7 extra", 0), ("100 1 nope 7", 0),
    ("100 1 50 -7", 0), ("100 1 50 1.5", 0), ("100 1 50 7\n100 1 50 8", 0)))
def test_unavailable_malformed_duplicate_monitor_never_returns_empty_table(helper, stdout, code):
    helper.subprocess = types.SimpleNamespace(run=lambda *args, **kwargs:
        types.SimpleNamespace(returncode=code, stdout=stdout))
    with pytest.raises(RuntimeError): helper.process_table()


@pytest.mark.parametrize("timeout", (0, -1, 1.001, True, None, "1", float("nan"), float("inf")))
def test_invalid_sample_timeout_refuses_before_ps(helper, timeout):
    helper.subprocess = types.SimpleNamespace(run=lambda *args, **kwargs: pytest.fail("No ps allowed"))
    with pytest.raises(ValueError): helper.process_table(timeout)


def test_monitor_exception_propagates_instead_of_zero_rss(helper):
    def fail(*args, **kwargs): raise TimeoutError("INVENTED_MONITOR_TIMEOUT")
    helper.subprocess = types.SimpleNamespace(run=fail)
    with pytest.raises(TimeoutError): helper.process_table(.5)


@pytest.mark.parametrize("root,expected", ((200, {200, 201, 202}), (999, set()), (400, {400})))
def test_descendant_closure_is_transitive_not_a_pgid_or_unrelated_tree_query(helper, root, expected):
    table = {200: dict(ppid=100, pgid=200), 201: dict(ppid=200, pgid=999),
        202: dict(ppid=201, pgid=999), 300: dict(ppid=1, pgid=200), 400: dict(ppid=1, pgid=400)}
    before = json.dumps(table, sort_keys=True)
    assert helper.descendants(table, root) == expected
    assert json.dumps(table, sort_keys=True) == before


def test_clean_environment_is_an_allowlist_not_an_inherited_secret_or_thread_pool(helper, monkeypatch):
    monkeypatch.setenv("INVENTED_SECRET", "do-not-inherit")
    monkeypatch.setenv("PYTHONPATH", "/unrelated")
    answer = helper.clean_environment(Path("/invented/cache"))
    assert answer == dict(PATH="/usr/bin:/bin", HOME="/invented/cache", TMPDIR="/invented/cache",
        XDG_CACHE_HOME="/invented/cache", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


@pytest.mark.parametrize("tree,parent,limit,outer,parent_limit,expected", (
    (7, 3, 10, True, None, (10, None)), (8, 3, 10, True, None, (11, "GLOBAL_RSS_INCLUDING_SUPERVISOR_LIMIT")),
    (10, 99, 10, False, None, (10, None)), (11, 1, 10, False, None, (11, "TREE_RSS_LIMIT")),
    (7, 3, 100, False, 3, (7, None)), (7, 4, 100, False, 3, (7, "COORDINATOR_RSS_LIMIT"))))
def test_memory_budget_preserves_scope_equality_and_coordinator_limit(helper, tree, parent, limit, outer, parent_limit, expected):
    assert helper.memory_budget(tree, parent, limit, include_parent=outer, parent_rss_limit=parent_limit) == expected


@pytest.mark.parametrize("value", (-1, True, 1.0, None))
@pytest.mark.parametrize("index", (0, 1, 2))
def test_invalid_rss_samples_are_not_coerced(helper, value, index):
    values = [1, 2, 3]
    values[index] = value
    with pytest.raises(ValueError): helper.memory_budget(*values)


@pytest.mark.parametrize("key,value", (("wall_seconds", 0), ("wall_seconds", -1),
    ("wall_seconds", True), ("wall_seconds", float("nan")), ("wall_seconds", float("inf")),
    ("wall_seconds", "2"), ("rss_limit", 0), ("rss_limit", True), ("rss_limit", 2.),
    ("log_limit", 0), ("log_limit", -1), ("log_limit", True),
    ("parent_rss_limit", 0), ("parent_rss_limit", True), ("parent_rss_limit", float("nan")),
    ("new_session", 1)))
def test_invalid_supervision_budget_refuses_before_selector_or_launch(helper, key, value):
    machine = _Machine(helper)
    helper.selectors.DefaultSelector = lambda: pytest.fail("No selector before budget admission")
    with pytest.raises(ValueError, match="INVALID_SUPERVISION_BUDGET"): machine.run(**{key: value})
    assert machine.starts == [] and machine.signals == []


@pytest.mark.parametrize("outer", (False, True))
def test_normal_completion_reaps_and_counts_log_and_scope_without_signals(helper, outer):
    machine = _Machine(helper)
    receipt = machine.run(new_session=outer)
    assert receipt["exit_code"] == 0 and receipt["budget_stop"] is None
    assert receipt["direct_child_reaped"] is True and receipt["cleanup_errors"] == []
    assert receipt["control_close_errors"] == [] and machine.signals == []
    assert receipt["owned_group_empty_confirmed"] is (True if outer else None)
    assert receipt["owned_group_id"] == (200 if outer else None)
    assert receipt["sampled_tree_peak_rss_bytes"] == 2000
    assert receipt["sampled_parent_peak_rss_bytes"] == 1000
    assert receipt["sampled_budget_peak_rss_bytes"] == (3000 if outer else 2000)
    assert receipt["child_tree_rss_sample_count"] == receipt["child_rss_sample_count"] == 1
    assert receipt["forwarded_log_bytes"] == receipt["stored_log_bytes"] == len(machine.log.blob) == 3
    assert receipt["log_write_attempted_bytes"] == receipt["log_write_accepted_bytes"] == 3
    assert receipt["log_storage_bytes_uncertain"] is False
    assert receipt["log_storage_status"] == "WRITE_AND_FLUSH_RETURNED_NOT_DURABILITY_ATTESTATION"
    assert receipt["log_created"] is True and bytes(machine.log.blob) == b"ok\n"
    assert machine.child.stdin.writes == [b'{"answer":42}'] and machine.child.stdin.closed
    command, kwargs = machine.starts[0]
    assert command == ["invented-python", "invented-child"]
    assert kwargs == dict(cwd=machine.cache, env=helper.clean_environment(machine.cache),
        stdin=-1, stdout=-1, stderr=-2, start_new_session=outer)


def test_fast_exit_reports_unobserved_child_peak_as_none_not_zero(helper):
    machine = _Machine(helper)
    machine.blocks = [b""]
    receipt = machine.run()
    assert receipt["exit_code"] == 0 and receipt["budget_stop"] is None
    assert receipt["owned_group_empty_confirmed"] is True and receipt["direct_child_reaped"] is True
    assert receipt["sampled_tree_peak_rss_bytes"] is None
    assert receipt["child_tree_rss_sample_count"] == receipt["child_rss_sample_count"] == 0
    assert receipt["child_rss_observation_status"] == "UNOBSERVED_CHILD_TREE_PEAK_NOT_ZERO_REQUIRES_SEPARATE_HIGHWATER_EVIDENCE"


def test_leader_exit_and_eof_do_not_hide_reparented_owned_group_survivor(helper):
    machine = _Machine(helper)
    machine.extra[201] = dict(ppid=1, pgid=200, rss=4000)
    receipt = machine.run()
    assert receipt["budget_stop"] == "OWNED_PROCESSES_REMAIN_AFTER_DIRECT_CHILD_EXIT"
    assert receipt["sampled_tree_peak_rss_bytes"] == 6000
    assert machine.signals == [("group", 200, signal.SIGKILL)]
    assert receipt["direct_child_reaped"] is True and receipt["owned_group_empty_confirmed"] is True
    assert receipt["cleanup_errors"] == [] and 201 in machine.killed


def test_unrelated_foreign_groups_do_not_enter_accounting_or_signals(helper):
    machine = _Machine(helper)
    machine.extra[901] = dict(ppid=1, pgid=901, rss=10**12)
    receipt = machine.run()
    assert receipt["budget_stop"] is None and receipt["sampled_tree_peak_rss_bytes"] == 2000
    assert receipt["sampled_budget_peak_rss_bytes"] == 3000 and machine.signals == []


def test_outer_foreign_descendant_is_measured_vetoed_and_not_foreign_group_killed(helper):
    machine = _Machine(helper)
    machine.extra[201] = dict(ppid=200, pgid=777, rss=4000)
    receipt = machine.run()
    assert receipt["budget_stop"] == "OBSERVED_DESCENDANT_OUTSIDE_OWNED_GROUP"
    assert receipt["sampled_tree_peak_rss_bytes"] == 6000
    assert receipt["observed_foreign_group_escape_detected"] is True
    assert receipt["observed_foreign_pid_count_peak"] == 1
    assert receipt["observed_foreign_group_escape_unconfirmed_after_cleanup"] is True
    assert receipt["owned_group_empty_confirmed"] is True  # Only this group, NOT the foreign descendant.
    assert machine.signals == [("group", 200, signal.SIGKILL)] and 201 not in machine.killed


def test_inner_known_reparented_leaf_is_cleaned_by_pid_without_killing_shared_group(helper):
    machine = _Machine(helper)
    machine.extra[201] = dict(ppid=200, pgid=50, rss=1000)
    machine.extra[999] = dict(ppid=100, pgid=50, rss=10**12)
    def table(state):
        rows = state.rows()
        if state.child.returncode is not None and 201 in rows: rows[201]["ppid"] = 1
        return rows
    machine.table_override = table
    receipt = machine.run(new_session=False)
    assert receipt["budget_stop"] == "OWNED_PROCESSES_REMAIN_AFTER_DIRECT_CHILD_EXIT"
    assert receipt["sampled_tree_peak_rss_bytes"] == 3000
    assert receipt["owned_group_empty_confirmed"] is None
    assert machine.signals == [("pid", 201, signal.SIGKILL), ("pid", 200, signal.SIGKILL)]
    assert 999 not in machine.killed and receipt["cleanup_errors"] == []


@pytest.mark.parametrize("initial_group", (200, 777))
def test_known_pid_reparent_and_group_change_cannot_disappear_as_successful_cleanup(helper, initial_group):
    machine = _Machine(helper)
    machine.extra[201] = dict(ppid=200, pgid=initial_group, rss=1000)
    def table(state):
        rows = state.rows()
        if state.child.returncode is not None:
            # It is no longer rooted, and its observed PGID identity changed.
            # Neither silently drop it nor signal the new, potentially foreign group.
            rows[201] = dict(ppid=1, pgid=300, rss=1000)
        return rows
    machine.table_override = table
    receipt = machine.run()
    assert receipt["budget_stop"] == ("OBSERVED_PROCESS_IDENTITY_CHANGED_OR_REUSED" if initial_group == 200
        else "OBSERVED_DESCENDANT_OUTSIDE_OWNED_GROUP")
    assert receipt["observed_process_identity_change_detected"] is True
    assert receipt["observed_identity_unknown_still_present"] is True
    assert receipt["observed_identity_unknown_pid_count_peak"] == 1
    if initial_group == 777:
        assert receipt["observed_foreign_group_escape_unconfirmed_after_cleanup"] is True
        assert receipt["observed_foreign_identity_changed_or_unknown"] is True
    assert all(kind != "group" or pid == 200 for kind, pid, _ in machine.signals)
    assert not any(pid in (201, 300, 777) for _, pid, _ in machine.signals)


@pytest.mark.parametrize("fault", ("absent_child", "absent_parent", "own_parent_in_tree", "leader_wrong_group", "monitor"))
def test_missing_or_invalid_monitor_evidence_vetoes_never_zero_success(helper, fault):
    machine = _Machine(helper)
    def table(state):
        rows = state.rows()
        if fault == "absent_child": rows.pop(200, None)
        elif fault == "absent_parent": rows.pop(100)
        elif fault == "own_parent_in_tree": rows[100]["ppid"] = 200
        elif 200 in rows: rows[200]["pgid"] = 777
        return rows
    if fault == "monitor": machine.failures["table"] = RuntimeError
    else: machine.table_override = table
    receipt = machine.run()
    assert receipt["budget_stop"].startswith("SUPERVISOR_FAILURE:")
    assert machine.signals == [("group", 200, signal.SIGKILL)]
    assert receipt["child_tree_rss_sample_count"] == 0
    assert receipt["sampled_tree_peak_rss_bytes"] is None
    if fault in ("absent_parent", "monitor"): assert receipt["owned_group_empty_confirmed"] is False


@pytest.mark.parametrize("outer,limit,parent_limit,reason", (
    (True, 2999, None, "GLOBAL_RSS_INCLUDING_SUPERVISOR_LIMIT"),
    (False, 1999, None, "TREE_RSS_LIMIT"),
    (False, 10000, 999, "COORDINATOR_RSS_LIMIT")))
def test_orchestration_enforces_exact_memory_scope_and_keeps_first_stop(helper, outer, limit, parent_limit, reason):
    machine = _Machine(helper)
    receipt = machine.run(new_session=outer, rss_limit=limit, parent_rss_limit=parent_limit)
    assert receipt["budget_stop"] == reason and receipt["direct_child_reaped"] is True
    assert machine.signals and receipt["wall_seconds"] >= 0


@pytest.mark.parametrize("size", (4, 5))
def test_observed_and_stored_log_bytes_have_different_overflow_meanings(helper, size):
    machine = _Machine(helper)
    machine.blocks = [b"x"*size, b""]
    receipt = machine.run(log_limit=4)
    assert receipt["forwarded_log_bytes"] == size
    assert receipt["stored_log_bytes"] == len(machine.log.blob) == min(4, size)
    assert receipt["budget_stop"] == ("LOG_BYTE_LIMIT" if size == 5 else None)
    assert bool(machine.signals) is (size == 5)


@pytest.mark.parametrize("fault,accepted,retained", (("write_exception", 0, 1),
    ("short_write", 1, 1), ("flush_exception", 3, 3), ("invalid_write_result", 0, 3)))
def test_failed_log_publication_preserves_observed_bytes_and_unknown_storage(helper, fault, accepted, retained):
    machine = _Machine(helper)
    if fault == "flush_exception":
        machine.failures["log_flush"] = OSError
    else:
        def write(blob):
            machine.log.blob.extend(blob[:retained])
            if fault == "write_exception": raise OSError("INVENTED_PARTIAL_WRITE")
            return 1 if fault == "short_write" else None
        machine.log.write = write
    receipt = machine.run()
    assert receipt["budget_stop"] == "SUPERVISOR_FAILURE:OSError"
    assert receipt["forwarded_log_bytes"] == receipt["log_write_attempted_bytes"] == 3
    assert receipt["log_write_accepted_bytes"] == accepted and len(machine.log.blob) == retained
    assert receipt["stored_log_bytes"] is None and receipt["log_storage_bytes_uncertain"] is True
    assert receipt["log_storage_status"] == "WRITE_OR_FLUSH_FAILED_STORED_BYTE_COUNT_UNKNOWN"
    assert receipt["direct_child_reaped"] is True and receipt["owned_group_empty_confirmed"] is True


@pytest.mark.parametrize("fault", ("stdin_write", "stdin_close", "register", "log_open", "log_write", "log_flush", "read", "select"))
def test_post_launch_io_failure_always_runs_owned_cleanup(helper, fault):
    machine = _Machine(helper)
    machine.failures[fault] = OSError
    receipt = machine.run()
    assert receipt["budget_stop"] == "SUPERVISOR_FAILURE:OSError"
    assert machine.signals == [("group", 200, signal.SIGKILL)] and receipt["direct_child_reaped"] is True
    assert receipt["owned_group_empty_confirmed"] is True and machine.child.stdout.closed


@pytest.mark.parametrize("fault", ("selector_close", "stdout_close"))
def test_control_close_exception_cannot_bypass_prior_child_cleanup(helper, fault):
    machine = _Machine(helper)
    machine.failures["read"] = OSError
    machine.failures[fault] = OSError
    receipt = machine.run()
    assert receipt["direct_child_reaped"] is True and receipt["owned_group_empty_confirmed"] is True
    assert machine.events.index("wait") < machine.events.index(fault)
    assert receipt["control_close_errors"] == [("SELECTOR:" if fault == "selector_close" else "STDOUT:")+"OSError"]
    assert receipt["budget_stop"] == "SUPERVISOR_FAILURE:OSError"  # First failure retained.


@pytest.mark.parametrize("fault", ("selector_close", "stdout_close"))
def test_late_control_close_error_vetoes_otherwise_normal_completion(helper, fault):
    machine = _Machine(helper)
    machine.failures[fault] = OSError
    receipt = machine.run()
    assert receipt["exit_code"] == 0 and receipt["direct_child_reaped"] is True
    assert receipt["owned_group_empty_confirmed"] is True
    assert receipt["budget_stop"] == "SUPERVISOR_CONTROL_CLOSE_FAILURE"
    assert receipt["control_close_errors"] and machine.signals == []


@pytest.mark.parametrize("fault", ("killpg", "wait", "persistent_survivor"))
def test_cleanup_failure_cannot_confirm_owned_group_or_erase_primary_failure(helper, fault):
    machine = _Machine(helper)
    machine.blocks = [b"overflow", b""]
    if fault == "persistent_survivor":
        machine.extra[201] = dict(ppid=1, pgid=200, rss=100)
        machine.sticky_group = True
        machine.sample_advance = 1.
    else: machine.failures[fault] = OSError
    receipt = machine.run(log_limit=1)
    assert receipt["budget_stop"] == "LOG_BYTE_LIMIT"
    assert receipt["cleanup_errors"] and receipt["owned_group_empty_confirmed"] is False
    assert receipt["wall_seconds"] <= 7
    assert machine.events[-2:] == ["selector_close", "stdout_close"]


def test_absolute_work_deadline_and_failed_cleanup_time_are_both_retained(helper):
    machine = _Machine(helper)
    machine.blocks, machine.exit_at_eof = [b""], False
    machine.select_advance = .04
    receipt = machine.run(wall_seconds=.03)
    assert receipt["budget_stop"] == "SUPERVISOR_FAILURE:TimeoutError"
    assert receipt["wall_seconds"] > .03 and receipt["direct_child_reaped"] is True
    assert machine.select_timeouts == [.03]
    assert all(0 < timeout <= 1 for timeout in machine.samples)


def test_late_final_sample_wall_exceedance_cannot_return_success(helper):
    machine = _Machine(helper)
    machine.blocks = [b""]
    machine.sample_advance = .05
    receipt = machine.run(wall_seconds=.03)
    assert receipt["exit_code"] == 0 and receipt["owned_group_empty_confirmed"] is True
    assert receipt["budget_stop"] == "WALL_LIMIT_AFTER_FINAL_PROCESS_SAMPLE"


@pytest.mark.parametrize("payload", ({"bad": float("nan")}, {"large": "x"*16385}))
def test_dispatch_preflight_failure_starts_no_process(helper, payload):
    machine = _Machine(helper)
    receipt = helper.supervise_process(["not-executed"], payload, cache=machine.cache,
        log_path=machine.log_path, wall_seconds=2, rss_limit=100, log_limit=100, new_session=True)
    assert receipt["budget_stop"] == "SUPERVISOR_FAILURE:ValueError"
    assert receipt["child_pid"] is None and receipt["owned_group_empty_confirmed"] is None
    assert machine.starts == machine.signals == [] and receipt["sampled_tree_peak_rss_bytes"] is None


def test_popen_failure_has_no_owned_child_or_fabricated_reap(helper):
    machine = _Machine(helper)
    machine.failures["popen"] = OSError
    receipt = machine.run()
    assert receipt["budget_stop"] == "SUPERVISOR_FAILURE:OSError"
    assert len(machine.starts) == 1 and machine.signals == machine.waits == []
    assert receipt["child_pid"] is receipt["owned_group_empty_confirmed"] is None
    assert receipt["direct_child_reaped"] is False and receipt["log_created"] is False


def test_nonzero_direct_exit_is_retained_not_rewritten_into_supervisor_success(helper):
    machine = _Machine(helper)
    machine.eof_code = 7
    receipt = machine.run()
    assert receipt["exit_code"] == 7 and receipt["direct_child_reaped"] is True
    assert receipt["owned_group_empty_confirmed"] is True
    # No resource/monitor failure was invented; caller must also require exit0.
    assert receipt["budget_stop"] is None


_SMOKE_PROGRAM = """import json, os, resource, signal, sys
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
signal.alarm(3)
payload = sys.stdin.buffer.read(1025)
assert len(payload) <= 1024
assert json.loads(payload) == {'invented': 'tiny'}
print(json.dumps({'status': 'INVENTED_SAME_GROUP_SUPERVISION_COMPLETE',
    'pid': os.getpid(), 'pgid': os.getpgrp(), 'parent_pid': os.getppid()}), flush=True)
"""


def test_optional_smoke_has_no_group_creation_or_native_io_and_limits_precede_read():
    tree = ast.parse(_SMOKE_PROGRAM)
    assert {alias.name for node in tree.body if isinstance(node, ast.Import)
        for alias in node.names} == {"json", "os", "resource", "signal", "sys"}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = {ast.unparse(node.func) for node in calls}
    assert not {"os.setsid", "os.setpgid", "subprocess.Popen", "open", "exec", "__import__"} & names
    read, = [node for node in calls if ast.unparse(node.func) == "sys.stdin.buffer.read"]
    limits = [node for node in calls if ast.unparse(node.func) in ("resource.setrlimit", "signal.alarm")]
    assert len(limits) == 3 and all(node.lineno < read.lineno for node in limits)
    assert RESOURCE_LIMITS == dict(wall_seconds=60, worker_cpu_seconds=45,
        process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)
    assert CHILD_LIMITS == dict(wall_seconds=5, cpu_seconds=2, alarm_seconds=3,
        output_bytes=65536, process_tree_rss_bytes=1024**3)


@pytest.mark.skipif(os.environ.get(SMOKE_OPT_IN) != "1", reason="Requires independently frozen same-group invented child release")
def test_opt_in_actual_tiny_same_group_child_uses_helper_without_new_pgid(helper, tmp_path):
    cache = tmp_path/"invented_supervisor_child"
    cache.mkdir()
    log = cache/"OUTPUT.txt"
    receipt = helper.supervise_process([sys.executable, "-I", "-S", "-B", "-u", "-c", _SMOKE_PROGRAM],
        {"invented": "tiny"}, cache=cache, log_path=log,
        wall_seconds=CHILD_LIMITS["wall_seconds"], rss_limit=CHILD_LIMITS["process_tree_rss_bytes"],
        log_limit=CHILD_LIMITS["output_bytes"], new_session=False)
    output, log_error = None, None
    try:
        with log.open("rb") as stream: output = stream.read(CHILD_LIMITS["output_bytes"]+1)
    except OSError as error:
        log_error = type(error).__name__
    # Complete receipt and bounded log are forwarded before assertions; no
    # transient pytest directory is the sole retained record of a failure.
    print(json.dumps({"supervisor_receipt": receipt, "log_read_error": log_error,
        "child_log": output.decode("utf-8", errors="replace") if output is not None else None},
        sort_keys=True), flush=True)
    assert log_error is None and output is not None
    assert len(output) <= CHILD_LIMITS["output_bytes"]
    assert receipt["exit_code"] == 0 and receipt["budget_stop"] is None
    assert receipt["direct_child_reaped"] is True and receipt["cleanup_errors"] == receipt["control_close_errors"] == []
    assert receipt["owned_group_id"] is receipt["owned_group_empty_confirmed"] is None
    assert receipt["forwarded_log_bytes"] == receipt["stored_log_bytes"] == len(output)
    record = json.loads(output)
    assert record == dict(status="INVENTED_SAME_GROUP_SUPERVISION_COMPLETE", pid=receipt["child_pid"],
        pgid=os.getpgrp(), parent_pid=os.getpid())
