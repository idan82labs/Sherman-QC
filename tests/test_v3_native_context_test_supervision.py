"""Independent invented process models; no subprocess/native work is dispatched.

Only the named inert supervisor source and harness AST are read. All retained
output/cache fixtures are synthetic files beneath pytest's private tmp_path.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/v3_native_context_test_supervision.py"
HARNESS = ROOT / "scripts/check_cad_guided_measurement.py"
PID = 4242
OTHER_PID = 4243


@pytest.fixture
def subject():
    spec = importlib.util.spec_from_file_location("invented_v3_test_supervisor", SUPERVISOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Reviewed inert standard-library imports only.
    return module


class ProcessModel:
    """A finite scripted model; never calls real ps, Popen, selectors or kill."""

    def __init__(self, module, tmp_path, *, steps=None, samples=None,
                 popen_time=0., wait_time=None, wait_error=None,
                 popen_error=None, kill_error=None):
        self.module = module
        self.now = 0.
        self.steps = list(steps if steps is not None else [{"data": b"", "code": 0}])
        self.samples = list(samples if samples is not None else [[(PID, 4096)], []])
        self.popen_time = popen_time
        self.wait_time, self.wait_error = wait_time, wait_error
        self.popen_error, self.kill_error = popen_error, kill_error
        self.poll_code = None
        self.waits, self.selects, self.group_calls, self.sleeps, self.kills = [], [], [], [], []
        self.popen_calls, self.reads = [], []
        self.registered = False
        self.selector_closed = self.stdout_closed = False
        self.block = None
        self.retained = tmp_path / "receipt"
        self.retained.mkdir()
        self.cache = tmp_path / "cache"
        self.cache.mkdir()
        self.stdout = SimpleNamespace(fileno=lambda: 17, close=self.close_stdout)
        self.child = SimpleNamespace(pid=PID, stdout=self.stdout, poll=lambda: self.poll_code,
                                     wait=self.wait)
        module.time = SimpleNamespace(monotonic=lambda: self.now, sleep=self.sleep)
        module.tempfile = SimpleNamespace(mkdtemp=self.mkdtemp)
        module.os = SimpleNamespace(read=self.read, killpg=self.killpg)
        module.subprocess = SimpleNamespace(Popen=self.popen, PIPE=-1, STDOUT=-2)
        module.selectors = SimpleNamespace(DefaultSelector=lambda: self, EVENT_READ=1)
        module._group_sample = self.sample

    def mkdtemp(self, *, prefix):
        assert prefix == "native-context-v3-tests-"
        return str(self.cache)

    def popen(self, command, **kwargs):
        self.popen_calls.append((command, kwargs))
        self.now = self.popen_time
        if self.popen_error:
            raise self.popen_error
        return self.child

    def register(self, fileobj, event):
        assert fileobj is self.stdout and event == 1
        self.registered = True

    def select(self, *, timeout):
        self.selects.append((self.now, timeout))
        assert self.steps, "Unplanned monitoring iteration"
        step = self.steps.pop(0)
        self.now = step.get("at", self.now)
        if "code" in step:
            self.poll_code = step["code"]
        if "error" in step:
            raise step["error"]
        if "data" not in step:
            return []
        self.block = step["data"]
        return [(SimpleNamespace(fileobj=self.stdout), 1)]

    def unregister(self, fileobj):
        assert fileobj is self.stdout
        self.registered = False

    def get_map(self):
        return {17: self.stdout} if self.registered else {}

    def close(self):
        self.selector_closed = True

    def close_stdout(self):
        self.stdout_closed = True

    def read(self, fd, size):
        assert fd == 17 and size == 65536
        self.reads.append((fd, size))
        if isinstance(self.block, Exception):
            raise self.block
        assert len(self.block) <= size
        return self.block

    def sample(self, pgid, deadline):
        timeout = self.module._remaining(deadline, 2)
        self.group_calls.append((self.now, pgid, deadline, timeout))
        assert self.samples, "Unplanned process-group sample"
        value = self.samples.pop(0)
        if isinstance(value, dict):
            self.now = value.get("at", self.now)
            value = value.get("error", value.get("rows", []))
        if isinstance(value, Exception):
            raise value
        return value

    def wait(self, *, timeout):
        self.waits.append((self.now, timeout))
        if self.wait_time is not None:
            self.now = self.wait_time
        if self.wait_error:
            raise self.wait_error
        assert self.poll_code is not None, "Model requires kill or observed exit before reap"
        return self.poll_code

    def killpg(self, pgid, sig):
        assert pgid == PID and sig == signal.SIGKILL
        self.kills.append((pgid, sig))
        if self.kill_error:
            raise self.kill_error
        if self.poll_code is None:
            self.poll_code = -9

    def sleep(self, duration):
        self.sleeps.append((self.now, duration))
        self.now += duration

    def run(self, env=None):
        return self.module.supervise_native_context_test_v3(
            ["invented-python", "invented-test-only"], cwd="invented-cwd",
            env=env or {"OMP_NUM_THREADS": "1"}, retained=self.retained)


def assert_confirmed_terminal(result):
    assert result["process_group_confirmed_empty"] is True
    assert result["requires_group_attention"] is False
    assert result["cleanup_error"] is None


def test_fixed_limits_are_not_the_legacy_three_hundred_second_route(subject):
    assert (subject.WALL_SECONDS, subject.WORK_SECONDS) == (120, 115)
    assert subject.RSS_BYTES == 1024**3
    assert subject.OUTPUT_BYTES == 1024**2


@pytest.mark.parametrize("now,deadline,cap,want", [(0, 120, 5, 5), (119, 120, 5, 1),
    (114.99, 115, .02, .01), (119.999, 120, 2, .001)])
def test_remaining_clamps_every_wait_to_its_absolute_deadline(subject, now, deadline, cap, want):
    subject.time = SimpleNamespace(monotonic=lambda: now)
    assert subject._remaining(deadline, cap) == pytest.approx(want)


@pytest.mark.parametrize("now", [120, 120.001, 1000])
def test_remaining_refuses_expired_deadline(subject, now):
    subject.time = SimpleNamespace(monotonic=lambda: now)
    with pytest.raises(TimeoutError, match="ABSOLUTE_DEADLINE"):
        subject._remaining(120, 5)


def test_actual_process_table_parser_filters_group_and_converts_kib(subject):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="10 10 8\n4242 4242 10\n4243 4242 2\n")
    subject.subprocess = SimpleNamespace(run=run)
    subject.time = SimpleNamespace(monotonic=lambda: 119.)
    assert subject._group_sample(PID, 120) == [(PID, 10240), (OTHER_PID, 2048)]
    assert calls == [(["/bin/ps", "-axo", "pid=,pgid=,rss="],
                      dict(capture_output=True, text=True, timeout=1.))]


@pytest.mark.parametrize("stdout,code,reason", [
    ("", 0, "INVALID"), ("\n  \n", 0, "INVALID"), ("4242 4242", 0, "INVALID"),
    ("4242 4242 1 extra", 0, "INVALID"), ("4242 4242 -1", 0, "INVALID"),
    ("4242 4242 1.5", 0, "INVALID"), ("x 4242 1", 0, "INVALID"),
    ("4242 4242 1", 1, "INVALID"),
    ("4242 4242 1\n4242 8 2", 0, "DUPLICATE_PID"),
])
def test_process_table_parser_refuses_malformed_global_snapshot(subject, stdout, code, reason):
    subject.subprocess = SimpleNamespace(run=lambda *a, **k: SimpleNamespace(returncode=code, stdout=stdout))
    subject.time = SimpleNamespace(monotonic=lambda: 0.)
    with pytest.raises(RuntimeError, match=reason):
        subject._group_sample(PID, 120)


def test_process_table_timeout_is_bounded_and_propagates(subject):
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired("invented-ps", kwargs["timeout"])
    subject.subprocess = SimpleNamespace(run=run)
    subject.time = SimpleNamespace(monotonic=lambda: 114.99)
    with pytest.raises(subprocess.TimeoutExpired):
        subject._group_sample(PID, 115)
    assert calls == [pytest.approx(.01)]


def test_expired_process_query_does_not_dispatch_ps(subject):
    subject.subprocess = SimpleNamespace(run=lambda *a, **k: pytest.fail("ps after deadline"))
    subject.time = SimpleNamespace(monotonic=lambda: 120.)
    with pytest.raises(TimeoutError):
        subject._group_sample(PID, 120)


def test_normal_exit_still_reaps_and_confirms_empty_group(subject, tmp_path, capsys):
    model = ProcessModel(subject, tmp_path, steps=[{"data": b"invented output\n"}, {"data": b"", "code": 0}],
                         samples=[[(PID, 4096)], [], []])
    result = model.run()
    assert result["exit_code"] == 0 and result["budget_stop"] is None
    assert result["primary_error"] is None
    assert_confirmed_terminal(result)
    assert model.kills == [] and len(model.waits) == 1
    assert [call[2] for call in model.group_calls] == [115, 115, 120]
    assert model.selector_closed and model.stdout_closed
    output = (model.retained / "OUTPUT.txt").read_bytes()
    assert output == b"invented output\n"
    assert result["output_sha256"] == hashlib.sha256(output).hexdigest()
    assert result["output_bytes_observed"] == len(output)
    assert capsys.readouterr().out == ""  # Native stdout is retained, not forwarded.


def test_private_cache_is_retained_and_env_is_copied(subject, tmp_path):
    model = ProcessModel(subject, tmp_path)
    env = {"HOME": "not-used", "TMPDIR": "not-used", "XDG_CACHE_HOME": "not-used", "TOKEN": "invented"}
    result = model.run(env)
    command, options = model.popen_calls[0]
    assert command == ["invented-python", "invented-test-only"]
    assert options["start_new_session"] is True and options["cwd"] == "invented-cwd"
    assert all(options["env"][key] == str(model.cache) for key in ("HOME", "TMPDIR", "XDG_CACHE_HOME"))
    assert env["HOME"] == "not-used"
    assert model.cache.is_dir()
    assert result["private_cache_retained_for_custody"] == str(model.cache)


def test_nonzero_worker_exit_is_preserved_despite_clean_custody(subject, tmp_path):
    result = ProcessModel(subject, tmp_path, steps=[{"data": b"", "code": 7}]).run()
    assert result["exit_code"] == 7 and result["budget_stop"] is None
    assert_confirmed_terminal(result)


def test_main_near_115_never_spends_the_cleanup_reserve_on_select_or_ps(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, popen_time=114.99)
    result = model.run()
    assert result["budget_stop"] is None
    assert model.selects[0][1] == pytest.approx(.01)
    assert model.group_calls[0][2:] == (115, pytest.approx(.01))
    assert model.group_calls[-1][2] == 120
    assert model.waits[0][1] == 5


def test_work_deadline_before_next_dispatch_kills_then_uses_cleanup_deadline(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, popen_time=115, steps=[], samples=[[]])
    result = model.run()
    assert result["budget_stop"] == "WORK_WALL_LIMIT"
    assert model.selects == [] and model.kills == [(PID, signal.SIGKILL)]
    assert model.waits == [(115, 5)]
    assert [call[2] for call in model.group_calls] == [120]
    assert_confirmed_terminal(result)


def test_cleanup_wait_and_ps_share_only_the_remaining_absolute_second(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, popen_time=119, steps=[], samples=[[]])
    result = model.run()
    assert result["budget_stop"] == "WORK_WALL_LIMIT"
    assert model.waits == [(119, 1)]
    assert model.group_calls == [(119, PID, 120, 1)]
    assert_confirmed_terminal(result)


def test_ps_timeout_at_work_boundary_retains_error_and_work_classification(subject, tmp_path):
    error = subprocess.TimeoutExpired("invented-ps", .01)
    model = ProcessModel(subject, tmp_path, popen_time=114.99,
                         steps=[{}], samples=[{"at": 115, "error": error}, []])
    result = model.run()
    assert result["budget_stop"] == "WORK_WALL_LIMIT"
    assert result["primary_error"].startswith("TimeoutExpired:")
    assert_confirmed_terminal(result)


def test_late_zero_exit_after_work_limit_is_not_a_success(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, steps=[{"at": 116, "data": b"", "code": 0}], samples=[[]])
    result = model.run()
    assert result["exit_code"] == 0
    assert result["budget_stop"] == "WORK_WALL_LIMIT"
    assert_confirmed_terminal(result)


def test_late_zero_exit_cleanup_after_120_stays_failure_even_if_empty(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, samples=[[(PID, 4096)], {"at": 120.01, "rows": []}])
    result = model.run()
    assert result["exit_code"] == 0 and result["budget_stop"] == "TOTAL_WALL_LIMIT"
    assert result["process_group_confirmed_empty"] is True
    assert result["phase_times"]["terminal_decision_seconds"] >= 120


def test_missing_live_child_is_monitor_failure_not_empty_success(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, steps=[{}], samples=[[], []])
    result = model.run()
    assert result["budget_stop"] == "NATIVE_TEST_SUPERVISION_FAILURE"
    assert "ACTIVE_CHILD_MISSING" in result["primary_error"]
    assert model.kills == [(PID, signal.SIGKILL)]
    assert_confirmed_terminal(result)


@pytest.mark.parametrize("excess", [0, 1])
def test_rss_exact_boundary_and_one_byte_over(subject, tmp_path, excess):
    model = ProcessModel(subject, tmp_path, samples=[[(PID, subject.RSS_BYTES + excess)], []])
    result = model.run()
    assert result["sampled_group_peak_rss_bytes"] == subject.RSS_BYTES + excess
    assert result["budget_stop"] == ("SAMPLED_RSS_LIMIT" if excess else None)
    assert_confirmed_terminal(result)


@pytest.mark.parametrize("excess", [0, 1])
def test_output_cap_retains_only_prefix_but_counts_every_byte(subject, tmp_path, excess, capsys):
    chunk = b"x" * 65536
    steps = [{"data": chunk} for _ in range(16)]
    steps.append({"data": b"x"} if excess else {"data": b"", "code": 0})
    model = ProcessModel(subject, tmp_path, steps=steps,
                         samples=[[(PID, 4096)]] * 17 + [[]])
    result = model.run()
    assert result["output_bytes_observed"] == 1024**2 + excess
    assert (model.retained / "OUTPUT.txt").stat().st_size == 1024**2
    assert result["budget_stop"] == ("OUTPUT_LIMIT" if excess else None)
    assert_confirmed_terminal(result)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("failure", [RuntimeError("bad invented ps"), subprocess.TimeoutExpired("ps", 2)])
def test_monitor_failures_retained_with_independent_cleanup(subject, tmp_path, failure):
    model = ProcessModel(subject, tmp_path, steps=[{}], samples=[failure, []])
    result = model.run()
    assert result["budget_stop"] == "NATIVE_TEST_SUPERVISION_FAILURE"
    assert result["primary_error"].startswith(type(failure).__name__ + ":")
    assert_confirmed_terminal(result)


def test_selector_failure_still_reaps_and_confirms_owned_group(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, steps=[{"error": OSError("invented selector")}], samples=[[]])
    result = model.run()
    assert "invented selector" in result["primary_error"]
    assert model.kills and model.selector_closed and model.stdout_closed
    assert_confirmed_terminal(result)


def test_observed_surviving_child_group_is_killed_and_rechecked(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, steps=[{"data": b"", "code": 0}],
                         samples=[[(OTHER_PID, 8192)], [(OTHER_PID, 8192)], []])
    result = model.run()
    assert result["budget_stop"] == "UNEXPECTED_NATIVE_TEST_CHILD_PROCESS"
    assert result["observed_group_pids"] == [OTHER_PID]
    assert model.kills == [(PID, signal.SIGKILL)]
    assert model.sleeps == [(0., .02)]
    assert [call[2] for call in model.group_calls] == [115, 120, 120]
    assert_confirmed_terminal(result)


def test_new_unknown_postexit_group_identity_is_not_signaled_or_claimed_empty(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, samples=[[(PID, 4096)], [(OTHER_PID, 8192)]])
    result = model.run()
    assert model.kills == []
    assert result["process_group_confirmed_empty"] is False
    assert result["requires_group_attention"] is True
    assert result["budget_stop"] == "NATIVE_TEST_GROUP_CUSTODY_UNCONFIRMED"
    assert str(OTHER_PID) in result["cleanup_error"]
    assert model.cache.is_dir()


def test_known_survivor_cannot_extend_cleanup_past_absolute_deadline(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, samples=[[(PID, 4096)],
                         {"at": 119.99, "rows": [(PID, 4096)]}])
    result = model.run()
    assert result["process_group_confirmed_empty"] is False
    assert result["requires_group_attention"] is True
    assert result["budget_stop"] == "TOTAL_WALL_LIMIT"
    assert "ABSOLUTE_DEADLINE" in result["cleanup_error"]
    assert model.sleeps[0][1] == pytest.approx(.01)
    assert len(model.kills) == 1


def test_cleanup_query_failure_keeps_uncertainty_and_original_monitor_error(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, steps=[{}],
                         samples=[RuntimeError("first monitor"), RuntimeError("cleanup monitor")])
    result = model.run()
    assert "first monitor" in result["primary_error"]
    assert "cleanup monitor" in result["cleanup_error"]
    assert result["process_group_confirmed_empty"] is False
    assert result["requires_group_attention"] is True


def test_kill_failure_never_claims_reaped_or_empty(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, popen_time=115, steps=[], samples=[],
                         kill_error=PermissionError("invented kill refusal"))
    result = model.run()
    assert model.waits == []
    assert result["process_group_confirmed_empty"] is False
    assert result["requires_group_attention"] is True
    assert result["cleanup_error"].startswith("PermissionError:")


def test_reap_timeout_is_bounded_and_does_not_claim_empty(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, popen_time=115, steps=[], samples=[],
                         wait_error=subprocess.TimeoutExpired("invented child", 5))
    result = model.run()
    assert model.waits == [(115, 5)]
    assert result["cleanup_error"].startswith("TimeoutExpired:")
    assert result["process_group_confirmed_empty"] is False
    assert result["requires_group_attention"] is True


def test_popen_failure_retains_empty_output_and_cache_without_invented_child(subject, tmp_path):
    model = ProcessModel(subject, tmp_path, steps=[], samples=[], popen_error=OSError("invented spawn"))
    result = model.run()
    assert result["exit_code"] is None and result["requires_group_attention"] is False
    assert result["primary_error"].startswith("OSError:")
    assert model.kills == [] and model.waits == []
    assert (model.retained / "OUTPUT.txt").read_bytes() == b""
    assert model.cache.is_dir()


def test_existing_output_is_not_overwritten_or_followed_by_dispatch(subject, tmp_path):
    model = ProcessModel(subject, tmp_path)
    (model.retained / "OUTPUT.txt").write_bytes(b"invented prior terminal output")
    with pytest.raises(FileExistsError):
        model.run()
    assert model.popen_calls == []
    assert (model.retained / "OUTPUT.txt").read_bytes() == b"invented prior terminal output"


def test_source_is_inert_has_bounded_waits_and_has_no_cleanup_or_forwarding_route():
    tree = ast.parse(SUPERVISOR.read_text())
    assert all(isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef))
               or isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
               for node in tree.body)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in
                   ("rmtree", "TemporaryDirectory", "communicate") for node in calls)
    assert not any(isinstance(node.func, ast.Name) and node.func.id == "print" for node in calls)
    for call in calls:
        if isinstance(call.func, ast.Attribute) and call.func.attr in ("wait", "run", "select"):
            assert any(keyword.arg == "timeout" for keyword in call.keywords)
    assert "sys.stdout" not in SUPERVISOR.read_text()


@pytest.fixture
def harness_tree():
    return ast.parse(HARNESS.read_text())


def _main(tree):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")


def _native_parent_route(tree):
    return next(node for node in _main(tree).body if isinstance(node, ast.If)
                and "args.suite == 'native_context_collection_v3'" == ast.unparse(node.test)
                and "supervise_native_context_test_v3" in ast.unparse(node))


def test_harness_named_routes_and_source_closure_are_explicit(harness_tree):
    tests = ast.literal_eval(next(node.value for node in harness_tree.body if isinstance(node, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "TESTS" for t in node.targets)))
    assert tests["native_context_collection_supervision_v3"] == "tests/test_v3_native_context_test_supervision.py"
    assert tests["native_context_collection_v3_source"] == tests["native_context_collection_v3"]
    bindings = [ast.unparse(node) for node in harness_tree.body if isinstance(node, ast.Assign)
                and "V1_SOURCE_BINDINGS['native_context_collection_v3']" in ast.unparse(node)]
    assert any("scripts/v3_native_context_test_supervision.py" in line and
               "tests/test_v3_native_context_test_supervision.py" in line for line in bindings)
    pure_binding = next(ast.unparse(node) for node in harness_tree.body if isinstance(node, ast.Assign)
                        and "V1_SOURCE_BINDINGS['native_context_collection_supervision_v3']" in ast.unparse(node))
    assert "scripts/v3_native_context_test_supervision.py" in pure_binding


def test_native_route_returns_before_legacy_unbounded_cleanup_and_requires_clean_receipt(harness_tree):
    main = _main(harness_tree)
    route = _native_parent_route(harness_tree)
    later = main.body[main.body.index(route) + 1:]
    assert any(isinstance(node, ast.Return) for node in route.body)
    assert any("cache_scope" in ast.unparse(node) for node in later)
    code = ast.unparse(route)
    assert "COMPLETED.json" in code and ".open('x')" in code
    assert "source_sha256_after" in code and "source_before_after_consistent" in code
    final = next(node for node in route.body if isinstance(node, ast.Return))
    decision = ast.unparse(final.value)
    assert all(name in decision for name in ("exit_code", "budget_stop", "source_before_after_consistent", "cleanup_error"))
    assert decision.endswith("else 1")
    dispatches = [node for node in ast.walk(route) if isinstance(node, ast.Call)
                  and "supervise_native_context_test_v3" in ast.unparse(node.func)]
    assert len(dispatches) == 1
    command = dispatches[0].args[0]
    assert isinstance(command, ast.List)
    assert [node.value for node in command.elts if isinstance(node, ast.Constant)] == [
        "-I", "-S", "-B", "-u", "--worker", "--suite"]


def test_harness_native_worker_limits_and_native_opt_in_are_route_scoped(harness_tree):
    main = _main(harness_tree)
    worker = next(node for node in main.body if isinstance(node, ast.If) and ast.unparse(node.test) == "args.worker")
    limit_call = next(node for node in ast.walk(worker) if isinstance(node, ast.Call)
                      and ast.unparse(node.func) == "resource.setrlimit")
    assert ast.unparse(limit_call.args[0]) == "resource.RLIMIT_CPU"
    assert ast.unparse(limit_call.args[1]) == "(CPU_SECONDS, CPU_SECONDS)"
    opt_in = [node for node in ast.walk(worker) if node is not worker and isinstance(node, ast.If)
              and "SHERMAN_RUN_NATIVE_CONTEXT_COLLECTION_V3" in ast.unparse(node)]
    assert len(opt_in) == 1
    assert ast.unparse(opt_in[0].test) == "args.suite == 'native_context_collection_v3'"
    parents = [node for node in main.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "env" for target in node.targets)]
    assert len(parents) == 1 and isinstance(parents[0].value, ast.Dict)
    assert "SHERMAN_RUN_NATIVE_CONTEXT_COLLECTION_V3" not in ast.unparse(parents[0])
    native_limits = next(node for node in main.body if isinstance(node, ast.If)
                         and ast.unparse(node.test) == "args.suite == 'native_context_collection_v3'"
                         and "120, 90" in ast.unparse(node))
    assert "1024 ** 3" in ast.unparse(native_limits)
    assert any(isinstance(node, ast.If) and "native_context_collection_supervision_v3" in ast.unparse(node.test)
               and "60, 45" in ast.unparse(node) for node in main.body)


def test_native_artifact_destination_is_fresh_parent_owned_and_fixed_eight_cases(harness_tree):
    main = _main(harness_tree)
    artifact_route = next(node for node in ast.walk(main) if isinstance(node, ast.If)
        and ast.unparse(node.test) == "args.suite == 'native_context_collection_v3'"
        and "SHERMAN_NATIVE_CONTEXT_COLLECTION_V3_ARTIFACT_DIRECTORY" in ast.unparse(node))
    code = ast.unparse(artifact_route)
    assert "retained / 'ARTIFACTS'" in code
    assert "'case_count': 8" in code and "'per_case_json_bytes': 768 * 1024" in code
    assert "'total_case_json_bytes': 6 * 1024 ** 2" in code
    assert "cleanup_reserve_seconds" in code
