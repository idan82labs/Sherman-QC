"""Source-only orchestration contracts; no freeze, Gold, geometry or native job.

Tests use the whole runner module with inert top-level constants. Only pytest's
owned temporary directory is written. Native loading/process execution is never
performed; policy hooks and process samples receive explicit invented values.
Root must review/run this named suite with project conftest/plugins disabled.
"""
import ast
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1]/"scripts/run_gold11_mesh_development_v1.py"


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("gold11_mesh_runner_source_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_top_level_imports_are_only_stdlib_no_app_or_native(runner):
    tree = ast.parse(SOURCE.read_text())
    names = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import)
             for alias in node.names}
    names |= {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert names <= {"argparse", "ast", "contextlib", "fcntl", "hashlib", "importlib", "json", "os", "pathlib",
                     "resource", "secrets", "selectors", "signal", "stat", "subprocess", "sys", "tempfile", "time", "types"}
    assert "OCP" not in names and "numpy" not in names and "backend" not in names


def test_fixed_limits_separate_worker_coordinator_and_scorer(runner):
    limits = runner.LIMITS
    assert (limits["case_wall_seconds"], limits["case_cpu_seconds"], limits["sampled_worker_tree_rss_bytes"]) == (180, 150, 8*1024**3)
    assert (limits["scorer_wall_seconds"], limits["scorer_cpu_seconds"], limits["scorer_rss_bytes"]) == (60, 50, 2*1024**3)
    assert limits["global_tree_rss_bytes"] == 12*1024**3 < 16*1024**3
    assert limits["coordinator_rss_bytes"] == 4*1024**3
    assert limits["total_wall_seconds"] == 4200


@pytest.mark.parametrize("role,entry", (("INFERENCE", "worker"), ("SCORER", "scorer_worker")))
def test_direct_leaf_cli_paths_are_rejected_before_any_read_or_import(runner, monkeypatch, role, entry):
    direct = dict(snapshot="/arbitrary/source", cache="/arbitrary/input", arm="v1",
                  scan_sha256="a"*64, cad_sha256="b"*64)
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(runner.encoded(direct))))
    limits = []
    monkeypatch.setattr(runner.resource, "setrlimit", lambda key, value: limits.append((key, value)))
    def forbidden(*args, **kwargs):
        pytest.fail("A direct leaf bypass reached source/control/input access or initialization")
    monkeypatch.setattr(runner, "bounded_read", forbidden)
    monkeypatch.setattr(runner, "initialize_worker", forbidden)
    with pytest.raises(ValueError, match="COORDINATOR_LEAF_DISPATCH_REQUIRED"):
        getattr(runner, entry)()
    cpu = runner.LIMITS["case_cpu_seconds" if role == "INFERENCE" else "scorer_cpu_seconds"]
    assert limits == [(runner.resource.RLIMIT_CPU, (cpu, cpu))]


@pytest.mark.parametrize("capability", ({}, None, [], {"dispatch_id": "../outside", "token": "a"*64},
                                    {"dispatch_id": "a"*64, "token": "b"*64, "cache": "/outside"}))
def test_malformed_leaf_capability_has_no_file_access(runner, monkeypatch, capability):
    monkeypatch.setattr(runner, "bounded_read", lambda *a, **k: pytest.fail("unexpected file read"))
    with pytest.raises(ValueError, match="COORDINATOR_LEAF_DISPATCH_REQUIRED"):
        runner.authorize_leaf(capability, "INFERENCE")


@pytest.fixture
def leaf_attempt(runner, tmp_path, monkeypatch):
    """Invented neutral controls only: never creates the real attempt/freeze."""
    tmp_path = tmp_path.resolve()
    monkeypatch.setattr(runner, "RUN", tmp_path)
    monkeypatch.setattr(runner.os, "getpid", lambda: 100)
    monkeypatch.setattr(runner.os, "getppid", lambda: 99)
    monkeypatch.setattr(runner.os, "getpgrp", lambda: 100)
    runtime = {"invented_runtime": True}
    monkeypatch.setattr(runner, "runtime_metadata", lambda: runtime)
    source_blobs = {name: ("invented neutral source: "+name).encode() for name in runner.SOURCES}
    bundle = dict(source_sha256={name: runner.digest(blob) for name, blob in source_blobs.items()},
                  runtime_metadata=runtime)
    audit = {"invented_audit": True}
    expected = dict(parent_pid=99, token_sha256="e"*64)
    start = dict(bundle_sha256=runner.digest(runner.encoded(bundle)),
                 audit_sha256=runner.digest(runner.encoded(audit)), status="STARTED_NO_RETRY", limits=runner.LIMITS)
    runner.exclusive_json(tmp_path/"STARTED.json", start)
    runner.exclusive_json(tmp_path/"COORDINATOR_DISPATCH.json", expected)
    runner.activate_coordinator(bundle, audit, expected, start)
    cache = tmp_path/"neutral-cache"
    cache.mkdir(mode=0o700)
    monkeypatch.chdir(cache)
    monkeypatch.setenv("TMPDIR", str(cache))
    monkeypatch.setattr(runner, "__file__", str(tmp_path/"source"/runner.RUNNER_SOURCE))

    def prepare(role="INFERENCE"):
        binding = (dict(arm="v1", scan_sha256="a"*64, cad_sha256="b"*64)
                   if role == "INFERENCE" else dict(request_sha256="c"*64))
        request = runner.issue_leaf_dispatch(role, cache, **binding)
        for name in runner.leaf_sources(role):
            path = tmp_path/"source"/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source_blobs[name])
        if role == "SCORER":
            runner.exclusive_json(tmp_path/"INFERENCE_TERMINAL.json",
                                  dict(case_count=11, arm_count=2, terminal_record_count=22))
        monkeypatch.setattr(runner.os, "getpid", lambda: 101)
        monkeypatch.setattr(runner.os, "getppid", lambda: 100)
        return dict(request=request, cache=cache, binding=binding, role=role,
                    record=tmp_path/"leaf_dispatch"/(request["dispatch_id"]+".json"),
                    claimed=tmp_path/"leaf_dispatch"/(request["dispatch_id"]+".CLAIMED.json"))
    return prepare


@pytest.mark.parametrize("role", ("INFERENCE", "SCORER"))
def test_valid_neutral_leaf_is_parent_bound_and_one_use(runner, leaf_attempt, role, monkeypatch):
    attempt = leaf_attempt(role)
    assert set(attempt["request"]) == {"dispatch_id", "token"}
    snapshot, cache, binding = runner.authorize_leaf(attempt["request"], role)
    assert (snapshot, cache, binding) == (runner.RUN/"source", attempt["cache"], attempt["binding"])
    assert attempt["claimed"].is_file()
    original = runner.bounded_read
    def no_source_after_claim(path, *args, **kwargs):
        assert not Path(path).is_relative_to(snapshot), "replayed leaf reached source access"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(runner, "bounded_read", no_source_after_claim)
    with pytest.raises(FileExistsError):
        runner.authorize_leaf(attempt["request"], role)


@pytest.mark.parametrize("fault", ("token", "parent", "group", "role", "context", "started", "terminal"))
def test_invalid_leaf_binding_fails_before_claim_source_or_input(runner, leaf_attempt, monkeypatch, fault):
    attempt = leaf_attempt()
    request = dict(attempt["request"])
    if fault == "token":
        request["token"] = "0"*64
    elif fault == "parent":
        monkeypatch.setattr(runner.os, "getppid", lambda: 200)
    elif fault == "group":
        monkeypatch.setattr(runner.os, "getpgrp", lambda: 200)
    elif fault in ("role", "context"):
        record = runner.decode(attempt["record"].read_bytes())
        record["role" if fault == "role" else "context_sha256"] = "SCORER" if fault == "role" else "0"*64
        attempt["record"].write_bytes(runner.encoded(record))
    elif fault == "started":
        path = runner.RUN/"STARTED.json"
        start = runner.decode(path.read_bytes())
        start["audit_sha256"] = "0"*64
        path.write_bytes(runner.encoded(start))
    else:
        runner.exclusive_json(runner.RUN/"COMPLETED.json", {"invented_terminal": True})
    original = runner.bounded_read
    def controls_only(path, *args, **kwargs):
        assert not Path(path).is_relative_to(runner.RUN/"source")
        assert not Path(path).is_relative_to(attempt["cache"])
        return original(path, *args, **kwargs)
    monkeypatch.setattr(runner, "bounded_read", controls_only)
    with pytest.raises(ValueError):
        runner.authorize_leaf(request, "INFERENCE")
    assert not attempt["claimed"].exists()


def test_leaf_source_drift_burns_capability_without_reading_inputs(runner, leaf_attempt, monkeypatch):
    attempt = leaf_attempt()
    (runner.RUN/"source"/runner.RUNNER_SOURCE).write_bytes(b"invented source drift")
    original = runner.bounded_read
    def no_inputs(path, *args, **kwargs):
        assert not Path(path).is_relative_to(attempt["cache"])
        return original(path, *args, **kwargs)
    monkeypatch.setattr(runner, "bounded_read", no_inputs)
    with pytest.raises(ValueError, match="DECLARED_HASH_MISMATCH"):
        runner.authorize_leaf(attempt["request"], "INFERENCE")
    assert attempt["claimed"].is_file()


def test_leaf_cache_inode_change_is_terminal_before_source_or_input_read(runner, leaf_attempt, monkeypatch):
    attempt = leaf_attempt()
    record = runner.decode(attempt["record"].read_bytes())
    record["cache_identity"]["inode"] += 1
    attempt["record"].write_bytes(runner.encoded(record))
    original = runner.bounded_read
    def controls_only(path, *args, **kwargs):
        assert not Path(path).is_relative_to(runner.RUN/"source")
        assert not Path(path).is_relative_to(attempt["cache"])
        return original(path, *args, **kwargs)
    monkeypatch.setattr(runner, "bounded_read", controls_only)
    with pytest.raises(ValueError, match="LEAF_CACHE_BINDING_CHANGED"):
        runner.authorize_leaf(attempt["request"], "INFERENCE")
    assert attempt["claimed"].is_file()


def test_leaf_issuance_requires_live_verified_coordinator(runner, tmp_path):
    with pytest.raises(ValueError, match="VERIFIED_COORDINATOR_REQUIRED"):
        runner.issue_leaf_dispatch("INFERENCE", tmp_path, arm="v1", scan_sha256="a"*64, cad_sha256="b"*64)


def test_per_arm_dispatch_cannot_be_reissued_even_before_worker_claim(runner, leaf_attempt, monkeypatch):
    attempt = leaf_attempt()
    monkeypatch.setattr(runner.os, "getpid", lambda: 100)
    monkeypatch.setattr(runner.os, "getppid", lambda: 99)
    with pytest.raises(FileExistsError):
        runner.issue_leaf_dispatch("INFERENCE", attempt["cache"], **attempt["binding"])
    assert not attempt["claimed"].exists()


def test_scorer_dispatch_cannot_open_reference_request_before_22_terminals(runner, leaf_attempt):
    attempt = leaf_attempt("SCORER")
    (runner.RUN/"INFERENCE_TERMINAL.json").write_bytes(runner.encoded(
        dict(case_count=11, arm_count=2, terminal_record_count=21)))
    with pytest.raises(ValueError, match="SCORER_REQUIRES_ALL_22_TERMINALS"):
        runner.authorize_leaf(attempt["request"], "SCORER")
    assert not attempt["claimed"].exists()


def test_global_rss_includes_parent_even_when_child_tree_is_empty(runner):
    cap = runner.LIMITS["global_tree_rss_bytes"]
    assert runner.memory_budget(0, cap+1, cap, include_parent=True) == (cap+1, "GLOBAL_RSS_INCLUDING_SUPERVISOR_LIMIT")
    assert runner.memory_budget(cap-20, 21, cap, include_parent=True) == (cap+1, "GLOBAL_RSS_INCLUDING_SUPERVISOR_LIMIT")
    assert runner.memory_budget(cap-20, 20, cap, include_parent=True) == (cap, None)
    assert runner.memory_budget(1, 5, 8, parent_rss_limit=4) == (1, "COORDINATOR_RSS_LIMIT")


def test_global_supervision_wires_parent_into_budget_and_receipt():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    supervisor = ast.get_source_segment(source, functions["supervise_process"])
    execute = ast.get_source_segment(source, functions["execute"])
    assert "include_parent=new_session" in supervisor and "sampled_budget_peak_rss_bytes" in supervisor
    assert "new_session=True" in execute and 'LIMITS["global_tree_rss_bytes"]' in execute


@pytest.mark.parametrize("value", (None, [], "not a result", {"status": None}))
def test_compact_result_is_safe_for_malformed_workflow(runner, value):
    assert runner.compact_result(value)["predictions"] == []
    assert runner.compact_result(value)["status"] == "UNAVAILABLE"


def test_compaction_keeps_entire_predictions_conflicts_and_strict_fields(runner):
    prediction = {"bend_id": "invented", "line_start": [0, 0, 0], "line_end": [1, 0, 0],
                  "span_endpoints": [[0, 0, 0], [1, 0, 0]]}
    payload = dict(status="PARTIAL", phase="COMPLETE", predictions=[prediction], case_id="wrong-case",
                   family_id="wrong-family", observed_bend_count=None, production_use_allowed=False,
                   conformance="INDETERMINATE", uncertainty_status="UNCALIBRATED", prediction_frame="raw",
                   units_status="provisional", reference_frame_compatibility="not verified", reasons=[],
                   measurements={"large_unused_rows": list(range(10))}, native_features={"large_inventory": []})
    result = runner.compact_result(payload)
    assert result == {key: payload[key] for key in runner.SCORING_FIELDS if key in payload}
    assert result["predictions"] == [prediction]  # alternate carrier is NOT erased
    assert result["case_id"] == "wrong-case" and result["family_id"] == "wrong-family"
    assert "measurements" not in result and "native_features" not in result


def test_projection_overflow_refuses_whole_case(runner, monkeypatch):
    monkeypatch.setitem(runner.LIMITS, "compact_case_bytes", 50)
    result = runner.compact_result({"status": "PARTIAL", "predictions": [{"x": "x"*100}]})
    assert result["status"] == "UNAVAILABLE" and result["predictions"] == []
    assert result["reasons"] == ["SCORING_PROJECTION_BYTE_BUDGET"]


@pytest.mark.parametrize("value,allowed", (
    (None, True), ("/runtime/numpy/libtest.so", True), ("/runtime/scipy/libtest.dylib", True),
    ("/usr/lib/libSystem.B.dylib", True), ("/usr/lib/libc.dylib", True),
    ("libc.dylib", False), ("../lib.so", False), ("/owned/cache/lib.so", False),
    ("/runtime/source.py", False), ("/runtime-other/lib.so", False), (42, False),
))
def test_native_library_allowlist_is_not_blanket(runner, value, allowed):
    assert runner.allowed_native_library(value, ("/runtime",)) is allowed


@pytest.fixture
def hook(runner, monkeypatch):
    monkeypatch.setattr(runner, "runtime_roots", lambda: ("/invented/runtime",))
    return runner.worker_audit_hook(Path("/invented/snapshot"), Path("/invented/cache"))


def test_guard_admits_only_backend_runtime_and_neutral_cache(hook):
    hook("open", ("/invented/snapshot/backend/entry.py", "r", 0))
    hook("open", ("/invented/runtime/numpy/__init__.py", "r", 0))
    hook("open", ("/invented/cache/RESULT.json", "x", 0))
    hook("os.listdir", ("/invented/snapshot/backend",))
    hook("os.scandir", ("/invented/cache",))
    for path in ("/invented/snapshot/scripts/review.py", "/invented/snapshot/protocol.md",
                 "/invented/case/STARTED.json", "/not-in-scope/catalog.json"):
        with pytest.raises(PermissionError):
            hook("open", (path, "r", 0))
        with pytest.raises(PermissionError):
            hook("os.scandir", (path,))


@pytest.mark.parametrize("event,args", (
    ("os.remove", ("/invented/snapshot/backend/entry.py", -1)),
    ("os.mkdir", ("/not-in-scope", 0o700, -1)),
    ("os.rename", ("/invented/cache/a", "/not-in-scope/b", -1, -1)),
    ("os.link", ("/not-in-scope/a", "/invented/cache/b", -1, -1)),
    ("os.symlink", ("/invented/cache/a", "/invented/cache/b", -1)),
    ("socket.connect", (None, None)), ("subprocess.Popen", ("x", [], None, {})),
))
def test_guard_denies_unscoped_mutation_and_external_actions(hook, event, args):
    with pytest.raises(PermissionError):
        hook(event, args)


def test_fd_cleanup_paths_are_resolved_not_blanket_allowed(runner, monkeypatch):
    monkeypatch.setattr(runner, "runtime_roots", lambda: ("/invented/runtime",))
    monkeypatch.setattr(runner.sys, "platform", "darwin")
    monkeypatch.setattr(runner.fcntl, "fcntl", lambda fd, op, buffer: b"/invented/cache/subdir\0")
    hook = runner.worker_audit_hook(Path("/invented/snapshot"), Path("/invented/cache"))
    hook("os.scandir", (7,))
    hook("os.remove", ("temporary.step", 7))
    hook("os.rename", ("a", "b", 7, 7))
    monkeypatch.setattr(runner.fcntl, "fcntl", lambda fd, op, buffer: b"/not-in-scope\0")
    with pytest.raises(PermissionError):
        hook("os.scandir", (7,))
    with pytest.raises(PermissionError):
        hook("os.remove", ("temporary.step", 7))


def test_inert_namespace_does_not_execute_package_initializer(runner, monkeypatch):
    name = "invented_frozen_namespace"
    monkeypatch.setitem(runner.sys.modules, name, SimpleNamespace(__path__=["/wrong-package"]))
    runner.install_namespace(name, Path("/invented/snapshot/backend"))
    assert runner.sys.modules[name].__path__ == ["/invented/snapshot/backend"]
    assert runner.sys.modules[name].__package__ == name


@pytest.mark.parametrize("returncode,stdout", ((1, "1 0 1 42"), (0, ""), (0, "broken"),
                                             (0, "1 0 1 42\n1 0 1 44")))
def test_rss_monitor_fails_closed(runner, monkeypatch, returncode, stdout):
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=stdout))
    with pytest.raises(RuntimeError):
        runner.process_table()


def test_descendant_tree_excludes_unrelated_processes_and_includes_grandchildren(runner, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="1 0 1 20\n2 1 2 30\n3 2 2 40\n4 0 4 99"))
    table = runner.process_table()
    assert runner.descendants(table, 1) == {1, 2, 3}
    assert runner.descendants(table, 2) == {2, 3}
    assert runner.descendants(table, 9) == set()
    assert sum(table[pid]["rss"] for pid in runner.descendants(table, 1)) == 90*1024


def test_terminalization_retains_exact_22_without_source_or_reference_access(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUN", tmp_path)
    bundle = {"arms": ["r0", "v1"], "cases": [{"case_token": f"invented-{index}"} for index in range(11)]}
    rows = runner.terminalize_case_records(bundle, "INVENTED_GLOBAL_STOP")
    assert len(rows) == 22 and all(row["durable"] for row in rows)
    first = tmp_path/"cases/invented-0/r0/SCORING_INPUT.json"
    before = first.read_bytes()
    again = runner.terminalize_case_records(bundle, "MUST_NOT_REPLACE")
    assert len(again) == 22 and first.read_bytes() == before
    assert runner.decode(before)["predictions"] == []


def test_source_orders_gold_after_22_terminals_and_verifies_after_scoring(runner):
    source = SOURCE.read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    coordinator = ast.get_source_segment(source, functions["internal_coordinator"])
    assert coordinator.index('RUN/"INFERENCE_TERMINAL.json"') < coordinator.index("score_after_terminal(bundle, outcomes)")
    scorer = ast.get_source_segment(source, functions["score_after_terminal"])
    assert scorer.index("selected_references") < scorer.index('runner_command("--score-worker")')
    assert scorer.rindex("verify_bundle()") > scorer.index('runner_command("--score-worker")')
    worker = ast.get_source_segment(source, functions["worker"])
    assert "destination" not in worker and "case_token" not in worker and "part_key" not in worker
    assert 'cache/"RESULT.json"' in worker
    assert "token_sha256" in coordinator and "os.getppid()" in coordinator


def test_exclusive_one_shot_record_cannot_be_replaced(runner, tmp_path):
    target = tmp_path/"STARTED.json"
    runner.exclusive_json(target, {"status": "STARTED"})
    with pytest.raises(FileExistsError):
        runner.exclusive_json(target, {"status": "RETRY"})
    assert runner.decode(target.read_bytes()) == {"status": "STARTED"}


@pytest.mark.skipif(os.environ.get("SHERMAN_RUN_GUARDED_RUNTIME_SMOKE") != "1",
                    reason="Root-reviewed tiny native subprocess opt-in required")
def test_real_guard_imports_and_fd_safe_owned_temp_cleanup(runner, tmp_path):
    """Root-only runtime smoke: no mesh/STEP, inference, Gold or source freeze."""
    cache, snapshot = tmp_path/"cache", tmp_path/"source"
    cache.mkdir()
    (snapshot/"backend").mkdir(parents=True)
    program = '''import importlib.util, json, pathlib, sys, tempfile
source, snapshot, cache = map(pathlib.Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("invented_guard_runner", source)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.initialize_worker(snapshot, cache, "backend", 10)
import numpy
import scipy
from OCP.gp import gp_Pnt
point = gp_Pnt(1., 2., 3.)
assert (point.X(), point.Y(), point.Z()) == (1., 2., 3.)
with tempfile.TemporaryDirectory(dir=cache) as directory:
    path = pathlib.Path(directory)/"invented.txt"
    with path.open("xb") as stream:
        stream.write(b"invented guard smoke")
    assert path.read_bytes() == b"invented guard smoke"
print(json.dumps({"status": "INVENTED_GUARD_SMOKE_PASS"}), flush=True)
'''
    command = [str(runner.ROOT/".venv/bin/python"), "-I", "-S", "-B", "-u", "-c", program,
               str(SOURCE), str(snapshot), str(cache)]
    receipt = runner.supervise_process(command, {}, cache=cache, log_path=tmp_path/"SMOKE_OUTPUT.txt",
        wall_seconds=20, rss_limit=512*1024**2, log_limit=1024**2)
    assert receipt["exit_code"] == 0 and receipt["budget_stop"] is None, receipt
    assert '"status": "INVENTED_GUARD_SMOKE_PASS"' in (tmp_path/"SMOKE_OUTPUT.txt").read_text()
