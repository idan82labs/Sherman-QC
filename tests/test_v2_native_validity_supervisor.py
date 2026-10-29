"""Mocked supervisor controls and explicitly opt-in invented stdlib smoke.

All writes and mock input reads are in pytest-owned temporary directories.
The one real support-source read is the explicitly frozen V1 Python source,
not a control/data file or application import. Run only under root's isolated
source-test harness, with repository conftest/plugin autoload disabled. The two
real-process smoke controls require SHERMAN_RUN_V2_NATIVE_VALIDITY_GUARD_SMOKE=1.
They use only invented text/stdlib subprocesses: no OCP, CAD, scan or Gold.
"""
import ast
import builtins
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"scripts/run_v2_native_validity_diagnostic.py"


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location("_single_cad_supervisor_tests", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_and_cli_are_inert_stdlib_only(driver, monkeypatch):
    tree = ast.parse(SOURCE.read_bytes())
    imports = {alias.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for alias in n.names}
    imports |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)}
    assert imports <= {"argparse", "ast", "contextlib", "hashlib", "json", "os", "pathlib",
                       "resource", "secrets", "selectors", "shutil", "signal", "stat", "subprocess",
                       "sys", "tempfile", "time", "types"}
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(("OCP", "backend", "diagnostic"))
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("_another_inert_driver", SOURCE)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    monkeypatch.setattr(driver.sys, "argv", [str(SOURCE), "--cad-path", "/not-allowed"])
    with pytest.raises(SystemExit):
        driver.main()


def test_exact_limits_and_no_general_purpose_dispatch(driver):
    assert driver.LIMITS == dict(total_wall_seconds=120, terminalization_reserve_seconds=10,
        worker_cpu_seconds=90, sampled_complete_tree_rss_bytes=1024**3,
        combined_worker_output_bytes=1024**2, result_bytes=768*1024,
        parent_io_seconds=5, input_bytes=260402, dispatch_bytes=4096)
    assert driver.WORKER_SOURCES == ("driver.py", "support.py", "diagnostic/v2_native_validity_diagnostic.py")
    assert driver.SUPERVISION_CONTRACT["production_use_allowed"] is False


def test_only_pinned_generic_v1_functions_are_loaded(driver):
    support = (ROOT/driver.SOURCES["support.py"]).read_bytes()
    module = driver.generic_support(support)
    assert module.memory_budget(100, 20, 119, include_parent=True) == (120, "GLOBAL_RSS_INCLUDING_SUPERVISOR_LIMIT")
    assert not {"RUN", "CATALOG", "CASES", "worker", "main", "execute", "freeze",
                "internal_coordinator", "score_after_terminal", "issue_leaf_dispatch",
                "supervise_process"}.intersection(vars(module))
    assert set(driver.SUPPORT_FUNCTIONS) <= set(vars(module))
    with pytest.raises(ValueError, match="GENERIC_SUPPORT_SOURCE_MISMATCH"):
        driver.generic_support(support+b"\n")


@pytest.mark.parametrize("blob", (b"{\"x\":1,\"x\":2}", b"{\"x\":NaN}", b"{\"x\":Infinity}"))
def test_controls_reject_duplicate_and_nonfinite_json(driver, blob):
    with pytest.raises(ValueError):
        driver.decode(blob)


def test_descriptor_snapshot_is_bounded_and_hash_bound(driver, tmp_path):
    path = tmp_path/"invented.step"
    path.write_bytes(b"invented fixture bytes")
    expected = driver.digest(path.read_bytes())
    snapshot = {}
    assert driver.read_file(path, 64, expected, observation=snapshot) == b"invented fixture bytes"
    assert snapshot["sha256"] == expected and snapshot["bytes"] == len(b"invented fixture bytes")
    assert snapshot["regular_descriptor"] and snapshot["descriptor_metadata_stable"]
    with pytest.raises(ValueError, match="BYTE_BUDGET"):
        driver.read_file(path, 3)
    with pytest.raises(ValueError, match="DECLARED_HASH_MISMATCH"):
        driver.read_file(path, 64, "0"*64)
    link = tmp_path/"link.step"
    link.symlink_to(path)
    with pytest.raises(OSError):
        driver.read_file(link, 64)
    with pytest.raises(ValueError, match="NONREGULAR"):
        driver.read_file(tmp_path, 64)


def test_descriptor_change_during_read_is_rejected(driver, tmp_path, monkeypatch):
    path = tmp_path/"invented.step"
    path.write_bytes(b"invented")
    actual, count = driver.os.fstat, [0]
    def changing(fd):
        row = actual(fd)
        count[0] += 1
        return SimpleNamespace(**{k: getattr(row, k)+(1 if k == "st_mtime_ns" and count[0] == 2 else 0)
            for k in ("st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")})
    monkeypatch.setattr(driver.os, "fstat", changing)
    with pytest.raises(ValueError, match="SOURCE_CHANGED_DURING_READ"):
        driver.read_file(path, 64)


def test_empty_failed_attempt_log_can_be_hashed_but_empty_source_is_rejected(driver, tmp_path):
    path = tmp_path/"empty-output.txt"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="NONREGULAR_OR_BYTE_BUDGET"):
        driver.read_file(path, 64)
    assert driver.read_file(path, 64, allow_empty=True) == b""


def test_approved_binding_is_derived_without_opening_any_input(driver, monkeypatch):
    blob = (ROOT/driver.SOURCES["review.py"]).read_bytes()
    monkeypatch.setattr(driver, "read_file", lambda *a, **k: pytest.fail("unexpected input/source access"))
    binding = driver.approved_input(blob)
    assert binding["cad_sha256"] == driver.CAD_SHA and binding["cad_bytes"] == 260402
    assert binding["provenance_case"] == "CG191DEC7F81" and binding["provenance_family"] == "49001000"
    with pytest.raises(ValueError, match="REVIEW_SOURCE_BINDING_MISMATCH"):
        driver.approved_input(blob+b"\n")


@pytest.fixture
def invented_bundle(driver, tmp_path, monkeypatch):
    root, run = tmp_path/"repo", tmp_path/"attempt"
    root.mkdir()
    run.mkdir()
    monkeypatch.setattr(driver, "ROOT", root)
    monkeypatch.setattr(driver, "RUN", run)
    sources = {alias: f"invented-{index}.txt" for index, alias in enumerate(driver.SOURCES)}
    monkeypatch.setattr(driver, "SOURCES", sources)
    monkeypatch.setattr(driver, "FIXED_HASHES", {})
    runtime = {"invented_runtime": True}
    helper = SimpleNamespace(runtime_metadata=lambda: runtime)
    monkeypatch.setattr(driver, "generic_support", lambda blob: helper)
    binding = dict(cad_path=str(tmp_path/"must-not-open.step"), cad_sha256=driver.CAD_SHA,
                   cad_bytes=260402, provenance_case="invented")
    monkeypatch.setattr(driver, "approved_input", lambda blob: dict(binding))
    records = {}
    for alias, name in sources.items():
        blob = ("invented source "+alias).encode()
        (root/name).write_bytes(blob)
        dest = run/"source"/alias
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        records[alias] = dict(path=name, sha256=driver.digest(blob))
    bundle = dict(schema=driver.SCHEMA, limits=driver.LIMITS,
        supervision_contract=driver.SUPERVISION_CONTRACT, approved_input=dict(binding),
        runtime_metadata=runtime, sources=records)
    driver.exclusive_json(run/"FROZEN.json", bundle)
    return bundle, binding, helper


def test_bundle_checks_every_live_and_snapshot_source_not_manifest_input_path(driver, invented_bundle):
    bundle, binding, _ = invented_bundle
    assert driver.verify_bundle()[1] == binding
    bundle["approved_input"]["cad_path"] = "/unapproved/input.step"
    (driver.RUN/"FROZEN.json").write_bytes(driver.encoded(bundle))
    with pytest.raises(ValueError, match="FROZEN_APPROVED_INPUT_CHANGED"):
        driver.verify_bundle()


@pytest.mark.parametrize("side", ("live", "snapshot"))
def test_bundle_refuses_source_drift(driver, invented_bundle, side):
    path = (driver.ROOT/driver.SOURCES["driver.py"] if side == "live" else driver.RUN/"source/driver.py")
    path.write_bytes(b"invented drift")
    with pytest.raises(ValueError, match="DECLARED_HASH_MISMATCH"):
        driver.verify_bundle()


def test_freeze_reads_only_declared_sources_and_never_input(driver, invented_bundle, monkeypatch, tmp_path):
    bundle, _, _ = invented_bundle
    monkeypatch.setattr(driver, "RUN", tmp_path/"new-invented-freeze")
    original, visited = driver.read_file, []
    def source_only(path, *args, **kwargs):
        assert Path(path) in {driver.ROOT/n for n in driver.SOURCES.values()}
        visited.append(Path(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(driver, "read_file", source_only)
    driver.freeze()
    assert len(visited) == len(driver.SOURCES)
    assert driver.decode((driver.RUN/"FROZEN.json").read_bytes()) == bundle
    with pytest.raises(FileExistsError):
        driver.freeze()


def test_audit_requires_exact_bundle_and_independent_reviewer(driver, invented_bundle):
    bundle, _, _ = invented_bundle
    audit = dict(decision=driver.AUDIT_DECISION, bundle_sha256=driver.digest(driver.encoded(bundle)),
                 reviewer="invented independent reviewer", blocking_findings=[])
    driver.exclusive_json(driver.RUN/"PRE_RUN_AUDIT.json", audit)
    assert driver.verify_audit(bundle) == audit
    audit["bundle_sha256"] = "0"*64
    (driver.RUN/"PRE_RUN_AUDIT.json").write_bytes(driver.encoded(audit))
    with pytest.raises(ValueError, match="EXACT_INDEPENDENT_AUDIT_REQUIRED"):
        driver.verify_audit(bundle)


@pytest.mark.parametrize("field", ("limits", "runtime_metadata", "source_path"))
def test_frozen_limit_runtime_and_source_path_changes_fail_closed(driver, invented_bundle, field):
    bundle, _, _ = invented_bundle
    if field == "limits":
        bundle["limits"] = {**bundle["limits"], "worker_cpu_seconds": 91}
    elif field == "runtime_metadata":
        bundle["runtime_metadata"] = {"invented_runtime": False}
    else:
        bundle["sources"]["driver.py"]["path"] = "/unapproved/source.py"
    (driver.RUN/"FROZEN.json").write_bytes(driver.encoded(bundle))
    with pytest.raises(ValueError):
        driver.verify_bundle()


@pytest.mark.parametrize("capability", ({}, None, [], {"token": "a"}, {"token": "a"*64, "path": "/outside"}))
def test_direct_worker_rejects_arbitrary_requests_before_access(driver, monkeypatch, capability):
    monkeypatch.setattr(driver, "read_json", lambda *a, **k: pytest.fail("control read before gate"))
    with pytest.raises(ValueError, match="ONE_USE_PARENT_DISPATCH_REQUIRED"):
        driver.authorize_worker(capability)


@pytest.fixture
def leaf(driver, tmp_path, monkeypatch):
    run, cache = tmp_path/"attempt", tmp_path/"cache"
    run.mkdir()
    cache.mkdir(mode=0o700)
    monkeypatch.setattr(driver, "RUN", run)
    monkeypatch.setattr(driver, "__file__", str(run/"source/driver.py"))
    monkeypatch.setattr(driver.os, "getpid", lambda: 101)
    monkeypatch.setattr(driver.os, "getppid", lambda: 100)
    monkeypatch.setattr(driver.os, "getpgrp", lambda: 100)
    monkeypatch.chdir(cache)
    monkeypatch.setenv("TMPDIR", str(cache))
    runtime = {"invented_runtime": True}
    identity = {"device": 1, "inode": 2, "uid": 3}
    helper = SimpleNamespace(runtime_metadata=lambda: runtime, cache_identity=lambda p: identity)
    monkeypatch.setattr(driver, "generic_support", lambda blob: helper)
    source_hashes = {}
    for alias in driver.SOURCES:
        blob = ("invented payload "+alias).encode()
        source_hashes[alias] = driver.digest(blob)
        path = run/"source"/alias
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    monkeypatch.setattr(driver, "SUPPORT_SHA", source_hashes["support.py"])
    monkeypatch.setattr(driver, "DIAGNOSTIC_SHA", source_hashes["diagnostic/v2_native_validity_diagnostic.py"])
    audit = dict(decision=driver.AUDIT_DECISION, bundle_sha256="b"*64,
                 blocking_findings=[], reviewer="invented independent reviewer")
    started = dict(status="STARTED_NO_RETRY", bundle_sha256="b"*64,
                   audit_sha256=driver.digest(driver.encoded(audit)), limits=driver.LIMITS)
    context = dict(schema=driver.SCHEMA, parent_pid=100, bundle_sha256="b"*64,
        audit_sha256=started["audit_sha256"], started_sha256=driver.digest(driver.encoded(started)),
        limits=driver.LIMITS, source_sha256=source_hashes, runtime_metadata=runtime,
        cad_sha256=driver.CAD_SHA, cad_bytes=driver.CAD_BYTES)
    request = dict(token="a"*64)
    dispatch = dict(schema=driver.SCHEMA, role="SINGLE_CAD_VALIDITY_WORKER", parent_pid=100,
        context_sha256=driver.digest(driver.encoded(context)), token_sha256=driver.digest(request["token"].encode()),
        cache=str(cache), cache_identity=identity)
    coordinator_dispatch = dict(parent_pid=99, role="ONE_USE_COORDINATOR")
    coordinator_claim = dict(coordinator_pid=100, parent_pid=99,
        dispatch_sha256=driver.digest(driver.encoded(coordinator_dispatch)), no_retry=True)
    for name, value in (("PRE_RUN_AUDIT.json", audit), ("STARTED.json", started),
                        ("VERIFIED_CONTEXT.json", context), ("DISPATCH.json", dispatch),
                        ("COORDINATOR_DISPATCH.json", coordinator_dispatch),
                        ("COORDINATOR_CLAIMED.json", coordinator_claim)):
        driver.exclusive_json(run/name, value)
    return SimpleNamespace(request=request, cache=cache, helper=helper,
                           context=context, dispatch=dispatch, runtime=runtime)


def test_leaf_is_parent_bound_one_use_and_never_reads_review_or_protocol(driver, leaf, monkeypatch):
    original, visited = driver.read_file, []
    def tracked(path, *args, **kwargs):
        if (driver.RUN/"source") in Path(path).parents:
            assert str(Path(path).relative_to(driver.RUN/"source")) in driver.WORKER_SOURCES
            visited.append(Path(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(driver, "read_file", tracked)
    assert driver.authorize_worker(leaf.request) == (leaf.cache, leaf.helper)
    assert len(visited) == 3
    with pytest.raises(FileExistsError):
        driver.authorize_worker(leaf.request)
    assert len(visited) == 3


@pytest.mark.parametrize("fault", ("token", "parent", "group", "context", "terminal", "audit", "coordinator"))
def test_invalid_capability_stops_before_claim_or_payload(driver, leaf, monkeypatch, fault):
    request = dict(leaf.request)
    if fault == "token":
        request["token"] = "c"*64
    elif fault == "parent":
        monkeypatch.setattr(driver.os, "getppid", lambda: 999)
    elif fault == "group":
        monkeypatch.setattr(driver.os, "getpgrp", lambda: 999)
    elif fault == "context":
        leaf.context["cad_bytes"] += 1
        (driver.RUN/"VERIFIED_CONTEXT.json").write_bytes(driver.encoded(leaf.context))
    elif fault == "terminal":
        driver.exclusive_json(driver.RUN/"COMPLETED.json", {"status": "TERMINAL"})
    elif fault == "audit":
        audit = driver.read_json(driver.RUN/"PRE_RUN_AUDIT.json")
        audit["decision"] = "NOT_GREEN"
        (driver.RUN/"PRE_RUN_AUDIT.json").write_bytes(driver.encoded(audit))
    else:
        claim = driver.read_json(driver.RUN/"COORDINATOR_CLAIMED.json")
        claim["coordinator_pid"] = 999
        (driver.RUN/"COORDINATOR_CLAIMED.json").write_bytes(driver.encoded(claim))
    original = driver.read_file
    def no_payload(path, *args, **kwargs):
        assert (driver.RUN/"source") not in Path(path).parents
        return original(path, *args, **kwargs)
    monkeypatch.setattr(driver, "read_file", no_payload)
    with pytest.raises(ValueError):
        driver.authorize_worker(request)
    assert not (driver.RUN/"CLAIMED.json").exists()


def test_payload_drift_burns_claim_without_input_or_native(driver, leaf):
    (driver.RUN/"source/driver.py").write_bytes(b"changed invented driver")
    with pytest.raises(ValueError, match="DECLARED_HASH_MISMATCH"):
        driver.authorize_worker(leaf.request)
    assert (driver.RUN/"CLAIMED.json").exists()


def test_cache_identity_change_burns_claim(driver, leaf):
    leaf.dispatch["cache_identity"] = {"device": 1, "inode": 999, "uid": 3}
    (driver.RUN/"DISPATCH.json").write_bytes(driver.encoded(leaf.dispatch))
    with pytest.raises(ValueError, match="WORKER_CACHE_CHANGED"):
        driver.authorize_worker(leaf.request)
    assert (driver.RUN/"CLAIMED.json").exists()


def test_worker_entry_sets_cpu_then_rejects_direct_bypass(driver, monkeypatch):
    calls = []
    monkeypatch.setattr(driver.resource, "setrlimit", lambda *args: calls.append(args))
    monkeypatch.setattr(driver.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"{}")))
    monkeypatch.setattr(driver, "read_json", lambda *a, **k: pytest.fail("bypass reached I/O"))
    with pytest.raises(ValueError, match="ONE_USE_PARENT_DISPATCH_REQUIRED"):
        driver.worker()
    assert calls == [(driver.resource.RLIMIT_CPU, (90, 90))]


def _invented_result(driver):
    return dict(schema="V2_NATIVE_VALIDITY_DIAGNOSTIC", status="DIAGNOSTIC_COMPLETE",
        expected_cad_sha256=driver.CAD_SHA, expected_cad_bytes=driver.CAD_BYTES,
        production_use_allowed=False, observed_bend_count=None, predictions_not_computed=True,
        native_analyzer_count=1, step_import_count=1, evidence_complete=True, native_valid=False)


@pytest.fixture
def execution(driver, tmp_path, monkeypatch):
    run = tmp_path/"attempt"
    run.mkdir()
    monkeypatch.setattr(driver, "RUN", run)
    monkeypatch.setattr(driver.tempfile, "tempdir", str(tmp_path))
    blob = b"invented CAD bytes; never native input"
    path = tmp_path/"invented-input.step"
    path.write_bytes(blob)
    monkeypatch.setattr(driver, "CAD_SHA", driver.digest(blob))
    monkeypatch.setattr(driver, "CAD_BYTES", len(blob))
    binding = dict(cad_path=str(path), cad_sha256=driver.CAD_SHA, cad_bytes=len(blob))
    bundle = dict(sources={alias: {"sha256": "c"*64} for alias in driver.SOURCES},
                  runtime_metadata={"invented_runtime": True}, approved_input=binding)
    audit = dict(decision=driver.AUDIT_DECISION, bundle_sha256=driver.digest(driver.encoded(bundle)),
                 blocking_findings=[], reviewer="invented independent reviewer")
    calls, reads = [], []
    helper = SimpleNamespace(cache_identity=lambda cache: {"invented_identity": True})
    def supervise(command, request, **kwargs):
        calls.append((command, request, kwargs))
        assert (run/"STARTED.json").exists()
        assert not (kwargs["cache"]/"cad.step").exists()
        coordinator_dispatch = driver.read_json(run/"COORDINATOR_DISPATCH.json")
        driver.exclusive_json(run/"COORDINATOR_CLAIMED.json", dict(coordinator_pid=100,
            parent_pid=driver.os.getpid(), dispatch_sha256=driver.digest(driver.encoded(coordinator_dispatch)), no_retry=True))
        result_blob = driver.encoded(_invented_result(driver))
        (run/"RESULT.json").write_bytes(result_blob)
        dispatch = dict(parent_pid=100, role="SINGLE_CAD_VALIDITY_WORKER")
        driver.exclusive_json(run/"DISPATCH.json", dispatch)
        claim = dict(worker_pid=101, parent_pid=100,
            dispatch_sha256=driver.digest(driver.encoded(dispatch)), no_retry=True)
        driver.exclusive_json(run/"CLAIMED.json", claim)
        (run/"OUTPUT.txt").write_bytes(driver.encoded(dict(status="SINGLE_CAD_WORKER_TERMINAL",
            result_sha256=driver.digest(result_blob), worker_pid=101)))
        driver.exclusive_json(run/"COORDINATOR_COMPLETED.json", dict(
            status="COMPLETE_DEVELOPMENT_DIAGNOSTIC", coordinator_pid=100, worker_pid=101,
            worker_exit_code=0, cache_cleanup_complete=True, result_sha256=driver.digest(result_blob),
            worker_claim_sha256=driver.digest(driver.encoded(claim))))
        # A mock-only cleanup claim must correspond to the mock's owned cache.
        kwargs["cache"].rmdir()
        return dict(exit_code=0, budget_stop=None, coordinator_pid=100, group_gone=True,
                    sampled_budget_peak_rss_bytes=1)
    monkeypatch.setattr(driver, "watch_coordinator", supervise)
    monkeypatch.setattr(driver, "verify_bundle", lambda: (bundle, binding, helper))
    monkeypatch.setattr(driver, "verify_audit", lambda b: audit)
    monkeypatch.setattr(driver, "memory_check", lambda h: None)
    original = driver.read_file
    def observed_read(given, *args, **kwargs):
        if Path(given) == path:
            reads.append(given)
            pytest.fail("outer watchdog must never read original input")
        return original(given, *args, **kwargs)
    monkeypatch.setattr(driver, "read_file", observed_read)
    return SimpleNamespace(helper=helper, calls=calls, reads=reads, path=path, bundle=bundle, binding=binding)


def test_outer_execute_never_reads_input_and_dispatches_one_coordinator(driver, execution):
    driver.execute()
    assert len(execution.reads) == 0 and len(execution.calls) == 1
    command, request, args = execution.calls[0]
    assert command[-1] == "--coordinator" and "-I" in command and "-S" in command
    assert str(execution.path) not in command and set(request) == {"token"}
    started = driver.read_json(driver.RUN/"STARTED.json")
    assert args["work_stop"] == started["began_monotonic"]+110
    assert args["end"] == args["work_stop"]+10
    receipt = driver.read_json(driver.RUN/"COMPLETED.json")
    assert receipt["status"] == "COMPLETE_DEVELOPMENT_DIAGNOSTIC"
    assert receipt["source_before_after_consistent"] and receipt["production_use_allowed"] is False
    assert receipt["scratch_cleanup"] == "COORDINATOR_CONFIRMED"
    assert receipt["result_sha256"] == driver.digest((driver.RUN/"RESULT.json").read_bytes())
    before = (driver.RUN/"COMPLETED.json").read_bytes()
    with pytest.raises(FileExistsError):
        driver.execute()
    assert not execution.reads and len(execution.calls) == 1
    assert (driver.RUN/"COMPLETED.json").read_bytes() == before


def test_preflight_audit_failure_has_no_start_input_or_worker(driver, execution, monkeypatch):
    def refuse(_):
        raise ValueError("EXACT_INDEPENDENT_AUDIT_REQUIRED")
    monkeypatch.setattr(driver, "verify_audit", refuse)
    with pytest.raises(ValueError, match="EXACT_INDEPENDENT_AUDIT_REQUIRED"):
        driver.execute()
    assert not execution.reads and not execution.calls
    assert not (driver.RUN/"STARTED.json").exists()
    assert not (driver.RUN/"COMPLETED.json").exists()


def test_budget_stop_is_terminal_without_result_acceptance(driver, execution, monkeypatch):
    def stopped(*args, **kwargs):
        return dict(exit_code=-9, budget_stop="COMPLETE_GROUP_PLUS_OUTER_RSS_LIMIT", group_gone=True)
    monkeypatch.setattr(driver, "watch_coordinator", stopped)
    driver.execute()
    receipt = driver.read_json(driver.RUN/"COMPLETED.json")
    assert receipt["status"] == "TERMINAL_FAILURE" and receipt["result_sha256"] is None
    assert receipt["coordinator_receipt"]["budget_stop"] == "COMPLETE_GROUP_PLUS_OUTER_RSS_LIMIT"
    assert receipt["scratch_cleanup"] == "NOT_CONFIRMED_NO_OUTER_RECURSIVE_CLEANUP"


def test_post_worker_source_drift_invalidates_diagnostic(driver, execution, monkeypatch):
    calls = []
    def verify():
        calls.append(True)
        if len(calls) > 1:
            raise ValueError("DECLARED_HASH_MISMATCH")
        return execution.bundle, execution.binding, execution.helper
    monkeypatch.setattr(driver, "verify_bundle", verify)
    driver.execute()
    receipt = driver.read_json(driver.RUN/"COMPLETED.json")
    assert receipt["status"] == "TERMINAL_FAILURE"
    assert receipt["source_before_after_consistent"] is False


def test_receipt_write_failure_leaves_durable_started_and_never_readmits(driver, execution, monkeypatch):
    original = driver.exclusive_json
    def fail_receipt(path, value):
        if Path(path) == driver.RUN/"COMPLETED.json":
            raise OSError("invented terminal receipt write failure")
        return original(path, value)
    monkeypatch.setattr(driver, "exclusive_json", fail_receipt)
    with pytest.raises(OSError, match="invented terminal receipt"):
        driver.execute()
    started = (driver.RUN/"STARTED.json").read_bytes()
    assert driver.decode(started)["status"] == "STARTED_NO_RETRY"
    assert not (driver.RUN/"COMPLETED.json").exists() and len(execution.calls) == 1
    with pytest.raises(FileExistsError):
        driver.execute()
    assert (driver.RUN/"STARTED.json").read_bytes() == started and len(execution.calls) == 1


@pytest.mark.parametrize("field,value", (("production_use_allowed", True), ("observed_bend_count", 5),
    ("predictions_not_computed", False), ("step_import_count", 2), ("native_analyzer_count", 2),
    ("evidence_complete", False), ("native_valid", "true")))
def test_malformed_or_unsafe_report_is_not_accepted(driver, field, value):
    result = _invented_result(driver)
    result[field] = value
    with pytest.raises(ValueError):
        driver.validate_result(result)


def test_expired_absolute_budget_has_no_extra_worker_time(driver, monkeypatch):
    monkeypatch.setattr(driver.time, "monotonic", lambda: 200.)
    with pytest.raises(ValueError, match="TOTAL_WALL_EXHAUSTED"):
        driver.remaining(200.)
    assert driver.remaining(205.) == 5.


@pytest.fixture
def coordinator_gate(driver, tmp_path, monkeypatch):
    run, cache = tmp_path/"attempt", tmp_path/"private-cache"
    run.mkdir()
    cache.mkdir(mode=0o700)
    monkeypatch.setattr(driver, "RUN", run)
    monkeypatch.setattr(driver, "__file__", str(run/"source/driver.py"))
    monkeypatch.setattr(driver.os, "getpid", lambda: 100)
    monkeypatch.setattr(driver.os, "getppid", lambda: 99)
    monkeypatch.setattr(driver.os, "getpgrp", lambda: 100)
    monkeypatch.chdir(cache)
    monkeypatch.setenv("TMPDIR", str(cache))
    identity = dict(device=1, inode=2, uid=3)
    helper = SimpleNamespace(cache_identity=lambda p: identity,
                             clean_environment=lambda p: {"TMPDIR": str(p)})
    binding = dict(cad_path=str(tmp_path/"invented.step"), cad_sha256=driver.CAD_SHA,
                   cad_bytes=driver.CAD_BYTES)
    bundle = dict(sources={alias: {"sha256": "c"*64} for alias in driver.SOURCES},
                  runtime_metadata={"invented": True})
    audit = dict(decision=driver.AUDIT_DECISION, bundle_sha256=driver.digest(driver.encoded(bundle)),
                 blocking_findings=[], reviewer="invented independent reviewer")
    began = driver.time.monotonic()
    started = dict(status="STARTED_NO_RETRY", bundle_sha256=driver.digest(driver.encoded(bundle)),
        audit_sha256=driver.digest(driver.encoded(audit)), limits=driver.LIMITS,
        began_monotonic=began, work_stop_monotonic=began+110)
    monkeypatch.setattr(driver, "verify_bundle", lambda: (bundle, binding, helper))
    monkeypatch.setattr(driver, "verify_audit", lambda b: audit)
    driver.exclusive_json(run/"STARTED.json", started)
    driver.exclusive_json(run/"PRE_RUN_AUDIT.json", audit)
    # Issue as the invented outer parent, then authorize as its coordinator.
    monkeypatch.setattr(driver.os, "getpid", lambda: 99)
    request = driver.issue_coordinator(bundle, audit, started, cache, helper)
    monkeypatch.setattr(driver.os, "getpid", lambda: 100)
    return SimpleNamespace(request=request, cache=cache, helper=helper, bundle=bundle,
        binding=binding, audit=audit, started=started, identity=identity)


@pytest.mark.parametrize("capability", ({}, [], {"token": "a"}, {"token": "a"*64, "path": "/outside"}))
def test_coordinator_rejects_malformed_capability_before_io(driver, monkeypatch, capability):
    monkeypatch.setattr(driver, "verify_bundle", lambda: pytest.fail("read before shape gate"))
    with pytest.raises(ValueError, match="ONE_USE_COORDINATOR_REQUIRED"):
        driver.authorize_coordinator(capability)


def test_coordinator_requires_actual_parent_group_and_one_use_claim(driver, coordinator_gate):
    gate = coordinator_gate
    assert driver.authorize_coordinator(gate.request) == (
        gate.bundle, gate.binding, gate.helper, gate.audit, gate.started, gate.cache, gate.identity)
    claim = driver.read_json(driver.RUN/"COORDINATOR_CLAIMED.json")
    assert claim["coordinator_pid"] == 100 and claim["parent_pid"] == 99 and claim["no_retry"]
    with pytest.raises(FileExistsError):
        driver.authorize_coordinator(gate.request)


@pytest.mark.parametrize("fault", ("parent", "group", "token", "started", "terminal", "cache", "wall"))
def test_coordinator_refuses_invalid_control_before_original_read(driver, coordinator_gate, monkeypatch, fault):
    gate = coordinator_gate
    request = dict(gate.request)
    if fault == "parent":
        monkeypatch.setattr(driver.os, "getppid", lambda: 999)
    elif fault == "group":
        monkeypatch.setattr(driver.os, "getpgrp", lambda: 999)
    elif fault == "token":
        request["token"] = "0"*64
    elif fault == "terminal":
        driver.exclusive_json(driver.RUN/"COMPLETED.json", {"status": "TERMINAL"})
    elif fault == "cache":
        gate.helper.cache_identity = lambda p: {"changed": True}
    else:
        started = dict(gate.started)
        if fault == "started":
            started["audit_sha256"] = "0"*64
        else:
            started["work_stop_monotonic"] += 1
        (driver.RUN/"STARTED.json").write_bytes(driver.encoded(started))
        # Preserve the hash binding in the wall-fault control, so its specific
        # absolute-time invariant is reached after the one-use claim is burnt.
        if fault == "wall":
            dispatch = driver.read_json(driver.RUN/"COORDINATOR_DISPATCH.json")
            dispatch["started_sha256"] = driver.digest(driver.encoded(started))
            (driver.RUN/"COORDINATOR_DISPATCH.json").write_bytes(driver.encoded(dispatch))
    original = driver.read_file
    def controls_only(path, *args, **kwargs):
        assert Path(path) != Path(gate.binding["cad_path"])
        return original(path, *args, **kwargs)
    monkeypatch.setattr(driver, "read_file", controls_only)
    with pytest.raises(ValueError):
        driver.authorize_coordinator(request)
    assert (driver.RUN/"COORDINATOR_CLAIMED.json").exists() == (fault in ("cache", "wall"))


@pytest.fixture
def coordinating(driver, coordinator_gate, monkeypatch):
    gate = coordinator_gate
    blob = b"invented text; never a CAD fixture"
    path = Path(gate.binding["cad_path"])
    path.write_bytes(blob)
    monkeypatch.setattr(driver, "CAD_SHA", driver.digest(blob))
    monkeypatch.setattr(driver, "CAD_BYTES", len(blob))
    gate.binding.update(cad_sha256=driver.CAD_SHA, cad_bytes=len(blob))
    calls, reads = [], []
    original = driver.read_file
    def read_after_claim(given, *args, **kwargs):
        if Path(given) == path:
            assert driver.read_json(driver.RUN/"STARTED.json")["status"] == "STARTED_NO_RETRY"
            assert (driver.RUN/"COORDINATOR_CLAIMED.json").exists()
            reads.append(given)
        return original(given, *args, **kwargs)
    monkeypatch.setattr(driver, "read_file", read_after_claim)
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        assert (gate.cache/"cad.step").read_bytes() == blob
        assert kwargs["start_new_session"] is False and kwargs["stdout"] is None
        assert kwargs["stderr"] == driver.subprocess.STDOUT
        dispatch = driver.read_json(driver.RUN/"DISPATCH.json")
        driver.exclusive_json(driver.RUN/"CLAIMED.json", dict(worker_pid=101, parent_pid=100,
            dispatch_sha256=driver.digest(driver.encoded(dispatch)), no_retry=True))
        (gate.cache/"RESULT.json").write_bytes(driver.encoded(_invented_result(driver)))
        return SimpleNamespace(pid=101, stdin=io.BytesIO(), wait=lambda **kw: 0,
                               poll=lambda: 0, returncode=0)
    monkeypatch.setattr(driver.subprocess, "Popen", popen)
    gate.calls, gate.reads, gate.path = calls, reads, path
    return gate


def test_coordinator_reads_exactly_one_claimed_snapshot_and_cleans_inside_group(driver, coordinating):
    gate = coordinating
    driver.coordinator(gate.request)
    assert len(gate.reads) == len(gate.calls) == 1
    assert gate.calls[0][0][-1] == "--worker" and str(gate.path) not in gate.calls[0][0]
    report = driver.read_json(driver.RUN/"COORDINATOR_COMPLETED.json")
    assert report["status"] == "COMPLETE_DEVELOPMENT_DIAGNOSTIC"
    assert report["input_snapshot"]["sha256"] == driver.CAD_SHA
    assert report["input_snapshot"]["bytes"] == driver.CAD_BYTES
    assert report["source_before_after_consistent"] and report["cache_cleanup_complete"]
    assert not gate.cache.exists() and not (driver.RUN/"COMPLETED.json").exists()


def test_coordinator_snapshot_hash_failure_terminalizes_without_native(driver, coordinating):
    gate = coordinating
    gate.path.write_bytes(b"different invented bytes")
    driver.coordinator(gate.request)
    report = driver.read_json(driver.RUN/"COORDINATOR_COMPLETED.json")
    assert report["status"] == "TERMINAL_FAILURE" and not gate.calls and len(gate.reads) == 1
    assert report["phase"] == "APPROVED_DESCRIPTOR_SNAPSHOT" and report["cache_cleanup_complete"]
    assert (driver.RUN/"COORDINATOR_CLAIMED.json").exists()


def test_coordinator_cleanup_failure_is_not_a_success(driver, coordinating, monkeypatch):
    def fail(_):
        raise OSError("invented cleanup failure")
    monkeypatch.setattr(driver.shutil, "rmtree", fail)
    driver.coordinator(coordinating.request)
    report = driver.read_json(driver.RUN/"COORDINATOR_COMPLETED.json")
    assert report["status"] == "TERMINAL_FAILURE" and not report["cache_cleanup_complete"]
    assert report["reason"] == "COORDINATOR_CLEANUP_FAILED"


def test_watchdog_counts_orphaned_group_plus_outer_monitor_without_double_count(driver, monkeypatch):
    monkeypatch.setattr(driver.os, "getpid", lambda: 10)
    table = {10: dict(ppid=1, pgid=10, rss=11), 20: dict(ppid=10, pgid=20, rss=22),
             21: dict(ppid=1, pgid=20, rss=33), 30: dict(ppid=10, pgid=10, rss=44),
             99: dict(ppid=1, pgid=99, rss=999)}
    helper = SimpleNamespace(process_table=lambda: table, descendants=lambda rows, parent: {20, 30})
    assert driver.watchdog_sample(helper, 20) == ({20, 21}, 110, 11)
    del table[10]
    with pytest.raises(ValueError, match="OUTER_RSS_MISSING"):
        driver.watchdog_sample(helper, 20)


def test_total_alarm_kills_group_without_wait_or_exception_cleanup(driver, monkeypatch):
    calls = []
    monkeypatch.setattr(driver, "ACTIVE_GROUP", 123)
    monkeypatch.setattr(driver, "kill_group", lambda group: calls.append(("kill", group)))
    def exited(code):
        calls.append(("exit", code))
        raise SystemExit(code)
    monkeypatch.setattr(driver.os, "_exit", exited)
    with pytest.raises(SystemExit, match="124"):
        driver.hard_exit()
    assert calls == [("kill", 123), ("exit", 124)]


def test_caught_soft_deadline_rearms_total_alarm_before_raising(driver, monkeypatch):
    clock, events, handlers = [100.], [], []
    monkeypatch.setattr(driver, "HARD_END", 120.)
    monkeypatch.setattr(driver.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(driver.signal, "getitimer", lambda *_: (20., 0.))
    def install(signum, handler):
        handlers.append(handler)
        return driver.hard_exit
    monkeypatch.setattr(driver.signal, "signal", install)
    monkeypatch.setattr(driver.signal, "setitimer", lambda *args: events.append(args))
    with driver.deadline(10.):
        soft_handler = handlers[-1]
        clock[0] = 110.
        with pytest.raises(TimeoutError):
            soft_handler()
        assert handlers[-1] is driver.hard_exit and events[-1][1] == 10.
    assert handlers[-1] is driver.hard_exit and events[-1][1] == 10.


def test_outer_watchdog_has_no_cooperative_wait_or_recursive_cleanup(driver):
    tree = ast.parse(SOURCE.read_bytes())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("watch_coordinator", "execute", "hard_exit"):
        attributes = {node.attr for node in ast.walk(functions[name]) if isinstance(node, ast.Attribute)}
        assert not {"wait", "rmtree", "TemporaryDirectory"}.intersection(attributes)
    assert "supervise_process" not in driver.SUPPORT_FUNCTIONS
    assert "hard_deadline(end)" in ast.unparse(functions["execute"])
    # Outer log/ps I/O is also interrupted at the work boundary, with the
    # separately tested repaired soft-handler/hard-alarm composition.
    work_guards = [node for node in ast.walk(functions["execute"]) if isinstance(node, ast.With)
                   and any(ast.unparse(item.context_expr) == "deadline(remaining(work_stop))"
                           for item in node.items)]
    assert any(any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                   and call.func.id == "watch_coordinator" for call in ast.walk(guard))
               for guard in work_guards)


@pytest.mark.parametrize("fault", ("rss", "monitor", "active_row", "output", "startup"))
def test_pure_watchdog_stops_on_missing_monitor_rss_output_or_startup(driver, tmp_path, monkeypatch, fault):
    run, cache = tmp_path/"attempt", tmp_path/"cache"
    run.mkdir()
    cache.mkdir(mode=0o700)
    monkeypatch.setattr(driver, "RUN", run)
    alive, kills, calls = [True], [], []
    output = io.BytesIO()
    output.fileno = lambda: 123  # Owned mock descriptor; os.read is replaced below.
    child = SimpleNamespace(pid=100, stdin=io.BytesIO(), stdout=output,
                            poll=lambda: None if alive[0] else -9)
    def popen(command, **kwargs):
        calls.append(kwargs)
        if fault == "startup":
            raise OSError("invented startup failure")
        return child
    class Selector:
        def register(self, stream, events):
            self.stream = stream
        def select(self, **kwargs):
            return [(SimpleNamespace(fileobj=self.stream), None)]
        def unregister(self, stream):
            pass
        def get_map(self):
            return {123: self.stream}
        def close(self):
            pass
    def kill(group):
        kills.append(group)
        alive[0] = False
    def sample(helper, group):
        if not alive[0]:
            return set(), 1, 1
        if fault == "monitor":
            raise RuntimeError("invented RSS unavailable")
        if fault == "active_row":
            return set(), 1, 1
        return {100}, (1024**3+1 if fault == "rss" else 2), 1
    monkeypatch.setattr(driver.subprocess, "Popen", popen)
    monkeypatch.setattr(driver.selectors, "DefaultSelector", Selector)
    monkeypatch.setattr(driver.os, "read", lambda *args: b"invented output")
    monkeypatch.setattr(driver, "kill_group", kill)
    monkeypatch.setattr(driver, "watchdog_sample", sample)
    if fault == "output":
        monkeypatch.setattr(driver, "LIMITS", {**driver.LIMITS, "combined_worker_output_bytes": 3})
    helper = SimpleNamespace(clean_environment=lambda cache: {"invented": "only"})
    began = driver.time.monotonic()
    receipt = driver.watch_coordinator(["invented-only-command"], {"token": "a"*64},
        cache=cache, helper=helper, work_stop=began+10, end=began+15)
    assert calls[0]["start_new_session"] is True and calls[0]["stderr"] == driver.subprocess.STDOUT
    assert receipt["budget_stop"] is not None and receipt["group_gone"]
    assert driver.ACTIVE_GROUP is None
    assert bool(kills) == (fault != "startup")
    if fault == "rss":
        assert receipt["budget_stop"] == "COMPLETE_GROUP_PLUS_OUTER_RSS_LIMIT"
        assert receipt["sampled_budget_peak_rss_bytes"] == 1024**3+1
    elif fault == "output":
        assert receipt["budget_stop"] == "COMBINED_OUTPUT_BYTE_LIMIT"
        assert (run/"OUTPUT.txt").read_bytes() == b"inv"
        assert receipt["observed_output_bytes"] == len(b"invented output")
    else:
        assert receipt["budget_stop"].startswith("OUTER_WATCHDOG_FAILURE:")


@pytest.mark.parametrize("rows", ([], [{"status": "SINGLE_CAD_WORKER_TERMINAL"}]*2))
def test_outer_requires_exactly_one_native_terminal(driver, rows):
    with pytest.raises(ValueError, match="EXACT_ONE_NATIVE_WORKER_LOG_TERMINAL_REQUIRED"):
        driver.worker_terminal_from_output(b"\n".join(driver.encoded(row) for row in rows))


def _install_smoke_group_proxy(driver, monkeypatch, harness, directory, nonce):
    """Return no new-session child to the watchdog until the OUTER harness ACKs.

    watch_coordinator writes stdin only after Popen returns. A dead pytest
    process closes that pipe; both invented coordinators reject EOF before work.
    """
    harness.smoke_registry_identity(directory)
    original = driver.subprocess.Popen
    def registered_popen(*args, **kwargs):
        child = original(*args, **kwargs)
        if kwargs.get("start_new_session") is True:
            try:
                harness.register_smoke_child(child, directory, nonce)
            except BaseException as original_error:
                cleanup_errors = []
                # Before ACK, the fixed coordinator cannot spawn. If already
                # reaped, never signal its now-unowned numeric PGID again.
                if child.poll() is None:
                    try:
                        driver.os.killpg(child.pid, driver.signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except OSError as error:
                        cleanup_errors.append(harness.smoke_failure(error, "SMOKE_CLIENT_KILL"))
                try:
                    child.wait(timeout=2.)
                except BaseException as error:
                    cleanup_errors.append(harness.smoke_failure(error, "SMOKE_CLIENT_REAP"))
                # Cleanup must not replace the original registration failure.
                try:
                    print(harness.smoke_encoded(dict(status="SMOKE_REGISTRATION_FAILURE",
                        primary_error=harness.smoke_failure(original_error, "SMOKE_CLIENT_REGISTRATION"),
                        cleanup_errors=cleanup_errors)).decode(), flush=True)
                except OSError:
                    pass
                raise
        return child
    monkeypatch.setattr(driver.subprocess, "Popen", registered_popen)


_SMOKE_STDIN_GATE = '''
if sys.stdin.buffer.read(4097) != b'{"token":"' + b'a'*64 + b'"}':
    raise SystemExit("INVENTED_SMOKE_RELEASE_REQUIRED")
'''


def _real_smoke_setup(driver, tmp_path, monkeypatch):
    if os.environ.get("SHERMAN_RUN_V2_NATIVE_VALIDITY_GUARD_SMOKE") != "1":
        pytest.skip("explicit invented subprocess smoke opt-in required")
    spec = importlib.util.spec_from_file_location("_smoke_outer_registry_helpers", ROOT/"scripts/check_cad_guided_measurement.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)  # Inert stdlib helper definitions only.
    assert os.environ.get(harness.SMOKE_REGISTRY_ENV) and os.environ.get(harness.SMOKE_NONCE_ENV), \
        "real smoke requires the acknowledged outer-harness registry"
    _install_smoke_group_proxy(driver, monkeypatch, harness,
        Path(os.environ[harness.SMOKE_REGISTRY_ENV]), os.environ[harness.SMOKE_NONCE_ENV])
    run, cache = tmp_path/"invented-attempt", tmp_path/"invented-cache"
    run.mkdir()
    cache.mkdir(mode=0o700)
    monkeypatch.setattr(driver, "RUN", run)
    helper = driver.generic_support((ROOT/driver.SOURCES["support.py"]).read_bytes())
    return run, cache, helper


def test_optin_real_guarded_stdlib_worker_success(driver, tmp_path, monkeypatch):
    run, cache, helper = _real_smoke_setup(driver, tmp_path, monkeypatch)
    snapshot = tmp_path/"invented-source"
    (snapshot/"diagnostic").mkdir(parents=True)
    (snapshot/"diagnostic/probe.py").write_text("VALUE = 'invented-only'\n")
    outside = tmp_path/"unrelated-invented.txt"
    outside.write_text("must be denied")
    # This fresh process imports only the reviewed stdlib driver/support and
    # the invented namespace. No diagnostic source or original path is used.
    guarded = r'''
import importlib.util, pathlib, socket, subprocess, sys, tempfile
spec = importlib.util.spec_from_file_location("invented_driver", sys.argv[1])
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
support = (driver.ROOT/driver.SOURCES["support.py"]).read_bytes()
helper = driver.generic_support(support)
snapshot, cache, outside = map(pathlib.Path, sys.argv[2:5])
helper.initialize_worker(snapshot, cache, "diagnostic", 10)
from diagnostic.probe import VALUE
assert VALUE == "invented-only"
with tempfile.TemporaryDirectory(dir=cache) as nested:
    path = pathlib.Path(nested)/"invented.txt"
    path.write_text("invented safe cleanup")
    assert path.read_text() == "invented safe cleanup"
denied = 0
for operation in (lambda: outside.read_text(), lambda: socket.socket(),
                  lambda: subprocess.Popen([sys.executable, "-c", "pass"])):
    try:
        operation()
    except PermissionError:
        denied += 1
assert denied == 3
assert not any(name == "OCP" or name.startswith("OCP.") for name in sys.modules)
print("INVENTED_GUARD_OK", flush=True)
'''
    python = str(ROOT/".venv/bin/python")
    worker_command = [python, "-I", "-S", "-B", "-u", "-c", guarded,
                      str(SOURCE), str(snapshot), str(cache), str(outside)]
    coordinator = ("import json,subprocess,sys\n" + _SMOKE_STDIN_GATE +
                   "child=subprocess.Popen(json.loads(sys.argv[1]),start_new_session=False)\nsys.exit(child.wait())")
    command = [python, "-I", "-S", "-B", "-u", "-c", coordinator,
               driver.encoded(worker_command).decode()]
    began = driver.time.monotonic()
    with driver.hard_deadline(began+15.):
        receipt = driver.watch_coordinator(command, {"token": "a"*64}, cache=cache,
                                          helper=helper, work_stop=began+10., end=began+15.)
    assert receipt["exit_code"] == 0 and receipt["budget_stop"] is None and receipt["group_gone"]
    assert b"INVENTED_GUARD_OK" in (run/"OUTPUT.txt").read_bytes()
    assert 0 < receipt["sampled_budget_peak_rss_bytes"] <= 1024**3
    assert receipt["observed_output_bytes"] == receipt["retained_output_bytes"] <= 1024**2
    assert not any(row["pgid"] == receipt["coordinator_pid"] for row in helper.process_table().values())


def test_optin_real_stuck_cleanup_is_killed_with_entire_child_group(driver, tmp_path, monkeypatch):
    run, cache, helper = _real_smoke_setup(driver, tmp_path, monkeypatch)
    python = str(ROOT/".venv/bin/python")
    # Simulates an exception being caught followed by stuck cleanup while a
    # stdlib worker is still alive. It cannot cooperate with the outer timer.
    program = "import subprocess, sys, time\n" + _SMOKE_STDIN_GATE + r'''
child = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", "import time; time.sleep(30)"], start_new_session=False)
print("INVENTED_STUCK_CLEANUP_READY", flush=True)
try:
    raise TimeoutError("invented work deadline")
except TimeoutError:
    time.sleep(30)
'''
    began = driver.time.monotonic()
    with driver.hard_deadline(began+5.):
        receipt = driver.watch_coordinator([python, "-I", "-S", "-B", "-u", "-c", program],
            {"token": "a"*64}, cache=cache, helper=helper, work_stop=began+1., end=began+5.)
    assert b"INVENTED_STUCK_CLEANUP_READY" in (run/"OUTPUT.txt").read_bytes()
    assert receipt["budget_stop"] == "TOTAL_WORK_WALL_LIMIT"
    assert receipt["exit_code"] != 0 and receipt["group_gone"] and not receipt["requires_group_attention"]
    assert driver.time.monotonic()-began < 5.
    assert not any(row["pgid"] == receipt["coordinator_pid"] for row in helper.process_table().values())
