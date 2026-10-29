"""V3 source-only orchestration contracts; no freeze, Gold, geometry or native job.

Tests use the whole runner module with inert top-level constants. Only pytest's
owned temporary directory is written. Native loading/process execution is never
performed; policy hooks and process samples receive explicit invented values.
Root must review/run this named suite with project conftest/plugins disabled.

The initial V2 controls are copied, not imported at runtime. Explicit V3 changes
are fresh V8/V10 constructor identities, shared-supervisor delegation, richer
unknown-stage descriptions and pending-release status. Geometry/configuration,
neutral leaf authority and scorer oracles remain unchanged. New controls use
small invented files or clearly marked orchestration mocks; no real CLI runs.
"""
import ast
from contextlib import contextmanager
import hashlib
import builtins
from copy import deepcopy
import math
import importlib.util
import io
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1]/"scripts/run_gold11_mesh_development_v3.py"
SUPERVISOR = SOURCE.with_name("gold11_owned_process_supervision_v1.py")
PREDECESSOR = SOURCE.with_name("run_gold11_mesh_development_v2.py")
PREDECESSOR_SHA = "94d51b90faef0d88b15ed2d3d84d5b7d36ba94bfb6f63f90b9cc555a35e26422"
RESOURCE_LIMITS = dict(wall_seconds=60, worker_cpu_seconds=45,
    sampled_group_rss_bytes=1024**3, output_bytes=1024**2)


@pytest.fixture
def runner(monkeypatch):
    spec = importlib.util.spec_from_file_location("gold11_mesh_runner_v3_source_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    helper = ModuleType("_invented_runner_supervision_source")
    exec(compile(SUPERVISOR.read_bytes(), str(SUPERVISOR), "exec"), helper.__dict__)
    module._real_supervision_loader = module.supervision_module
    monkeypatch.setattr(module, "supervision_module", lambda: helper)
    def no_process(*args, **kwargs):
        pytest.fail("This pure suite cannot start or inspect an actual process")
    monkeypatch.setattr(module.subprocess, "Popen", no_process)
    monkeypatch.setattr(module.subprocess, "run", no_process)
    return module


def test_top_level_imports_are_only_stdlib_no_app_or_native(runner):
    tree = ast.parse(SOURCE.read_text())
    names = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import)
             for alias in node.names}
    names |= {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert names <= {"argparse", "ast", "contextlib", "fcntl", "hashlib", "importlib", "json", "os", "pathlib",
                     "math", "resource", "secrets", "selectors", "signal", "stat", "subprocess", "sys", "tempfile", "time", "types"}
    assert "OCP" not in names and "numpy" not in names and "backend" not in names


def test_fixed_limits_separate_worker_coordinator_and_scorer(runner):
    limits = runner.LIMITS
    assert (limits["case_wall_seconds"], limits["case_cpu_seconds"], limits["sampled_worker_tree_rss_bytes"]) == (180, 150, 8*1024**3)
    assert (limits["scorer_wall_seconds"], limits["scorer_cpu_seconds"], limits["scorer_rss_bytes"]) == (60, 50, 2*1024**3)
    assert limits["global_tree_rss_bytes"] == 12*1024**3 < 16*1024**3
    assert limits["coordinator_rss_bytes"] == 4*1024**3
    assert limits["total_wall_seconds"] == 4200
    assert RESOURCE_LIMITS == dict(wall_seconds=60, worker_cpu_seconds=45,
        sampled_group_rss_bytes=1024**3, output_bytes=1024**2)


@pytest.mark.parametrize("role,entry", (("INFERENCE", "worker"), ("SCORER", "scorer_worker")))
def test_direct_leaf_cli_paths_are_rejected_before_any_read_or_import(runner, monkeypatch, role, entry):
    direct = dict(snapshot="/arbitrary/source", cache="/arbitrary/input", arm="control",
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
        binding = (dict(arm="control", scan_sha256="a"*64, cad_sha256="b"*64)
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
        runner.issue_leaf_dispatch("INFERENCE", tmp_path, arm="control", scan_sha256="a"*64, cad_sha256="b"*64)


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
    source, _, functions = _source_nodes(SOURCE)
    delegate = ast.get_source_segment(source, functions["supervise_process"])
    assert "supervision_module().supervise_process(" in delegate
    execute = ast.get_source_segment(source, functions["execute"])
    assert "new_session=True" in execute and 'LIMITS["global_tree_rss_bytes"]' in execute
    assert 'receipt.get("owned_group_empty_confirmed") is not True' in execute
    assert 'receipt.get("direct_child_reaped") is not True' in execute


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
    bundle = {"arms": ["control", "challenger"], "cases": [{"case_token": f"invented-{index}"} for index in range(11)]}
    rows = runner.terminalize_case_records(bundle, "INVENTED_GLOBAL_STOP")
    assert len(rows) == 22 and all(row["durable"] for row in rows)
    first = tmp_path/"cases/invented-0/control/SCORING_INPUT.json"
    before = first.read_bytes()
    again = runner.terminalize_case_records(bundle, "MUST_NOT_REPLACE")
    assert len(again) == 22 and first.read_bytes() == before
    assert runner.decode(before)["predictions"] == []


def test_source_orders_gold_after_22_terminals_and_verifies_after_scoring(runner):
    source = SOURCE.read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    coordinator = ast.get_source_segment(source, functions["internal_coordinator"])
    assert coordinator.index('RUN/"INFERENCE_TERMINAL.json"') < coordinator.index("score_after_terminal(bundle, outcomes, receipts)")
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



@pytest.mark.skip(reason="Actual guarded runtime smoke is disabled pending a separately reviewed release")
def test_actual_guarded_runtime_smoke_is_not_released():
    """No environment variable enables native/process execution in this suite."""
    raise AssertionError("No actual runtime smoke body is authorized or included")


# Independent literal oracle for the nine explicit challenger constructor
# dictionaries. No application module is imported to discover these defaults.
EXPECTED_V8_CONFIG = {
    "noisy_policy": dict(tile_triangles=[128, 1024], halo_rings=1, min_support_rows=3,
        max_normal_angle_radians=math.pi/18., min_inplane_covariance_ratio=1e-8),
    "noisy_budgets": dict(max_raw_rows=1_000_000, max_triangles=2_000_000,
        max_edge_records=6_000_000, max_core_records=65_536, max_core_triangle_rows=4096,
        max_triangle_memberships=20_000_000, max_raw_memberships=32_000_000,
        max_full_support_row_tests=64_000_000, max_cone_row_tests=16_000_000,
        max_closure_triangle_tests=32_000_000, max_closure_edge_tests=96_000_000,
        max_fit_evaluations=65_552, max_closure_schedules=16,
        max_retained_array_bytes=536_870_912, max_metadata_bytes=67_108_864, chunk_rows=65_536),
    "cad_patch_policy": dict(max_pair_normal_sine=1e-8, max_pair_anchored_distance_mm=1e-8,
        max_full_support_distance_mm=1e-8, min_support_vertices=12,
        min_second_extent_mm=.25, min_inplane_covariance_ratio=1e-8),
    "cad_patch_budgets": dict(max_vertices=1_000_000, max_triangles=2_000_000,
        max_edge_records=6_000_000, max_components=65_536, max_output_patches=4096,
        max_support_memberships=6_000_000, max_full_vertex_tests=12_000_000, chunk_triangles=65_536),
    "search_policy": dict(min_scan_support_count=3, max_scan_support_rms_mm=.1,
        min_normal_condition_ratio=.02, max_normal_alignment_extra_radians=.02,
        max_intersection_backward_residual_mm=1e-8, pair_angle_numeric_guard_radians=1e-10),
    "search_budgets": dict(max_scan_planes=65_552, max_cad_planes=2048, max_bank_planes=128,
        max_scan_triplets=256, max_scan_partner_tests=1_048_576, max_cad_pair_records=2_096_128,
        max_pair_join_tests=32_000_000, max_cad_pair_samples_per_triplet=12,
        max_unsigned_per_scan_triplet=4, max_compatible_rows_per_triplet=49_104,
        max_unsigned_assignments=1024, max_signed_assignments=8192,
        max_report_bytes=67_108_864, chunk_pair_rows=65_536),
    "initialization_budgets": dict(max_hypothesis_records=8192, max_prefilter_unsigned_assignments=1024,
        max_cached_cad_faces=2048, max_indexed_face_triangles=500_000,
        max_prefilter_query_calls=32768, max_prefilter_query_rows=64_000_000,
        chunk_rows=65_536, max_node_visits=512_000_000, max_triangle_pairs=256_000_000,
        max_normal_pairs=32_000_000, max_work_units=4_000_000_000,
        max_face_build_nodes=131_071, max_face_build_triangle_visits=10_000_000,
        max_refined_start_groups=48, max_report_bytes=100_663_296),
    "surface_budgets": dict(max_triangles=500_000, max_query_rows=200_000, leaf_size=16,
        max_build_nodes=131_071, max_build_triangle_visits=10_000_000,
        max_node_visits=32_000_000, max_triangle_pairs=32_000_000,
        max_normal_pairs=32_000_000, max_work_units=512_000_000,
        max_pair_batch=65_536, max_ties_per_query=32),
    "registration_budgets": dict(max_query_calls=4096, max_query_rows=64_000_000,
        max_node_visits=2_000_000_000, max_triangle_pairs=2_000_000_000,
        max_normal_pairs=500_000_000, max_work_units=20_000_000_000),
}
EXPECTED_V8_TYPES = {
    "noisy_policy": ("cad_guided_noisy_mesh_planes_v5", "NoisyMeshPlanePolicyV5"),
    "noisy_budgets": ("cad_guided_noisy_mesh_planes_v5", "NoisyMeshPlaneBudgetsV5"),
    "cad_patch_policy": ("cad_guided_mesh_plane_proposals_v4", "CoplanarPatchPolicyV4"),
    "cad_patch_budgets": ("cad_guided_mesh_plane_proposals_v4", "CoplanarPatchBudgetsV4"),
    "search_policy": ("cad_guided_plane_search_v6", "PlaneSearchPolicyV6"),
    "search_budgets": ("cad_guided_plane_search_v6", "PlaneSearchBudgetsV6"),
    "initialization_budgets": ("cad_guided_plane_initialization_v8", "PlaneInitializationBudgetsV8"),
    "surface_budgets": ("cad_guided_triangle_surface_v2", "TriangleSurfaceBudgets"),
    "registration_budgets": ("cad_guided_global_registration_v2", "SurfaceRegistrationBudgetsV2"),
}


def _source_nodes(path):
    """Read only the exact named runner source; no imports or AST execution."""
    text = path.read_text()
    tree = ast.parse(text)
    assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    return text, assignments, functions


def test_original_scoring_and_four_common_configs_are_unchanged(runner):
    assert hashlib.sha256(PREDECESSOR.read_bytes()).hexdigest() == PREDECESSOR_SHA
    old_text, old, old_functions = _source_nodes(PREDECESSOR)
    text, new, functions = _source_nodes(SOURCE)
    for name in ("CONFIG", "LIMITS", "SCORING_FIELDS", "REVIEW", "REVIEW_SHA", "CATALOG", "CATALOG_SHA", "SELECTED_SHA"):
        assert ast.dump(old[name]) == ast.dump(new[name]), name
    for name in ("compact_result", "selected_references", "approved_cases_from_review"):
        assert ast.get_source_segment(old_text, old_functions[name]) == ast.get_source_segment(text, functions[name])
    assert runner.CERTIFICATE_CAPS == dict(initialization_budgets=1_024_000_000,
        surface_budgets=128_000_000, registration_budgets=8_000_000_000)
    scorer = functions["scorer_worker"]
    imports = [node for node in ast.walk(scorer) if isinstance(node, ast.ImportFrom)]
    assert [(node.module, [alias.name for alias in node.names]) for node in imports] == [
        ("scripts.gold11_mesh_scoring_v1", ["score_gold11_mesh_workflows_v1"])]
    calls = [node for node in ast.walk(scorer) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "score_gold11_mesh_workflows_v1"]
    assert len(calls) == 1 and ast.unparse(calls[0]) == "score_gold11_mesh_workflows_v1(**payload)"
    assert runner.RETAINED_CONTROL_POLICY == dict(
        primary_nonzero_control="SEPARATELY_FROZEN_CACHED_UNION",
        cached_control_value="NOT_ASSUMED_OR_LOADED_BY_THIS_RUNNER",
        comparison="SEPARATE_PROVISIONAL_RAW_ENDPOINT_GEOMETRY_LEDGER_ONLY",
        requires="INDEPENDENT_CONTROL_RELEASE_COMMON71_REFERENCE_SHA_UNCHANGED_MATCHER",
        cached_payload_as_modern_measured_carrier=False, historical_replay_claim_allowed=False,
        fresh_v8_gain_is_not_cached_union_superiority=True)
    assert not any(isinstance(node, ast.ImportFrom) and node.module and
        ("fusion" in node.module or "cached" in node.module) for node in ast.walk(ast.parse(text)))


def test_explicit_constructor_defaults_are_json_stable_and_uncalibrated(runner):
    assert {key: runner.CONFIG[key] for key in EXPECTED_V8_CONFIG} == EXPECTED_V8_CONFIG
    assert runner.decode(runner.encoded(runner.CONFIG)) == runner.CONFIG
    assert type(runner.CONFIG["noisy_policy"]["tile_triangles"]) is list
    assert type(runner.CONFIG["position_error_bound_mm"]) is float
    assert runner.CONFIG["position_error_bound_mm"] == .05
    assert runner.POSITION_ERROR_DECLARATION == dict(
        status="EXPLORATORY_STIPULATED_UNCALIBRATED_PROPOSAL_BOUND",
        scope="BOTH_ARMS_PLANE_INITIALIZATION_ONLY_NOT_MEASUREMENT_UNCERTAINTY",
        uniform_across_all_cases=True, physical_scanner_bound_established=False,
        selected_from_gold=False, constructed_point005_bound_is_not_physical_evidence=True)
    assert runner.ARMS == ["control", "challenger"]
    assert runner.ARM_DEFINITIONS == dict(
        control=dict(workflow="CAD_GUIDED_MESH_OFFLINE_V8", native_adapter_version="v1",
            registration="V8_ALL_GENERATED_NOISY_PLANE_STARTS_FINITE_TRIANGLE_CORRESPONDENCE_V2_REFINEMENT",
            inference="FRESH_ONCE_FOR_EACH_OF_THE_FIXED_ELEVEN_CASES"),
        challenger=dict(workflow="CAD_GUIDED_MESH_OFFLINE_V10", native_adapter_version="v1",
            registration="V10_FULL_FACE_BOX_REJECTION_AND_V3_BOUND_AWARE_FINITE_SURFACE_CONSUMERS",
            inference="FRESH_ONCE_FOR_EACH_OF_THE_FIXED_ELEVEN_CASES"))
    assert runner.CONFIG["initialization_budgets"]["max_prefilter_unsigned_assignments"] == (
        runner.CONFIG["search_budgets"]["max_unsigned_assignments"]) == 1024
    assert runner.COMPARATOR == "FRESH_V8_VS_V10_FIXED_NATIVE_V1_NOT_BOX_ONLY_AND_NOT_CACHED_UNION_COMPARISON"


def test_leaf_source_denominators_do_not_expose_review_or_gold_to_inference(runner):
    assert len(runner.SOURCES) == len(set(runner.SOURCES)) == 38
    inference = runner.leaf_sources("INFERENCE")
    scorer = runner.leaf_sources("SCORER")
    assert set(inference) == {runner.RUNNER_SOURCE} | {
        name for name in runner.SOURCES if name.startswith("backend/")}
    assert len(inference) == 30
    assert set(scorer) == {runner.RUNNER_SOURCE, "scripts/gold11_mesh_scoring_v1.py",
        "scripts/evaluate_f1_against_manual_gold.py", "scripts/f1_physical_evaluation_contract_v2.py"}
    assert runner.REVIEW not in inference and runner.PROTOCOL not in inference
    assert runner.SUPERVISION_SOURCE not in inference
    assert not any("catalog" in name or "review_gold11_inputs" in name for name in inference)
    _, _, functions = _source_nodes(SOURCE)
    backend_imports = {node.module for node in ast.walk(functions["workflow_entry_and_policies"])
                       if isinstance(node, ast.ImportFrom) and node.module.startswith("backend.")}
    assert {name.replace(".", "/")+".py" for name in backend_imports} <= set(inference)


@pytest.fixture
def mocked_worker(runner, tmp_path, monkeypatch):
    """Constructor/API doubles only; no actual backend import or input bytes."""
    modules, calls, imports = {}, [], []
    pairs = list(EXPECTED_V8_TYPES.values()) + [
        ("cad_guided_global_registration_v1", "WholePartRegistrationPolicyV1"),
        ("cad_guided_native_features", "NativeGuidedPolicy"),
        ("cad_guided_instances", "SupportPolicy"),
        ("cad_guided_mesh_offline_v8", "MeshWorkflowPolicyV8"),
        ("cad_guided_mesh_offline_v10", "MeshWorkflowPolicyV10"),
        ("cad_guided_plane_initialization_v10", "PlaneInitializationBudgetsV10"),
        ("cad_guided_triangle_surface_v3", "TriangleSurfaceBudgets"),
        ("cad_guided_global_registration_v3", "SurfaceRegistrationBudgetsV3")]
    def construct(self, **kwargs): self.kwargs = deepcopy(kwargs)
    for module_name, class_name in pairs:
        full_name = "backend."+module_name
        module = modules.setdefault(full_name, ModuleType(full_name))
        setattr(module, class_name, type(class_name, (), {"__init__": construct, "__module__": full_name}))
    result = dict(status="UNAVAILABLE", phase="INVENTED_MOCK_ONLY", predictions=[],
                  observed_bend_count=None, production_use_allowed=False)
    def inspect(**kwargs):
        calls.append(kwargs)
        return result
    modules["backend.cad_guided_mesh_offline_v8"].inspect_cad_guided_mesh_v8 = inspect
    modules["backend.cad_guided_mesh_offline_v10"].inspect_cad_guided_mesh_v10 = inspect
    original_import = builtins.__import__
    def import_stub(name, globals=None, locals=None, fromlist=(), level=0):
        if name in modules:
            imports.append(name)
            return modules[name]
        if name.split(".")[0] in {"backend", "OCP", "numpy", "scipy"}:
            pytest.fail("Unexpected application/native import in mocked worker")
        return original_import(name, globals, locals, fromlist, level)
    cache, snapshot = tmp_path/"neutral-cache", tmp_path/"inert-source-not-read"
    cache.mkdir()
    reads, initialized, cpu_calls = [], [], []
    request = dict(dispatch_id="a"*64, token="b"*64)
    def invoke(arm):
        binding = dict(arm=arm, scan_sha256="c"*64, cad_sha256="d"*64)
        def authorize(value, role):
            assert value == request and role == "INFERENCE"
            return snapshot, cache, binding
        def read(path, cap, sha=None):
            assert Path(path) in {cache/"scan.ply", cache/"cad.step"}
            reads.append((Path(path).name, cap, sha))
            return b"INVENTED_SCAN_BYTES" if Path(path).suffix == ".ply" else b"INVENTED_CAD_BYTES"
        with monkeypatch.context() as scoped:
            scoped.setattr(builtins, "__import__", import_stub)
            scoped.setattr(runner, "authorize_leaf", authorize)
            scoped.setattr(runner, "bounded_read", read)
            scoped.setattr(runner, "initialize_worker", lambda *args: initialized.append(args))
            scoped.setattr(runner.resource, "setrlimit", lambda *args: cpu_calls.append(args))
            scoped.setattr(runner.resource, "getrusage", lambda *args: SimpleNamespace(ru_utime=.1, ru_stime=.2, ru_maxrss=1024))
            scoped.setattr(runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(runner.encoded(request))))
            runner.worker()
    return SimpleNamespace(invoke=invoke, result=result, cache=cache, snapshot=snapshot,
        calls=calls, reads=reads, initialized=initialized, cpu_calls=cpu_calls, imports=imports)


@pytest.mark.parametrize("arm", ("control", "challenger"))
def test_worker_uses_fixed_native_v1_and_exact_versioned_kwargs(runner, mocked_worker, arm):
    before = deepcopy(runner.CONFIG)
    mocked_worker.invoke(arm)
    assert len(mocked_worker.calls) == 1
    call = mocked_worker.calls[0]
    common = {"scan_bytes", "cad_bytes", "registration_policy", "measurement_policy",
              "instance_policy", "allow_ocp", "workflow_policy"}
    assert set(call) == common | set(EXPECTED_V8_CONFIG) | {"position_error_bound_mm"}
    assert call["scan_bytes"] == b"INVENTED_SCAN_BYTES" and call["cad_bytes"] == b"INVENTED_CAD_BYTES"
    assert call["allow_ocp"] is True
    for key, name, config in (("registration_policy", "WholePartRegistrationPolicyV1", "registration"),
                              ("measurement_policy", "NativeGuidedPolicy", "measurement"),
                              ("instance_policy", "SupportPolicy", "support")):
        assert type(call[key]).__name__ == name and call[key].kwargs == runner.CONFIG[config]
    version = "8" if arm == "control" else "10"
    assert type(call["workflow_policy"]).__name__ == "MeshWorkflowPolicyV"+version
    assert call["workflow_policy"].kwargs == dict(runner.CONFIG["workflow"], native_adapter_version="v1")
    expected_types = dict(EXPECTED_V8_TYPES)
    if arm == "challenger":
        expected_types.update(
            initialization_budgets=("cad_guided_plane_initialization_v10", "PlaneInitializationBudgetsV10"),
            surface_budgets=("cad_guided_triangle_surface_v3", "TriangleSurfaceBudgets"),
            registration_budgets=("cad_guided_global_registration_v3", "SurfaceRegistrationBudgetsV3"))
    for key, (module_name, name) in expected_types.items():
        expected = deepcopy(EXPECTED_V8_CONFIG[key])
        if key == "noisy_policy": expected["tile_triangles"] = (128, 1024)
        if arm == "challenger" and key in runner.CERTIFICATE_CAPS:
            expected["max_certificate_pairs"] = runner.CERTIFICATE_CAPS[key]
        assert (type(call[key]).__module__, type(call[key]).__name__) == ("backend."+module_name, name)
        assert call[key].kwargs == expected
    assert call["position_error_bound_mm"] == .05 and runner.CONFIG == before
    assert "backend.cad_guided_mesh_offline_v1" not in mocked_worker.imports
    assert mocked_worker.reads == [("scan.ply", 64*1024**2, "c"*64), ("cad.step", 8*1024**2, "d"*64)]
    assert mocked_worker.initialized == [(mocked_worker.snapshot, mocked_worker.cache, "backend", 150)]
    assert mocked_worker.cpu_calls == [(runner.resource.RLIMIT_CPU, (150, 150))]
    assert runner.decode((mocked_worker.cache/"RESULT.json").read_bytes()) == mocked_worker.result


@pytest.mark.parametrize("arm", ("control", "challenger"))
@pytest.mark.parametrize("extra", (0, -1))
def test_equal_transport_cap_applies_after_each_mocked_arm(runner, mocked_worker, monkeypatch, arm, extra):
    blob = runner.encoded(mocked_worker.result)
    monkeypatch.setitem(runner.LIMITS, "case_result_bytes", len(blob)+extra)
    if extra == 0:
        mocked_worker.invoke(arm)
        assert (mocked_worker.cache/"RESULT.json").read_bytes() == blob
    else:
        with pytest.raises(ValueError, match="RESULT_BYTE_BUDGET"):
            mocked_worker.invoke(arm)
        assert not (mocked_worker.cache/"RESULT.json").exists()


@pytest.mark.parametrize("arm", ("r0", "v1", "v7", "unknown", None))
def test_old_or_arbitrary_arm_cannot_be_used_as_neutral_binding(runner, arm):
    with pytest.raises(ValueError, match="LEAF_ARM_SCHEMA"):
        runner.validate_leaf_binding("INFERENCE", dict(arm=arm, scan_sha256="a"*64, cad_sha256="b"*64))


@pytest.mark.parametrize("arm,registration,native", (
    ("control", "CONDITIONAL_BEST_FIT", "UNAVAILABLE"),
    ("challenger", "CONDITIONAL_BEST_FIT", "UNAVAILABLE"),
    ("control", "UNAVAILABLE", "UNAVAILABLE"),
    ("control", "CONDITIONAL_BEST_FIT", "PARTIAL"),
))
def test_nested_stages_preserve_returned_phase_without_legacy_v1_reclassification(runner, arm, registration, native):
    payload = dict(status="UNAVAILABLE", phase="WHOLE_PART_REGISTRATION", predictions=[],
        registration=dict(status=registration, reasons=["INVENTED_REGISTRATION_REASON"], pose="must-not-leak"),
        native_features=dict(status=native, reasons=["INVENTED_NATIVE_REASON"]),
        measurements=None)
    before = deepcopy(payload)
    result = runner.workflow_stage_diagnostics(payload, arm)
    assert "legacy_stale_phase_observed" not in result
    assert result["missing_stage_policy"] == "UNKNOWN_NEVER_INFER_NOT_REACHED_OR_COMPLETED"
    assert result["reported_workflow_phase"] == "WHOLE_PART_REGISTRATION"
    assert result["stages"]["registration"]["status"] == registration
    assert result["stages"]["native_features"]["status"] == native
    assert result["stages"]["measurements"]["evidence"] == "NULL_UNKNOWN"
    assert result["stages"]["scan_topology"]["evidence"] == "MISSING_UNKNOWN"
    assert "must-not-leak" not in runner.encoded(result).decode() and payload == before
    assert runner.compact_result(payload)["status"] == "UNAVAILABLE"
    assert runner.compact_result(payload)["phase"] == "WHOLE_PART_REGISTRATION"


@pytest.mark.parametrize("value", (None, [], "bad result"))
def test_malformed_workflow_stage_projection_remains_unknown(runner, value):
    assert runner.workflow_stage_diagnostics(value, "control") == dict(
        evidence="NO_RETURNED_WORKFLOW_INTERNAL_STAGES_UNKNOWN", stages={}, reported_workflow_phase=None,
        arm="control", missing_stage_policy="UNKNOWN_NEVER_INFER_NOT_REACHED_OR_COMPLETED")


@pytest.mark.parametrize("reasons", (["x"]*33, ["x"*257], "not-a-list", [None]))
def test_stage_projection_refuses_malformed_reasons_without_inventing_status(runner, reasons):
    payload = dict(phase="x"*129, registration=dict(status="x"*129, reasons=reasons,
                   detailed_report_discarded_whole=1), native_features=[])
    result = runner.workflow_stage_diagnostics(payload, "control")
    assert result["reported_workflow_phase"] is None
    row = result["stages"]["registration"]
    assert row == dict(evidence="PRESENT", status=None, reasons=[], reasons_complete=False,
                      detailed_report_discarded_whole=False)
    assert result["stages"]["native_features"]["evidence"] == "MALFORMED_UNKNOWN"
    assert "legacy_stale_phase_observed" not in result


@pytest.mark.parametrize("abort_after", (None, 0, 11, 21))
def test_mocked_coordinator_scores_only_after_all_22_neutral_terminals(runner, tmp_path, monkeypatch, abort_after):
    """Entire authority/source/process/score boundaries are invented mocks."""
    monkeypatch.setattr(runner, "RUN", tmp_path)
    bundle = dict(arms=["control", "challenger"], cases=[dict(case_token=f"invented-{i}") for i in range(11)])
    audit, token = {"invented_audit": True}, "e"*64
    expected = dict(parent_pid=90, token_sha256=runner.digest(token.encode()))
    start = dict(status="STARTED_NO_RETRY", limits=runner.LIMITS,
        bundle_sha256=runner.digest(runner.encoded(bundle)), audit_sha256=runner.digest(runner.encoded(audit)))
    controls = {"COORDINATOR_DISPATCH.json": expected, "STARTED.json": start}
    events, records, terminal_fills = [], {}, []
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(runner.encoded(dict(token=token, parent_pid=90)))))
    monkeypatch.setattr(runner.os, "getpid", lambda: 91)
    monkeypatch.setattr(runner.os, "getppid", lambda: 90)
    monkeypatch.setattr(runner.os, "getpgrp", lambda: 91)
    monkeypatch.setattr(runner, "verify_bundle", lambda: bundle)
    monkeypatch.setattr(runner, "verify_pre_run_audit", lambda value: audit)
    def read_control(path, *args):
        assert Path(path).parent == tmp_path and Path(path).name in controls
        return runner.encoded(controls[Path(path).name])
    monkeypatch.setattr(runner, "bounded_read", read_control)
    monkeypatch.setattr(runner, "activate_coordinator", lambda *args: events.append("ACTIVATED"))
    monkeypatch.setattr(runner, "coordinator_memory_check", lambda: None)
    attempted = []
    def supervise(case, arm, destination, remaining):
        assert destination == tmp_path/"cases"/case["case_token"]/arm and 0 < remaining <= 4170
        if len(attempted) == abort_after:
            raise RuntimeError("INVENTED_TERMINAL_STORAGE_FAILURE")
        attempted.append((case["case_token"], arm))
        events.append("INFERENCE")
        return runner.unavailable("INVENTED_NEGATIVE"), dict(result_sha256="f"*64)
    monkeypatch.setattr(runner, "supervise_case", supervise)
    def exclusive(path, payload):
        assert path.parent == tmp_path and path.name not in records
        records[path.name] = deepcopy(payload)
        events.append(path.name)
    monkeypatch.setattr(runner, "exclusive_json", exclusive)
    def score(value, outcomes, receipts):
        assert value is bundle and len(attempted) == 22
        assert events[-1] == "INFERENCE_TERMINAL.json"
        assert records[events[-1]]["terminal_record_count"] == 22
        assert set(outcomes) == set(receipts) == {"control", "challenger"}
        assert all(len(outcomes[arm]) == len(receipts[arm]) == 11 for arm in receipts)
        assert all(row["predictions"] == [] for arm in outcomes.values() for row in arm.values())
        events.append("POST_TERMINAL_SCORE_MOCK_NO_GOLD")
    monkeypatch.setattr(runner, "score_after_terminal", score)
    def fill(value, reason):
        terminal_fills.append((value, reason))
        return [dict(durable=True) for _ in range(22)]
    monkeypatch.setattr(runner, "terminalize_case_records", fill)
    runner.internal_coordinator()
    wanted = [(f"invented-{i}", arm) for i in range(11) for arm in ("control", "challenger")]
    assert attempted == (wanted if abort_after is None else wanted[:abort_after])
    if abort_after is None:
        assert "POST_TERMINAL_SCORE_MOCK_NO_GOLD" in events and not terminal_fills
        assert records["COORDINATOR_COMPLETED.json"]["status"] == "COMPUTED_PENDING_INDEPENDENT_POST_AUDIT"
        assert "SCORING_UNAVAILABLE.json" not in records
    else:
        assert "POST_TERMINAL_SCORE_MOCK_NO_GOLD" not in events and "INFERENCE_TERMINAL.json" not in records
        assert terminal_fills == [(bundle, "COORDINATOR_ABORT_NO_RETRY")]
        assert len(records["SCORING_UNAVAILABLE.json"]["terminal_case_records"]) == 22
        assert records["COORDINATOR_COMPLETED.json"]["status"] == "TERMINAL_SCORING_UNAVAILABLE"
    assert records["COORDINATOR_COMPLETED.json"]["production_use_allowed"] is False


@pytest.mark.parametrize("scorer", (False, True))
@pytest.mark.parametrize("mode", ("complete", "partial", "empty_partial", "linked", "none"))
def test_actual_owned_retention_keeps_complete_or_partial_bytes_once(runner, tmp_path, scorer, mode):
    cache, destination = tmp_path/"cache", tmp_path/"retained"
    cache.mkdir(); destination.mkdir()
    name = "SCORER_RESULT.json" if scorer else "RESULT.json"
    partial = cache/("."+name+".partial-"+"a"*16)
    blob = b'{"invented":true}'
    if mode in ("complete", "linked"): (cache/name).write_bytes(blob)
    if mode == "linked": os.link(cache/name, partial)
    if mode in ("partial", "empty_partial"):
        blob = blob if mode == "partial" else b""
        partial.write_bytes(blob)
    result = runner.retain_worker_result(cache, destination, scorer=scorer)
    if mode == "none":
        assert result == dict(status="NO_RETURNED_RESULT_INTERNAL_STAGES_UNKNOWN", artifacts=[])
        assert list(destination.iterdir()) == []
        return
    full = mode in ("complete", "linked")
    target = name if full else name.removesuffix(".json")+"_PARTIAL.bin"
    assert result["status"] == ("COMPLETE_RESULT_BYTES_RETAINED" if full else "PARTIAL_BYTES_NOT_SCORABLE")
    assert result["artifacts"] == [dict(path=target, sha256=runner.digest(blob), bytes=len(blob))]
    assert (destination/target).read_bytes() == blob
    assert result["partial_alias_bytes_retained_once"] is (mode == "linked")
    assert len(list(destination.iterdir())) == 1


@pytest.mark.parametrize("fault", ("second_partial", "bad_suffix", "unbound_complete", "symlink", "over_cap"))
def test_retention_refuses_unbound_or_unbounded_bytes_without_publishing(runner, tmp_path, monkeypatch, fault):
    cache, destination = tmp_path/"cache", tmp_path/"retained"
    cache.mkdir(); destination.mkdir()
    partial = cache/(".RESULT.json.partial-"+"a"*16)
    partial.write_bytes(b"invented")
    if fault == "second_partial": (cache/(".RESULT.json.partial-"+"b"*16)).write_bytes(b"other")
    elif fault == "bad_suffix": partial.rename(cache/".RESULT.json.partial-INVALID")
    elif fault == "unbound_complete": (cache/"RESULT.json").write_bytes(b"different")
    elif fault == "symlink":
        partial.unlink(); partial.symlink_to(tmp_path/"unread_missing")
    else: monkeypatch.setitem(runner.LIMITS, "case_result_bytes", 2)
    with pytest.raises((ValueError, OSError)):
        runner.retain_worker_result(cache, destination)
    assert list(destination.iterdir()) == []


def _artifact_fixture(runner, tmp_path, monkeypatch):
    """Actual invented owned files; scorer semantics are NOT evaluated here."""
    monkeypatch.setattr(runner, "RUN", tmp_path)
    cases = [dict(case_token=f"invented-{i}", part_key=f"family-{i%6}") for i in range(11)]
    bundle = dict(cases=cases, arms=["control", "challenger"])
    def put(path, value):
        target = tmp_path/path
        target.parent.mkdir(parents=True, exist_ok=True)
        blob = runner.encoded(value)+b"\n"
        target.write_bytes(blob)
        return runner.digest(blob)
    def get(path): return runner.decode((tmp_path/path).read_bytes())
    outcomes, bindings = {arm: {} for arm in runner.ARMS}, {arm: {} for arm in runner.ARMS}
    for case in cases:
        for arm in runner.ARMS:
            key, prefix = case["case_token"], f'cases/{case["case_token"]}/{arm}/'
            result_sha = put(prefix+"RESULT.json", dict(status="UNAVAILABLE", predictions=[], invented=True))
            outcome = runner.unavailable("INVENTED_GEOMETRIC_REFUSAL")
            input_sha = put(prefix+"SCORING_INPUT.json", outcome)
            put(prefix+"STARTED.json", dict(case_id=key, arm=arm, limits=runner.LIMITS, no_resume=True))
            (tmp_path/(prefix+"OUTPUT.txt")).write_bytes(b"ok\n")
            put(prefix+"COMPLETED.json", dict(status="TERMINAL", exit_code=0, budget_stop=None,
                source_binding_valid=True, inference_started=True, derived_result_retention_complete=True,
                result_sha256=result_sha, scoring_input_sha256=input_sha,
                log_created=True, forwarded_log_bytes=3, stored_log_bytes=3,
                retained_result_artifacts=dict(status="COMPLETE_RESULT_BYTES_RETAINED",
                    artifacts=[dict(path="RESULT.json", sha256=result_sha,
                        bytes=(tmp_path/(prefix+"RESULT.json")).stat().st_size)])))
            outcomes[arm][key], bindings[arm][key] = outcome, result_sha
    put("COORDINATOR_COMPLETED.json", dict(status="COMPUTED_PENDING_INDEPENDENT_POST_AUDIT"))
    put("INFERENCE_TERMINAL.json", dict(case_count=11, arm_count=2, terminal_record_count=22,
        outcomes_sha256=runner.digest(runner.encoded(outcomes)), result_bindings=bindings))
    supervisor = dict(exit_code=0, budget_stop=None, log_created=True, forwarded_log_bytes=3, stored_log_bytes=3)
    (tmp_path/"COORDINATOR_OUTPUT.txt").write_bytes(b"ok\n")
    (tmp_path/"SCORER_OUTPUT.txt").write_bytes(b"ok\n")
    put("SCORER_RECEIPT.json", supervisor)
    lane = dict(case_dispositions={case["case_token"]: dict(family_id=case["part_key"]) for case in cases},
        families={case["part_key"]: dict(invented=True) for case in cases})
    original = dict(schema="GOLD11_MESH_GEOMETRIC_DEVELOPMENT_COMPARISON_V1",
        status="COMPUTED_DEVELOPMENT_DIAGNOSTIC", case_count=11, family_count=6, reference_line_count=71,
        control_identity="NEW_COMMON_WRAPPER_CONTROL_NOT_HISTORICAL_INCUMBENT",
        control=deepcopy(lane), challenger=deepcopy(lane), production_use_allowed=False,
        conformance="INDETERMINATE", observed_bend_count=None)
    score_sha = put("SCORER_RESULT.json", original)
    put("SCORER_ARTIFACT_RETENTION.json", dict(status="COMPLETE_RESULT_BYTES_RETAINED",
        artifacts=[dict(path="SCORER_RESULT.json", sha256=score_sha,
            bytes=(tmp_path/"SCORER_RESULT.json").stat().st_size)]))
    selected_sha = put("SELECTED_REFERENCES.json", dict(invented_reference_bytes_only=True))
    monkeypatch.setattr(runner, "SELECTED_SHA", selected_sha)
    score = dict(original, experiment_schema="GOLD11_MESH_DEVELOPMENT_V3",
        control_identity="FRESH_V8_WORKFLOW_NATIVE_V1_NOT_HISTORICAL_INCUMBENT",
        bundle_sha256=runner.digest(runner.encoded(bundle)), reference_snapshot_sha256=selected_sha,
        retained_control_policy=runner.RETAINED_CONTROL_POLICY, arm_definitions=runner.ARM_DEFINITIONS,
        comparator=runner.COMPARATOR, certificate_caps=runner.CERTIFICATE_CAPS,
        scorer_result_sha256=score_sha)
    put("SCORED.json", score)
    return SimpleNamespace(bundle=bundle, supervisor=supervisor, put=put, get=get,
        prefix="cases/invented-0/control/", original=original, score=score)


def test_complete_owned_22_case_manifest_is_hash_bound_not_scoring_or_inference(runner, tmp_path, monkeypatch):
    fixture = _artifact_fixture(runner, tmp_path, monkeypatch)
    evidence = {}
    runner.verify_parent_artifacts(fixture.bundle, fixture.supervisor, evidence)
    assert evidence["complete"] is True and evidence["case_count"] == 11 and evidence["arm_count"] == 2
    assert len(evidence["files"]) == 119 and len(evidence["logs"]) == 24
    for path, row in evidence["files"].items():
        blob = (tmp_path/path).read_bytes()
        assert row == dict(sha256=runner.digest(blob), bytes=len(blob))
    assert evidence["scored_report_sha256"] == runner.digest((tmp_path/"SCORED.json").read_bytes())
    assert len(runner.encoded(evidence)) < 256*1024


@pytest.mark.parametrize("fault", ("case_missing", "case_nonterminal", "case_input_hash", "case_start_arm",
    "result_changed", "result_absent", "result_sha", "unlisted_partial", "failed_usable", "log_missing",
    "log_short", "log_overflow", "stored_unknown", "barrier_count", "barrier_outcomes", "barrier_result", "scorer_failed",
    "scorer_result_changed", "score_metric_changed", "score_bundle", "selected_changed"))
def test_owned_manifest_faults_preserve_partial_evidence_and_refuse_whole_release(runner, tmp_path, monkeypatch, fault):
    f = _artifact_fixture(runner, tmp_path, monkeypatch)
    cp = f.prefix+"COMPLETED.json"
    receipt = f.get(cp)
    if fault == "case_missing": (tmp_path/cp).unlink()
    elif fault == "case_nonterminal": receipt["status"] = "RUNNING"; f.put(cp, receipt)
    elif fault == "case_input_hash": receipt["scoring_input_sha256"] = "0"*64; f.put(cp, receipt)
    elif fault == "case_start_arm":
        row = f.get(f.prefix+"STARTED.json"); row["arm"] = "challenger"; f.put(f.prefix+"STARTED.json", row)
    elif fault == "result_changed": (tmp_path/(f.prefix+"RESULT.json")).write_bytes(b"changed")
    elif fault == "result_absent": (tmp_path/(f.prefix+"RESULT.json")).unlink()
    elif fault == "result_sha": receipt["result_sha256"] = "0"*64; f.put(cp, receipt)
    elif fault == "unlisted_partial": (tmp_path/(f.prefix+"RESULT_PARTIAL.bin")).write_bytes(b"unbound")
    elif fault == "failed_usable":
        receipt["exit_code"] = 7
        receipt["scoring_input_sha256"] = f.put(f.prefix+"SCORING_INPUT.json", dict(status="PARTIAL", predictions=[]))
        f.put(cp, receipt)
    elif fault == "log_missing": (tmp_path/(f.prefix+"OUTPUT.txt")).unlink()
    elif fault == "log_short": (tmp_path/(f.prefix+"OUTPUT.txt")).write_bytes(b"ok")
    elif fault == "log_overflow": receipt["forwarded_log_bytes"] = runner.LIMITS["case_log_bytes"]+1; f.put(cp, receipt)
    elif fault == "stored_unknown":
        receipt.update(stored_log_bytes=None, budget_stop="INVENTED_LOG_STORAGE_FAILURE")
        f.put(cp, receipt)
    elif fault.startswith("barrier_"):
        row = f.get("INFERENCE_TERMINAL.json")
        if fault == "barrier_count": row["terminal_record_count"] = 21
        elif fault == "barrier_outcomes": row["outcomes_sha256"] = "0"*64
        else: row["result_bindings"]["control"]["invented-0"] = "0"*64
        f.put("INFERENCE_TERMINAL.json", row)
    elif fault == "scorer_failed":
        row = f.get("SCORER_RECEIPT.json"); row["exit_code"] = 7; f.put("SCORER_RECEIPT.json", row)
    elif fault == "scorer_result_changed": (tmp_path/"SCORER_RESULT.json").write_bytes(b"changed")
    elif fault in ("score_metric_changed", "score_bundle"):
        row = f.get("SCORED.json")
        row["observed_bend_count" if fault == "score_metric_changed" else "bundle_sha256"] = 99 if fault == "score_metric_changed" else "0"*64
        f.put("SCORED.json", row)
    else: (tmp_path/"SELECTED_REFERENCES.json").write_bytes(b"changed")
    evidence = {}
    with pytest.raises((ValueError, OSError)):
        runner.verify_parent_artifacts(f.bundle, f.supervisor, evidence)
    assert evidence["complete"] is False and evidence["files"]


@pytest.mark.parametrize("fault", ("case", "family", "family_identity"))
def test_coherently_hash_bound_but_incomplete_scoring_lane_is_still_refused(runner, tmp_path, monkeypatch, fault):
    f = _artifact_fixture(runner, tmp_path, monkeypatch)
    original = f.get("SCORER_RESULT.json")
    if fault == "case": original["control"]["case_dispositions"].pop("invented-0")
    elif fault == "family": original["control"]["families"].pop("family-0")
    else: original["control"]["case_dispositions"]["invented-0"]["family_id"] = "family-5"
    sha = f.put("SCORER_RESULT.json", original)
    f.put("SCORER_ARTIFACT_RETENTION.json", dict(status="COMPLETE_RESULT_BYTES_RETAINED", artifacts=[
        dict(path="SCORER_RESULT.json", sha256=sha, bytes=(tmp_path/"SCORER_RESULT.json").stat().st_size)]))
    score = f.get("SCORED.json"); score["control"] = original["control"]; score["scorer_result_sha256"] = sha
    f.put("SCORED.json", score)
    with pytest.raises(ValueError, match="COMPLETE_CASE_FAMILY_SCORE_LANE_REQUIRED"):
        runner.verify_parent_artifacts(f.bundle, f.supervisor, {})


def test_manifest_byte_cap_clears_complete_flag_after_otherwise_complete_validation(runner, tmp_path, monkeypatch):
    f = _artifact_fixture(runner, tmp_path, monkeypatch)
    encode = runner.encoded
    monkeypatch.setattr(runner, "encoded", lambda value: b"x"*(256*1024+1)
        if type(value) is dict and value.get("schema") == "GOLD11_OWNED_ARTIFACT_CUSTODY_V3" else encode(value))
    evidence = {}
    with pytest.raises(ValueError, match="OWNED_ARTIFACT_MANIFEST_BYTE_BUDGET"):
        runner.verify_parent_artifacts(f.bundle, f.supervisor, evidence)
    assert evidence["complete"] is False and len(evidence["files"]) == 119


def test_actual_nested_manifest_read_caps_attempted_entries_at_129(runner):
    _, _, functions = _source_nodes(SOURCE)
    nested, = [node for node in functions["verify_parent_artifacts"].body
        if isinstance(node, ast.FunctionDef) and node.name == "read"]
    evidence = dict(complete=False, files={str(i): {} for i in range(128)})
    namespace = dict(evidence=evidence, owned_read=lambda *args, **kwargs: b"invented", digest=runner.digest)
    exec(compile(ast.Module(body=[nested], type_ignores=[]), str(SOURCE), "exec"), namespace)
    with pytest.raises(ValueError, match="OWNED_ARTIFACT_MANIFEST_ENTRY_BUDGET"):
        namespace["read"]("next", 100)
    assert len(evidence["files"]) == 129 and evidence["complete"] is False


@pytest.mark.parametrize("fault", (None, "read", "changed_loaded", "missing_sha"))
def test_supervision_loader_compiles_exact_verified_buffer_or_refuses(runner, monkeypatch, fault):
    blob = b"marker = 947\n"
    sha = runner.digest(blob)
    monkeypatch.setattr(runner, "_VERIFIED_SUPERVISION_SHA", None if fault == "missing_sha" else sha)
    monkeypatch.setattr(runner, "_SUPERVISION_MODULE", None)
    calls = []
    def read(path, cap, expected):
        calls.append((path, cap, expected))
        assert path == runner.RUN/"source"/runner.SUPERVISION_SOURCE and cap == 2*1024**2 and expected == sha
        if fault == "read": raise ValueError("INVENTED_HASH_FAILURE")
        return blob
    monkeypatch.setattr(runner, "_bounded_read", read)
    if fault in ("missing_sha", "read"):
        with pytest.raises(ValueError): runner._real_supervision_loader()
        assert runner._SUPERVISION_MODULE is None
        assert len(calls) == int(fault == "read")
    else:
        module = runner._real_supervision_loader()
        assert module.marker == 947 and module.__spec__ is None
        assert module.__verified_source_sha256 == sha and len(calls) == 1
        if fault == "changed_loaded":
            monkeypatch.setattr(runner, "_VERIFIED_SUPERVISION_SHA", "0"*64)
            with pytest.raises(ValueError, match="LOADED_SUPERVISION_SOURCE_BINDING_CHANGED"):
                runner._real_supervision_loader()
        else: assert runner._real_supervision_loader() is module and len(calls) == 1


def _execute_fixture(runner, tmp_path, monkeypatch, fault=None):
    """Mock orchestration only; actual artifact validator is tested separately."""
    monkeypatch.setattr(runner, "RUN", tmp_path)
    state = SimpleNamespace(now=0., active=[], deadlines=[], publications={}, worker_calls=0,
        audit_calls=0, artifact_calls=0, terminal_calls=0, print_calls=0, started=False)
    bundle = dict(invented_bundle=True, source_sha256={}, runtime_metadata={})
    audit = dict(invented_audit=True)
    @contextmanager
    def deadline(seconds):
        assert 0 < seconds <= 4200
        state.deadlines.append(seconds); state.active.append(seconds)
        try: yield
        finally: state.active.pop()
    monkeypatch.setattr(runner, "deadline", deadline)
    monkeypatch.setattr(runner.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(runner.resource, "getrusage", lambda *args: SimpleNamespace(ru_maxrss=1024))
    monkeypatch.setattr(runner, "verify_bundle", lambda: bundle)
    monkeypatch.setattr(runner, "verify_pre_run_audit", lambda value: audit)
    controls = {"FROZEN.json": bundle, "PRE_RUN_AUDIT.json": audit}
    monkeypatch.setattr(runner, "owned_read", lambda name, *args: runner.encoded(controls[name]))
    publish = runner.exclusive_json
    def publication(path, value):
        assert state.active, "Every publication must be inside an active deadline"
        publish(path, value)
        state.publications[path.name] = deepcopy(value)
        if path.name == "STARTED.json": state.started = True
        if fault == "started" and path.name == "STARTED.json": raise TimeoutError("INVENTED_POST_LINK_STARTED")
        if fault == "dispatch" and path.name == "COORDINATOR_DISPATCH.json": raise TimeoutError("INVENTED_POST_LINK_DISPATCH")
        if fault == "completed" and path.name == "COMPLETED.json": raise TimeoutError("INVENTED_POST_LINK_COMPLETED")
        if fault == "final_gate" and path.name == "FINAL_GATE.json": raise TimeoutError("INVENTED_POST_LINK_FINAL_GATE")
    monkeypatch.setattr(runner, "exclusive_json", publication)
    original_token = runner.secrets.token_hex
    def token(count):
        if count == 32:
            assert state.active == [4170] and state.started
            if fault == "token": raise TimeoutError("INVENTED_TOKEN_TIMEOUT")
        return original_token(count)
    monkeypatch.setattr(runner.secrets, "token_hex", token)
    mkdir = Path.mkdir
    def make(path, *args, **kwargs):
        if path == tmp_path/"coordinator_work":
            assert state.active == [4170]
            if fault == "mkdir": raise TimeoutError("INVENTED_SETUP_MKDIR")
        return mkdir(path, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", make)
    receipt = dict(exit_code=0, budget_stop=None, direct_child_reaped=True,
        owned_group_empty_confirmed=True, cleanup_errors=[], control_close_errors=[],
        observed_foreign_group_escape_detected=False, child_pid=200)
    if fault == "group_false": receipt["owned_group_empty_confirmed"] = False
    elif fault == "group_missing": receipt.pop("owned_group_empty_confirmed")
    elif fault == "not_reaped": receipt["direct_child_reaped"] = False
    elif fault == "cleanup": receipt["cleanup_errors"] = ["INVENTED"]
    elif fault == "control_close": receipt["control_close_errors"] = ["INVENTED"]
    elif fault == "escape": receipt["observed_foreign_group_escape_detected"] = True
    elif fault == "worker_budget": receipt["budget_stop"] = "INVENTED_LIMIT"
    elif fault == "worker_exit": receipt["exit_code"] = 7
    def supervise(command, request, **kwargs):
        assert state.active == [4170]
        state.worker_calls += 1
        assert kwargs["new_session"] is True and kwargs["wall_seconds"] == 4170
        assert kwargs["rss_limit"] == 12*1024**3 and kwargs["log_limit"] == 2*1024**2
        if fault == "supervise": raise TimeoutError("INVENTED_SUPERVISION")
        return deepcopy(receipt)
    monkeypatch.setattr(runner, "supervise_process", supervise)
    def source_audit(value, evidence, *, control_bindings):
        assert state.active and value is bundle
        state.audit_calls += 1
        evidence.update(complete=True, unchanged=fault != "source_drift", controls=dict(control_bindings))
    monkeypatch.setattr(runner, "final_source_audit", source_audit)
    def terminalize(value, reason):
        state.terminal_calls += 1
        assert state.active and value is bundle
        return [dict(durable=True) for _ in range(21 if fault == "terminal_missing" else 22)]
    monkeypatch.setattr(runner, "terminalize_case_records", terminalize)
    def artifacts(value, received, evidence):
        state.artifact_calls += 1
        assert state.active and value is bundle and received == receipt
        evidence.update(complete=fault != "artifact", scored_report_sha256="a"*64,
            scorer_result_sha256="b"*64, reference_snapshot_sha256=runner.SELECTED_SHA)
        if fault == "artifact": raise ValueError("INVENTED_ARTIFACT_FAILURE")
    monkeypatch.setattr(runner, "verify_parent_artifacts", artifacts)
    def printed(*args, **kwargs):
        state.print_calls += 1
        assert state.active and "FINAL_GATE.json" in state.publications
        if fault == "print": raise TimeoutError("INVENTED_PRINT_TIMEOUT")
        if fault == "post_stdout": state.now = 4201.
    monkeypatch.setattr(runner, "print", printed, raising=False)
    return state


def test_mocked_parent_success_binds_pending_terminal_and_all_22_not_physical_release(runner, tmp_path, monkeypatch):
    state = _execute_fixture(runner, tmp_path, monkeypatch)
    assert runner.execute() == 0
    assert state.worker_calls == state.audit_calls == state.terminal_calls == state.artifact_calls == state.print_calls == 1
    assert state.deadlines[:2] == [20, 4170]
    assert "FINALIZATION_FAILED.json" not in state.publications
    completed, gate = state.publications["COMPLETED.json"], state.publications["FINAL_GATE.json"]
    assert completed["status"] == "COMPUTED_PENDING_INDEPENDENT_POST_AUDIT" and completed["failures"] == []
    assert len(completed["terminal_case_records"]) == 22 and completed["production_use_allowed"] is False
    assert gate["completed_canonical_sha256"] == runner.digest(runner.encoded(completed))
    assert gate["owned_artifact_custody_sha256"] == runner.digest(runner.encoded(completed["owned_artifact_custody"]))
    assert gate["status"] == "PENDING_INDEPENDENT_POST_AUDIT"


def test_post_link_started_timeout_vetoes_before_coordinator_or_input_access(runner, tmp_path, monkeypatch):
    state = _execute_fixture(runner, tmp_path, monkeypatch, "started")
    assert runner.execute() == 1
    assert state.worker_calls == state.audit_calls == state.terminal_calls == state.artifact_calls == 0
    assert set(state.publications) == {"STARTED.json", "FINALIZATION_FAILED.json"}
    assert state.publications["FINALIZATION_FAILED.json"]["phase"] == "STARTED_PUBLICATION"
    assert state.publications["FINALIZATION_FAILED.json"]["worker_dispatched"] is False
    assert state.deadlines == [20, 1]


@pytest.mark.parametrize("fault", ("token", "mkdir", "dispatch", "supervise", "group_false", "group_missing",
    "not_reaped", "cleanup", "control_close", "escape", "worker_budget", "worker_exit", "source_drift",
    "terminal_missing", "artifact", "completed", "final_gate", "print", "post_stdout"))
def test_mocked_parent_failures_are_terminal_vetoes_not_partial_campaign_success(runner, tmp_path, monkeypatch, fault):
    state = _execute_fixture(runner, tmp_path, monkeypatch, fault)
    assert runner.execute() == 1
    assert state.worker_calls == int(fault not in ("token", "mkdir", "dispatch"))
    assert state.audit_calls == state.terminal_calls == 1
    assert state.publications["FINALIZATION_FAILED.json"]["status"] == "TERMINAL_VETO_NO_RELEASE"
    assert state.deadlines[-1] == 1
    assert len(state.publications["COMPLETED.json"]["terminal_case_records"]) == (21 if fault == "terminal_missing" else 22)
    if fault in ("final_gate", "print", "post_stdout"):
        assert "FINAL_GATE.json" in state.publications  # Additive late veto dominates.
    else: assert "FINAL_GATE.json" not in state.publications
    if fault in ("token", "mkdir", "dispatch", "supervise", "group_false", "group_missing", "not_reaped",
                 "cleanup", "control_close", "escape", "worker_budget", "worker_exit", "terminal_missing"):
        assert state.artifact_calls == 0


@pytest.mark.parametrize("marker", ("STARTED.json", "FINALIZATION_FAILED.json", "INFERENCE_TERMINAL.json",
    "COORDINATOR_OUTPUT.txt", "cases", "leaf_dispatch"))
def test_any_spent_marker_even_dangling_symlink_blocks_before_dispatch_without_veto(runner, tmp_path, monkeypatch, marker):
    state = _execute_fixture(runner, tmp_path, monkeypatch)
    (tmp_path/marker).symlink_to(tmp_path/"unread_missing_target")
    with pytest.raises(ValueError, match="STARTED_OR_TERMINAL_ATTEMPT_NO_RETRY"):
        runner.execute()
    assert state.worker_calls == state.audit_calls == state.terminal_calls == 0
    assert state.publications == {} and (tmp_path/marker).is_symlink()


def test_scorer_synchronous_supervision_is_inside_exact_local_60_second_deadline():
    _, _, functions = _source_nodes(SOURCE)
    scorer = functions["score_after_terminal"]
    call, = [node for node in ast.walk(scorer) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "supervise_process"]
    contexts = [node for node in ast.walk(scorer) if isinstance(node, ast.With)
        and any(descendant is call for descendant in ast.walk(node))]
    assert any(any(ast.unparse(item.context_expr) == "deadline(LIMITS['scorer_wall_seconds'])"
        for item in node.items) for node in contexts)
    assert ast.unparse(next(item.value for item in call.keywords if item.arg == "wall_seconds")) == "LIMITS['scorer_wall_seconds']"


@pytest.mark.parametrize("fault", (None, "source", "timeout"))
def test_final_source_audit_retains_attempted_hashes_and_never_clears_prior_failure(runner, monkeypatch, fault):
    names = ("scripts/invented_a.py", "scripts/invented_b.py")
    monkeypatch.setattr(runner, "SOURCES", names)
    runtime, sha = dict(invented=True), runner.digest(b"inert")
    monkeypatch.setattr(runner, "runtime_metadata", lambda: runtime)
    calls = []
    def read(path, cap):
        calls.append(path)
        if len(calls) == 2 and fault:
            raise TimeoutError("INVENTED") if fault == "timeout" else ValueError("INVENTED")
        return b"inert"
    monkeypatch.setattr(runner, "_bounded_read", read)
    monkeypatch.setattr(runner, "owned_read", lambda *args: pytest.fail("No controls declared by this mock"))
    evidence, bundle = {}, dict(source_sha256={name: sha for name in names}, runtime_metadata=runtime)
    if fault == "timeout":
        with pytest.raises(TimeoutError): runner.final_source_audit(bundle, evidence, control_bindings={})
        assert len(calls) == 2 and evidence["complete"] is evidence["unchanged"] is False
    else:
        runner.final_source_audit(bundle, evidence, control_bindings={})
        assert len(calls) == 4 and evidence["complete"] is True
        assert evidence["unchanged"] is (fault is None)
        assert bool(evidence["errors"]) is (fault == "source")
    assert evidence["current"][names[0]] == sha


def test_nested_funnel_preserves_zero_vs_unknown_and_never_decodes_geometry(runner):
    initialization = dict(status="UNAVAILABLE", selected_policy_evaluation_complete=False,
        numerical_consumer_complete=False, box_screening_complete=False,
        statistics=dict(query_rows=0, certificate_pairs=0, completed_box_screen_rows=None,
            work_units=float("nan"), planned_box_screen_rows={"not": "scalar"}),
        patch_evidence=dict(scan=dict(format="INVENTED_ENCODED", payload="NEVER_DECODE")),
        hypothesis_evidence=dict(status="UNAVAILABLE", statistics=dict(input_scan_planes=27,
            proposed_hypotheses=0, scan_triplets_evaluated=48)))
    payload = dict(phase="WHOLE_PART_REGISTRATION", registration=dict(status="UNAVAILABLE",
        evidence=dict(initialization=initialization, candidate_count=0)))
    result = runner.workflow_stage_diagnostics(payload, "challenger")
    work = result["stages"]["initialization_work"]
    assert work["values"]["query_rows"] == work["values"]["certificate_pairs"] == 0
    assert work["values"]["completed_box_screen_rows"] is None
    assert set(work["omitted_fields"]) == {"work_units", "planned_box_screen_rows"}
    assert result["stages"]["plane_search_work"]["values"]["input_scan_planes"] == 27
    assert "NEVER_DECODE" not in runner.encoded(result).decode()
    assert result["full_internal_ledger_recoverable_after_worker_kill"] is False
