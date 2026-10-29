"""Separate opt-in runtime-import smoke; never inspection or physical input.

The pure tests use invented bytes and mocked process boundaries. The sole native
test is separately released by the parent after its source/closure audit.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SOURCE = "scripts/run_gold11_mesh_development_v2.py"
SOURCE = ROOT/RUNNER_SOURCE
SOURCE_FILE_CAP = 2*1024**2
SOURCE_TOTAL_CAP = 48*1024**2
OUTPUT_CAP = 1024**2
OPT_IN = "SHERMAN_RUN_GUARDED_RUNTIME_SMOKE_V2"
ARTIFACT_DIRECTORY = "SHERMAN_GOLD11_GUARD_SMOKE_V2_ARTIFACT_DIRECTORY"
SOURCE_BINDINGS = {
    RUNNER_SOURCE: "94d51b90faef0d88b15ed2d3d84d5b7d36ba94bfb6f63f90b9cc555a35e26422",
    "backend/cad_guided_mesh_offline_v1.py": "55243b2a557e77a9c50e27a7fdef9257d60ed2468bc23c1788c391d7a836bdba",
    "backend/cad_guided_mesh_input_v1.py": "c4d936e2f60837d7c5221c9a46b64abd68c4ccee01ea72e9fd3c4050bde47917",
    "backend/cad_guided_global_registration_v1.py": "7700541063c4bfb40c72fb66baf2a78fea1c631e5e3f93eabbb51edfdb59eff7",
    "backend/cad_guided_cad_sampling_v1.py": "f1d32159a50b455d6b9a0b5f646586f0d79e7f9c26e8fffbe67f97c5b74e181a",
    "backend/cad_guided_native_features_v1.py": "3804a583f5642df41cd3eb02be36b3f5024347b6514861d404389e85c1ccf98b",
    "backend/cad_guided_native_features.py": "f8c681027af231546974633a02f12038d522c604bec14196f2ae5d49b439c99f",
    "backend/cad_guided_local_measurement_v1.py": "5b07e8c14d3eededfb7cbdb2cb628e6d631a30662280fc2d39b2528d862b541b",
    "backend/cad_guided_measurement.py": "d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb",
    "backend/cad_guided_instances.py": "d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda",
    "backend/cad_guided_mesh_offline_v2.py": "fa30bc1ed575f9174414e3737eaaffecc6eafdcbcd9e3b1d68475f1af7ea8cbf",
    "backend/cad_guided_mesh_offline_v8.py": "afdfc7f859eb5fdbf0fa7d796cc06a4180d40c02f6b0e590362f83e3f339e7ec",
    "backend/cad_guided_cad_surface_v2.py": "b7364234a716f283f46246a452451f47c8efb07c18b2d37375c6fda0636da0d1",
    "backend/cad_guided_global_registration_v2.py": "5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3",
    "backend/cad_guided_triangle_surface_v2.py": "7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1",
    "backend/cad_guided_mesh_topology_v4.py": "4d061b56a39aee6b7d9e8c04eb7af88b0573a7916d3d679c610c4aca81ee0734",
    "backend/cad_guided_plane_initialization_v4.py": "a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724",
    "backend/cad_guided_plane_initialization_v8.py": "cd225b4642816f76cfd4b8bd725e3b8ffa5d9f6177ddbfa9b03fb6a22128c28b",
    "backend/cad_guided_mesh_plane_proposals_v4.py": "55356d900ce933b4dd4ff9fc8d055fdb46038422b69b6e05f7f9509d73f5ed48",
    "backend/cad_guided_plane_hypotheses_v4.py": "7e69fce8bf79216342071aacfbbb7c0d73972c78334cc81d7f0fe15c831f1370",
    "backend/cad_guided_noisy_mesh_planes_v5.py": "9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75",
    "backend/cad_guided_noisy_mesh_planes_v7.py": "b0822289cf3abbde9c7cbf263b45c8d8ef4c4f45d7913e452c47a46ed881449f",
    "backend/cad_guided_plane_search_v6.py": "aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676",
    "backend/cad_guided_report_envelope_v6.py": "c7cc45a362c5972aff78265b2ce63fa4d41ec023b662640d7d197c23003ba28e",
}


PROGRAM = '''import hashlib, importlib.util, json, pathlib, resource, sys, tempfile
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
source, snapshot, cache = map(pathlib.Path, sys.argv[1:4])
with source.open("rb") as stream:
    source_blob = stream.read(2*1024**2+1)
assert 0 < len(source_blob) <= 2*1024**2
assert hashlib.sha256(source_blob).hexdigest() == sys.argv[4]
spec = importlib.util.spec_from_file_location("guarded_v2_runner", source)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.initialize_worker(snapshot, cache, "backend", 10)
import numpy
import scipy
from OCP.gp import gp_Pnt
from backend.cad_guided_mesh_offline_v8 import MeshWorkflowPolicyV8, inspect_cad_guided_mesh_v8
from backend.cad_guided_noisy_mesh_planes_v5 import NoisyMeshPlanePolicyV5, NoisyMeshPlaneBudgetsV5
from backend.cad_guided_mesh_plane_proposals_v4 import CoplanarPatchPolicyV4, CoplanarPatchBudgetsV4
from backend.cad_guided_plane_search_v6 import PlaneSearchPolicyV6, PlaneSearchBudgetsV6
from backend.cad_guided_plane_initialization_v8 import PlaneInitializationBudgetsV8
from backend.cad_guided_triangle_surface_v2 import TriangleSurfaceBudgets
from backend.cad_guided_global_registration_v2 import SurfaceRegistrationBudgetsV2
assert callable(inspect_cad_guided_mesh_v8)
assert runner.CONFIG["position_error_bound_mm"] == .05
assert runner.CONFIG["initialization_budgets"]["max_prefilter_unsigned_assignments"] == 1024
noisy_policy = dict(runner.CONFIG["noisy_policy"])
noisy_policy["tile_triangles"] = tuple(noisy_policy["tile_triangles"])
workflow = MeshWorkflowPolicyV8(**runner.CONFIG["workflow"], native_adapter_version="v1")
assert workflow.native_adapter_version == "v1"
constructors = {
    "noisy_policy": NoisyMeshPlanePolicyV5(**noisy_policy),
    "noisy_budgets": NoisyMeshPlaneBudgetsV5(**runner.CONFIG["noisy_budgets"]),
    "cad_patch_policy": CoplanarPatchPolicyV4(**runner.CONFIG["cad_patch_policy"]),
    "cad_patch_budgets": CoplanarPatchBudgetsV4(**runner.CONFIG["cad_patch_budgets"]),
    "search_policy": PlaneSearchPolicyV6(**runner.CONFIG["search_policy"]),
    "search_budgets": PlaneSearchBudgetsV6(**runner.CONFIG["search_budgets"]),
    "initialization_budgets": PlaneInitializationBudgetsV8(**runner.CONFIG["initialization_budgets"]),
    "surface_budgets": TriangleSurfaceBudgets(**runner.CONFIG["surface_budgets"]),
    "registration_budgets": SurfaceRegistrationBudgetsV2(**runner.CONFIG["registration_budgets"]),
}
for name, value in constructors.items():
    expected = noisy_policy if name == "noisy_policy" else runner.CONFIG[name]
    for key, expected_value in expected.items():
        assert getattr(value, key) == expected_value
point = gp_Pnt(1., 2., 3.)
assert (point.X(), point.Y(), point.Z()) == (1., 2., 3.)
with tempfile.TemporaryDirectory(dir=cache, prefix="owned-v2-smoke-") as directory:
    path = pathlib.Path(directory)/"invented.txt"
    with path.open("xb") as stream:
        stream.write(b"invented guarded V2 runtime smoke")
    with path.open("rb") as stream:
        assert stream.read(128) == b"invented guarded V2 runtime smoke"
assert not pathlib.Path(directory).exists()
print(json.dumps(dict(status="GUARDED_V2_RUNTIME_IMPORT_SMOKE_PASS", constructor_count=10,
    inference_executed=False, owned_temp_removed=True)), flush=True)
'''


def _enabled(environment):
    return environment.get(OPT_IN) == "1"


def _new_artifact_directory(environment):
    value = environment.get(ARTIFACT_DIRECTORY)
    if (type(value) is not str or not 0 < len(value) <= 4096
            or not Path(value).is_absolute() or str(Path(value)) != value
            or ".." in Path(value).parts):
        raise ValueError("SMOKE_FRESH_ABSOLUTE_ARTIFACT_DIRECTORY_REQUIRED")
    path = Path(value)
    # Root supplies an owned retained parent. Never reuse or follow an existing
    # directory/symlink, create parents, or fall back to pytest's disposable tmp.
    path.mkdir(mode=0o700, exist_ok=False)
    return path


@pytest.fixture
def runner():
    with SOURCE.open("rb") as stream:
        blob = stream.read(SOURCE_FILE_CAP+1)
    assert 0 < len(blob) <= SOURCE_FILE_CAP
    assert hashlib.sha256(blob).hexdigest() == SOURCE_BINDINGS[RUNNER_SOURCE]
    spec = importlib.util.spec_from_file_location("guard_smoke_v2_inert_runner", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_sources(runner, snapshot, bindings):
    # The literal scope is independent of caller bindings; no path discovery.
    names = tuple(SOURCE_BINDINGS)
    if (type(bindings) is not dict or set(bindings) != set(names)
            or tuple(runner.leaf_sources("INFERENCE")) != names
            or any(type(value) is not str or len(value) != 64
                   or any(c not in "0123456789abcdef" for c in value) for value in bindings.values())):
        raise ValueError("SMOKE_SOURCE_DENOMINATOR_OR_HASH_SCHEMA")
    snapshot.mkdir(exist_ok=False)
    blobs, total = {}, 0
    for name in names:
        blob = runner.bounded_read(runner.ROOT/name, SOURCE_FILE_CAP, bindings[name])
        if type(blob) is not bytes or not 0 < len(blob) <= SOURCE_FILE_CAP:
            raise ValueError("SMOKE_SOURCE_BYTE_CAP")
        if hashlib.sha256(blob).hexdigest() != bindings[name]:
            raise ValueError("SMOKE_SOURCE_HASH_MISMATCH")
        total += len(blob)
        if total > SOURCE_TOTAL_CAP:
            raise ValueError("SMOKE_SOURCE_TOTAL_CAP")
        blobs[name] = blob
    # No successful prefix is published if any source/hash admission fails.
    for name, blob in blobs.items():
        destination = snapshot/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(blob)
    return total


def _launch_smoke(runner, snapshot, cache, log):
    command = [str(runner.ROOT/".venv/bin/python"), "-I", "-S", "-B", "-u", "-c", PROGRAM,
               str(snapshot/RUNNER_SOURCE), str(snapshot), str(cache), SOURCE_BINDINGS[RUNNER_SOURCE]]
    receipt = runner.supervise_process(command, {}, cache=cache, log_path=log,
        wall_seconds=20, rss_limit=512*1024**2, log_limit=OUTPUT_CAP,
        new_session=False, parent_rss_limit=1024**3)
    # Retain bounded metadata before interpreting child success. The full log
    # itself already resides in the independently retained artifact directory.
    output = None
    try:
        output = runner.bounded_read(log, OUTPUT_CAP)
        log_metadata = dict(status="AVAILABLE", bytes=len(output),
                            sha256=hashlib.sha256(output).hexdigest())
    except Exception as error:
        log_metadata = dict(status="UNAVAILABLE", bytes=None, sha256=None,
                            error_type=type(error).__name__)
    metadata = json.dumps(dict(status="GUARDED_V2_SMOKE_CHILD_TERMINAL", receipt=receipt,
                               child_log=log_metadata), sort_keys=True, allow_nan=False)
    assert len(metadata.encode()) <= 4096
    print(metadata, flush=True)
    assert receipt["exit_code"] == 0 and receipt["budget_stop"] is None, receipt
    assert output is not None, log_metadata
    final = json.loads(output.splitlines()[-1])
    assert final == dict(status="GUARDED_V2_RUNTIME_IMPORT_SMOKE_PASS", constructor_count=10,
                         inference_executed=False, owned_temp_removed=True)
    return receipt, hashlib.sha256(output).hexdigest()


@pytest.fixture
def invented_sources(tmp_path):
    blobs = {name: ("invented source "+name).encode() for name in SOURCE_BINDINGS}
    bindings = {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}
    reads = []
    def read(path, cap, expected):
        name = str(path.relative_to(tmp_path))
        assert name in blobs and cap == SOURCE_FILE_CAP and expected == bindings[name]
        reads.append(name)
        return blobs[name]
    fake = SimpleNamespace(ROOT=tmp_path, leaf_sources=lambda role: tuple(SOURCE_BINDINGS), bounded_read=read)
    return SimpleNamespace(runner=fake, blobs=blobs, bindings=bindings, reads=reads)


def test_exact_named_runtime_closure_matches_runner_without_identity_sources(runner):
    assert len(SOURCE_BINDINGS) == 24 and tuple(SOURCE_BINDINGS) == runner.leaf_sources("INFERENCE")
    assert set(SOURCE_BINDINGS) == {RUNNER_SOURCE} | {name for name in runner.SOURCES if name.startswith("backend/")}
    assert not any("review" in name or "scoring" in name or "protocol" in name for name in SOURCE_BINDINGS)
    assert all(len(sha) == 64 and set(sha) <= set("0123456789abcdef") for sha in SOURCE_BINDINGS.values())


def test_copy_admits_full_named_hash_bound_snapshot_only(invented_sources, tmp_path):
    data, snapshot = invented_sources, tmp_path/"snapshot"
    assert _copy_sources(data.runner, snapshot, data.bindings) == sum(map(len, data.blobs.values()))
    assert data.reads == list(SOURCE_BINDINGS)
    for name, blob in data.blobs.items():
        assert (snapshot/name).read_bytes() == blob
    assert sorted(str(path.relative_to(snapshot)) for path in snapshot.rglob("*.py")) == sorted(SOURCE_BINDINGS)


@pytest.mark.parametrize("fault", ("missing", "extra", "nonhex"))
def test_copy_refuses_wrong_scope_before_any_read(invented_sources, tmp_path, fault):
    data, bindings = invented_sources, dict(invented_sources.bindings)
    if fault == "missing":
        bindings.pop(RUNNER_SOURCE)
    elif fault == "extra":
        bindings["../outside.py"] = "a"*64
    else:
        bindings[RUNNER_SOURCE] = "g"*64
    with pytest.raises(ValueError, match="SMOKE_SOURCE_DENOMINATOR_OR_HASH_SCHEMA"):
        _copy_sources(data.runner, tmp_path/"snapshot", bindings)
    assert not data.reads and not (tmp_path/"snapshot").exists()


@pytest.mark.parametrize("fault", ("first_hash", "last_hash", "last_bytes"))
def test_copy_late_failures_publish_no_successful_prefix(invented_sources, tmp_path, fault):
    data = invented_sources
    name = next(iter(SOURCE_BINDINGS)) if fault == "first_hash" else tuple(SOURCE_BINDINGS)[-1]
    data.blobs[name] = b"wrong digest" if fault != "last_bytes" else bytearray(b"not exact bytes")
    with pytest.raises(ValueError, match="SMOKE_SOURCE_HASH_MISMATCH|SMOKE_SOURCE_BYTE_CAP"):
        _copy_sources(data.runner, tmp_path/"snapshot", data.bindings)
    assert not list((tmp_path/"snapshot").rglob("*.py"))


def test_copy_never_reuses_snapshot(invented_sources, tmp_path):
    (tmp_path/"snapshot").mkdir()
    with pytest.raises(FileExistsError):
        _copy_sources(invented_sources.runner, tmp_path/"snapshot", invented_sources.bindings)
    assert not invented_sources.reads


@pytest.mark.parametrize("extra", (0, -1))
def test_copy_total_byte_boundary_is_exact(invented_sources, tmp_path, monkeypatch, extra):
    data = invented_sources
    total = sum(map(len, data.blobs.values()))
    monkeypatch.setitem(globals(), "SOURCE_TOTAL_CAP", total+extra)
    if extra == 0:
        assert _copy_sources(data.runner, tmp_path/"snapshot", data.bindings) == total
    else:
        with pytest.raises(ValueError, match="SMOKE_SOURCE_TOTAL_CAP"):
            _copy_sources(data.runner, tmp_path/"snapshot", data.bindings)
        assert not list((tmp_path/"snapshot").rglob("*.py"))


@pytest.mark.parametrize("value,expected", ((None, False), ("", False), ("0", False), ("1", True), ("true", False)))
def test_only_new_explicit_optin_enables_actual_smoke(value, expected):
    environment = {"SHERMAN_RUN_GUARDED_RUNTIME_SMOKE": "1"}
    if value is not None:
        environment[OPT_IN] = value
    assert _enabled(environment) is expected


@pytest.mark.parametrize("value", (None, "", "relative", "/owned/../other", "/owned//artifact"))
def test_artifact_directory_requires_explicit_normal_absolute_path(value, monkeypatch):
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("Invalid artifact path reached filesystem mutation"))
    with pytest.raises(ValueError, match="SMOKE_FRESH_ABSOLUTE_ARTIFACT_DIRECTORY_REQUIRED"):
        _new_artifact_directory({ARTIFACT_DIRECTORY: value})


def test_artifact_directory_is_fresh_owned_and_never_reused(tmp_path):
    target = tmp_path/"ARTIFACTS"
    environment = {ARTIFACT_DIRECTORY: str(target)}
    assert _new_artifact_directory(environment) == target
    assert target.is_dir() and target.stat().st_mode & 0o777 == 0o700
    sentinel = target/"SMOKE_OUTPUT.txt"
    sentinel.write_bytes(b"invented retained failure")
    with pytest.raises(FileExistsError):
        _new_artifact_directory(environment)
    assert sentinel.read_bytes() == b"invented retained failure"


def test_artifact_directory_does_not_follow_existing_symlink(tmp_path):
    owned = tmp_path/"owned"
    owned.mkdir()
    target = tmp_path/"ARTIFACTS"
    target.symlink_to(owned, target_is_directory=True)
    with pytest.raises(FileExistsError):
        _new_artifact_directory({ARTIFACT_DIRECTORY: str(target)})
    assert not list(owned.iterdir())


def test_child_program_guards_before_native_imports_and_never_invokes_inference():
    tree = ast.parse(PROGRAM)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    named = lambda call: call.func.id if isinstance(call.func, ast.Name) else call.func.attr if isinstance(call.func, ast.Attribute) else ""
    guards = [node for node in calls if named(node) == "initialize_worker"]
    assert len(guards) == 1 and [ast.literal_eval(node) for node in guards[0].args[2:]] == ["backend", 10]
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    roots = lambda node: {item.name.split(".")[0] for item in node.names} if isinstance(node, ast.Import) else {node.module.split(".")[0]}
    assert all(node.lineno > guards[0].lineno for node in imports if roots(node) & {"numpy", "scipy", "OCP", "backend"})
    forbidden = {"freeze", "execute", "worker", "authorize_leaf", "supervise_case", "scorer_worker",
                 "selected_references", "inspect_cad_guided_mesh_v8", "read_mesh", "read_step", "setsid", "setpgid"}
    assert not ({named(node) for node in calls} & forbidden)
    assert not any(token in PROGRAM for token in ("scan.ply", "cad.step", "CATALOG", "STARTED.json", "FROZEN.json"))
    assert PROGRAM.index("resource.setrlimit") < PROGRAM.index("exec_module") < PROGRAM.index("initialize_worker")
    assert 'resource.setrlimit(resource.RLIMIT_CPU, (10, 10))' in PROGRAM
    assert 'TemporaryDirectory(dir=cache' in PROGRAM and 'assert not pathlib.Path(directory).exists()' in PROGRAM


def test_program_constructor_set_matches_exact_v8_and_nine_policy_types():
    tree = ast.parse(PROGRAM)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "constructors" for target in node.targets))
    assert isinstance(assignment.value, ast.Dict)
    pairs = {ast.literal_eval(key): value.func.id for key, value in zip(assignment.value.keys, assignment.value.values)}
    assert pairs == dict(noisy_policy="NoisyMeshPlanePolicyV5", noisy_budgets="NoisyMeshPlaneBudgetsV5",
        cad_patch_policy="CoplanarPatchPolicyV4", cad_patch_budgets="CoplanarPatchBudgetsV4",
        search_policy="PlaneSearchPolicyV6", search_budgets="PlaneSearchBudgetsV6",
        initialization_budgets="PlaneInitializationBudgetsV8", surface_budgets="TriangleSurfaceBudgets",
        registration_budgets="SurfaceRegistrationBudgetsV2")
    assert 'MeshWorkflowPolicyV8(**runner.CONFIG["workflow"], native_adapter_version="v1")' in PROGRAM


def test_runner_initialization_installs_cpu_namespace_and_guard_without_application_bootstrap():
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "initialize_worker")
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    names = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id for node in calls}
    assert names == {"setrlimit", "insert", "runtime_roots", "install_namespace", "addaudithook", "worker_audit_hook"}
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(function))


def test_launch_uses_same_group_and_fixed_small_budgets_without_process_execution(tmp_path):
    calls = []
    receipt = dict(exit_code=0, budget_stop=None)
    payload = dict(status="GUARDED_V2_RUNTIME_IMPORT_SMOKE_PASS", constructor_count=10,
                   inference_executed=False, owned_temp_removed=True)
    def supervise(command, request, **kwargs):
        calls.append((command, request, kwargs))
        return receipt
    log = tmp_path/"SMOKE_OUTPUT.txt"
    output = json.dumps(payload).encode()+b"\n"
    def read(path, cap):
        assert path == log and cap == OUTPUT_CAP
        return output
    fake = SimpleNamespace(ROOT=tmp_path, supervise_process=supervise, bounded_read=read)
    assert _launch_smoke(fake, tmp_path/"snapshot", tmp_path/"cache", log) == (receipt, hashlib.sha256(output).hexdigest())
    command, request, kwargs = calls[0]
    assert len(calls) == 1 and request == {}
    assert command == [str(tmp_path/".venv/bin/python"), "-I", "-S", "-B", "-u", "-c", PROGRAM,
                       str(tmp_path/"snapshot"/RUNNER_SOURCE), str(tmp_path/"snapshot"), str(tmp_path/"cache"), SOURCE_BINDINGS[RUNNER_SOURCE]]
    assert kwargs == dict(cache=tmp_path/"cache", log_path=log, wall_seconds=20,
        rss_limit=512*1024**2, log_limit=1024**2, new_session=False, parent_rss_limit=1024**3)


@pytest.mark.parametrize("receipt", (dict(exit_code=1, budget_stop=None), dict(exit_code=-9, budget_stop="WALL_LIMIT")))
def test_failed_child_retains_receipt_and_log_metadata_before_refusal(tmp_path, receipt, capsys):
    output, reads = b"invented child exception, not a success record\n", []
    def read(path, cap):
        reads.append((path, cap))
        return output
    fake = SimpleNamespace(ROOT=tmp_path, supervise_process=lambda *a, **k: receipt,
                           bounded_read=read)
    with pytest.raises(AssertionError):
        _launch_smoke(fake, tmp_path/"snapshot", tmp_path/"cache", tmp_path/"log")
    assert reads == [(tmp_path/"log", OUTPUT_CAP)]
    metadata = json.loads(capsys.readouterr().out)
    assert metadata == dict(status="GUARDED_V2_SMOKE_CHILD_TERMINAL", receipt=receipt,
        child_log=dict(status="AVAILABLE", bytes=len(output), sha256=hashlib.sha256(output).hexdigest()))


@pytest.mark.parametrize("error", (FileNotFoundError, PermissionError))
def test_log_unavailable_retains_original_receipt_and_explicit_unknown(tmp_path, error, capsys):
    receipt = dict(exit_code=0, budget_stop=None)
    def read(*args):
        raise error("invented path detail must not be printed")
    fake = SimpleNamespace(ROOT=tmp_path, supervise_process=lambda *a, **k: receipt,
                           bounded_read=read)
    with pytest.raises(AssertionError):
        _launch_smoke(fake, tmp_path/"snapshot", tmp_path/"cache", tmp_path/"log")
    metadata = json.loads(capsys.readouterr().out)
    assert metadata == dict(status="GUARDED_V2_SMOKE_CHILD_TERMINAL", receipt=receipt,
        child_log=dict(status="UNAVAILABLE", bytes=None, sha256=None, error_type=error.__name__))


@pytest.mark.skipif(not _enabled(os.environ), reason="Separate root-reviewed V2 guarded runtime opt-in required")
def test_actual_v8_guarded_runtime_imports_and_owned_temp_cleanup(runner, tmp_path):
    """One small native import job, no physical bytes or inspection entrypoint."""
    artifacts = _new_artifact_directory(os.environ)
    snapshot, cache = tmp_path/"snapshot", tmp_path/"cache"
    source_bytes = _copy_sources(runner, snapshot, SOURCE_BINDINGS)
    cache.mkdir(mode=0o700)
    # The supervisor writes directly into retained storage before interpreting
    # success/failure. A failing child never depends on pytest tmp retention.
    receipt, output_sha256 = _launch_smoke(runner, snapshot, cache, artifacts/"SMOKE_OUTPUT.txt")
    print(json.dumps(dict(status="GUARDED_V2_RUNTIME_SMOKE_TERMINAL", source_count=24,
        source_bytes=source_bytes, output_sha256=output_sha256, receipt=receipt)), flush=True)
