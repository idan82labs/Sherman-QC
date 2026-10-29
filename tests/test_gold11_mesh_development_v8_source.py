"""Focused source controls for the V8 runner: identities, closure, config deltas,
frozen request shapes, an inert freeze/verify round trip and real constructor
bindings. No attempt directory, raw input, reference, native call or process is
touched: every read/publish is redirected to an in-memory store.
"""
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT/"scripts/run_gold11_mesh_development_v8.py"
V7 = ROOT/"scripts/run_gold11_mesh_development_v7.py"
V7_SHA = "e910d562e55a62de80f5e448f944c239b5bdf95ef11c6d0b6f0b83b24bfc7a52"


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


@pytest.fixture(scope="module")
def runner():
    prefix = "backend"
    saved = {name: value for name, value in sys.modules.items() if name == prefix or name.startswith(prefix+".")}
    for name in saved:
        del sys.modules[name]
    package = types.ModuleType(prefix)
    package.__path__ = [str(ROOT/"backend")]
    sys.modules[prefix] = package
    spec = importlib.util.spec_from_file_location("_v8_runner_source_test", V8)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix+"."):
                del sys.modules[name]
        sys.modules.update(saved)


def test_basis_identities_and_declared_config_deltas(runner):
    assert sha(V7.read_bytes()) == V7_SHA == runner.SOURCE_BASIS_SHA
    assert runner.SOURCE_BASIS == "scripts/run_gold11_mesh_development_v7.py"
    assert runner.SOURCE_PROFILE["implementation"] == "GOLD11_MESH_DEVELOPMENT_V8"
    assert runner.SOURCE_PROFILE["scientific_config_or_scorer_change"] is True
    assert runner.SOURCE_PROFILE["scorer_change"] is False
    assert runner.RUN == ROOT/"docs/research/gold11_mesh_development_v8_attempt"
    assert runner.LEAF_SCHEMA == "ONE_USE_PARENT_BOUND_LEAF_V8"
    assert runner.RUNNER_SOURCE == "scripts/run_gold11_mesh_development_v8.py"
    assert runner.CONFIG["registration"]["max_inlier_plane_p95_mm"] == 1.
    assert runner.CONFIG["surface_budgets"]["max_ties_per_query"] == 256
    assert runner.CONTEXT_V4_REQUEST["budgets"]["max_extrema_coordinate_visits"] == 512_000_000
    assert runner.CONTEXT_V4_REQUEST["mode"] == "v4"
    assert set(runner.CONFIG_DELTA_FROM_V7) == {"registration_max_inlier_plane_p95_mm",
        "surface_max_ties_per_query", "context_max_extrema_coordinate_visits",
        "workflow_max_measurement_contexts"}
    assert runner.SCORING_TRANSPORT["wrapper_schema"] == "GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V3"
    assert runner.RUNNER_SOURCE in runner.STAGE_SAMPLE_SOURCE_PATHS
    assert [name for name in runner.STAGE_SAMPLE_SOURCE_PATHS if "run_gold11_" in name] == [runner.RUNNER_SOURCE]
    assert runner.POSE_ADMISSION_DECLARATION["max_inlier_plane_p95_mm"] == runner.CONFIG["registration"]["max_inlier_plane_p95_mm"]
    assert runner.SCORING_SOURCES[-1] == "scripts/gold11_schedule_projection_v3.py"
    assert "scripts/gold11_schedule_projection_v2.py" not in runner.SCORING_SOURCES
    for arm in runner.ARMS:
        definition = runner.ARM_DEFINITIONS[arm]
        assert definition["implementation"] == "CAD_GUIDED_MESH_OFFLINE_V15_CONTEXT_V4"
        assert definition["initialization_implementation"] == "CAD_GUIDED_PLANE_INITIALIZATION_V14_CONTEXT_V4"
        assert definition["two_plane_context_mode"] == "v4"


def test_unchanged_v7_values_outside_the_declared_deltas(runner):
    tree = ast.parse(V7.read_bytes())
    namespace = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("CONFIG", "LIMITS", "CERTIFICATE_CAPS", "SEEDED_PLANE_BUDGETS",
                                           "COMPOSITE_PLANE_BUDGETS", "TWO_PLANE_DECLARATION", "SCHEDULE_V2_BUDGETS"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<v7-constants>", "exec"),
                 dict(math=importlib.import_module("math")), namespace)
    v7 = namespace["CONFIG"]
    v8 = json.loads(json.dumps(runner.CONFIG))
    assert v8["registration"].pop("max_inlier_plane_p95_mm") == 1. and v7["registration"].pop("max_inlier_plane_p95_mm") == .5
    assert v8["surface_budgets"].pop("max_ties_per_query") == 256 and v7["surface_budgets"].pop("max_ties_per_query") == 32
    assert v8["workflow"].pop("max_measurement_contexts") == 8 and "max_measurement_contexts" not in v7["workflow"]
    assert wire(v8) == wire(json.loads(json.dumps(v7)))
    assert runner.LIMITS == namespace["LIMITS"]
    assert runner.CERTIFICATE_CAPS == namespace["CERTIFICATE_CAPS"]
    assert runner.SEEDED_PLANE_BUDGETS == namespace["SEEDED_PLANE_BUDGETS"]
    assert runner.COMPOSITE_PLANE_BUDGETS == namespace["COMPOSITE_PLANE_BUDGETS"]
    assert runner.SCHEDULE_V2_BUDGETS == namespace["SCHEDULE_V2_BUDGETS"]
    assert runner.TWO_PLANE_DECLARATION["schedule_budgets"] == namespace["TWO_PLANE_DECLARATION"]["schedule_budgets"]


def test_declared_sources_exist_and_cover_the_engine_import_closure(runner):
    missing = [name for name in runner.SOURCES if not (ROOT/name).is_file()]
    assert missing == []
    assert len(set(runner.SOURCES)) == len(runner.SOURCES)
    for name in ("backend/cad_guided_raw_extrema_index_v3.py", "backend/cad_guided_two_plane_context_v4.py",
                 "backend/cad_guided_triangle_surface_v4.py", "backend/cad_guided_global_registration_v4.py",
                 "backend/cad_guided_stage_policies_v2.py", "backend/cad_guided_plane_initialization_v14.py",
                 "backend/cad_guided_mesh_offline_v15.py", "scripts/gold11_schedule_projection_v3.py",
                 "scripts/run_gold11_mesh_development_v8.py", runner.SOURCE_BASIS, runner.PROTOCOL,
                 "docs/research/2026-09-13_gold11_mesh_development_v8_delta_design.md"):
        assert name in runner.SOURCES, name
    for lazy in ("cad_guided_mesh_input_v1", "cad_guided_mesh_topology_v4", "cad_guided_native_features",
                 "cad_guided_native_features_v1", "cad_guided_cad_surface_v2", "cad_guided_support_cover_measurement_v2",
                 "cad_guided_context_surface_roles_v1", "cad_guided_composite_surface_roles_v1",
                 "cad_guided_scan_plane_preparation_v1", "cad_guided_seeded_plane_proposals_v1",
                 "cad_guided_two_plane_schedule_v2", "cad_guided_mesh_offline_v15"):
        importlib.import_module("backend."+lazy)
    closure = {name[len("backend."):]+".py" for name in sys.modules
               if name.startswith("backend.") and sys.modules[name] is not None}
    declared = {Path(name).name for name in runner.SOURCES if name.startswith("backend/")}
    assert closure <= declared, sorted(closure-declared)


def test_frozen_arm_requests_are_json_stable_and_bind_v4_identities(runner):
    for arm in runner.ARMS:
        request = runner.arm_request(arm)
        assert wire(request) == wire(json.loads(wire(request)))
        assert request["implementation"] == "CAD_GUIDED_MESH_OFFLINE_V15_CONTEXT_V4"
        assert request["registration_implementation"] == "SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V4"
        assert request["context_request"] == runner.CONTEXT_V4_REQUEST
        assert request["two_plane_context_mode"] == "v4"
    assert runner.arm_request("control")["scan_plane_mode"] == "v7"
    assert runner.arm_request("challenger")["scan_plane_mode"] == "composite_v1"
    assert set(runner.scan_plane_requests()) == {"control", "challenger"}


def test_real_constructor_bindings_for_both_arms(runner):
    for arm in runner.ARMS:
        entry, policies = runner.workflow_entry_and_policies(arm)
        assert entry.__name__ == "inspect_cad_guided_mesh_v15"
        assert type(policies["workflow_policy"]).__name__ == "MeshWorkflowPolicyV15"
        assert type(policies["two_plane_context_budgets"]).__name__ == "TwoPlaneContextBudgetsV4"
        assert type(policies["surface_budgets"]).__name__ == "TriangleSurfaceBudgetsV4"
        assert type(policies["registration_budgets"]).__name__ == "SurfaceRegistrationBudgetsV4"
        assert policies["surface_budgets"].max_ties_per_query == 256
        assert policies["registration_policy"].max_inlier_plane_p95_mm == 1.
        assert policies["workflow_policy"].max_measurement_contexts == 8
        assert policies["two_plane_context_mode"] == "v4"
        assert policies["two_plane_schedule_mode"] == ("v2" if arm == "challenger" else "v1")


def test_inert_freeze_and_verify_round_trip_bind_the_v7_basis(runner, monkeypatch):
    store, blobs = {}, {name: (ROOT/name).read_bytes() for name in runner.SOURCES}
    run = Path("/invented/v8/run")
    monkeypatch.setattr(runner, "RUN", run)
    def read(path, cap, expected=None):
        path = Path(path)
        if path == run/"FROZEN.json":
            value = store["FROZEN.json"]
        elif str(path).startswith(str(run/"source")):
            value = store["source/"+path.relative_to(run/"source").as_posix()]
        else:
            value = blobs[path.relative_to(ROOT).as_posix()]
        assert len(value) <= cap
        if expected is not None and sha(value) != expected:
            raise ValueError("INVENTED_STORE_HASH_MISMATCH")
        return value
    def publish(path, value):
        name = Path(path).relative_to(run).as_posix()
        assert name not in store
        store[name] = value
    monkeypatch.setattr(runner, "bounded_read", read)
    monkeypatch.setattr(runner, "publish_bytes", publish)
    monkeypatch.setattr(runner, "exclusive_json", lambda path, value: publish(path, wire(value)+b"\n"))
    monkeypatch.setattr(runner, "runtime_metadata", lambda: {"scope": "INVENTED_RUNTIME"})
    cases = [dict(case_token=f"INVENTED_{i:02}", part_key=f"family_{i % 6}", scan_path=f"/invented/scan_{i}.ply",
        cad_path=f"/invented/cad_{i % 6}.step", scan_sha256=sha(f"scan{i}".encode()), cad_sha256=sha(f"cad{i % 6}".encode()))
        for i in range(11)]
    monkeypatch.setattr(runner, "approved_cases_from_review", lambda _blob: cases)
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: None)
    runner.freeze()
    bundle = runner.verify_bundle()
    assert bundle["schema"] == "GOLD11_MESH_DEVELOPMENT_V8"
    assert bundle["config_delta_from_v7"] == runner.CONFIG_DELTA_FROM_V7
    assert bundle["pose_admission_declaration"] == runner.POSE_ADMISSION_DECLARATION
    assert bundle["source_sha256"][runner.SOURCE_BASIS] == V7_SHA
    assert set(bundle["source_sha256"]) == set(runner.SOURCES)
    assert len(wire(bundle)) < 256*1024
    blobs[runner.SOURCE_BASIS] += b"changed"
    with pytest.raises(ValueError, match="V8_FROZEN_V7_SOURCE_BASIS_CHANGED"):
        runner.freeze()
    changed = json.loads(store["FROZEN.json"])
    changed["source_profile"] = dict(changed["source_profile"], implementation="INVENTED")
    store["FROZEN.json"] = wire(changed)
    with pytest.raises(ValueError, match="FROZEN_PROTOCOL_CONSTANTS_CHANGED"):
        runner.verify_bundle()
