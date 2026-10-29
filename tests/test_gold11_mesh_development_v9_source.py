"""Focused source controls for the V9 runner (ported from the V8 source test): identities, closure, config deltas,
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
V9 = ROOT/"scripts/run_gold11_mesh_development_v9.py"
V8 = ROOT/"scripts/run_gold11_mesh_development_v8.py"
V8_SHA = "0e9aa2f64f3fc14e0da137687e44c8cd0dc0712f9e8bd783726df4187b1360c5"


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
    spec = importlib.util.spec_from_file_location("_v9_runner_source_test", V9)
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
    assert sha(V8.read_bytes()) == V8_SHA == runner.SOURCE_BASIS_SHA
    assert runner.SOURCE_BASIS == "scripts/run_gold11_mesh_development_v8.py"
    assert runner.SOURCE_PROFILE["implementation"] == "GOLD11_MESH_DEVELOPMENT_V9"
    assert runner.SOURCE_PROFILE["scientific_config_or_scorer_change"] is True
    assert runner.SOURCE_PROFILE["scorer_change"] is False
    assert runner.RUN == ROOT/"docs/research/gold11_mesh_development_v9_attempt"
    assert runner.LEAF_SCHEMA == "ONE_USE_PARENT_BOUND_LEAF_V9"
    assert runner.RUNNER_SOURCE == "scripts/run_gold11_mesh_development_v9.py"
    assert runner.CONFIG["measurement"]["max_fit_residual_mm"] == .03
    assert runner.CONFIG["measurement"]["plane_exclusion_mm"] == .04
    assert runner.CONFIG["layer_policy"]["enabled"] is True and runner.CONFIG["completion_policy"]["enabled"] is True
    assert runner.MEASUREMENT_POLICY_DECLARATION["frozen_v8_policy_feasible_for_cohort_geometry"] is False
    assert runner.CONFIG["registration"]["max_inlier_plane_p95_mm"] == 1.
    assert runner.CONFIG["surface_budgets"]["max_ties_per_query"] == 256
    assert runner.CONTEXT_V4_REQUEST["budgets"]["max_extrema_coordinate_visits"] == 512_000_000
    assert runner.CONTEXT_V4_REQUEST["mode"] == "v4"
    assert set(runner.CONFIG_DELTA_FROM_V8) == {"measurement_max_fit_residual_mm", "measurement_plane_exclusion_mm",
        "support_max_angular_gap_deg", "initialization_completion_lane", "measurement_layer_selection", "inherited_v7_to_v8"}
    assert runner.CONFIG["support"]["max_angular_gap_deg"] == 30. and {k: runner.CONFIG_DELTA_FROM_V8["support_max_angular_gap_deg"][k] for k in ("v8", "v9", "reason")} == \
        dict(v8=5., v9=30., reason=runner.CONFIG_DELTA_FROM_V8["support_max_angular_gap_deg"]["reason"])
    assert runner.CONFIG["support"]["min_arc_sweep_deg"] == 20. and runner.CONFIG["support"]["max_axial_gap_mm"] == 4.
    assert runner.MEASUREMENT_POLICY_DECLARATION["uniqueness"].startswith("EXPLICIT_PROSPECTIVE_DEVELOPMENT_CANDIDATE_NOT")
    gap = runner.CONFIG_DELTA_FROM_V8["support_max_angular_gap_deg"]
    assert gap["spacing_assumption"].startswith("ASSUMED_CROSS_SECTIONAL_ANGULAR_COLUMN_MODEL_OF_THE_SYNTHETIC_FIXTURE_NOT_A_MEASURED")
    assert gap["value_status"].startswith("ONE_PLAUSIBLE_DEVELOPMENT_MARGIN_NOT_AN_ANALYTICALLY_UNIQUE_OPTIMUM")
    assert "22.5 deg" in gap["reason"] and "regardless of the fit" not in gap["reason"]
    assert "NOT_A_PROOF_THAT_EVERY_PHYSICAL_CARRIER_FAILS" in gap["evidence_scope"]
    declaration = runner.CONFIG["completion_budget_declaration"]
    assert set(declaration["active_fields"]) | set(declaration["inactive_fields"]) == set(runner.CONFIG["completion_budgets"])
    assert not set(declaration["active_fields"]) & set(declaration["inactive_fields"])
    assert runner.MEASUREMENT_POLICY_DECLARATION["physical_claim"].endswith("NOT_A_UNIVERSAL_PHYSICAL_IMPOSSIBILITY")
    assert set(runner.CONFIG_DELTA_FROM_V8["inherited_v7_to_v8"]) == {"registration_max_inlier_plane_p95_mm",
        "surface_max_ties_per_query", "context_max_extrema_coordinate_visits", "workflow_max_measurement_contexts"}
    assert runner.SCORING_TRANSPORT["wrapper_schema"] == "GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V4"
    assert runner.RUNNER_SOURCE in runner.STAGE_SAMPLE_SOURCE_PATHS
    assert [name for name in runner.STAGE_SAMPLE_SOURCE_PATHS if "run_gold11_" in name] == [runner.RUNNER_SOURCE]
    assert runner.POSE_ADMISSION_DECLARATION["max_inlier_plane_p95_mm"] == runner.CONFIG["registration"]["max_inlier_plane_p95_mm"]
    assert runner.SCORING_SOURCES[-1] == "scripts/gold11_schedule_projection_v4.py"
    assert "scripts/gold11_schedule_projection_v3.py" not in runner.SCORING_SOURCES
    for arm in runner.ARMS:
        definition = runner.ARM_DEFINITIONS[arm]
        assert definition["implementation"] == "CAD_GUIDED_MESH_OFFLINE_V16_CONTEXT_V4_COMPLETION_V1_KERNEL_V2"
        assert definition["initialization_implementation"] == "CAD_GUIDED_PLANE_INITIALIZATION_V15_CONTEXT_V4_COMPLETION_V1"
        assert definition["two_plane_context_mode"] == "v4"


def test_unchanged_v8_values_outside_the_declared_deltas(runner):
    tree = ast.parse(V8.read_bytes())
    namespace = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("CONFIG", "LIMITS", "CERTIFICATE_CAPS", "SEEDED_PLANE_BUDGETS",
                                           "COMPOSITE_PLANE_BUDGETS", "TWO_PLANE_DECLARATION", "SCHEDULE_V2_BUDGETS"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<v8-constants>", "exec"),
                 dict(math=importlib.import_module("math")), namespace)
    v8 = namespace["CONFIG"]
    v9 = json.loads(json.dumps(runner.CONFIG))
    assert v9["measurement"].pop("max_fit_residual_mm") == .03 and v8["measurement"].pop("max_fit_residual_mm") == .05
    assert v9["measurement"].pop("plane_exclusion_mm") == .04 and v8["measurement"].pop("plane_exclusion_mm") == .075
    assert v9["support"].pop("max_angular_gap_deg") == 30. and v8["support"].pop("max_angular_gap_deg") == 5.
    assert v9.pop("completion_budgets") == v8["initialization_budgets"]
    assert set(v9.pop("completion_budget_declaration")) == {"schema", "active_fields", "inactive_fields", "inactive_scope",
        "work_units_formula", "combined_allowance", "ledger_cap"}
    assert set(v9.pop("completion_policy")) == {"enabled", "max_parents", "max_completions_per_parent",
        "max_completions_total", "max_shift_mm", "min_axis_projection", "min_pair_sine"}
    assert set(v9.pop("layer_policy")) == {"enabled", "layer_split_gap_mm", "min_layer_rows", "max_layers_reported",
        "max_layer_normal_angle_deg", "max_layer_gap_mm", "min_projected_interval_overlap", "max_layer_rms_mm"}
    assert wire(v9) == wire(json.loads(json.dumps(v8)))
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
    assert "scripts/gold11_contact_scoring_v3.py" in runner.SCORING_SOURCES and "scripts/gold11_contact_scoring_v2.py" in runner.SCORING_SOURCES
    for name in ("backend/cad_guided_raw_extrema_index_v3.py", "backend/cad_guided_two_plane_context_v4.py",
                 "backend/cad_guided_report_envelope_v7.py", "scripts/gold11_contact_scoring_v3.py",
                 "backend/cad_guided_triangle_surface_v4.py", "backend/cad_guided_global_registration_v4.py",
                 "backend/cad_guided_stage_policies_v2.py", "backend/cad_guided_plane_initialization_v15.py",
                 "backend/cad_guided_mesh_offline_v16.py", "scripts/gold11_schedule_projection_v4.py",
                 "backend/cad_guided_measurement_v2.py", "backend/cad_guided_context_measurement_v2.py",
                 "backend/cad_guided_support_cover_measurement_v3.py",
                 "scripts/run_gold11_mesh_development_v9.py", runner.SOURCE_BASIS, runner.PROTOCOL,
                 "scripts/run_gold11_mesh_development_v7.py", "scripts/gold11_v9_execute_waiter.py",
                 "docs/research/2026-09-13_gold11_mesh_development_v9_delta_design.md"):
        assert name in runner.SOURCES, name
    for lazy in ("cad_guided_mesh_input_v1", "cad_guided_mesh_topology_v4", "cad_guided_native_features",
                 "cad_guided_native_features_v1", "cad_guided_cad_surface_v2", "cad_guided_support_cover_measurement_v3",
                 "cad_guided_context_surface_roles_v1", "cad_guided_composite_surface_roles_v1",
                 "cad_guided_scan_plane_preparation_v1", "cad_guided_seeded_plane_proposals_v1",
                 "cad_guided_two_plane_schedule_v2", "cad_guided_mesh_offline_v16",
                 "cad_guided_context_measurement_v2", "cad_guided_measurement_v2", "cad_guided_report_envelope_v7"):
        importlib.import_module("backend."+lazy)
    closure = {name[len("backend."):]+".py" for name in sys.modules
               if name.startswith("backend.") and sys.modules[name] is not None}
    declared = {Path(name).name for name in runner.SOURCES if name.startswith("backend/")}
    assert closure <= declared, sorted(closure-declared)


def test_frozen_arm_requests_are_json_stable_and_bind_v4_identities(runner):
    for arm in runner.ARMS:
        request = runner.arm_request(arm)
        assert wire(request) == wire(json.loads(wire(request)))
        assert request["implementation"] == "CAD_GUIDED_MESH_OFFLINE_V16_CONTEXT_V4_COMPLETION_V1_KERNEL_V2"
        assert request["registration_implementation"] == "SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V4"
        assert request["context_request"] == runner.CONTEXT_V4_REQUEST
        assert request["two_plane_context_mode"] == "v4"
    assert runner.arm_request("control")["scan_plane_mode"] == "v7"
    assert runner.arm_request("challenger")["scan_plane_mode"] == "composite_v1"
    assert set(runner.scan_plane_requests()) == {"control", "challenger"}


def test_real_constructor_bindings_for_both_arms(runner):
    for arm in runner.ARMS:
        entry, policies = runner.workflow_entry_and_policies(arm)
        assert entry.__name__ == "inspect_cad_guided_mesh_v16"
        assert type(policies["workflow_policy"]).__name__ == "MeshWorkflowPolicyV16"
        assert type(policies["completion_policy"]).__name__ == "CompletionLanePolicyV1"
        assert type(policies["completion_budgets"]).__name__ == "PlaneInitializationBudgetsV12"
        assert type(policies["layer_policy"]).__name__ == "LayerSelectionPolicyV1"
        assert policies["measurement_policy"].max_fit_residual_mm == .03 and policies["measurement_policy"].plane_exclusion_mm == .04
        assert type(policies["two_plane_context_budgets"]).__name__ == "TwoPlaneContextBudgetsV4"
        assert type(policies["surface_budgets"]).__name__ == "TriangleSurfaceBudgetsV4"
        assert type(policies["registration_budgets"]).__name__ == "SurfaceRegistrationBudgetsV4"
        assert policies["surface_budgets"].max_ties_per_query == 256
        assert policies["registration_policy"].max_inlier_plane_p95_mm == 1.
        assert policies["workflow_policy"].max_measurement_contexts == 8
        assert policies["two_plane_context_mode"] == "v4"
        assert policies["two_plane_schedule_mode"] == ("v2" if arm == "challenger" else "v1")


def test_inert_freeze_and_verify_round_trip_bind_the_v8_basis(runner, monkeypatch):
    store, blobs = {}, {name: (ROOT/name).read_bytes() for name in runner.SOURCES}
    run = Path("/invented/v9/run")
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
    assert bundle["schema"] == "GOLD11_MESH_DEVELOPMENT_V9"
    assert bundle["config_delta_from_v8"] == runner.CONFIG_DELTA_FROM_V8
    assert bundle["pose_admission_declaration"] == runner.POSE_ADMISSION_DECLARATION
    assert bundle["measurement_policy_declaration"] == runner.MEASUREMENT_POLICY_DECLARATION
    assert bundle["source_sha256"][runner.SOURCE_BASIS] == V8_SHA
    assert set(bundle["source_sha256"]) == set(runner.SOURCES)
    assert len(wire(bundle)) < 256*1024
    blobs[runner.SOURCE_BASIS] += b"changed"
    with pytest.raises(ValueError, match="V9_FROZEN_V8_SOURCE_BASIS_CHANGED"):
        runner.freeze()
    changed = json.loads(store["FROZEN.json"])
    changed["source_profile"] = dict(changed["source_profile"], implementation="INVENTED")
    store["FROZEN.json"] = wire(changed)
    with pytest.raises(ValueError, match="FROZEN_PROTOCOL_CONSTANTS_CHANGED"):
        runner.verify_bundle()
