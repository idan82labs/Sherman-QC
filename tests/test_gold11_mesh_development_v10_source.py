"""Focused source controls for the V10 runner (ported from the V9 source test): identities, closure, config deltas,
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
V10 = ROOT/"scripts/run_gold11_mesh_development_v10.py"
V9 = ROOT/"scripts/run_gold11_mesh_development_v9.py"
V9_SHA = "4567a0f584e15b75339238b3f50ebb19209f127bfcb3a55e5fa73fd36b4a85b1"


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
    spec = importlib.util.spec_from_file_location("_v10_runner_source_test", V10)
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
    assert sha(V9.read_bytes()) == V9_SHA == runner.SOURCE_BASIS_SHA
    assert runner.SOURCE_BASIS == "scripts/run_gold11_mesh_development_v9.py"
    assert runner.SOURCE_PROFILE["implementation"] == "GOLD11_MESH_DEVELOPMENT_V10"
    assert runner.SOURCE_PROFILE["scientific_config_or_scorer_change"] is True
    assert runner.SOURCE_PROFILE["scorer_change"] is True and "MATCHER_GATES_AND_CARRIER_UNCHANGED" in runner.SOURCE_PROFILE["scorer_change_scope"]
    assert runner.RUN == ROOT/"docs/research/gold11_mesh_development_v10_attempt"
    assert runner.LEAF_SCHEMA == "ONE_USE_PARENT_BOUND_LEAF_V10"
    assert runner.RUNNER_SOURCE == "scripts/run_gold11_mesh_development_v10.py"
    assert runner.TIMING_RELEASE_STATUS == "REVIEWED_V10_EXACT_TIMING_RELEASE"
    # Inherited V9 policy values (unchanged): derived measurement policy, 30 deg gap, completion and layer policies.
    assert runner.CONFIG["measurement"]["max_fit_residual_mm"] == .03 and runner.CONFIG["measurement"]["plane_exclusion_mm"] == .04
    assert runner.CONFIG["support"]["max_angular_gap_deg"] == 30. and runner.CONFIG["layer_policy"]["enabled"] is True
    assert runner.CONFIG["completion_policy"]["enabled"] is True and runner.CONFIG["registration"]["max_inlier_plane_p95_mm"] == 1.
    assert runner.CONFIG["surface_budgets"]["max_ties_per_query"] == 256 and runner.CONFIG["workflow"]["max_measurement_contexts"] == 8
    assert runner.CONTEXT_V4_REQUEST["budgets"]["max_extrema_coordinate_visits"] == 512_000_000 and runner.CONTEXT_V4_REQUEST["mode"] == "v4"
    # NEW allowance inside the unchanged caps.
    allowance = runner.CONFIG["refinement_allowance"]
    assert allowance == dict(enabled=True, main_refined_groups=12, completion_refined_groups=4, main_query_rows=6_000_000,
                             completion_query_rows=3_000_000, cpu_budget_seconds=1860., cpu_guard_seconds=1500., cooperative_chunk_rows=32_768)
    assert allowance["cpu_budget_seconds"] == runner.LIMITS["case_cpu_seconds"] == 1860 and allowance["cpu_guard_seconds"] < allowance["cpu_budget_seconds"]
    assert set(runner.CONFIG_DELTA_FROM_V9) == {"refinement_allowance", "native_feature_bridge", "query_reuse",
        "completion_budget_declaration", "worker_stage_checkpoints", "inherited_v8_to_v9", "inherited_v7_to_v8"}
    assert runner.CONFIG_DELTA_FROM_V9["refinement_allowance"]["v10"] == allowance and runner.CONFIG_DELTA_FROM_V9["refinement_allowance"]["v9"] is None
    assert "not a raised cap" in runner.CONFIG_DELTA_FROM_V9["refinement_allowance"]["reason"]
    assert runner.CONFIG_DELTA_FROM_V9["refinement_allowance"]["timing_dependence"].startswith("CPU_GUARD_STOPS_ARE_TIMING_DEPENDENT")
    assert runner.CONFIG_DELTA_FROM_V9["inherited_v8_to_v9"]["support_max_angular_gap_deg"] == dict(v8=5., v9=30.)
    assert set(runner.CONFIG_DELTA_FROM_V9["inherited_v7_to_v8"]) == {"registration_max_inlier_plane_p95_mm",
        "surface_max_ties_per_query", "context_max_extrema_coordinate_visits", "workflow_max_measurement_contexts"}
    declaration = runner.CONFIG["completion_budget_declaration"]
    assert set(declaration["active_fields"]) | set(declaration["inactive_fields"]) == set(runner.CONFIG["completion_budgets"])
    assert not set(declaration["active_fields"]) & set(declaration["inactive_fields"])
    assert {"max_node_visits", "max_triangle_pairs", "max_normal_pairs"} <= set(declaration["active_fields"])   # V10 correction
    assert set(declaration["inactive_fields"]) == {"max_prefilter_unsigned_assignments", "max_refined_start_groups"}
    assert declaration["finite_query_charging"].startswith("V10_CORRECTION")
    assert runner.MEASUREMENT_POLICY_DECLARATION["physical_claim"].endswith("NOT_A_UNIVERSAL_PHYSICAL_IMPOSSIBILITY")
    assert runner.SCORING_TRANSPORT["wrapper_schema"] == "GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V5"
    assert runner.RUNNER_SOURCE in runner.STAGE_SAMPLE_SOURCE_PATHS
    assert [name for name in runner.STAGE_SAMPLE_SOURCE_PATHS if "run_gold11_" in name] == [runner.RUNNER_SOURCE]
    assert runner.POSE_ADMISSION_DECLARATION["max_inlier_plane_p95_mm"] == runner.CONFIG["registration"]["max_inlier_plane_p95_mm"]
    assert runner.SCORING_SOURCES[-2:] == ("scripts/gold11_contact_scoring_v4.py", "scripts/gold11_schedule_projection_v5.py")
    assert runner.STAGE_LOG_DECLARATION["checkpoint_stdout_bytes"] == 65536
    assert (runner.STAGE_LOG_DECLARATION["sample_stream_bytes"]+runner.STAGE_LOG_DECLARATION["receipt_bytes"]
            +runner.STAGE_LOG_DECLARATION["checkpoint_stdout_bytes"]+runner.STAGE_LOG_DECLARATION["checkpoint_cap_notice_bytes"]
            +runner.STAGE_LOG_DECLARATION["other_output_remaining_bytes"]) == runner.STAGE_LOG_DECLARATION["whole_leaf_log_bytes"] == runner.LIMITS["case_log_bytes"]
    for arm in runner.ARMS:
        definition = runner.ARM_DEFINITIONS[arm]
        assert definition["implementation"] == "CAD_GUIDED_MESH_OFFLINE_V17_CONTEXT_V4_COMPLETION_V1_KERNEL_V2_NATIVE_BRIDGE_V1_ALLOWANCE_V1"
        assert definition["initialization_implementation"] == "CAD_GUIDED_PLANE_INITIALIZATION_V16_CONTEXT_V4_COMPLETION_V1_ALLOWANCE_V1"
        assert definition["registration_implementation"] == "SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V5_CHUNKED_QUERIES_REUSE_V1_ALLOWANCE_V1"
        assert definition["two_plane_context_mode"] == "v4" and "COVER_V4" in definition["measurement"]


def test_unchanged_v9_values_outside_the_declared_deltas(runner):
    tree = ast.parse(V9.read_bytes())
    namespace = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("CONFIG", "LIMITS", "CERTIFICATE_CAPS", "SEEDED_PLANE_BUDGETS",
                                           "COMPOSITE_PLANE_BUDGETS", "TWO_PLANE_DECLARATION", "SCHEDULE_V2_BUDGETS",
                                           "MEASUREMENT_POLICY_DECLARATION", "POSE_ADMISSION_DECLARATION", "POSITION_ERROR_DECLARATION"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<v9-constants>", "exec"),
                 dict(math=importlib.import_module("math")), namespace)
    v9 = json.loads(json.dumps(namespace["CONFIG"]))
    v10 = json.loads(json.dumps(runner.CONFIG))
    assert set(v10.pop("refinement_allowance")) == {"enabled", "main_refined_groups", "completion_refined_groups", "main_query_rows",
                                                    "completion_query_rows", "cpu_budget_seconds", "cpu_guard_seconds", "cooperative_chunk_rows"}
    old, new = v9.pop("completion_budget_declaration"), v10.pop("completion_budget_declaration")
    assert set(new) == set(old) | {"finite_query_charging", "allowance_interaction"}
    assert set(old["active_fields"]) | {"max_node_visits", "max_triangle_pairs", "max_normal_pairs"} == set(new["active_fields"])
    assert new["schema"] == old["schema"] and new["combined_allowance"] == old["combined_allowance"] and new["ledger_cap"] == old["ledger_cap"]
    assert wire(v10) == wire(v9)                                  # every other CONFIG value byte-identical to V9
    assert v10["completion_budgets"] == v9["completion_budgets"] == v9["initialization_budgets"]
    for name in ("LIMITS", "CERTIFICATE_CAPS", "SEEDED_PLANE_BUDGETS", "COMPOSITE_PLANE_BUDGETS", "SCHEDULE_V2_BUDGETS",
                 "MEASUREMENT_POLICY_DECLARATION", "POSE_ADMISSION_DECLARATION", "POSITION_ERROR_DECLARATION"):
        assert getattr(runner, name) == namespace[name], name
    assert runner.TWO_PLANE_DECLARATION["schedule_budgets"] == namespace["TWO_PLANE_DECLARATION"]["schedule_budgets"]


def test_declared_sources_exist_and_cover_the_engine_import_closure(runner):
    missing = [name for name in runner.SOURCES if not (ROOT/name).is_file()]
    assert missing == []
    assert len(set(runner.SOURCES)) == len(runner.SOURCES)
    assert "scripts/gold11_contact_scoring_v4.py" in runner.SCORING_SOURCES and "scripts/gold11_contact_scoring_v3.py" in runner.SCORING_SOURCES
    for name in ("backend/cad_guided_raw_extrema_index_v3.py", "backend/cad_guided_two_plane_context_v4.py",
                 "backend/cad_guided_report_envelope_v7.py", "scripts/gold11_contact_scoring_v3.py",
                 "backend/cad_guided_triangle_surface_v4.py", "backend/cad_guided_global_registration_v4.py",
                 "backend/cad_guided_stage_policies_v2.py", "backend/cad_guided_plane_initialization_v15.py",
                 "backend/cad_guided_mesh_offline_v16.py", "scripts/gold11_schedule_projection_v4.py",
                 "backend/cad_guided_measurement_v2.py", "backend/cad_guided_context_measurement_v2.py",
                 "backend/cad_guided_support_cover_measurement_v3.py",
                 "scripts/run_gold11_mesh_development_v10.py", runner.SOURCE_BASIS, runner.PROTOCOL,
                 "scripts/run_gold11_mesh_development_v8.py", "scripts/run_gold11_mesh_development_v7.py",
                 "scripts/gold11_v10_execute_waiter.py", "backend/cad_guided_native_feature_bridge_v1.py",
                 "backend/cad_guided_global_registration_v5.py", "backend/cad_guided_plane_initialization_v16.py",
                 "backend/cad_guided_context_measurement_v3.py", "backend/cad_guided_support_cover_measurement_v4.py",
                 "backend/cad_guided_mesh_offline_v17.py", "scripts/gold11_contact_scoring_v4.py",
                 "scripts/gold11_schedule_projection_v5.py",
                 "docs/research/2026-09-13_gold11_mesh_development_v10_delta_design.md"):
        assert name in runner.SOURCES, name
    for lazy in ("cad_guided_mesh_input_v1", "cad_guided_mesh_topology_v4", "cad_guided_native_features",
                 "cad_guided_native_features_v1", "cad_guided_cad_surface_v2", "cad_guided_support_cover_measurement_v3",
                 "cad_guided_context_surface_roles_v1", "cad_guided_composite_surface_roles_v1",
                 "cad_guided_scan_plane_preparation_v1", "cad_guided_seeded_plane_proposals_v1",
                 "cad_guided_two_plane_schedule_v2", "cad_guided_mesh_offline_v16", "cad_guided_mesh_offline_v17",
                 "cad_guided_context_measurement_v2", "cad_guided_context_measurement_v3", "cad_guided_measurement_v2",
                 "cad_guided_support_cover_measurement_v4", "cad_guided_native_feature_bridge_v1",
                 "cad_guided_global_registration_v5", "cad_guided_plane_initialization_v16", "cad_guided_report_envelope_v7"):
        importlib.import_module("backend."+lazy)
    closure = {name[len("backend."):]+".py" for name in sys.modules
               if name.startswith("backend.") and sys.modules[name] is not None}
    declared = {Path(name).name for name in runner.SOURCES if name.startswith("backend/")}
    assert closure <= declared, sorted(closure-declared)


def test_frozen_arm_requests_are_json_stable_and_bind_v4_identities(runner):
    for arm in runner.ARMS:
        request = runner.arm_request(arm)
        assert wire(request) == wire(json.loads(wire(request)))
        assert request["implementation"] == "CAD_GUIDED_MESH_OFFLINE_V17_CONTEXT_V4_COMPLETION_V1_KERNEL_V2_NATIVE_BRIDGE_V1_ALLOWANCE_V1"
        assert request["initialization_implementation"] == "CAD_GUIDED_PLANE_INITIALIZATION_V16_CONTEXT_V4_COMPLETION_V1_ALLOWANCE_V1"
        assert request["registration_implementation"] == "SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V5_CHUNKED_QUERIES_REUSE_V1_ALLOWANCE_V1"
        assert request["common_config"]["refinement_allowance"] == runner.CONFIG["refinement_allowance"]
        assert request["context_request"] == runner.CONTEXT_V4_REQUEST
        assert request["two_plane_context_mode"] == "v4"
    assert runner.arm_request("control")["scan_plane_mode"] == "v7"
    assert runner.arm_request("challenger")["scan_plane_mode"] == "composite_v1"
    assert set(runner.scan_plane_requests()) == {"control", "challenger"}


def test_real_constructor_bindings_for_both_arms(runner):
    for arm in runner.ARMS:
        entry, policies = runner.workflow_entry_and_policies(arm)
        assert entry.__name__ == "inspect_cad_guided_mesh_v17"
        assert type(policies["workflow_policy"]).__name__ == "MeshWorkflowPolicyV17"
        allowance = policies["refinement_allowance"]
        assert type(allowance).__name__ == "RefinementAllowanceV1" and allowance.enabled is True
        assert allowance.cpu_budget_seconds == runner.LIMITS["case_cpu_seconds"] and allowance.cpu_guard_seconds == 1500.
        assert allowance.main_refined_groups == 12 and allowance.completion_refined_groups == 4 and allowance.cooperative_chunk_rows == 32768
        assert "stage_checkpoint" not in policies                      # the worker attaches the bounded printer at dispatch
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


def test_inert_freeze_and_verify_round_trip_bind_the_v9_basis(runner, monkeypatch):
    store, blobs = {}, {name: (ROOT/name).read_bytes() for name in runner.SOURCES}
    run = Path("/invented/v10/run")
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
    assert bundle["schema"] == "GOLD11_MESH_DEVELOPMENT_V10"
    assert bundle["config_delta_from_v9"] == runner.CONFIG_DELTA_FROM_V9
    assert bundle["config"]["refinement_allowance"] == runner.CONFIG["refinement_allowance"]
    assert bundle["pose_admission_declaration"] == runner.POSE_ADMISSION_DECLARATION
    assert bundle["measurement_policy_declaration"] == runner.MEASUREMENT_POLICY_DECLARATION
    assert bundle["source_sha256"][runner.SOURCE_BASIS] == V9_SHA
    assert set(bundle["source_sha256"]) == set(runner.SOURCES)
    assert len(wire(bundle)) < 256*1024
    blobs[runner.SOURCE_BASIS] += b"changed"
    with pytest.raises(ValueError, match="V10_FROZEN_V9_SOURCE_BASIS_CHANGED"):
        runner.freeze()
    changed = json.loads(store["FROZEN.json"])
    changed["source_profile"] = dict(changed["source_profile"], implementation="INVENTED")
    store["FROZEN.json"] = wire(changed)
    with pytest.raises(ValueError, match="FROZEN_PROTOCOL_CONSTANTS_CHANGED"):
        runner.verify_bundle()


def test_stage_checkpoint_printer_is_bounded_and_never_raises(runner, capsys):
    emit = runner.stage_checkpoint_printer()
    emit(dict(stage="REFINEMENT_BEGIN", candidates=3, process_cpu_seconds=1.5))
    emit(dict(stage="UNENCODABLE", value=float("nan")))          # allow_nan=False -> omitted whole, not raised
    big = dict(stage="X", payload="y"*70000)
    emit(big)                                                    # exceeds the cap -> notice line, then silence
    emit(dict(stage="AFTER_CAP"))
    out = capsys.readouterr().out.splitlines()
    assert json.loads(out[0]) == dict(status="STAGE_CHECKPOINT", checkpoint=dict(stage="REFINEMENT_BEGIN", candidates=3, process_cpu_seconds=1.5))
    assert json.loads(out[1]) == dict(status="STAGE_CHECKPOINT_UNENCODABLE_OMITTED")
    assert json.loads(out[2]) == dict(status="STAGE_CHECKPOINT_STDOUT_CAP_REACHED_FURTHER_CHECKPOINTS_OMITTED") and len(out) == 3
    assert emit.state["truncated"] is True and emit.state["bytes"] <= runner.STAGE_LOG_DECLARATION["checkpoint_stdout_bytes"]
