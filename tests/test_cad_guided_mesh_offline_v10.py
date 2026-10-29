"""UNFROZEN V10 byte-workflow publication controls with native/pose stubs.

These execute the complete Python orchestration only with explicitly invented
bytes and mocked native, sampler, pose and measurement boundaries. They establish
dispatch/lineage/failure propagation, not geometry, physical accuracy or F1.
The unchanged actual four-case native workflow remains a separate required gate.
All44 predecessor controls remain exact apart from explicit V9-to-V10 names.
The mocked box-completion label is not evidence of actual face/row binding,
box screening, geometric exclusion, speedup or native support qualification.
Prospective54 outcomes:44 inherited,8 new refusal cases and2 source/positive guards.
"""

from dataclasses import asdict
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_offline_v9.py")
PREVIOUS_SHA256 = "b3bb60f5b5e5e2dceb5a1bb87e3b9290178218b169095b90f244a8f13c3ccd00"
BOX_POLICY = "REJECTION_ONLY_COMPLETE_BOUND_FACE_AND_SUPPORT_V1"
MOCK_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_offline_v2.py")
POSITION_ERROR_BOUND_MM = .005


def _load(path):
    spec = importlib.util.spec_from_file_location("_v10_workflow_mock_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plumbing(monkeypatch):
    """Existing independently invented V2 stubs; NO native/pose-recovery claim."""
    old = _load(MOCK_SOURCE)
    state = old.fixture.__wrapped__(monkeypatch)
    package = old.workflow.__package__
    workflow = importlib.import_module(package+".cad_guided_mesh_offline_v10")
    topology_module = importlib.import_module(package+".cad_guided_mesh_topology_v4")
    initialize = importlib.import_module(package+".cad_guided_plane_initialization_v10")
    raw = state["scan"]["raw_points"]
    triangles = old.immutable(np.arange(72).reshape(24, 3), "<i8")
    state["scan"]["input_admission"] = dict(face_count=24, face_index_count=72)
    state["topology"] = dict(status="AVAILABLE", reasons=[], points=raw, triangle_raw_rows=triangles,
        vertex_count=72, triangle_count=24, source_sha256=hashlib.sha256(old.SCAN_BYTES).hexdigest(),
        points_sha256=old.digest(raw), triangle_raw_rows_sha256=old.digest(triangles),
        raw_row_order="ORIGINAL_PLY_VERTEX_ROWS", triangle_row_order="ORIGINAL_PLY_FACE_ROWS")
    monkeypatch.setattr(topology_module, "parse_triangle_mesh_ply_v4", lambda blob: state["topology"])
    def register(points, rows, cad, faces, local, **kwargs):
        state["events"].append("register_v10")
        state["register_v10_args"] = points, rows, cad, faces, local, kwargs
        digests = dict(scan_points_sha256=old.digest(points), scan_triangle_raw_rows_sha256=old.digest(rows),
            cad_triangles_sha256=old.digest(cad), cad_face_ids_sha256=old.digest(faces), cad_source_triangle_ids_sha256=old.digest(local),
            position_error_bounds_mm_sha256=old.digest(np.full(len(points), kwargs["position_error_bounds_mm"], dtype="<f8")))
        if state.get("binding_fault"):
            digests[state["binding_fault"]] = "0"*64
        evidence = state["registration"]["evidence"]
        evidence.update(initialization=dict(status="COMPLETE", search_complete=True, selected_policy_evaluation_complete=True,
            search_exhaustive=False, input_digests=dict(digests)), input_digests=dict(digests),
            candidate_evaluation_complete=not state.get("incomplete"), prefilter_full_scan_raw_rows=72, registration_input_rows=36,
            registration_input_points_sha256=old.digest(state["scan"]["points"]), registration_raw_rows=dict(
                selection="EXPLICIT_ORIGINAL_RAW_IDS_NO_HIDDEN_SAMPLING", count=36, sha256=old.digest(state["scan"]["engine_to_raw_row_indices"])),
            cad_initialization_points_sha256=old.digest(np.ascontiguousarray(np.unique(kwargs["cad_initialization_points"], axis=0), dtype="<f8")))
        state["registration"]["schema_version"] = "noisy_plane_initialized_finite_registration_v10"
        evidence.update(schema="NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V10",
            numerical_consumer_complete=True,
            numerical_support_policy="DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION",
            distance_model="CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL",
            optimizer_objective="MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS",
            selected_candidate_index=0,
            candidates=[dict(index=0, execution_status="ATTEMPTED", reasons=[],
                rotation=old.ROTATION.tolist(), translation=old.TRANSLATION.tolist(),
                numerical_support_policy="DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION",
                distance_model="CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL",
                optimizer_objective="MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS",
                numerical_consumer_status="COMPLETE_V3_INTERVAL_SUPPORT_POLICY")])
        evidence["initialization"].update(schema_version="noisy_plane_initialization_prefilter_v10",
            box_screening_complete=True, box_screening_policy=BOX_POLICY,
            numerical_consumer_complete=True,
            numerical_support_policy="DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION")
        if state.get("numerical_fault") is not None:
            state["numerical_fault"](state["registration"])
        return state["registration"]
    monkeypatch.setattr(initialize, "register_whole_part_planes_v10", register)
    def run(**overrides):
        policies = old.policies()
        policies["workflow_policy"] = workflow.MeshWorkflowPolicyV10(**asdict(policies["workflow_policy"]))
        args = dict(scan_bytes=old.SCAN_BYTES, cad_bytes=old.CAD_BYTES, position_error_bound_mm=POSITION_ERROR_BOUND_MM,
                    allow_ocp=True, **policies)
        args.update(overrides)
        return workflow.inspect_cad_guided_mesh_v10(**args)
    return old, workflow, state, run


def test_mock_bytes_epsilon_full_raw_measurement_and_inverse_publication(plumbing):
    old, _, state, run = plumbing
    result = run()
    assert result["status"] == "PARTIAL" and result["phase"] == "COMPLETE"
    assert state["read_bytes"] is old.CAD_BYTES
    assert state["extract_args"][0] is state["sample_args"][0] is state["shape"]
    points, rows, cad, faces, local, kwargs = state["register_v10_args"]
    assert points is state["scan"]["raw_points"] and len(points) == 72
    assert rows is state["topology"]["triangle_raw_rows"] and cad is state["cad"]["triangles"]
    assert faces is state["cad"]["source_face_indices"] and local is state["cad"]["source_triangle_indices"]
    assert kwargs["position_error_bounds_mm"] == .005
    assert kwargs["registration_raw_rows"] is state["scan"]["engine_to_raw_row_indices"]
    measured, _, args = state["measure_args"]
    np.testing.assert_array_equal(measured, points@old.ROTATION.T+old.TRANSLATION)
    assert len(measured) == 72 and args["scan_sha256"] == hashlib.sha256(old.SCAN_BYTES).hexdigest()
    assert "position_error_bounds_mm" not in args
    for i, key in enumerate(old.ENDPOINT_FIELDS):
        np.testing.assert_array_equal(result["predictions"][0][key], [i+.25, 2., 3.])
    assert result["measurements"]["features"][0]["raw_support"]["arc_rows"] == [1, 5, 69]
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ("epsilon_digest", "scan_byte_digest", "incomplete", "refusal", "late_publication"))
def test_mock_bad_binding_refusal_or_late_output_cannot_publish(plumbing, fault):
    _, _, state, run = plumbing
    if fault == "epsilon_digest":
        state["binding_fault"] = "position_error_bounds_mm_sha256"
    elif fault == "scan_byte_digest":
        state["topology"]["source_sha256"] = "0"*64
    elif fault == "incomplete":
        state["incomplete"] = True
    elif fault == "refusal":
        state["registration"].update(status="UNAVAILABLE", transform_scan_to_cad=None, reasons=["MOCK_GEOMETRIC_REFUSAL"])
    else:
        state["measured"]["finite_support_instances"].append(dict(state["measured"]["finite_support_instances"][0], line_end=[float("nan"), 0., 0.]))
    result = run()
    assert result["status"] == "UNAVAILABLE" and result["predictions"] == [] and result["measurements"] is None
    if fault != "late_publication":
        assert "measure" not in state["events"]
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False


@pytest.mark.parametrize("epsilon", (None, False, -.1, float("nan"), float("inf"), "0.005"))
def test_mock_invalid_position_declaration_stops_before_loader(plumbing, epsilon):
    _, _, state, run = plumbing
    result = run(position_error_bound_mm=epsilon)
    assert result["reasons"] == ["EXPLICIT_FINITE_POSITION_ERROR_BOUND_REQUIRED"]
    assert state["events"] == [] and result["predictions"] == []



@pytest.mark.parametrize("fault", (
    "registration_schema", "evidence_schema", "initialization_schema",
    "initialization_complete", "registration_complete", "initialization_policy",
    "registration_policy", "distance_model", "optimizer_objective", "selected_negative",
    "selected_boolean", "selected_reasons", "selected_numerical_status",
))
def test_missing_or_mismatched_numerical_consumer_contract_never_reaches_measurement(plumbing, fault):
    _, _, state, run = plumbing
    def damage(registration):
        evidence = registration["evidence"]
        initialization = evidence["initialization"]
        if fault == "registration_schema":
            registration["schema_version"] = "noisy_plane_initialized_finite_registration_v8"
        elif fault == "evidence_schema":
            evidence["schema"] = "LEGACY"
        elif fault == "initialization_schema":
            initialization["schema_version"] = "LEGACY"
        elif fault == "initialization_complete":
            initialization["numerical_consumer_complete"] = False
        elif fault == "registration_complete":
            evidence["numerical_consumer_complete"] = False
        elif fault == "initialization_policy":
            initialization["numerical_support_policy"] = "NOMINAL_ONLY"
        elif fault == "registration_policy":
            evidence["numerical_support_policy"] = "NOMINAL_ONLY"
        elif fault == "distance_model":
            evidence["distance_model"] = "CERTIFIED_NATIVE_CAD"
        elif fault == "optimizer_objective":
            evidence["optimizer_objective"] = "CERTIFIED_SURFACE_OBJECTIVE"
        elif fault == "selected_negative":
            evidence["selected_candidate_index"] = -1
        elif fault == "selected_boolean":
            evidence["selected_candidate_index"] = False
        elif fault == "selected_reasons":
            evidence["candidates"][0]["reasons"] = ["INCOMPLETE"]
        else:
            evidence["candidates"][0].pop("numerical_consumer_status")
    state["numerical_fault"] = damage
    answer = run()
    assert answer["status"] == "UNAVAILABLE"
    assert answer["measurements"] is None and answer["predictions"] == []
    assert "measure" not in state["events"]
    assert answer["registration"]["status"] == "UNAVAILABLE"
    assert answer["registration"]["transform_scan_to_cad"] is None
    assert answer["observed_bend_count"] is None


def test_explicit_new_certificate_budgets_reach_new_registration_types(plumbing):
    _, workflow, state, run = plumbing
    initialization = importlib.import_module(workflow.__package__+".cad_guided_plane_initialization_v10")
    init_budget = initialization.PlaneInitializationBudgetsV10(max_certificate_pairs=123)
    reg_budget = initialization._v3.SurfaceRegistrationBudgetsV3(max_certificate_pairs=456)
    surface_budget = initialization._surface.TriangleSurfaceBudgets(max_certificate_pairs=17)
    result = run(initialization_budgets=init_budget, registration_budgets=reg_budget, surface_budgets=surface_budget)
    assert result["status"] == "PARTIAL"
    kwargs = state["register_v10_args"][-1]
    assert kwargs["budgets"].max_certificate_pairs == 123
    assert kwargs["registration_budgets"].max_certificate_pairs == 456
    assert kwargs["surface_budgets"].max_certificate_pairs == 17
    assert kwargs["budgets"] is not init_budget and kwargs["registration_budgets"] is not reg_budget
    assert kwargs["surface_budgets"] is not surface_budget


@pytest.mark.parametrize("fault", (
    "candidate_boolean_index", "candidate_numpy_index",
    "published_translation", "published_rotation", "selected_translation", "selected_rotation",
    "selected_missing_rotation", "selected_nan_translation", "selected_boolean_rotation",
    "selected_wrong_rotation_shape", "selected_reflection", "selected_legacy_translation_alias",
    "selected_missing_policy", "selected_wrong_policy", "selected_missing_model",
    "selected_wrong_model", "selected_missing_objective", "selected_wrong_objective",
))
def test_selected_pose_and_its_numerical_labels_bind_the_published_transform(plumbing, fault):
    _, _, state, run = plumbing
    def damage(registration):
        candidate = registration["evidence"]["candidates"][0]
        if fault == "candidate_boolean_index":
            candidate["index"] = False
        elif fault == "candidate_numpy_index":
            candidate["index"] = np.int64(0)
        elif fault == "published_translation":
            registration["transform_scan_to_cad"][0][3] += .125
        elif fault == "published_rotation":
            for row in range(3):
                registration["transform_scan_to_cad"][row][:3] = np.eye(3)[row].tolist()
        elif fault == "selected_translation":
            candidate["translation"][0] += .125
        elif fault == "selected_rotation":
            candidate["rotation"] = np.eye(3).tolist()
        elif fault == "selected_missing_rotation":
            candidate.pop("rotation")
        elif fault == "selected_nan_translation":
            candidate["translation"][0] = float("nan")
        elif fault == "selected_boolean_rotation":
            candidate["rotation"] = np.eye(3, dtype=bool).tolist()
        elif fault == "selected_wrong_rotation_shape":
            candidate["rotation"] = [1., 0., 0.]
        elif fault == "selected_reflection":
            candidate["rotation"] = np.diag([-1., 1., 1.]).tolist()
        elif fault == "selected_legacy_translation_alias":
            candidate["translation_mm"] = candidate.pop("translation")
        else:
            key = {"policy": "numerical_support_policy", "model": "distance_model",
                   "objective": "optimizer_objective"}[fault.rsplit("_", 1)[1]]
            if "missing" in fault:
                candidate.pop(key)
            else:
                candidate[key] = "WRONG_SELECTED_CANDIDATE_CONTRACT"
    state["numerical_fault"] = damage
    result = run()
    assert result["status"] == "UNAVAILABLE" and result["predictions"] == []
    assert result["measurements"] is None and "measure" not in state["events"]
    if fault in ("candidate_numpy_index", "selected_nan_translation"):
        # The invalid retained evidence also cannot be serialized as strict JSON.
        # Whole-report refusal is stronger than a nested pose-only refusal.
        assert result["registration"] is None and result["phase"] == "REPORT_PUBLICATION"
        assert result["detailed_report_discarded_whole"] is True
    else:
        assert result["registration"]["status"] == "UNAVAILABLE"
        assert result["registration"]["transform_scan_to_cad"] is None


@pytest.mark.parametrize("fault", (
    "missing_box_complete", "false_box_complete", "integer_box_complete",
    "missing_box_policy", "wrong_box_policy", "v9_registration_schema",
    "v9_evidence_schema", "v9_initialization_schema",
))
def test_missing_incomplete_or_legacy_box_contract_cannot_publish(plumbing, fault):
    _, _, state, run = plumbing
    def damage(registration):
        evidence = registration["evidence"]
        initialization = evidence["initialization"]
        if fault == "missing_box_complete":
            initialization.pop("box_screening_complete")
        elif fault == "false_box_complete":
            initialization["box_screening_complete"] = False
        elif fault == "integer_box_complete":
            initialization["box_screening_complete"] = 1
        elif fault == "missing_box_policy":
            initialization.pop("box_screening_policy")
        elif fault == "wrong_box_policy":
            initialization["box_screening_policy"] = "BOX_COMPLEMENT_ADMITTED_AS_SURFACE_SUPPORT"
        elif fault == "v9_registration_schema":
            registration["schema_version"] = "noisy_plane_initialized_finite_registration_v9"
        elif fault == "v9_evidence_schema":
            evidence["schema"] = "NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V9"
        else:
            initialization["schema_version"] = "noisy_plane_initialization_prefilter_v9"
    state["numerical_fault"] = damage
    result = run()
    assert result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V10"
    assert result["status"] == "UNAVAILABLE" and result["phase"] == "WHOLE_PART_REGISTRATION"
    assert result["measurements"] is None and result["predictions"] == []
    assert "measure" not in state["events"]
    assert result["registration"]["status"] == "UNAVAILABLE"
    assert result["registration"]["transform_scan_to_cad"] is None
    assert result["registration"]["reasons"] == ["INVALID_REGISTRATION_TRANSFORM_OR_BINDING_CONTRACT"]
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False
    json.dumps(result, allow_nan=False)


def test_explicit_v10_mock_box_contract_preserves_successful_publication_boundary(plumbing):
    _, _, state, run = plumbing
    result = run()
    assert result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V10"
    assert result["registration_initialization"] == "PLANE_ONLY_FINITE_ASSIGNED_FACE_PREFILTER_V10"
    assert result["status"] == "PARTIAL" and result["phase"] == "COMPLETE"
    initialization = result["registration"]["evidence"]["initialization"]
    assert initialization["box_screening_complete"] is True
    assert initialization["box_screening_policy"] == BOX_POLICY
    assert result["registration"]["schema_version"] == "noisy_plane_initialized_finite_registration_v10"
    assert initialization["schema_version"] == "noisy_plane_initialization_prefilter_v10"
    assert state["events"] == ["prepare", "read", "extract:v1", "sample", "register_v10", "measure"]
    assert len(result["predictions"]) == 1
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False


def test_all44_frozen_v9_controls_are_preserved_under_explicit_version_adaptation():
    previous = PREVIOUS_SOURCE.read_bytes()
    assert hashlib.sha256(previous).hexdigest() == PREVIOUS_SHA256
    source = ast.parse(previous)
    current = ast.parse(Path(__file__).read_text())
    inherited = {node.name: node for node in source.body
                 if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}
    successors = {node.name: node for node in current.body
                  if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}
    assert set(inherited) == {
        "test_mock_bytes_epsilon_full_raw_measurement_and_inverse_publication",
        "test_mock_bad_binding_refusal_or_late_output_cannot_publish",
        "test_mock_invalid_position_declaration_stops_before_loader",
        "test_missing_or_mismatched_numerical_consumer_contract_never_reaches_measurement",
        "test_explicit_new_certificate_budgets_reach_new_registration_types",
        "test_selected_pose_and_its_numerical_labels_bind_the_published_transform",
    }
    class VersionOnly(ast.NodeTransformer):
        @staticmethod
        def renamed(value):
            return value.replace("V9", "V10").replace("v9", "v10")
        def visit_Name(self, node):
            node.id = self.renamed(node.id)
            return node
        def visit_Attribute(self, node):
            self.generic_visit(node)
            node.attr = self.renamed(node.attr)
            return node
        def visit_Constant(self, node):
            if type(node.value) is str:
                node.value = self.renamed(node.value)
            return node
    for name, node in inherited.items():
        assert ast.dump(VersionOnly().visit(node), include_attributes=False) == ast.dump(
            successors[name], include_attributes=False), name

