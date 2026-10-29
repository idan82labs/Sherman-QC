"""Invented V2 plumbing checks with explicitly mocked native/pose boundaries.

No native import, actual registration search, raw file, Gold, or physical
accuracy test is performed. The pure V2 export supplies invented finite
triangles; a stipulated transform tests only forward/inverse frame plumbing.
Root owns source review and bounded test execution; authors do not run this.
"""

from dataclasses import replace
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


_PACKAGE = "_cad_guided_mesh_offline_v2_tests"
_package = types.ModuleType(_PACKAGE)
_package.__path__ = [str(Path(__file__).resolve().parents[1] / "backend")]
sys.modules[_PACKAGE] = _package
workflow = importlib.import_module(_PACKAGE + ".cad_guided_mesh_offline_v2")
legacy = importlib.import_module(_PACKAGE + ".cad_guided_mesh_offline_v1")
mesh_input = importlib.import_module(_PACKAGE + ".cad_guided_mesh_input_v1")
registration_v1 = importlib.import_module(_PACKAGE + ".cad_guided_global_registration_v1")
registration = importlib.import_module(_PACKAGE + ".cad_guided_global_registration_v2")
primitive = importlib.import_module(_PACKAGE + ".cad_guided_triangle_surface_v2")
native_r0 = importlib.import_module(_PACKAGE + ".cad_guided_native_features")
native_v1 = importlib.import_module(_PACKAGE + ".cad_guided_native_features_v1")
surface = importlib.import_module(_PACKAGE + ".cad_guided_cad_surface_v2")
local = importlib.import_module(_PACKAGE + ".cad_guided_local_measurement_v1")

SCAN_BYTES = b"INVENTED_SCAN_BYTES_FOR_MOCKED_LOADER_ONLY\x00\xff"
CAD_BYTES = b"INVENTED_STEP_BYTES_FOR_MOCKED_READER_ONLY\x00\xff"
ROTATION = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
TRANSLATION = np.array([7., -4., 3.])
TRANSFORM = np.eye(4)
TRANSFORM[:3, :3], TRANSFORM[:3, 3] = ROTATION, TRANSLATION
ENDPOINT_FIELDS = ("line_start", "line_end", "axis_support_line_start", "axis_support_line_end")


def immutable(value, dtype="<f8"):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def policies(branch="v1"):
    return dict(workflow_policy=workflow.MeshWorkflowPolicyV2(
        max_registration_scan_rows=36, cad_sample_count=36, native_adapter_version=branch),
        registration_policy=registration_v1.WholePartRegistrationPolicyV1(
            coarse_points=36, refine_points=36, coarse_iterations=1, refine_iterations=1),
        measurement_policy=native_r0.NativeGuidedPolicy(
            flange_window_mm=(2.5, 12.5), search_margin_mm=3., max_fit_residual_mm=.05,
            plane_exclusion_mm=.075, max_tangency_error_mm=.1, fit_model="free_circle"),
        instance_policy=local.SupportPolicy(4., 5., 20., 3))


@pytest.fixture
def fixture(monkeypatch):
    """Every external/native/inference boundary is a deterministic test stub."""
    state = dict(events=[], shape=object(), feature=object())
    i = np.arange(72, dtype=float)
    raw = immutable(np.column_stack((i % 6, (i // 6) % 4, i // 24)))
    rows = immutable(np.arange(0, 72, 2), "<i8")
    points = immutable(raw[rows])
    state["scan"] = dict(status="AVAILABLE", reasons=[], raw_points=raw, points=points,
        engine_to_raw_row_indices=rows, coverage={"fixture": "NOT_COVERAGE_EVIDENCE"},
        measurement_cloud_contract="FULL_VALIDATED_RAW_XYZ_REQUIRED_FOR_LOCAL_FITTING",
        sample_cloud_contract="REGISTRATION_AND_PROPOSALS_ONLY", units_status="UNVERIFIED",
        coordinate_frame_status="UNVERIFIED", source_bindings=dict(
            scan_sha256=hashlib.sha256(SCAN_BYTES).hexdigest(), raw_xyz_sha256=digest(raw),
            sample_xyz_sha256=digest(points), engine_to_raw_sha256=digest(rows)))
    state["cad"] = surface.prepare_cad_triangle_surface_v2(
        np.array([[[0., 0., 0.], [8., 0., 0.], [0., 5., 0.]],
                  [[0., 0., 2.], [8., 0., 2.], [0., 5., 2.]]]),
        source_face_indices=np.array([2, 5]), source_triangle_indices=np.array([1, 1]),
        sample_count=36)
    assert state["cad"]["status"] == "AVAILABLE"
    state["native"] = dict(status="AVAILABLE", features=(state["feature"],),
                            metadata={"status": "AVAILABLE", "fixture": "MOCKED_NATIVE"})
    state["registration"] = dict(status="CONDITIONAL_BEST_FIT", reasons=[],
        transform_scan_to_cad=TRANSFORM.tolist(), evidence={"fixture": "STIPULATED_POSE_NOT_RECOVERY"})
    instance = dict(feature_id="invented-window", line_convention="MEASURED_INNER_ARC_MID_ANGLE_GENERATOR")
    for index, field in enumerate(ENDPOINT_FIELDS):
        instance[field] = (np.array([index + .25, 2., 3.]) @ ROTATION.T + TRANSLATION).tolist()
    state["measured"] = dict(status="PARTIAL", observed_bend_count=None,
        production_use_allowed=False, finite_support_instances=[instance], features=[dict(
            feature_id="invented-window", raw_support=dict(
                arc_rows=immutable([1, 5, 69], "<i8"),
                flange_rows=(immutable([3, 7, 71], "<i8"), immutable([9, 11, 67], "<i8"))))])

    def prepare(**kwargs):
        state["events"].append("prepare")
        state["prepare_args"] = kwargs
        return state["scan"]

    def read(blob):
        state["events"].append("read")
        state["read_bytes"] = blob
        return state["shape"], dict(target_unit="mm", physical_calibration="NOT_ESTABLISHED",
            target_unit_action="SET_SYSTEM_LENGTH_UNIT_1_AFTER_READ_BEFORE_TRANSFER",
            transferred_roots=1, source_unit_names={"length": ["MILLIMETRE"]})

    def extract(branch, shape, **kwargs):
        state["events"].append("extract:" + branch)
        state["extract_args"] = shape, kwargs
        return state["native"]

    def sample(shape, **kwargs):
        state["events"].append("sample")
        state["sample_args"] = shape, kwargs
        return state["cad"]

    def register(scan, triangles, face_ids, **kwargs):
        state["events"].append("register")
        state["register_args"] = scan, triangles, face_ids, kwargs
        return state["registration"]

    def measure(points, features, **kwargs):
        state["events"].append("measure")
        state["measure_args"] = points.copy(), features, kwargs
        return state["measured"]

    monkeypatch.setattr(mesh_input, "prepare_mesh_scan_v1", prepare)
    monkeypatch.setattr(workflow, "_read_step_owned_snapshot", read)
    monkeypatch.setattr(native_r0, "extract_ocp_guided_features", lambda *a, **k: extract("r0", *a, **k))
    monkeypatch.setattr(native_v1, "extract_ocp_guided_features_v1", lambda *a, **k: extract("v1", *a, **k))
    monkeypatch.setattr(surface, "sample_native_cad_surface_v2", sample)
    monkeypatch.setattr(registration, "register_whole_part_v2", register)
    monkeypatch.setattr(local, "inspect_full_raw_windows_v1", measure)
    return state


def run(branch="v1", **overrides):
    args = dict(scan_bytes=SCAN_BYTES, cad_bytes=CAD_BYTES, allow_ocp=True, **policies(branch))
    args.update(overrides)
    return workflow.inspect_cad_guided_mesh_v2(**args)


@pytest.mark.parametrize("branch", ["r0", "v1"])
def test_single_snapshot_finite_surface_and_full_raw_plumbing(fixture, branch):
    result = run(branch)
    assert result["status"] == "PARTIAL" and result["phase"] == "COMPLETE"
    assert fixture["events"] == ["prepare", "read", "extract:" + branch, "sample", "register", "measure"]
    assert fixture["read_bytes"] is CAD_BYTES
    assert fixture["prepare_args"] == dict(scan_bytes=SCAN_BYTES, max_engine_rows=36)
    assert fixture["extract_args"][0] is fixture["sample_args"][0] is fixture["shape"]
    assert fixture["sample_args"][1] == dict(allow_ocp=True, native_binding="ocp", sample_count=36,
        linear_deflection_mm=.1, angular_deflection_radians=.2)
    supplied_scan, triangles, faces, kwargs = fixture["register_args"]
    assert supplied_scan is fixture["scan"]["points"] and len(supplied_scan) == 36
    assert triangles is fixture["cad"]["triangles"] and faces is fixture["cad"]["source_face_indices"]
    assert kwargs["cad_initialization_points"] is fixture["cad"]["points"]
    assert kwargs["unit_assumption"] == "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"
    aligned, features, kwargs = fixture["measure_args"]
    assert len(aligned) == 72 and features == (fixture["feature"],)
    np.testing.assert_array_equal(aligned, fixture["scan"]["raw_points"] @ ROTATION.T + TRANSLATION)
    assert kwargs["scan_sha256"] == hashlib.sha256(SCAN_BYTES).hexdigest()
    assert kwargs["registration_status"] == "CONDITIONAL_BEST_FIT"
    assert "source_raw_row_indices" not in kwargs  # All original rows, identity order.
    for index, field in enumerate(ENDPOINT_FIELDS):
        np.testing.assert_array_equal(result["predictions"][0][field], [index + .25, 2., 3.])
    assert result["predictions"][0]["reference_frame"] == result["prediction_frame"]
    assert result["measurement_geometry_frame"] == "CAD_ALIGNED_COORDINATES_ASSUMED_MM"
    assert result["measurements"]["geometry_coordinate_frame"] == result["measurement_geometry_frame"]
    assert result["measurements"]["geometry_transform_reference"] == "registration.transform_scan_to_cad"
    assert result["measurements"]["raw_support_identity_frame"] == "ORIGINAL_SCAN_VERTEX_ROW_IDENTITIES"
    for index, field in enumerate(ENDPOINT_FIELDS):
        expected_cad = np.array([index + .25, 2., 3.]) @ ROTATION.T + TRANSLATION
        np.testing.assert_array_equal(result["measurements"]["finite_support_instances"][0][field],
                                      expected_cad)
        np.testing.assert_array_equal(fixture["measured"]["finite_support_instances"][0][field],
                                      expected_cad)
    assert result["observed_bend_count"] is None and result["observed_support_instance_count"] == 1
    assert result["production_use_allowed"] is False and result["conformance"] == "INDETERMINATE"
    assert result["uncertainty_status"] == "UNCALIBRATED"
    assert result["triangle_sample_binding"]["triangle_count"] == 2
    assert result["cad_sha256"] == hashlib.sha256(CAD_BYTES).hexdigest()
    assert "triangles" not in result["cad_surface_sampling"]
    support = result["measurements"]["features"][0]["raw_support"]
    assert support["arc_rows"] == [1, 5, 69]
    assert support["arc_rows_sha256"] == digest(immutable([1, 5, 69], "<i8"))
    assert support["flange_rows"] == [[3, 7, 71], [9, 11, 67]]
    json.dumps(result, allow_nan=False)


def test_explicit_execution_budgets_are_forwarded_unchanged(fixture):
    surface_limits = primitive.TriangleSurfaceBudgets(max_node_visits=1000)
    execution_limits = registration.SurfaceRegistrationBudgetsV2(max_query_calls=3)
    result = run(surface_budgets=surface_limits, registration_budgets=execution_limits)
    assert result["status"] == "PARTIAL"
    kwargs = fixture["register_args"][3]
    assert kwargs["surface_budgets"] is surface_limits
    assert kwargs["registration_budgets"] is execution_limits
    assert result["policies"]["registration_budgets"]["max_query_calls"] == 3


@pytest.mark.parametrize("status,has_features,expected", [
    ("AVAILABLE", False, "UNAVAILABLE"), ("UNAVAILABLE", False, "UNAVAILABLE"),
    ("UNAVAILABLE", True, "UNAVAILABLE"), ("PARTIAL", True, "PARTIAL")])
def test_native_refusal_phase_is_not_accepted_registration(fixture, status, has_features, expected):
    fixture["native"]["status"] = status
    fixture["native"]["features"] = (fixture["feature"],) if has_features else ()
    result = run()
    assert result["status"] == expected
    assert result["registration"]["status"] == "CONDITIONAL_BEST_FIT"
    if expected == "UNAVAILABLE":
        assert result["phase"] == "NATIVE_FEATURES"
        assert result["reasons"] == ["NATIVE_FEATURES_UNAVAILABLE"]
        assert "measure" not in fixture["events"]


@pytest.mark.parametrize("value", [False, 0, 1, "true", None, np.bool_(True)])
def test_native_opt_in_is_exact_boolean(fixture, value):
    assert run(allow_ocp=value)["reasons"] == ["EXPLICIT_OCP_OPT_IN_REQUIRED"]
    assert fixture["events"] == []


@pytest.mark.parametrize("field,value", [
    ("cad_bytes", b""), ("cad_bytes", bytearray(b"x")), ("cad_bytes", "input.step"),
    ("scan_bytes", b""), ("scan_bytes", bytearray(b"x")), ("scan_bytes", "input.ply"),
    ("workflow_policy", None), ("registration_policy", None), ("measurement_policy", None),
    ("instance_policy", None), ("surface_budgets", {}), ("registration_budgets", {})])
def test_malformed_admission_never_enters_native_or_loader(fixture, field, value):
    result = run(**{field: value})
    assert result["status"] == "UNAVAILABLE" and result["predictions"] == []
    assert fixture["events"] == []


@pytest.mark.parametrize("field,value", [
    ("max_registration_scan_rows", True), ("cad_sample_count", 35),
    ("cad_linear_deflection_mm", float("nan")), ("cad_angular_deflection_radians", 0.),
    ("native_adapter_version", "v2"), ("unit_assumption", "CALIBRATED_MM")])
def test_versioned_policy_keeps_legacy_admission(field, value):
    with pytest.raises(ValueError):
        workflow.MeshWorkflowPolicyV2(**{field: value})


@pytest.mark.parametrize("policy_name,field,value", [
    ("workflow_policy", "cad_sample_count", True),
    ("registration_policy", "trim_fraction", -1.),
    ("measurement_policy", "search_margin_mm", float("nan")),
    ("instance_policy", "min_local_points", False)])
def test_bypassed_frozen_policy_is_revalidated(fixture, policy_name, field, value):
    args = policies()
    object.__setattr__(args[policy_name], field, value)
    result = run(**args)
    assert result["status"] == "UNAVAILABLE" and fixture["events"] == []


@pytest.mark.parametrize("field,value", [("max_pair_batch", 0), ("leaf_size", 7),
    ("max_ties_per_query", 33), ("max_query_rows", True), ("max_triangle_pairs", -1)])
def test_invalid_surface_budget_is_rejected_before_native(fixture, field, value):
    result = run(surface_budgets=replace(primitive.TriangleSurfaceBudgets(), **{field: value}))
    assert result["status"] == "UNAVAILABLE" and fixture["events"] == []


@pytest.mark.parametrize("value", [-1, True, 4097])
def test_invalid_execution_budget_is_rejected_before_native(fixture, value):
    result = run(registration_budgets=registration.SurfaceRegistrationBudgetsV2(max_query_calls=value))
    assert result["status"] == "UNAVAILABLE" and fixture["events"] == []


@pytest.mark.parametrize("field", ["raw_xyz_sha256", "sample_xyz_sha256", "engine_to_raw_sha256", "scan_sha256"])
def test_bad_scan_digest_refuses_before_native(fixture, field):
    fixture["scan"]["source_bindings"][field] = "0" * 64
    result = run()
    assert result["phase"] == "INPUT_PREPARATION" and result["status"] == "UNAVAILABLE"
    assert fixture["events"] == ["prepare"]


def test_rebound_scan_sample_must_still_be_original_rows(fixture):
    fixture["scan"]["points"] = immutable(fixture["scan"]["points"] + .5)
    fixture["scan"]["source_bindings"]["sample_xyz_sha256"] = digest(fixture["scan"]["points"])
    assert run()["status"] == "UNAVAILABLE"
    assert fixture["events"] == ["prepare"]


@pytest.mark.parametrize("field", ["oriented_triangle_xyz_sha256", "source_face_ids_sha256",
    "source_triangle_ids_sha256", "sample_points_sha256", "sample_normals_sha256",
    "sample_triangle_rows_sha256", "sampling_policy_sha256"])
def test_bad_surface_digest_never_reaches_registration(fixture, field):
    fixture["cad"]["digests"][field] = "0" * 64
    result = run()
    assert result["status"] == "UNAVAILABLE" and result["phase"] == "CAD_SURFACE_SAMPLING"
    assert "register" not in fixture["events"] and result["measurements"] is None


@pytest.mark.parametrize("field", ["points", "normals", "triangles"])
def test_rebound_geometry_must_match_triangle_incidence(fixture, field):
    fixture["cad"][field] = immutable(fixture["cad"][field] + .01)
    key = {"points": "sample_points_sha256", "normals": "sample_normals_sha256",
           "triangles": "oriented_triangle_xyz_sha256"}[field]
    fixture["cad"]["digests"][key] = digest(fixture["cad"][field])
    assert run()["status"] == "UNAVAILABLE"
    assert "register" not in fixture["events"]


@pytest.mark.parametrize("field", ["sample_source_face_indices", "sample_source_triangle_indices",
                                  "sample_barycentric_weights"])
def test_unhashed_memberships_are_checked_directly(fixture, field):
    fixture["cad"][field] = immutable(fixture["cad"][field] + 1,
                                     "<f8" if field.endswith("weights") else "<i8")
    assert run()["status"] == "UNAVAILABLE" and "register" not in fixture["events"]


def test_readonly_flag_without_immutable_ownership_is_insufficient(fixture):
    points = fixture["cad"]["points"].copy()
    points.setflags(write=False)
    fixture["cad"]["points"] = points
    assert run()["status"] == "UNAVAILABLE" and "register" not in fixture["events"]


@pytest.mark.parametrize("stage,phase", [("scan", "INPUT_PREPARATION"),
    ("cad", "CAD_SURFACE_SAMPLING"), ("registration", "WHOLE_PART_REGISTRATION")])
def test_unavailable_stage_stops_downstream_with_unknown_count(fixture, stage, phase):
    fixture[stage].update(status="UNAVAILABLE", reasons=["INVENTED_REFUSAL"])
    result = run()
    assert result["phase"] == phase and result["status"] == "UNAVAILABLE"
    assert result["observed_bend_count"] is None and result["predictions"] == []
    assert "INVENTED_REFUSAL" in result["reasons"] and "measure" not in fixture["events"]


@pytest.mark.parametrize("module,name,phase", [
    (mesh_input, "prepare_mesh_scan_v1", "INPUT_PREPARATION"),
    (workflow, "_read_step_owned_snapshot", "NATIVE_IMPORT"),
    (native_v1, "extract_ocp_guided_features_v1", "NATIVE_FEATURES"),
    (surface, "sample_native_cad_surface_v2", "CAD_SURFACE_SAMPLING"),
    (registration, "register_whole_part_v2", "WHOLE_PART_REGISTRATION"),
    (local, "inspect_full_raw_windows_v1", "FULL_RAW_LOCAL_MEASUREMENT")])
def test_exception_has_correct_phase_no_partial_answer_or_library_text(fixture, monkeypatch, module, name, phase):
    def fail(*args, **kwargs):
        raise RuntimeError("ARBITRARY_NATIVE_OR_INPUT_MESSAGE_MUST_NOT_BE_PUBLISHED")
    monkeypatch.setattr(module, name, fail)
    result = run()
    assert result["status"] == "UNAVAILABLE" and result["phase"] == phase
    assert result["predictions"] == [] and result["measurements"] is None
    assert result["observed_bend_count"] is None
    assert result["reasons"][0].endswith(":RuntimeError")
    assert "ARBITRARY" not in json.dumps(result)


@pytest.mark.parametrize("transform", [None, np.eye(3).tolist(), np.zeros((4, 4)).tolist(),
    np.diag([-1., 1., 1., 1.]).tolist(), np.diag([2., 1., 1., 1.]).tolist(),
    np.full((4, 4), float("nan")).tolist()])
def test_accepted_but_invalid_transform_is_not_measured(fixture, transform):
    fixture["registration"]["transform_scan_to_cad"] = transform
    result = run()
    assert result["status"] == "UNAVAILABLE" and result["phase"] == "WHOLE_PART_REGISTRATION"
    assert "measure" not in fixture["events"]
    assert result["registration"]["transform_scan_to_cad"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad_endpoint", [None, [1, 2], [True, False, True], [1., 2., float("nan")]])
def test_bad_late_instance_clears_entire_publication(fixture, bad_endpoint):
    later = dict(fixture["measured"]["finite_support_instances"][0], line_end=bad_endpoint)
    fixture["measured"]["finite_support_instances"].append(later)
    result = run()
    assert result["status"] == "UNAVAILABLE" and result["phase"] == "FULL_RAW_LOCAL_MEASUREMENT"
    assert result["predictions"] == [] and result["measurements"] is None
    assert "observed_support_instance_count" not in result


@pytest.mark.parametrize("rows", [[1, 1], [-1, 2], [0, 72]])
def test_invalid_raw_support_clears_publication(fixture, rows):
    fixture["measured"]["features"][0]["raw_support"]["arc_rows"] = immutable(rows, "<i8")
    result = run()
    assert result["predictions"] == [] and result["measurements"] is None


def test_zero_measurements_remain_unavailable_not_zero_observed_bends(fixture):
    fixture["measured"].update(status="UNAVAILABLE", features=[], finite_support_instances=[])
    result = run()
    assert result["status"] == "UNAVAILABLE" and result["phase"] == "COMPLETE"
    assert result["observed_support_instance_count"] == 0 and result["observed_bend_count"] is None
    assert result["reasons"] == ["NO_FINITE_SCAN_SUPPORTED_PREDICTIONS"]


def test_unavailable_measurement_cannot_publish_instances(fixture):
    fixture["measured"]["status"] = "UNAVAILABLE"
    assert run()["measurements"] is None


def test_no_gold_case_pose_or_count_interface_and_no_v1_routing():
    names = set(inspect.signature(workflow.inspect_cad_guided_mesh_v2).parameters)
    assert names == {"scan_bytes", "cad_bytes", "workflow_policy", "registration_policy",
        "measurement_policy", "instance_policy", "surface_budgets", "registration_budgets", "allow_ocp"}
    assert workflow.MeshWorkflowPolicyV2 is not legacy.MeshWorkflowPolicyV1


def test_snapshot_wrapper_delegates_exact_bytes_to_frozen_importer(monkeypatch):
    token = object()
    calls = []
    def read(blob):
        calls.append(blob)
        return token
    monkeypatch.setattr(legacy, "_read_step_owned_snapshot", read)
    assert workflow._read_step_owned_snapshot(CAD_BYTES) is token
    assert len(calls) == 1 and calls[0] is CAD_BYTES
