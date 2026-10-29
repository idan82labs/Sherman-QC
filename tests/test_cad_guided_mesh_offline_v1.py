"""Invented byte-to-workflow glue tests with mocked native/pose boundaries.

No application initialization, native CAD reader, registration search, raw files,
Gold, or physical benchmark is executed by these tests. The analytic checks use
the real mesh byte parser and full-row measurement wrapper; mocked known poses
test only transform plumbing, not pose recovery. Root owns bounded execution.
"""
from dataclasses import replace
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


_PACKAGE = "_cad_guided_mesh_offline_v1_tests"
_package = types.ModuleType(_PACKAGE)
_package.__path__ = [str(Path(__file__).resolve().parents[1] / "backend")]
sys.modules[_PACKAGE] = _package
workflow = importlib.import_module(_PACKAGE + ".cad_guided_mesh_offline_v1")
mesh_input = importlib.import_module(_PACKAGE + ".cad_guided_mesh_input_v1")
registration = importlib.import_module(_PACKAGE + ".cad_guided_global_registration_v1")
native_r0 = importlib.import_module(_PACKAGE + ".cad_guided_native_features")
native_v1 = importlib.import_module(_PACKAGE + ".cad_guided_native_features_v1")
sampler = importlib.import_module(_PACKAGE + ".cad_guided_cad_sampling_v1")
local = importlib.import_module(_PACKAGE + ".cad_guided_local_measurement_v1")


CAD_BYTES = b"INVENTED_STEP_SNAPSHOT_FOR_MOCKED_READER_ONLY\x00\xff"
ROTATION = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
TRANSLATION = np.array([7., -4., 3.])
TRANSFORM = np.eye(4)
TRANSFORM[:3, :3] = ROTATION
TRANSFORM[:3, 3] = TRANSLATION


def policies(branch="v1"):
    return dict(
        workflow_policy=workflow.MeshWorkflowPolicyV1(
            max_registration_scan_rows=36, cad_sample_count=36,
            native_adapter_version=branch),
        registration_policy=registration.WholePartRegistrationPolicyV1(
            coarse_points=36, refine_points=36, coarse_iterations=1, refine_iterations=1),
        measurement_policy=native_r0.NativeGuidedPolicy(
            flange_window_mm=(2.5, 12.5), search_margin_mm=3.,
            max_fit_residual_mm=.05, plane_exclusion_mm=.075,
            max_tangency_error_mm=.1, fit_model="free_circle"),
        instance_policy=local.SupportPolicy(4., 5., 20., 3),
    )


def ply_bytes(points):
    """One invented mesh face is consumed; vertices remain exact ordered rows."""
    lines = ["ply", "format ascii 1.0", "comment invented workflow-only fixture",
             f"element vertex {len(points)}", "property double x", "property double y",
             "property double z", "element face 1", "property list uchar int vertex_indices",
             "end_header"]
    lines.extend(" ".join(format(float(value), ".17g") for value in point) for point in points)
    lines.append("3 0 1 2")
    return ("\n".join(lines) + "\n").encode("ascii")


def generic_raw_points():
    i = np.arange(72, dtype=float)
    return np.column_stack((i % 6, (i // 6) % 4, i // 24))


def analytic_fixture():
    """CAD declares R5/90 degrees; scan has a tangent R6/95-degree extrusion."""
    radius, phi = 6., math.radians(95.)
    stations = np.linspace(0., 30., 9)
    arc = np.array([[radius * math.cos(a), radius * math.sin(a), z]
                    for a in np.linspace(0., phi, 81) for z in stations])
    contacts = (np.array([radius, 0., 0.]), radius * np.array([math.cos(phi), math.sin(phi), 0.]))
    rays = (np.array([0., -1., 0.]), np.array([-math.sin(phi), math.cos(phi), 0.]))
    flanges = [np.array([contact + d * ray + [0., 0., z]
                        for d in np.linspace(3., 12., 17) for z in stations])
               for contact, ray in zip(contacts, rays)]
    cad_frame_points = np.vstack((arc, *flanges))
    raw = (cad_frame_points - TRANSLATION) @ ROTATION
    declared = native_r0.CADGuidedFeature(
        feature_id="native-local-window:0", nominal_radius_mm=5., nominal_sweep_deg=90.,
        origin=(0., 0., 0.), axis=(0., 0., 1.), start_radial=(1., 0., 0.),
        span_mm=(0., 30.), flange_window_mm=(2.5, 12.5), search_margin_mm=3.,
        max_fit_residual_mm=.05, plane_exclusion_mm=.075, max_tangency_error_mm=.1,
        nominal_source="invented native extraction stub, not manufacturing identity",
    )
    return raw, cad_frame_points, declared


def install_boundaries(monkeypatch, *, features=None, real_measurement=False):
    """Patch source module attributes resolved by workflow's local imports."""
    events, calls = [], {}
    shape = object()
    native_features = (object(),) if features is None else tuple(features)
    sample_points = generic_raw_points()[:36].copy()
    sample_normals = np.tile([0., 0., 1.], (36, 1))

    def read(blob):
        events.append("read")
        calls.setdefault("read", []).append(blob)
        assert blob is CAD_BYTES
        return shape, {"target_unit": "mm", "physical_calibration": "NOT_ESTABLISHED"}

    def extractor(branch):
        def extract(got_shape, **kwargs):
            events.append("extract:" + branch)
            calls["extract"] = (got_shape, kwargs)
            assert got_shape is shape
            assert kwargs["allow_ocp"] is True
            assert kwargs["unit_status"] == "DECLARED_MM"
            return {"status": "AVAILABLE" if native_features else "UNAVAILABLE",
                    "features": native_features,
                    "metadata": {"stub_branch": branch, "manufacturing_event_identity": "UNKNOWN"}}
        return extract

    def sample(got_shape, **kwargs):
        events.append("sample")
        calls["sample"] = (got_shape, kwargs)
        assert got_shape is shape
        return {"status": "AVAILABLE", "reasons": [], "points": sample_points,
                "normals": sample_normals, "sample_count": 36,
                "physical_metrology_coverage": "NOT_ESTABLISHED"}

    def register(scan_points, cad_points, cad_normals, **kwargs):
        events.append("register")
        calls["register"] = (scan_points.copy(), cad_points, cad_normals, kwargs)
        assert cad_points is sample_points
        assert cad_normals is sample_normals
        return {"status": "CONDITIONAL_BEST_FIT", "reasons": [],
                "transform_scan_to_cad": TRANSFORM.tolist(),
                "physical_pose_identity": "NOT_VERIFIED"}

    real_local = local.inspect_full_raw_windows_v1

    def measure(points, got_features, **kwargs):
        events.append("measure")
        calls["measure"] = (points.copy(), got_features, kwargs)
        assert got_features is native_features
        if real_measurement:
            return real_local(points, got_features, **kwargs)
        payload = {"status": "PARTIAL", "features": [{
            "feature_id": "invented-window", "raw_support": {
                "arc_rows": np.array([0, 2, 4], dtype="<i8"),
                "flange_rows": (np.array([1, 3, 5], dtype="<i8"),
                                np.array([6, 7, 8], dtype="<i8")),
            },
        }], "finite_support_instances": [{
            "bend_id": "invented-window", "line_start": [4., 5., 6.], "line_end": [4., 5., 16.],
            "axis_support_line_start": [1., 2., 6.], "axis_support_line_end": [1., 2., 16.],
            "line_convention": "MEASURED_INNER_ARC_MID_ANGLE_GENERATOR",
            "physical_trim_endpoints": None, "manufacturing_event_identity": "UNKNOWN",
        }], "observed_bend_count": None}
        calls["measured_payload"] = payload
        return payload

    monkeypatch.setattr(workflow, "_read_step_owned_snapshot", read)
    monkeypatch.setattr(native_r0, "extract_ocp_guided_features", extractor("r0"))
    monkeypatch.setattr(native_v1, "extract_ocp_guided_features_v1", extractor("v1"))
    monkeypatch.setattr(sampler, "sample_native_cad_surface_v1", sample)
    monkeypatch.setattr(registration, "register_whole_part_v1", register)
    monkeypatch.setattr(local, "inspect_full_raw_windows_v1", measure)
    return events, calls


def run(*, scan_blob=None, branch="v1", **overrides):
    arguments = {"scan_bytes": ply_bytes(generic_raw_points()) if scan_blob is None else scan_blob,
                 "cad_bytes": CAD_BYTES, "allow_ocp": True, **policies(branch)}
    arguments.update(overrides)
    return workflow.inspect_cad_guided_mesh_v1(**arguments)


def assert_not_qualified(result):
    assert result["observed_bend_count"] is None
    assert result["production_use_allowed"] is False
    assert result["conformance"] == "INDETERMINATE"
    assert result["uncertainty_status"] == "UNCALIBRATED"
    assert result["units_status"] == "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"
    assert result["datum_status"] == "NO_VERIFIED_STABLE_DATUM"
    assert result["reference_frame_compatibility"] == "NOT_ASSERTED_BY_INFERENCE"


@pytest.mark.parametrize("branch", ["r0", "v1"])
def test_single_exact_byte_import_same_shape_and_only_extractor_branch_changes(monkeypatch, branch):
    events, calls = install_boundaries(monkeypatch)
    blob = ply_bytes(generic_raw_points())
    result = run(scan_blob=blob, branch=branch)
    assert events == ["read", "extract:" + branch, "sample", "register", "measure"]
    assert calls["read"] == [CAD_BYTES]
    assert calls["extract"][0] is calls["sample"][0]
    assert result["cad_sha256"] == hashlib.sha256(CAD_BYTES).hexdigest()
    assert result["input_preparation"]["source_bindings"]["scan_sha256"] == hashlib.sha256(blob).hexdigest()
    assert result["native_features"]["stub_branch"] == branch
    assert result["policies"]["workflow"]["native_adapter_version"] == branch
    assert calls["sample"][1] == {
        "allow_ocp": True, "native_binding": "ocp", "sample_count": 36,
        "linear_deflection_mm": .1, "angular_deflection_radians": .2,
    }
    assert calls["register"][3]["unit_assumption"] == "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"
    assert_not_qualified(result)


def test_registration_receives_bounded_sample_measurement_receives_every_raw_row(monkeypatch):
    _, calls = install_boundaries(monkeypatch)
    raw = generic_raw_points()
    blob = ply_bytes(raw)
    expected = mesh_input.prepare_mesh_scan_v1(scan_bytes=blob, max_engine_rows=36)
    result = run(scan_blob=blob)
    assert expected["status"] == "AVAILABLE"
    assert 12 <= len(calls["register"][0]) <= 36 < len(raw)
    np.testing.assert_array_equal(calls["register"][0], expected["points"])
    np.testing.assert_allclose(calls["measure"][0], raw @ ROTATION.T + TRANSLATION, atol=1e-14, rtol=0)
    assert len(calls["measure"][0]) == 72
    assert calls["measure"][2]["scan_sha256"] == hashlib.sha256(blob).hexdigest()
    assert calls["measure"][2]["registration_status"] == "CONDITIONAL_BEST_FIT"
    assert result["input_preparation"]["coverage"]["raw_vertex_count"] == 72
    assert result["input_preparation"]["sample_cloud_contract"] == "REGISTRATION_AND_PROPOSALS_ONLY"
    assert "points" not in result["cad_surface_sampling"]
    assert "normals" not in result["cad_surface_sampling"]


def test_inverse_transform_applies_to_all_four_endpoint_fields_without_mutating_measurements(monkeypatch):
    install_boundaries(monkeypatch)
    result = run()
    assert result["status"] == "PARTIAL"
    assert result["phase"] == "COMPLETE"
    raw_prediction = result["predictions"][0]
    cad_instance = result["measurements"]["finite_support_instances"][0]
    for field in ("line_start", "line_end", "axis_support_line_start", "axis_support_line_end"):
        expected = (np.asarray(cad_instance[field]) - TRANSLATION) @ ROTATION
        np.testing.assert_allclose(raw_prediction[field], expected, atol=1e-14, rtol=0)
        # Forward mapping is an independent round-trip check of the inverse.
        np.testing.assert_allclose(np.asarray(raw_prediction[field]) @ ROTATION.T + TRANSLATION,
                                   cad_instance[field], atol=1e-14, rtol=0)
    assert cad_instance["line_start"] == [4., 5., 6.]
    assert raw_prediction["reference_frame"] == "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"
    assert result["observed_support_instance_count"] == 1
    assert_not_qualified(result)


def test_successful_output_is_strict_json_with_explicit_raw_support_and_digests(monkeypatch):
    _, calls = install_boundaries(monkeypatch)
    result = run()
    decoded = json.loads(json.dumps(result, allow_nan=False))
    support = decoded["measurements"]["features"][0]["raw_support"]
    original = calls["measured_payload"]["features"][0]["raw_support"]
    assert support["arc_rows"] == [0, 2, 4]
    assert support["flange_rows"] == [[1, 3, 5], [6, 7, 8]]
    assert support["arc_rows_sha256"] == hashlib.sha256(original["arc_rows"].tobytes()).hexdigest()
    assert support["flange_rows_sha256"] == [hashlib.sha256(rows.tobytes()).hexdigest()
                                            for rows in original["flange_rows"]]
    assert support["digest_encoding"] == "LITTLE_ENDIAN_INT64_ORIGINAL_RAW_ROW_INDICES"
    assert isinstance(original["arc_rows"], np.ndarray)
    assert all(isinstance(rows, np.ndarray) for rows in original["flange_rows"])


@pytest.mark.parametrize("branch", ["r0", "v1"])
def test_real_parser_and_full_raw_fitter_measure_deviation_under_mocked_pose(monkeypatch, branch):
    raw, cad_points, declared = analytic_fixture()
    events, calls = install_boundaries(monkeypatch, features=(declared,), real_measurement=True)
    result = run(scan_blob=ply_bytes(raw), branch=branch)
    assert events == ["read", "extract:" + branch, "sample", "register", "measure"]
    assert result["phase"] == "COMPLETE"
    assert result["status"] == "PARTIAL"
    # Exercise the full actual measurement payload, not only a shortened stub.
    decoded = json.loads(json.dumps(result, allow_nan=False))
    assert decoded["measurements"]["features"][0]["raw_support"]["arc_rows"]
    assert len(calls["register"][0]) <= 36 < len(calls["measure"][0])
    assert len(calls["measure"][0]) == len(raw) == 1035
    np.testing.assert_allclose(calls["measure"][0], cad_points, atol=2e-14, rtol=0)
    row = result["measurements"]["features"][0]
    assert row["status"] == "MEASURED_CONDITIONAL"
    assert row["measurements"]["radius_deviation_mm"] == pytest.approx(1., abs=1e-7)
    assert row["measurements"]["sweep_deviation_deg"] == pytest.approx(5., abs=1e-8)
    assert len(result["predictions"]) == 1
    prediction = result["predictions"][0]
    expected_cad_offset = 6. * np.array([math.cos(math.radians(47.5)), math.sin(math.radians(47.5)), 0.])
    for point_field, axis_field in (("line_start", "axis_support_line_start"),
                                   ("line_end", "axis_support_line_end")):
        observed_raw_offset = np.asarray(prediction[point_field]) - prediction[axis_field]
        np.testing.assert_allclose(observed_raw_offset, expected_cad_offset @ ROTATION, atol=1e-7, rtol=0)
    assert_not_qualified(result)


@pytest.mark.parametrize("stage,phase", [
    ("prepare", "INPUT_PREPARATION"), ("read", "NATIVE_IMPORT"),
    ("extract", "NATIVE_FEATURES"), ("sample", "CAD_SURFACE_SAMPLING"),
    ("register", "WHOLE_PART_REGISTRATION"), ("measure", "FULL_RAW_LOCAL_MEASUREMENT"),
])
def test_stage_exception_retains_phase_and_never_publishes_predictions(monkeypatch, stage, phase):
    events, _ = install_boundaries(monkeypatch)

    def fail(*_args, **_kwargs):
        raise RuntimeError("invented internal failure; not a defect or trusted report text")

    target, name = {
        "prepare": (mesh_input, "prepare_mesh_scan_v1"),
        "read": (workflow, "_read_step_owned_snapshot"),
        "extract": (native_v1, "extract_ocp_guided_features_v1"),
        "sample": (sampler, "sample_native_cad_surface_v1"),
        "register": (registration, "register_whole_part_v1"),
        "measure": (local, "inspect_full_raw_windows_v1"),
    }[stage]
    monkeypatch.setattr(target, name, fail)
    result = run()
    assert result["status"] == "UNAVAILABLE"
    assert result["phase"] == phase
    assert result["predictions"] == []
    assert result["reasons"] == [
        ("INPUT_PREPARATION_EXCEPTION:" if stage == "prepare" else "WORKFLOW_EXCEPTION:") + "RuntimeError"]
    assert "invented internal failure" not in str(result["reasons"])
    if stage == "prepare":
        assert events == []
    assert_not_qualified(result)


def test_invalid_mesh_bytes_stop_before_native_import(monkeypatch):
    events, _ = install_boundaries(monkeypatch)
    result = run(scan_blob=b"not a PLY\n")
    assert result["phase"] == "INPUT_PREPARATION"
    assert result["status"] == "UNAVAILABLE"
    assert result["predictions"] == []
    assert result["reasons"][0] == "MESH_INPUT_UNAVAILABLE"
    assert events == []


def test_malformed_preparation_receipt_cannot_escape_or_enter_native(monkeypatch):
    events, _ = install_boundaries(monkeypatch)
    monkeypatch.setattr(mesh_input, "prepare_mesh_scan_v1", lambda **_kwargs: {"status": "AVAILABLE"})
    result = run()
    assert result["status"] == "UNAVAILABLE"
    assert result["phase"] == "INPUT_PREPARATION"
    assert result["reasons"] == ["INPUT_PREPARATION_EXCEPTION:KeyError"]
    assert result["predictions"] == []
    assert events == []


def test_empty_native_feature_set_still_records_sampling_and_registration(monkeypatch):
    events, _ = install_boundaries(monkeypatch, features=())
    result = run()
    assert events == ["read", "extract:v1", "sample", "register"]
    assert result["registration"]["status"] == "CONDITIONAL_BEST_FIT"
    assert result["reasons"] == ["NATIVE_FEATURES_UNAVAILABLE"]
    assert result["measurements"] is None
    assert result["predictions"] == []


def test_unavailable_cad_sample_does_not_call_registration(monkeypatch):
    events, _ = install_boundaries(monkeypatch)
    monkeypatch.setattr(sampler, "sample_native_cad_surface_v1", lambda *_args, **_kwargs:
                        {"status": "UNAVAILABLE", "reasons": ["INVENTED_SAMPLING_REFUSAL"]})
    result = run()
    assert result["phase"] == "CAD_SURFACE_SAMPLING"
    assert result["reasons"] == ["CAD_SURFACE_SAMPLING_UNAVAILABLE"]
    assert result["cad_surface_sampling"]["reasons"] == ["INVENTED_SAMPLING_REFUSAL"]
    assert "register" not in events and "measure" not in events
    assert result["predictions"] == []


def test_unavailable_registration_does_not_measure_or_emit_nominal_lines(monkeypatch):
    events, _ = install_boundaries(monkeypatch)
    monkeypatch.setattr(registration, "register_whole_part_v1", lambda *_args, **_kwargs:
                        {"status": "UNAVAILABLE", "reasons": ["INVENTED_AMBIGUITY"]})
    result = run()
    assert result["phase"] == "WHOLE_PART_REGISTRATION"
    assert result["reasons"] == ["PROVISIONAL_REGISTRATION_UNAVAILABLE", "INVENTED_AMBIGUITY"]
    assert "measure" not in events
    assert result["predictions"] == []


def test_measurement_refusals_finish_without_substituting_native_feature_count(monkeypatch):
    install_boundaries(monkeypatch)
    monkeypatch.setattr(local, "inspect_full_raw_windows_v1", lambda *_args, **_kwargs:
                        {"status": "UNAVAILABLE", "features": [], "finite_support_instances": [],
                         "expected_window_count": 7, "measured_conditional_window_count": 0,
                         "observed_bend_count": None})
    result = run()
    assert result["phase"] == "COMPLETE"
    assert result["status"] == "UNAVAILABLE"
    assert result["reasons"] == ["NO_FINITE_SCAN_SUPPORTED_PREDICTIONS"]
    assert result["predictions"] == []
    assert result["observed_support_instance_count"] == 0
    assert result["observed_bend_count"] is None


def test_bad_second_instance_cannot_publish_successful_prefix(monkeypatch):
    install_boundaries(monkeypatch)
    first = {field: [1., 2., 3.] for field in
             ("line_start", "line_end", "axis_support_line_start", "axis_support_line_end")}
    monkeypatch.setattr(local, "inspect_full_raw_windows_v1", lambda *_args, **_kwargs:
                        {"features": [], "finite_support_instances": [first, {"line_start": [4., 5., 6.]}]})
    result = run()
    assert result["status"] == "UNAVAILABLE"
    assert result["phase"] == "FULL_RAW_LOCAL_MEASUREMENT"
    assert result["predictions"] == []
    assert result["reasons"] == ["WORKFLOW_EXCEPTION:KeyError"]
    assert first["line_start"] == [1., 2., 3.]


@pytest.mark.parametrize("opt_in", [False, None, 1, "true"])
def test_native_opt_in_is_strict_boolean_before_any_reader(monkeypatch, opt_in):
    events, _ = install_boundaries(monkeypatch)
    result = run(allow_ocp=opt_in)
    assert result["phase"] == "ADMISSION"
    assert result["reasons"] == ["EXPLICIT_OCP_OPT_IN_REQUIRED"]
    assert events == []


@pytest.mark.parametrize("cad", [b"", bytearray(b"not exact immutable bytes"), "path.step", None])
def test_cad_requires_nonempty_owned_bytes_not_path_or_mutable_buffer(monkeypatch, cad):
    events, _ = install_boundaries(monkeypatch)
    result = run(cad_bytes=cad)
    assert result["reasons"] == ["CAD_BYTE_BUDGET"]
    assert events == []


@pytest.mark.parametrize("name", ["workflow_policy", "registration_policy", "measurement_policy", "instance_policy"])
def test_workflow_rejects_untyped_policies_before_native(monkeypatch, name):
    events, _ = install_boundaries(monkeypatch)
    result = run(**{name: {"looks_valid": True}})
    expected = "VALIDATED_WORKFLOW_POLICY_REQUIRED" if name == "workflow_policy" else "VALIDATED_INFERENCE_POLICIES_REQUIRED"
    assert result["reasons"] == [expected]
    assert events == []


@pytest.mark.parametrize("field,value", [
    ("max_registration_scan_rows", 35), ("max_registration_scan_rows", 200001),
    ("max_registration_scan_rows", True), ("cad_sample_count", 35),
    ("cad_sample_count", 100001), ("cad_sample_count", 36.),
    ("cad_linear_deflection_mm", 0.), ("cad_linear_deflection_mm", float("nan")),
    ("cad_linear_deflection_mm", True), ("cad_angular_deflection_radians", 1.1),
    ("native_adapter_version", "legacy"), ("native_adapter_version", 1),
    ("unit_assumption", "VERIFIED_MM"), ("unit_assumption", "mm"),
])
def test_policy_requires_strict_budgets_adapter_and_provisional_units(field, value):
    with pytest.raises(ValueError):
        replace(workflow.MeshWorkflowPolicyV1(), **{field: value})


def test_public_workflow_signature_has_no_truth_identity_or_known_pose_inputs():
    parameters = inspect.signature(workflow.inspect_cad_guided_mesh_v1).parameters
    assert set(parameters) == {"scan_bytes", "cad_bytes", "workflow_policy", "registration_policy",
                               "measurement_policy", "instance_policy", "allow_ocp"}
    assert all(value.kind == inspect.Parameter.KEYWORD_ONLY for value in parameters.values())


@pytest.mark.parametrize("forbidden", ["gold", "reference_lines", "case_id", "part_id", "known_pose", "expected_bend_count"])
def test_truth_or_identity_keyword_is_not_accepted(forbidden):
    with pytest.raises(TypeError):
        run(**{forbidden: "not an inference input"})
