"""Invented-input tests of exact engine/report source, without application imports.

The incumbent control executes only the existing dense-refinement construction
assignment. It is not an incumbent detector, alignment or end-to-end benchmark.
"""

import ast
import builtins
import copy
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT_SOURCE = ROOT / "backend" / "feature_detection" / "bend_detector.py"
PIPELINE_SOURCE = ROOT / "backend" / "bend_inspection_pipeline.py"
SEMANTICS_SOURCE = ROOT / "domains" / "bend" / "services" / "runtime_semantics.py"
KERNEL_SOURCE = ROOT / "backend" / "cad_guided_measurement.py"


def _source_namespace():
    namespace = {
        "__name__": "cad_guided_report_source_contract",
        "np": np,
        "dataclass": dataclass,
        "field": field,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    # This named module contains only typing, constants and pure semantic helpers.
    exec(compile(SEMANTICS_SOURCE.read_text(), str(SEMANTICS_SOURCE), "exec"), namespace)
    names = {
        "SourcePlaneSupport", "SourceBendCandidate", "PlaneSegment",
        "DetectedBend", "BendSpecification", "BendMatch", "BendInspectionReport",
    }
    tree = ast.parse(REPORT_SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert {node.name for node in selected} == names
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(REPORT_SOURCE), "exec"), namespace)
    return namespace


def _entrypoint(namespace, import_function):
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "inspect_cad_guided_offline"]
    assert len(selected) == 1
    local = dict(namespace)
    local["__builtins__"] = {**vars(builtins), "__import__": import_function}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(PIPELINE_SOURCE), "exec"), local)
    return local["inspect_cad_guided_offline"]


def _payload():
    return {
        "schema_version": 1,
        "scope": "CAD_GUIDED_OFFLINE_SINGLE_INNER_FACE",
        "status": "UNTRUSTED_REGISTRATION",
        "features": [],
        "conformance": "INDETERMINATE",
        "production_use_allowed": False,
        "observed_bend_count": None,
    }


def _kernel_module():
    """Load only the reviewed NumPy kernel source, never a backend initializer."""
    name = "cad_guided_measurement_direct_source_contract"
    module = ModuleType(name)
    module.__file__ = str(KERNEL_SOURCE)
    # dataclasses may consult the defining module while processing annotations.
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(KERNEL_SOURCE.read_text(), str(KERNEL_SOURCE), "exec"), module.__dict__)
    finally:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous
    return module


def _declared_feature(kernel):
    return kernel.CADGuidedFeature(
        feature_id="constructed-radius-only", nominal_radius_mm=5., nominal_sweep_deg=90.,
        origin=(0., 0., 0.), axis=(0., 0., 1.), start_radial=(1., 0., 0.),
        span_mm=(0., 20.), flange_window_mm=(4., 12.), search_margin_mm=2.,
        max_fit_residual_mm=.01, plane_exclusion_mm=.1,
        max_tangency_error_mm=.05, nominal_source="DECLARED",
    )


def _constructed_inner_face(actual_radius=6.):
    """Independent explicit circle and tangent-plane positions; no kernel calls."""
    z = np.linspace(0., 20., 9)
    distance = np.linspace(4., 12., 9)
    angles = np.deg2rad(np.linspace(10., 80., 25))
    arc = [[actual_radius * np.cos(theta), actual_radius * np.sin(theta), axial]
           for axial in z for theta in angles]
    first = [[actual_radius, -along, axial] for axial in z for along in distance]
    second = [[-along, actual_radius, axial] for axial in z for along in distance]
    return np.asarray(arc + first + second, dtype=np.float64)


def test_default_report_serialization_is_unchanged():
    report = _source_namespace()["BendInspectionReport"](part_id="legacy", matches=[])
    assert report.cad_guided_measurements is None
    assert report.to_dict() == {
        "part_id": "legacy",
        "summary": {
            "total_bends": 0, "countable_total_bends": 0, "detected": 0,
            "countable_completed_bends": 0, "passed": 0, "failed": 0,
            "warnings": 0, "progress_pct": 0, "completed_bends": 0,
            "completed_in_spec": 0, "completed_out_of_spec": 0,
            "remaining_bends": 0, "observed_bends": 0,
            "partially_observed_bends": 0, "unobserved_bends": 0,
            "total_features": 0, "detected_features": 0,
            "process_features": 0, "detected_process_features": 0,
        },
        "matches": [], "unmatched_detections": [], "scan_quality": {},
        "alignment": {}, "operator_actions": [], "processing_time_ms": 0,
    }


@pytest.mark.parametrize("status", ["PASS", "FAIL", "WARNING", "NOT_DETECTED"])
def test_default_nonempty_report_keeps_legacy_summary_contract(status):
    report_type = _source_namespace()["BendInspectionReport"]
    match = SimpleNamespace(
        cad_bend=SimpleNamespace(countable_in_regression=True, bend_id="legacy-bend"),
        status=status, action_item="legacy action", issue_location="bend_line",
        observability_state="UNKNOWN", physical_completion_state="UNKNOWN",
        to_dict=lambda: {"bend_id": "legacy-bend", "status": status},
    )
    report = report_type(part_id="legacy", matches=[match])
    output = report.to_dict()
    assert "scope" not in output and "features" not in output
    assert output["summary"]["total_bends"] == 1
    assert output["summary"]["detected"] == int(status != "NOT_DETECTED")
    assert output["summary"]["passed"] == int(status == "PASS")
    assert output["summary"]["failed"] == int(status == "FAIL")
    assert output["summary"]["warnings"] == int(status == "WARNING")
    assert output["matches"] == [{"bend_id": "legacy-bend", "status": status}]


def test_opt_in_report_bypasses_legacy_completion_summary():
    payload = _payload()
    report = _source_namespace()["BendInspectionReport"](
        part_id="constructed", matches=[], cad_guided_measurements=payload,
    )

    def forbidden():
        raise AssertionError("Offline measurement payload must not compute legacy completion")

    report.compute_summary = forbidden
    output = report.to_dict()
    assert output == {**payload, "part_id": "constructed"}
    assert "summary" not in output and "matches" not in output
    assert "part_id" not in payload
    assert output["observed_bend_count"] is None
    assert output["conformance"] == "INDETERMINATE"
    json.dumps(output, allow_nan=False)


@pytest.mark.parametrize("consumer", ["compute_summary", "progress_percentage", "to_table_string"])
def test_offline_measurements_explicitly_reject_legacy_summary_interfaces(consumer):
    report = _source_namespace()["BendInspectionReport"](
        part_id="constructed", matches=[], cad_guided_measurements=_payload(),
    )
    with pytest.raises(ValueError, match="to_dict"):
        value = getattr(report, consumer)
        if callable(value):
            value()


@pytest.mark.parametrize("consumer,offline_side", [
    ("_report_rank", "primary"),
    ("merge_bend_reports", "primary"),
    ("merge_bend_reports", "local"),
])
def test_exact_legacy_rank_and_merge_cannot_promote_an_offline_measurement_payload(consumer, offline_side):
    namespace = _source_namespace()
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == consumer)
    # The offline guard must reject before any detector/matcher/report-selection
    # dependency is reached; none of those dependencies is available here.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PIPELINE_SOURCE), "exec"), namespace)
    report_type = namespace["BendInspectionReport"]
    offline = report_type(part_id="offline", matches=[], cad_guided_measurements=_payload())
    legacy = report_type(part_id="legacy", matches=[])
    with pytest.raises(ValueError, match="to_dict"):
        if consumer == "_report_rank":
            namespace[consumer](offline, expected_bend_count=1)
        else:
            primary, local = (offline, legacy) if offline_side == "primary" else (legacy, offline)
            namespace[consumer](primary, local, part_id="constructed")


def test_exact_legacy_rank_keeps_default_tuple():
    namespace = _source_namespace()
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_report_rank")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PIPELINE_SOURCE), "exec"), namespace)
    report = namespace["BendInspectionReport"](part_id="legacy", matches=[])
    report.detected_count, report.pass_count, report.fail_count, report.warning_count = 3, 1, 1, 1
    assert namespace["_report_rank"](report, expected_bend_count=2) == (2, 1, -1, -1, -1)


def test_exact_legacy_empty_merge_keeps_default_report():
    namespace = _source_namespace()
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "merge_bend_reports")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PIPELINE_SOURCE), "exec"), namespace)
    report_type = namespace["BendInspectionReport"]
    result = namespace["merge_bend_reports"](
        report_type(part_id="primary", matches=[]), report_type(part_id="local", matches=[]),
        part_id="merged",
    )
    assert result.to_dict() == report_type(part_id="merged", matches=[]).to_dict()


@pytest.mark.parametrize("legacy_import", [False, True])
def test_entrypoint_is_lazy_and_passes_explicit_inputs_without_legacy_detection(legacy_import):
    namespace = _source_namespace()
    points = np.zeros((0, 3))
    features = (object(),)
    calls = []
    imports = []

    def inspect(actual_points, actual_features, *, registration_status):
        calls.append((actual_points, actual_features, registration_status))
        return _payload()

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        imports.append(name)
        if name == "backend.cad_guided_measurement":
            if legacy_import:
                raise ModuleNotFoundError("No module named backend", name="backend")
            return SimpleNamespace(inspect_cad_guided_scan=inspect)
        if legacy_import and name == "cad_guided_measurement":
            return SimpleNamespace(inspect_cad_guided_scan=inspect)
        raise AssertionError(f"Unexpected application/native import: {name}")

    entrypoint = _entrypoint(namespace, guarded_import)
    assert not imports
    report = entrypoint(points, features, "constructed", registration_status="EXPLICIT_TEST_FRAME")
    assert isinstance(report, namespace["BendInspectionReport"])
    assert report.matches == [] and report.unmatched_detections == []
    assert calls[0][0] is points and calls[0][1] is features
    assert calls[0][2] == "EXPLICIT_TEST_FRAME"
    assert imports == (["backend.cad_guided_measurement", "cad_guided_measurement"]
                       if legacy_import else ["backend.cad_guided_measurement"])
    assert report.to_dict()["part_id"] == "constructed"


def test_missing_kernel_dependency_is_not_hidden_by_import_fallback():
    imports = []

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        imports.append(name)
        raise ModuleNotFoundError("Missing dependency", name="numpy")

    entrypoint = _entrypoint(_source_namespace(), guarded_import)
    with pytest.raises(ModuleNotFoundError) as exc:
        entrypoint(np.zeros((0, 3)), (), "constructed")
    assert exc.value.name == "numpy"
    assert imports == ["backend.cad_guided_measurement"]


@pytest.mark.parametrize("registration_status", ["UNTRUSTED", "UNSUPPORTED_FRAME_CLAIM"])
def test_actual_kernel_unsupported_inputs_cannot_create_pass_or_count_claims(registration_status):
    kernel = _kernel_module()

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        assert name == "backend.cad_guided_measurement"
        return kernel

    entrypoint = _entrypoint(_source_namespace(), guarded_import)
    report = entrypoint(np.zeros((0, 3)), (_declared_feature(kernel),), "unsupported",
                        registration_status=registration_status)
    output = report.to_dict()
    assert output["scope"] == "CAD_GUIDED_OFFLINE_SINGLE_INNER_FACE"
    assert output["conformance"] == "INDETERMINATE"
    assert output["production_use_allowed"] is False
    assert output["observed_bend_count"] is None
    assert "summary" not in output and "passed" not in output and "failed" not in output
    assert report.matches == [] and report.unmatched_detections == []
    assert output["features"][0]["measurements"] is None
    assert output["features"][0]["conformance"] == "INDETERMINATE"
    json.dumps(output, allow_nan=False)


def _incumbent_dense_construction(namespace, cad_bend, measured_angle):
    """Execute one exact incumbent assignment, not the surrounding pipeline."""
    tree = ast.parse(PIPELINE_SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "refine_dense_scan_bends_via_alignment")
    assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "detected" for target in node.targets)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                   and node.value.func.id == "DetectedBend"]
    assert len(assignments) == 1
    local = dict(namespace)
    plane = namespace["PlaneSegment"](
        plane_id=1, normal=np.array([1., 0., 0.]), d=0., centroid=np.zeros(3),
        points=np.empty((0, 3)), boundary_points=np.empty((0, 3)), area=0., inlier_count=0,
    )
    local.update(cad_bend=cad_bend, measured_angle=measured_angle,
                 confidence=.9, plane1=plane, plane2=plane, point_count=100)
    exec(compile(ast.Module(body=[copy.deepcopy(assignments[0])], type_ignores=[]),
                 str(PIPELINE_SOURCE), "exec"), local)
    return local["detected"]


@pytest.mark.parametrize("nominal_radius,actual_radius", [(1., 1.5), (3., 4.), (6., 9.)])
def test_exact_incumbent_assignment_copies_nominal_radius_and_endpoints(nominal_radius, actual_radius):
    namespace = _source_namespace()
    cad = namespace["BendSpecification"](
        bend_id="constructed", target_angle=90., target_radius=nominal_radius,
        bend_line_start=np.array([0., 0., -10.]), bend_line_end=np.array([0., 0., 10.]),
    )
    detected = _incumbent_dense_construction(namespace, cad, measured_angle=90.)
    # Independently stipulated actual radius is deliberately different. The
    # current assignment has no measured-radius input, so it reports zero delta.
    assert actual_radius != nominal_radius
    assert detected.measured_radius == nominal_radius
    assert detected.measured_radius - cad.target_radius == 0.
    assert abs(detected.measured_radius - actual_radius) == abs(nominal_radius - actual_radius)
    np.testing.assert_array_equal(detected.bend_line_start, cad.bend_line_start)
    np.testing.assert_array_equal(detected.bend_line_end, cad.bend_line_end)


@pytest.mark.parametrize("fit_model", ["free_circle", "measured_flange_tangent"])
def test_connected_actual_kernel_reports_radius_change_missed_by_incumbent_assignment(fit_model):
    kernel = _kernel_module()
    namespace = _source_namespace()
    feature = replace(_declared_feature(kernel), fit_model=fit_model)
    actual_radius = 6.
    scan = _constructed_inner_face(actual_radius)
    original_scan = scan.copy()

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        assert name == "backend.cad_guided_measurement"
        return kernel

    entrypoint = _entrypoint(namespace, guarded_import)
    report = entrypoint(scan, (feature,), "constructed-radius-only",
                        registration_status="DECLARED_ALIGNED")
    output = report.to_dict()
    row = output["features"][0]
    assert row["status"] == "MEASURED_CONDITIONAL", row
    measured = row["measurements"]
    assert measured["fit_model"] == fit_model
    assert measured["tangency_check_role"] == (
        "ENFORCED_CONSTRAINT_NOT_INDEPENDENT_VALIDATION"
        if fit_model == "measured_flange_tangent"
        else "FREE_CIRCLE_MODEL_COMPATIBILITY_CHECK"
    )
    assert measured["measured_radius_mm"] == pytest.approx(actual_radius, abs=1e-8)
    assert measured["radius_deviation_mm"] == pytest.approx(1., abs=1e-8)
    assert measured["measured_sweep_deg"] == pytest.approx(90., abs=1e-8)
    assert measured["physical_trim_endpoints"] is None
    assert measured["physical_extent_deviation_mm"] is None
    assert row["physical_completion_state"] == "UNKNOWN"
    assert row["uncertainty_status"] == "UNCALIBRATED"
    assert row["conformance"] == output["conformance"] == "INDETERMINATE"
    assert output["observed_bend_count"] is None
    assert output["production_use_allowed"] is False
    assert "summary" not in output and report.matches == []
    np.testing.assert_array_equal(scan, original_scan)
    json.dumps(output, allow_nan=False)

    # Paired radius-only control of the exact old construction assignment. Its
    # scan-angle input is stipulated correctly, isolating the copied-radius bug.
    # This is NOT execution or evaluation of the full incumbent pipeline.
    cad = namespace["BendSpecification"](
        bend_id=feature.feature_id, target_angle=90., target_radius=5.,
        bend_line_start=np.array([0., 0., 0.]), bend_line_end=np.array([0., 0., 20.]),
    )
    incumbent = _incumbent_dense_construction(namespace, cad, measured_angle=90.)
    incumbent_error = abs(incumbent.measured_radius - actual_radius)
    challenger_error = abs(measured["measured_radius_mm"] - actual_radius)
    assert incumbent.measured_radius - cad.target_radius == 0.
    assert incumbent_error == 1.
    assert challenger_error < 1e-8


def test_connected_missing_flange_stays_unavailable_not_defective():
    kernel = _kernel_module()
    scan = _constructed_inner_face()
    # The independent generator appends the second 9-by-9 flange last.
    scan = scan[:-81]

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        assert name == "backend.cad_guided_measurement"
        return kernel

    entrypoint = _entrypoint(_source_namespace(), guarded_import)
    report = entrypoint(scan, (_declared_feature(kernel),), "constructed-partial",
                        registration_status="DECLARED_ALIGNED")
    output = report.to_dict()
    row = output["features"][0]
    assert row["status"] == "UNAVAILABLE"
    assert row["measurements"] is None
    assert "INSUFFICIENT_FLANGE_SUPPORT" in row["reasons"]
    assert row["physical_completion_state"] == "UNKNOWN"
    assert row["conformance"] == output["conformance"] == "INDETERMINATE"
    assert output["measured_conditional_feature_count"] == 0
    assert output["observed_bend_count"] is None
    assert output["production_use_allowed"] is False
    assert "summary" not in output and report.matches == []
    json.dumps(output, allow_nan=False)
