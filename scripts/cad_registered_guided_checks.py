"""Fixed invented datum registration -> native CAD -> scan fit -> finite score.

Run only inside the named bounded native worker's fresh cache. No application
package is imported and no existing CAD/scan file is read. Separate reference
surface datums are supplied explicitly; their acquisition/segmentation is not
tested here. Scanner poses and per-row point correspondences are NEVER inputs
to the registration or measurement code. All points come from constructed
analytic geometry, not native tessellation or extracted feature parameters.

Frozen before execution: 54 positives (3 sweeps x 3 deviations x 3 independent
scanner poses x 2 datum coverages), and 12 fail-closed controls (4 kinds x 3
poses). Positive radius, sweep, recovered point positions and finite support
endpoints must be within 2e-6 mm/degrees. Finite lines are measured cylinder-axis
SAMPLED SUPPORT, not physical trims or manufacturing-event evidence. No policy,
fixture or denominator rescue follows a failed run. This is not corpus accuracy.
"""

import ast
import builtins
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
import itertools
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple


GEOMETRY_ABSOLUTE_TOLERANCE_MM = 2e-6
ANGLE_ABSOLUTE_TOLERANCE_DEG = 2e-6
RADIUS_MM = 5.
AXIAL_LENGTH_MM = 30.
SWEEPS_DEG = (60., 90., 135.)
DEVIATIONS = ("nominal", "radius_plus_1mm", "sweep_plus_5deg")
SCANNER_POSES = ("translated", "oblique", "axial_flip")
DATUM_COVERAGES = ("complete", "partial_three_planes")
POSITIVE_MATRIX = tuple(itertools.product(SWEEPS_DEG, DEVIATIONS, SCANNER_POSES,
                                         DATUM_COVERAGES))
NEGATIVE_KINDS = ("arc_unobserved", "central_axial_gap", "missing_z_datum", "symmetric_cube_datums")
NEGATIVE_MATRIX = tuple(itertools.product(NEGATIVE_KINDS, SCANNER_POSES))
LINE_CONVENTION = "CYLINDER_CENTER_AXIS_SUPPORT"


def _isolated_cache():
    cache = Path(os.environ["TMPDIR"]).resolve()
    root = Path(__file__).resolve().parents[1]
    if (cache != Path.cwd().resolve() or cache != Path(os.environ["HOME"]).resolve()
            or cache == root or root in cache.parents):
        raise RuntimeError("Root-owned isolated cache required")
    return cache


def _scanner_pose(np, name):
    """Truth-only scan creation; return maps CAD positions INTO scanner frame."""
    if name == "translated":
        return np.eye(3), np.array([17., -19., 11.])
    if name == "axial_flip":
        return np.diag([1., -1., -1.]), np.array([-33., 24., 42.])
    assert name == "oblique"
    axis = np.array([2., -1., 3.]) / math.sqrt(14.)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = 1.23
    rotation = (math.cos(angle) * np.eye(3) + (1. - math.cos(angle)) * np.outer(axis, axis)
                + math.sin(angle) * cross)
    assert np.max(np.abs(rotation @ rotation.T - np.eye(3))) < 1e-14
    assert abs(np.linalg.det(rotation) - 1.) < 1e-14
    return rotation, np.array([31., -47., 19.])


def _datum_surfaces(np, mode):
    """Separate stable surfaces, not bend nominals or coded point fiducials.

    CAD-side reference includes positions and surface normals. Scan-side output
    contains only positions, without per-row reference identity or normals.
    Partial removes one quarter of each surface's rows without moving points.
    """
    if mode == "symmetric_cube_datums":
        center, half_side = np.array([-60., -70., -80.]), 12.
        interior = np.linspace(-10., 10., 7)
        patches, normals = [], []
        for axis in range(3):
            tangents = [index for index in range(3) if index != axis]
            for sign in (-1., 1.):
                rows = []
                for first, second in itertools.product(interior, repeat=2):
                    point = center.copy()
                    point[axis] += sign * half_side
                    point[tangents[0]] += first
                    point[tangents[1]] += second
                    rows.append(point)
                patches.append(np.asarray(rows))
                normals.append(np.tile(sign * np.eye(3)[axis], (len(rows), 1)))
        reference = np.vstack(patches)
        return reference, np.vstack(normals), reference.copy()
    patches = [
        np.array([[-40., y, z] for y in np.linspace(-30., -8., 12)
                  for z in np.linspace(-5., 23., 15)]),
        np.array([[x, -50., z] for x in np.linspace(-18., 16., 18)
                  for z in np.linspace(8., 20., 7)]),
        np.array([[x, y, -30.] for x in np.linspace(-32., 7., 14)
                  for y in np.linspace(-13., 6., 10)]),
    ]
    normals = [np.tile(-np.eye(3)[index], (len(rows), 1))
               for index, rows in enumerate(patches)]
    # Deficiency is in both supplied reference and scan datum surfaces. Even an
    # exact point-to-point fit must not create the unobserved surface constraint.
    if mode == "missing_z_datum":
        patches, normals = patches[:2], normals[:2]
    assert mode in ("complete", "partial_three_planes", "missing_z_datum")
    observed = [rows[np.arange(len(rows)) % 4 != 0] if mode == "partial_three_planes" else rows
                for rows in patches]
    return np.vstack(patches), np.vstack(normals), np.vstack(observed)


def _bend_scan(np, sweep, deviation, control=None):
    """Independent scan positions; no CAD adapter output enters this oracle."""
    radius = RADIUS_MM + (1. if deviation == "radius_plus_1mm" else 0.)
    actual_sweep = sweep + (5. if deviation == "sweep_plus_5deg" else 0.)
    phi = math.radians(actual_sweep)
    axial = np.linspace(0., AXIAL_LENGTH_MM, 9)
    if control == "central_axial_gap":
        axial = axial[(axial <= 7.5) | (axial >= 22.5)]
    distance = np.linspace(3., 12., 17)
    arc = [] if control == "arc_unobserved" else [
        [radius * math.cos(theta), radius * math.sin(theta), z]
        for theta in np.linspace(0., phi, 81) for z in axial]
    first = [[radius, -along, z] for along in distance for z in axial]
    last = [[radius * math.cos(phi) - along * math.sin(phi),
             radius * math.sin(phi) + along * math.cos(phi), z]
            for along in distance for z in axial]
    return np.asarray(arc + first + last, dtype=np.float64), radius, actual_sweep


def _observed_cloud(np, bend, datum_positions, pose):
    source = np.vstack((bend, datum_positions))
    is_datum = np.arange(len(source)) >= len(bend)
    rotation, translation = _scanner_pose(np, pose)
    observed = source @ rotation.T + translation
    # Fixed nonsemantic ordering, applied jointly to all acquired positions.
    # Datum segmentation is carried as row indices, NOT datum correspondence.
    order = np.r_[np.arange(1, len(source), 2), np.arange(0, len(source), 2)][::-1]
    return observed[order], np.flatnonzero(is_datum[order]), source[order]


def _native_bend(np, sweep):
    """Independent closed CCW annular-sector strip extruded to one solid."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec

    phi, thickness, length = math.radians(sweep), 1., 15.

    def polar(radius, angle):
        return np.array([radius * math.cos(angle), radius * math.sin(angle), 0.])

    def point(value):
        return gp_Pnt(*map(float, value))

    def line(first, second):
        maker = BRepBuilderAPI_MakeEdge(point(first), point(second))
        assert maker.IsDone()
        return maker.Edge()

    def arc(radius, start, stop):
        curve = GC_MakeArcOfCircle(point(polar(radius, start)),
                                  point(polar(radius, (start + stop) / 2.)), point(polar(radius, stop)))
        assert curve.IsDone()
        maker = BRepBuilderAPI_MakeEdge(curve.Value())
        assert maker.IsDone()
        return maker.Edge()

    inner0, inner1 = polar(RADIUS_MM, 0.), polar(RADIUS_MM, phi)
    outer0, outer1 = polar(RADIUS_MM + thickness, 0.), polar(RADIUS_MM + thickness, phi)
    first_ray, last_ray = np.array([0., -1., 0.]), np.array([-math.sin(phi), math.cos(phi), 0.])
    edges = [line(outer0 + length * first_ray, outer0), arc(RADIUS_MM + thickness, 0., phi),
             line(outer1, outer1 + length * last_ray),
             line(outer1 + length * last_ray, inner1 + length * last_ray),
             line(inner1 + length * last_ray, inner1), arc(RADIUS_MM, phi, 0.),
             line(inner0, inner0 + length * first_ray),
             line(inner0 + length * first_ray, outer0 + length * first_ray)]
    wire = BRepBuilderAPI_MakeWire()
    for edge in edges:
        wire.Add(edge)
        assert wire.IsDone()
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0., 0., 0.), gp_Dir(0., 0., 1.)), wire.Wire(), True)
    assert face.IsDone()
    prism = BRepPrimAPI_MakePrism(face.Face(), gp_Vec(0., 0., AXIAL_LENGTH_MM), True, True)
    assert prism.IsDone()
    shape = prism.Shape()
    assert shape.ShapeType() == TopAbs_SOLID and BRepCheck_Analyzer(shape).IsValid()
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    expected_volume = (.5 * ((RADIUS_MM + thickness)**2 - RADIUS_MM**2) * phi
                       + 2. * length * thickness) * AXIAL_LENGTH_MM
    assert abs(properties.Mass() - expected_volume) <= 1e-9 * expected_volume
    return shape


def _point_set_error(np, actual, expected):
    distance = np.linalg.norm(actual[:, None, :] - expected[None, :, :], axis=2)
    return float(max(np.max(np.min(distance, axis=0)), np.max(np.min(distance, axis=1))))


@contextmanager
def _source_modules(np):
    """Import complete engine modules; AST selects only legacy report/wrapper."""
    import importlib
    root = Path(__file__).resolve().parents[1]
    instances = importlib.import_module("backend.cad_guided_instances")
    adapter = importlib.import_module("backend.cad_guided_native_features")
    registration = importlib.import_module("backend.cad_guided_registration")
    evaluation = importlib.import_module("backend.cad_guided_evaluation")
    importlib.import_module("backend.cad_guided_offline")
    namespace = dict(__name__=__name__, np=np, dataclass=dataclass, field=field,
                     Any=Any, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple,
                     BendMatch=Any, DetectedBend=Any)
    report_source = root / "backend" / "feature_detection" / "bend_detector.py"
    selected = [node for node in ast.parse(report_source.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == "BendInspectionReport"]
    assert len(selected) == 1
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(report_source), "exec"), namespace)
    pipeline_source = root / "backend" / "bend_inspection_pipeline.py"
    selected = [node for node in ast.parse(pipeline_source.read_text()).body
                if isinstance(node, ast.FunctionDef)
                and node.name == "inspect_registered_native_cad_guided_offline"]
    assert len(selected) == 1
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(pipeline_source), "exec"), namespace)
    yield SimpleNamespace(adapter=adapter, registration=registration, instances=instances,
                          evaluation=evaluation,
                          entrypoint=namespace["inspect_registered_native_cad_guided_offline"])


def run_checks():
    _isolated_cache()
    import numpy as np
    import OCP

    results, counts, maxima = [], Counter(), Counter()

    def error(name, value):
        value = float(value)
        limit = ANGLE_ABSOLUTE_TOLERANCE_DEG if name.endswith("_deg") else GEOMETRY_ABSOLUTE_TOLERANCE_MM
        assert math.isfinite(value) and value <= limit, (name, value)
        maxima[name] = max(maxima[name], value)
        return value

    def safety(output):
        assert output["production_use_allowed"] is False
        assert output["observed_bend_count"] is None
        assert output["conformance"] == "INDETERMINATE"
        assert "summary" not in output and "matches" not in output
        assert "passed" not in output and "failed" not in output
        assert output["reference_frame"] == "CAD_REFERENCE"
        assert output["unit"] == "mm" and output["line_convention"] == LINE_CONVENTION
        assert output["endpoint_semantics"] == "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM"
        assert output["observed_support_instance_count"] == len(output["finite_support_instances"])
        registration = output["registration"]
        assert registration["evidence"]["candidate_search_status"] == "SEARCH_BOUNDED_NOT_GLOBAL"
        assert registration["evidence"]["uncertainty_status"] == "UNCALIBRATED"
        for row in output["features"]:
            assert row["uncertainty_status"] == "UNCALIBRATED"
            assert row["physical_completion_state"] == "UNKNOWN"
        json.dumps(output, allow_nan=False)

    with _source_modules(np) as modules:
        registration_policy = modules.registration.RegistrationPolicy()
        measurement_policy = modules.adapter.NativeGuidedPolicy(
            flange_window_mm=(2.5, 12.5), search_margin_mm=3., max_fit_residual_mm=.005,
            plane_exclusion_mm=.075, max_tangency_error_mm=.1,
            fit_model="measured_flange_tangent")
        instance_policy = modules.instances.SupportPolicy(
            max_axial_gap_mm=4., max_angular_gap_deg=2., min_arc_sweep_deg=20., min_local_points=12)
        # Evaluator thresholds are a separate fixed contract from the much
        # tighter fixture geometry oracle; neither is an inference input.
        thresholds = modules.evaluation.Thresholds()
        reference_records = [{"bend_id": "independent-truth:1", "line_start": [0., 0., 0.],
                              "line_end": [0., 0., AXIAL_LENGTH_MM]}]
        matrix = [("positive", item) for item in POSITIVE_MATRIX]
        matrix += [("negative", item) for item in NEGATIVE_MATRIX]
        for index, (kind, item) in enumerate(matrix):
            record = dict(case_id=f"registered-native:{index}", kind=kind, parameters=item)
            stage = "PREPARE_INDEPENDENT_NATIVE_AND_SCAN_GEOMETRY"
            try:
                if kind == "positive":
                    sweep, deviation, pose, datum_mode = item
                    control = None
                else:
                    control, pose = item
                    sweep, deviation = 90., "radius_plus_1mm"
                    datum_mode = control if control in ("missing_z_datum", "symmetric_cube_datums") else "complete"
                retained = SimpleNamespace(success=True, native_shape=_native_bend(np, sweep), native_binding="ocp")
                bend, actual_radius, actual_sweep = _bend_scan(np, sweep, deviation, control)
                cad_datums, cad_normals, scan_datums = _datum_surfaces(np, datum_mode)
                scan, datum_indices, oracle_positions = _observed_cloud(np, bend, scan_datums, pose)
                assert len(scan) < 200_000 and len(datum_indices) >= 36
                counts["generated_scan_rows"] += len(scan)
                record.update(scan_rows=len(scan), reference_datum_rows=len(cad_datums),
                              scanned_datum_rows=len(datum_indices))
                before = (scan.copy(), cad_datums.copy(), cad_normals.copy(), datum_indices.copy())
                stage = "ACTUAL_REGISTERED_ENTRY_AND_REPORT"
                # The known creation transform, per-row oracle positions and
                # reference truth records are intentionally absent from inputs.
                output = modules.entrypoint(
                    retained, scan, f"invented-registered:{index}",
                    cad_datum_points=cad_datums, cad_datum_normals=cad_normals,
                    scan_datum_row_indices=datum_indices, registration_policy=registration_policy,
                    measurement_policy=measurement_policy, instance_policy=instance_policy,
                    unit_status="DECLARED_MM", allow_ocp=True).to_dict()
                safety(output)
                assert all(np.array_equal(actual, original) for actual, original in
                           zip((scan, cad_datums, cad_normals, datum_indices), before))
                registration = output["registration"]
                record["registration_status"] = registration["status"]
                record["registration_reasons"] = registration["reasons"]
                record["generated_pose_count"] = registration["evidence"]["candidate_count"]
                registration_negative = control in ("missing_z_datum", "symmetric_cube_datums")
                stage = "VERIFY_RECOVERED_REGISTRATION_WITH_INDEPENDENT_TRUTH"
                if registration_negative:
                    assert registration["status"] == "UNAVAILABLE", registration["reasons"]
                    assert registration["transform_scan_to_cad"] is None
                    assert output["features"] == [] and output["measured_conditional_feature_count"] == 0
                    if control == "symmetric_cube_datums":
                        assert "AMBIGUOUS_GENERATED_POSES" in registration["reasons"]
                    counts["registration_abstentions"] += 1
                else:
                    assert registration["status"] == "ACCEPTED_BOUNDED", registration["reasons"]
                    transform = np.asarray(registration["transform_scan_to_cad"])
                    recovered = scan @ transform[:3, :3].T + transform[:3, 3]
                    record["registration_position_error_mm"] = error(
                        "registration_position_error_mm", np.linalg.norm(recovered - oracle_positions, axis=1).max())
                    assert output["datum_input_row_count"] == len(datum_indices)
                    assert output["inspection_input_row_count"] == len(bend)
                    assert output["datum_selection_status"] == "EXPLICIT_STABLE_REGION_SELECTION_NOT_AUTOMATIC"
                    counts["accepted_bounded_registrations"] += 1
                stage = "VERIFY_MEASURED_GEOMETRY_AND_FINITE_SUPPORT"
                if kind == "positive":
                    assert output["status"] == "COMPLETE"
                    assert output["measured_conditional_feature_count"] == 1
                    assert output["instance_extraction_status"] == "COMPLETE"
                    assert output["observed_support_instance_count"] == 1
                    measurement = output["features"][0]["measurements"]
                    record["measurement_errors"] = {
                        "radius_error_mm": error("radius_error_mm", abs(measurement["measured_radius_mm"] - actual_radius)),
                        "radius_deviation_error_mm": error("radius_deviation_error_mm", abs(
                            measurement["radius_deviation_mm"] - (actual_radius - RADIUS_MM))),
                        "sweep_error_deg": error("sweep_error_deg", abs(abs(measurement["measured_sweep_deg"]) - actual_sweep)),
                        "sweep_deviation_error_deg": error("sweep_deviation_error_deg", abs(
                            measurement["sweep_deviation_deg"] - (actual_sweep - sweep) * math.copysign(
                                1., output["features"][0]["nominal_sweep_deg"]))),
                    }
                    instance = output["finite_support_instances"][0]
                    endpoints = np.asarray([instance["line_start"], instance["line_end"]])
                    record["support_endpoint_error_mm"] = error("support_endpoint_error_mm", _point_set_error(
                        np, endpoints, np.array([[0., 0., 0.], [0., 0., AXIAL_LENGTH_MM]])))
                    assert instance["manufacturing_event_identity"] == "UNKNOWN"
                    assert instance["physical_trim_endpoints"] is None
                    assert instance["continuous_surface_coverage"] == "NOT_ESTABLISHED"
                    assert measurement["physical_trim_endpoints"] is None
                    assert measurement["physical_extent_deviation_mm"] is None
                    counts["positive_finite_instances"] += 1
                    counts[f"positive_{deviation}"] += 1
                    counts[f"positive_{datum_mode}"] += 1
                else:
                    assert output["instance_extraction_status"] == "UNAVAILABLE"
                    assert output["finite_support_instances"] == []
                    assert output["observed_support_instance_count"] == 0
                    # Gap rejection cannot be credited to a missing measurement.
                    if control == "central_axial_gap":
                        assert output["measured_conditional_feature_count"] == 1
                        assert output["features"][0]["finite_support"]["status"] == "UNAVAILABLE"
                        assert "DISCONNECTED_AXIAL_SUPPORT" in output["features"][0]["finite_support"]["reasons"]
                    counts["expected_abstentions"] += 1
                stage = "ACTUAL_FINITE_REFERENCE_EVALUATOR"
                score = modules.evaluation.score_support_case(
                    case_id=record["case_id"], reference_records=reference_records,
                    prediction_payload=output, reference_frame="CAD_REFERENCE", reference_unit="mm",
                    reference_convention=LINE_CONVENTION, thresholds=thresholds)
                record["evaluation"] = score
                # Both unavailable capture and failed registration remain a
                # selected evaluation case. A recall penalty is not a defect.
                assert score["true_positive"] == (1 if kind == "positive" else 0)
                assert score["false_positive"] == 0 and score["false_negative"] == (0 if kind == "positive" else 1)
                assert score["exact_count"] is (kind == "positive")
                assert score["exact_localized_target_set"] is (kind == "positive")
                counts["evaluated_cases"] += 1
                record["status"] = "PASS"
            except Exception as exc:
                trace = exc.__traceback__
                while trace is not None and trace.tb_next is not None:
                    trace = trace.tb_next
                record.update(status="FAIL", failure_stage=stage, failure_type=type(exc).__name__,
                              failure_message=str(exc)[:1000],
                              failure_function=None if trace is None else trace.tb_frame.f_code.co_name,
                              failure_source_line=None if trace is None else trace.tb_lineno)
            results.append(record)
        # Never silently summarize the subset that reached the evaluator.
        # Failed stage records remain in the matrix; an incomplete scored set
        # suppresses its aggregate instead of shortening the denominator.
        aggregate = None
        if all("evaluation" in record for record in results):
            aggregate = modules.evaluation.score_support_cases(
                [record["evaluation"] for record in results],
                expected_case_ids=[record["case_id"] for record in results])
    assert len(results) == 66
    failed = sum(record["status"] != "PASS" for record in results)
    return dict(status="FAIL" if failed else "PASS", matrix_cases=len(results), failed_cases=failed,
                counts=dict(counts), maximum_errors=dict(maxima),
                fixed_geometry_tolerance_mm=GEOMETRY_ABSOLUTE_TOLERANCE_MM,
                fixed_angle_tolerance_deg=ANGLE_ABSOLUTE_TOLERANCE_DEG, ocp_version=OCP.__version__,
                pose_input="NO_TRANSFORM_OR_POINT_CORRESPONDENCE_SUPPLIED_TO_INFERENCE",
                datum_scope="EXPLICIT_SEPARATE_STABLE_PLANAR_SURFACE_REGIONS",
                truth_origin="INDEPENDENT_NATIVE_CROSS_SECTION_AND_ANALYTIC_SCAN_POSITIONS",
                reference_line_convention=LINE_CONVENTION,
                endpoint_semantics="SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM",
                units_status="DECLARED_MM_NOT_CALIBRATED", manufacturing_event_count="NOT_CLAIMED",
                corpus_accuracy="NOT_TESTED", support_evaluation_aggregate=aggregate, cases=results)
