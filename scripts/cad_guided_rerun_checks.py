"""Fixed invented CAD/PLY/datum files -> normally imported offline engine.

Run only in the root bounded native worker's own isolated cache. The prior
registered script supplies invented-shape/analytic-point helpers only; its AST
runtime loader is NEVER called. All input, engine and evaluator modules use
normal backend namespace imports. No existing part, application, corpus,
incumbent, truth file or email input is read.

Frozen matrix: 12 positives = STEP/BREP x ASCII/little-endian/big-endian PLY x
independent +1 mm radius/+5 degree sweep deviation, all under oblique pose and
partial three-plane datum coverage. Four input-admission controls remain in the
16-case denominator. Policies and 2e-6 mm/degree analytic error limits are fixed
before the first run; failures do not justify smaller fixtures or relaxed gates.
"""
import hashlib
import importlib.util
import importlib.machinery
import itertools
import json
import math
import os
from pathlib import Path
import tempfile


CAD_FORMATS = ("step", "brep")
PLY_ENCODINGS = ("ascii", "binary_little_endian", "binary_big_endian")
DEVIATIONS = ("radius_plus_1mm", "sweep_plus_5deg")
POSITIVE_MATRIX = tuple(itertools.product(CAD_FORMATS, PLY_ENCODINGS, DEVIATIONS))
NEGATIVE_MATRIX = (
    ("stale_scan_binding", "DATUM_SOURCE_BINDING_MISMATCH"),
    ("wrong_datum_frame", "UNSUPPORTED_DATUM_DECLARATIONS"),
    ("unsupported_face_element", "UNSUPPORTED_PLY_ELEMENTS"),
    ("malformed_cad", "NATIVE_CAD_IMPORT_UNAVAILABLE"),
)
MATRIX_CASE_IDS = tuple(f"rerun-input:{index}" for index in range(16))
GEOMETRY_ABSOLUTE_TOLERANCE_MM = 2e-6
ANGLE_ABSOLUTE_TOLERANCE_DEG = 2e-6
LINE_CONVENTION = "CYLINDER_CENTER_AXIS_SUPPORT"


def _isolated_cache():
    cache = Path(os.environ["TMPDIR"]).resolve()
    root = Path(__file__).resolve().parents[1]
    if (cache != Path.cwd().resolve() or cache != Path(os.environ["HOME"]).resolve()
            or cache == root or root in cache.parents):
        raise RuntimeError("Root-owned isolated cache required")
    return cache


def _fixture_helpers():
    source = Path(__file__).resolve().parent / "cad_registered_guided_checks.py"
    class FixtureSource(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            return compile(self.get_data(self.path), self.path, "exec", dont_inherit=True)

    name = "invented_rerun_fixture_helpers"
    spec = importlib.util.spec_from_file_location(name, source, loader=FixtureSource(name, str(source)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cad(shape, path, format_):
    from OCP.BRepTools import BRepTools
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    if format_ == "step":
        writer = STEPControl_Writer()
        assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
        assert writer.Write(str(path)) == IFSelect_RetDone
    else:
        assert format_ == "brep"
        assert BRepTools.Write_s(shape, str(path)) is True
    # This is an after-write acceptance check. Native allocation/write execution
    # is separately bounded by the fixed parent worker, not by this byte check.
    assert 0 < path.stat().st_size <= 8 * 1024**2
    return path.read_bytes()


def _ply_bytes(np, points, encoding, *, face_element=False):
    header = ("ply\nformat " + encoding + " 1.0\n"
              "comment invented adapter-control positions, raw row order retained\n"
              f"element vertex {len(points)}\n"
              "property double z\nproperty double x\nproperty double y\nproperty uchar red\n")
    if face_element:
        header += "element face 0\nproperty list uchar int vertex_indices\n"
    header += "end_header\n"
    colors = np.arange(len(points)) % 251
    if encoding == "ascii":
        body = "".join(f"{row[2]:.17g} {row[0]:.17g} {row[1]:.17g} {int(color)}\n"
                       for row, color in zip(points, colors)).encode("ascii")
    else:
        assert encoding in ("binary_little_endian", "binary_big_endian")
        endian = "<" if encoding == "binary_little_endian" else ">"
        records = np.empty(len(points), dtype=[("z", endian + "f8"), ("x", endian + "f8"),
                                               ("y", endian + "f8"), ("red", "u1")])
        records["z"], records["x"], records["y"] = points[:, 2], points[:, 0], points[:, 1]
        records["red"] = colors
        body = records.tobytes()
    blob = header.encode("ascii") + body
    assert 0 < len(blob) <= 64 * 1024**2
    return blob


def _input_abstention(reasons):
    """Scoring adapter for failed input admission, explicitly NOT engine output."""
    return {
        "status": "UNAVAILABLE", "instance_extraction_status": "UNAVAILABLE",
        "finite_support_instances": [], "observed_support_instance_count": 0,
        "observed_bend_count": None, "reference_frame": "CAD_REFERENCE", "unit": "mm",
        "line_convention": LINE_CONVENTION, "endpoint_semantics": "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM",
        "production_use_allowed": False, "conformance": "INDETERMINATE", "reasons": list(reasons),
        "prediction_payload_source": "INPUT_ADMISSION_ABSTENTION_NOT_ENGINE_RUN",
    }


def run_checks(control_policies=None):
    cache = _isolated_cache()
    import numpy as np
    import OCP
    # Actual normal imports: no AST or module-source substitution for runtime.
    from backend.cad_guided_case_input import load_registered_case
    from backend.cad_guided_offline import inspect_registered_native
    from backend.cad_guided_native_features import NativeGuidedPolicy
    from backend.cad_guided_registration import RegistrationPolicy
    from backend.cad_guided_instances import SupportPolicy
    from backend.cad_guided_evaluation import Thresholds, score_support_case, score_support_cases

    helpers = _fixture_helpers()
    registration_policy = RegistrationPolicy()
    measurement_policy = NativeGuidedPolicy(
        flange_window_mm=(2.5, 12.5), search_margin_mm=3., max_fit_residual_mm=.005,
        plane_exclusion_mm=.075, max_tangency_error_mm=.1, fit_model="measured_flange_tangent")
    instance_policy = SupportPolicy(
        max_axial_gap_mm=4., max_angular_gap_deg=2., min_arc_sweep_deg=20., min_local_points=12)
    thresholds = Thresholds()
    if control_policies is not None:
        # The fixed frozen-control worker supplies these from its captured
        # policy module. No CLI policy override or physical input is accepted.
        from dataclasses import asdict
        expected = {"registration_policy": registration_policy,
                    "measurement_policy": measurement_policy,
                    "instance_policy": instance_policy, "thresholds": thresholds}
        assert set(control_policies) == set(expected)
        assert all(asdict(control_policies[key]) == asdict(value) for key, value in expected.items()), "R0 policy differs from the fixed rehearsal"
        registration_policy = control_policies["registration_policy"]
        measurement_policy = control_policies["measurement_policy"]
        instance_policy = control_policies["instance_policy"]
        thresholds = control_policies["thresholds"]
    reference_records = [{"bend_id": "independent-truth:1", "line_start": [0., 0., 0.],
                          "line_end": [0., 0., 30.]}]
    matrix = [("positive", item) for item in POSITIVE_MATRIX]
    matrix += [("negative", item) for item in NEGATIVE_MATRIX]
    assert len(matrix) == len(MATRIX_CASE_IDS) == 16
    results, maxima = [], {}
    engine_calls = 0

    def error(name, value):
        value = float(value)
        limit = ANGLE_ABSOLUTE_TOLERANCE_DEG if name.endswith("_deg") else GEOMETRY_ABSOLUTE_TOLERANCE_MM
        assert math.isfinite(value) and value <= limit, (name, value)
        maxima[name] = max(maxima.get(name, 0.), value)
        return value

    with tempfile.TemporaryDirectory(prefix="invented-rerun-cases-", dir=str(cache)) as directory:
        owned = Path(directory)
        for index, (kind, item) in enumerate(matrix):
            record = {"case_id": MATRIX_CASE_IDS[index], "kind": kind, "parameters": item,
                      "engine_called": False}
            stage = "PREPARE_INDEPENDENT_INVENTED_FILES"
            try:
                if kind == "positive":
                    format_, encoding, deviation = item
                    control = expected_input_reason = None
                else:
                    control, expected_input_reason = item
                    format_, encoding, deviation = "step", "ascii", "radius_plus_1mm"
                cad_path = owned / f"invented_{index}.{format_}"
                scan_path = owned / f"invented_{index}.ply"
                datum_path = owned / f"invented_{index}.json"
                if control == "malformed_cad":
                    cad_bytes = b"INVENTED INVALID CAD\n"
                    cad_path.write_bytes(cad_bytes)
                else:
                    cad_bytes = _write_cad(helpers._native_bend(np, 90.), cad_path, format_)
                bend, actual_radius, actual_sweep = helpers._bend_scan(np, 90., deviation)
                cad_datums, cad_normals, scan_datums = helpers._datum_surfaces(np, "partial_three_planes")
                scan, datum_indices, oracle_positions = helpers._observed_cloud(np, bend, scan_datums, "oblique")
                scan_bytes = _ply_bytes(np, scan, encoding, face_element=control == "unsupported_face_element")
                scan_path.write_bytes(scan_bytes)
                datum = {
                    "schema_version": 1, "scan_sha256": hashlib.sha256(scan_bytes).hexdigest(),
                    "cad_sha256": hashlib.sha256(cad_bytes).hexdigest(),
                    "reference_frame": "CAD_REFERENCE", "unit": "mm",
                    "selection_method": "EXPLICIT_STABLE_REGION_SELECTION",
                    "cad_datum_points": cad_datums.tolist(), "cad_datum_normals": cad_normals.tolist(),
                    "scan_datum_row_indices": datum_indices.tolist(),
                }
                if control == "stale_scan_binding":
                    datum["scan_sha256"] = "0" * 64
                if control == "wrong_datum_frame":
                    datum["reference_frame"] = "SCANNER_FRAME"
                datum_bytes = json.dumps(datum, allow_nan=False, sort_keys=True).encode("utf-8")
                assert 0 < len(datum_bytes) <= 2 * 1024**2
                datum_path.write_bytes(datum_bytes)
                record.update(cad_bytes=len(cad_bytes), scan_bytes=len(scan_bytes),
                              datum_bytes=len(datum_bytes), input_scan_rows=len(scan),
                              input_datum_rows=len(datum_indices))
                stage = "NORMAL_IMPORT_FILE_INPUT_ADAPTER"
                loaded = load_registered_case(cad_path=cad_path, scan_path=scan_path,
                                              datum_path=datum_path, allow_ocp=True)
                record["input_status"] = loaded["status"]
                record["input_reasons"] = loaded["reasons"]
                assert loaded["production_use_allowed"] is False
                if kind == "negative":
                    assert loaded["status"] == "UNAVAILABLE", loaded["reasons"]
                    assert loaded["reasons"] == [expected_input_reason], loaded["reasons"]
                    assert "cad_result" not in loaded and "scan_input" not in loaded
                    output = _input_abstention(loaded["reasons"])
                    record["prediction_payload_source"] = "INPUT_ADMISSION_ABSTENTION_NOT_ENGINE_RUN"
                    assert record["engine_called"] is False
                else:
                    assert loaded["status"] == "AVAILABLE", loaded["reasons"]
                    retained, snapshot = loaded["cad_result"], loaded["scan_input"]
                    assert retained.success is True and retained.native_binding == "ocp"
                    assert retained.native_shape is not None
                    assert np.array_equal(snapshot.scan_points, scan), "PLY decoding changed row order or double values"
                    assert np.array_equal(snapshot.scan_datum_row_indices, datum_indices)
                    assert np.array_equal(snapshot.cad_datum_points, cad_datums)
                    assert np.array_equal(snapshot.cad_datum_normals, cad_normals)
                    assert all(not values.flags.writeable for values in (
                        snapshot.scan_points, snapshot.cad_datum_points,
                        snapshot.cad_datum_normals, snapshot.scan_datum_row_indices))
                    binding = loaded["source_bindings"]
                    assert binding["scan_sha256"] == hashlib.sha256(scan_bytes).hexdigest()
                    assert binding["cad_sha256"] == hashlib.sha256(cad_bytes).hexdigest()
                    assert binding["datum_sha256"] == hashlib.sha256(datum_bytes).hexdigest()
                    assert binding["binding_status"] == "VERIFIED_SNAPSHOT_BYTES_ONLY"
                    assert binding["datum_rigidity_and_frame_status"] == "DECLARED_NOT_PHYSICALLY_VERIFIED"
                    stage = "NORMAL_IMPORT_REGISTERED_NATIVE_ENGINE"
                    record["engine_called"] = True
                    engine_calls += 1
                    output = inspect_registered_native(
                        retained, **snapshot.inference_arguments(), registration_policy=registration_policy,
                        measurement_policy=measurement_policy, instance_policy=instance_policy, allow_ocp=True)
                    record["prediction_payload_source"] = "ACTUAL_NORMALLY_IMPORTED_ENGINE"
                    stage = "VERIFY_INDEPENDENT_REGISTRATION_GEOMETRY_AND_FINITE_SUPPORT"
                    assert output["status"] == "COMPLETE", output.get("reasons")
                    assert output["registration_status"] == "ACCEPTED_BOUNDED"
                    assert output["registration"]["evidence"]["candidate_search_status"] == "SEARCH_BOUNDED_NOT_GLOBAL"
                    transform = np.asarray(output["registration"]["transform_scan_to_cad"])
                    recovered = scan @ transform[:3, :3].T + transform[:3, 3]
                    record["registration_position_error_mm"] = error(
                        "registration_position_error_mm", np.linalg.norm(recovered - oracle_positions, axis=1).max())
                    assert output["datum_input_row_count"] == len(datum_indices)
                    assert output["inspection_input_row_count"] == len(bend)
                    assert output["measured_conditional_feature_count"] == 1
                    assert output["observed_support_instance_count"] == 1
                    assert output["instance_extraction_status"] == "COMPLETE"
                    assert len(output["features"]) == len(output["finite_support_instances"]) == 1
                    row = output["features"][0]
                    measurement = row["measurements"]
                    record["measurement_errors"] = {
                        "radius_error_mm": error("radius_error_mm", abs(measurement["measured_radius_mm"] - actual_radius)),
                        "radius_deviation_error_mm": error("radius_deviation_error_mm", abs(
                            measurement["radius_deviation_mm"] - (actual_radius - 5.))),
                        "sweep_error_deg": error("sweep_error_deg", abs(abs(measurement["measured_sweep_deg"]) - actual_sweep)),
                        "sweep_deviation_error_deg": error("sweep_deviation_error_deg", abs(
                            measurement["sweep_deviation_deg"] - (actual_sweep - 90.)
                            * math.copysign(1., row["nominal_sweep_deg"]))),
                    }
                    instance = output["finite_support_instances"][0]
                    endpoints = np.asarray([instance["line_start"], instance["line_end"]])
                    record["support_endpoint_error_mm"] = error("support_endpoint_error_mm", helpers._point_set_error(
                        np, endpoints, np.array([[0., 0., 0.], [0., 0., 30.]])))
                    assert instance["physical_trim_endpoints"] is None
                    assert instance["manufacturing_event_identity"] == "UNKNOWN"
                    assert instance["continuous_surface_coverage"] == "NOT_ESTABLISHED"
                    assert measurement["physical_extent_deviation_mm"] is None
                    assert output["uncertainty_status"] == "UNCALIBRATED"
                assert output["production_use_allowed"] is False and output["observed_bend_count"] is None
                assert output["conformance"] == "INDETERMINATE"
                assert output["reference_frame"] == "CAD_REFERENCE" and output["unit"] == "mm"
                assert output["line_convention"] == LINE_CONVENTION
                assert output["endpoint_semantics"] == "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM"
                assert "summary" not in output and "passed" not in output and "failed" not in output
                json.dumps(output, allow_nan=False)
                stage = "NORMAL_IMPORT_FINITE_REFERENCE_EVALUATOR"
                score = score_support_case(
                    case_id=record["case_id"], reference_records=reference_records,
                    prediction_payload=output, reference_frame="CAD_REFERENCE", reference_unit="mm",
                    reference_convention=LINE_CONVENTION, thresholds=thresholds)
                record["evaluation"] = score
                assert score["true_positive"] == (1 if kind == "positive" else 0)
                assert score["false_positive"] == 0
                assert score["false_negative"] == (0 if kind == "positive" else 1)
                assert score["exact_count"] is (kind == "positive")
                assert score["exact_localized_target_set"] is (kind == "positive")
                assert score["promotion_eligible"] is False
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
    assert len(results) == 16 and [row["case_id"] for row in results] == list(MATRIX_CASE_IDS)
    aggregate = None
    if all("evaluation" in record for record in results):
        aggregate = score_support_cases([record["evaluation"] for record in results],
                                        expected_case_ids=list(MATRIX_CASE_IDS))
    failed = sum(record["status"] != "PASS" for record in results)
    return {
        "status": "FAIL" if failed else "PASS", "matrix_cases": 16, "failed_cases": failed,
        "positive_case_count": 12, "negative_case_count": 4, "engine_call_count": engine_calls,
        "maximum_errors": maxima, "support_evaluation_aggregate": aggregate,
        "fixed_geometry_tolerance_mm": GEOMETRY_ABSOLUTE_TOLERANCE_MM,
        "fixed_angle_tolerance_deg": ANGLE_ABSOLUTE_TOLERANCE_DEG,
        "runtime_import_mode": "NORMAL_BACKEND_NAMESPACE_NOT_AST_EXTRACTED",
        "fixture_helper_scope": "INVENTED_NATIVE_SHAPE_AND_ANALYTIC_POINTS_ONLY",
        "input_origin": "OWN_ISOLATED_CACHE_INVENTED_FILES",
        "scan_property_order": ["z", "x", "y", "red"], "scan_xyz_type": "double",
        "reference_line_convention": LINE_CONVENTION,
        "endpoint_semantics": "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM",
        "units_status": "DECLARED_MM_NOT_CALIBRATED", "ocp_version": OCP.__version__,
        "physical_corpus_accuracy": "NOT_TESTED", "incumbent_comparison": "NOT_RUN",
        "promotion_eligible": False, "cases": results,
    }
