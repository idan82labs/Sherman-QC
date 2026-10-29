"""Fixed invented Open3D transport qualification, invoked only by guarded worker.

No native import occurs at module load. This callable neither reads CAD/scans nor
writes artifacts. The parent owns process limits, isolated startup directories,
filesystem/network guards, the sole execution and retention of its first result.
This is native preprocessing/lineage validation, not physical bend discovery.
"""

import ast
import builtins
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple


MATRIX_STRATA = ("perpendicular_flange_grids", "quarter_cylinder_samples",
                 "disconnected_planar_patches", "voxel_boundary_neighbors")
MATRIX_SCALES = (0.125, 1.0, 8.0)
MATRIX_DENSITIES = (1, 4)
MATRIX_ROTATIONS = ("identity", "axis_cycle", "rational_z", "rational_quaternion")
MATRIX_CASE_COUNT = 96
RAW_VERTEX_CAP = 200_000
SOURCE_CLASS_NAMES = {"SourcePlaneSupport", "SourceBendCandidate", "PlaneSegment",
                      "DetectedBend", "BendSpecification", "BendDetector", "CADBendExtractor"}
SOURCE_CAP_NAMES = {"MAX_CAD_LINEAGE_RAW_VERTICES", "MAX_CAD_LINEAGE_CANDIDATES",
                    "MAX_CAD_LINEAGE_SUPPORT_REFERENCES"}


def rational_rotations():
    """Exact proper rotations; float conversion is a separate numerical step."""
    f = Fraction
    return {
        "identity": ((f(1), f(0), f(0)), (f(0), f(1), f(0)), (f(0), f(0), f(1))),
        "axis_cycle": ((f(0), f(1), f(0)), (f(0), f(0), f(1)), (f(1), f(0), f(0))),
        "rational_z": ((f(3, 5), f(-4, 5), f(0)), (f(4, 5), f(3, 5), f(0)),
                       (f(0), f(0), f(1))),
        "rational_quaternion": ((f(1, 9), f(8, 9), f(4, 9)),
                                (f(8, 9), f(1, 9), f(-4, 9)),
                                (f(-4, 9), f(4, 9), f(-7, 9))),
    }


def exact_rotation_check(matrix):
    orthogonal = all(sum(matrix[k][i] * matrix[k][j] for k in range(3)) == int(i == j)
                     for i in range(3) for j in range(3))
    a, b, c = matrix
    determinant = (a[0] * (b[1] * c[2] - b[2] * c[1])
                   - a[1] * (b[0] * c[2] - b[2] * c[0])
                   + a[2] * (b[0] * c[1] - b[1] * c[0]))
    return orthogonal and determinant == 1


def exact_float_floor(value, rounded_bound, voxel_size):
    """Exact arithmetic on the actual binary64 operands, not idealized truth."""
    quotient = ((Fraction.from_float(float(value)) - Fraction.from_float(float(rounded_bound)))
                / Fraction.from_float(float(voxel_size)))
    return quotient.numerator // quotient.denominator


def _source_namespace(np, o3d):
    source_path = Path(__file__).resolve().parents[1] / "backend/feature_detection/bend_detector.py"
    source = source_path.read_bytes()
    tree = ast.parse(source, filename=str(source_path))
    nodes = [node for node in tree.body if
             (isinstance(node, ast.ClassDef) and node.name in SOURCE_CLASS_NAMES) or
             (isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in SOURCE_CAP_NAMES
                 for target in node.targets))]
    if {node.name for node in nodes if isinstance(node, ast.ClassDef)} != SOURCE_CLASS_NAMES:
        raise RuntimeError("detector source capsule is incomplete")

    def guarded_import(name, *args, **kwargs):
        if name == "open3d":
            return o3d
        raise RuntimeError("detector capsule attempted an unauthorized import")

    env = {"__name__": "cad_native_open3d_capsule", "np": np, "dataclass": dataclass,
           "field": field, "Any": Any, "Dict": Dict, "List": List, "Optional": Optional,
           "Tuple": Tuple, "logger": logging.getLogger("cad-native-open3d"),
           "__builtins__": dict(vars(builtins), __import__=guarded_import)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), env)
    if env["MAX_CAD_LINEAGE_RAW_VERTICES"] != RAW_VERTEX_CAP:
        raise RuntimeError("declared raw-row cap differs from frozen source")
    return env, hashlib.sha256(source).hexdigest()


def _fixtures(np):
    x, y = np.meshgrid(np.arange(21) / 4., np.arange(13) / 4., indexing="ij")
    flat = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    upright = np.column_stack((x.ravel(), np.zeros(x.size), y.ravel()))
    flange_points = np.vstack((flat, upright))
    flange_normals = np.vstack((np.tile([0., 0., 1.], (len(flat), 1)),
                                np.tile([0., -1., 0.], (len(upright), 1))))
    axis, theta = np.meshgrid(np.arange(21) / 4., np.arange(17) * (np.pi / 32.), indexing="ij")
    cylinder_points = np.column_stack((axis.ravel(), 2. * np.cos(theta).ravel(),
                                       2. * np.sin(theta).ravel()))
    cylinder_normals = np.column_stack((np.zeros(axis.size), np.cos(theta).ravel(),
                                        np.sin(theta).ravel()))
    patch_x, patch_y = np.meshgrid(np.arange(13) / 4., np.arange(13) / 4., indexing="ij")
    patch = np.column_stack((patch_x.ravel(), patch_y.ravel(), np.zeros(patch_x.size)))
    patches = np.vstack((patch, patch + [8., 0., 0.]))
    boundary_x = np.array([0., np.nextafter(.125, 0.), .125, np.nextafter(.125, 1.),
                           .25, np.nextafter(.375, 0.), .375, np.nextafter(.375, 1.), .5, .75])
    bx, by, bz = np.meshgrid(boundary_x, np.arange(5) / 2., np.arange(5) / 2., indexing="ij")
    boundaries = np.column_stack((bx.ravel(), by.ravel(), bz.ravel()))
    return {
        MATRIX_STRATA[0]: (flange_points, flange_normals),
        MATRIX_STRATA[1]: (cylinder_points, cylinder_normals),
        MATRIX_STRATA[2]: (patches, np.tile([0., 0., 1.], (len(patches), 1))),
        MATRIX_STRATA[3]: (boundaries, np.tile([0., 0., 1.], (len(boundaries), 1))),
    }


def _floor_diagnostic(np, points, voxel_size):
    # Duplicate operands need only one arithmetic check; multiplicities count all
    # original rows. Reference uses the rounded native grid bound as an operand.
    unique, counts = np.unique(points, axis=0, return_counts=True)
    minimum = np.min(points, axis=0) - voxel_size * .5
    floating = np.floor((unique - minimum) / voxel_size)
    different_rows = 0
    different_axes = 0
    max_cell_difference = 0
    for row, count, float_cells in zip(unique, counts, floating):
        row_difference = False
        for axis in range(3):
            reference = exact_float_floor(row[axis], minimum[axis], voxel_size)
            difference = abs(int(float_cells[axis]) - reference)
            if difference:
                different_axes += int(count)
                row_difference = True
                max_cell_difference = max(max_cell_difference, difference)
        if row_difference:
            different_rows += int(count)
    return {"unique_operand_rows": len(unique), "raw_operand_rows": len(points),
            "rounded_arithmetic_floor_different_rows": different_rows,
            "rounded_arithmetic_floor_different_axes": different_axes,
            "maximum_cell_difference": max_cell_difference,
            "interpretation": "BOUNDARY_ROUNDOFF_DIAGNOSTIC_NOT_PHYSICAL_GEOMETRY_ERROR"}


def _new_detector(env):
    return env["BendDetector"](min_plane_points=3, min_plane_area=0.,
                                adaptive_threshold=False, merge_coplanar_planes=False)


def _case(np, env, case_id, stratum, points, normals, voxel_size, expected_available=True):
    begin = time.perf_counter()
    result = {"case_id": case_id, "stratum": stratum, "raw_vertex_count": len(points),
              "expected_lineage_available": expected_available, "failure_codes": []}
    try:
        baseline = _new_detector(env)
        baseline._begin_source_lineage(points, False)
        before, before_normals = baseline._preprocess_point_cloud(points, normals, voxel_size, 20, 2.)
        candidate = _new_detector(env)
        candidate._begin_source_lineage(points, True)
        after, after_normals = candidate._preprocess_point_cloud(points, normals, voxel_size, 20, 2.)
        state = candidate.last_source_lineage
        geometry_equal = np.array_equal(before, after) and (
            (before_normals is None and after_normals is None) or
            (before_normals is not None and after_normals is not None and
             np.array_equal(before_normals, after_normals)))
        result.update(retained_vertex_count=len(after), legacy_geometry_exact=bool(geometry_equal),
                      lineage_status=state["status"], lineage_reason=state["reason"])
        if not geometry_equal:
            result["failure_codes"].append("LEGACY_GEOMETRY_CHANGED")
        if len(after) == 0:
            result["failure_codes"].append("EMPTY_NATIVE_RESULT")
        if expected_available and state["status"] != "COMPLETE":
            result["failure_codes"].append("NATIVE_TRACE_UNAVAILABLE")
        if not expected_available and state["status"] != "UNAVAILABLE":
            result["failure_codes"].append("OVER_CAP_LINEAGE_NOT_REJECTED")
        if state["status"] == "COMPLETE":
            groups = state["processed_source_vertex_indices"]
            rejected = state["preprocessing_rejected_source_vertex_indices"]
            flat = [index for group in groups for index in group]
            complete = (len(groups) == len(after) and
                        sorted(flat + list(rejected)) == list(range(len(points))))
            if not complete:
                result["failure_codes"].append("RAW_ROW_PARTITION_INVALID")
            maximum_error = 0.
            means_equal = True
            for index, group in enumerate(groups):
                mean = np.mean(points[list(group)], axis=0)
                maximum_error = max(maximum_error, float(np.max(np.abs(mean - after[index]))))
                means_equal = means_equal and np.array_equal(mean, after[index])
            if not means_equal:
                result["failure_codes"].append("RAW_CENTROID_NOT_EXACT")
            result.update(raw_partition_complete=bool(complete), retained_raw_rows=len(flat),
                          rejected_raw_rows=len(rejected), maximum_centroid_error=maximum_error)
        elif any(state[key] for key in ("processed_source_vertex_indices",
                                       "preprocessing_rejected_source_vertex_indices",
                                       "segmentation_passes", "candidate_records")):
            result["failure_codes"].append("UNAVAILABLE_LEDGER_RETAINED_PREFIX")
        result["floor_reference"] = _floor_diagnostic(np, points, voxel_size)
    except Exception as exc:
        result["failure_codes"].append("CASE_EXCEPTION")
        result["exception_type"] = type(exc).__name__
        result["exception_summary"] = " ".join(str(exc).splitlines()[:1])[:200]
    result["status"] = "PASS" if not result["failure_codes"] else "FAIL"
    result["elapsed_seconds"] = round(time.perf_counter() - begin, 6)
    return result


def _trace_controls(np, o3d, env, points, normals):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    size = .25
    legacy = pcd.voxel_down_sample(voxel_size=size)
    traced, cube, native_rows = pcd.voxel_down_sample_and_trace(
        size, points.min(axis=0) - size * .5, points.max(axis=0) + size * .5, False)
    old_points, old_normals = np.asarray(legacy.points), np.asarray(legacy.normals)
    trace_points, trace_normals = np.asarray(traced.points), np.asarray(traced.normals)
    rows = [list(map(int, group)) for group in native_rows]
    detector = _new_detector(env)
    names = ("untouched_native_trace", "late_missing_row", "late_duplicate_row",
             "late_boolean_row", "late_out_of_range_row", "late_float_row",
             "changed_trace_position", "changed_trace_normal")
    results = []
    for name in names:
        changed_rows = [list(group) for group in rows]
        changed_points = trace_points.copy()
        changed_normals = trace_normals.copy()
        if name == "late_missing_row":
            changed_rows[-1] = []
        elif name == "late_duplicate_row":
            changed_rows[-1][-1] = changed_rows[0][0]
        elif name == "late_boolean_row":
            changed_rows[-1][-1] = True
        elif name == "late_out_of_range_row":
            changed_rows[-1][-1] = len(points)
        elif name == "late_float_row":
            changed_rows[-1][-1] = float(changed_rows[-1][-1])
        elif name == "changed_trace_position":
            changed_points[-1, 0] = np.nextafter(changed_points[-1, 0], float("inf"))
        elif name == "changed_trace_normal":
            changed_normals[-1, 0] = np.nextafter(changed_normals[-1, 0], float("inf"))
        accepted = False
        try:
            detector._validate_voxel_source_trace(points, size, old_points, old_normals,
                                                   changed_points, changed_normals, changed_rows)
            accepted = True
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            pass
        expected = name == "untouched_native_trace"
        results.append({"case_id": name, "accepted": accepted, "expected_accepted": expected,
                        "status": "PASS" if accepted == expected else "FAIL"})
    # Do not hide a vacuous negative matrix behind an invalid untouched control.
    return {"cases": results, "untouched_native_trace_accepted": results[0]["accepted"],
            "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL"}


def _numeric_controls(np, env):
    detector = _new_detector(env)
    cases = []
    for name, points, size in (
        ("subnormal_voxel_unit_span", np.array([[0., 0., 0.], [1., 0., 0.]]),
         float(np.nextafter(0., 1.))),
        ("finite_extreme_subtraction_overflow", np.array([[-1e308, 0., 0.], [1e308, 0., 0.]]), 1.),
        ("underflowed_zero_voxel", np.array([[0., 0., 0.], [1., 0., 0.]]),
         float(np.nextafter(0., 1.) / 2.)),
    ):
        rejected = False
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            try:
                detector._validate_voxel_source_trace(points, size, points, None, points, None, [[0], [1]])
            except (ValueError, TypeError, OverflowError, FloatingPointError):
                rejected = True
        cases.append({"case_id": name, "status": "PASS" if rejected else "FAIL",
                      "native_call": False, "unsupported_arithmetic_rejected": rejected})
    bad = np.zeros((6, 3))
    bad[-1, -1] = np.nan
    detector._begin_source_lineage(bad, True)
    unavailable = detector.last_source_lineage["status"] == "UNAVAILABLE"
    cases.append({"case_id": "late_nonfinite_original_row", "status": "PASS" if unavailable else "FAIL",
                  "native_call": False, "unsupported_arithmetic_rejected": unavailable})
    return {"cases": cases, "status": "PASS" if all(item["status"] == "PASS" for item in cases) else "FAIL"}


def _nonempty_transport(np, env, points, normals):
    """Native preprocessing + exact source transport; proposal math is stipulated."""
    detector = _new_detector(env)
    detector._compute_ransac_iterations = lambda count: 1
    detector._ransac_plane_fit = lambda rows, **kwargs: (
        np.array([0., 0., 1.]), 0., np.arange(len(rows)) < 12)
    detector._estimate_area = lambda rows: 100.
    detector._extract_boundary = lambda rows: rows.copy()
    detector._find_adjacent_planes = lambda planes: [(planes[0], planes[1]), (planes[1], planes[2])]

    def proposed(a, b, all_points, bend_id):
        offset = 0. if bend_id == "B1" else 1.
        return env["DetectedBend"](bend_id, 90., 10., np.array([offset, 0., 0.]),
                                    np.array([offset + 4., 0., 0.]), np.array([1., 0., 0.]),
                                    .9 if bend_id == "B1" else .8, a, b)

    detector._analyze_bend = proposed
    found = detector.detect_bends(points, normals, preprocess=True, voxel_size=.25,
                                   multiscale=True, preserve_source_lineage=True)
    state = detector.last_source_lineage
    records = state["candidate_records"]
    original_segments_ok = all(record["bend_line_start"] in ([0., 0., 0.], [1., 0., 0.]) and
                               record["bend_line_end"] in ([4., 0., 0.], [5., 0., 0.])
                               for record in records)
    candidates_ok = (len(found) == 1 and len(records) == 6 and
                     len(found[0].source_candidates) == 6 and
                     {record["candidate_id"] for record in records} == {f"candidate:{i}" for i in range(6)})
    # Cross-check every segment's raw memberships against this pass's exact
    # processed-row mapping, without coordinate matching or native root labels.
    groups = state["processed_source_vertex_indices"]
    expected_first = tuple(sorted(row for group in groups[:12] for row in group))
    support_ok = bool(records) and all(
        tuple(record["flange1_source_vertex_indices"]) == expected_first
        for record in records if int(record["candidate_id"].split(":")[-1]) % 2 == 0)
    passed = state["status"] == "COMPLETE" and candidates_ok and original_segments_ok and support_ok
    return {"case_id": "native_preprocess_stipulated_proposal_multiscale_transport",
            "status": "PASS" if passed else "FAIL", "lineage_status": state["status"],
            "lineage_reason": state["reason"], "selected_geometry_count": len(found),
            "original_candidate_count": len(records), "finite_segments_preserved": original_segments_ok,
            "original_raw_support_checked": support_ok,
            "proposal_math": "STIPULATED_NOT_NATIVE_RANSAC_OR_PHYSICAL_DETECTION"}


def _retained_exception(case_id, exc):
    return {"case_id": case_id, "status": "FAIL", "exception_type": type(exc).__name__,
            "exception_summary": " ".join(str(exc).splitlines()[:1])[:200]}


def run_checks():
    """Run the frozen matrix once inside the parent's guarded native worker."""
    import numpy as np
    import open3d as o3d

    started = time.perf_counter()
    env, source_hash = _source_namespace(np, o3d)
    fixtures = _fixtures(np)
    rational = rational_rotations()
    translations = {"identity": [0., 0., 0.], "axis_cycle": [32., -16., 8.],
                    "rational_z": [.375, -.625, 1.125], "rational_quaternion": [-7.25, 11.5, 2.125]}
    rotation_checks = []
    for name in MATRIX_ROTATIONS:
        matrix = np.array(rational[name], dtype=float)
        error = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
        exact_ok = exact_rotation_check(rational[name])
        rotation_checks.append({"rotation": name, "exact_rational_proper_rotation": exact_ok,
                                "binary64_orthogonality_max_error": error,
                                "status": "PASS" if exact_ok and error <= 8 * np.finfo(float).eps else "FAIL"})
    cases = []
    for stratum in MATRIX_STRATA:
        base, base_normals = fixtures[stratum]
        for rotation in MATRIX_ROTATIONS:
            matrix = np.array(rational[rotation], dtype=float)
            for scale in MATRIX_SCALES:
                transformed = (base * scale) @ matrix.T + translations[rotation]
                transformed_normals = base_normals @ matrix.T
                for density in MATRIX_DENSITIES:
                    points = np.repeat(transformed, density, axis=0)
                    normals = np.repeat(transformed_normals, density, axis=0)
                    identity = f"{stratum}/{rotation}/scale={scale}/duplicates={density}"
                    cases.append(_case(np, env, identity, stratum, points, normals, .25 * scale))
    if len(cases) != MATRIX_CASE_COUNT:
        raise RuntimeError("matrix cardinality differs from frozen declaration")
    gx, gy = np.meshgrid(np.arange(100, dtype=float), np.arange(100, dtype=float), indexing="ij")
    grid = np.column_stack((gx.ravel(), gy.ravel(), np.zeros(gx.size)))
    dense = np.repeat(grid, 20, axis=0)
    dense_normals = np.tile([0., 0., 1.], (len(dense), 1))
    cases.append(_case(np, env, "dense_at_200000_raw_cap", "raw_cap_boundary", dense, dense_normals, .25))
    over = np.vstack((dense, [[10000., 10000., 10000.]]))
    over_normals = np.vstack((dense_normals, [[0., 0., 1.]]))
    cases.append(_case(np, env, "dense_over_200000_raw_cap", "raw_cap_boundary", over, over_normals, .25, False))
    del dense, dense_normals, over, over_normals
    try:
        controls = _trace_controls(np, o3d, env, *fixtures[MATRIX_STRATA[0]])
    except Exception as exc:
        controls = {"status": "FAIL", "cases": [_retained_exception("trace_controls_exception", exc)]}
    try:
        numerical = _numeric_controls(np, env)
    except Exception as exc:
        numerical = {"status": "FAIL", "cases": [_retained_exception("numeric_controls_exception", exc)]}
    try:
        transport = _nonempty_transport(np, env, *fixtures[MATRIX_STRATA[0]])
    except Exception as exc:
        transport = _retained_exception("nonempty_transport_exception", exc)
    all_results = cases + rotation_checks + controls["cases"] + numerical["cases"] + [transport]
    failed = [item for item in all_results if item["status"] != "PASS"]
    strata = {name: {"cases": sum(item["stratum"] == name for item in cases),
                     "passed": sum(item["stratum"] == name and item["status"] == "PASS" for item in cases)}
              for name in MATRIX_STRATA + ("raw_cap_boundary",)}
    return {
        "schema_version": 1, "kind": "CAD_NATIVE_OPEN3D_FIXED_MATRIX_V1",
        "status": "PASS" if not failed else "FAIL", "open3d_version": str(o3d.__version__),
        "detector_source_sha256": source_hash,
        "declared_matrix": {"cases": MATRIX_CASE_COUNT, "strata": list(MATRIX_STRATA),
                            "rotations": list(MATRIX_ROTATIONS), "scales": list(MATRIX_SCALES),
                            "duplicate_densities": list(MATRIX_DENSITIES), "raw_vertex_cap": RAW_VERTEX_CAP,
                            "cap_cases": 2, "trace_controls": 8, "numeric_controls": 4,
                            "nonempty_stipulated_transport_cases": 1},
        "checks": {"total": len(all_results), "passed": len(all_results) - len(failed), "failed": len(failed)},
        "numeric_summary": {
            "maximum_centroid_error": max((item["maximum_centroid_error"] for item in cases
                                            if "maximum_centroid_error" in item), default=None),
            "maximum_binary64_rotation_error": max(item["binary64_orthogonality_max_error"]
                                                     for item in rotation_checks),
            "floor_reference_cases_completed": sum("floor_reference" in item for item in cases),
            "floating_floor_disagreement_raw_rows": sum(
                item.get("floor_reference", {}).get("rounded_arithmetic_floor_different_rows", 0)
                for item in cases),
            "maximum_floor_cell_difference": max((
                item["floor_reference"]["maximum_cell_difference"] for item in cases
                if "floor_reference" in item), default=None),
        },
        "strata": strata, "cases": cases, "rotation_checks": rotation_checks,
        "trace_controls": controls, "numeric_controls": numerical, "nonempty_transport": transport,
        "first_failure": failed[0] if failed else None,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "claims": {"physical_inputs_used": False, "native_preprocessing_exercised": True,
                   "native_ransac_accuracy_established": False, "physical_generalization_established": False,
                   "thresholds_or_caps_retuned": False},
    }
