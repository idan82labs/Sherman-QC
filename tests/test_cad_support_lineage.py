"""Invented, native-free exact-source transport checks; no backend import.

Only the named source file is read. AST-selected classes execute with NumPy,
stdlib and guarded Open3D doubles; no CAD file, physical input or kernel opens.
"""

import ast
import builtins
from dataclasses import FrozenInstanceError, dataclass, field
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import weakref

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend/feature_detection/bend_detector.py"
CLASS_NAMES = {
    "SourcePlaneSupport", "SourceBendCandidate", "PlaneSegment", "DetectedBend",
    "BendSpecification", "BendDetector", "CADBendExtractor",
}
CAP_NAMES = {
    "MAX_CAD_LINEAGE_RAW_VERTICES", "MAX_CAD_LINEAGE_CANDIDATES",
    "MAX_CAD_LINEAGE_SUPPORT_REFERENCES",
}


@pytest.fixture
def ns():
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if
                (isinstance(node, ast.ClassDef) and node.name in CLASS_NAMES) or
                (isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in CAP_NAMES
                    for target in node.targets))]
    assert {node.name for node in selected if isinstance(node, ast.ClassDef)} == CLASS_NAMES
    imports = []
    env = {
        "__name__": "cad_lineage_source_contract", "np": np, "dataclass": dataclass,
        "field": field, "Any": Any, "Dict": Dict, "List": List, "Optional": Optional,
        "Tuple": Tuple, "logger": logging.getLogger("cad-lineage-test"),
    }

    def guarded_import(name, *args, **kwargs):
        imports.append(name)
        if name == "open3d" and "fake_open3d" in env:
            return env["fake_open3d"]
        raise AssertionError(f"Unexpected source import: {name}")

    env["__builtins__"] = dict(vars(builtins), __import__=guarded_import)
    env["imports"] = imports
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), env)
    return env


def detector(ns):
    return ns["BendDetector"](min_plane_points=3, min_plane_area=0,
                              adaptive_threshold=False, merge_coplanar_planes=False)


def points_fixture():
    return np.array([[0., 0., 0.], [.1, 0., 0.], [1., 0., 0.],
                     [2., 0., 0.], [2.1, 0., 0.], [3., 0., 0.]])


def fake_open3d(ns, fault=None, with_normals=True):
    raw = points_fixture()
    groups = [[0, 1], [2], [3, 4], [5]]
    voxel = np.array([raw[group].mean(axis=0) for group in groups])
    normals = np.tile([0., 0., 1.], (4, 1)) if with_normals else None
    calls = []

    class Cloud:
        def __init__(self, points=None, normal_rows=None):
            self.points = np.empty((0, 3)) if points is None else np.array(points, copy=True)
            self.normals = np.empty((0, 3)) if normal_rows is None else np.array(normal_rows, copy=True)

        def has_normals(self):
            return len(self.normals) > 0 and len(self.normals) == len(self.points)

        def voxel_down_sample(self, **kwargs):
            calls.append(("legacy_voxel", kwargs))
            return Cloud(voxel, normals)

        def voxel_down_sample_and_trace(self, **kwargs):
            calls.append(("trace", kwargs))
            if fault == "missing_trace":
                raise AttributeError("invented unavailable version")
            trace_points = voxel.copy()
            trace_normals = None if normals is None else normals.copy()
            rows = [list(group) for group in groups]
            if fault == "trace_order":
                trace_points = trace_points[::-1]
                rows = rows[::-1]
            elif fault == "trace_geometry":
                trace_points[-1, 0] += .001
            elif fault == "trace_normals":
                trace_normals[-1, 0] = .01
            elif fault == "trace_missing_normal":
                trace_normals = None
            elif fault == "trace_missing_row":
                rows[-1] = []
            elif fault == "trace_duplicate":
                rows[-1] = [0]
            elif fault == "trace_bool":
                rows[0][0] = False
            elif fault == "trace_float":
                rows[-1] = [5.0]
            elif fault == "trace_out_of_range":
                rows[-1] = [6]
            elif fault == "trace_wrong_cell":
                rows[0], rows[1] = [0, 2], [1]
            return Cloud(trace_points, trace_normals), np.empty((0, 8)), rows

        def remove_statistical_outlier(self, **kwargs):
            calls.append(("outliers", kwargs))
            keep = [0, 2, 3]
            result_points = voxel[keep].copy()
            result_normals = None if normals is None else normals[keep].copy()
            if fault == "outlier_index":
                keep[-1] = 1
            elif fault == "outlier_duplicate":
                keep[-1] = 2
            elif fault == "outlier_bool":
                keep[0] = False
            elif fault == "outlier_geometry":
                result_points[-1, 0] += .001
            elif fault == "outlier_normals":
                result_normals[-1, 1] = .01
            return Cloud(result_points, result_normals), keep

        def estimate_normals(self, **kwargs):
            calls.append(("estimate_normals", kwargs))
            self.normals = np.tile([0., 0., 1.], (len(self.points), 1))

    ns["fake_open3d"] = SimpleNamespace(
        geometry=SimpleNamespace(PointCloud=Cloud, KDTreeSearchParamHybrid=lambda **kw: kw),
        utility=SimpleNamespace(Vector3dVector=lambda rows: np.array(rows, copy=True)),
    )
    return raw, voxel, calls


def run_preprocess(ns, fault=None, enabled=True, with_normals=True):
    raw, voxel, calls = fake_open3d(ns, fault, with_normals)
    det = detector(ns)
    det._begin_source_lineage(raw, enabled)
    raw_normals = np.tile([0., 0., 1.], (len(raw), 1)) if with_normals else None
    processed, normals = det._preprocess_point_cloud(raw, raw_normals, 1., 20, 2.)
    return det, processed, normals, voxel, calls


def plane(ns, rows, plane_id=0, z=0.):
    coords = np.array([[float(index), z, 0.] for index in rows])
    segment = ns["SourcePlaneSupport"](f"pass:0:plane:{plane_id}", tuple(rows))
    return ns["PlaneSegment"](
        plane_id, np.array([0., 0., 1.]), 0., coords.mean(axis=0), coords,
        coords.copy(), 10., len(rows), tuple(rows), (segment,), "COMPLETE")


def bend(ns, first, second, offset=0., confidence=.9):
    return ns["DetectedBend"](
        "original", 90., 10., np.array([offset, 0., 0.]),
        np.array([offset + 4., 0., 0.]), np.array([1., 0., 0.]), confidence,
        first, second)


def configure_geometry(ns, det, offsets=(0., 1.), coplanar=False):
    """Actual transport stages, with stipulated segmentation/geometry proposals."""
    det._compute_ransac_iterations = lambda count: 1

    def fit(rows, **kwargs):
        mask = np.arange(len(rows)) < 3
        return np.array([0., 0., 1.]), 0., mask

    det._ransac_plane_fit = fit
    det._estimate_area = lambda rows: 10.
    det._extract_boundary = lambda rows: rows.copy()
    det._least_squares_plane = lambda rows: (np.array([0., 0., 1.]), 0.)
    det._find_adjacent_planes = lambda planes: [(planes[0], planes[-1])] if coplanar else [
        (planes[0], planes[1]), (planes[1], planes[2])]
    det._planes_coplanar_and_adjacent = lambda a, b: coplanar and a.plane_id < 2 and b.plane_id < 2
    det.merge_coplanar_planes = coplanar
    det._analyze_bend = lambda a, b, points, bend_id: bend(
        ns, a, b, offset=offsets[int(bend_id[1:]) - 1],
        confidence=.9 - .1 * (int(bend_id[1:]) - 1))
    return det


def extraction_fixture(ns, offsets=(0., 1.), multiscale=False, coplanar=False):
    real_class = ns["BendDetector"]

    def configured(**kwargs):
        kwargs.update(min_plane_points=3, min_plane_area=0, adaptive_threshold=False)
        return configure_geometry(ns, real_class(**kwargs), offsets, coplanar)

    ns["BendDetector"] = configured
    ext = ns["CADBendExtractor"](detect_call_kwargs={"preprocess": False, "multiscale": multiscale})
    raw = np.array([[float(i), float(i % 2), 0.] for i in range(9)])
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=int)
    return ext, raw, faces


def test_exact_preprocessing_keeps_every_contributor_and_rejected_rows(ns):
    det, processed, normals, voxel, calls = run_preprocess(ns)
    np.testing.assert_array_equal(processed, voxel[[0, 2, 3]])
    np.testing.assert_array_equal(normals, np.tile([0., 0., 1.], (3, 1)))
    assert det.last_source_lineage["status"] == "COMPLETE"
    assert det.last_source_lineage["processed_source_vertex_indices"] == ((0, 1), (3, 4), (5,))
    assert det.last_source_lineage["preprocessing_rejected_source_vertex_indices"] == (2,)
    trace = next(kwargs for name, kwargs in calls if name == "trace")
    np.testing.assert_array_equal(trace["min_bound"], [-.5, -.5, -.5])
    np.testing.assert_array_equal(trace["max_bound"], [3.5, .5, .5])
    assert trace["approximate_class"] is False
    assert ns["imports"] == ["open3d"]


def test_default_preprocessing_does_not_call_trace(ns):
    det, processed, _, voxel, calls = run_preprocess(ns, enabled=False)
    assert det.last_source_lineage is None
    assert [name for name, _ in calls] == ["legacy_voxel", "outliers"]
    np.testing.assert_array_equal(processed, voxel[[0, 2, 3]])


@pytest.mark.parametrize("fault", [
    "missing_trace", "trace_order", "trace_geometry", "trace_normals", "trace_missing_normal",
    "trace_missing_row", "trace_duplicate", "trace_bool", "trace_float", "trace_out_of_range",
    "trace_wrong_cell", "outlier_index", "outlier_duplicate", "outlier_bool",
    "outlier_geometry", "outlier_normals",
])
def test_invalid_trace_abstains_wholly_without_changing_legacy_geometry(ns, fault):
    det, actual, actual_normals, _, _ = run_preprocess(ns, fault=fault)
    state = det.last_source_lineage
    assert state["status"] == "UNAVAILABLE"
    assert state["processed_source_vertex_indices"] == ()
    assert state["preprocessing_rejected_source_vertex_indices"] == ()
    assert state["candidate_records"] == []
    assert state["segmentation_passes"] == []
    _, baseline, baseline_normals, _, _ = run_preprocess(ns, fault=fault, enabled=False)
    np.testing.assert_array_equal(actual, baseline)
    np.testing.assert_array_equal(actual_normals, baseline_normals)


def test_missing_input_normals_estimated_only_after_valid_selection(ns):
    det, _, normals, _, calls = run_preprocess(ns, with_normals=False)
    assert det.last_source_lineage["status"] == "COMPLETE"
    assert calls[-1][0] == "estimate_normals"
    assert normals.shape == (3, 3)


def test_coincident_raw_vertices_keep_distinct_row_identity(ns):
    det = detector(ns)
    raw = np.zeros((3, 3))
    det._begin_source_lineage(raw, True)
    result = det._validate_voxel_source_trace(raw, 1., np.zeros((1, 3)), None,
                                              np.zeros((1, 3)), None, [[0, 1, 2]])
    assert result == ((0, 1, 2),)


def test_segment_indices_follow_voxel_contributors_not_coordinate_search(ns):
    det = detector(ns)
    raw = np.array([[float(i), 0., 0.] for i in range(8)])
    det._begin_source_lineage(raw, True)
    det._point_source_indices = ((0, 1), (2,), (3, 4), (5,), (6,), (7,))
    det._lineage_pass = 0
    det._compute_ransac_iterations = lambda count: 1
    det._ransac_plane_fit = lambda rows, **kw: (
        np.array([0., 0., 1.]), 0., np.arange(len(rows)) < 3)
    det._estimate_area = lambda rows: 10. if rows[0, 0] == 0 else -1.
    det._extract_boundary = lambda rows: rows.copy()
    planes = det._segment_planes(raw[:6])
    assert len(planes) == 1
    assert planes[0].source_vertex_indices == (0, 1, 2, 3, 4)
    assert planes[0].source_segments[0].source_vertex_indices == (0, 1, 2, 3, 4)
    passage = det.last_source_lineage["segmentation_passes"][0]
    assert passage["rejected_source_vertex_indices"] == (5, 6, 7)
    assert passage["unassigned_source_vertex_indices"] == ()


def test_unassigned_rows_remain_explicit(ns):
    det = detector(ns)
    raw = np.arange(21, dtype=float).reshape(7, 3)
    det._begin_source_lineage(raw, True)
    det._lineage_pass = 0
    det._compute_ransac_iterations = lambda count: 1
    det._ransac_plane_fit = lambda rows, **kw: None
    assert det._segment_planes(raw) == []
    assert det.last_source_lineage["segmentation_passes"][0]["unassigned_source_vertex_indices"] == tuple(range(7))


def test_coplanar_merge_preserves_original_segment_pieces(ns):
    det = configure_geometry(ns, detector(ns), coplanar=True)
    raw = np.arange(27, dtype=float).reshape(9, 3)
    result = det.detect_bends(raw, preprocess=False, preserve_source_lineage=True)
    assert len(result) == 1
    record = result[0].source_candidates[0]
    assert record.flange1_source_vertex_indices == tuple(range(6))
    assert [part.source_vertex_indices for part in record.flange1_segments] == [(0, 1, 2), (3, 4, 5)]
    assert record.flange2_source_vertex_indices == (6, 7, 8)
    assert [part.segment_id for part in record.flange1_segments] == ["pass:0:plane:0", "pass:0:plane:1"]


def test_candidate_identity_and_finite_segment_snapshot_survive_geometry_mutation(ns):
    det = detector(ns)
    det._begin_source_lineage(np.zeros((6, 3)), True)
    det._lineage_pass = 0
    original = bend(ns, plane(ns, (0, 1, 2)), plane(ns, (3, 4, 5), 1))
    det._record_source_candidate(original)
    record = original.source_candidates[0]
    original.bend_id = "renumbered"
    original.bend_line_start[:] = 90.
    assert record.candidate_id == "candidate:0"
    assert record.bend_line_start == (0., 0., 0.)
    with pytest.raises(FrozenInstanceError):
        record.candidate_id = "changed"
    exported = record.to_dict()
    exported["flange1_source_vertex_indices"].clear()
    assert record.flange1_source_vertex_indices == (0, 1, 2)


@pytest.mark.parametrize("multiscale,expected", [(False, 2), (True, 6)])
def test_dedup_and_multiscale_preserve_all_original_candidates_with_legacy_winner(ns, multiscale, expected):
    det = configure_geometry(ns, detector(ns))
    raw = np.arange(27, dtype=float).reshape(9, 3)
    retained = det.detect_bends(raw, preprocess=False, multiscale=multiscale, preserve_source_lineage=True)
    assert len(retained) == 1
    actual = retained[0].to_dict()
    records = retained[0].source_candidates
    assert len(records) == expected
    assert {record.candidate_id for record in records} == {f"candidate:{i}" for i in range(expected)}
    assert {record.detection_pass for record in records} == ({0, 1, 2} if multiscale else {0})
    assert [record["candidate_id"] for record in det.last_source_lineage["candidate_records"]] == [
        f"candidate:{i}" for i in range(expected)]
    baseline = det.detect_bends(raw, preprocess=False, multiscale=multiscale)
    assert baseline[0].to_dict() == actual
    assert baseline[0].source_candidates == ()
    assert det.last_source_lineage is None


@pytest.mark.parametrize("offsets,expected", [((0., 1.), 2), ((0., 11.), 2)])
def test_extractor_early_dedup_and_rolled_merge_keep_premerge_ledger(ns, offsets, expected):
    ext, raw, faces = extraction_fixture(ns, offsets)
    actual = ext.extract_from_mesh(raw, faces, provenance={"native_topology": "UNKNOWN"})
    assert len(actual) == 1
    saved = actual[0].to_dict()
    start = actual[0].bend_line_start.copy()
    end = actual[0].bend_line_end.copy()
    records = actual[0].cad_provenance["candidate_records"]
    assert len(records) == expected
    assert [record["bend_line_start"][0] for record in records] == list(offsets)
    assert [record["bend_line_end"][0] for record in records] == [offset + 4 for offset in offsets]
    assert ext.last_source_lineage["candidate_records"] == records
    assert actual[0].cad_provenance["status"] == "COMPLETE"
    baseline = ext.extract_from_mesh(raw, faces)
    assert baseline[0].to_dict() == saved
    np.testing.assert_array_equal(baseline[0].bend_line_start, start)
    np.testing.assert_array_equal(baseline[0].bend_line_end, end)
    assert baseline[0].cad_provenance is None


def test_extractor_multiscale_ledger_contains_unique_originals_once(ns):
    ext, raw, faces = extraction_fixture(ns, multiscale=True)
    specs = ext.extract_from_mesh(raw, faces, preserve_source_lineage=True)
    records = specs[0].cad_provenance["candidate_records"]
    assert len(records) == 6
    assert len({record["candidate_id"] for record in records}) == 6
    assert len(ext.last_source_lineage["candidate_records"]) == 6


@pytest.mark.parametrize("bad", [None, 0, 1, "yes", np.bool_(True)])
def test_public_lineage_flag_is_strict_boolean(ns, bad):
    with pytest.raises(TypeError):
        detector(ns).detect_bends(np.zeros((9, 3)), preprocess=False, preserve_source_lineage=bad)
    with pytest.raises(TypeError):
        ns["CADBendExtractor"]().extract_from_mesh(np.zeros((9, 3)), np.empty((0, 3)),
                                                   preserve_source_lineage=bad)


@pytest.mark.parametrize("budget", ["MAX_CAD_LINEAGE_RAW_VERTICES", "MAX_CAD_LINEAGE_CANDIDATES",
                                   "MAX_CAD_LINEAGE_SUPPORT_REFERENCES"])
def test_budget_failure_clears_global_and_all_surviving_spec_records(ns, budget):
    ext, raw, faces = extraction_fixture(ns, offsets=(0., 11.))
    ns[budget] = {"MAX_CAD_LINEAGE_RAW_VERTICES": 8, "MAX_CAD_LINEAGE_CANDIDATES": 1,
                  "MAX_CAD_LINEAGE_SUPPORT_REFERENCES": 12}[budget]
    specs = ext.extract_from_mesh(raw, faces, provenance={})
    assert len(specs) == 1
    assert specs[0].cad_provenance["status"] == "UNAVAILABLE"
    assert specs[0].cad_provenance["candidate_records"] == []
    assert ext.last_source_lineage["candidate_records"] == []
    assert ext.last_source_lineage["processed_source_vertex_indices"] == ()
    assert ext.last_source_lineage["segmentation_passes"] == []


@pytest.mark.parametrize("bad_end", [[np.nan, 0., 0.], [np.inf, 0., 0.], [0., 0., 0.],
                                  [1e308, 1e308, 1e308]])
def test_late_bad_candidate_clears_earlier_complete_ledger(ns, bad_end):
    det = detector(ns)
    det._begin_source_lineage(np.zeros((6, 3)), True)
    det._lineage_pass = 0
    p1, p2 = plane(ns, (0, 1, 2)), plane(ns, (3, 4, 5), 1)
    first = bend(ns, p1, p2)
    det._record_source_candidate(first)
    second = bend(ns, p1, p2)
    second.bend_line_end = np.array(bad_end)
    with np.errstate(over="ignore", invalid="ignore"):
        det._record_source_candidate(second)
    det._finish_source_lineage([first, second])
    assert first.source_candidates == second.source_candidates == ()
    assert det.last_source_lineage["status"] == "UNAVAILABLE"
    assert det.last_source_lineage["candidate_records"] == []


def test_lost_or_duplicated_candidate_ledger_abstains(ns):
    det = detector(ns)
    det._begin_source_lineage(np.zeros((6, 3)), True)
    det._lineage_pass = 0
    first = bend(ns, plane(ns, (0, 1, 2)), plane(ns, (3, 4, 5), 1))
    det._record_source_candidate(first)
    det._finish_source_lineage([first, first])
    assert det.last_source_lineage["status"] == "UNAVAILABLE"
    assert first.source_candidates == ()


def test_no_native_face_owner_or_root_is_invented_by_lineage(ns):
    ext, raw, faces = extraction_fixture(ns)
    specs = ext.extract_from_mesh(raw, faces, provenance={"native_topology": "CLAIMED_COMPLETE"})
    record = specs[0].cad_provenance["candidate_records"][0]
    assert "owner_id" not in record and "root_face_ids" not in record
    assert "native_topology" not in specs[0].cad_provenance
    assert ext.last_source_lineage["identity_scope"] == "DETECTION_INVOCATION_LOCAL_NOT_NATIVE_IDENTITY"
    assert ns["imports"] == []


@pytest.mark.parametrize("enabled", [False, True])
def test_temporary_native_clouds_and_trace_cube_released_before_outliers(ns, enabled):
    raw, _, _ = fake_open3d(ns)
    cloud_type = ns["fake_open3d"].geometry.PointCloud
    original_remove = cloud_type.remove_statistical_outlier
    original_trace = cloud_type.voxel_down_sample_and_trace
    refs = {}
    inspected = []

    def make_original():
        cloud = cloud_type()
        refs["original"] = weakref.ref(cloud)
        return cloud

    def trace(self, **kwargs):
        result = original_trace(self, **kwargs)
        refs["traced"] = weakref.ref(result[0])
        refs["cube"] = weakref.ref(result[1])
        return result

    def remove(self, **kwargs):
        assert refs["original"]() is None
        if enabled:
            assert refs["traced"]() is None
            assert refs["cube"]() is None
        inspected.append(True)
        return original_remove(self, **kwargs)

    ns["fake_open3d"].geometry.PointCloud = make_original
    cloud_type.voxel_down_sample_and_trace = trace
    cloud_type.remove_statistical_outlier = remove
    det = detector(ns)
    det._begin_source_lineage(raw, enabled)
    det._preprocess_point_cloud(raw, np.tile([0., 0., 1.], (6, 1)), 1., 20, 2.)
    assert inspected == [True]


@pytest.mark.parametrize("empty", [False, True])
def test_custom_legacy_detector_missing_lineage_attribute_preserves_compatibility(ns, empty):
    raw = np.zeros((6, 3))
    detected = [] if empty else [bend(ns, plane(ns, (0, 1, 2)), plane(ns, (3, 4, 5), 1))]
    ns["BendDetector"] = lambda **kwargs: SimpleNamespace(detect_bends=lambda *args, **kw: detected)
    ext = ns["CADBendExtractor"]()
    baseline = ext.extract_from_mesh(raw, np.empty((0, 3), dtype=int))
    assert ext.last_source_lineage is None
    assert all(spec.cad_provenance is None for spec in baseline)
    traced = ext.extract_from_mesh(raw, np.empty((0, 3), dtype=int), provenance={})
    assert [spec.to_dict() for spec in traced] == [spec.to_dict() for spec in baseline]
    assert ext.last_source_lineage["status"] == "UNAVAILABLE"
    assert ext.last_source_lineage["reason"] == "DETECTOR_LINEAGE_UNAVAILABLE"
    assert all(spec.cad_provenance["candidate_records"] == [] for spec in traced)


def test_late_invalidation_clears_both_seed_and_merged_plane_objects(ns):
    det = configure_geometry(ns, detector(ns), coplanar=True)
    result = det.detect_bends(np.arange(27, dtype=float).reshape(9, 3), preprocess=False,
                              preserve_source_lineage=True)
    merged = result[0].flange1
    assert len(merged.source_segments) == 2
    seed_planes = list(det._lineage_planes)
    det._invalidate_source_lineage("INVENTED_LATE_FAILURE")
    assert result[0].source_candidates == ()
    assert det._lineage_original_candidates == []
    for item in seed_planes + [merged, result[0].flange2]:
        assert item.source_lineage_status == "UNAVAILABLE"
        assert item.source_vertex_indices == item.source_segments == ()
    assert det.last_source_lineage["candidate_records"] == []
