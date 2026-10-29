"""Invented byte/geometry fixtures only; no file input, app, CAD or native jobs."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
from time import perf_counter

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend/cad_guided_mesh_input_v1.py"
ENCODINGS = ("ascii", "binary_little_endian", "binary_big_endian")


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location("invented_mesh_input_v1", SOURCE)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _ply(points=None, faces=None, encoding="binary_little_endian", scalar="double"):
    if points is None:
        points = np.array([[.1, 3., 8.], [1., 2., 7.], [2., 1., 6.], [3., 0., 5.]])
    header = (f"ply\nformat {encoding} 1.0\ncomment invented units mm remain unverified\n"
              f"obj_info invented frame scanner remains unverified\nelement vertex {len(points)}\n"
              f"property {scalar} z\nproperty {scalar} x\nproperty {scalar} y\nproperty uchar red\n")
    if faces is not None:
        header += f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
    header += "end_header\n"
    if encoding == "ascii":
        body = b"".join(f"{z:.17g} {x:.17g} {y:.17g} 7\n".encode() for x, y, z in points)
        if faces is not None:
            body += b"".join((str(len(face)) + " " + " ".join(map(str, face)) + "\n").encode()
                             for face in faces)
    else:
        endian = "<" if encoding == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + ("f4" if scalar == "float" else "f8"))
                          for name in ("z", "x", "y")] + [("red", "u1")])
        records = np.empty(len(points), dtype=dtype)
        for index, axis in enumerate(("x", "y", "z")):
            records[axis] = points[:, index]
        records["red"] = 7
        body = records.tobytes()
        if faces is not None:
            body += b"".join(struct.pack(endian + "B" + "i" * len(face), len(face), *face)
                             for face in faces)
    return header.encode() + body


def _grid(side=8):
    return np.stack(np.meshgrid(*([np.arange(side, dtype=float)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("scalar", ["float", "double"])
@pytest.mark.parametrize("faces", [None, [], [[0, 1, 2]], [[0, 1, 2, 3], [0, 2, 3]]])
def test_parser_preserves_every_raw_row_and_validates_all_face_lists(m, encoding, scalar, faces):
    points = np.array([[.1, 3., 8.], [1.00000006, 2., 7.], [2., 1., 6.], [3., 0., 5.]])
    blob = _ply(points, faces, encoding, scalar)
    parsed = m.parse_mesh_ply_v1(blob)
    expected = points.astype(np.float32 if scalar == "float" else np.float64).astype(np.float64)
    assert parsed["status"] == "AVAILABLE", parsed["reasons"]
    np.testing.assert_array_equal(parsed["points"], expected)
    assert parsed["source_sha256"] == hashlib.sha256(blob).hexdigest()
    assert parsed["vertex_count"] == 4
    assert parsed["face_count"] == (0 if faces is None else len(faces))
    assert parsed["face_index_count"] == (0 if faces is None else sum(map(len, faces)))
    assert parsed["validated_unused_vertex_properties"] == ["red"]
    assert parsed["units_status"] == parsed["coordinate_frame_status"] == "UNVERIFIED"
    assert parsed["mesh_validity_status"] == "NOT_QUALIFIED"
    assert parsed["header"]["header_declarations_verified"] is False
    assert parsed["production_use_allowed"] is False
    with pytest.raises(ValueError):
        parsed["points"].setflags(write=True)


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("faces", [[[0, 1, -1]], [[0, 1, 4]], [[0, 1]], [[0, 1, 1]],
                                   [[0, 1, 2, 0]], [[0, 1, 2], [0, 3, 4]]])
def test_no_invalid_or_repeated_face_index_is_silently_discarded(m, encoding, faces):
    result = m.parse_mesh_ply_v1(_ply(faces=faces, encoding=encoding))
    assert result["status"] == "UNAVAILABLE" and result["points"] is None


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("change", ["trailing", "short_vertex", "short_face", "extra_geometry",
                                    "missing_xyz", "duplicate_property", "face_float_index",
                                    "face_extra_property", "vertex_list", "unknown_header"])
def test_malformed_or_unhandled_input_refuses_entire_coordinate_result(m, encoding, change):
    blob = _ply(faces=[[0, 1, 2]], encoding=encoding)
    if change == "trailing": blob += b" "
    elif change == "short_vertex": blob = _ply(encoding=encoding)[:-3]
    elif change == "short_face": blob = blob[:-3]
    elif change == "extra_geometry":
        blob = blob.replace(b"end_header", b"element edge 0\nproperty int a\nend_header")
    elif change == "missing_xyz": blob = blob.replace(b"double x", b"double u")
    elif change == "duplicate_property": blob = blob.replace(b"double x", b"double z")
    elif change == "face_float_index": blob = blob.replace(b"list uchar int", b"list uchar float")
    elif change == "face_extra_property": blob = blob.replace(b"end_header", b"property uchar red\nend_header")
    elif change == "vertex_list": blob = blob.replace(b"property double x", b"property list uchar float x")
    elif change == "unknown_header": blob = blob.replace(b"end_header", b"unexpected instruction\nend_header")
    result = m.parse_mesh_ply_v1(blob)
    assert result["status"] == "UNAVAILABLE" and result["points"] is None


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1_000_001.])
def test_no_nonfinite_or_out_of_range_vertex_can_be_filtered_to_success(m, encoding, value):
    points = np.zeros((15, 3))
    points[-1, 1] = value
    result = m.prepare_mesh_scan_v1(scan_bytes=_ply(points, encoding=encoding), max_engine_rows=12)
    assert result["status"] == "UNAVAILABLE" and result["raw_points"] is None and result["points"] is None


@pytest.mark.parametrize("change", ["bool_blob", "bytearray_blob", "format", "version", "vertex_cap",
                                    "face_cap", "no_terminator", "bad_magic"])
def test_strict_bytes_and_header_limits(m, change):
    blob = _ply(faces=[[0, 1, 2]])
    if change == "bool_blob": blob = True
    elif change == "bytearray_blob": blob = bytearray(blob)
    elif change == "format": blob = blob.replace(b"binary_little_endian 1.0", b"compressed 1.0")
    elif change == "version": blob = blob.replace(b"endian 1.0", b"endian 2.0")
    elif change == "vertex_cap": blob = blob.replace(b"vertex 4", b"vertex 1000001")
    elif change == "face_cap": blob = blob.replace(b"face 1", b"face 2000001")
    elif change == "no_terminator": blob = blob.replace(b"end_header\n", b"")
    elif change == "bad_magic": blob = b"not" + blob
    result = m.parse_mesh_ply_v1(blob)
    assert result["status"] == "UNAVAILABLE" and result["points"] is None


@pytest.mark.parametrize("budget,limit", [("MAX_PLY_BYTES", 4), ("MAX_HEADER_BYTES", 32),
                                         ("MAX_PROPERTIES", 3), ("MAX_FACE_INDICES", 2),
                                         ("MAX_ASCII_ROW_BYTES", 3), ("MAX_SCALAR_TOKEN_BYTES", 1)])
def test_declared_cardinality_byte_and_token_limits(m, monkeypatch, budget, limit):
    monkeypatch.setattr(m, budget, limit)
    result = m.parse_mesh_ply_v1(_ply(faces=[[0, 1, 2]], encoding="ascii"))
    assert result["status"] == "UNAVAILABLE" and result["points"] is None


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_face_sixteen_vertex_boundary_and_no_implied_surface_validity(m, encoding):
    points = np.zeros((17, 3))  # Distinct indices need not form a valid physical surface.
    accepted = m.parse_mesh_ply_v1(_ply(points, [list(range(16))], encoding))
    rejected = m.parse_mesh_ply_v1(_ply(points, [list(range(17))], encoding))
    assert accepted["status"] == "AVAILABLE" and accepted["mesh_validity_status"] == "NOT_QUALIFIED"
    assert rejected["status"] == "UNAVAILABLE"


@pytest.mark.parametrize("encoding", ["binary_little_endian", "binary_big_endian"])
def test_vectorized_triangle_path_validates_final_face_not_just_prefix(m, encoding):
    faces = [[0, 1, 2]] * 4096
    good = m.parse_mesh_ply_v1(_ply(faces=faces, encoding=encoding))
    assert good["status"] == "AVAILABLE" and good["face_index_count"] == 12_288
    faces[-1] = [0, 1, 4]
    bad = m.parse_mesh_ply_v1(_ply(faces=faces, encoding=encoding))
    assert bad["status"] == "UNAVAILABLE" and bad["points"] is None


def test_identity_keeps_full_cloud_and_explicitly_datum_free_mode(m):
    points = _grid(4)
    blob = _ply(points, [[0, 1, 2]])
    result = m.prepare_mesh_scan_v1(scan_bytes=blob)
    assert result["status"] == "AVAILABLE", result["reasons"]
    np.testing.assert_array_equal(result["raw_points"], points)
    np.testing.assert_array_equal(result["points"], points)
    np.testing.assert_array_equal(result["engine_to_raw_row_indices"], np.arange(len(points)))
    assert result["registration_preparation_mode"] == "PROVISIONAL_WHOLE_PART_NO_STABLE_DATUMS"
    assert result["datum_stability_status"] == "NOT_ESTABLISHED"
    assert result["coverage"]["maximum_point_displacement_raw_units"] == 0
    assert result["coverage"]["sampling_status"] == "IDENTITY_ALL_RAW_ROWS_RETAINED"
    assert result["sample_cloud_contract"] == "REGISTRATION_AND_PROPOSALS_ONLY"
    assert result["measurement_cloud_contract"] == "FULL_VALIDATED_RAW_XYZ_REQUIRED_FOR_LOCAL_FITTING"
    assert result["measurement_role_contract"] == "LOCAL_FITTING_EXCLUDES_ALL_DECLARED_RAW_DATUM_ROWS"
    assert result["coverage"]["datum_displacement_raw_units"] is None


def test_voxel_selection_matches_independent_small_grid_center_argmin_and_finest_capacity(m):
    raw = _grid(8)
    points = np.concatenate((raw[::2], raw[1::2]))
    result = m.prepare_mesh_scan_v1(scan_bytes=_ply(points), max_engine_rows=64)
    assert result["status"] == "AVAILABLE", result["reasons"]
    receipt = result["coverage"]
    level = receipt["voxel_level"]
    width = receipt["voxel_width_raw_units"]
    assert level == 2 and receipt["occupied_inspection_voxels"] == 64
    groups = {}
    for row, point in enumerate(points):
        cell = tuple(min(int(np.floor(value / width)), (1 << level) - 1) for value in point)
        center = (np.asarray(cell) + .5) * width
        squared = float(np.dot(point - center, point - center))
        groups.setdefault(cell, []).append((squared, row))
    expected = sorted(min(members)[1] for members in groups.values())
    np.testing.assert_array_equal(result["engine_to_raw_row_indices"], expected)
    assert len(np.unique(np.minimum(np.floor(points / (width / 2)), (1 << (level + 1)) - 1), axis=0)) > 64
    assigned_raw = result["engine_to_raw_row_indices"][result["raw_to_representative_engine_row_indices"]]
    distances = np.linalg.norm(points - points[assigned_raw], axis=1)
    assert receipt["maximum_point_displacement_raw_units"] == pytest.approx(float(np.max(distances)), abs=1e-12)
    assert receipt["assigned_vertex_count"] == len(points)
    assert receipt["continuous_surface_coverage"] == receipt["bend_feature_preservation"] == "NOT_ESTABLISHED"


def test_rare_spatial_islands_are_not_lost_to_dense_cluster_row_ranks(m):
    points = np.zeros((1000, 3))
    points[:998, 0] = np.linspace(0, 1, 998)
    points[-2:, 0] = [100., 1000.]
    result = m.prepare_mesh_scan_v1(scan_bytes=_ply(points), max_engine_rows=48)
    assert result["status"] == "AVAILABLE"
    assert {998, 999}.issubset(set(result["engine_to_raw_row_indices"]))
    assert len(result["raw_points"]) == 1000  # Full support is not replaced by representatives.


def test_every_declared_datum_and_caller_order_survive_without_inspection_reuse(m):
    points = _grid(8)
    datums = list(reversed(range(36)))
    result = m.prepare_mesh_scan_v1(scan_bytes=_ply(points), datum_raw_row_indices=datums, max_engine_rows=64)
    assert result["status"] == "AVAILABLE", result["reasons"]
    to_raw = result["engine_to_raw_row_indices"]
    np.testing.assert_array_equal(result["datum_raw_row_indices"], datums)
    np.testing.assert_array_equal(to_raw[result["datum_engine_row_indices"]], datums)
    assert set(result["datum_engine_row_indices"]).isdisjoint(result["inspection_engine_row_indices"])
    assert set(result["datum_raw_row_indices"]).isdisjoint(result["inspection_raw_row_indices"])
    assert set(result["datum_raw_row_indices"]) | set(result["inspection_raw_row_indices"]) == set(range(len(points)))
    assert len(result["inspection_engine_row_indices"]) >= 12
    assert set(result["datum_engine_row_indices"]) | set(result["inspection_engine_row_indices"]) == set(range(len(to_raw)))
    np.testing.assert_array_equal(result["raw_to_engine_row_indices"][to_raw], np.arange(len(to_raw)))
    np.testing.assert_array_equal(to_raw[result["raw_to_representative_engine_row_indices"]][datums], datums)
    assigned = result["raw_to_representative_engine_row_indices"][36:]
    assert not np.isin(assigned, result["datum_engine_row_indices"]).any()
    assert result["registration_preparation_mode"] == "CALLER_DECLARED_DATUM_ROWS_NOT_PHYSICALLY_VERIFIED"
    datums.clear()
    assert len(result["datum_raw_row_indices"]) == 36


def test_coincident_rows_preserve_raw_multiplicity_and_deterministic_minimum_supplement(m):
    result = m.prepare_mesh_scan_v1(scan_bytes=_ply(np.zeros((256, 3))), max_engine_rows=48)
    assert result["status"] == "AVAILABLE"
    assert len(result["raw_points"]) == 256 and len(result["points"]) == 12
    np.testing.assert_array_equal(result["engine_to_raw_row_indices"], np.arange(12))
    np.testing.assert_array_equal(result["raw_to_representative_engine_row_indices"][12:], np.zeros(244, dtype=int))
    assert result["coverage"]["maximum_point_displacement_raw_units"] == 0
    assert result["coverage"]["occupied_inspection_voxels"] == 1
    assert result["coverage"]["minimum_row_supplement_count"] == 11


def test_exact_geometry_not_centroids_is_returned_and_replay_hashes_are_identical(m):
    points = _grid(9)
    blob = _ply(points)
    first = m.prepare_mesh_scan_v1(scan_bytes=blob, max_engine_rows=100)
    second = m.prepare_mesh_scan_v1(scan_bytes=blob, max_engine_rows=100)
    assert first["status"] == second["status"] == "AVAILABLE"
    assert first["source_bindings"] == second["source_bindings"]
    np.testing.assert_array_equal(first["points"], points[first["engine_to_raw_row_indices"]])
    assert first["source_bindings"]["scan_sha256"] == hashlib.sha256(blob).hexdigest()
    encoded = json.dumps(first["sampling_policy"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert first["source_bindings"]["sampling_policy_sha256"] == hashlib.sha256(encoded).hexdigest()
    for key in ["raw_points", "points", "engine_to_raw_row_indices", "datum_raw_row_indices",
                "datum_engine_row_indices", "inspection_engine_row_indices", "raw_to_engine_row_indices",
                "raw_to_representative_engine_row_indices", "inspection_raw_row_indices"]:
        assert not first[key].flags.writeable
        with pytest.raises(ValueError):
            first[key].setflags(write=True)


def test_predeclared_displacement_limit_fails_closed_and_is_never_auto_relaxed(m):
    blob = _ply(_grid(8))
    rejected = m.prepare_mesh_scan_v1(scan_bytes=blob, max_engine_rows=64, max_point_displacement_raw_units=0)
    assert rejected["status"] == "UNAVAILABLE"
    assert rejected["reasons"] == ["POINT_DISPLACEMENT_LIMIT_EXCEEDED"]
    assert rejected["points"] is None and rejected["raw_points"] is None
    assert rejected["coverage"]["displacement_limit_status"] == "EXCEEDED"
    admitted = m.prepare_mesh_scan_v1(scan_bytes=blob, max_engine_rows=64, max_point_displacement_raw_units=10)
    assert admitted["status"] == "AVAILABLE"
    assert admitted["coverage"]["displacement_limit_status"] == "WITHIN_DECLARED_LIMIT"


@pytest.mark.parametrize("field,value", [("max_engine_rows", True), ("max_engine_rows", 64.),
    ("max_engine_rows", np.uint64(64)), ("max_engine_rows", 11), ("max_engine_rows", 200001),
    ("max_point_displacement_raw_units", True), ("max_point_displacement_raw_units", "1"),
    ("max_point_displacement_raw_units", -1), ("max_point_displacement_raw_units", float("nan")),
    ("max_point_displacement_raw_units", float("inf")), ("max_point_displacement_raw_units", 2**10000),
    ("datum_raw_row_indices", set(range(36))), ("datum_raw_row_indices", np.arange(36)),
    ("datum_raw_row_indices", list(range(35))), ("datum_raw_row_indices", [0] * 36),
    ("datum_raw_row_indices", [True] + list(range(1, 36))),
    ("datum_raw_row_indices", [0.] + list(range(1, 36))),
    ("datum_raw_row_indices", [np.uint64(2**64 - 1)] + list(range(1, 36))),
    ("datum_raw_row_indices", [1000] + list(range(1, 36))),
    ("datum_raw_row_indices", list(range(5001)))])
def test_strict_selection_declarations_have_no_silent_bool_float_or_uint_coercion(m, field, value):
    result = m.prepare_mesh_scan_v1(scan_bytes=_ply(_grid(4)), **{field: value})
    assert result["status"] == "UNAVAILABLE" and result["points"] is None and result["raw_points"] is None


@pytest.mark.parametrize("raw_count,datums,cap", [(11, (), 12), (48, list(range(37)), 48),
                                                 (100, list(range(36)), 47)])
def test_inspection_minimum_is_not_satisfied_by_reusing_datum_rows(m, raw_count, datums, cap):
    result = m.prepare_mesh_scan_v1(scan_bytes=_ply(np.zeros((raw_count, 3))),
                                    datum_raw_row_indices=datums, max_engine_rows=cap)
    assert result["status"] == "UNAVAILABLE" and result["points"] is None


def test_iterator_declarations_are_refused_before_consumption(m):
    def forbidden():
        raise AssertionError("must not consume arbitrary iterable")
        yield 0
    result = m.prepare_mesh_scan_v1(scan_bytes=b"never parsed", datum_raw_row_indices=forbidden())
    assert result["reasons"] == ["DATUM_ROW_CONTAINER_OR_BUDGET"]


def test_full_one_million_vertex_preparation_keeps_immutable_raw_support_and_bounded_samples(m, record_property):
    points = _grid(100)
    datums = tuple(reversed(range(5000)))
    blob = _ply(points, faces=[[0, 1, 100]])
    started = perf_counter()
    result = m.prepare_mesh_scan_v1(scan_bytes=blob, datum_raw_row_indices=datums)
    elapsed = perf_counter() - started
    record_property("mesh_input_v1_one_million_vertex_seconds", elapsed)
    assert result["status"] == "AVAILABLE", result["reasons"]
    assert result["raw_points"].shape == (1_000_000, 3)
    assert result["raw_points"].nbytes == 24_000_000
    assert 5012 <= len(result["points"]) <= 200_000
    assert len(result["datum_raw_row_indices"]) == 5000
    assert result["coverage"]["assigned_vertex_count"] == 1_000_000
    assert len(result["raw_to_representative_engine_row_indices"]) == 1_000_000
    np.testing.assert_array_equal(result["raw_points"], points)
    np.testing.assert_array_equal(result["engine_to_raw_row_indices"][result["datum_engine_row_indices"]], datums)
    assert result["coverage"]["datum_displacement_raw_units"] == 0
    assert np.isfinite(result["coverage"]["maximum_point_displacement_raw_units"])


def test_source_has_no_file_app_native_research_or_reference_runtime_imports():
    tree = ast.parse(SOURCE.read_bytes())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert imported == {"hashlib", "json", "io", "math", "re", "numpy"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in {"open", "exec", "eval", "__import__"} for node in ast.walk(tree))
    public = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "prepare_mesh_scan_v1")
    assert {arg.arg for arg in public.args.kwonlyargs} == {
        "scan_bytes", "datum_raw_row_indices", "max_engine_rows", "max_point_displacement_raw_units"
    }


@pytest.mark.parametrize("step", [1e-200, 1e-310])
def test_tiny_nonzero_displacement_cannot_underflow_into_zero_certificate(m, step):
    points = np.zeros((20, 3))
    points[:, 0] = np.arange(20) * step
    blob = _ply(points)
    result = m.prepare_mesh_scan_v1(scan_bytes=blob, max_engine_rows=12)
    assert result["status"] == "AVAILABLE"
    assigned = result["engine_to_raw_row_indices"][result["raw_to_representative_engine_row_indices"]]
    # Independent one-dimensional absolute difference, no norm/squaring.
    expected = float(np.max(np.abs(points[:, 0]-points[assigned, 0])))
    assert expected > 0
    assert result["coverage"]["maximum_point_displacement_raw_units"] == expected
    rejected = m.prepare_mesh_scan_v1(scan_bytes=blob, max_engine_rows=12,
                                      max_point_displacement_raw_units=0)
    assert rejected["reasons"] == ["POINT_DISPLACEMENT_LIMIT_EXCEEDED"]
