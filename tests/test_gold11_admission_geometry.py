"""Invented PLY bytes/reference declarations only; no file or native geometry input."""

import ast
import copy
import importlib.util
from pathlib import Path
import struct

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/gold11_admission_geometry.py"
ENCODINGS = ("ascii", "binary_little_endian", "binary_big_endian")


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location("invented_gold11_admission_geometry", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ply(encoding="ascii", *, faces=None, points=None, scalar="double", comments=True):
    if points is None:
        points = np.array([[.1, 3., 8.], [1., 2., 7.], [2., 1., 6.], [3., 0., 5.]])
    header = f"ply\nformat {encoding} 1.0\n"
    if comments:
        header += "comment units mm are only a declaration\nobj_info frame scanner is unverified\n"
    header += f"element vertex {len(points)}\nproperty {scalar} z\nproperty {scalar} x\nproperty {scalar} y\nproperty uchar red\n"
    if faces is not None:
        header += f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
    header += "end_header\n"
    if encoding == "ascii":
        body = b"".join(f"{z:.17g} {x:.17g} {y:.17g} {index}\n".encode()
                        for index, (x, y, z) in enumerate(points))
        if faces is not None:
            body += b"".join((str(len(face)) + " " + " ".join(map(str, face)) + "\n").encode()
                             for face in faces)
    else:
        endian = "<" if encoding == "binary_little_endian" else ">"
        code = "f" if scalar == "float" else "d"
        body = b"".join(struct.pack(endian + code * 3 + "B", z, x, y, index)
                        for index, (x, y, z) in enumerate(points))
        if faces is not None:
            body += b"".join(struct.pack(endian + "B" + "i" * len(face), len(face), *face)
                             for face in faces)
    return header.encode() + body, np.array(points, dtype=np.float32 if scalar == "float" else np.float64).astype(np.float64)


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("faces", [None, [], [[0, 1, 2]], [[3, 2, 1, 0], [0, 2, 3]]])
@pytest.mark.parametrize("scalar", ["float", "double"])
def test_all_supported_encodings_preserve_rows_and_declared_scalar_precision(m, encoding, faces, scalar):
    blob, expected = _ply(encoding, faces=faces, scalar=scalar)
    result = m.parse_diagnostic_ply(blob)
    assert result["status"] == "AVAILABLE", result["reasons"]
    assert np.array_equal(result["points"], expected)
    assert result["points"].dtype == np.float64 and not result["points"].flags.writeable
    assert result["face_index_count"] == (0 if faces is None else sum(map(len, faces)))
    assert result["header"]["comments"] == ["units mm are only a declaration"]
    assert result["header"]["obj_info"] == ["frame scanner is unverified"]
    assert result["header"]["header_terminated"] is True
    assert result["header"]["header_declarations_verified"] is False
    assert result["coordinate_frame_status"] == result["units_status"] == "UNVERIFIED"
    assert result["production_use_allowed"] is False
    assert result["trailing_geometry_status"] == ("NO_TRAILING_ELEMENTS" if faces is None
                                                 else "FACE_LIST_VALIDATED_NOT_USED_FOR_COORDINATES")


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("change", ["extra_byte", "truncated_vertex", "truncated_face",
                                    "negative_face_index", "out_of_range_face_index", "two_vertex_face"])
def test_invalid_body_or_face_refuses_all_coordinates(m, encoding, change):
    if change == "negative_face_index":
        blob, _ = _ply(encoding, faces=[[0, -1, 2]])
    elif change == "out_of_range_face_index":
        blob, _ = _ply(encoding, faces=[[0, 1, 4]])
    elif change == "two_vertex_face":
        blob, _ = _ply(encoding, faces=[[0, 1]])
    elif change == "truncated_face":
        blob, _ = _ply(encoding, faces=[[0, 1, 2]])
        blob = blob[:-3]
    else:
        blob, _ = _ply(encoding)
        blob = blob + b" " if change == "extra_byte" else blob[:-3]
    result = m.parse_diagnostic_ply(blob)
    assert result["status"] == "UNAVAILABLE" and result["points"] is None
    assert result["header"]["elements"][0]["count"] == 4
    assert result["trailing_geometry_status"] == "NOT_VALIDATED_COORDINATES_WITHHELD"


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1_000_001.])
def test_nonfinite_or_unsupported_coordinates_are_never_returned(m, encoding, value):
    points = np.array([[0., 0., value], [1., 1., 1.], [2., 2., 2.]])
    blob, _ = _ply(encoding, points=points)
    result = m.parse_diagnostic_ply(blob)
    assert result["status"] == "UNAVAILABLE" and result["points"] is None


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("face_vertices,expected", [(16, "AVAILABLE"), (17, "UNAVAILABLE")])
def test_frozen_sixteen_vertex_face_boundary(m, encoding, face_vertices, expected):
    points = np.column_stack((np.arange(17), np.zeros(17), np.ones(17)))
    blob, _ = _ply(encoding, points=points, faces=[list(range(face_vertices))])
    result = m.parse_diagnostic_ply(blob)
    assert result["status"] == expected
    if expected == "UNAVAILABLE":
        assert result["points"] is None


@pytest.mark.parametrize("bad_row", [b"nan 0 0 0\n", b"0x1p0 0 0 0\n", b"1e999 0 0 0\n",
                                     b"0 0 0 -1\n", b"0 0 0 256\n", b"0 0 0 1.5\n",
                                     b"0 0 0 0 trailing\n", b"0 0 0\n", b"\n"])
def test_ascii_scalars_are_whole_typed_tokens_not_partial_parses(m, bad_row):
    blob, _ = _ply(points=np.zeros((1, 3)))
    header = blob[:blob.index(b"end_header\n") + len(b"end_header\n")]
    result = m.parse_diagnostic_ply(header + bad_row)
    assert result["status"] == "UNAVAILABLE" and result["points"] is None


@pytest.mark.parametrize("change", ["format", "version", "extra_element", "vertex_list",
                                    "face_extra_property", "face_float_index", "missing_xyz",
                                    "duplicate_property", "duplicate_element", "vertex_after_face",
                                    "unknown_header", "vertex_budget", "face_budget", "no_terminator"])
def test_unsupported_header_retains_facts_without_silently_stripping_geometry(m, change):
    blob, _ = _ply(faces=[[0, 1, 2]])
    if change == "format":
        blob = blob.replace(b"format ascii", b"format compressed")
    elif change == "version":
        blob = blob.replace(b"ascii 1.0", b"ascii 2.0")
    elif change == "extra_element":
        blob = blob.replace(b"end_header", b"element edge 1\nproperty int vertex1\nproperty int vertex2\nend_header")
    elif change == "vertex_list":
        blob = blob.replace(b"property double z", b"property list uchar double z")
    elif change == "face_extra_property":
        blob = blob.replace(b"end_header", b"property uchar color\nend_header")
    elif change == "face_float_index":
        blob = blob.replace(b"list uchar int", b"list uchar float")
    elif change == "missing_xyz":
        blob = blob.replace(b"property double x", b"property double u")
    elif change == "duplicate_property":
        blob = blob.replace(b"property double x", b"property double z")
    elif change == "duplicate_element":
        blob = blob.replace(b"element face 1", b"element vertex 1")
    elif change == "vertex_after_face":
        blob = blob.replace(b"element vertex 4", b"element face 4").replace(b"element face 1", b"element vertex 1")
    elif change == "unknown_header":
        blob = blob.replace(b"end_header", b"unrecognized instruction\nend_header")
    elif change == "vertex_budget":
        blob = blob.replace(b"element vertex 4", b"element vertex 1000001")
    elif change == "face_budget":
        blob = blob.replace(b"element face 1", b"element face 2000001")
    else:
        blob = blob.replace(b"end_header\n", b"")
    result = m.parse_diagnostic_ply(blob)
    assert result["status"] == "UNAVAILABLE" and result["points"] is None
    assert result["header"]["comments"] == ["units mm are only a declaration"]
    if change == "extra_element":
        assert result["header"]["elements"][-1]["name"] == "edge"
    if change == "unknown_header":
        assert result["header"]["unhandled_header_fields"] == ["unrecognized instruction"]


@pytest.mark.parametrize("budget", ["MAX_PLY_BYTES", "MAX_HEADER_BYTES", "MAX_PROPERTIES",
                                     "MAX_SCALAR_TOKEN_BYTES", "MAX_ASCII_ROW_BYTES", "MAX_FACE_INDICES"])
def test_all_dynamic_byte_and_cardinality_budgets_refuse_invented_excess(m, monkeypatch, budget):
    blob, _ = _ply(faces=[[0, 1, 2]])
    limit = {"MAX_PLY_BYTES": len(blob) - 1, "MAX_HEADER_BYTES": 32, "MAX_PROPERTIES": 3,
             "MAX_SCALAR_TOKEN_BYTES": 1, "MAX_ASCII_ROW_BYTES": 3, "MAX_FACE_INDICES": 2}[budget]
    monkeypatch.setattr(m, budget, limit)
    result = m.parse_diagnostic_ply(blob)
    assert result["status"] == "UNAVAILABLE" and result["points"] is None


def test_parser_is_pure_and_does_not_call_product_reader_or_file_apis():
    tree = ast.parse(SOURCE.read_bytes())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert imported == {"io", "math", "re", "numpy"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in {"open", "exec", "eval", "__import__"} for node in ast.walk(tree))


@pytest.mark.parametrize("record,representation", [
    ({"span_endpoints": [[0, 0, 0], [0, 0, 20]]}, "span_endpoints"),
    ({"line_start": [0, 0, 0], "line_end": [0, 0, 20]}, "line_start_line_end"),
    ({"anchor": [0, 0, 10], "axis_direction": [0, 0, 3], "span_mm": 20}, "anchor_axis_span"),
    ({"span_endpoints": [], "line_start": [0, 0, 0], "line_end": [0, 0, 20]}, "line_start_line_end"),
])
def test_reference_representation_matches_historical_endpoint_precedence(m, record, representation):
    source = ROOT / "scripts/evaluate_f1_against_manual_gold.py"
    node = next(node for node in ast.parse(source.read_bytes()).body
                if isinstance(node, ast.FunctionDef) and node.name == "_endpoints")
    namespace = {"np": np, "Any": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<historical-source-endpoints-only>", "exec"), namespace)
    original = copy.deepcopy(record)
    actual = m.parse_reference_endpoints(record)
    assert np.array_equal(actual["endpoints"], namespace["_endpoints"](record))
    assert actual["selected_representation"] == representation
    assert actual["physical_carrier_convention_status"] == "UNVERIFIED"
    assert actual["production_use_allowed"] is False and not actual["endpoints"].flags.writeable
    assert record == original


def test_agreeing_reversed_representations_preserve_priority_and_original_orientation(m):
    record = {"span_endpoints": [[0, 0, 20], [0, 0, 0]],
              "line_start": [0, 0, 0], "line_end": [0, 0, 20],
              "anchor": [0, 0, 10], "axis_direction": [0, 0, 1], "span_mm": 20}
    result = m.parse_reference_endpoints(record)
    assert result["selected_representation"] == "span_endpoints"
    assert result["representations_present"] == ["span_endpoints", "line_start_line_end", "anchor_axis_span"]
    assert np.array_equal(result["endpoints"], record["span_endpoints"])


@pytest.mark.parametrize("record", [None, {}, {"span_endpoints": [1, 2]},
    {"span_endpoints": [[0, 0, 0], [0, 0, 0]]},
    {"span_endpoints": [[0, 0, 0], [float("nan"), 0, 1]]},
    {"span_endpoints": [[0, 0, 0], [1_000_001, 0, 1]]},
    {"span_endpoints": [[False, False, False], [True, True, True]]},
    {"line_start": [0, 0, 0]},
    {"anchor": [0, 0, 0], "axis_direction": [0, 0, 0], "span_mm": 20},
    {"anchor": [0, 0, 0], "axis_direction": [0, 0, 1], "span_mm": -20},
    {"anchor": [0, 0, 0], "axis_direction": [0, 0, 1], "span_mm": True},
    {"anchor": [0, 0, 0], "axis_direction": [0, 0, 1]},
    {"span_endpoints": [[0, 0, 0], [0, 0, 20]], "line_start": [1, 0, 0], "line_end": [1, 0, 20]},
    {"span_endpoints": [[0, 0, 0], [0, 0, 20]], "anchor": [0, 0, 10], "axis_direction": [0, 0, 1], "span_mm": 21},
])
def test_invalid_or_contradictory_references_refuse_without_fallback(m, record):
    with pytest.raises(m.GeometryAdmissionUnavailable):
        m.parse_reference_endpoints(record)


def test_numeric_consistency_check_is_explicit_not_an_inference_or_scorer_tolerance(m):
    record = {"span_endpoints": [[0, 0, 0], [0, 0, 20]],
              "line_start": [5e-9, 0, 0], "line_end": [5e-9, 0, 20]}
    result = m.parse_reference_endpoints(record)
    assert result["numeric_consistency_atol_coordinate_units"] == 1e-8
    assert np.array_equal(result["endpoints"], record["span_endpoints"])
    record["line_start"][0] = 1e-4
    with pytest.raises(m.GeometryAdmissionUnavailable, match="CONTRADICTORY_REFERENCE_REPRESENTATIONS"):
        m.parse_reference_endpoints(record)
