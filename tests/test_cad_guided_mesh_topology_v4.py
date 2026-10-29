"""Invented-byte-only triangle/raw-row lineage controls; no files/native CAD."""
import hashlib

import numpy as np
import pytest

from backend.cad_guided_mesh_topology_v4 import parse_triangle_mesh_ply_v4


POINTS = ((2., 3., 4.), (7., 3., 4.), (2., 9., 4.), (7., 9., 4.))
FACES = ((2, 0, 1), (2, 1, 3), (2, 0, 1))


def _blob(encoding="binary_little_endian", faces=FACES, points=POINTS,
          count_type="uchar", index_type="int", include_faces=True):
    header = (f"ply\nformat {encoding} 1.0\ncomment invented topology fixture\n"
              f"element vertex {len(points)}\nproperty double z\nproperty uchar quality\n"
              "property double x\nproperty double y\n")
    if include_faces:
        header += f"element face {len(faces)}\nproperty list {count_type} {index_type} vertex_indices\n"
    header = (header + "end_header\n").encode()
    if encoding == "ascii":
        rows = [f"{z:.17g} 17 {x:.17g} {y:.17g}\n" for x, y, z in points]
        if include_faces:
            rows += [str(len(face)) + " " + " ".join(map(str, face)) + "\n" for face in faces]
        return header + "".join(rows).encode()
    endian = "<" if encoding == "binary_little_endian" else ">"
    dt = np.dtype([("z", endian+"f8"), ("q", "u1"), ("x", endian+"f8"), ("y", endian+"f8")])
    raw = np.empty(len(points), dtype=dt)
    raw["x"], raw["y"], raw["z"] = np.asarray(points).T
    raw["q"] = 17
    types = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4"}
    body = raw.tobytes()
    if include_faces:
        for face in faces:
            body += np.asarray([len(face)], dtype=endian+types[count_type]).tobytes()
            body += np.asarray(face, dtype=endian+types[index_type]).tobytes()
    return header + body


def _withheld(result):
    assert result["status"] == "UNAVAILABLE" and result["reasons"]
    assert result["points"] is None and result["triangle_raw_rows"] is None
    assert result["points_sha256"] is None and result["triangle_raw_rows_sha256"] is None
    assert result["production_use_allowed"] is False


@pytest.mark.parametrize("encoding", ("ascii", "binary_little_endian", "binary_big_endian"))
@pytest.mark.parametrize("count_type,index_type", (("uchar", "int"), ("short", "uint"), ("uint", "ushort")))
def test_raw_vertex_triangle_order_winding_duplicates_and_hashes_are_exact(encoding, count_type, index_type):
    blob = _blob(encoding, count_type=count_type, index_type=index_type)
    result = parse_triangle_mesh_ply_v4(blob)
    assert result["status"] == "AVAILABLE", result["reasons"]
    assert result["source_sha256"] == hashlib.sha256(blob).hexdigest()
    assert np.array_equal(result["points"], POINTS)
    assert np.array_equal(result["triangle_raw_rows"], FACES)
    assert result["points"].dtype == np.dtype("<f8")
    assert result["triangle_raw_rows"].dtype == np.dtype("<i8")
    assert result["points_sha256"] == hashlib.sha256(np.asarray(POINTS, dtype="<f8").tobytes()).hexdigest()
    assert result["triangle_raw_rows_sha256"] == hashlib.sha256(np.asarray(FACES, dtype="<i8").tobytes()).hexdigest()
    assert result["vertex_count"] == 4 and result["triangle_count"] == 3
    assert result["units_status"] == result["coordinate_frame_status"] == "UNVERIFIED"
    assert result["production_use_allowed"] is False
    assert result["topology_status"] == "OBSERVED_MESH_CONNECTIVITY_NOT_MATERIAL_CERTIFICATION"
    for array in (result["points"], result["triangle_raw_rows"]):
        with pytest.raises(ValueError):
            array.flat[0] = 99
        base = array
        while isinstance(base, np.ndarray):
            with pytest.raises(ValueError):
                base.setflags(write=True)
            base = base.base
        assert isinstance(base, bytes)


@pytest.mark.parametrize("encoding", ("ascii", "binary_little_endian", "binary_big_endian"))
@pytest.mark.parametrize("faces", (((0, 1, 2, 3),), ((0, 1, 2), (0, 1, 2, 3))))
def test_polygon_and_late_polygon_refuse_every_coordinate_not_triangulated(encoding, faces):
    result = parse_triangle_mesh_ply_v4(_blob(encoding, faces=faces))
    _withheld(result)
    assert result["reasons"] == ["NONTRIANGULAR_MESH_UNSUPPORTED"]


@pytest.mark.parametrize("encoding", ("ascii", "binary_little_endian", "binary_big_endian"))
@pytest.mark.parametrize("faces,include_faces", (((), True), ((), False)))
def test_empty_or_absent_mesh_is_explicitly_unavailable(encoding, faces, include_faces):
    result = parse_triangle_mesh_ply_v4(_blob(encoding, faces=faces, include_faces=include_faces))
    _withheld(result)
    assert result["reasons"] == ["TRIANGLE_MESH_REQUIRED"]


@pytest.mark.parametrize("encoding", ("ascii", "binary_little_endian", "binary_big_endian"))
@pytest.mark.parametrize("faces", (((0, 1, 4),), ((0, 1, -1),), ((0, 1, 1),)))
def test_bad_row_identity_never_escapes_validated_parser(encoding, faces):
    _withheld(parse_triangle_mesh_ply_v4(_blob(encoding, faces=faces)))


@pytest.mark.parametrize("encoding", ("ascii", "binary_little_endian", "binary_big_endian"))
@pytest.mark.parametrize("mutation", ("truncated", "trailing", "wrong_count", "unknown_element"))
def test_late_body_and_layout_failures_withhold_prefix(encoding, mutation):
    blob = _blob(encoding)
    if mutation == "truncated":
        blob = blob[:-2]
    elif mutation == "trailing":
        blob += b"unexpected"
    elif mutation == "wrong_count":
        blob = blob.replace(b"element face 3", b"element face 4")
    else:
        blob = blob.replace(b"end_header", b"element edge 0\nproperty int a\nend_header")
    _withheld(parse_triangle_mesh_ply_v4(blob))


@pytest.mark.parametrize("value", (None, bytearray(b"ply"), memoryview(b"ply"), "ply", b""))
def test_only_nonempty_immutable_byte_input(value):
    _withheld(parse_triangle_mesh_ply_v4(value))


def test_same_coordinates_different_source_comments_keep_distinct_byte_identity():
    first = _blob()
    second = first.replace(b"invented topology fixture", b"different invented provenance")
    a, b = parse_triangle_mesh_ply_v4(first), parse_triangle_mesh_ply_v4(second)
    assert a["source_sha256"] != b["source_sha256"]
    assert a["points_sha256"] == b["points_sha256"]
    assert a["triangle_raw_rows_sha256"] == b["triangle_raw_rows_sha256"]


def test_rigidly_moved_vertices_do_not_change_topology_or_original_row_identity():
    moved = np.asarray(POINTS)[:, [1, 2, 0]] + [100., -20., 4.]
    a, b = parse_triangle_mesh_ply_v4(_blob()), parse_triangle_mesh_ply_v4(_blob(points=moved))
    assert np.array_equal(b["points"], moved)
    assert np.array_equal(a["triangle_raw_rows"], b["triangle_raw_rows"])
    assert a["triangle_raw_rows_sha256"] == b["triangle_raw_rows_sha256"]
    assert a["points_sha256"] != b["points_sha256"]


def test_degenerate_geometric_triangle_is_preserved_for_downstream_explicit_rejection():
    result = parse_triangle_mesh_ply_v4(_blob(points=((0., 0., 0.), (1., 0., 0.),
                                                  (2., 0., 0.), (3., 0., 0.))))
    assert result["status"] == "AVAILABLE"
    assert np.array_equal(result["triangle_raw_rows"], FACES)
    assert result["topology_status"] == "OBSERVED_MESH_CONNECTIVITY_NOT_MATERIAL_CERTIFICATION"
