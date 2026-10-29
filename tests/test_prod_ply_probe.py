"""C02 scan probe: PLY ascii/binary (mesh and points) and STL ascii/binary.

Synthetic bytes in tmp_path only; file names copy the real ZEISS shapes.
"""
from __future__ import annotations

import struct

import pytest

from apps.production_line.ingest.ply_probe import probe

VERTS = [(0.0, 0.0, 0.0), (10.0, 0.0, -2.0), (0.0, 20.5, 3.0), (4.0, 5.0, 6.0)]
FACES = [(0, 1, 2), (0, 2, 3)]


def ply_bytes(fmt: str = "binary_little_endian", *, verts=VERTS, faces=FACES, extra_vertex_prop: bool = False,
              double: bool = False) -> bytes:
    vtype = "double" if double else "float"
    header = ["ply", f"format {fmt} 1.0", "comment synthetic", f"element vertex {len(verts)}",
              f"property {vtype} x", f"property {vtype} y", f"property {vtype} z"]
    if extra_vertex_prop:
        header.append("property uchar red")
    if faces is not None:
        header += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    header.append("end_header")
    head = ("\n".join(header) + "\n").encode("ascii")
    if fmt == "ascii":
        lines = [" ".join(f"{c:g}" for c in v) + (" 7" if extra_vertex_prop else "") for v in verts]
        lines += [f"3 {a} {b} {c}" for a, b, c in (faces or [])]
        return head + ("\n".join(lines) + "\n").encode("ascii")
    e = "<" if fmt == "binary_little_endian" else ">"
    body = b""
    for v in verts:
        body += struct.pack(e + ("ddd" if double else "fff"), *v)
        if extra_vertex_prop:
            body += struct.pack(e + "B", 7)
    for f in faces or []:
        body += struct.pack(e + "Biii", 3, *f)
    return head + body


def stl_binary(tris, header: bytes = b"binary stl") -> bytes:
    out = header.ljust(80, b" ")[:80] + struct.pack("<I", len(tris))
    for tri in tris:
        out += struct.pack("<3f", 0, 0, 1) + b"".join(struct.pack("<3f", *v) for v in tri) + b"\x00\x00"
    return out


def stl_ascii(tris, *, terminated: bool = True) -> bytes:
    lines = ["solid part"]
    for tri in tris:
        lines += ["  facet normal 0 0 1", "    outer loop"]
        lines += [f"      vertex {x} {y} {z}" for x, y, z in tri]
        lines += ["    endloop", "  endfacet"]
    if terminated:
        lines.append("endsolid part")
    return ("\n".join(lines) + "\n").encode("ascii")


TRIS = [((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 1, 0), (0, 0, -5))]


def _write(tmp_path, name: str, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return p


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [("binary_little_endian", "PLY_BINARY_LE"), ("binary_big_endian", "PLY_BINARY_BE"), ("ascii", "PLY_ASCII")],
)
def test_ply_mesh_formats(tmp_path, fmt, expected):
    r = probe(_write(tmp_path, "23553000_B_FULL_SCAN 1.ply", ply_bytes(fmt)))
    assert r.ok, r
    assert (r.fmt, r.vertex_count, r.face_count) == (expected, 4, 2)
    assert r.reason is None and r.extension == ".ply"
    if fmt != "ascii":
        assert r.bbox_mm == ((0.0, 0.0, -2.0), (10.0, 20.5, 6.0))
        assert r.expected_min_bytes == r.size_bytes


def test_ply_point_cloud_and_extra_props(tmp_path):
    r = probe(_write(tmp_path, "BEFORE BENDING  11979003_C.ply", ply_bytes(faces=None, extra_vertex_prop=True, double=True)))
    assert r.ok and r.face_count == 0 and r.vertex_count == 4
    assert r.bbox_mm == ((0.0, 0.0, -2.0), (10.0, 20.5, 6.0))
    r = probe(_write(tmp_path, "pts.ply", ply_bytes("ascii", faces=None, extra_vertex_prop=True)))
    assert r.ok and r.fmt == "PLY_ASCII" and r.face_count == 0


def test_ply_crlf_header(tmp_path):
    head, _, body = ply_bytes().partition(b"end_header\n")
    data = head.replace(b"\n", b"\r\n") + b"end_header\r\n" + body
    r = probe(_write(tmp_path, "crlf.ply", data))
    assert r.ok and r.vertex_count == 4


@pytest.mark.parametrize("cut", [1, 13, 40])
def test_binary_ply_truncated(tmp_path, cut):
    data = ply_bytes()
    r = probe(_write(tmp_path, "38172000_01_SCAN_2 1.ply", data[:-cut]))
    assert not r.ok and r.reason == "PLY_TRUNCATED"
    assert r.expected_min_bytes == len(data) and r.fmt == "PLY_BINARY_LE"


def test_binary_ply_truncated_inside_vertices(tmp_path):
    data = ply_bytes(faces=None)
    r = probe(_write(tmp_path, "p.ply", data[:-5]))
    assert r.reason == "PLY_TRUNCATED"


def test_ascii_ply_truncated(tmp_path):
    data = ply_bytes("ascii")
    lines = data.split(b"\n")
    truncated = b"\n".join(lines[:-3]) + b"\n"  # drop the last face line and part of another
    r = probe(_write(tmp_path, "a.ply", truncated))
    assert not r.ok and r.reason == "PLY_TRUNCATED"
    # last line without trailing newline still counts
    r = probe(_write(tmp_path, "b.ply", data.rstrip(b"\n")))
    assert r.ok


@pytest.mark.parametrize(
    "header",
    [
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nend_header\n",
        b"ply\nformat binary_middle_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n",
        b"ply\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n",
        b"ply\nformat ascii 1.0\nproperty float x\nend_header\n",
        b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float128 x\nend_header\n",
        b"ply\nformat ascii 1.0\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n",
        b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n",  # no end_header
        b"ply\nformat ascii 1.0\nbogus line\nend_header\n",
    ],
)
def test_ply_header_invalid(tmp_path, header):
    r = probe(_write(tmp_path, "h.ply", header + b"\x00" * 64))
    assert not r.ok and r.reason == "PLY_HEADER_INVALID", r


def test_ply_zero_vertices_is_empty(tmp_path):
    r = probe(_write(tmp_path, "e.ply", ply_bytes(verts=[], faces=[])))
    assert not r.ok and r.reason == "EMPTY_SCAN" and r.vertex_count == 0


def test_empty_and_unsupported_and_unreadable(tmp_path):
    assert probe(_write(tmp_path, "z.ply", b"")).reason == "EMPTY_SCAN"
    assert probe(_write(tmp_path, "x.ply", b"OFF\n3 1 0\n" + b"0 " * 400)).reason == "UNSUPPORTED_FORMAT"
    assert probe(tmp_path / "missing.ply").reason == "READ_ERROR"
    assert probe(_write(tmp_path, "magic.ply", b"plyX" + b"\x00" * 200)).reason == "PLY_HEADER_INVALID"


def test_binary_stl(tmp_path):
    r = probe(_write(tmp_path, "28870000_A_FULL_SCAN 1.stl", stl_binary(TRIS)))
    assert r.ok and (r.fmt, r.vertex_count, r.face_count) == ("STL_BINARY", 6, 2)
    assert r.bbox_mm == ((0.0, 0.0, -5.0), (1.0, 1.0, 0.0)) and r.extension == ".stl"


def test_binary_stl_with_solid_header_is_binary(tmp_path):
    # SolidWorks-style exports start the 80-byte header with "solid"
    r = probe(_write(tmp_path, "s.stl", stl_binary(TRIS, header=b"solid exported by cad")))
    assert r.ok and r.fmt == "STL_BINARY"


def test_binary_stl_size_mismatch(tmp_path):
    data = stl_binary(TRIS)
    r = probe(_write(tmp_path, "t.stl", data[:-7]))
    assert not r.ok and r.reason == "STL_INVALID"
    r = probe(_write(tmp_path, "zero.stl", stl_binary([])))
    assert r.reason == "EMPTY_SCAN"


def test_ascii_stl(tmp_path):
    r = probe(_write(tmp_path, "a.stl", stl_ascii(TRIS)))
    assert r.ok and (r.fmt, r.face_count, r.vertex_count) == ("STL_ASCII", 2, 6)
    r = probe(_write(tmp_path, "upper.stl", stl_ascii(TRIS).upper()))
    assert r.ok and r.face_count == 2
    r = probe(_write(tmp_path, "cut.stl", stl_ascii(TRIS, terminated=False)))
    assert not r.ok and r.reason == "STL_INVALID"
    r = probe(_write(tmp_path, "none.stl", stl_ascii([])))
    assert r.reason == "EMPTY_SCAN"


def test_ascii_stl_facet_count_across_chunks(tmp_path):
    many = [((i, 0, 0), (i + 1, 0, 0), (i, 1, 0)) for i in range(30_000)]  # > 1 MiB of text
    data = stl_ascii(many)
    assert len(data) > 2 * 1024 * 1024
    r = probe(_write(tmp_path, "big.stl", data))
    assert r.ok and r.face_count == 30_000


def test_content_decides_format_not_extension(tmp_path):
    r = probe(_write(tmp_path, "really_a_ply.stl", ply_bytes()))
    assert r.ok and r.fmt == "PLY_BINARY_LE" and r.extension == ".ply"
    r = probe(_write(tmp_path, "really_an_stl.ply", stl_binary(TRIS)))
    assert r.ok and r.fmt == "STL_BINARY"
    assert probe(_write(tmp_path, "junk.stl", b"\x01\x02" * 100)).reason == "STL_INVALID"


def test_bbox_skipped_without_float_coordinates(tmp_path):
    data = ply_bytes().replace(b"property float x", b"property int x")
    # int x changes nothing in size, so the file is complete; bbox needs float/double
    r = probe(_write(tmp_path, "i.ply", data))
    assert r.ok and r.bbox_mm is None
    r = probe(_write(tmp_path, "nobbox.ply", ply_bytes()), compute_bbox=False)
    assert r.ok and r.bbox_mm is None


# ---------------------------------------------------------------- final-record check (fix stage)
@pytest.mark.parametrize(
    ("cut_at", "faces"),
    [
        (b"3 0 2 3\n", True),  # cut inside the final face line: "3 0 " (list needs 1 + 3 tokens)
        (b"4 5 6\n", False),  # point cloud cut inside the final vertex line: "4 5"
    ],
)
def test_ascii_ply_cut_inside_final_line(tmp_path, cut_at, faces):
    data = ply_bytes("ascii", faces=FACES if faces else None)
    assert data.endswith(cut_at)
    keep = len(data) - len(cut_at) + (4 if faces else 3)  # "3 0 " / "4 5"
    r = probe(_write(tmp_path, "23553000_B_FULL_SCAN 1.ply", data[:keep]))
    assert not r.ok and r.reason == "PLY_TRUNCATED" and "final" in r.detail, r
    assert probe(_write(tmp_path, "whole.ply", data)).ok


def test_ascii_ply_final_record_token_count(tmp_path):
    data = ply_bytes("ascii")
    r = probe(_write(tmp_path, "extra.ply", data.rstrip(b"\n") + b" 9\n"))  # one token too many
    assert r.reason == "PLY_TRUNCATED"
    r = probe(_write(tmp_path, "badcount.ply", data.rstrip(b"\n")[: -len(b"3 0 2 3")] + b"x 0 2 3\n"))
    assert r.reason == "PLY_TRUNCATED" and "not an integer" in r.detail
    r = probe(_write(tmp_path, "nonl.ply", data.rstrip(b"\n")))  # complete, just no trailing newline
    assert r.ok
